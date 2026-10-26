# Host-RAM residency for a model's materialized weights, with an optional node-shared backend.
#
# A model that is pinned in system RAM keeps its weights materialized as host Arrays. The default
# `PrivateWeightStore` gives each worker its own copy. `SharedWeightStore` (opt-in via
# `shared_host_weights`) places one copy per model in a node-level POSIX shared-memory region so
# that several same-node GPU workers share it, each transferring to its own device from the shared
# mapping.
#
# Coordination is peer-to-peer with no daemon; crash safety comes from an advisory file lock
# (`flock`), never an in-band flag. The region is content-addressed (`/rsw-<model>-<digest>`,
# where the digest covers the model name, the tensor layout, AND a size+mtime token of the
# weights file), so any fully-populated region with a matching digest is immutable and safe to
# reuse, including one left by a prior run. A weights-only update changes the token and thus the
# region key: reloading workers materialize the new version while stragglers keep holding the old
# region until the last releases it (rolling updates never serve a mix within one worker), and
# holder-free old-version regions are swept when the new version attaches. The lock file lives in `/dev/shm` next to the region: with Docker `ipc:
# host` (one container per worker), `/dev/shm` is the host's single tmpfs shared across all worker
# containers, so `flock` (a kernel lock on the shared inode) coordinates them with no extra
# volumes and no involvement of the image's overlay filesystem. Without a shared `/dev/shm` the
# regions cannot be shared anyway, so the caller falls back to the private store.
#
# Protocol (per model, on the lock file's flock). Readers hold LOCK_SH for their lifetime, so
# acquire never takes a blocking LOCK_EX (which would wait behind them):
#   acquire: take LOCK_SH and reuse a READY region with the digest (readers run concurrently).
#            If absent/partial, drop to a non-blocking LOCK_EX to become the creator: re-check,
#            then (re)create, populate, msync, set READY, and downgrade to LOCK_SH. If another
#            worker is mid-create (holds EX), spin on the fast path until it publishes READY.
#   release: try to upgrade to LOCK_EX non-blocking; success means last holder, so shm_unlink the
#            region. The lock file itself is left in place (tiny, reused). A crash auto-releases.
#
# Node budget (`budget` set). The regions in /dev/shm are then a cache bounded across every worker
# on the node, and their sizes are the whole ledger: nothing else records what is cached, so a
# crash cannot leave the accounting wrong. A release never unlinks; the region stays cached until
# evicted. Before creating a region a worker takes LOCK_EX on `/dev/shm/rsw-budget.lock`, sums the
# regions, and evicts holder-free ones (a non-blocking LOCK_EX on the region's lock succeeds), least
# recently loaded first, until the new one fits; it creates and reserves the region while still
# holding that lock (so the next admitter counts it) and fills it after dropping it. A worker holds
# a region's lock only while pinned (`residency: system`) or while uploading from it, so a model
# that stays on a device ages out of the cache. "Recently loaded" is the region lock file's mtime,
# touched on every load from the region. The creator holds the region's LOCK_EX while it waits for
# the budget lock, and an evictor only ever tries region locks without blocking, so the two locks
# cannot deadlock.

"""
    WeightStore

Host-RAM residency backend for a model's materialized weights. [`PrivateWeightStore`](@ref)
(the default) gives each worker its own copy; [`SharedWeightStore`](@ref) shares one copy per
model across same-node workers through a POSIX shared-memory region. See the file header for
the coordination protocol.
"""
abstract type WeightStore end

"""
    PrivateWeightStore()

The default store: each worker materializes its own host copy of a model's weights.
"""
struct PrivateWeightStore <: WeightStore end

"""
    SharedWeightStore(; mode=0o666, budget=nothing)

Opt-in store backing each model's host weights with a node-level POSIX shared-memory region so
same-node workers share one copy. See the file header for the flock-coordinated protocol.
`mode` sets the permission bits for the regions and their lock files. The default `0o666`
lets workers running as unrelated UIDs (for example different containers) share the regions,
but is world-writable; `0o660` is recommended for production and multi-user systems.

`budget` (bytes) caps the regions in `/dev/shm` across every worker on the node; see "Node
budget" in the file header. `nothing` (the default) leaves them unbudgeted. `prefix` names the
regions and lock files (`/dev/shm/<prefix>-...`); stores with different prefixes share neither
regions nor a budget. Every worker on a node must use the same one, which the server does.
"""
mutable struct SharedWeightStore <: WeightStore
    attached::Dict{String, NamedTuple{(:shm, :fd, :name, :lockpath), Tuple{SharedMemory, Cint, String, String}}}
    lock::ReentrantLock
    mode::UInt16                  # POSIX permission bits for the regions and lock files
    budget::Union{Int, Nothing}   # node-wide byte cap on the regions; nothing = unbudgeted
    prefix::String                # region and lock file name prefix in /dev/shm
end
function SharedWeightStore(;
        mode::Integer = 0o666, budget::Union{Integer, Nothing} = nothing, prefix::AbstractString = "rsw"
    )
    occursin(r"^[A-Za-z0-9]+$", prefix) || throw(ArgumentError("SharedWeightStore prefix must be alphanumeric, got $(repr(prefix))"))
    budget === nothing || budget >= 0 || throw(ArgumentError("SharedWeightStore budget must be non-negative, got $budget"))
    return SharedWeightStore(
        Dict{String, NamedTuple{(:shm, :fd, :name, :lockpath), Tuple{SharedMemory, Cint, String, String}}}(),
        ReentrantLock(), UInt16(mode), budget === nothing ? nothing : Int(budget), String(prefix)
    )
end

const _RSW_MAGIC = 0x52535701      # 'RSW' + format version 1
const _RSW_DATA_OFFSET = 64        # header occupies the first bytes; tensor data starts here
const _RSW_ALIGN = 64

# Linux flock / open / msync constants.
const _LOCK_SH = Cint(1)
const _LOCK_EX = Cint(2)
const _LOCK_NB = Cint(4)
const _LOCK_UN = Cint(8)
const _O_RDWR = Cint(2)
const _O_CREAT = Cint(0o100)
const _MS_SYNC = Cint(4)

# A blocking flock waits on another process (a peer filling a region holds the exclusive lock for
# the whole fill). It is declared gc_safe so that, while it waits, a collection requested by any
# other thread can proceed: a plain ccall would make every thread wait for the peer as well. No
# Julia object crosses the call, so gc_safe is sound here.
_flock(fd::Cint, op::Cint) = @ccall gc_safe = true flock(fd::Cint, op::Cint)::Cint
_open_lock(path::AbstractString, mode::Integer = 0o666) =
    ccall(:open, Cint, (Cstring, Cint, Cint), path, _O_RDWR | _O_CREAT, Cint(mode))
_close_fd(fd::Cint) = ccall(:close, Cint, (Cint,), fd)
_shm_unlink(name::AbstractString) = ccall(:shm_unlink, Cint, (Cstring,), name)
_msync(shm::SharedMemory) =
    ccall(:msync, Cint, (Ptr{Cvoid}, Csize_t, Cint), pointer(shm), Csize_t(sizeof(shm)), _MS_SYNC)

_align(n::Integer, a::Integer = _RSW_ALIGN) = ((Int(n) + a - 1) ÷ a) * a

# A 64-bit FNV-1a over a byte string. Used for the content digest because it is stable across
# processes and Julia versions (unlike `hash`, which can fold object identities).
function _fnv1a64(bytes)
    h = 0xcbf29ce484222325
    for b in bytes
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end

_sanitize_name(s::AbstractString) = replace(String(s), r"[^A-Za-z0-9_]" => "_")

"""
    weights_digest(key, specs; content=0) -> UInt64

Digest over a model's identity, weight layout (name, per-tensor dtype/size/shape, a format
version), and a `content` token identifying the weight file's on-disk version (see
[`weights_file_token`](@ref)). Two workers computing this for the same model and the same file
agree on the same region key. Without the content token a weights-only update (same name, same
tensor layout) would collide with the previous version's region and silently serve stale weights,
including across server restarts (regions in `/dev/shm` outlive the process by design).
"""
function weights_digest(key::AbstractString, specs; content::UInt64 = UInt64(0))
    io = IOBuffer()
    print(io, "rsw2|", key, "|c", string(content; base = 16))
    for (T, dims) in specs
        print(io, "|", string(T), ":", sizeof(T), ":", join(dims, ","))
    end
    return _fnv1a64(take!(io))
end

"""
    WeightsFileId

Which on-disk version of a weights file a process read: its device, inode, size and mtime. Take it
from the open file (`weights_file_id(stat(io))`) at the moment the file is mapped, so it names the
bytes actually mapped even if the path is replaced afterwards. [`NO_WEIGHTS_FILE`](@ref) stands for
no file (hand-built entries).
"""
struct WeightsFileId
    device::UInt64
    inode::UInt64
    size::Int64
    mtime::Float64
end

"""
    NO_WEIGHTS_FILE

The [`WeightsFileId`](@ref) of an entry with no weights file; its token is 0.
"""
const NO_WEIGHTS_FILE = WeightsFileId(0, 0, 0, 0.0)

weights_file_id(st::Base.Filesystem.StatStruct) =
    st.mtime == 0 ? NO_WEIGHTS_FILE : WeightsFileId(UInt64(st.device), UInt64(st.inode), Int64(st.size), st.mtime)

"""
    weights_file_id(path) -> WeightsFileId

The [`WeightsFileId`](@ref) of whatever the path names now; [`NO_WEIGHTS_FILE`](@ref) for an empty
or missing path.
"""
weights_file_id(path::AbstractString) = isempty(path) ? NO_WEIGHTS_FILE : weights_file_id(stat(String(path)))

"""
    weights_file_token(id_or_path) -> UInt64

Identity token for one on-disk version of a weights file, folded into [`weights_digest`](@ref) via
its `content` keyword. Derived from the device, inode, size and mtime: cheap (no content read, so
the shared store's zero-copy attach path stays zero-read) and identical across same-node workers
that mapped the same file. The inode makes an atomic replacement (write a new file, rename it over
the old one) a new version even when size and mtime match. Pass the [`WeightsFileId`](@ref) taken
when the file was mapped; a path is stat'ed now, which can name a newer file than the one mapped.
Returns 0 for no file (hand-built test entries), which reproduces the layout-only digest.
"""
function weights_file_token(id::WeightsFileId)
    id == NO_WEIGHTS_FILE && return UInt64(0)
    io = IOBuffer()
    print(io, id.device, ":", id.inode, ":", id.size, ":", id.mtime)
    return _fnv1a64(take!(io))
end
weights_file_token(path::AbstractString) = weights_file_token(weights_file_id(path))

"""
    SharedStoreFull

Raised by `materialize_host_weights!` on a [`SharedWeightStore`](@ref) when the node cannot back a
new region: `/dev/shm` is full or the memory limit refuses its pages. Nothing is left behind (the
partial region is unlinked); the caller can fall back to a private copy.
"""
struct SharedStoreFull <: Exception
    name::String
    bytes::Int
    errno::Cint
end
Base.showerror(io::IO, e::SharedStoreFull) = print(
    io, "shared weight region ", e.name, " (", Base.format_bytes(e.bytes), ") could not be reserved: ",
    Libc.strerror(e.errno)
)

# Back a freshly created region with real pages before writing to it. Sizing a POSIX shared memory
# object only sets its length (ftruncate); its pages are allocated on first touch, and when
# /dev/shm is full or the memory cgroup refuses a page, that touch kills the process with SIGBUS.
# posix_fallocate allocates them up front and reports failure as an error number instead. Linux
# only (/dev/shm is where the region's file lives); elsewhere it reports success. Returns 0 or an
# errno. A Ref so tests can simulate a full /dev/shm.
function _posix_fallocate(name::AbstractString, bytes::Int)
    Sys.islinux() || return Cint(0)
    fd = ccall(:open, Cint, (Cstring, Cint, Cint), "/dev/shm" * name, _O_RDWR, Cint(0))
    fd < 0 && return Cint(Libc.errno())
    rc = @ccall gc_safe = true posix_fallocate(fd::Cint, 0::Int64, Int64(bytes)::Int64)::Cint
    _close_fd(fd)
    return rc
end
const _RESERVE_REGION = Ref{Function}(_posix_fallocate)

# (offsets per tensor, total region bytes) from the layout specs.
function _layout(specs)
    offs = Int[]
    off = _RSW_DATA_OFFSET
    for (T, dims) in specs
        push!(offs, off)
        off += _align(sizeof(T) * prod(dims))
    end
    return offs, off
end

# Build host Arrays aliasing the region at each tensor's offset (zero-copy). The caller keeps the
# SharedMemory alive (in `attached`) for as long as these arrays are used.
function _build_arrays(shm::SharedMemory, specs, offs)
    base = convert(Ptr{UInt8}, pointer(shm))
    arrays = Any[]
    for ((T, dims), o) in zip(specs, offs)
        push!(arrays, unsafe_wrap(Array, convert(Ptr{T}, base + o), Tuple(dims); own = false))
    end
    return arrays
end

_hdr_read(p::Ptr{UInt8}, ::Type{T}, off::Int) where {T} = unsafe_load(convert(Ptr{T}, p + off))
_hdr_write!(p::Ptr{UInt8}, v::T, off::Int) where {T} = unsafe_store!(convert(Ptr{T}, p + off), v)
_is_ready(p::Ptr{UInt8}, digest::UInt64) =
    _hdr_read(p, UInt32, 0) == _RSW_MAGIC && _hdr_read(p, UInt32, 4) == UInt32(1) &&
    _hdr_read(p, UInt64, 8) == digest

"""
    materialize_host_weights!(store, key, digest, specs, fill!) -> Vector{Any}

Return a model's host weight Arrays (in weight order), populating them via `fill!(arrays)` when
this worker is the one that must materialize them. `specs` is a vector of `(eltype, dims)` per
tensor. For [`PrivateWeightStore`](@ref) the arrays are freshly allocated; for
[`SharedWeightStore`](@ref) they alias a node-shared region.
"""
function materialize_host_weights!(::PrivateWeightStore, key, digest, specs, fill!)
    arrays = Any[Array{T}(undef, Tuple(dims)) for (T, dims) in specs]
    fill!(arrays)
    return arrays
end

# Attach an existing READY region with the digest, or return nothing. Caller holds some flock.
function _attach_ready(name, digest::UInt64, total::Int)
    existing = try
        SharedMemory(name)
    catch
        return nothing
    end
    if sizeof(existing) >= total && _is_ready(convert(Ptr{UInt8}, pointer(existing)), digest)
        return existing
    end
    finalize(existing)                      # munmap the stale/partial mapping
    return nothing
end

# Unlink stale sibling versions of one model's region: same sanitized model name, a different
# digest, and no holders (a non-blocking exclusive flock on the sibling's lock file succeeds).
# Called by a worker right after it has attached its own region, while holding that region's
# shared lock, so the version being kept can never be swept by a peer running the same code.
# Content-keyed digests orphan the previous version's region when a model's weights are updated;
# live holders release it through the normal last-one-out unlink, but a crashed run would leak it
# in /dev/shm (a RAM-backed tmpfs) forever. This reclaims exactly those, touching nothing in use.
function _sweep_stale_versions(store::SharedWeightStore, key::AbstractString, keep_digest::UInt64)
    san = _sanitize_name(key)
    keep = store.prefix * "-" * san * "-" * string(keep_digest; base = 16)
    rx = Regex("^" * store.prefix * "-" * san * "-[0-9a-f]+\$")
    entries = try
        readdir("/dev/shm")
    catch
        return nothing
    end
    for f in entries
        (f != keep && match(rx, f) !== nothing) || continue
        fd = _open_lock("/dev/shm/" * f * ".lock", store.mode)
        fd < 0 && continue
        if _flock(fd, _LOCK_EX | _LOCK_NB) == 0   # no holders: safe to unlink
            _shm_unlink("/" * f)
            _flock(fd, _LOCK_UN)
        end
        _close_fd(fd)
    end
    return nothing
end

# Blocking exclusive lock on the node budget lock file; returns its fd. Held only for the
# bookkeeping around a region's creation, never across a fill.
function _lock_budget(store::SharedWeightStore)
    path = "/dev/shm/" * store.prefix * "-budget.lock"
    fd = _open_lock(path, store.mode)
    fd < 0 && throw(SystemError("open weight-store budget lock $path"))
    while _flock(fd, _LOCK_EX) != 0
        if Libc.errno() != Libc.EINTR
            err = Libc.errno()
            _close_fd(fd)
            throw(SystemError("locking $path", err))
        end
    end
    return fd
end

function _unlock_budget(fd::Cint)
    _flock(fd, _LOCK_UN)
    _close_fd(fd)
    return nothing
end

# Every weight region in /dev/shm: (file name, bytes, last load time). The last load time is the
# region lock file's mtime (0 when it has none, so an orphan is evicted first).
function _regions(store::SharedWeightStore)
    rx = Regex("^" * store.prefix * "-[A-Za-z0-9_]+-[0-9a-f]+\$")
    out = Tuple{String, Int, Float64}[]
    entries = try
        readdir("/dev/shm")
    catch
        return out
    end
    for f in entries
        match(rx, f) === nothing && continue
        st = stat("/dev/shm/" * f)
        isfile(st) || continue
        lk = stat("/dev/shm/" * f * ".lock")
        push!(out, (f, Int(st.size), isfile(lk) ? lk.mtime : 0.0))
    end
    return out
end

"""
    host_cache_bytes(store) -> Int

Bytes held by every weight region on the node (the quantity the node budget bounds). 0 for the
private store.
"""
host_cache_bytes(::WeightStore) = 0
host_cache_bytes(store::SharedWeightStore) = sum((r[2] for r in _regions(store)); init = 0)

# Evict holder-free regions, least recently loaded first, until `need` more bytes fit the budget.
# The caller holds the budget lock. `keep` (a region file name) is never evicted. Evicts nothing
# unless that makes the bytes fit: every candidate is locked first (a non-blocking LOCK_EX succeeds
# only with no holder and no creator), and the locks are dropped untouched when the evictable total
# is short, unless `partial` (a trim), which evicts what it can. Returns whether the bytes fit, and
# the number of regions evicted.
function _make_room!(store::SharedWeightStore, need::Int; keep::AbstractString = "", partial::Bool = false)
    budget = store.budget::Int
    regions = _regions(store)
    total = sum((r[2] for r in regions); init = 0)
    total + need <= budget && return true, 0
    sort!(regions; by = r -> r[3])
    locked = Tuple{String, Int, Cint}[]
    freeable = 0
    for (f, bytes, _) in regions
        f == keep && continue
        total - freeable + need <= budget && break
        fd = _open_lock("/dev/shm/" * f * ".lock", store.mode)
        fd < 0 && continue
        if _flock(fd, _LOCK_EX | _LOCK_NB) == 0
            push!(locked, (f, bytes, fd))
            freeable += bytes
        else
            _close_fd(fd)
        end
    end
    fits = total - freeable + need <= budget
    evict = fits || partial
    for (f, _, fd) in locked
        evict && _shm_unlink("/" * f)
        _flock(fd, _LOCK_UN)
        _close_fd(fd)
    end
    return fits, evict ? length(locked) : 0
end

"""
    trim_host_cache!(store) -> Int

Evict holder-free regions until the node fits the store's budget (a lowered budget, or regions an
earlier run left behind). Returns the number evicted; a no-op without a budget.
"""
trim_host_cache!(::WeightStore) = 0
function trim_host_cache!(store::SharedWeightStore)
    store.budget === nothing && return 0
    bfd = _lock_budget(store)
    try
        return _make_room!(store, 0; partial = true)[2]
    finally
        _unlock_budget(bfd)
    end
end

# Attach a READY region for `digest`, or create and fill it, and return it with a shared lock held:
# `(; shm, fd, name, lockpath, arrays, hit)`. Under a budget a new region is admitted first; when
# it cannot fit, a `pinned` caller creates it anyway (with a warning: a pinned floor is a
# guarantee) and any other caller gets `nothing`, holding no lock.
function _acquire_region(store::SharedWeightStore, key, digest::UInt64, specs, fill!; pinned::Bool)
    offs, total = _layout(specs)
    tag = _sanitize_name(key) * "-" * string(digest; base = 16)
    name = "/" * store.prefix * "-" * tag
    lockpath = "/dev/shm/" * store.prefix * "-" * tag * ".lock"

    fd = _open_lock(lockpath, store.mode)
    fd < 0 && throw(SystemError("open weight-store lock $lockpath"))
    local shm::SharedMemory
    local arrays
    created = false                 # true once this call created the region (and must clean it up on failure)
    hit = true
    bfd = Cint(-1)                  # the budget lock, while this call holds it
    try
        while true
            # Fast path: shared lock, reuse a published region (readers run concurrently).
            _flock(fd, _LOCK_SH)
            ready = _attach_ready(name, digest, total)
            if ready !== nothing
                shm = ready
                arrays = _build_arrays(shm, specs, offs)
                break
            end
            _flock(fd, _LOCK_UN)
            # Slow path: try to become the creator without blocking behind shared holders.
            if _flock(fd, _LOCK_EX | _LOCK_NB) == 0
                ready2 = _attach_ready(name, digest, total)
                if ready2 !== nothing
                    shm = ready2
                    arrays = _build_arrays(shm, specs, offs)
                    _flock(fd, _LOCK_SH)            # downgrade, hold while in use
                    break
                end
                hit = false
                # Remove any non-ready remnant, then create exclusively and populate.
                stale = try
                    SharedMemory(name)
                catch
                    nothing
                end
                if stale !== nothing
                    finalize(stale)
                    _shm_unlink(name)
                end
                if store.budget !== nothing
                    bfd = _lock_budget(store)
                    fits, _ = _make_room!(store, total; keep = store.prefix * "-" * tag)
                    if !fits
                        if !pinned
                            _unlock_budget(bfd)
                            bfd = Cint(-1)
                            _flock(fd, _LOCK_UN)
                            _close_fd(fd)
                            return nothing
                        end
                        @warn "host weight budget overcommitted: pinned (residency: system) weights exceed it" model = String(key) budget = Base.format_bytes(store.budget) cached = Base.format_bytes(host_cache_bytes(store)) adding = Base.format_bytes(total)
                    end
                end
                shm = SharedMemory(name, total; perms = store.mode, volatile = false)
                # A signal can interrupt a large reservation; tmpfs undoes the partial allocation and
                # reports EINTR, so retry rather than mistake it for a full /dev/shm.
                rc = _RESERVE_REGION[](name, total)
                while rc == Libc.EINTR
                    rc = _RESERVE_REGION[](name, total)
                end
                if rc != 0
                    finalize(shm)
                    _shm_unlink(name)
                    throw(SharedStoreFull(name, total, rc))
                end
                created = true
                # The region now counts toward the budget; let other admitters proceed during the fill.
                bfd >= 0 && (_unlock_budget(bfd); bfd = Cint(-1))
                p = convert(Ptr{UInt8}, pointer(shm))
                _hdr_write!(p, _RSW_MAGIC, 0)
                _hdr_write!(p, UInt32(0), 4)        # ready = 0 until the data is durable
                _hdr_write!(p, digest, 8)
                arrays = _build_arrays(shm, specs, offs)
                fill!(arrays)
                _msync(shm)
                _hdr_write!(p, UInt32(1), 4)        # READY written last
                _msync(shm)
                _flock(fd, _LOCK_SH)                # downgrade, hold while in use
                break
            end
            # Another worker is mid-create; back off and retry the fast path until it publishes.
            sleep(0.005)
        end
    catch
        bfd >= 0 && _unlock_budget(bfd)
        # A region this call created but did not publish (the fill threw) is removed now, while the
        # exclusive lock is still held, rather than left for the next creator to find.
        if created
            finalize(shm)
            _shm_unlink(name)
        end
        _flock(fd, _LOCK_UN)
        _close_fd(fd)
        rethrow()
    end
    # A load from this region: it becomes the most recently loaded for eviction.
    store.budget === nothing || try
        touch(lockpath)
    catch
    end
    return (; shm, fd, name, lockpath, arrays, hit)
end

function materialize_host_weights!(store::SharedWeightStore, key, digest::UInt64, specs, fill!)
    r = _acquire_region(store, key, digest, specs, fill!; pinned = true)::NamedTuple
    @lock store.lock store.attached[String(key)] = (; r.shm, r.fd, r.name, r.lockpath)
    # Our region is attached (shared lock held), so reclaim any holder-free older versions of this
    # model now: a weights update moved the digest, and only crashed runs leave the old ones behind.
    _sweep_stale_versions(store, String(key), digest)
    return r.arrays
end

"""
    with_cached_host_weights(f, store, key, digest, specs, fill!) -> Union{Tuple{Bool, Any}, Nothing}

Run `f(arrays)` on a model's host weights from the node cache. The store must have a budget. The
region is held only while `f` runs (an upload) and then detached without unlinking, so it stays
cached for the next load by any worker. A missing region
is admitted (evicting least recently loaded ones) and filled via `fill!(arrays)` first. Returns
`(hit, f(arrays))`, where `hit` says the region already existed, or `nothing` when the region
cannot fit the budget; the caller then loads without the cache.
"""
function with_cached_host_weights(f, store::SharedWeightStore, key, digest::UInt64, specs, fill!)
    r = _acquire_region(store, key, digest, specs, fill!; pinned = false)
    r === nothing && return nothing
    try
        return (r.hit, f(r.arrays))
    finally
        finalize(r.shm)                     # munmap; the region itself stays cached
        _flock(r.fd, _LOCK_UN)
        _close_fd(r.fd)
        r.hit || _sweep_stale_versions(store, String(key), digest)
    end
end

"""
    release_host_weights!(store, key) -> nothing

Release a model's host weights. For [`SharedWeightStore`](@ref) this detaches the region and,
if this was the last holder on the node (a non-blocking upgrade to an exclusive flock succeeds),
unlinks the region and its lock file. A no-op for the private store. The caller must drop all
references to the arrays first.
"""
release_host_weights!(::PrivateWeightStore, key) = nothing

"""
    rename_host_weights!(store, old, new) -> nothing

Rekey a model's attached host-weight region from `old` to `new` (a model rename; the weights are
unchanged). The region itself keeps its original content-addressed SHM name; only this worker's
bookkeeping key moves, so a later `release_host_weights!(store, new)` detaches the same region.
A no-op for the private store and when nothing is attached under `old`.
"""
rename_host_weights!(::PrivateWeightStore, old, new) = nothing

function rename_host_weights!(store::SharedWeightStore, old, new)
    @lock store.lock begin
        ent = pop!(store.attached, String(old), nothing)
        ent === nothing || (store.attached[String(new)] = ent)
    end
    return nothing
end

function release_host_weights!(store::SharedWeightStore, key)
    ent = @lock store.lock get(store.attached, String(key), nothing)
    ent === nothing && return nothing
    # Under a node budget the region stays cached when released; eviction reclaims it.
    last_holder = store.budget === nothing && _flock(ent.fd, _LOCK_EX | _LOCK_NB) == 0   # success => no other shared holders
    finalize(ent.shm)                                        # munmap our mapping
    # Last holder unlinks the region. The lock file is left in place (tiny, reused); unlinking it
    # would race a worker that just opened it, so we keep it.
    last_holder && _shm_unlink(ent.name)
    _flock(ent.fd, _LOCK_UN)
    _close_fd(ent.fd)
    @lock store.lock delete!(store.attached, String(key))
    return nothing
end
