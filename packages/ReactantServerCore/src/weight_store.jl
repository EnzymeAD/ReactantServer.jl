# Host-RAM residency for a model's materialized weights, with an optional node-shared backend.
#
# A model that is pinned in system RAM keeps its weights materialized as host Arrays. The default
# `PrivateWeightStore` gives each worker its own copy. `SharedWeightStore` (opt-in via
# `shared_host_weights`) places one copy per model in a node-level POSIX shared-memory region so
# that several same-node GPU workers share it, each transferring to its own device from the shared
# mapping.
#
# Coordination is peer-to-peer with no daemon; crash safety comes from an advisory file lock
# (`flock`), never an in-band flag. The region is content-addressed (`/rsw-<model>-<digest>`), so
# any fully-populated region with a matching digest is immutable and safe to reuse, including one
# left by a prior run. The lock file lives in `/dev/shm` next to the region: with Docker `ipc:
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
    SharedWeightStore(; mode=0o666)

Opt-in store backing each model's host weights with a node-level POSIX shared-memory region so
same-node workers share one copy. See the file header for the flock-coordinated protocol.
`mode` sets the permission bits for the regions and their lock files. The default `0o666`
lets workers running as unrelated UIDs (for example different containers) share the regions,
but is world-writable; `0o660` is recommended for production and multi-user systems.
"""
mutable struct SharedWeightStore <: WeightStore
    attached::Dict{String,NamedTuple{(:shm, :fd, :name, :lockpath),Tuple{SharedMemory,Cint,String,String}}}
    lock::ReentrantLock
    mode::UInt16                  # POSIX permission bits for the regions and lock files
end
SharedWeightStore(; mode::Integer=0o666) = SharedWeightStore(
    Dict{String,NamedTuple{(:shm, :fd, :name, :lockpath),Tuple{SharedMemory,Cint,String,String}}}(),
    ReentrantLock(), UInt16(mode))

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

_flock(fd::Cint, op::Cint) = ccall(:flock, Cint, (Cint, Cint), fd, op)
_open_lock(path::AbstractString, mode::Integer=0o666) =
    ccall(:open, Cint, (Cstring, Cint, Cint), path, _O_RDWR | _O_CREAT, Cint(mode))
_close_fd(fd::Cint) = ccall(:close, Cint, (Cint,), fd)
_shm_unlink(name::AbstractString) = ccall(:shm_unlink, Cint, (Cstring,), name)
_msync(shm::SharedMemory) =
    ccall(:msync, Cint, (Ptr{Cvoid}, Csize_t, Cint), pointer(shm), Csize_t(sizeof(shm)), _MS_SYNC)

_align(n::Integer, a::Integer=_RSW_ALIGN) = ((Int(n) + a - 1) ÷ a) * a

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
    weights_digest(key, specs) -> UInt64

Content digest over a model's identity and weight layout (name, per-tensor dtype/size/shape, and
a format version). Two workers computing this for the same model agree on the same region key.
"""
function weights_digest(key::AbstractString, specs)
    io = IOBuffer()
    print(io, "rsw1|", key)
    for (T, dims) in specs
        print(io, "|", string(T), ":", sizeof(T), ":", join(dims, ","))
    end
    return _fnv1a64(take!(io))
end

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
        push!(arrays, unsafe_wrap(Array, convert(Ptr{T}, base + o), Tuple(dims); own=false))
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

function materialize_host_weights!(store::SharedWeightStore, key, digest::UInt64, specs, fill!)
    offs, total = _layout(specs)
    tag = _sanitize_name(key) * "-" * string(digest; base=16)
    name = "/rsw-" * tag
    lockpath = "/dev/shm/rsw-" * tag * ".lock"

    fd = _open_lock(lockpath, store.mode)
    fd < 0 && throw(SystemError("open weight-store lock $lockpath"))
    local shm::SharedMemory
    local arrays
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
                    _flock(fd, _LOCK_SH)            # downgrade, hold while pinned
                    break
                end
                # Remove any non-ready remnant, then create exclusively and populate.
                stale = try SharedMemory(name) catch; nothing end
                if stale !== nothing
                    finalize(stale)
                    _shm_unlink(name)
                end
                shm = SharedMemory(name, total; perms=store.mode, volatile=false)
                p = convert(Ptr{UInt8}, pointer(shm))
                _hdr_write!(p, _RSW_MAGIC, 0)
                _hdr_write!(p, UInt32(0), 4)        # ready = 0 until the data is durable
                _hdr_write!(p, digest, 8)
                arrays = _build_arrays(shm, specs, offs)
                fill!(arrays)
                _msync(shm)
                _hdr_write!(p, UInt32(1), 4)        # READY written last
                _msync(shm)
                _flock(fd, _LOCK_SH)                # downgrade, hold while pinned
                break
            end
            # Another worker is mid-create; back off and retry the fast path until it publishes.
            sleep(0.005)
        end
    catch
        _flock(fd, _LOCK_UN)
        _close_fd(fd)
        rethrow()
    end
    @lock store.lock store.attached[String(key)] = (; shm, fd, name, lockpath)
    return arrays
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
    last_holder = _flock(ent.fd, _LOCK_EX | _LOCK_NB) == 0   # success => no other shared holders
    finalize(ent.shm)                                        # munmap our mapping
    # Last holder unlinks the region. The lock file is left in place (tiny, reused); unlinking it
    # would race a worker that just opened it, so we keep it.
    last_holder && _shm_unlink(ent.name)
    _flock(ent.fd, _LOCK_UN)
    _close_fd(ent.fd)
    @lock store.lock delete!(store.attached, String(key))
    return nothing
end
