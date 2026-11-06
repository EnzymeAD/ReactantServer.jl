# Server side of the Triton-compatible system shared-memory extension.
#
# Clients create a POSIX shared-memory object, register it (name, key, offset, byte_size),
# and then reference it from input/output tensors. This file attaches the client's region
# read-write and hands out byte views. The data plane is one copy from the mapped region into
# a host array for input, and one copy back into the region for output.
#
# Concurrency: the registry-wide lock guards only the regions Dict (register/unregister/lookup
# are short critical sections). Each ShmRegion carries its own lock, so concurrent shm_read /
# shm_write! against *distinct* regions proceed in parallel. A read/write and an unregister of
# the *same* region serialize on that region's lock, and unregister flips `attached` to false
# under that lock before detaching, so a copy can never touch a munmap'd mapping (no
# use-after-free).
#
# Identity: a registration records the (device, inode) of the object it mapped, read from the
# descriptor it mapped (see shm_object.jl), and keeps that descriptor. Registering a name again is
# decided by object, never by key string (CORE-2): the same object is an idempotent success, a key
# that no longer resolves to the registered object (unlinked, or unlinked and created again) is
# stale and replaced, and a different object that still exists is refused (ALREADY_EXISTS). Stale
# registrations are also dropped at every registration and periodically (CORE-11).

# Thrown when a tensor references a shared-memory region name the registry does not know. This is
# a distinct type (not a bare ArgumentError) so the transport can map it to a distinct gRPC status
# (FAILED_PRECONDITION): it is the exact signal a client gets after a server restart wiped the
# in-memory registry, and it is recoverable by re-registering, unlike a malformed-argument error.
struct UnregisteredRegionError <: Exception
    name::String
end
Base.showerror(io::IO, e::UnregisteredRegionError) =
    print(io, "unregistered shared memory region: ", e.name)

"""
    ShmRegionExistsError(name, key)

Raised by [`shm_register!`](@ref) when `name` is already registered for a different shared-memory
object that still exists (`key` is the key it is registered with). The transport maps it to
ALREADY_EXISTS, as Triton does; nothing about the existing registration changes.
"""
struct ShmRegionExistsError <: Exception
    name::String
    key::String
end
Base.showerror(io::IO, e::ShmRegionExistsError) =
    print(
    io, "shared memory region '", e.name, "' is already registered for a different object ",
    "that still exists (key '", e.key, "')"
)

mutable struct ShmRegion
    const name::String
    const key::String
    const offset::Int
    const byte_size::Int
    const fd::ShmFd                  # the descriptor the object was mapped through, kept open
    const ptr::Ptr{UInt8}            # base of the mapping
    const len::Int                   # mapped length: the object's size when it was registered
    const identity::ShmIdentity      # (device, inode) of the mapped object, read from `fd`
    const lock::ReentrantLock
    attached::Bool
    function ShmRegion(name, key, offset, byte_size, fd, ptr, len, identity)
        r = new(name, key, offset, byte_size, fd, ptr, len, identity, ReentrantLock(), true)
        return finalizer(_release_mapping!, r)
    end
end

# Unmap and close. Called by `_detach!` under `r.lock`, or by the GC once the region is unreachable,
# when nothing can hold its lock any more.
function _release_mapping!(r::ShmRegion)
    r.attached || return nothing
    r.attached = false
    _shm_munmap(r.ptr, r.len)
    close(r.fd)
    return nothing
end

"""
    SHM_REAP_INTERVAL_SECONDS

How often (seconds) a [`SharedMemoryRegistry`](@ref) drops registrations whose object was unlinked
or recreated (CORE-11). Low frequency on purpose: a check is one `shm_open` and `fstat` per region.
"""
const SHM_REAP_INTERVAL_SECONDS = 60.0

"""
    SharedMemoryRegistry(; reap_interval = SHM_REAP_INTERVAL_SECONDS)

The registry of client shared-memory regions. Once something is registered, a timer re-checks
every `reap_interval` seconds that each registration's object still exists under its key and drops
those that do not (see [`shm_reap!`](@ref)); `shm_teardown!` stops it. A non-positive
interval disables the timer (registrations still sweep).
"""
struct SharedMemoryRegistry
    regions::Dict{String, ShmRegion}
    lock::ReentrantLock
    reap_interval::Float64
    reaper::Base.RefValue{Union{Nothing, Timer}}
end
SharedMemoryRegistry(; reap_interval::Real = SHM_REAP_INTERVAL_SECONDS) =
    SharedMemoryRegistry(
    Dict{String, ShmRegion}(), ReentrantLock(), Float64(reap_interval),
    Ref{Union{Nothing, Timer}}(nothing)
)

# Detach our mapping of a region. Takes `r.lock`, so it waits for a copy in progress, and flips
# `attached` so any reader blocked on the same lock bails instead of touching the munmap'd page.
# This munmaps and closes our descriptor only; it never shm_unlinks, so the client's underlying
# object is left intact.
function _detach!(r::ShmRegion)
    @lock r.lock _release_mapping!(r)
    return
end

# Whether `r`'s object is still the one its key names. A key that is gone, or resolves to another
# inode, means the client unlinked (and maybe recreated) it: the registration is stale. An open
# that fails for any other reason counts as live, so an existing registration is never replaced or
# dropped on an ambiguous answer.
function _object_live(r::ShmRegion)
    id = shm_key_identity(r.key)
    id === :gone && return false
    id isa ShmIdentity && return id == r.identity
    return true
end

"""
    shm_register!(registry, name, key, offset, byte_size)

Attach the existing POSIX shared-memory object `key` read-write and register it under `name`. The
worker opens `key`, records the (device, inode) of the object through the descriptor it then maps,
and keeps that descriptor. When `name` is already registered:

- the same object (equal device and inode): idempotent success; the registration is kept, or
  replaced by a fresh mapping of the same object when `offset` or `byte_size` changed;
- the registered key no longer resolves to the registered object (it was unlinked, or unlinked and
  created again): the registration is stale and is replaced;
- a different object that still exists: [`ShmRegionExistsError`](@ref) (ALREADY_EXISTS), and the
  existing registration is untouched.

Every registration also sweeps the registry for stale entries ([`shm_reap!`](@ref)).
"""
function shm_register!(
        reg::SharedMemoryRegistry, name::AbstractString, key::AbstractString,
        offset::Integer, byte_size::Integer
    )
    isempty(name) && throw(ArgumentError("shared memory region name must not be empty"))
    isempty(key) && throw(ArgumentError("shared memory key must not be empty"))
    # Validate and narrow the untrusted sizes before any mapping exists, so an out-of-range
    # value is a clean ArgumentError rather than an InexactError after the mmap was created.
    (0 <= offset <= typemax(Int)) ||
        throw(ArgumentError("shared memory offset must be non-negative and fit in Int64, got $offset"))
    (0 < byte_size <= typemax(Int)) ||
        throw(ArgumentError("shared memory byte_size must be positive and fit in Int64, got $byte_size"))
    off = Int(offset)
    bs = Int(byte_size)

    fd, ptr, total, identity = shm_open_existing(String(key))
    # Subtraction form: `off + bs` could wrap for large values.
    if bs > total || off > total - bs
        _shm_munmap(ptr, total)
        close(fd)
        throw(
            ArgumentError(
                "region '$name' offset+byte_size exceeds " *
                    "shared memory object '$key' size ($total)"
            )
        )
    end
    new = ShmRegion(String(name), String(key), off, bs, fd, ptr, total, identity)

    # The decision and the insertion are one critical section, so two concurrent registrations of a
    # name cannot both act on the same previous state. `_object_live` is one shm_open and fstat.
    decision, prev = @lock reg.lock begin
        prev = get(reg.regions, name, nothing)
        d = if prev === nothing
            :registered
        elseif prev.identity == identity
            (prev.offset == off && prev.byte_size == bs) ? :same : :window
        elseif _object_live(prev)
            :exists
        else
            :stale
        end
        d in (:registered, :window, :stale) && (reg.regions[name] = new)
        d, prev
    end
    if decision === :same || decision === :exists
        _detach!(new)                      # keep the existing registration and its mapping
    elseif prev !== nothing
        _detach!(prev)                     # requests bound to it fail as unregistered
    end
    decision === :exists && throw(ShmRegionExistsError(String(name), prev.key))
    _start_reaper!(reg)
    shm_reap!(reg)
    return nothing
end

"""
    shm_reap!(registry) -> Vector{String}

Drop every registration whose key no longer resolves to the object the worker mapped (the client
unlinked it, or unlinked it and created another under the same key), and return the names dropped.
A normal client exit unlinks its objects, so this is what eventually releases their mappings; an
object a crashed client left linked cannot be told from a live one and stays registered. Runs at
every registration and on the registry's timer.
"""
function shm_reap!(reg::SharedMemoryRegistry)
    stale = ShmRegion[r for r in values(shm_regions(reg)) if !_object_live(r)]   # outside the lock
    isempty(stale) && return String[]
    removed = @lock reg.lock begin
        out = ShmRegion[]
        for r in stale
            # Skip a name that was unregistered or registered again since the check above.
            get(reg.regions, r.name, nothing) === r || continue
            delete!(reg.regions, r.name)
            push!(out, r)
        end
        out
    end
    for r in removed
        _detach!(r)
    end
    names = String[r.name for r in removed]
    isempty(names) || @info "shm: dropped registrations whose object is gone" regions = names
    return names
end

# Start the periodic sweep once, on first registration. Never while generating precompile output,
# where an open timer would keep the process waiting.
function _start_reaper!(reg::SharedMemoryRegistry)
    ccall(:jl_generating_output, Cint, ()) == 0 || return
    reg.reap_interval > 0 || return
    @lock reg.lock begin
        reg.reaper[] === nothing || return
        reg.reaper[] = Timer(reg.reap_interval; interval = reg.reap_interval) do _
            # The sweep can wait on a region lock behind a large copy, so it runs on the default
            # pool rather than on the timer's task.
            errormonitor(
                Threads.@spawn try
                    shm_reap!(reg)
                catch err
                    @warn "shm: periodic sweep of stale registrations failed" exception = err
                end
            )
        end
    end
    return
end

function _stop_reaper!(reg::SharedMemoryRegistry)
    t = @lock reg.lock begin
        t = reg.reaper[]
        reg.reaper[] = nothing
        t
    end
    t === nothing || close(t)
    return
end

"""
    shm_unregister!(registry, name)

Unregister and detach a region, or all regions when `name` is empty. Idempotent: unregistering a
name that is not registered is a successful no-op (it matches KServe semantics and lets the gateway
fan-out and its rollback unregister succeed without a spurious error).
Registration, by contrast, fails loudly (see [`shm_register!`](@ref)) so a bad region surfaces at
register time rather than during inference.
"""
function shm_unregister!(reg::SharedMemoryRegistry, name::AbstractString)
    removed = @lock reg.lock begin
        if isempty(name)
            rs = collect(values(reg.regions))
            empty!(reg.regions)
            rs
        else
            r = get(reg.regions, name, nothing)
            if r === nothing
                ShmRegion[]                 # not registered: idempotent no-op
            else
                delete!(reg.regions, name)
                ShmRegion[r]
            end
        end
    end
    for r in removed
        _detach!(r)
    end
    return nothing
end

"""
    same_ipc_namespace(name) -> Bool

Return whether the POSIX shared-memory object `name` is visible in this process's IPC
namespace. The client passes the name of an object it created; we answer `true` only if we can
open that same object ourselves, which is what determines whether system shared-memory transport
can work between the two processes. The open is read-only and detached immediately; nothing is
registered or kept mapped. Any failure (object absent because we are in a different namespace,
permission error, malformed name) is reported as `false`.
"""
function same_ipc_namespace(name::AbstractString)
    isempty(name) && return false
    shm = try
        IPC.SharedMemory(String(name); readonly = true)
    catch
        return false
    end
    finalize(shm)
    return true
end

shm_regions(reg::SharedMemoryRegistry) = (@lock reg.lock copy(reg.regions))

# Unregister everything and stop the periodic sweep.
shm_teardown!(reg::SharedMemoryRegistry) = (_stop_reaper!(reg); shm_unregister!(reg, ""))

# Look up a region by name under the registry lock. The returned region carries its own lock;
# the caller takes that lock and re-checks `attached` before touching the mapping.
function _lookup_region(reg::SharedMemoryRegistry, name::AbstractString)
    r = @lock reg.lock get(reg.regions, name, nothing)
    r === nothing && throw(UnregisteredRegionError(String(name)))
    return r
end

# Bounds-checked raw byte base within a region. Caller MUST hold `r.lock` and have verified
# `r.attached`.
function _region_base(r::ShmRegion, offset::Integer, byte_size::Integer)
    # Range-check the untrusted values before narrowing, and use the subtraction form for the
    # bounds check: `off + bs` could wrap for hostile int64 parameters.
    (0 <= offset <= typemax(Int) && 0 <= byte_size <= typemax(Int)) ||
        throw(
        ArgumentError(
            "shared memory access (offset $offset, byte_size $byte_size) is out " *
                "of bounds for region '$(r.name)' of $(r.byte_size) bytes"
        )
    )
    off = Int(offset)
    bs = Int(byte_size)
    (bs <= r.byte_size && off <= r.byte_size - bs) ||
        throw(
        ArgumentError(
            "shared memory access [$off, $off + $bs) is out of bounds for region " *
                "'$(r.name)' of $(r.byte_size) bytes"
        )
    )
    return r.ptr + r.offset + off, bs
end

"""
    shm_read(registry, name, offset, byte_size) -> Vector{UInt8}

Copy `[offset, offset+byte_size)` of the named region into a fresh byte vector. The copy is
done while the region's lock is held, so a concurrent `shm_unregister!` of the same region
cannot detach the mapping underneath the read.
"""
function shm_read(reg::SharedMemoryRegistry, name::AbstractString, offset::Integer, byte_size::Integer)
    r = _lookup_region(reg, name)
    @lock r.lock begin
        r.attached || throw(UnregisteredRegionError(String(name)))
        base, bs = _region_base(r, offset, byte_size)
        out = Vector{UInt8}(undef, bs)
        GC.@preserve r out unsafe_copyto!(pointer(out), base, bs)
        return out
    end
end

"""
    shm_write!(registry, name, offset, bytes)

Copy `bytes` into `[offset, offset+length(bytes))` of the named region. The copy is done while
the region's lock is held, so a concurrent `shm_unregister!` of the same region cannot detach
the mapping underneath the write.
"""
function shm_write!(
        reg::SharedMemoryRegistry, name::AbstractString, offset::Integer,
        bytes::Vector{UInt8}
    )
    return shm_write!(_lookup_region(reg, name), offset, bytes)
end

# Write into a region already resolved, such as the one a request was decoded against: if that
# registration has since been replaced or dropped, the write fails as unregistered rather than
# landing in whatever object now carries the name.
function shm_write!(r::ShmRegion, offset::Integer, bytes::Vector{UInt8})
    @lock r.lock begin
        r.attached || throw(UnregisteredRegionError(r.name))
        base, bs = _region_base(r, offset, length(bytes))
        GC.@preserve r bytes unsafe_copyto!(base, pointer(bytes), bs)
        return nothing
    end
end
