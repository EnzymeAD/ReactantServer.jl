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

import InterProcessCommunication as IPC

mutable struct ShmRegion
    name::String
    key::String
    offset::Int
    byte_size::Int
    shm::IPC.SharedMemory{String}
    lock::ReentrantLock
    attached::Bool
end
ShmRegion(name, key, offset, byte_size, shm) =
    ShmRegion(name, key, offset, byte_size, shm, ReentrantLock(), true)

struct SharedMemoryRegistry
    regions::Dict{String,ShmRegion}
    lock::ReentrantLock
end
SharedMemoryRegistry() = SharedMemoryRegistry(Dict{String,ShmRegion}(), ReentrantLock())

# Detach our mapping of a region. Must be called holding `r.lock`; flips `attached` so any
# reader blocked on the same lock bails instead of touching the munmap'd page. This munmaps
# only; it never shm_unlinks, so the client's underlying object is left intact.
function _detach!(r::ShmRegion)
    @lock r.lock begin
        r.attached || return
        r.attached = false
        finalize(r.shm)
    end
    return
end

"""
    shm_register!(registry, name, key, offset, byte_size)

Attach the existing POSIX shared-memory object `key` read-write and register it under
`name`. Re-registering a name replaces (and detaches) the previous mapping.
"""
function shm_register!(reg::SharedMemoryRegistry, name::AbstractString, key::AbstractString,
                       offset::Integer, byte_size::Integer)
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

    shm = IPC.SharedMemory(String(key); readonly=false)
    total = sizeof(shm)
    # Subtraction form: `off + bs` could wrap for large values.
    if bs > total || off > total - bs
        finalize(shm)
        throw(ArgumentError("region '$name' offset+byte_size exceeds " *
                            "shared memory object '$key' size ($total)"))
    end

    old = @lock reg.lock begin
        prev = get(reg.regions, name, nothing)
        reg.regions[name] = ShmRegion(String(name), String(key), off, bs, shm)
        prev
    end
    old === nothing || _detach!(old)
    return nothing
end

"""
    shm_unregister!(registry, name)

Unregister and detach a region, or all regions when `name` is empty. Idempotent: unregistering a
name that is not registered is a successful no-op (it matches KServe semantics and lets the gateway
fan-out and the client's pre-emptive cleanup unregister succeed without a spurious error).
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
        IPC.SharedMemory(String(name); readonly=true)
    catch
        return false
    end
    finalize(shm)
    return true
end

shm_regions(reg::SharedMemoryRegistry) = (@lock reg.lock copy(reg.regions))

shm_teardown!(reg::SharedMemoryRegistry) = shm_unregister!(reg, "")

# Look up a region by name under the registry lock. The returned region carries its own lock;
# the caller takes that lock and re-checks `attached` before touching the mapping.
function _lookup_region(reg::SharedMemoryRegistry, name::AbstractString)
    r = @lock reg.lock get(reg.regions, name, nothing)
    r === nothing && throw(ArgumentError("unregistered shared memory region: $name"))
    return r
end

# Bounds-checked raw byte base within a region. Caller MUST hold `r.lock` and have verified
# `r.attached`.
function _region_base(r::ShmRegion, offset::Integer, byte_size::Integer)
    # Range-check the untrusted values before narrowing, and use the subtraction form for the
    # bounds check: `off + bs` could wrap for hostile int64 parameters.
    (0 <= offset <= typemax(Int) && 0 <= byte_size <= typemax(Int)) ||
        throw(ArgumentError("shared memory access (offset $offset, byte_size $byte_size) is out " *
                            "of bounds for region '$(r.name)' of $(r.byte_size) bytes"))
    off = Int(offset)
    bs = Int(byte_size)
    (bs <= r.byte_size && off <= r.byte_size - bs) ||
        throw(ArgumentError("shared memory access [$off, $off + $bs) is out of bounds for region " *
                            "'$(r.name)' of $(r.byte_size) bytes"))
    return convert(Ptr{UInt8}, pointer(r.shm)) + r.offset + off, bs
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
        r.attached || throw(ArgumentError("unregistered shared memory region: $name"))
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
function shm_write!(reg::SharedMemoryRegistry, name::AbstractString, offset::Integer,
                    bytes::Vector{UInt8})
    r = _lookup_region(reg, name)
    @lock r.lock begin
        r.attached || throw(ArgumentError("unregistered shared memory region: $name"))
        base, bs = _region_base(r, offset, length(bytes))
        GC.@preserve r bytes unsafe_copyto!(base, pointer(bytes), bs)
        return nothing
    end
end
