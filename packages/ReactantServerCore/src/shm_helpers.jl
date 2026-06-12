# Shared-memory aliasing helpers, vendored from the (unpublished) MMISHM.jl.
#
# These let both inbound RPC traffic (protobuf or SHM) and outbound transport (inline pool
# or SHM pool) converge on a single concrete type, FixedSizeArray{T,N,Memory{T}}, so
# downstream dispatch is reachable at precompile time and copies are limited to a single bulk
# memcpy on the protobuf path. The SHM path is zero-copy: the Memory aliases the SHM pointer
# and a finalizer closure keeps the SharedMemory alive for the lifetime of the Memory.

# Random, per-process suffix for every SHM key produced by `shm_key`. Combining the PID with
# 64 bits of entropy makes collisions vanishingly unlikely even when two processes share an
# IPC namespace, while keeping the PID prefix so orphans in /dev/shm can be attributed to a
# process during ops triage. Set by `_init_shm_naming!` from the module `__init__`.
const _naming_token = Ref{String}("")

_init_shm_naming!() = (_naming_token[] = string(getpid(), "-", bytes2hex(rand(UInt8, 8))); nothing)

shm_key(name::AbstractString) = "/shm-$(name)-$(_naming_token[])"

WrappedFArray(shm::SharedMemory, ::Type{T}, shape) where {T} = WrappedArray(shm, T, shape...)

# With the (W, H) Julia convention, a Python C-order array of shape [H, W] has the same memory
# layout as a Julia column-major array of shape (W, H). Reversing the shape is all that is
# needed; no transpose or permutedims required.
WrappedCArray(shm::SharedMemory, ::Type{T}, shape) where {T} = WrappedArray(shm, T, reverse(shape)...)

# Public version of Base.FastContiguousSubArray{T}
const MemCopySafeSubArray = SubArray{
    T,N,P,I,true,
} where {T,N,P,I<:Union{Tuple{Vararg{Real}},Tuple{AbstractUnitRange,Vararg{Any}}}}

const MemCopySafeReshapedArray{T,N} = Base.ReshapedArray{
    T,N,P,Tuple{},
} where {P<:Union{DenseArray{T},MemCopySafeSubArray{T}}}

# All array types contiguous in memory (safe to use with unsafe_copyto!).
const MemCopySafeArray{T,N} = Union{
    DenseArray{T,N},MemCopySafeSubArray{T,N},MemCopySafeReshapedArray{T,N},
} where {T,N}

memcpy_safe_arr_n_bytes(a::MemCopySafeArray) = Base.checked_mul(length(a), Base.elsize(typeof(a)))

# Pinned SharedMemory references keyed by the aliased Memory's objectid. Pins the SharedMemory
# so its own finalizer (_destroy in InterProcessCommunication) cannot fire and munmap the page
# out from under the Memory while the alias is in use. Keyed by objectid(mem) rather than `mem`
# itself: an IdDict{Memory,...} would hold the Memory strongly as a key, preventing its
# finalizer from ever running and leaking the dict entry.
const _SHM_KEEPALIVE = Dict{UInt,SharedMemory}()
const _SHM_KEEPALIVE_LOCK = ReentrantLock()

# Wrap a SHM region as a Memory{T} that aliases the SHM bytes. own=false because IPC owns the
# underlying region, not the Julia GC.
function memory_from_shm(shm::SharedMemory, ::Type{T}, n_elem::Integer) where {T}
    n = Int(n_elem)
    n >= 0 || throw(ArgumentError("memory_from_shm: n_elem must be non-negative (got $n)"))
    nbytes = Base.checked_mul(Int(sizeof(T)), n)
    nbytes <= sizeof(shm) ||
        throw(ArgumentError("memory_from_shm: $(nbytes) bytes exceeds SHM region of $(sizeof(shm)) bytes"))
    ptr = convert(Ptr{T}, pointer(shm))
    mem = unsafe_wrap(Memory{T}, ptr, n; own = false)
    key = objectid(mem)
    @lock _SHM_KEEPALIVE_LOCK _SHM_KEEPALIVE[key] = shm
    finalizer(mem) do m
        @lock _SHM_KEEPALIVE_LOCK delete!(_SHM_KEEPALIVE, objectid(m))
    end
    return mem
end

# One bulk memcpy from protobuf-decoded bytes into a typed Memory{T}.
function memory_from_bytes(bytes::AbstractVector{UInt8}, ::Type{T}, n_elem::Integer) where {T}
    n = Int(n_elem)
    n >= 0 || throw(ArgumentError("memory_from_bytes: n_elem must be non-negative (got $n)"))
    nbytes = sizeof(T) * n
    length(bytes) == nbytes ||
        throw(DimensionMismatch("memory_from_bytes: expected $nbytes bytes for $n elements of $T, got $(length(bytes))"))
    mem = Memory{T}(undef, n)
    GC.@preserve mem bytes unsafe_copyto!(convert(Ptr{UInt8}, pointer(mem)), pointer(bytes), nbytes)
    return mem
end

# Adopt an existing Memory as the storage of a FixedSizeArray without copying.
fsa_from_memory(mem::Memory{T}, size::NTuple{N,<:Integer}) where {T,N} =
    FixedSizeArrays.new_fixed_size_array(mem, map(Int, size))

fsa_from_memory(mem::Memory{T}, size::Vararg{Integer,N}) where {T,N} = fsa_from_memory(mem, size)
