# Client-local definitions. The dtype vocabulary (TritonType, KSERVE_OUTPUT_DTYPE_TABLE and
# its reverse) is shared from ReactantServerCore; only the batch iterator lives here.

# For batched inference: yield successive UnitRanges of at most `max_batch_size` items covering
# 1:n. Allocation-free.
struct BatchIterator
    n::Int
    max_batch_size::Int
    BatchIterator(n, max_batch_size) = new(Int(n), Int(max_batch_size))
end

function Base.iterate(it::BatchIterator, a::Int = 1)
    a > it.n && return nothing
    b = min(a + it.max_batch_size - 1, it.n)
    return a:b, b + 1
end

Base.IteratorSize(::Type{BatchIterator}) = Base.SizeUnknown()
Base.eltype(::Type{BatchIterator}) = UnitRange{Int}

# Why system shared memory is unavailable on this platform, or `nothing` when it is available.
# On macOS the server cannot reopen a region the client created (shm_open fails with EACCES, even
# within one process), so the IsSameIPCNamespace probe can never succeed there; the regions it would
# create are not trustworthy either. `:auto` falls back to inline and `:on` is refused. A Ref so the
# tests can exercise the unsupported path on Linux.
const _shm_unsupported = Ref{Union{Nothing, String}}(
    Sys.isapple() ?
        "system shared memory is not supported on macOS yet (the server cannot reopen a region " *
        "the client creates)" :
        nothing
)
