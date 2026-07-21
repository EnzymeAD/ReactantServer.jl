# Client-local definitions. The dtype vocabulary (TritonType, KSERVE_OUTPUT_DTYPE_TABLE and
# its reverse) is shared from ReactantServerCore; only the batch iterator lives here.

# For batched inference: yield successive UnitRanges of at most `max_batch_size` items covering
# 1:n. A plain iterator rather than a ResumableFunctions @resumable coroutine: the behavior is a
# handful of lines and the generated-function machinery @resumable relies on is fragile across
# Julia versions (it mis-dispatches on Julia 1.12), while this is stable and allocation-free.
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
