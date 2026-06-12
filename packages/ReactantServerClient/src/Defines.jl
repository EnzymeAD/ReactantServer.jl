# Client-local definitions. The dtype vocabulary (TritonType, KSERVE_OUTPUT_DTYPE_TABLE and
# its reverse) is shared from ReactantServerCore; only the batch iterator, which needs
# ResumableFunctions, lives here.

# For batched inference: yield successive UnitRanges of at most `max_batch_size` items.
@resumable function BatchIterator(n, max_batch_size)
    a = 1
    while a <= n
        b = min(a + max_batch_size - 1, n)
        @yield a:b
        a += max_batch_size
    end
end
