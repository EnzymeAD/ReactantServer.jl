# Concurrency contract of the fixed-slot allocator: under many threads hammering
# acquire_slot!/release_slot!, the pool must never hand the same slot index to two holders at
# once, and concurrently-held slots must occupy disjoint byte ranges. This is the property the
# old fixed-index/lockstep scheme lacked across independent top-level callers.

@testset "buffer pool slot allocator" begin
    @testset "construction + geometry" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        @test p.n_slots == 4
        @test p.slot_bytes == 1024
        @test sizeof(p) == 4096
        @test !ReactantServerCore.is_shm_backed(p)

        s1 = ReactantServerCore.acquire_slot!(p)
        s2 = ReactantServerCore.acquire_slot!(p)
        @test s1.offset != s2.offset
        @test s1.offset + s1.capacity <= sizeof(p)
        # disjoint ranges
        @test s1.offset + s1.capacity <= s2.offset || s2.offset + s2.capacity <= s1.offset
        ReactantServerCore.release_slot!(s1)
        ReactantServerCore.release_slot!(s2)
    end

    @testset "subslot carving" begin
        p = ReactantServerCore.BufferPool(1024; n_slots = 1, use_shm = false)
        s = ReactantServerCore.acquire_slot!(p)
        a = ReactantServerCore.subslot(s, 100)
        b = ReactantServerCore.subslot(s, 200)
        @test a.offset == s.offset
        @test b.offset == s.offset + 100
        @test_throws Exception ReactantServerCore.subslot(s, 1024)  # overruns remaining capacity
        ReactantServerCore.release_slot!(s)
    end

    @testset "contiguous span acquisition" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        s = ReactantServerCore.acquire_slot!(p, 3)
        @test s.span == 3
        @test s.capacity == 3 * p.slot_bytes
        @test s.offset == (s.index - 1) * p.slot_bytes
        # A concurrent single-slot acquire lands outside the run.
        s1 = ReactantServerCore.acquire_slot!(p)
        @test s1.offset + s1.capacity <= s.offset || s.offset + s.capacity <= s1.offset
        ReactantServerCore.release_slot!(s1)
        # Releasing the span frees all of its indices: a full drain succeeds without blocking.
        ReactantServerCore.release_slot!(s)
        drained = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]
        @test length(unique(d.index for d in drained)) == 4
        foreach(ReactantServerCore.release_slot!, drained)
    end

    @testset "span exceeding the pool throws immediately" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        @test_throws ArgumentError ReactantServerCore.acquire_slot!(p, 5)
        @test_throws ArgumentError ReactantServerCore.acquire_slot!(p, 0)
        # The failed request left no residue: the pool still drains fully.
        drained = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]
        foreach(ReactantServerCore.release_slot!, drained)
    end

    @testset "double release throws" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        s = ReactantServerCore.acquire_slot!(p)
        ReactantServerCore.release_slot!(s)
        @test_throws ArgumentError ReactantServerCore.release_slot!(s)
    end

    @testset "span waiter blocks until an adjacent run frees" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        singles = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]
        byidx = Dict(s.index => s for s in singles)
        # Free indices 2 and 4: two slots are free but no two are adjacent.
        ReactantServerCore.release_slot!(byidx[2])
        ReactantServerCore.release_slot!(byidx[4])

        got = Channel{Any}(1)
        t = Threads.@spawn put!(got, ReactantServerCore.acquire_slot!(p, 2))
        @test timedwait(() -> isready(got), 0.5) == :timed_out

        ReactantServerCore.release_slot!(byidx[3])   # 2,3,4 free; a run of 2 now exists
        @test timedwait(() -> isready(got), 5.0) == :ok
        s = take!(got)
        @test s.span == 2
        @test s.offset == (s.index - 1) * p.slot_bytes  # physically adjacent run
        wait(t)
        ReactantServerCore.release_slot!(s)
        ReactantServerCore.release_slot!(byidx[1])
    end

    @testset "FIFO fairness: large waiter is not starved" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        singles = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]

        order = Channel{Symbol}(8)
        # The big waiter takes the whole pool. This is deliberate: with any smaller span the
        # final release would satisfy big and leave a slot over that immediately satisfies the
        # next (small) waiter, so big and that small become runnable together and their `put!`s
        # race. With span == n_slots no slot is free while big holds the pool, so a small cannot
        # be served until big releases (strictly after big's `put!(:big)`), making the completion
        # order deterministic and not dependent on scheduler timing on a loaded CI runner.
        big = Threads.@spawn begin
            s = ReactantServerCore.acquire_slot!(p, 4)
            put!(order, :big)
            ReactantServerCore.release_slot!(s)
        end
        # Make sure the big waiter is enqueued before the small ones.
        @test timedwait(() -> (@lock p.alloc_lock !isempty(p.waitq)), 5.0) == :ok
        smalls = map(1:2) do _
            Threads.@spawn begin
                s = ReactantServerCore.acquire_slot!(p)
                put!(order, :small)
                ReactantServerCore.release_slot!(s)
            end
        end
        # Wait until both small waiters have parked behind the big one, so the FIFO ordering is
        # actually under test. Polling the queue length is robust where a fixed sleep is not.
        @test timedwait(() -> (@lock p.alloc_lock length(p.waitq) == 3), 5.0) == :ok

        # Release singles one at a time; freed slots must not be handed past the big waiter.
        for s in singles
            ReactantServerCore.release_slot!(s)
        end
        foreach(wait, [big; smalls])
        @test take!(order) == :big
        @test take!(order) == :small
        @test take!(order) == :small
    end

    @testset "subslot carving across the physical slot boundary" begin
        p = ReactantServerCore.BufferPool(1024; n_slots = 2, use_shm = false)  # 512-byte slots
        s = ReactantServerCore.acquire_slot!(p, 2)
        sub = ReactantServerCore.subslot(s, 700)   # straddles byte 512
        v = ReactantServerCore.pool_view(sub, UInt8, 700)
        v .= UInt8.((0:699) .% 256)
        @test ReactantServerCore.pool_view(sub, UInt8, 700) == UInt8.((0:699) .% 256)
        ReactantServerCore.release_slot!(s)
    end

    @testset "cancelled waiter leaves the queue" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        singles = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]
        t = Threads.@spawn ReactantServerCore.acquire_slot!(p)
        @test timedwait(() -> (@lock p.alloc_lock !isempty(p.waitq)), 5.0) == :ok
        schedule(t, InterruptException(); error = true)
        @test_throws TaskFailedException wait(t)
        # The abandoned token is gone and the pool still serves a fresh acquire.
        @test timedwait(() -> (@lock p.alloc_lock isempty(p.waitq)), 5.0) == :ok
        ReactantServerCore.release_slot!(singles[1])
        s = ReactantServerCore.acquire_slot!(p)
        ReactantServerCore.release_slot!(s)
        foreach(ReactantServerCore.release_slot!, singles[2:end])
    end

    @testset "deadline-bounded acquire times out instead of parking forever" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        singles = [ReactantServerCore.acquire_slot!(p) for _ in 1:4]   # pool fully held
        # No slot will free; a short deadline must throw rather than block indefinitely. The timer
        # has to wake the parked waiter on its own, since nothing is released here.
        dl = Int64(time_ns()) + 200_000_000   # 200 ms
        @test_throws ReactantServerCore.PoolAcquireTimeout ReactantServerCore.acquire_slot!(p; deadline_ns = dl)
        # The timed-out waiter dequeued itself; the queue is clean and a later acquire still works.
        @test timedwait(() -> (@lock p.alloc_lock isempty(p.waitq)), 5.0) == :ok

        # A deadline a release beats is satisfied normally (no spurious timeout).
        got = Channel{Any}(1)
        t = Threads.@spawn put!(got,
            ReactantServerCore.acquire_slot!(p; deadline_ns = Int64(time_ns()) + 5_000_000_000))
        @test timedwait(() -> (@lock p.alloc_lock !isempty(p.waitq)), 5.0) == :ok
        ReactantServerCore.release_slot!(singles[1])
        @test timedwait(() -> isready(got), 5.0) == :ok
        ReactantServerCore.release_slot!(take!(got))
        wait(t)

        # deadline_ns = 0 (the default) keeps the original unbounded behavior.
        s = ReactantServerCore.acquire_slot!(p)   # slot 1 is free again
        ReactantServerCore.release_slot!(s)
        foreach(ReactantServerCore.release_slot!, singles[2:end])
    end

    @testset "no overlap under contention" begin
        n_slots = 8
        p = ReactantServerCore.BufferPool(8 * 4096; n_slots = n_slots, use_shm = false)

        held = falses(n_slots)          # which slot indices are currently checked out
        held_lock = ReentrantLock()
        bad = Threads.Atomic{Int}(0)    # count of any invariant violation
        iters_per_task = 200
        ntasks = 32

        tasks = map(1:ntasks) do task_i
            Threads.@spawn begin
                span = 1 + task_i % 2   # mix span-1 and span-2 acquisitions
                for _ in 1:iters_per_task
                    s = ReactantServerCore.acquire_slot!(p, span)
                    rng = s.index:(s.index + s.span - 1)
                    @lock held_lock begin
                        # The allocator must never hand out an index already held.
                        any(held[rng]) && Threads.atomic_add!(bad, 1)
                        held[rng] .= true
                        # Offset and capacity must match the run's exclusive range.
                        s.offset == (s.index - 1) * p.slot_bytes || Threads.atomic_add!(bad, 1)
                        s.capacity == span * p.slot_bytes || Threads.atomic_add!(bad, 1)
                    end
                    # Touch the bytes to surface any aliasing under TSan-like stress.
                    v = ReactantServerCore.pool_view(s, UInt8, s.capacity)
                    @inbounds v[1] = UInt8(s.index % 256)
                    @inbounds v[end] = UInt8(s.index % 256)
                    @lock held_lock held[rng] .= false
                    ReactantServerCore.release_slot!(s)
                end
            end
        end
        foreach(wait, tasks)
        @test bad[] == 0
        # All slots returned: a fresh full drain must succeed without blocking.
        drained = [ReactantServerCore.acquire_slot!(p) for _ in 1:n_slots]
        @test length(unique(s.index for s in drained)) == n_slots
        foreach(ReactantServerCore.release_slot!, drained)
    end

    @testset "scratch carves several typed buffers at once" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = false)
        s = ReactantServerCore.acquire_slot!(p, 2)
        f, i = ReactantServerCore.scratch(s, [(4,) => Float32, (2,) => Int32])
        # Arrays aliasing the pool backing (handed to NamedTensor by reference downstream).
        @test f isa Vector{Float32}
        @test i isa Vector{Int32}
        @test size(f) == (4,) && size(i) == (2,)
        # Disjoint, contiguous offsets within the slot.
        base = Int(ReactantServerCore.pool_base_pointer(p))
        @test Int(pointer(f)) - base == s.offset
        @test Int(pointer(i)) - base == s.offset + 4 * sizeof(Float32)
        # Writes land in the right bytes and the buffers do not alias.
        f .= Float32[1, 2, 3, 4]
        i .= Int32[10, 20]
        @test f == Float32[1, 2, 3, 4] && i == Int32[10, 20]
        # Scalar form returns a single buffer with the given shape.
        s2 = ReactantServerCore.acquire_slot!(p)
        m = ReactantServerCore.scratch(s2, (3, 3), Float64)
        @test size(m) == (3, 3) && m isa Matrix{Float64}
        ReactantServerCore.release_slot!(s2)
        ReactantServerCore.release_slot!(s)
    end
end
