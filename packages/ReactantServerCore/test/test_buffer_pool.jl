# Concurrency contract of the fixed-slot allocator: under many threads hammering
# acquire_slot!/release_slot!, the pool must never hand the same slot index to two holders at
# once, and concurrently-held slots must occupy disjoint byte ranges. This is the property the
# old fixed-index/lockstep scheme lacked across independent top-level callers.

using InterProcessCommunication: SharedMemory

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
        t = Threads.@spawn put!(
            got,
            ReactantServerCore.acquire_slot!(p; deadline_ns = Int64(time_ns()) + 5_000_000_000)
        )
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

    @testset "shm keys fit the POSIX name budget on every platform" begin
        # macOS (XNU) rejects shm names longer than 31 bytes after the leading slash; Linux allows
        # 255, so an overlong key only fails on macOS. Every key we mint must fit the tighter
        # budget regardless of the semantic pool name handed to the pool.
        names = (
            "reactant_server_pool", "reactant_server_client_pool",
            "a", "abcde", "abcdef",
            repeat("long_name_", 20),
            "spaced name/with/slashes and:colons",
        )
        for name in names
            key = ReactantServerCore.shm_key(name)
            @test startswith(key, "/shm-")
            @test length(key) <= 31
            @test !occursin('/', key[2:end])            # only the leading slash survives
        end
        # Distinct names in one process map to distinct keys, and the mapping is stable for the
        # process's lifetime (the per-process token does not move under a live pool).
        @test ReactantServerCore.shm_key("reactant_server_pool") !=
            ReactantServerCore.shm_key("reactant_server_client_pool")
        @test ReactantServerCore.shm_key("x") == ReactantServerCore.shm_key("x")
    end

    @testset "shm-backed pool round-trips through a short key" begin
        # End to end on the POSIX path: the minted key must be creatable by shm_open (the
        # BufferPool constructor) under the 31-byte budget the key was designed for, the region
        # name clients register is the key without its leading slash, and a second mapping of the
        # same key (what the server does for registration and IsSameIPCNamespace) observes
        # writes from the first.
        p = ReactantServerCore.BufferPool(4096; n_slots = 4, use_shm = true)
        try
            @test ReactantServerCore.is_shm_backed(p)
            region = ReactantServerCore.pool_region_name(p)
            @test length(region) + 1 <= 31                       # + leading slash
            @test !occursin('/', region)
            s = ReactantServerCore.acquire_slot!(p)
            v = ReactantServerCore.pool_view(s, UInt8, 8)
            v .= 0x5a
            peer = SharedMemory("/" * region; readonly = true)
            GC.@preserve peer begin
                got = unsafe_wrap(Vector{UInt8}, convert(Ptr{UInt8}, pointer(peer)), 8)
                @test all(==(0x5a), got)
            end
            finalize(peer)
            ReactantServerCore.release_slot!(s)
        finally
            rm(p.backing)
        end
    end

    @testset "keepalive finalizer defers instead of blocking on a held lock" begin
        # A finalizer that blocks on a contended ReentrantLock either throws "task switch not
        # allowed from inside gc finalizer" (the pin then leaks for the life of the process) or
        # stalls the thread running the GC until the lock is free. The unpin must instead give up,
        # re-register, and complete at a later GC once the lock is free.
        RSC = ReactantServerCore
        p = RSC.BufferPool(4096; n_slots = 1, use_shm = false)
        s = RSC.acquire_slot!(p)
        pinned() = @lock RSC._POOL_KEEPALIVE_LOCK count(v -> v === p, values(RSC._POOL_KEEPALIVE))
        # Held until the lock is taken, so a GC on another thread cannot collect them early.
        aliases = [RSC.pool_memory(s, UInt8, 16) for _ in 1:32]
        @test pinned() == 32

        held = Threads.Event()
        done = Threads.Event()
        holder = Threads.@spawn begin
            lock(RSC._POOL_KEEPALIVE_LOCK)
            try
                notify(held)
                timedwait(() -> done.set, 10.0)   # bounded, so a regression fails rather than hangs
            finally
                unlock(RSC._POOL_KEEPALIVE_LOCK)
            end
        end
        wait(held)
        empty!(aliases)
        t = @elapsed GC.gc(true)
        @test pinned() > 0                         # deferred, not dropped and not blocked on
        notify(done)
        wait(holder)
        @test t < 5.0                              # the GC did not wait for the lock holder
        for _ in 1:5
            GC.gc(true)
            pinned() == 0 && break
        end
        @test pinned() == 0                        # the deferred unpins completed
        RSC.release_slot!(s)
    end

    @testset "close releases an idle shm pool's mapping at once" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 2, use_shm = true)
        region = "/" * ReactantServerCore.pool_region_name(p)
        @test isopen(p)
        close(p)
        @test !isopen(p)
        @test p.released
        @test pointer(p.backing) == C_NULL         # munmapped (IPC's PARANOID destroy)
        @test_throws Exception SharedMemory(region; readonly = true)   # unlinked
        @test_throws ReactantServerCore.PoolClosedError ReactantServerCore.acquire_slot!(p)
        close(p)                                   # idempotent
    end

    @testset "close waits for outstanding slots before releasing" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 2, use_shm = true)
        region = "/" * ReactantServerCore.pool_region_name(p)
        a = ReactantServerCore.acquire_slot!(p)
        b = ReactantServerCore.acquire_slot!(p)
        close(p)
        @test_throws Exception SharedMemory(region; readonly = true)   # name gone immediately
        @test !p.released                          # but in-flight slots keep the mapping
        ReactantServerCore.pool_view(a, UInt8, 8) .= 0x11              # still writable
        ReactantServerCore.release_slot!(a)
        @test !p.released
        ReactantServerCore.release_slot!(b)
        @test p.released                           # the last release frees it
        @test pointer(p.backing) == C_NULL
    end

    @testset "close wakes parked waiters with an error" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 1, use_shm = true)
        s = ReactantServerCore.acquire_slot!(p)
        waiter = Threads.@spawn ReactantServerCore.acquire_slot!(p)
        @test timedwait(() -> !isempty(@lock(p.alloc_lock, copy(p.waitq))), 5.0) == :ok
        close(p)
        @test timedwait(() -> istaskdone(waiter), 5.0) == :ok
        @test_throws TaskFailedException fetch(waiter)
        @test isempty(p.waitq)
        ReactantServerCore.release_slot!(s)
        @test p.released
    end

    @testset "close leaves the mapping to the GC while a pool_memory alias lives" begin
        p = ReactantServerCore.BufferPool(4096; n_slots = 1, use_shm = true)
        s = ReactantServerCore.acquire_slot!(p)
        mem = ReactantServerCore.pool_memory(s, UInt8, 8)
        close(p)
        ReactantServerCore.release_slot!(s)
        @test !p.released                          # the alias may still be read
        GC.@preserve mem @test pointer(p.backing) != C_NULL
        finalize(mem)                              # drop the pin (the alias's own finalizer)
        @test !ReactantServerCore._has_pinned_alias(p)
        finalize(p.backing)                        # what the GC would now do
    end
end

@testset "slot boundaries and scratch buffers stay aligned (CORE-15)" begin
    RSC = ReactantServerCore
    page = RSC._page_size()
    # Odd totals and odd slot counts: 10^6 ÷ 3 put slot 2 at an odd offset, so every typed view in it
    # failed; 1000 ÷ 3 is below a page and aligns to 64 bytes.
    for (bytes, n) in ((10^6, 3), (7 * page + 13, 7), (1000, 3), (3 * page + 100, 3)), use_shm in (false, true)
        p = RSC.BufferPool(bytes; n_slots = n, use_shm = use_shm)
        try
            @test p.slot_bytes % (p.slot_bytes >= page ? page : 64) == 0
            @test 0 < p.slot_bytes * n <= bytes
            slots = [RSC.acquire_slot!(p) for _ in 1:n]
            for s in slots, k in (1, 2, 3, 5, 7)            # odd chunk sizes
                RSC.reset_slot!(s)
                specs = Pair[(3, k) => UInt8, (4, k) => Float32, (k,) => Float64, (5, k) => Float16, (k,) => Int16, (2, k) => Bool]
                total = sum(sizeof(last(sp)) * prod(first(sp)) for sp in specs)
                total <= s.capacity || continue
                bufs = RSC.scratch(s, specs)
                for (b, sp) in zip(bufs, specs)
                    T = last(sp)
                    @test b isa Array{T} && size(b) == first(sp)
                    @test UInt(pointer(b)) % Base.datatype_alignment(T) == 0
                    fill!(b, one(T))
                end
                @test s.cursor == total                     # aligned with no padding at all
                ranges = sort!([UInt(pointer(b)):(UInt(pointer(b)) + sizeof(b) - 1) for b in bufs]; by = first)
                @test all(last(ranges[i]) < first(ranges[i + 1]) for i in 1:(length(ranges) - 1))
                @test all(b -> all(==(one(eltype(b))), b), bufs)   # no buffer overwrote another
            end
            foreach(RSC.release_slot!, slots)
        finally
            close(p)
        end
    end

    # The tracker's case: a 3-byte-per-item UInt8 input before a Float32 one, on a final chunk of 3.
    p = RSC.BufferPool(4096; n_slots = 1, use_shm = false)
    s = RSC.acquire_slot!(p)
    m, x = RSC.scratch(s, Pair[(3, 3) => UInt8, (4, 3) => Float32])
    @test size(m) == (3, 3) && size(x) == (4, 3) && UInt(pointer(x)) % 4 == 0

    # subslot: an explicit alignment, a clear error for a misaligned typed view, and no wraparound in
    # the bounds check.
    RSC.reset_slot!(s)
    RSC.subslot(s, 3)
    b = RSC.subslot(s, 8; align = 8)
    @test b.offset == s.offset + 8 && s.cursor == 16
    RSC.subslot(s, 1)
    c = RSC.subslot(s, 4)                                   # offset 17, misaligned for Float32
    err = try
        RSC.pool_view(c, Float32, 1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("not aligned for Float32", err.msg)
    @test_throws ArgumentError RSC.subslot(s, 8; align = 3)
    cur = s.cursor
    @test_throws ArgumentError RSC.subslot(s, typemax(Int))
    @test_throws ArgumentError RSC.subslot(s, typemax(Int) - 7)
    @test_throws ArgumentError RSC.subslot(s, s.capacity; align = 8)
    @test s.cursor == cur                                   # a refused carve moves nothing
    RSC.release_slot!(s)
end
