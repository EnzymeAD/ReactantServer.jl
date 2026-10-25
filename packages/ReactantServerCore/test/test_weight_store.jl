# WeightStore tests. The private store is exercised in-process; the shared store's cross-process
# coordination (reuse of a READY region and last-one-out unlink) is exercised by spawning a second
# Julia process that attaches the same POSIX SHM region. Two processes on one host share /dev/shm,
# which is exactly what Docker `ipc: host` gives same-node worker containers.

@testset "PrivateWeightStore materializes fresh per-call arrays" begin
    specs = [(Float32, (4,)), (Float64, (2, 2))]
    dg = weights_digest("m", specs)
    arrs = materialize_host_weights!(
        PrivateWeightStore(), "m", dg, specs,
        a -> (a[1] .= Float32[1, 2, 3, 4]; a[2] .= 9.0)
    )
    @test arrs[1] == Float32[1, 2, 3, 4]
    @test size(arrs[2]) == (2, 2) && all(==(9.0), arrs[2])
    @test eltype(arrs[1]) == Float32 && eltype(arrs[2]) == Float64
    release_host_weights!(PrivateWeightStore(), "m")   # no-op
end

@testset "weights_digest is stable, layout-sensitive, and content-sensitive" begin
    @test weights_digest("m", [(Float32, (4,))]) == weights_digest("m", [(Float32, (4,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("m", [(Float32, (8,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("n", [(Float32, (4,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("m", [(Float64, (4,))])
    # A weights-only update (same name, same layout) must move the digest, or a stale region in
    # /dev/shm would be reused and serve the old weights, even across restarts.
    @test weights_digest("m", [(Float32, (4,))]; content = UInt64(1)) !=
        weights_digest("m", [(Float32, (4,))])
    @test weights_digest("m", [(Float32, (4,))]; content = UInt64(1)) !=
        weights_digest("m", [(Float32, (4,))]; content = UInt64(2))
end

@testset "weights_file_token tracks the on-disk file version" begin
    dir = mktempdir()
    p = joinpath(dir, "weights.safetensors")
    write(p, rand(UInt8, 64))
    t1 = weights_file_token(p)
    @test t1 != 0
    @test weights_file_token(p) == t1                 # stable while the file is untouched
    sleep(0.01)                                       # ensure a distinct mtime
    write(p, rand(UInt8, 64))                         # same size, new mtime: new version
    @test weights_file_token(p) != t1
    @test weights_file_token("") == 0                 # hand-built entries with no file
    @test weights_file_token(joinpath(dir, "missing")) == 0
    rm(dir; recursive = true, force = true)
end

@testset "weights_file_id names the mapped file; an atomic replace is a new version" begin
    dir = mktempdir()
    p = joinpath(dir, "weights.safetensors")
    write(p, rand(UInt8, 64))
    mapped = open(io -> weights_file_id(stat(io)), p)       # taken from the open file, as at load
    @test mapped == weights_file_id(p)
    @test weights_file_token(mapped) == weights_file_token(p)
    # Deploy a new file with the same size and the same mtime by writing it beside the old one and
    # renaming it over: only the inode tells the versions apart, and it must.
    q = joinpath(dir, "weights.new")
    write(q, rand(UInt8, 64))
    run(`touch -r $p $q`)
    mv(q, p; force = true)
    now = weights_file_id(p)
    @test now.size == mapped.size && now.mtime == mapped.mtime
    @test now.inode != mapped.inode
    @test weights_file_token(now) != weights_file_token(mapped)
    @test weights_file_id("") == NO_WEIGHTS_FILE && weights_file_token(NO_WEIGHTS_FILE) == 0
    rm(dir; recursive = true, force = true)
end

@testset "SharedWeightStore: a region the node cannot back fails cleanly (no SIGBUS)" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsfull_" * string(getpid())
        specs = [(Float32, (4,))]
        dg = weights_digest(key, specs)
        region = "/dev/shm/rsw-" * key * "-" * string(dg; base = 16)
        filled = Ref(false)
        # Simulate a full /dev/shm (ENOSPC = 28): the reservation fails before any page is touched.
        saved = ReactantServerCore._RESERVE_REGION[]
        ReactantServerCore._RESERVE_REGION[] = (name, bytes) -> Cint(28)
        try
            err = try
                materialize_host_weights!(SharedWeightStore(), key, dg, specs, a -> (filled[] = true))
                nothing
            catch e
                e
            end
            @test err isa SharedStoreFull
            @test !filled[]                                  # nothing was written
            @test !isfile(region)                            # and nothing is left behind
        finally
            ReactantServerCore._RESERVE_REGION[] = saved
        end
        # The real reservation succeeds for a normal region.
        store = SharedWeightStore()
        arrs = materialize_host_weights!(store, key, dg, specs, a -> (a[1] .= 7.0f0))
        @test arrs[1] == fill(7.0f0, 4)
        arrs = nothing
        release_host_weights!(store, key)
        @test !isfile(region)
    end
end

@testset "SharedWeightStore: waiting on a peer's lock does not block garbage collection" begin
    if !(Sys.islinux() && isdir("/dev/shm")) || Threads.nthreads() < 2
        @test_skip "needs Linux /dev/shm and at least two threads"
    else
        lockpath = "/dev/shm/rsw-gcwait_" * string(getpid()) * ".lock"
        holder = ReactantServerCore._open_lock(lockpath)
        waiter = ReactantServerCore._open_lock(lockpath)
        # flock conflicts between separate open file descriptions, even within one process, so the
        # waiter below blocks exactly as it would behind a peer worker filling the region.
        @test ReactantServerCore._flock(holder, ReactantServerCore._LOCK_EX) == 0
        t = Threads.@spawn ReactantServerCore._flock(waiter, ReactantServerCore._LOCK_SH)
        sleep(0.2)                                           # let the waiter block in flock
        @test !istaskdone(t)
        gc_seconds = @elapsed GC.gc(false)
        @test gc_seconds < 0.15                              # a plain ccall would hold it until release
        ReactantServerCore._flock(holder, ReactantServerCore._LOCK_UN)
        @test fetch(t) == 0
        ReactantServerCore._flock(waiter, ReactantServerCore._LOCK_UN)
        ReactantServerCore._close_fd(holder); ReactantServerCore._close_fd(waiter)
        rm(lockpath; force = true)
    end
end

@testset "SharedWeightStore: single-process create, reuse, and unlink" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsone_" * string(getpid())
        specs = [(Float32, (4,))]
        dg = weights_digest(key, specs)
        region = "/dev/shm/rsw-" * key * "-" * string(dg; base = 16)

        store = SharedWeightStore()
        arrs = materialize_host_weights!(store, key, dg, specs, a -> (a[1] .= Float32[10, 20, 30, 40]))
        @test arrs[1] == Float32[10, 20, 30, 40]
        @test isfile(region)

        # A model rename rekeys the attachment; the region (content-addressed under the OLD name)
        # stays mapped and a release under the NEW key detaches and unlinks it.
        rename_host_weights!(store, key, key * "_renamed")
        @test isfile(region)
        release_host_weights!(store, key)                  # old key: no longer attached, no-op
        @test isfile(region)
        release_host_weights!(store, key * "_renamed")
        @test !isfile(region)         # sole holder unlinks on release
        rename_host_weights!(PrivateWeightStore(), "a", "b")   # no-op
    end
end

@testset "SharedWeightStore: attaching a new version sweeps a holder-free stale one" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsswp_" * string(getpid())
        specs = [(Float32, (4,))]

        # Simulate a crashed run's leftover: materialize version 1, then release WITHOUT unlink by
        # renaming away and dropping the store (the region stays because we fake the crash by
        # keeping it: populate v1 and abandon the store object after closing its lock).
        store1 = SharedWeightStore()
        dg1 = weights_digest(key, specs; content = UInt64(1))
        materialize_host_weights!(store1, key, dg1, specs, a -> (a[1] .= 1.0f0))
        region1 = "/dev/shm/rsw-" * key * "-" * string(dg1; base = 16)
        @test isfile(region1)
        # Drop the holder's lock the way a crashed process would (kernel releases flocks on exit)
        # but leave the region in place: close the fd directly without the release path's unlink.
        ent = store1.attached[key]
        ReactantServerCore._flock(ent.fd, ReactantServerCore._LOCK_UN)
        ReactantServerCore._close_fd(ent.fd)
        finalize(ent.shm)
        delete!(store1.attached, key)
        @test isfile(region1)                              # leaked, holder-free

        # A new version (different content token) attaches under a new region and sweeps the leak.
        store2 = SharedWeightStore()
        dg2 = weights_digest(key, specs; content = UInt64(2))
        arrs = materialize_host_weights!(store2, key, dg2, specs, a -> (a[1] .= 2.0f0))
        region2 = "/dev/shm/rsw-" * key * "-" * string(dg2; base = 16)
        @test arrs[1] == Float32[2, 2, 2, 2]               # freshly filled, not the stale content
        @test isfile(region2)
        @test !isfile(region1)                             # stale version swept
        release_host_weights!(store2, key)
        @test !isfile(region2)
        # Sweep leaves lock files behind by design; clean them for a tidy /dev/shm.
        for f in readdir("/dev/shm"; join = true)
            occursin("rsw-" * key, f) && endswith(f, ".lock") && rm(f; force = true)
        end
    end
end

@testset "SharedWeightStore: a second process reuses the region; last out unlinks" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsx_" * string(getpid())
        specs = [(Float32, (4,))]
        dg = weights_digest(key, specs)
        region = "/dev/shm/rsw-" * key * "-" * string(dg; base = 16)

        dir = mktempdir()
        readyf = joinpath(dir, "ready")
        stopf = joinpath(dir, "stop")
        script = joinpath(dir, "holder.jl")
        write(
            script, """
            using ReactantServerCore
            key, readyf, stopf = ARGS[1], ARGS[2], ARGS[3]
            specs = [(Float32, (4,))]
            store = SharedWeightStore()
            materialize_host_weights!(store, key, weights_digest(key, specs), specs,
                a -> (a[1] .= Float32[10, 20, 30, 40]))
            write(readyf, "ok")
            t0 = time(); while !isfile(stopf) && time() - t0 < 30; sleep(0.05); end
            release_host_weights!(store, key)
            """
        )

        proc = run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script $key $readyf $stopf`; wait = false)
        try
            t0 = time()
            while !isfile(readyf) && time() - t0 < 90
                sleep(0.1)
            end
            @test isfile(readyf)          # holder populated the region
            @test isfile(region)

            # This process attaches the same region and must REUSE it (fill! not called).
            called = Ref(false)
            store = SharedWeightStore()
            arrs = materialize_host_weights!(store, key, dg, specs, a -> (called[] = true; a[1] .= 0.0f0))
            @test called[] == false
            @test arrs[1] == Float32[10, 20, 30, 40]

            release_host_weights!(store, key)   # not the last holder
            @test isfile(region)                # the holder still has it mapped
        finally
            touch(stopf)
            wait(proc)
        end
        @test !isfile(region)             # last holder out: region unlinked
        rm(dir; recursive = true, force = true)
    end
end

# The node budget counts every region under a store's prefix, so these tests use a prefix of their
# own: they neither see nor evict the regions of other runs on the machine.
_budget_cleanup(prefix) = for f in readdir("/dev/shm"; join = true)
    startswith(basename(f), prefix * "-") && rm(f; force = true)
end

@testset "SharedWeightStore node budget: admit, evict least recently loaded, keep held regions" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        prefix = "tb" * string(getpid())
        specs = [(Float32, (1024,))]
        R = ReactantServerCore._layout(specs)[2]               # bytes per region
        region(k) = "/dev/shm/" * prefix * "-" * k * "-" * string(weights_digest(k, specs); base = 16)
        store = SharedWeightStore(; budget = 2R, prefix = prefix)
        fills = String[]
        load(k) = with_cached_host_weights(
            a -> copy(a[1]), store, k, weights_digest(k, specs), specs,
            a -> (push!(fills, k); a[1] .= length(k))
        )
        try
            @test trim_host_cache!(store) == 0 && host_cache_bytes(store) == 0

            hit, v = load("m1")
            @test !hit && v == fill(2.0f0, 1024) && isfile(region("m1"))
            sleep(0.02)
            @test load("m22")[1] == false
            @test host_cache_bytes(store) == 2R
            sleep(0.02)
            @test load("m1")[1] == true                    # a hit: m1 is now the most recently loaded
            @test fills == ["m1", "m22"]                   # the hit did not refill
            sleep(0.02)
            @test load("m333")[1] == false                 # evicts m22, the least recently loaded
            @test !isfile(region("m22")) && isfile(region("m1")) && isfile(region("m333"))
            @test host_cache_bytes(store) == 2R

            # A pinned floor is admitted like any region but held, so it is never evicted.
            sleep(0.02)
            p1 = materialize_host_weights!(store, "p1", weights_digest("p1", specs), specs, a -> (a[1] .= 7))
            @test p1[1][1] == 7 && !isfile(region("m1")) && isfile(region("m333"))
            sleep(0.02)
            materialize_host_weights!(store, "p2", weights_digest("p2", specs), specs, a -> (a[1] .= 8))
            @test !isfile(region("m333")) && host_cache_bytes(store) == 2R
            # Pinned floors past the budget still load (a guarantee), with a warning.
            @test_logs (:warn, r"overcommitted") match_mode = :any materialize_host_weights!(
                store, "p3", weights_digest("p3", specs), specs, a -> (a[1] .= 9)
            )
            @test host_cache_bytes(store) == 3R
            # Nothing evictable: an unpinned load is not cached and its fill never runs.
            n = length(fills)
            @test load("m4") === nothing && length(fills) == n && !isfile(region("m4"))
            # Room is made only when it suffices: a region too large for what can be evicted leaves
            # the evictable regions cached.
            store.budget = 3R
            release_host_weights!(store, "p3")             # p3 becomes evictable, p2 stays held
            big = [(Float32, (4096,))]
            @test with_cached_host_weights(a -> 0, store, "big", weights_digest("big", big), big, a -> nothing) === nothing
            @test isfile(region("p3"))
            materialize_host_weights!(store, "p3", weights_digest("p3", specs), specs, a -> (a[1] .= 9))
            store.budget = 2R

            # Under a budget a released floor stays cached; trimming to a lower budget reclaims it.
            release_host_weights!(store, "p1")
            @test isfile(region("p1"))
            store.budget = 0
            @test trim_host_cache!(store) == 1
            @test !isfile(region("p1")) && isfile(region("p2")) && isfile(region("p3"))
            release_host_weights!(store, "p2")
            release_host_weights!(store, "p3")
            @test trim_host_cache!(store) == 2 && host_cache_bytes(store) == 0
            # A zero budget caches nothing.
            @test load("m1") === nothing && host_cache_bytes(store) == 0
        finally
            for k in collect(keys(store.attached))
                release_host_weights!(store, k)
            end
            _budget_cleanup(prefix)
        end
    end
end

@testset "SharedWeightStore node budget: a region another process is uploading from is not evicted" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        prefix = "tx" * string(getpid())
        specs = [(Float32, (1024,))]
        R = ReactantServerCore._layout(specs)[2]
        dir = mktempdir()
        readyf = joinpath(dir, "ready")
        stopf = joinpath(dir, "stop")
        script = joinpath(dir, "uploader.jl")
        write(
            script, """
            using ReactantServerCore
            prefix, readyf, stopf, R = ARGS[1], ARGS[2], ARGS[3], parse(Int, ARGS[4])
            specs = [(Float32, (1024,))]
            store = SharedWeightStore(; budget = R, prefix = prefix)
            with_cached_host_weights(store, "held", weights_digest("held", specs), specs, a -> (a[1] .= 1)) do arrays
                write(readyf, "ok")         # mid-upload: the region is held until this returns
                t0 = time(); while !isfile(stopf) && time() - t0 < 30; sleep(0.05); end
            end
            """
        )
        proc = run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script $prefix $readyf $stopf $R`; wait = false)
        store = SharedWeightStore(; budget = R, prefix = prefix)
        other(k) = with_cached_host_weights(a -> a[1][1], store, k, weights_digest(k, specs), specs, a -> (a[1] .= 2))
        try
            t0 = time()
            while !isfile(readyf) && time() - t0 < 90
                sleep(0.1)
            end
            @test isfile(readyf)
            @test other("mine") === nothing                # the only region is in use: no room
            touch(stopf)
            wait(proc)
            @test other("mine") == (false, 2.0f0)          # released, so now it can be evicted
            @test host_cache_bytes(store) == R
        finally
            touch(stopf)
            wait(proc)
            _budget_cleanup(prefix)
            rm(dir; recursive = true, force = true)
        end
    end
end
