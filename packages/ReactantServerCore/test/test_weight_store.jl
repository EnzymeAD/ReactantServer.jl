# WeightStore tests. The private store is exercised in-process; the shared store's cross-process
# coordination (reuse of a READY region and last-one-out unlink) is exercised by spawning a second
# Julia process that attaches the same POSIX SHM region. Two processes on one host share /dev/shm,
# which is exactly what Docker `ipc: host` gives same-node worker containers.

@testset "PrivateWeightStore materializes fresh per-call arrays" begin
    specs = [(Float32, (4,)), (Float64, (2, 2))]
    dg = weights_digest("m", specs)
    arrs = materialize_host_weights!(PrivateWeightStore(), "m", dg, specs,
        a -> (a[1] .= Float32[1, 2, 3, 4]; a[2] .= 9.0))
    @test arrs[1] == Float32[1, 2, 3, 4]
    @test size(arrs[2]) == (2, 2) && all(==(9.0), arrs[2])
    @test eltype(arrs[1]) == Float32 && eltype(arrs[2]) == Float64
    release_host_weights!(PrivateWeightStore(), "m")   # no-op
end

@testset "weights_digest is stable and layout-sensitive" begin
    @test weights_digest("m", [(Float32, (4,))]) == weights_digest("m", [(Float32, (4,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("m", [(Float32, (8,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("n", [(Float32, (4,))])
    @test weights_digest("m", [(Float32, (4,))]) != weights_digest("m", [(Float64, (4,))])
end

@testset "SharedWeightStore: single-process create, reuse, and unlink" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsone_" * string(getpid())
        specs = [(Float32, (4,))]
        dg = weights_digest(key, specs)
        region = "/dev/shm/rsw-" * key * "-" * string(dg; base=16)

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

@testset "SharedWeightStore: a second process reuses the region; last out unlinks" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        key = "wsx_" * string(getpid())
        specs = [(Float32, (4,))]
        dg = weights_digest(key, specs)
        region = "/dev/shm/rsw-" * key * "-" * string(dg; base=16)

        dir = mktempdir()
        readyf = joinpath(dir, "ready")
        stopf = joinpath(dir, "stop")
        script = joinpath(dir, "holder.jl")
        write(script, """
        using ReactantServerCore
        key, readyf, stopf = ARGS[1], ARGS[2], ARGS[3]
        specs = [(Float32, (4,))]
        store = SharedWeightStore()
        materialize_host_weights!(store, key, weights_digest(key, specs), specs,
            a -> (a[1] .= Float32[10, 20, 30, 40]))
        write(readyf, "ok")
        t0 = time(); while !isfile(stopf) && time() - t0 < 30; sleep(0.05); end
        release_host_weights!(store, key)
        """)

        proc = run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script $key $readyf $stopf`; wait=false)
        try
            t0 = time()
            while !isfile(readyf) && time() - t0 < 90; sleep(0.1); end
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
        rm(dir; recursive=true, force=true)
    end
end
