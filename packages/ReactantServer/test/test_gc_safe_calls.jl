# GcSafeXLA (runtime/gc_safe_calls.jl): the hot path's libReactantExtra calls declared gc_safe.
# Checked on Reactant's CPU client: results match, frees are idempotent, a C++ error thrown from
# inside a gc_safe call unwinds cleanly and leaves the runtime usable, and garbage collections
# forced on another thread while the calls run cannot corrupt the data they read or write.

const _GCX = ReactantServer.GcSafeXLA

@testset "gc_safe calls (CPU)" begin
    backend = ReactantServer.ReactantBackend()
    pool = ReactantServer.resolve_client(backend, ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
    client, device = pool.client, pool.device
    # y = x .* w over 1024 floats.
    mlir = """
    module {
      func.func @main(%x: tensor<1024xf32>, %w: tensor<1024xf32>) -> tensor<1024xf32> {
        %0 = stablehlo.multiply %x, %w : tensor<1024xf32>
        return %0 : tensor<1024xf32>
      }
    }
    """
    exec = ReactantServer.compile_artifact(backend, pool, stablehlo_artifact(mlir), 2, 1)
    run(x, w) = begin
        bx = ReactantServer.to_device(backend, client, x, device)
        bw = ReactantServer.to_device(backend, client, w, device)
        outs = ReactantServer.execute_single_device(backend, exec, device, Any[bx, bw], [false, false], 1)
        y = ReactantServer.to_host!(backend, outs[1], Vector{Float32}(undef, 1024))
        foreach(b -> ReactantServer.free_buffer!(backend, b), (bx, bw, outs[1]))
        y
    end

    @testset "round trip and idempotent free" begin
        a = rand(Float32, 3, 5)
        b = _GCX.upload(client, a, device)
        @test size(b) == (5, 3)                                   # XLA row-major shape, as before
        @test _GCX.to_host!(b, Array{Float32}(undef, 3, 5)) == a
        _GCX.free!(b)
        @test b.buffer == C_NULL
        _GCX.free!(b)                                             # a second free is a no-op
        @test_throws ArgumentError _GCX.to_host!(b, Array{Float32}(undef, 3, 5))
        GC.gc()                                                   # the finalizer is gone: nothing double-frees
        x, w = rand(Float32, 1024), rand(Float32, 1024)
        @test run(x, w) == x .* w
    end

    @testset "a C++ error inside a gc_safe call unwinds and leaves the runtime usable" begin
        bx = _GCX.upload(client, rand(Float32, 1024), device)
        bw = _GCX.upload(client, rand(Float32, 1024), device)
        # Asking for two outputs from a one-output program makes XLAExecuteSharded report the
        # mismatch through Reactant's error callback, from inside the gc_safe call.
        err = try
            _GCX.execute(exec, device, Ptr{Cvoid}[bx.buffer, bw.buffer], UInt8[0, 0], 2)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("num_results", sprint(showerror, err))
        GC.gc()                                                   # the collector still works afterwards
        _GCX.free!(bx); _GCX.free!(bw)
        x, w = rand(Float32, 1024), rand(Float32, 1024)
        @test run(x, w) == x .* w                                 # and so does the runtime
    end

    # A program can call back into Julia mid-execution (Reactant's julia_callback custom call, an XLA
    # FFI handler calling a @cfunction). On the CPU client it runs on the calling thread, inside the
    # gc_safe execute: the cfunction entry must return the thread to GC-unsafe, and a collection
    # forced inside the callback, or on another thread meanwhile, must neither deadlock nor corrupt.
    # It uses a Reactant internal, so it is skipped on a Reactant without it.
    if isdefined(Reactant.Ops, :_make_c_callback)
        @testset "a Julia callback during a gc_safe execute" begin
            threads_seen = Int[]
            function cbf(out, x)
                garbage = [rand(256) for _ in 1:64]
                GC.gc(false)
                out .= 2 .* x
                push!(threads_seen, Threads.threadid() + 0 * length(garbage))
                return nothing
            end
            ptr = Reactant.Ops._make_c_callback(cbf, ((Float32, (1024,)),), ((Float32, (1024,)),))
            cb_mlir = """
            module {
              func.func @main(%x: tensor<1024xf32>) -> tensor<1024xf32> {
                %0 = stablehlo.custom_call @reactant_julia_callback(%x) {api_version = 4 : i32, backend_config = {callback_ptr = $(Int64(UInt(ptr))) : i64}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<1024xf32>) -> tensor<1024xf32>
                return %0 : tensor<1024xf32>
              }
            }
            """
            cb_exec = ReactantServer.compile_artifact(backend, pool, stablehlo_artifact(cb_mlir), 1, 1)
            stop = Threads.Atomic{Bool}(false)
            churn = Threads.@spawn while !stop[]
                GC.gc(false)
                sleep(0.001)
            end
            correct = 0
            for _ in 1:50
                x = rand(Float32, 1024)
                bx = _GCX.upload(client, x, device)
                outs = _GCX.execute(cb_exec, device, Ptr{Cvoid}[bx.buffer], UInt8[0], 1)
                y = _GCX.to_host!(outs[1], Vector{Float32}(undef, 1024))
                _GCX.free!(bx); _GCX.free!(outs[1])
                correct += y == 2 .* x
            end
            stop[] = true
            wait(churn)
            @test correct == 50
            @test length(threads_seen) == 50
        end
    end

    @testset "collections on another thread while the calls run" begin
        stop = Threads.Atomic{Bool}(false)
        collections = Threads.Atomic{Int}(0)
        churn = Threads.@spawn begin
            junk = Any[]
            while !stop[]
                push!(junk, rand(UInt8, 1 << 16))
                length(junk) > 64 && empty!(junk)
                if rand() < 0.05
                    GC.gc(false)
                    Threads.atomic_add!(collections, 1)
                end
                yield()
            end
        end
        # With threads, start once the churn task is collecting and keep going until at least ten
        # collections have interleaved with the calls, so the test cannot pass without overlap.
        if Threads.nthreads() > 1
            t0 = time()
            while collections[] == 0 && time() - t0 < 30
                sleep(0.01)
            end
        end
        ok = true
        rounds = 0
        while rounds < 300 || (Threads.nthreads() > 1 && collections[] < 10 && rounds < 20_000)
            x, w = rand(Float32, 1024), rand(Float32, 1024)
            ok &= run(x, w) == x .* w
            rounds += 1
        end
        stop[] = true
        wait(churn)
        @test ok
        Threads.nthreads() > 1 && @test collections[] >= 10       # collections really did interleave
    end
end
