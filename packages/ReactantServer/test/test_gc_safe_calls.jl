# GcSafeXLA (runtime/gc_safe_calls.jl): the hot path's libReactantExtra calls and the XLA compile
# declared gc_safe. Checked on Reactant's CPU client: results match, frees are idempotent, a C++
# error thrown from inside a gc_safe call (an execution or a compile) unwinds cleanly and leaves the
# runtime usable, garbage collections forced on another thread while the calls run cannot corrupt
# the data they read or write, and a collection does not wait for a compile to end.

const _GCX = ReactantServer.GcSafeXLA

# A program XLA takes a while to compile on the CPU (about 2 s at n = 1500 on netai02): a chain of n
# matrix products, each through a tanh so that nothing fuses away.
function _gcx_slow_module(n::Int)
    io = IOBuffer()
    println(io, "module {\n  func.func @main(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {")
    prev = "%x"
    for i in 1:n
        println(io, "    %d$i = stablehlo.dot_general $prev, $prev, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>")
        println(io, "    %t$i = stablehlo.tanh %d$i : tensor<64x64xf32>")
        prev = "%t$i"
    end
    println(io, "    return $prev : tensor<64x64xf32>\n  }\n}")
    return String(take!(io))
end

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
            # compile_artifact refuses a Julia callback (its pointer is only valid in the tracing
            # process, which here is this one), so compile with Reactant directly.
            cb_exec = Reactant.MLIR.IR.@with_context pool.ctx begin
                Reactant.XLA.compile(
                    client, parse(Reactant.MLIR.IR.Module, cb_mlir);
                    compile_options = ReactantServer._compile_options(pool, ReactantServer.device_ordinal(backend, device)),
                    num_parameters = Int64(1), num_outputs = Int64(1),
                    is_sharded = false, num_replicas = Int64(1), num_partitions = Int64(1),
                )
            end
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

    @testset "compile_artifact refuses a Julia callback" begin
        # The pointer is never followed, so any value will do.
        nested = """
        module {
          func.func @main(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
            %0 = "stablehlo.if"(%p) ({
              %1 = stablehlo.custom_call @reactant_julia_callback(%x) {api_version = 4 : i32, backend_config = {callback_ptr = 1 : i64}, has_side_effect = true} : (tensor<4xf32>) -> tensor<4xf32>
              stablehlo.return %1 : tensor<4xf32>
            }, {
              stablehlo.return %x : tensor<4xf32>
            }) : (tensor<i1>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        @test_throws "reactant_julia_callback" ReactantServer.compile_artifact(backend, pool, stablehlo_artifact(nested), 2, 1)
        # Another custom call target is not refused.
        other = replace(nested, "@reactant_julia_callback" => "@some_other_target")
        @test Reactant.MLIR.IR.@with_context(pool.ctx, ReactantServer.refuse_julia_callbacks(parse(Reactant.MLIR.IR.Module, other))) === nothing
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

    @testset "a compile error inside the gc_safe compile unwinds and leaves the runtime usable" begin
        # XLA rejects a custom call with no registered handler while it compiles, and reports it
        # through Reactant's error callback from inside the gc_safe call.
        bad = """
        module {
          func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.custom_call @no_such_target_xyz(%x) {api_version = 4 : i32} : (tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        err = try
            ReactantServer.compile_artifact(backend, pool, stablehlo_artifact(bad), 1, 1)
            nothing
        catch e
            e
        end
        @test err isa Reactant.XLA.ReactantInternalError
        @test occursin("no_such_target_xyz", sprint(showerror, err))
        GC.gc()                                                   # the collector still works afterwards
        again = ReactantServer.compile_artifact(backend, pool, stablehlo_artifact(mlir), 2, 1)
        @test again isa Reactant.XLA.PJRT.LoadedExecutable        # and so does compiling
        x, w = rand(Float32, 1024), rand(Float32, 1024)
        @test run(x, w) == x .* w
    end

    # The compile runs on a default-pool thread while this one collects every 10 ms. A collection
    # stops every thread at a safepoint, and a thread inside a plain ccall reaches none until the call
    # returns: with a plain compile the first collection requested during it waits for the whole
    # compile. With the gc_safe one each collection takes milliseconds. Needs a second thread.
    if Threads.nthreads(:default) + Threads.nthreads(:interactive) > 1
        @testset "a collection on another thread does not wait for a compile" begin
            artifact = stablehlo_artifact(_gcx_slow_module(1500))
            took = Ref(0.0)
            task = Threads.@spawn :default begin
                took[] = @elapsed ReactantServer.compile_artifact(backend, pool, artifact, 1, 1)
            end
            longest = 0.0
            collections = 0
            while !istaskdone(task)
                t0 = time()
                GC.gc(false)
                longest = max(longest, time() - t0)
                collections += 1
                sleep(0.01)
            end
            fetch(task)
            @test took[] > 0.5                                    # long enough for the check to mean something
            @test collections >= 10
            # A plain compile makes one collection last about as long as the compile itself.
            @test longest < took[] / 2
        end
    end
end
