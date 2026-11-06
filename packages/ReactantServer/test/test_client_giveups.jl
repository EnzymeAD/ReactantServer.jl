# A client that gives up on calls (its deadline passes, or it cancels) resets one HTTP/2 stream on
# its connection to the worker each time, and by default the worker must keep answering it promptly
# however many it has given up on (GW-19, the worker's side of GW-18). At HTTP.jl's default stream
# limit the listener sent GOAWAY(ENHANCE_YOUR_CALM) after the 200th reset; libcurl, under
# gRPCClient, then parked every later call behind the replacement connection until the call's
# deadline whenever that connection's first calls were give-ups too. `grpc.rapid_reset_protection`
# brings that limit back for an operator who wants it. Reuses `_write_slow_bundle`, `_slow_infer`,
# `_DrInf` and `_DrCore` from test_drain.jl.

import gRPCClient

# Admission uncapped (the three-argument EndpointsConfig), so every give-up reaches the listener as
# a reset rather than being shed with RESOURCE_EXHAUSTED while the worker still runs earlier ones.
_giveups_cfg(root, port; rapid_reset_protection::Bool = false) = ReactantServer.ServerConfig(
    [root], "",
    ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 1024, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port), String[], 0.0, ReactantServer.STATIC,
    ReactantServer.GrpcConfig(
        _DrCore.DEFAULT_GRPC_MSG_BYTES, _DrCore.DEFAULT_GRPC_MSG_BYTES,
        ReactantServer.ERROR_DETAIL_REDACTED, rapid_reset_protection
    )
)

@testset "a client that gives up on many calls keeps getting prompt answers from a worker (GW-19)" begin
    mktempdir() do root
        _write_slow_bundle(root)
        port = grpc_free_port()
        srv = ReactantServer.serve(_giveups_cfg(root, port); backend = ReactantServer.ReactantBackend(), blocking = false)
        # A handle of the test's own, so the calls share one connection to the worker, as the calls
        # of a client process do.
        grpc = gRPCClient.gRPCCURL()
        try
            # By default the listener's stream limit, and with it HTTP.jl's reset budget, is out of reach.
            @test !srv.config.grpc.rapid_reset_protection
            @test srv.server.config.max_concurrent_streams == typemax(Int32)
            chan = gRPCClient.gRPCChannel("127.0.0.1", port; grpc = grpc)
            # The worker answers y = 2x after sleeping x[1] = `secs` seconds in preprocess.
            function timed(secs, deadline)
                x = Float32[secs, 1, 2, 3]
                inp = _DrInf.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[4])
                req = _DrInf.ModelInferRequest(;
                    model_name = "slow4", inputs = [inp], raw_input_contents = [collect(reinterpret(UInt8, x))]
                )
                t = time()
                r = try
                    WorkerRPC.ModelInfer(chan, req; deadline = deadline)
                catch e
                    e
                end
                return r, time() - t
            end
            gave_up(r) = r isa gRPCClient.gRPCServiceCallException && r.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED
            answered(r) = r isa _DrInf.ModelInferResponse &&
                collect(reinterpret(Float32, r.raw_output_contents[1])) == Float32[0, 2, 4, 6]

            @test answered(first(timed(0, 10)))

            # 300 give-ups, more than the 200 resets that ended the connection at a limit of 100, from
            # 6 callers at once: each call wants 0.2 s of the worker and gives up after 20 ms.
            n = Threads.Atomic{Int}(0)
            @sync for _ in 1:6
                @async for _ in 1:50
                    gave_up(first(timed(0.2, 0.02))) && Threads.atomic_add!(n, 1)
                end
            end
            @test n[] == 300
            # And one more on its own, so that a replacement connection would open on a give-up.
            @test gave_up(first(timed(0.2, 0.02)))
            # Let the worker finish what it was handed, so the calls below find it idle.
            @test timedwait(() -> srv.drain.inflight[] == 0, 30) === :ok

            # Each answers in milliseconds; parked behind a replacement connection, each would take its
            # full 5 s deadline instead.
            for _ in 1:3
                r, dt = timed(0, 5)
                @test answered(r)
                @test dt < 2
            end
        finally
            gRPCClient.grpc_shutdown(grpc)
            ReactantServer.stop!(srv)
        end
    end
end

@testset "worker grpc.rapid_reset_protection: on restores the 100-stream limit" begin
    mktempdir() do root
        _write_slow_bundle(root)
        port = grpc_free_port()
        # On: the listener takes gRPCServer's default of 100 streams per connection, from which
        # HTTP.jl derives its budget of 200 resets per connection.
        srv = ReactantServer.serve(
            _giveups_cfg(root, port; rapid_reset_protection = true);
            backend = ReactantServer.ReactantBackend(), blocking = false
        )
        try
            @test srv.config.grpc.rapid_reset_protection
            @test srv.server.config.max_concurrent_streams == 100
            @test _slow_infer(port, 0) == Float32[0, 2, 4, 6]
        finally
            ReactantServer.stop!(srv)
        end
    end
end
