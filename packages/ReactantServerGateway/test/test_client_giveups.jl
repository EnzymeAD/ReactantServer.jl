# A client that gives up on calls (its deadline passes, or it cancels) resets one HTTP/2 stream on
# its connection to the gateway each time, and by default the gateway must keep answering it
# promptly however many it has given up on (GW-18). At HTTP.jl's default stream limit the gateway's
# listener sent GOAWAY(ENHANCE_YOUR_CALM) after the 200th reset; libcurl, under gRPCClient, then
# parked every later call behind the replacement connection until the call's deadline, because that
# connection's first calls were give-ups too and it never answered one. `grpc.rapid_reset_protection`
# brings that limit back for an operator who wants it. Reuses `_slow_worker` and `_gateway_file`
# from test_drain.jl.

import gRPCClient

@testset "a client that gives up on many calls keeps getting prompt answers (GW-18)" begin
    worker, wport = _slow_worker()
    # Uncapped admission: every give-up below must reach the listener as a reset, not be shed with
    # RESOURCE_EXHAUSTED while the gateway still runs the calls given up before it.
    path = _gateway_file(wport)
    write(path, replace(read(path, String), "endpoints:" => "grpc:\n  max_concurrent_requests_per_worker: 0\nendpoints:"))
    gw = ReactantServerGateway.serve_gateway(path; blocking = false)
    # A handle of the test's own, so the calls share one connection to the gateway, as the calls of
    # a client process do.
    grpc = gRPCClient.gRPCCURL()
    try
        # By default the listener's stream limit, and with it HTTP.jl's reset budget, is out of reach.
        @test !gw.cfg.rapid_reset_protection
        @test gw.server.config.max_concurrent_streams == typemax(Int32)
        gport, _ = ReactantServerGateway.bound_ports(gw)
        @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
        chan = gRPCClient.gRPCChannel("127.0.0.1", gport; grpc = grpc)
        # The worker answers after `secs` seconds.
        call(secs, deadline) = _DGInf.GRPCInferenceService.ModelInfer(
            chan, _DGInf.ModelInferRequest(; model_name = "slow", id = string(secs)); deadline = deadline
        )
        function timed(secs, deadline)
            t = time()
            r = try
                call(secs, deadline)
            catch e
                e
            end
            return r, time() - t
        end
        gave_up(r) = r isa gRPCClient.gRPCServiceCallException && r.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED

        r, _ = timed(0, 10)
        @test r isa _DGInf.ModelInferResponse && r.id == "0"

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
        # Let the gateway finish what it forwarded, so the calls below find the worker idle.
        @test timedwait(() -> gw.drain.inflight[] == 0, 30) === :ok

        # Each answers in milliseconds; parked behind a replacement connection, each would take its
        # full 5 s deadline instead.
        for _ in 1:3
            r, dt = timed(0, 5)
            @test r isa _DGInf.ModelInferResponse && r.id == "0"
            @test dt < 2
        end
    finally
        gRPCClient.grpc_shutdown(grpc)
        ReactantServerGateway.stop!(gw)
        gRPCServer.stop!(worker; force = true)
    end
end

@testset "grpc.rapid_reset_protection: off by default; on restores the 100-stream limit" begin
    GW = ReactantServerGateway
    eps = "endpoints:\n  - \"127.0.0.1:7001\"\n"
    function load(yaml; env = Pair{String, String}[])
        path = tempname() * ".yaml"
        write(path, yaml * eps)
        try
            return withenv(env...) do
                GW.load_gateway(path)
            end
        finally
            rm(path; force = true)
        end
    end
    @test !load("").rapid_reset_protection
    @test load("grpc:\n  rapid_reset_protection: true\n").rapid_reset_protection
    @test !load("grpc:\n  rapid_reset_protection: false\n").rapid_reset_protection
    @test load(""; env = ["REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION" => "true"]).rapid_reset_protection
    @test !load(
        "grpc:\n  rapid_reset_protection: true\n"; env = ["REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION" => "0"]
    ).rapid_reset_protection
    @test_throws ReactantServerCore.ConfigError load("grpc:\n  rapid_reset_protection: sometimes\n")

    # On: the listener takes gRPCServer's default of 100 streams per connection, from which HTTP.jl
    # derives its budget of 200 resets per connection.
    worker, wport = _slow_worker()
    path = _gateway_file(wport)
    write(path, replace(read(path, String), "endpoints:" => "grpc:\n  rapid_reset_protection: true\nendpoints:"))
    gw = GW.serve_gateway(path; blocking = false)
    try
        @test gw.cfg.rapid_reset_protection
        @test gw.server.config.max_concurrent_streams == 100
        gport, _ = GW.bound_ports(gw)
        @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
    finally
        GW.stop!(gw)
        gRPCServer.stop!(worker; force = true)
    end
end
