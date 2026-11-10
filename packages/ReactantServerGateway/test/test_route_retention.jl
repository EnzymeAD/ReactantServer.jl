# What the routing table keeps when a worker does not answer (GW-11). Before, a worker whose
# RepositoryIndex call timed out or failed got no routes for the round, a round every worker missed
# emptied the table, and clients got NOT_FOUND, which they do not retry; with one worker, every
# restart did that for its whole duration. Now a worker keeps the routes it last reported until it
# answers again, or until it has been unreachable or not ready for `route_retention_seconds`; while
# it is out, no request is sent to it: its models go to ready replicas when there are any, and a
# request for a model whose only workers are down is answered UNAVAILABLE without a worker call.
# Also the round-robin cursor, which every rebuild reset (GW-20). Reuses `MockWorker`, `_start_mock`,
# `_infer` and `_http_get` from test_gateway.jl.

const _RGW = ReactantServerGateway

function _retention_gateway(ports; retention = nothing, env = Pair{String, String}[])
    path = tempname() * ".yaml"
    yaml = "listen:\n  grpc: \"127.0.0.1:0\"\n  metrics: \"127.0.0.1:0\"\n"
    retention === nothing || (yaml *= "worker_client:\n  route_retention_seconds: $retention\n")
    yaml *= "endpoints:\n" * join(["  - \"127.0.0.1:$p\"\n" for p in ports])
    write(path, yaml)
    try
        return withenv(() -> _RGW.serve_gateway(path; blocking = false), env...)
    finally
        rm(path; force = true)
    end
end

# The gRPC status `f` ended with, or nothing when it succeeded.
_rstatus(f) = try
    f()
    nothing
catch e
    e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : rethrow()
end

_rurl(p) = "127.0.0.1:$p"

# A discovery scan now, as a request for an unrouted model would start one (past the refresher's
# one-second rate limit).
_rescan(gw) = (sleep(1.1); _RGW.refresh_now!(gw.server.context.refresher))

@testset "a discovery call that does not answer keeps the worker's routes (GW-11)" begin
    w0 = MockWorker("worker0", ["m"], false, false, false)
    w1 = MockWorker("worker1", ["m"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    gw = _retention_gateway([p0, p1])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> _rstatus(() -> _infer(port, "m")) === nothing, 30) === :ok
        @test Set(_RGW.replicas(gw.routes, "m")) == Set([_rurl(p0), _rurl(p1)])

        # One worker answers discovery too slowly (past the 5 s deadline): its routes stay.
        w1.index = :slow
        _rescan(gw)
        @test Set(something(_RGW.replicas(gw.routes, "m"), String[])) == Set([_rurl(p0), _rurl(p1)])

        # Every worker fails discovery in the same round: the table keeps every model, and requests
        # are still served instead of being answered NOT_FOUND.
        w0.index = w1.index = :down
        _rescan(gw)
        @test _rstatus(() -> _infer(port, "m")) === nothing
        @test Set(something(_RGW.replicas(gw.routes, "m"), String[])) == Set([_rurl(p0), _rurl(p1)])

        # An answer speaks for the worker: a model it no longer lists is dropped at once.
        w0.index = w1.index = :ok
        w1.models = String[]
        _rescan(gw)
        @test _RGW.replicas(gw.routes, "m") == [_rurl(p0)]
    finally
        _RGW.stop!(gw)
        close(s0)
        close(s1)
    end
end

@testset "requests go to ready replicas; what only a worker that is out serves is UNAVAILABLE, uncalled (GW-11)" begin
    w0 = MockWorker("worker0", ["shared", "only0"], false, false, false)
    w1 = MockWorker("worker1", ["shared"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    gw = _retention_gateway([p0, p1]; env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.5"])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> _rstatus(() -> _infer(port, "only0")) === nothing, 30) === :ok
        @test timedwait(() -> length(something(_RGW.replicas(gw.routes, "shared"), String[])) == 2, 30) === :ok

        # worker0 reports not ready (draining, say): the model it shares goes to worker1 alone...
        w0.ready = false
        @test timedwait(() -> _RGW.replicas(gw.routes, "shared") == [_rurl(p1)], 10) === :ok
        w0.infer_calls = w1.infer_calls = 0
        @test all(_ -> _infer(port, "shared").model_name == "worker1", 1:4)
        @test w0.infer_calls == 0
        # ...and the one only it serves is still known, so it is UNAVAILABLE rather than NOT_FOUND,
        # but answered without sending worker0 anything.
        @test _RGW.has_model(gw.routes, "only0")
        @test _RGW.replicas(gw.routes, "only0") == String[]
        @test _rstatus(() -> _infer(port, "only0")) == gRPCClient.GRPC_UNAVAILABLE
        @test w0.infer_calls == 0

        # Ready again: both replicas share the model.
        w0.ready = true
        @test timedwait(() -> length(something(_RGW.replicas(gw.routes, "shared"), String[])) == 2, 10) === :ok
    finally
        _RGW.stop!(gw)
        close(s0)
        close(s1)
    end
end

@testset "the only worker down: UNAVAILABLE, NOT_FOUND once its routes expire, routed when back (GW-11)" begin
    w = MockWorker("solo", ["m"], false, false, false)
    s, p = _start_mock(w)
    gw = _retention_gateway([p]; retention = 4, env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.5"])
    try
        @test gw.cfg.route_retention_seconds == 4.0
        port, aport = _RGW.bound_ports(gw)
        @test timedwait(() -> _rstatus(() -> _infer(port, "m")) === nothing, 30) === :ok

        # The worker goes away, as when it is killed and restarted.
        gRPCServer.stop!(s; force = true)
        t_down = time()
        @test timedwait(() -> _http_get(aport, "/readyz").status == 503, 10) === :ok   # the prober saw it
        sleep(1.0)                                     # and discovery has failed to reach it since
        # Its routes stay, so the model is still known, but no request is sent to a worker that is
        # down: the gateway answers UNAVAILABLE itself, and the client may retry.
        @test _rstatus(() -> _infer(port, "m")) == gRPCClient.GRPC_UNAVAILABLE
        @test _RGW.has_model(gw.routes, "m")
        @test _RGW.replicas(gw.routes, "m") == String[]

        # Once it has been out longer than the retention, its routes are dropped.
        @test timedwait(() -> _rstatus(() -> _infer(port, "m")) == gRPCClient.GRPC_NOT_FOUND, 15) === :ok
        @test time() - t_down >= 4

        # Back on its port: routed again.
        s, _ = _start_mock(w; port = p)
        @test timedwait(() -> _rstatus(() -> _infer(port, "m")) === nothing, 30) === :ok
    finally
        _RGW.stop!(gw)
        gRPCServer.stop!(s; force = true)
    end
end

@testset "round robin carries on across a routing table rebuild (GW-20)" begin
    w0 = MockWorker("worker0", ["r"], false, false, false)
    w1 = MockWorker("worker1", ["r"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    gw = _retention_gateway([p0, p1])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> length(something(_RGW.replicas(gw.routes, "r"), String[])) == 2, 30) === :ok
        # A rebuild between every request: each one must still go to the other replica, not back to
        # the first one in the list.
        seen = String[]
        for _ in 1:4
            push!(seen, _infer(port, "r").model_name)
            _rescan(gw)
        end
        @test all(i -> seen[i] != seen[i + 1], 1:3)
    finally
        _RGW.stop!(gw)
        close(s0)
        close(s1)
    end
end
