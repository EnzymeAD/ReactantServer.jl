# Graceful drain of the gateway. Once `stop!(gw; drain_seconds)` starts (or, for a blocking
# `serve_gateway`, a drain is requested), a new ModelInfer gets UNAVAILABLE and /readyz reports
# 503, while the requests already forwarded finish on the worker.

import gRPCServer
import gRPCClient
import HTTP
using Sockets

const _DGInf = ReactantServerGateway.inference

# A mock worker serving one model, "slow", whose ModelInfer sleeps for as many seconds as the
# request's id says, then answers with that id.
function _slow_worker()
    server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", 1)
    server.port = 0
    ReactantServerGateway.register_GRPCInferenceService!(
        server;
        ServerReady = (ctx, req) -> _DGInf.ServerReadyResponse(; ready = true),
        RepositoryIndex = (ctx, req) -> _DGInf.RepositoryIndexResponse(;
            models = [_DGInf.var"RepositoryIndexResponse.ModelIndex"(; name = "slow", version = "", state = "READY", reason = "")]
        ),
        ModelInfer = (ctx, req) -> begin
            sleep(parse(Float64, req.id))
            _DGInf.ModelInferResponse(; model_name = "slow", id = req.id)
        end,
    )
    return server, grpc_start_ephemeral!(server)
end

_gw_slow(port, secs) = grpc_call(
    _DGInf.GRPCInferenceService.ModelInfer, port,
    _DGInf.ModelInferRequest(; model_name = "slow", id = string(secs)); deadline = 30
)

_gw_status(f) = try
    f()
    nothing
catch e
    e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : e
end

_gw_readyz(port) = try
    HTTP.get("http://127.0.0.1:$port/readyz"; retry = false, status_exception = false, readtimeout = 10).status
catch
    0
end

_gw_open(port) = try
    close(Sockets.connect(Sockets.localhost, port))
    true
catch
    false
end

function _gateway_file(worker_port; grpc = "127.0.0.1:0", metrics = "127.0.0.1:0")
    path = tempname() * ".yaml"
    write(
        path, """
        listen:
          grpc: "$grpc"
          metrics: "$metrics"
        endpoints:
          - "127.0.0.1:$worker_port"
        """
    )
    return path
end

const _UNAVAILABLE = Int(gRPCServer.StatusCode.UNAVAILABLE)

@testset "gateway drain: refuses new requests, finishes the forwarded ones" begin
    worker, wport = _slow_worker()
    gw = ReactantServerGateway.serve_gateway(_gateway_file(wport); blocking = false)
    stopper = nothing
    try
        gport, aport = ReactantServerGateway.bound_ports(gw)
        @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
        @test timedwait(() -> _gw_readyz(aport) == 200, 30) === :ok

        inflight = Threads.@spawn _gw_slow(gport, 3)
        @test timedwait(() -> gw.drain.inflight[] == 1, 30) === :ok
        stopper = Threads.@spawn ReactantServerGateway.stop!(gw; drain_seconds = 30)
        @test timedwait(() -> ReactantServerCore.is_draining(gw.drain), 10) === :ok

        @test _gw_status(() -> _gw_slow(gport, 0)) == _UNAVAILABLE
        @test _gw_readyz(aport) == 503
        @test !istaskdone(stopper)                       # still waiting for the forwarded request

        r = fetch(inflight)
        @test (r.model_name, r.id) == ("slow", "3")
        @test timedwait(() -> istaskdone(stopper), 30) === :ok
        fetch(stopper)
        stopper = nothing
        @test !_gw_open(gport)
        @test !_gw_open(aport)
    finally
        stopper === nothing ? ReactantServerGateway.stop!(gw) : wait(stopper)
        gRPCServer.stop!(worker; force = true)
    end
end

@testset "gateway drain: a blocking serve_gateway drains and returns on request" begin
    worker, wport = _slow_worker()
    gport, aport = grpc_free_port(), grpc_free_port()
    ReactantServerCore._reset_drain!()
    server = Threads.@spawn ReactantServerGateway.serve_gateway(
        _gateway_file(wport; grpc = "127.0.0.1:$gport", metrics = "127.0.0.1:$aport")
    )
    try
        @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing || istaskdone(server), 60) === :ok
        @test !istaskdone(server)
        inflight = Threads.@spawn _gw_slow(gport, 2)
        sleep(0.5)                                       # let the request be forwarded
        @test ReactantServerCore.request_drain!(30)      # a serving gateway acts on it
        @test timedwait(() -> _gw_readyz(aport) == 503 || istaskdone(server), 10) === :ok
        r = fetch(inflight)
        @test (r.model_name, r.id) == ("slow", "2")
        @test timedwait(() -> istaskdone(server), 30) === :ok
        @test fetch(server) === nothing
        @test !_gw_open(gport)
        @test !_gw_open(aport)
    finally
        istaskdone(server) || (ReactantServerCore.request_drain!(0); wait(server))
        ReactantServerCore._reset_drain!()
        gRPCServer.stop!(worker; force = true)
    end
end
