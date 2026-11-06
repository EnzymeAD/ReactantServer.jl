# The worker's gRPC admission series on /metrics. With `endpoints.max_concurrent_requests: 0`
# (uncapped) the worker used to pass gRPCServer no cap at all, which also turned off gRPCServer's
# in-flight counter, so `worker_inflight_requests` read 0 at any load (WRK-16). Reuses
# `_write_slow_bundle` and `_slow_infer` from test_drain.jl.

import HTTP

_admission_cfg(root, port, mport, cap) = ReactantServer.ServerConfig(
    [root], "",
    ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 64, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port, mport, cap)
)

# The admission series from the worker's /metrics, by name.
function _worker_admission_metrics(mport)
    body = String(HTTP.get("http://127.0.0.1:$mport/metrics"; retry = false).body)
    out = Dict{String, Float64}()
    for l in split(body, '\n')
        m = match(r"^(worker_inflight_requests|worker_requests_shed_total|worker_max_concurrent_requests)(?:\{[^}]*\})? (\S+)$", l)
        m === nothing || (out[m[1]] = parse(Float64, m[2]))
    end
    return out
end

@testset "worker admission: uncapped, the calls in flight are still counted (WRK-16)" begin
    mktempdir() do root
        _write_slow_bundle(root)
        port, mport = grpc_free_port(), grpc_free_port()
        srv = ReactantServer.serve(_admission_cfg(root, port, mport, 0); backend = ReactantServer.ReactantBackend(), blocking = false)
        try
            @test _slow_infer(port, 0) == Float32[0, 2, 4, 6]       # warm the request path
            # Four calls held at once: none is shed, and the in-flight gauge counts every one.
            held = [Threads.@spawn(_slow_infer(port, 3)) for _ in 1:4]
            @test timedwait(() -> srv.server.inflight[] == 4, 30) === :ok
            m = _worker_admission_metrics(mport)
            @test m["worker_max_concurrent_requests"] == 0          # still reported as configured
            @test m["worker_inflight_requests"] == 4
            @test m["worker_requests_shed_total"] == 0
            @test all(==(Float32[6, 2, 4, 6]), fetch.(held))
            @test timedwait(() -> srv.server.inflight[] == 0, 30) === :ok
        finally
            ReactantServer.stop!(srv)
        end
    end
end
