# The gateway's inbound admission cap (GW-17): `grpc.max_concurrent_requests_per_worker` times the
# number of workers. A call past it is shed at once with RESOURCE_EXHAUSTED, before any gateway
# handler runs, and the admission metrics on /metrics say so; 0 means uncapped. Reuses
# `_slow_worker`, `_gw_slow` and `_gw_status` from test_drain.jl.

import HTTP

function _admission_gateway(worker_port, per_worker)
    path = tempname() * ".yaml"
    write(
        path, """
        listen:
          grpc: "127.0.0.1:0"
          metrics: "127.0.0.1:0"
        grpc:
          max_concurrent_requests_per_worker: $per_worker
        endpoints:
          - "127.0.0.1:$worker_port"
        """
    )
    return ReactantServerGateway.serve_gateway(path; blocking = false)
end

# The admission series from the gateway's /metrics, by name.
function _admission_metrics(admin_port)
    body = String(HTTP.get("http://127.0.0.1:$admin_port/metrics"; retry = false).body)
    out = Dict{String, Float64}()
    for l in split(body, '\n')
        m = match(r"^(gateway_inflight_requests|gateway_requests_shed_total|gateway_max_concurrent_requests) (\S+)$", l)
        m === nothing || (out[m[1]] = parse(Float64, m[2]))
    end
    return out
end

@testset "gateway admission: sheds past the inbound cap; 0 is uncapped (GW-17)" begin
    worker, wport = _slow_worker()
    try
        # A cap of 2 (2 per worker, one worker): with two calls held in flight, a third is shed.
        gw = _admission_gateway(wport, 2)
        try
            gport, aport = ReactantServerGateway.bound_ports(gw)
            @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
            held = [Threads.@spawn(_gw_slow(gport, 3)) for _ in 1:2]
            @test timedwait(() -> gw.server.inflight[] == 2, 30) === :ok
            t = time()
            @test _gw_status(() -> _gw_slow(gport, 0)) == Int(gRPCServer.StatusCode.RESOURCE_EXHAUSTED)
            @test time() - t < 2                                  # shed at once, not queued
            m = _admission_metrics(aport)
            @test m["gateway_max_concurrent_requests"] == 2
            @test m["gateway_inflight_requests"] == 2
            @test m["gateway_requests_shed_total"] == 1
            @test all(r -> r.id == "3", fetch.(held))             # the admitted calls finish
            @test timedwait(() -> gw.server.inflight[] == 0, 30) === :ok
        finally
            ReactantServerGateway.stop!(gw)
        end

        # 0 per worker: uncapped. Twice as many calls as the cap above run at once, none is shed,
        # and the in-flight gauge still counts them.
        gw = _admission_gateway(wport, 0)
        try
            gport, aport = ReactantServerGateway.bound_ports(gw)
            @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
            held = [Threads.@spawn(_gw_slow(gport, 2)) for _ in 1:4]
            @test timedwait(() -> gw.server.inflight[] == 4, 30) === :ok
            m = _admission_metrics(aport)
            @test m["gateway_max_concurrent_requests"] == 0
            @test m["gateway_inflight_requests"] == 4
            @test m["gateway_requests_shed_total"] == 0
            @test all(r -> r.id == "2", fetch.(held))
        finally
            ReactantServerGateway.stop!(gw)
        end
    finally
        gRPCServer.stop!(worker; force = true)
    end
end
