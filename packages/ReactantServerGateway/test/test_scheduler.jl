# The gateway scheduler interface: the factory picks the right scheduler per mode, round_robin
# rotates over discovered routes, and least_outstanding routes to the least in-flight replica and
# restores the count on release. The lpt_packing scheduler is covered in test_lpt_packing.jl.

const GW = ReactantServerGateway

# A GatewayConfig with the given scheduling mode and two dummy endpoints (no servers are contacted;
# the lightweight-scheduler select_replicas paths read only the routing table).
function _sched_cfg(mode)
    return GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", ["127.0.0.1:7001", "127.0.0.1:7002"],
                            String[], 60, 1, 1, "info", "json", mode, 30.0, 0.0, 0.8, 0.1, 30.0,
                            1, 1.0, "fill_rr", Dict{String,GW.GatewayModelConfig}(), 32, 64, :off, 0)
end

# Build a ScheduleContext for `model` over a routing table mapping each model to worker URLs.
function _ctx(model, table::Dict)
    cfg = _sched_cfg("round_robin")
    pool = GW.ClientPool(cfg)
    routes = GW.DiscoveredRoutes()
    GW.swap_table!(routes, GW.RoutingTable(table))
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics)
    ctx = GW.ScheduleContext(model, "id", pool, routes, metrics, refresher)
    return ctx, pool
end

@testset "make_scheduler: mode selects the scheduler type" begin
    @test GW.make_scheduler(_sched_cfg("round_robin")) isa GW.RoundRobinScheduler
    @test GW.make_scheduler(_sched_cfg("least_outstanding")) isa GW.LeastOutstandingScheduler
    @test GW.make_scheduler(_sched_cfg("lpt_packing")) isa GW.LptPackingState
end

@testset "RoundRobinScheduler: rotates over discovered replicas" begin
    s = GW.RoundRobinScheduler()
    ctx, pool = _ctx("m", Dict("m" => ["127.0.0.1:7001", "127.0.0.1:7002"]))
    try
        u1, r1 = GW.select_replicas(s, ctx)
        u2, r2 = GW.select_replicas(s, ctx)
        @test Set(u1) == Set(["127.0.0.1:7001", "127.0.0.1:7002"])   # both present as failover
        @test u1[1] != u2[1]                                         # the choice rotates
        @test r1 === nothing && r2 === nothing                       # nothing to reserve
        GW.release!(s, r1)                                           # no-op, must not throw
        # An unknown model has no route.
        ctxu, _ = _ctx("ghost", Dict("m" => ["127.0.0.1:7001"]))
        @test GW.select_replicas(s, ctxu) === nothing
    finally
        GW.close_pool!(pool)
    end
end

@testset "LeastOutstandingScheduler: routes to the least in-flight replica" begin
    s = GW.LeastOutstandingScheduler()
    ctx, pool = _ctx("m", Dict("m" => ["127.0.0.1:7001", "127.0.0.1:7002"]))
    try
        # First pick is the URL-tiebreak winner; holding it in flight pushes the next to the other.
        u1, res1 = GW.select_replicas(s, ctx)
        @test u1[1] == "127.0.0.1:7001"
        u2, res2 = GW.select_replicas(s, ctx)
        @test u2[1] == "127.0.0.1:7002"
        # With one in flight on each, the tie returns to the URL order.
        u3, res3 = GW.select_replicas(s, ctx)
        @test u3[1] == "127.0.0.1:7001"
        # Releasing 7001's two reservations makes it the least loaded again.
        GW.release!(s, res1)
        GW.release!(s, res3)
        u4, res4 = GW.select_replicas(s, ctx)
        @test u4[1] == "127.0.0.1:7001"
        foreach(r -> GW.release!(s, r), (res2, res4))
        @test all(c -> c[] == 0, values(@atomic s.inflight))        # every counter back to baseline
        # Unknown model: no route.
        ctxu, _ = _ctx("ghost", Dict("m" => ["127.0.0.1:7001"]))
        @test GW.select_replicas(s, ctxu) === nothing
    finally
        GW.close_pool!(pool)
    end
end
