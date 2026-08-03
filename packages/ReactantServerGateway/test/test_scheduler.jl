# The gateway scheduler interface: the factory picks the right scheduler per mode, round_robin
# rotates over discovered routes, and least_outstanding routes to the least in-flight replica and
# restores the count on release. The lpt_packing scheduler is covered in test_lpt_packing.jl.

const GW = ReactantServerGateway

# A GatewayConfig with the given scheduling mode and two dummy endpoints (no servers are contacted;
# the lightweight-scheduler select_replicas paths read only the routing table).
function _sched_cfg(mode; basis = "compute")
    return GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", ["127.0.0.1:7001", "127.0.0.1:7002"],
                            String[], String[], 60, 1, 1, "info", "json", mode, basis, 30.0, 0.0, 0.8, 0.1, 30.0,
                            1, 1.0, "fill_rr", "run", Dict{String,GW.GatewayModelConfig}(), 32, 64, :off, 0, false)
end

# Build a ScheduleContext for `model` over a routing table mapping each model to worker URLs.
# `units` is how many items the request carries, as `request_units` would have resolved it.
function _ctx(model, table::Dict; units::Int = 1)
    cfg = _sched_cfg("round_robin")
    pool = GW.ClientPool(cfg)
    routes = GW.DiscoveredRoutes()
    GW.swap_table!(routes, GW.RoutingTable(table))
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics)
    ctx = GW.ScheduleContext(model, "id", units, pool, routes, metrics, refresher)
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

# A poll result shaped like `poll_workers`, carrying per-model cumulative (compute, requests, rows).
# `cum` maps model -> (compute_seconds, requests, rows); `batch_at` maps model -> (input, axis).
function _poll(cum::Dict; batch_at::Dict = Dict(), max_batch::Dict = Dict(), workers = ["w0"])
    sums = Dict{String,Tuple{Float64,UInt64,UInt64}}(
        m => (Float64(c), UInt64(r), UInt64(w)) for (m, (c, r, w)) in cum)
    mb = Dict{String,Int}(m => Int(get(max_batch, m, 8)) for m in keys(cum))
    ba = Dict{String,Tuple{String,Int}}(m => v for (m, v) in batch_at)
    return (; sums,
            permodel_workers = Dict{String,Vector{String}}(m => copy(workers) for m in keys(cum)),
            mem = Dict{String,Float64}(m => 0.0 for m in keys(cum)),
            mem_cap = Dict{String,Float64}(w => 0.0 for w in workers),
            max_batch = mb, batch_at = ba, polled = collect(workers),
            fleet_compute = sum(Float64(c) for (c, _, _) in values(cum); init = 0.0))
end

# A ModelInferRequest body carrying one named input with the given shape, encoded as the wire sees it.
function _body(model, name, shape)
    QPB = ReactantServerGateway.PB
    QInf = ReactantServerCore.inference
    msg = QInf.ModelInferRequest(; model_name = model,
        inputs = [QInf.var"ModelInferRequest.InferInputTensor"(;
            name = name, datatype = "UINT8", shape = Int64[shape...])])
    io = IOBuffer()
    QPB.encode(QPB.ProtoEncoder(io), msg)
    return take!(io)
end

@testset "RoutingMeta: cost per item is not cost per request" begin
    rm = GW.RoutingMeta(30.0)
    @test GW.item_cost(GW.routing_meta(rm), "m") == 1.0        # nothing polled: one unit per item

    # First poll: 10 GPU-seconds over 5 requests carrying 40 rows, i.e. 8 items per request. Cost per
    # REQUEST is 2.0s and cost per ITEM is 0.25s, and the whole point of reporting rows_served is that
    # these are not interchangeable: charging 8 items at the per-request cost overstates the work 8x.
    # The opening sample is the counters as they stand (a lifetime average, the same prior lpt_packing
    # adopts on its first repack), so one poll is enough to have a usable cost.
    snap = GW.refresh_routing_meta!(rm, _poll(Dict("m" => (10.0, 5, 40)); batch_at = Dict("m" => ("IN", 1))))
    m = GW.model_meta(snap, "m")
    @test m.cost_per_request ≈ 2.0
    @test m.cost_per_item ≈ 0.25
    @test m.batch_input == "IN" && m.batch_axis == 1
    @test GW.item_cost(snap, "m") ≈ 0.25
    @test snap.mean_cost_per_item ≈ 0.25

    # A model with no measurement of its own borrows the fleet mean, so a cold model weighs like an
    # average one instead of looking free (and attracting every request).
    @test GW.item_cost(snap, "cold") ≈ 0.25
    @test GW.model_meta(snap, "cold").batch_axis == 0          # absent, not an error

    # Second poll: +8 GPU-seconds over 8 rows, a per-item sample of 1.0. The EWMA folds toward it
    # against the 8 fleet GPU-seconds consumed this interval, so the estimate moves without jumping.
    snap = GW.refresh_routing_meta!(rm, _poll(Dict("m" => (18.0, 7, 48)); batch_at = Dict("m" => ("IN", 1))))
    @test 0.25 < GW.model_meta(snap, "m").cost_per_item < 1.0
    settled = GW.model_meta(snap, "m").cost_per_item

    # A worker restart resets its counters: the negative delta re-baselines instead of folding a
    # nonsense sample, so the last good cost stands.
    GW.refresh_routing_meta!(rm, _poll(Dict("m" => (0.5, 1, 4)); batch_at = Dict("m" => ("IN", 1))))
    @test GW.model_meta(GW.routing_meta(rm), "m").cost_per_item ≈ settled

    # A round where the worker is mid-reload and reports no batch axis keeps the known one, rather
    # than degrading the model to one item per request until the next good poll.
    snap2 = GW.refresh_routing_meta!(rm, _poll(Dict("m" => (1.0, 2, 8))))
    @test GW.model_meta(snap2, "m").batch_input == "IN"
    @test GW.model_meta(snap2, "m").batch_axis == 1
end

@testset "RoutingMeta: request_items sizes a request by the reported axis" begin
    rm = GW.RoutingMeta(30.0)
    # Batch last, as an image bundle declares it (`whcn`).
    GW.refresh_routing_meta!(rm, _poll(Dict("img" => (1.0, 1, 1)); batch_at = Dict("img" => ("IN", 4))))
    snap = GW.routing_meta(rm)
    @test GW.request_items(snap, "img", _body("img", "IN", (1024, 768, 3, 5))) == 5
    # A model with no axis, a missing input, and an unpolled model all count as one item.
    @test GW.request_items(snap, "img", _body("img", "OTHER", (5, 5))) == 1
    @test GW.request_items(snap, "ghost", _body("ghost", "IN", (5, 5))) == 1
end

@testset "LeastOutstandingScheduler: the basis decides what 'least' means" begin
    URLS = ["127.0.0.1:7001", "127.0.0.1:7002"]

    # :requests is the pre-existing behavior, kept verbatim: the body is never read, so a 32-item
    # request and a 1-item request weigh the same.
    sr = GW.LeastOutstandingScheduler(; basis = :requests)
    @test GW.request_units(sr, "m", _body("m", "IN", (32, 4))) == 1
    @test GW.needs_routing_meta(sr) == false                   # and it needs no control-plane poll

    # :items charges the batch extent the worker pointed at, so one 32-item request is 32 units of
    # work and the next request goes to the other replica.
    rm = GW.RoutingMeta(30.0)
    GW.refresh_routing_meta!(rm, _poll(Dict("m" => (1.0, 1, 1)); batch_at = Dict("m" => ("IN", 1))))
    si = GW.LeastOutstandingScheduler(rm; basis = :items)
    @test GW.needs_routing_meta(si) == true
    ctx_big, pool = _ctx("m", Dict("m" => URLS); units = 32)
    try
        @test GW.request_units(si, "m", _body("m", "IN", (32, 4))) == 32
        ubig, rbig = GW.select_replicas(si, ctx_big)
        ctx_small, _ = _ctx("m", Dict("m" => URLS); units = 1)
        usmall, rsmall = GW.select_replicas(si, ctx_small)
        @test usmall[1] != ubig[1]                             # 32 items outweighs 1
        u3, r3 = GW.select_replicas(si, ctx_small)
        @test u3[1] == usmall[1]                               # 2 items still lighter than 32
        foreach(r -> GW.release!(si, r), (rbig, rsmall, r3))
        @test all(c -> c[] == 0.0, values(@atomic si.inflight))
    finally
        GW.close_pool!(pool)
    end

    # :compute weighs items by the model's measured cost per item, the only basis that is comparable
    # across models: 8 items of a model costing 1.0s/item outweigh 32 items of one costing 0.01s/item.
    rm2 = GW.RoutingMeta(30.0)
    GW.refresh_routing_meta!(rm2, _poll(Dict("dear" => (0.0, 0, 0), "cheap" => (0.0, 0, 0));
                                       batch_at = Dict("dear" => ("IN", 1), "cheap" => ("IN", 1))))
    GW.refresh_routing_meta!(rm2, _poll(Dict("dear" => (10.0, 1, 10), "cheap" => (0.1, 1, 10));
                                       batch_at = Dict("dear" => ("IN", 1), "cheap" => ("IN", 1))))
    sc = GW.LeastOutstandingScheduler(rm2; basis = :compute)
    snap = GW.routing_meta(rm2)
    @test GW.item_cost(snap, "dear") ≈ 1.0
    @test GW.item_cost(snap, "cheap") ≈ 0.01
    ctx_dear, pool2 = _ctx("dear", Dict("dear" => URLS, "cheap" => URLS); units = 8)
    try
        udear, rdear = GW.select_replicas(sc, ctx_dear)         # charges 8.0 GPU-seconds
        ctx_cheap, _ = _ctx("cheap", Dict("dear" => URLS, "cheap" => URLS); units = 32)
        ucheap, rcheap = GW.select_replicas(sc, ctx_cheap)      # charges 0.32
        @test ucheap[1] != udear[1]                             # the expensive model's GPU is busier
        # ...and stays busier: three more cheap batches still do not add up to 8 GPU-seconds.
        held = [GW.select_replicas(sc, ctx_cheap) for _ in 1:3]
        @test all(h -> h[1][1] == ucheap[1], held)
        foreach(h -> GW.release!(sc, h[2]), held)
        foreach(r -> GW.release!(sc, r), (rdear, rcheap))
        # The captured charge is what is released, so the counters return to baseline (to within the
        # float rounding of summing and unsumming fractional GPU-seconds).
        @test all(c -> isapprox(c[], 0.0; atol = 1e-9), values(@atomic sc.inflight))
    finally
        GW.close_pool!(pool2)
    end

    @test_throws ArgumentError GW.LeastOutstandingScheduler(; basis = :gpu_seconds)
end

@testset "gateway_worker_inflight_work exports what the scheduler compares" begin
    _scrape(m) = (io = IOBuffer(); GW.expose(io, m); String(take!(io)))

    # least_outstanding: the per-worker load, labelled with the basis, because the unit depends on it.
    rm = GW.RoutingMeta(30.0)
    GW.refresh_routing_meta!(rm, _poll(Dict("m" => (4.0, 1, 8)); batch_at = Dict("m" => ("IN", 1))))
    s = GW.LeastOutstandingScheduler(rm; basis = :compute)
    metrics = GW.GatewayMetrics(Dict("127.0.0.1:7001" => "worker0", "127.0.0.1:7002" => "worker1"))
    ctx, pool = _ctx("m", Dict("m" => ["127.0.0.1:7001", "127.0.0.1:7002"]); units = 4)
    try
        # Nothing routed yet: the family is registered but has no children, so no series is emitted
        # (an empty family would read as "zero work" rather than "no counters").
        GW.scheduler_start!(s, pool, metrics)
        @test !occursin("gateway_worker_inflight_work", _scrape(metrics))

        _, res = GW.select_replicas(s, ctx)                  # 4 items x 0.5 GPU-seconds per item
        body = _scrape(metrics)
        @test occursin("gateway_worker_inflight_work{worker=\"worker0\",basis=\"compute\"} 2\n", body)
        # A candidate that was not chosen still reports 0, which is what distinguishes an idle worker
        # the scheduler knows about from one it has never seen.
        @test occursin("gateway_worker_inflight_work{worker=\"worker1\",basis=\"compute\"} 0\n", body)
        @test occursin("# TYPE gateway_worker_inflight_work gauge", body)
        # Read at scrape time from the live counters, so a release is visible on the next scrape with
        # no publish step in between.
        GW.release!(s, res)
        @test occursin("gateway_worker_inflight_work{worker=\"worker0\",basis=\"compute\"} 0\n",
                       _scrape(metrics))
    finally
        GW.close_pool!(pool)
    end

    # The basis rides along as a label, so a fleet on items reports items under its own series.
    si = GW.LeastOutstandingScheduler(rm; basis = :items)
    mi = GW.GatewayMetrics(Dict("127.0.0.1:7001" => "worker0"))
    ctxi, pooli = _ctx("m", Dict("m" => ["127.0.0.1:7001"]); units = 7)
    try
        GW.scheduler_start!(si, pooli, mi)
        GW.select_replicas(si, ctxi)
        @test occursin("gateway_worker_inflight_work{worker=\"worker0\",basis=\"items\"} 7\n", _scrape(mi))
    finally
        GW.close_pool!(pooli)
    end

    # round_robin tracks no work, so it registers no collector rather than exporting an empty family.
    mr = GW.GatewayMetrics()
    rr = GW.RoundRobinScheduler()
    @test GW.inflight_work(rr) === nothing
    @test GW.register_worker_work!(rr, mr) === nothing
    @test !occursin("gateway_worker_inflight_work", _scrape(mr))
end

@testset "LeastOutstandingScheduler: routes to the least in-flight replica" begin
    s = GW.LeastOutstandingScheduler(; basis = :requests)
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
