# LPT-packing scheduling: the pure assignment math (concentration, balance, split, hysteresis),
# weighted sampling, and the gateway integration (startup preconditions, concentration after a
# rebalance, metrics) against mock workers that serve both the inference and control services.

import gRPCServer
import gRPCClient
import HTTP
using ReactantServerCore.control   # bare message types for the included server stubs

const AInf = ReactantServerCore.inference
const ACtl = ReactantServerCore.control

# Server-side control stubs for the mock workers (the gateway module ships only the client side).
include(ReactantServerCore.control_server_stubs_path())

const GW = ReactantServerGateway
const NOPREV = Dict{String,GW.Placement}()

@testset "control proto: max_batch_size round-trips" begin
    ms = ACtl.ModelStatus(; name = "m", max_batch_size = Int64(32))
    io = IOBuffer()
    GW.PB.encode(GW.PB.ProtoEncoder(io), ms)
    seekstart(io)
    got = GW.PB.decode(GW.PB.ProtoDecoder(io), ACtl.ModelStatus)
    @test got.name == "m"
    @test got.max_batch_size == 32
end

@testset "compute_assignment: concentration and LPT balance" begin
    W = ["w0", "w1"]
    asn = GW.compute_assignment(Dict("a" => 0.5, "b" => 0.4, "c" => 0.1), W, NOPREV)
    # every model on exactly one worker with weight 1
    for (m, pl) in asn
        @test length(pl) == 1
        @test pl[1][2] == 1.0
    end
    # LPT: a (0.5) -> w0; b (0.4) -> w1; c (0.1) -> the lighter w1 (0.4 < 0.5)
    @test asn["a"] == [("w0", 1.0)]
    @test asn["b"] == [("w1", 1.0)]
    @test asn["c"] == [("w1", 1.0)]
    # cold models (absent from u) are not placed
    @test !haskey(asn, "ghost")
end

@testset "compute_assignment: configured replicas on distinct GPUs" begin
    W = ["w0", "w1", "w2"]
    # Default is one GPU regardless of load: a hot model does not fan out on its own.
    one = GW.compute_assignment(Dict("hot" => 1.5), W, NOPREV)
    @test length(one["hot"]) == 1
    @test one["hot"][1][2] == 1.0

    # replicas = 2 places the model on two distinct workers with even weights summing to 1.
    two = GW.compute_assignment(Dict("m" => 0.9), W, NOPREV; replicas = Dict("m" => 2))
    @test length(two["m"]) == 2
    @test allunique(first.(two["m"]))                # no worker hosts the model twice
    @test sum(last.(two["m"])) ≈ 1.0
    @test all(==(0.5), last.(two["m"]))

    # replicas clamps to the worker count; default_replicas applies to unlisted models.
    @test length(GW.compute_assignment(Dict("m" => 0.9), W, NOPREV; replicas = Dict("m" => 9))["m"]) == 3
    @test length(GW.compute_assignment(Dict("m" => 0.9), W, NOPREV; default_replicas = 2)["m"]) == 2

    # default_replicas = all places every model on every worker (clamped to the worker count).
    everywhere = GW.compute_assignment(Dict("a" => 0.5, "b" => 0.1, "c" => 0.0), W, NOPREV;
                                       default_replicas = GW.REPLICAS_ALL)
    @test all(m -> length(everywhere[m]) == 3, keys(everywhere))
end

@testset "config: replica counts accept a positive integer or 'all'" begin
    function _load(yaml)
        path = tempname() * ".yaml"
        write(path, yaml)
        try
            return GW.load_gateway(path)
        finally
            rm(path; force = true)
        end
    end
    eps = "endpoints:\n  - \"127.0.0.1:7001\"\n"
    @test _load("scheduling:\n  default_replicas: all\n" * eps).default_replicas == GW.REPLICAS_ALL
    @test _load("scheduling:\n  default_replicas: 3\n" * eps).default_replicas == 3
    @test _load("scheduling:\n  models:\n    big:\n      replicas: all\n" * eps).models["big"].replicas == GW.REPLICAS_ALL
    @test_throws ReactantServerCore.ConfigError _load("scheduling:\n  default_replicas: 0\n" * eps)
    @test_throws ReactantServerCore.ConfigError _load("scheduling:\n  default_replicas: huge\n" * eps)
end

@testset "compute_assignment: memory dimension steers placement" begin
    W = ["w0", "w1"]
    GB = 1.0e9
    u = Dict("a" => 0.6, "b" => 0.5, "c" => 0.1)
    mem = Dict("a" => 1GB, "b" => 9GB, "c" => 9GB)
    caps = Dict("w0" => 10GB, "w1" => 10GB)
    # Compute-only packing co-locates c with b on the less compute-loaded w1.
    plain = GW.compute_assignment(u, W, NOPREV)
    @test plain["c"] == [("w1", 1.0)]
    # With memory in play, b's 9 GB fills w1; c's 9 GB no longer fits there and moves to w0
    # even though w0 carries more compute. Eviction churn avoided, no GPU idle.
    packed = GW.compute_assignment(u, W, NOPREV; mem = mem, mem_cap = caps)
    @test packed["a"] == [("w0", 1.0)]
    @test packed["b"] == [("w1", 1.0)]
    @test packed["c"] == [("w0", 1.0)]

    # Cold models (u = 0) still occupy memory and get concentrated homes spread by footprint.
    cold = GW.compute_assignment(Dict("c1" => 0.0, "c2" => 0.0), W, NOPREV;
                                 mem = Dict("c1" => 6GB, "c2" => 6GB), mem_cap = caps)
    @test length(cold["c1"]) == 1 && length(cold["c2"]) == 1
    @test cold["c1"][1][2] == 1.0 && cold["c2"][1][2] == 1.0
    @test cold["c1"][1][1] != cold["c2"][1][1]    # one per worker: 12 GB would overflow one budget

    # A worker with cap 0 (on-demand cache disabled) is memory-unconstrained.
    free = GW.compute_assignment(u, W, NOPREV; mem = mem,
                                 mem_cap = Dict("w0" => 0.0, "w1" => 0.0))
    @test free["c"] == [("w1", 1.0)]              # back to pure compute balance

    # Abundant memory degrades gracefully to compute-only LPT: when every model fits every
    # budget with room to spare, the max-norm is dominated by compute pressure and the placement
    # matches the no-memory packing exactly.
    roomy = GW.compute_assignment(u, W, NOPREV; mem = mem,
                                  mem_cap = Dict("w0" => 1000GB, "w1" => 1000GB))
    @test roomy == plain
end

@testset "compute_assignment: hysteresis keeps placements stable" begin
    W = ["w0", "w1"]
    prev = Dict{String,GW.Placement}("a" => [("w1", 1.0)])
    # small imbalance: a stays on its previous worker even though w0 is nominally least loaded
    asn = GW.compute_assignment(Dict("a" => 0.5), W, prev; hysteresis = 0.1)
    @test asn["a"] == [("w1", 1.0)]
    # large imbalance: a model stuck behind a hot one moves
    prev2 = Dict{String,GW.Placement}("hot" => [("w1", 1.0)], "a" => [("w1", 1.0)])
    asn2 = GW.compute_assignment(Dict("hot" => 0.7, "a" => 0.3), W, prev2; hysteresis = 0.1)
    @test asn2["hot"] == [("w1", 1.0)]               # sticky
    @test asn2["a"] == [("w0", 1.0)]                 # moved off the hot worker
    # a previous worker that no longer exists is ignored
    prev3 = Dict{String,GW.Placement}("a" => [("gone", 1.0)])
    asn3 = GW.compute_assignment(Dict("a" => 0.5), W, prev3)
    @test asn3["a"][1][1] in W
end

# Build a packing state directly for routing unit tests, with a two-replica placement installed.
function _pk_state(; routing_policy = "fill", fill_factor = 1.0, max_batch = 8)
    cfg = GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", String[], String[], 60, 1, 1, "info", "json",
                           "lpt_packing", 30.0, 0.0, 0.8, 0.1, 30.0, 1, fill_factor, routing_policy,
                           Dict{String,GW.GatewayModelConfig}())
    s = GW.LptPackingState(cfg)
    @atomic s.assignment = Dict{String,GW.Placement}("m" => [("w0", 0.5), ("w1", 0.5)])
    @atomic s.max_batch = Dict("m" => max_batch)
    GW._swap_outstanding!(s, (@atomic s.assignment))
    return s
end

@testset "route_replica: fill one replica before the next" begin
    s = _pk_state(; max_batch = 8)
    firsts = [GW.route_replica(s, "m")[1][1] for _ in 1:8]   # hold all 8 in flight
    @test all(==(firsts[1]), firsts)                          # first batch fills one replica
    ninth, _ = GW.route_replica(s, "m")
    @test ninth[1] != firsts[1]                               # then spill to the other
    @test Set(ninth) == Set(["w0", "w1"])                     # both present as failover

    # fill_factor over-provisions the per-replica target (1.5 * 8 = 12).
    s2 = _pk_state(; fill_factor = 1.5, max_batch = 8)
    f2 = [GW.route_replica(s2, "m")[1][1] for _ in 1:12]
    @test all(==(f2[1]), f2)
    @test GW.route_replica(s2, "m")[1][1] != f2[1]

    # single-replica fast path: no counters; cold model falls back to round robin.
    @atomic s.assignment = Dict{String,GW.Placement}("solo" => [("w0", 1.0)])
    GW._swap_outstanding!(s, (@atomic s.assignment))
    urls, counters = GW.route_replica(s, "solo")
    @test urls == ["w0"] && counters === nothing
    @test GW.route_replica(s, "unknown") === nothing
end

@testset "route_replica: release frees the counter on every path" begin
    s = _pk_state(; max_batch = 4)
    held = [GW.route_replica(s, "m")[2] for _ in 1:6]
    foreach(GW._release_route!, held)
    out = @atomic s.outstanding
    @test out[("m", "w0")][] == 0 && out[("m", "w1")][] == 0
end

@testset "route_replica: least_outstanding spreads by total in-flight" begin
    s = _pk_state(; routing_policy = "least_outstanding", max_batch = 8)
    # With equal totals, the first pick is deterministic; the second goes to the other replica
    # because the first reservation raised that worker's total.
    a, _ = GW.route_replica(s, "m")
    b, _ = GW.route_replica(s, "m")
    @test a[1] != b[1]
end

# --- Integration: mock workers serving inference + control ------------------------------------

mutable struct AffMockWorker
    name::String
    models::Vector{String}
    discipline::String
    served::Dict{String,Int}
    compute::Dict{String,Float64}
end
AffMockWorker(name, models; discipline = "fifo") =
    AffMockWorker(name, models, discipline, Dict{String,Int}(), Dict{String,Float64}())

function _aff_router()
    router = gRPCServer.gRPCRouter()
    GW.register_GRPCInferenceService!(router;
        ServerReady = (req, c) -> AInf.ServerReadyResponse(; ready = true),
        RepositoryIndex = (req, c) -> AInf.RepositoryIndexResponse(; models = [
            AInf.var"RepositoryIndexResponse.ModelIndex"(; name = m, version = "", state = "READY", reason = "")
            for m in c.payload.models]),
        ModelInfer = (req, c) -> begin
            w = c.payload
            w.served[req.model_name] = get(w.served, req.model_name, 0) + 1
            w.compute[req.model_name] = get(w.compute, req.model_name, 0.0) + 0.05
            AInf.ModelInferResponse(; model_name = w.name, id = req.id)
        end,
    )
    register_ControlService!(router;
        ModelControlStatus = (req, c) -> begin
            w = c.payload
            ACtl.ModelControlStatusResponse(;
                residency_mode = "self_managed", discipline = w.discipline,
                models = [ACtl.ModelStatus(; name = m,
                              weight_nbytes = Int64(256 * 1024 * 1024),
                              total_compute_seconds = get(w.compute, m, 0.0),
                              requests_served = UInt64(get(w.served, m, 0)),
                              dispatch_count = UInt64(get(w.served, m, 0)),
                              max_batch_size = Int64(8))
                          for m in w.models],
                weight_cache_max_bytes = UInt64(8) * 1024^3)
        end,
    )
    return router
end

_aff_infer(port, model) = grpc_call(AInf.ModelInferRequest, AInf.ModelInferResponse, "ModelInfer",
    port, AInf.ModelInferRequest(; model_name = model))

function _aff_gatewayfile(gw_port, admin_port, worker_ports)
    path = tempname() * ".yaml"
    eps = join(("  - \"127.0.0.1:$p\"" for p in worker_ports), "\n")
    write(path, """
    listen:
      grpc: "127.0.0.1:$gw_port"
      metrics: "127.0.0.1:$admin_port"
    scheduling:
      mode: lpt_packing
      rebalance_compute_seconds: 0.001
    endpoints:
    $eps
    """)
    return path
end

@testset "lpt_packing gateway: preconditions and concentration" begin
    models = ["alpha", "beta"]
    w0 = AffMockWorker("worker0", copy(models))
    w1 = AffMockWorker("worker1", copy(models))
    p0, p1 = grpc_free_port(), grpc_free_port()
    s0 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p0; context = w0)
    s1 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p1; context = w1)

    gw_port, admin_port = grpc_free_port(), grpc_free_port()
    gatewayfile = _aff_gatewayfile(gw_port, admin_port, [p0, p1])
    gw = GW.serve_gateway(gatewayfile; blocking = false)
    try
        # wait for routing
        routed = false
        for _ in 1:40
            try
                _aff_infer(gw_port, "alpha")
                routed = true
                break
            catch
                sleep(0.1)
            end
        end
        @test routed

        # Drive traffic so the gateway accumulates arrival rate and the mocks accumulate served
        # compute, then force a rebalance (deterministic; the prober would do this on its tick).
        for _ in 1:30
            _aff_infer(gw_port, "alpha")
            _aff_infer(gw_port, "beta")
        end
        sleep(1.1)   # ensure dt since the startup baseline rebalance is meaningful
        aff = gw.prober.packing
        @test aff !== nothing
        GW.rebalance!(aff, gw.pool, copy(gw.pool.order), gw.metrics)

        # Both models now have a single-worker placement (default replicas = 1): every route for a
        # model returns the same worker.
        for m in models
            routed = GW.route_replica(aff, m)
            @test routed !== nothing
            urls = routed[1]
            @test count(u -> u == urls[1], [GW.route_replica(aff, m)[1][1] for _ in 1:10]) == 10
        end

        # Concentration end to end: further traffic for alpha lands on one worker only.
        base0 = get(w0.served, "alpha", 0)
        base1 = get(w1.served, "alpha", 0)
        for _ in 1:20
            _aff_infer(gw_port, "alpha")
        end
        d0 = get(w0.served, "alpha", 0) - base0
        d1 = get(w1.served, "alpha", 0) - base1
        @test d0 + d1 == 20
        @test max(d0, d1) == 20                       # all on the placed worker

        # Compute-driven trigger: tick_packing! accumulates fleet compute and repacks only once the
        # budget is crossed.
        aff.rebalance_compute_seconds = 1.0e9      # effectively never
        before = aff.last_rebalance
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance == before          # not enough compute -> no repack
        @test aff.compute_accum > 0                 # but the compute was accounted

        aff.rebalance_compute_seconds = 1.0e-9      # any compute triggers
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance > before           # repacked
        @test aff.compute_accum == 0.0              # accumulator reset on repack

        # Placement is observable in the metrics.
        body = String(HTTP.get("http://127.0.0.1:$admin_port/metrics"; retry = false).body)
        @test occursin("gateway_placement_weight", body)
        @test occursin("gateway_model_utilization", body)
        @test occursin("gateway_model_replicas", body)
    finally
        GW.stop!(gw)
        close(s0)
        close(s1)
        rm(gatewayfile; force = true)
    end
end

@testset "lpt_packing gateway: startup hard-fail" begin
    models = ["alpha", "beta"]
    gw_port, admin_port = grpc_free_port(), grpc_free_port()

    # A worker reporting fair discipline is rejected.
    wfair = AffMockWorker("worker0", copy(models); discipline = "fair")
    pf = grpc_free_port()
    sf = gRPCServer.serve!(_aff_router(), "127.0.0.1", pf; context = wfair)
    f1 = _aff_gatewayfile(gw_port, admin_port, [pf])
    @test_throws ErrorException GW.serve_gateway(f1; blocking = false)
    close(sf)

    # Differing model sets are rejected.
    wa = AffMockWorker("worker0", ["alpha", "beta"])
    wb = AffMockWorker("worker1", ["alpha"])
    pa, pb = grpc_free_port(), grpc_free_port()
    sa = gRPCServer.serve!(_aff_router(), "127.0.0.1", pa; context = wa)
    sb = gRPCServer.serve!(_aff_router(), "127.0.0.1", pb; context = wb)
    f2 = _aff_gatewayfile(gw_port, admin_port, [pa, pb])
    @test_throws ErrorException GW.serve_gateway(f2; blocking = false)
    close(sa); close(sb)

    # An unreachable worker is rejected.
    f3 = _aff_gatewayfile(gw_port, admin_port, [grpc_free_port()])
    @test_throws ErrorException GW.serve_gateway(f3; blocking = false)

    rm(f1; force = true); rm(f2; force = true); rm(f3; force = true)
end
