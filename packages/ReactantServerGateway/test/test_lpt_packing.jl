# LPT-packing scheduling: the pure assignment math (concentration, balance, split, hysteresis),
# weighted sampling, and the gateway integration (startup preconditions, concentration after a
# rebalance, metrics) against mock workers that serve both the inference and control services.

import gRPCServer
import gRPCClient
import HTTP
import Sockets
import Logging
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

@testset "config: routing_policy accepts only the fill variants" begin
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
    _pol(p) = _load("scheduling:\n  routing_policy: $p\n" * eps).routing_policy
    _mode(m) = _load("scheduling:\n  mode: $m\n" * eps).scheduling_mode
    @test _pol("fill_rr") == "fill_rr"
    @test _pol("fill_least") == "fill_least"
    @test _load(eps).routing_policy == "fill_least"                   # default
    # least_outstanding is now a top-level scheduling mode, not a routing policy.
    @test _mode("least_outstanding") == "least_outstanding"
    @test_throws ReactantServerCore.ConfigError _pol("least_outstanding")
    # The old 'fill' alias is removed; it is now just an invalid value.
    @test_throws ReactantServerCore.ConfigError _pol("fill")
    @test_throws ReactantServerCore.ConfigError _pol("bogus")
    @test_throws ReactantServerCore.ConfigError _mode("bogus")

    # What both least_outstanding and fill_least call "busy". Defaults to compute: in-flight GPU-seconds.
    _basis(b) = _load("scheduling:\n  work_basis: $b\n" * eps).work_basis
    @test _load(eps).work_basis == "compute"                           # default
    @test _basis("items") == "items"
    @test _basis("requests") == "requests"
    @test _basis("COMPUTE") == "compute"                               # case-insensitive, like the rest
    @test_throws ReactantServerCore.ConfigError _basis("gpu_seconds")
end

@testset "verify_lpt_packing_preconditions!: gates on worker reachability" begin
    cfg = GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", ["127.0.0.1:1"], String[], String[], 1, 1, 1, "info",
                           "json", "lpt_packing", "compute", 30.0, 0.0, 0.8, 0.1, 30.0, 1, 1.0, "fill_rr", "run",
                           Dict{String,GW.GatewayModelConfig}(), 32, 64, :off, 0, false)
    pool = GW.ClientPool(cfg)
    # Default (wait_seconds = 0) fails fast when a worker is unreachable.
    @test_throws ErrorException GW.verify_lpt_packing_preconditions!(pool; wait_seconds = 0)
    # A bounded wait polls, then still errors if the worker never comes up.
    t0 = time()
    @test_throws ErrorException GW.verify_lpt_packing_preconditions!(pool; wait_seconds = 0.3,
                                                                     poll_interval = 0.05)
    @test time() - t0 >= 0.25
end

@testset "_bounded: timeout log level is controllable (quiet startup wait)" begin
    slow() = (sleep(5); 1)   # never finishes within the watchdog, forcing the timeout branch
    # Default :warn (the runtime health prober) warns on a timeout.
    @test_logs (:warn, r"timed out; treating as unavailable") match_mode = :any begin
        v, to = GW._bounded(slow, 0.05, nothing, "ModelControlStatus poll", "127.0.0.1:0")
        @test v === nothing && to === true
    end
    # :debug (the startup precondition wait) stays quiet: no record at Warn or above.
    @test_logs min_level = Logging.Warn begin
        v, to = GW._bounded(slow, 0.05, nothing, "ModelControlStatus poll", "127.0.0.1:0"; level = :debug)
        @test v === nothing && to === true
    end
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

@testset "EMA decay: gamma tracks compute elapsed against the halflife" begin
    # _ewma(old, sample, dt, h): one halflife of elapsed compute folds ~50% toward the sample, so a
    # halflife equal to the rebalance interval gives about 50% decay per rebalance.
    @test GW._ewma(0.0, 1.0, 300.0, 300.0) ≈ 0.5
    @test GW._ewma(2.0, 2.0, 300.0, 300.0) ≈ 2.0                 # steady state holds
    # A fraction of a halflife (the early first_rebalance budget, 60 of 300) folds gently.
    @test GW._ewma(0.0, 1.0, 60.0, 300.0) ≈ 1 - 2.0^(-0.2)
    # Zero elapsed compute (the startup rebalance) is a no-op: the EWMA does not move.
    @test GW._ewma(0.7, 5.0, 0.0, 300.0) ≈ 0.7
end

@testset "config: lpt_packing defaults and compute-clock EMA resolution" begin
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
    cfg = _load(eps)
    @test cfg.rebalance_compute_seconds == 300.0
    @test cfg.first_rebalance_compute_seconds == 60.0
    @test cfg.hysteresis == 0.0
    @test cfg.ema_halflife_compute_seconds == 0.0
    @test cfg.compaction_mode == :eager
    @test cfg.compaction_interval == 1
    @test cfg.forbid_memory_oversubscription == true
    # 0 resolves to the rebalance interval (~50% decay per rebalance); a set value is honored.
    @test GW.knobs(GW.LptPackingState(cfg)).ema_halflife_compute == cfg.rebalance_compute_seconds
    cfg2 = _load("scheduling:\n  ema_halflife_compute_seconds: 120\n  rebalance_compute_seconds: 300\n" * eps)
    @test GW.knobs(GW.LptPackingState(cfg2)).ema_halflife_compute == 120.0
    @test _load("scheduling:\n  forbid_memory_oversubscription: false\n" * eps).forbid_memory_oversubscription == false
end

@testset "forbid_memory_oversubscription: vacates an over-budget home, falls back when infeasible" begin
    W = ["w0", "w1"]
    GB = 1.0e9
    caps = Dict("w0" => 10GB, "w1" => 10GB)
    # cpu1 saturates w1's compute; occ0 + m together exceed w0's budget. Hysteresis (unconstrained)
    # pins m on w0 because moving it to the compute-busy w1 barely improves its max-norm pressure.
    prev = Dict{String,GW.Placement}("m" => [("w0", 1.0)], "occ0" => [("w0", 1.0)],
                                     "cpu1" => [("w1", 1.0)])
    u = Dict("cpu1" => 1.0, "occ0" => 0.0, "m" => 0.0)
    mem = Dict("cpu1" => 1GB, "occ0" => 7GB, "m" => 3.5GB)
    soft = GW.compute_assignment(u, W, prev; mem = mem, mem_cap = caps, hysteresis = 0.1)
    @test soft["m"] == [("w0", 1.0)]                            # pinned: w0 now holds 10.5 GB > 10
    hard = GW.compute_assignment(u, W, prev; mem = mem, mem_cap = caps, hysteresis = 0.1,
                                 forbid_memory_oversubscription = true)
    @test hard["m"] == [("w1", 1.0)]                            # forced onto the feasible worker
    for w in W                                                  # no worker exceeds its budget
        load = sum((get(mem, mm, 0.0) for (mm, pl) in hard for (ww, _) in pl if ww == w); init = 0.0)
        @test load <= caps[w]
    end

    # Roomy memory: the guarantee is a no-op (every worker fits), matching the unconstrained packing.
    u2 = Dict("a" => 0.6, "b" => 0.5, "c" => 0.1)
    mem2 = Dict("a" => 1GB, "b" => 1GB, "c" => 1GB)
    big = Dict("w0" => 1000GB, "w1" => 1000GB)
    @test GW.compute_assignment(u2, W, NOPREV; mem = mem2, mem_cap = big,
                                forbid_memory_oversubscription = true) ==
          GW.compute_assignment(u2, W, NOPREV; mem = mem2, mem_cap = big)

    # Genuinely infeasible (a model larger than any budget) is still placed, not dropped.
    huge = GW.compute_assignment(Dict("x" => 0.5), W, NOPREV;
                                 mem = Dict("x" => 20GB), mem_cap = caps,
                                 forbid_memory_oversubscription = true)
    @test length(huge["x"]) == 1 && huge["x"][1][1] in W
end

@testset "gateway compaction cadence: fires on the Nth repack that moves a model" begin
    mk(mode, interval) = GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", String[], String[], String[], 60, 1, 1,
        "info", "json", "lpt_packing", "compute", 30.0, 0.0, 0.8, 0.1, 30.0, 1, 1.0, "fill_rr", "run",
        Dict{String,GW.GatewayModelConfig}(), 32, 64, mode, interval, false)
    cfg = mk(:eager, 2)
    s = GW.LptPackingState(cfg)
    pool = GW.ClientPool(cfg)               # no workers; the ghost URL below is skipped (no network)
    moved = Set(["ghost:1"])
    none = Set{String}()

    GW._maybe_compact_fleet!(s, pool, nothing, moved)
    @test s.repacks_since_compact == 1      # below the interval: counts, no fan-out
    GW._maybe_compact_fleet!(s, pool, nothing, moved)
    @test s.repacks_since_compact == 0      # reached the interval with a move: fired and reset

    # A no-move repack still counts but cannot fire, so the trigger can land later than exactly N.
    GW._maybe_compact_fleet!(s, pool, nothing, none)
    @test s.repacks_since_compact == 1
    GW._maybe_compact_fleet!(s, pool, nothing, none)
    @test s.repacks_since_compact == 2      # at/over the interval but nothing moved: still waiting
    GW._maybe_compact_fleet!(s, pool, nothing, moved)
    @test s.repacks_since_compact == 0      # first move after the interval fires and resets

    # mode :off never counts or fires.
    s_off = GW.LptPackingState(mk(:off, 2))
    GW._maybe_compact_fleet!(s_off, GW.ClientPool(mk(:off, 2)), nothing, moved)
    @test s_off.repacks_since_compact == 0
end

# Build a packing state directly for routing unit tests. Defaults to a single two-replica model
# "m" on w0/w1; callers can install their own placement, per-model costs, and max batches.
#
# `costs` are per-REQUEST costs (what the `requests` work basis charges, published through
# `cost_snapshot`); `item_costs` are per-ITEM costs (what `compute` charges, published through the
# shared routing-metadata cache, which the prober would normally fill from a poll).
function _pk_state(; routing_policy = "fill_rr", fill_factor = 1.0, max_batch = 8,
                   fill_mode = "run", batch_at = nothing, work_basis = "compute",
                   assignment = Dict{String,GW.Placement}("m" => [("w0", 0.5), ("w1", 0.5)]),
                   costs = nothing, item_costs = nothing)
    cfg = GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", String[], String[], String[], 60, 1, 1, "info", "json",
                           "lpt_packing", work_basis, 30.0, 0.0, 0.8, 0.1, 30.0, 1, fill_factor, routing_policy, fill_mode,
                           Dict{String,GW.GatewayModelConfig}(), 32, 64, :off, 0, false)
    meta = GW.RoutingMeta(30.0)
    if item_costs !== nothing
        bi, ax = batch_at === nothing ? ("", 0) : batch_at
        models = Dict(m => GW.ModelRoutingMeta(max_batch, bi, ax, 0.0, Float64(c))
                      for (m, c) in item_costs)
        @atomic meta.snapshot = GW.RoutingMetaSnapshot(models,
            sum(Float64(c) for c in values(item_costs)) / length(item_costs))
    end
    s = GW.LptPackingState(cfg, meta)
    @atomic s.assignment = assignment
    mb = Dict(m => max_batch for m in keys(assignment))
    @atomic s.max_batch = mb
    at = batch_at === nothing ? Dict{String,Tuple{String,Int}}() :
         Dict(m => batch_at for m in keys(assignment))
    GW._publish_fill_plan!(s, GW.knobs(s), mb, at)   # resolve mode + quantum, as a prober tick would
    costs === nothing || (@atomic s.cost_snapshot = costs)
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

    # single-replica fast path: still reserves (so its load is visible to fill_least), but routes
    # to its sole worker. A cold/unknown model falls back to round robin (nothing).
    s3 = _pk_state(; assignment = Dict{String,GW.Placement}("solo" => [("w0", 1.0)]))
    urls, counters = GW.route_replica(s3, "solo")
    @test urls == ["w0"] && counters !== nothing
    @test (@atomic s3.worker_load)["w0"][] > 0                # the single-replica request loads w0
    GW._release_route!(counters)
    @test (@atomic s3.worker_load)["w0"][] == 0
    @test GW.route_replica(s3, "unknown") === nothing
end

@testset "route_replica: fill_rr rotates which replica opens each batch" begin
    s = _pk_state(; routing_policy = "fill_rr", max_batch = 8)
    # Each batch start (the model idle) opens on the next replica in rotation.
    picks = String[]
    for _ in 1:4
        urls, c = GW.route_replica(s, "m")
        push!(picks, urls[1])
        GW._release_route!(c)                                 # complete it: idle again for next start
    end
    @test picks == ["w0", "w1", "w0", "w1"]

    # Mid-fill never rotates: requests held in flight concentrate on the one open replica.
    held = [GW.route_replica(s, "m") for _ in 1:5]
    @test all(==(held[1][1][1]), [h[1][1] for h in held])
    foreach(h -> GW._release_route!(h[2]), held)
end

@testset "route_replica: fill_least opens on the least loaded GPU (requests basis)" begin
    # The pre-basis behavior, pinned by `work_basis: requests`: the load is the model's measured cost
    # per REQUEST, whatever the request's size.
    #
    # m is replicated on w0/w1; an expensive single-replica model "hot" lives on w0. Routing hot
    # loads w0 (single-replica load counts), so m's next batch opens on the idle w1 even though both
    # replicas hold zero of m's own requests.
    s = _pk_state(; routing_policy = "fill_least", work_basis = "requests",
                  assignment = Dict{String,GW.Placement}("m" => [("w0", 0.5), ("w1", 0.5)],
                                                         "hot" => [("w0", 1.0)]),
                  costs = Dict("hot" => 10.0, "m" => 1.0))
    _, hc = GW.route_replica(s, "hot")
    @test (@atomic s.worker_load)["w0"][] == 10.0             # hot's measured cost weights the load
    @test GW.route_replica(s, "m")[1][1] == "w1"             # m avoids the busy w0
    # Blind to batch size, which is the whole reason the other bases exist: a 32-item request of m
    # charges exactly what a 1-item request charges.
    _, big = GW.route_replica(s, "m", 32)
    @test (@atomic s.worker_load)["w1"][] == 2.0             # 1.0 + 1.0, not 1.0 + 32.0

    # With both GPUs equally loaded the choice falls back to the deterministic URL tiebreak.
    s2 = _pk_state(; routing_policy = "fill_least", work_basis = "requests")
    @test GW.route_replica(s2, "m")[1][1] == "w0"
end

@testset "route_replica: fill_least weighs items by cost per item (compute basis)" begin
    # The default basis. Same fleet, same traffic, three bases, three different answers about which
    # GPU is busier. "dear" costs 1.0 GPU-second per item and lives alone on w0; "m" is replicated on
    # w0/w1 and costs 0.01 per item.
    _fleet(basis) = _pk_state(; routing_policy = "fill_least", work_basis = basis, max_batch = 32,
                              batch_at = ("IN", 1),
                              assignment = Dict{String,GW.Placement}("m" => [("w0", 0.5), ("w1", 0.5)],
                                                                     "dear" => [("w0", 1.0)]),
                              item_costs = Dict("dear" => 1.0, "m" => 0.01),
                              costs = Dict("dear" => 8.0, "m" => 0.32))

    # compute: 8 items of dear is 8 GPU-seconds; 32 items of m is 0.32. dear's GPU is busier, which
    # is the truth, so m's next run opens on the other one.
    s = _fleet("compute")
    GW.route_replica(s, "dear", 8)
    m1 = GW.route_replica(s, "m", 32)
    @test m1[1][1] == "w1"
    load = @atomic s.worker_load
    @test load["w0"][] ≈ 8.0                                 # items x cost per item, not one request
    @test load["w1"][] ≈ 0.32

    # items: the same traffic says the opposite, because 32 cheap items outweigh 8 expensive ones.
    # This is the wart of the items basis for `fill_least`, which compares across models.
    si = _fleet("items")
    GW.route_replica(si, "dear", 8)
    GW.route_replica(si, "m", 32)
    iload = @atomic si.worker_load
    @test iload["w0"][] == 8.0
    @test iload["w1"][] == 32.0

    # requests: both charge one measured per-request cost, blind to the sizes actually sent.
    sr = _fleet("requests")
    GW.route_replica(sr, "dear", 8)
    GW.route_replica(sr, "m", 32)
    rload = @atomic sr.worker_load
    @test rload["w0"][] ≈ 8.0
    @test rload["w1"][] ≈ 0.32

    # Releasing subtracts the captured charge exactly, so a cost the prober refolds mid-flight cannot
    # leave the load drifting.
    GW._release_route!(m1[2])
    @test (@atomic s.worker_load)["w1"][] == 0.0

    # A model with no measured cost of its own borrows the fleet mean rather than looking free.
    s2 = _pk_state(; routing_policy = "fill_least", batch_at = ("IN", 1),
                   assignment = Dict{String,GW.Placement}("m" => [("w0", 0.5), ("w1", 0.5)],
                                                          "cold" => [("w0", 1.0)]),
                   item_costs = Dict("m" => 2.0))
    GW.route_replica(s2, "cold", 3)
    @test (@atomic s2.worker_load)["w0"][] ≈ 6.0             # 3 items x the 2.0 fleet mean
end

@testset "fill_least: a drained fleet still rotates, despite float residue" begin
    # The netai02 regression. Bursty traffic drains the fleet between bursts, so each burst opens from
    # what should be a four-way tie that the rotation cursor breaks. Adding and subtracting fractional
    # GPU-second charges in different orders leaves a couple of ulp behind, and comparing loads raw
    # turned every one of those ties into a strict inequality: the workers that happened to hold
    # exactly 0.0 won every burst, and a 4-replica model's work came out 1.9x skewed across its GPUs.
    #
    # The cost and shape here are netai02's: ~0.00326 GPU-seconds per item, 31-item requests, quantum
    # 32 items, 8 requests in flight per burst.
    FOUR = Dict{String,GW.Placement}("m" => [("w$i", 0.25) for i in 0:3])
    s = _pk_state(; routing_policy = "fill_least", max_batch = 32, batch_at = ("IN", 1),
                  assignment = FOUR, item_costs = Dict("m" => 0.00326))
    openers = String[]
    for _ in 1:16
        held = [GW.route_replica(s, "m", 31) for _ in 1:8]
        push!(openers, held[1][1][1])
        foreach(h -> GW._release_route!(h[2]), held)
        # Drained means drained: every counter back to EXACTLY zero, so the next burst opens on a
        # genuine tie. Without the settle-to-zero this reads a few e-16 on some subset of the workers.
        @test all(a -> a[] == 0.0, values(@atomic s.worker_load))
    end
    # ...and because the ties are real, the opening replica rotates over the whole set instead of
    # pinning to whichever workers carry no crumb.
    @test length(unique(openers)) == 4
    counts = [count(==("w$i"), openers) for i in 0:3]
    @test maximum(counts) - minimum(counts) <= 1        # even to within one burst

    # The same holds for the comparison itself: loads that differ by less than a nanosecond of GPU
    # time are a tie, so the cursor decides rather than the crumb.
    @test GW._load_key(0.0) == GW._load_key(4.440892098500626e-16)
    @test GW._load_key(0.5) < GW._load_key(0.5 + 1e-6)   # a real difference still orders
end

@testset "work_basis is runtime-mutable, so a fleet can be switched without a restart" begin
    s = _pk_state(; routing_policy = "fill_least", batch_at = ("IN", 1),
                  item_costs = Dict("m" => 0.5), costs = Dict("m" => 4.0))
    @test GW.knobs(s).work_basis === :compute
    GW.route_replica(s, "m", 6)
    w0 = (@atomic s.worker_load)["w0"]
    @test w0[] ≈ 3.0                                         # 6 items x 0.5 GPU-seconds

    # Flip the live fleet. The run already open on w0 keeps receiving m (fill_least chooses only at a
    # run boundary), so the change is visible in what the next request CHARGES.
    GW.set_knobs!(s; work_basis = :requests)
    @test GW.knobs(s).work_basis === :requests
    GW.route_replica(s, "m", 6)
    @test w0[] ≈ 7.0                                         # +4.0, the per-request cost, size-blind

    # Validated exactly as gateway.yml validates it, and rejected without disturbing the live knobs.
    @test_throws ReactantServerCore.ConfigError GW.set_knobs!(s; work_basis = :gpu_seconds)
    @test GW.knobs(s).work_basis === :requests
end

@testset "route_replica: release frees the counter on every path" begin
    s = _pk_state(; max_batch = 4)
    held = [GW.route_replica(s, "m")[2] for _ in 1:6]
    foreach(GW._release_route!, held)
    out = @atomic s.outstanding
    @test out[("m", "w0")][] == 0 && out[("m", "w1")][] == 0
end

# Drive `route_replica` under a closed-loop client: exactly `C` requests in flight at all times, the
# oldest released as each new one is routed. Returns the per-replica share of routed requests and the
# mean number of replicas holding in-flight work (how many GPUs the model uses at once). This is the
# shape of a fixed-size consumer pool or a benchmark harness, and it is the case the `inflight` basis
# parks on.
function _closed_loop(; C, n = 2, steps = 400, kw...)
    ws = ["w$(i - 1)" for i in 1:n]
    s = _pk_state(; assignment = Dict{String,GW.Placement}("m" => [(w, 1 / n) for w in ws]), kw...)
    held = Any[]
    counts = Dict{String,Int}()
    conc = 0
    for _ in 1:steps
        length(held) >= C && GW._release_route!(popfirst!(held))
        urls, res = GW.route_replica(s, "m")
        counts[urls[1]] = get(counts, urls[1], 0) + 1
        push!(held, res)
        out = @atomic s.outstanding
        conc += count(w -> out[("m", w)][] > 0, ws)
    end
    foreach(GW._release_route!, held)
    return [get(counts, w, 0) for w in ws], conc / steps
end

@testset "fill modes: no replica starves at any concurrency" begin
    # The regression test for the parking bug. Under the `inflight` basis every C in 2..Q sent 100% of
    # a model's traffic to one replica forever; `run` and `spread` must give every replica a share at
    # every concurrency, and stay within one quantum of even.
    Q = 8
    for mode in ("run", "spread"), policy in ("fill_rr", "fill_least"),
        n in (2, 3, 4), C in (1, 2, 4, 7, 8, 9, 16, 24)

        shares, _ = _closed_loop(; C = C, n = n, steps = 400, max_batch = Q,
                                 fill_mode = mode, routing_policy = policy)
        @test sum(shares) == 400
        @test minimum(shares) > 0                                  # nobody starves
        @test maximum(abs.(shares .- 400 / n)) <= Q                # even to within one run
    end
end

@testset "fill modes: run counts routed requests, not in-flight" begin
    # Two requests in flight is far below Q=8, so the `inflight` basis can never reach its threshold
    # and parks. The `run` basis rotates after Q *routed* requests regardless of how few are in flight.
    run_shares, run_conc = _closed_loop(; C = 2, n = 2, steps = 400, max_batch = 8, fill_mode = "run")
    @test run_shares == [200, 200]
    inflight_shares, _ = _closed_loop(; C = 2, n = 2, steps = 400, max_batch = 8,
                                      fill_mode = "inflight")
    @test inflight_shares == [400, 0]                              # legacy behavior, pinned on purpose

    # `run` keeps the model on one GPU at a time (deep batches, fair over runs); `spread` uses both at
    # once (concurrent service, shallower batches). This is the whole distinction between them.
    _, spread_conc = _closed_loop(; C = 4, n = 2, steps = 400, max_batch = 8, fill_mode = "spread")
    _, run_conc4 = _closed_loop(; C = 4, n = 2, steps = 400, max_batch = 8, fill_mode = "run")
    @test spread_conc > 1.99          # both replicas busy essentially always
    @test 1.0 < run_conc4 < 1.5       # mostly one at a time, straddling only at run boundaries
    @test spread_conc > run_conc4
end

@testset "fill quantum counts items, not requests" begin
    # The requirement: 32 requests of one item and one request of 32 items are the same amount of
    # work, so both spend a quantum of 32. A client that pre-batches to max_batch would otherwise
    # hold a replica for 32 full batches.
    mb = Dict("m" => 32)
    s = _pk_state(; max_batch = 32, fill_mode = "run")
    @test (@atomic s.fill_plan)["m"].quantum == 32

    # One request carrying a full batch spends the whole run: the next request rotates.
    urls1, c1 = GW.route_replica(s, "m", 32)
    urls2, c2 = GW.route_replica(s, "m", 32)
    @test urls1[1] != urls2[1]
    # ...and it is charged as 32 in flight, not 1.
    @test (@atomic s.outstanding)[("m", urls1[1])][] == 32
    GW._release_route!(c1)
    GW._release_route!(c2)
    @test all(a -> a[] == 0, values(@atomic s.outstanding))    # release subtracts the same items

    # 32 single-item requests spend exactly the same run.
    s2 = _pk_state(; max_batch = 32, fill_mode = "run")
    held = [GW.route_replica(s2, "m", 1) for _ in 1:32]
    @test all(h -> h[1][1] == held[1][1][1], held)             # one run
    @test GW.route_replica(s2, "m", 1)[1][1] != held[1][1][1]  # 33rd opens the next
    foreach(h -> GW._release_route!(h[2]), held)

    # Mixed sizes add up the same way: 16 + 8 + 8 = 32 closes the run.
    s3 = _pk_state(; max_batch = 32, fill_mode = "run")
    mixed = [GW.route_replica(s3, "m", n) for n in (16, 8, 8)]
    @test all(h -> h[1][1] == mixed[1][1][1], mixed)
    @test GW.route_replica(s3, "m", 1)[1][1] != mixed[1][1][1]
    foreach(h -> GW._release_route!(h[2]), mixed)
end

@testset "request_units: the worker says where the batch axis is" begin
    import ProtoBuf as QPB
    QInf = ReactantServerCore.inference
    enc(msg) = (io = IOBuffer(); QPB.encode(QPB.ProtoEncoder(io), msg); take!(io))
    body(name, shape) = enc(QInf.ModelInferRequest(; model_name = "m",
        inputs = [QInf.var"ModelInferRequest.InferInputTensor"(;
            name = name, datatype = "UINT8", shape = Int64[shape...])]))

    # Batch last, as an image bundle declares it (`whcn`): the item count is the 4th axis, and a
    # first-axis assumption would have charged the width.
    s = _pk_state(; max_batch = 32, fill_mode = "run", batch_at = ("IN", 4))
    @test GW.request_units(s, "m", body("IN", (1024, 768, 3, 5))) == 5

    # Batch first, as a tokenized bundle declares it (`nc`).
    s1 = _pk_state(; max_batch = 32, fill_mode = "run", batch_at = ("IN", 1))
    @test GW.request_units(s1, "m", body("IN", (7, 512))) == 7

    # A model that declares no batch axis charges one item per request whatever its shape says, and
    # so does a request that omits the named input or has too short a shape.
    s0 = _pk_state(; max_batch = 32, fill_mode = "run")          # no locator reported
    @test GW.request_units(s0, "m", body("IN", (3, 224, 224))) == 1
    @test GW.request_units(s1, "m", body("OTHER", (7, 512))) == 1
    @test GW.request_units(s1, "unknown-model", body("IN", (7, 512))) == 1
    s9 = _pk_state(; max_batch = 32, fill_mode = "run", batch_at = ("IN", 9))
    @test GW.request_units(s9, "m", body("IN", (7, 512))) == 1

    # Schedulers that route by request count never touch the body.
    @test GW.request_units(GW.RoundRobinScheduler(), "m", UInt8[]) == 1
    @test GW.request_units(GW.LeastOutstandingScheduler(), "m", UInt8[]) == 1
end

@testset "fill modes: spread weighs items, so one big request outranks two small ones" begin
    # w0 holding a single 32-item request is busier than w1 holding two 1-item requests, and spread
    # must see that. Counting requests would have picked the wrong replica.
    s = _pk_state(; max_batch = 32, fill_mode = "spread")
    big = GW.route_replica(s, "m", 32)
    small1 = GW.route_replica(s, "m", 1)
    small2 = GW.route_replica(s, "m", 1)
    @test small1[1][1] != big[1][1]                            # the empty replica
    @test small2[1][1] == small1[1][1]                         # still lighter than 32 items
    out = @atomic s.outstanding
    @test out[("m", big[1][1])][] == 32
    @test out[("m", small1[1][1])][] == 2
    foreach(h -> GW._release_route!(h[2]), (big, small1, small2))
end

@testset "fill modes: run rotates every Q requests" begin
    s = _pk_state(; max_batch = 8, fill_mode = "run")
    # Hold everything in flight so only the routed count can end the run.
    held = [GW.route_replica(s, "m") for _ in 1:8]
    first_w = held[1][1][1]
    @test all(h -> h[1][1] == first_w, held)                       # one run of Q = 8
    ninth, c9 = GW.route_replica(s, "m")
    @test ninth[1] != first_w                                      # the next run opens elsewhere
    @test Set(ninth) == Set(["w0", "w1"])                          # both present as failover
    foreach(h -> GW._release_route!(h[2]), held)
    GW._release_route!(c9)
end

@testset "fill modes: a replica a quantum behind loses its turn" begin
    # Backpressure: rotation alone would return to w0, but w0 is a whole quantum deeper in in-flight
    # work, so the run stays on the draining replica instead of queueing on the slow one.
    s = _pk_state(; max_batch = 8, fill_mode = "run")
    run1 = [GW.route_replica(s, "m") for _ in 1:8]                 # fills w0
    run2 = [GW.route_replica(s, "m") for _ in 1:8]                 # fills w1
    @test run1[1][1][1] != run2[1][1][1]
    foreach(h -> GW._release_route!(h[2]), run2)                   # only w1 drains
    urls, c = GW.route_replica(s, "m")
    @test urls[1] == run2[1][1][1]                                 # not the backed-up replica
    foreach(h -> GW._release_route!(h[2]), run1)
    GW._release_route!(c)
end

@testset "fill modes: an unknown max batch degrades to rotation" begin
    # max_batch 0 means the worker reported no compiled batch shape, so the quantum is 1: there is no
    # batch to protect and every request opens a new run.
    s = _pk_state(; max_batch = 0, fill_mode = "run")
    @test (@atomic s.fill_plan)["m"].quantum == 1
    picks = String[]
    for _ in 1:4
        urls, c = GW.route_replica(s, "m")
        push!(picks, urls[1])
        GW._release_route!(c)
    end
    @test picks == ["w0", "w1", "w0", "w1"]
end

@testset "fill modes: a run ends when the placement changes under it" begin
    # A repack can swap a model onto a different worker set mid-run. The carried run record indexes the
    # old placement, so routing must reopen instead of indexing out of bounds.
    s = _pk_state(; max_batch = 8, fill_mode = "run")
    held = [GW.route_replica(s, "m") for _ in 1:3]                 # a run is open on w0
    foreach(h -> GW._release_route!(h[2]), held)
    next = Dict{String,GW.Placement}("m" => [("w1", 0.5), ("w2", 0.5)])
    @atomic s.assignment = next
    GW._swap_outstanding!(s, next)
    shares = Dict{String,Int}()
    for _ in 1:40
        urls, c = GW.route_replica(s, "m")
        shares[urls[1]] = get(shares, urls[1], 0) + 1
        GW._release_route!(c)
    end
    @test sort(collect(keys(shares))) == ["w1", "w2"]               # only the new placement is used
    @test minimum(values(shares)) > 0
end

@testset "fill modes: per-model override beats the scheduling default" begin
    # Three-level resolution: a model's own fill_mode/fill_factor, else the `scheduling:` default, else
    # the built-in. This is what makes a default set at the scheduling level reach models with no block.
    cfg = GW.GatewayConfig("0.0.0.0:0", "0.0.0.0:0", String[], String[], String[], 60, 1, 1, "info",
        "json", "lpt_packing", "compute", 30.0, 0.0, 0.8, 0.1, 30.0, 1, 1.0, "fill_rr", "spread",
        Dict("pinned" => GW.GatewayModelConfig(2, :inflight, 0.0),
             "short" => GW.GatewayModelConfig(2, :inherit, 0.25)),
        32, 64, :off, 0, false)
    s = GW.LptPackingState(cfg)
    mb = Dict("pinned" => 8, "short" => 8, "plain" => 8)
    plan = GW._publish_fill_plan!(s, GW.knobs(s), mb)
    @test plan["pinned"].mode == :inflight && plan["pinned"].quantum == 8   # mode overridden
    @test plan["short"].mode == :spread && plan["short"].quantum == 2       # factor overridden only
    @test plan["plain"].mode == :spread && plan["plain"].quantum == 8       # inherits both

    # Clearing an override returns the model to the fleet default; setting the fleet default moves
    # every model that has no override of its own.
    GW.set_knobs!(s; model_overrides = Dict("short" => GW.ModelKnobs(2, :inherit, 0.0)))
    plan = GW._publish_fill_plan!(s, GW.knobs(s), mb)
    @test plan["short"].quantum == 8 && plan["pinned"].mode == :spread      # both back to the default
    GW.set_knobs!(s; routing_fill_mode = "run")
    plan = GW._publish_fill_plan!(s, GW.knobs(s), mb)
    @test all(p -> p.mode == :run, values(plan))
end

@testset "config: routing_fill_mode and the per-model overrides" begin
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
    @test _load(eps).routing_fill_mode == "run"                    # default
    @test _load("scheduling:\n  routing_fill_mode: spread\n" * eps).routing_fill_mode == "spread"
    @test _load("scheduling:\n  routing_fill_mode: inflight\n" * eps).routing_fill_mode == "inflight"
    @test_throws ReactantServerCore.ConfigError _load("scheduling:\n  routing_fill_mode: bogus\n" * eps)
    @test_throws ReactantServerCore.ConfigError _load("scheduling:\n  routing_fill_mode: inherit\n" * eps)
    withenv("REACTANT_GATEWAY_SCHEDULING_ROUTING_FILL_MODE" => "spread") do
        @test _load(eps).routing_fill_mode == "spread"
    end
    mc = _load("scheduling:\n  models:\n    a:\n      replicas: 2\n      fill_mode: inherit\n" *
               "      fill_factor: 0.5\n" * eps).models["a"]
    @test mc.replicas == 2 && mc.fill_mode == :inherit && mc.fill_factor == 0.5
    @test _load("scheduling:\n  models:\n    a:\n      fill_mode: spread\n" * eps).models["a"].fill_mode == :spread
    @test_throws ReactantServerCore.ConfigError _load(
        "scheduling:\n  models:\n    a:\n      fill_factor: 0\n" * eps)
end

@testset "reset_clients! recovers a poisoned (stalled) worker connection" begin
    # A server that accepts TCP but never speaks gRPC/HTTP-2: the connection establishes then stalls
    # on the HTTP/2 handshake, exactly like a worker caught in its brief silent-accept window at
    # startup. With PIPEWAIT (which the client keeps for multiplexing), libcurl pools the half-open
    # connection and later requests reuse it; a handle parked waiting for that connection to become
    # multiplexable never re-enters libcurl's state machine, so its own CURLOPT_TIMEOUT_MS never
    # fires. gRPCClient 1.1 backstops that with a client-side deadline watchdog, so every call on a
    # poisoned connection now fails at its deadline instead of wedging forever. The gateway still
    # calls reset_clients! when a probe to a worker hangs (the per-worker equivalent of a process
    # restart); this checks the handle is usable again afterwards.
    srv = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(srv)[2])
    acceptor = @async try
        while isopen(srv)
            Sockets.accept(srv)   # accept and hold; never respond
        end
    catch
    end

    grpc = gRPCClient.gRPCCURL(; sticky = true)
    client = gRPCClient.gRPCServiceClient{Vector{UInt8},false,Vector{UInt8},false}(
        "127.0.0.1", port, "/probe.Svc/Call"; grpc = grpc, deadline = 0.5)
    # Returns (whether the call returned within `cap`, the exception it threw).
    bounded_call(cap) = begin
        err = Ref{Any}(nothing)
        t = @async try
            gRPCClient.grpc_sync_request(client, UInt8[0x00, 0x00, 0x00, 0x00, 0x00])
        catch e
            err[] = e
        end
        (timedwait(() -> istaskdone(t), cap), err[])
    end
    expired(e) = e isa gRPCClient.gRPCServiceCallException &&
                 e.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED

    # Both calls end at their 0.5s deadline: the first driven by libcurl's own timeout, the second
    # (which reuses the poisoned pooled connection) by the client's watchdog.
    for _ in 1:2
        status, err = bounded_call(3.0)
        @test status == :ok
        @test expired(err)
    end

    wc = GW.WorkerClients("127.0.0.1:$port", grpc, client, client, client, client, client, client, client, client)
    GW.reset_clients!(wc)                    # close + reopen the handle: drops the poisoned connection

    status, err = bounded_call(3.0)          # fresh connection: still bounded by the deadline
    @test status == :ok
    @test expired(err)

    gRPCClient.grpc_shutdown(grpc)
    close(srv)
end

# --- Integration: mock workers serving inference + control ------------------------------------

mutable struct AffMockWorker
    name::String
    models::Vector{String}
    discipline::String
    served::Dict{String,Int}
    compute::Dict{String,Float64}
    # Rows (batch-axis items) served, tracked separately from requests so the pair can diverge the
    # way it does in production, and the wire batch-axis locator a real worker reports.
    rows::Dict{String,Int}
    rows_per_request::Int
    batch_at::Tuple{String,Int}
end
AffMockWorker(name, models; discipline = "fifo", rows_per_request = 1, batch_at = ("IN", 1)) =
    AffMockWorker(name, models, discipline, Dict{String,Int}(), Dict{String,Float64}(),
                  Dict{String,Int}(), rows_per_request, batch_at)

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
            w.rows[req.model_name] = get(w.rows, req.model_name, 0) + w.rows_per_request
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
                              rows_served = UInt64(get(w.rows, m, 0)),
                              dispatch_count = UInt64(get(w.served, m, 0)),
                              max_batch_size = Int64(8),
                              batch_input_name = w.batch_at[1],
                              batch_axis = Int64(w.batch_at[2]))
                          for m in w.models],
                weight_cache_max_bytes = UInt64(8) * 1024^3)
        end,
        # Eager compaction is the default now, so the gateway may fan CompactMemory out after a
        # placement-changing repack; answer it so the call never hits the generous client deadline.
        CompactMemory = (req, c) -> ACtl.CompactMemoryResponse(; reloaded_models = Int64(0)),
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
        aff = gw.prober.scheduler
        @test aff isa GW.LptPackingState
        GW.rebalance!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.did_first_tick_repack == false    # the startup cold placement does not count as the first run

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
        # budget is crossed. Disable the separate first-repack budget here so this block exercises the
        # steady-state threshold directly (the first-vs-steady budget is covered in the block below).
        GW.set_knobs!(aff; first_rebalance_compute_seconds = 0.0,
                      rebalance_compute_seconds = 1.0e9)   # effectively never
        before = aff.last_rebalance
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance == before          # not enough compute -> no repack
        @test aff.compute_accum > 0                 # but the compute was accounted

        GW.set_knobs!(aff; rebalance_compute_seconds = 1.0e-9)   # any compute triggers
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance > before           # repacked
        @test aff.compute_accum == 0.0              # accumulator reset on repack

        # First-run vs steady-state budget: the first tick-driven repack uses the smaller
        # first_rebalance_compute_seconds, then repacks after need the larger steady-state budget.
        aff.did_first_tick_repack = false
        GW.set_knobs!(aff; first_rebalance_compute_seconds = 1.0e-9,   # any compute triggers the first
                      rebalance_compute_seconds = 1.0e9)              # steady state: effectively never
        before2 = aff.last_rebalance
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance > before2             # first repack fired on the small budget
        @test aff.did_first_tick_repack               # flag now set

        before3 = aff.last_rebalance
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
        end
        GW.tick_packing!(aff, gw.pool, copy(gw.pool.order), gw.metrics)
        @test aff.last_rebalance == before3            # steady-state budget too large -> no repack

        # Placement is observable in the metrics.
        body = String(HTTP.get("http://127.0.0.1:$admin_port/metrics"; retry = false).body)
        @test occursin("gateway_placement_weight", body)
        @test occursin("gateway_model_utilization", body)
        @test occursin("gateway_model_replicas", body)
        # ...and so is the per-worker load fill_least compares, labelled with the basis in force, so a
        # routing decision can be explained after the fact rather than guessed at.
        @test occursin("# TYPE gateway_worker_inflight_work gauge", body)
        @test occursin("gateway_worker_inflight_work{worker=", body)
        @test occursin("basis=\"compute\"", body)
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

# The routing metadata has to survive the whole trip (worker control_status -> ModelStatus proto ->
# gateway poll -> cache), and every hop is a place it can be silently dropped: a handler that never
# populates the fields leaves the gateway counting requests, which is indistinguishable from working.
# So this checks it against live workers rather than a hand-built poll.
@testset "routing metadata reaches the gateway over the control plane" begin
    models = ["alpha", "beta"]
    # Each request carries 4 rows, so cost per item is a quarter of cost per request. A mock that
    # reported rows == requests could not tell the two apart.
    w0 = AffMockWorker("worker0", copy(models); rows_per_request = 4, batch_at = ("IN", 2))
    w1 = AffMockWorker("worker1", copy(models); rows_per_request = 4, batch_at = ("IN", 2))
    p0, p1 = grpc_free_port(), grpc_free_port()
    s0 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p0; context = w0)
    s1 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p1; context = w1)

    gw_port, admin_port = grpc_free_port(), grpc_free_port()
    gatewayfile = _aff_gatewayfile(gw_port, admin_port, [p0, p1])
    gw = GW.serve_gateway(gatewayfile; blocking = false)
    try
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
        for _ in 1:20
            _aff_infer(gw_port, "alpha")
        end

        # The prober owns the live cache and refreshes it from the round's poll. Driven directly here
        # rather than waiting out a probe interval, so the assertion is not a race.
        GW._check_once(gw.prober)
        live = GW.model_meta(GW.routing_meta(gw.prober.meta), "alpha")
        @test live.batch_input == "IN"
        @test live.batch_axis == 2

        meta = GW.RoutingMeta(30.0)
        snap = GW.refresh_routing_meta!(meta, GW.poll_workers(gw.pool, copy(gw.pool.order)))
        m = GW.model_meta(snap, "alpha")
        @test m.batch_input == "IN"                  # the locator, not a positional guess
        @test m.batch_axis == 2
        @test m.max_batch == 8
        # 0.05 GPU-seconds per request and 4 rows per request, so the per-item cost is a quarter of
        # the per-request cost. This is the number a work-weighted router multiplies an item count by.
        @test m.cost_per_request ≈ 0.05 rtol = 1e-6
        @test m.cost_per_item ≈ 0.0125 rtol = 1e-6
        @test GW.item_cost(snap, "alpha") ≈ 0.0125 rtol = 1e-6
        # A model nobody has called yet has no cost of its own and borrows the fleet mean.
        @test GW.model_meta(snap, "beta").cost_per_item == 0.0
        @test GW.item_cost(snap, "beta") ≈ 0.0125 rtol = 1e-6

        # And a request is sized from that locator: axis 2 of IN, not its first dimension.
        req = AInf.ModelInferRequest(; model_name = "alpha",
            inputs = [AInf.var"ModelInferRequest.InferInputTensor"(;
                name = "IN", datatype = "UINT8", shape = Int64[512, 6])])
        io = IOBuffer()
        ReactantServerGateway.PB.encode(ReactantServerGateway.PB.ProtoEncoder(io), req)
        body = take!(io)
        @test GW.request_items(snap, "alpha", body) == 6
    finally
        GW.stop!(gw)
        close(s0)
        close(s1)
        rm(gatewayfile; force = true)
    end
end
