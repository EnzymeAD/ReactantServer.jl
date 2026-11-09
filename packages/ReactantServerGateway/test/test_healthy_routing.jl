# The gateway never sends a request to a worker that is not healthy: once the prober finds a worker
# not ready, unreachable or hung, that worker gets no request, as a first choice or as a failover, in
# every scheduling mode and for shared-memory requests as for inline ones, until the prober finds it
# ready again. A model that only such workers serve is answered UNAVAILABLE without a worker call.
#
# Before, a model that only workers in an outage served was still routed to them, and lpt_packing
# chose among a model's placed workers without looking at health: a placed worker that was down kept
# first choice for its share of the traffic, a single-replica model placed on it always tried it
# first, a shared-memory request that failed there was never moved, and a hung worker held requests
# until their deadline.
#
# Under lpt_packing a change in the healthy workers also repacks on the round that sees it
# (`trigger = :membership`), where before a repack waited for the compute budget and then for a
# missing worker's 30 s grace.
#
# Reuses `MockWorker`, `_start_mock`, `_infer` (test_gateway.jl), `_retention_gateway`, `_rstatus`,
# `_rurl` (test_route_retention.jl), `_sched_cfg` (test_scheduler.jl), and `AffMockWorker`,
# `_aff_server`, `_aff_infer`, `_pk_state` (test_lpt_packing.jl).

const _HGW = ReactantServerGateway

# Whether the gateway has the worker at `url` marked down.
_hr_down(gw, url) = lock(() -> haskey(gw.routes.down_since, url), gw.routes.lock)

# The ModelInfer calls the gateway made to the worker at `url`, answered or failed, as its per-worker
# call latency histogram counts them. Counts calls to a worker that is gone, which no mock can.
function _hr_attempts(gw, url)
    body = String(HTTP.get("http://127.0.0.1:$(_HGW.bound_ports(gw)[2])/metrics"; retry = false).body)
    label = get(gw.metrics.worker_names, url, url)
    for line in eachsplit(body, '\n')
        startswith(line, "gateway_worker_latency_seconds_count{") || continue
        (occursin("rpc=\"ModelInfer\"", line) && occursin("worker=\"$label\"", line)) || continue
        return round(Int, parse(Float64, last(split(line))))
    end
    return 0
end

# A ModelInfer for `model` whose input lives in a registered shared-memory region.
_hr_shm_request(model) = GWInf.ModelInferRequest(;
    model_name = model,
    inputs = [
        GWInf.var"ModelInferRequest.InferInputTensor"(;
            name = "x", datatype = "FP32", shape = Int64[1],
            parameters = Dict(
                "shared_memory_region" =>
                    GWInf.InferParameter(; parameter_choice = TPB.OneOf(:string_param, "client-slot-0"))
            )
        ),
    ]
)
_hr_call(port, req; deadline = 10) = grpc_call(GWInf.GRPCInferenceService.ModelInfer, port, req; deadline = deadline)

@testset "every scheduling mode offers only healthy workers, first choice and failover alike" begin
    a, b = "127.0.0.1:7001", "127.0.0.1:7002"
    pool = GW.ClientPool(_sched_cfg("round_robin"))
    routes = GW.DiscoveredRoutes()
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics)
    ctx(model) = GW.ScheduleContext(model, "id", 1, pool, routes, metrics, refresher)
    # m is on both workers, n is placed on b alone, and only b serves onlyb.
    lpt() = _pk_state(; assignment = Dict{String, GW.Placement}("m" => [(a, 0.5), (b, 0.5)], "n" => [(b, 1.0)]))
    modes = () -> (GW.RoundRobinScheduler(), GW.LeastOutstandingScheduler(), lpt())
    try
        GW.record_answers!(routes, Dict(a => ["m", "n"], b => ["m", "n", "onlyb"]))
        # b goes down; the next discovery round cannot reach it.
        GW.note_worker_ready!(routes, b, false)
        GW.record_answers!(routes, Dict(a => ["m", "n"], b => nothing))
        for s in modes()
            for model in ("m", "n"), _ in 1:4
                urls, res = GW.select_replicas(s, ctx(model))
                @test urls == [a]
                GW.release!(s, res)
            end
            # Known, but no healthy worker serves it: an empty list, which the gateway answers
            # UNAVAILABLE without a worker call. A model no worker ever reported stays unknown.
            sel = GW.select_replicas(s, ctx("onlyb"))
            @test sel !== nothing && isempty(sel[1])
            @test GW.select_replicas(s, ctx("ghost")) === nothing
        end
        # b is ready again: every mode offers it again.
        GW.note_worker_ready!(routes, b, true)
        GW.record_answers!(routes, Dict(a => ["m", "n"], b => ["m", "n", "onlyb"]))
        for s in modes()
            urls, res = GW.select_replicas(s, ctx("m"))
            @test Set(urls) == Set([a, b])
            GW.release!(s, res)
            @test first(GW.select_replicas(s, ctx("onlyb"))) == [b]
        end
    finally
        GW.close_pool!(pool)
    end
end

@testset "route_replica: a down replica is skipped; the run, cursor and counters stay consistent" begin
    asn = Dict{String, GW.Placement}("m" => [("w0", 1 / 3), ("w1", 1 / 3), ("w2", 1 / 3)])
    for mode in ("run", "spread", "inflight"), policy in ("fill_rr", "fill_least")
        s = _pk_state(; max_batch = 4, fill_mode = mode, routing_policy = policy, assignment = asn)
        held = [GW.route_replica(s, "m"; down = Set(["w1"])) for _ in 1:20]
        @test all(h -> !("w1" in h[1]) && length(h[1]) == 2 && h[2] !== nothing, held)
        @test Set(h[1][1] for h in held) == Set(["w0", "w2"])
        run = (@atomic s.fill_runs)["m"]
        @test run.target in (1, 3) && 0 <= run.cursor < 3       # indices into the full placement
        @test (@atomic s.outstanding)[("m", "w1")][] == 0
        foreach(h -> GW._release_route!(h[2]), held)
        @test all(c -> c[] == 0, values(@atomic s.outstanding))
        @test all(c -> c[] == 0, values(@atomic s.worker_load))
        # Back: w1 serves again.
        firsts = Set{String}()
        for _ in 1:30
            urls, c = GW.route_replica(s, "m"; down = Set{String}())
            push!(firsts, urls[1])
            GW._release_route!(c)
        end
        @test firsts == Set(["w0", "w1", "w2"])
    end
    # A run open on a replica that goes down ends; the next request opens on a healthy one.
    s = _pk_state(; max_batch = 8, fill_mode = "run")
    urls, c = GW.route_replica(s, "m")
    open = urls[1]
    nxt, c2 = GW.route_replica(s, "m"; down = Set([open]))
    @test nxt == [only(setdiff(["w0", "w1"], [open]))]
    GW._release_route!(c); GW._release_route!(c2)
    # Every placed replica down: nothing offered and nothing reserved.
    @test GW.route_replica(s, "m"; down = Set(["w0", "w1"])) == (String[], nothing)
    s1 = _pk_state(; assignment = Dict{String, GW.Placement}("solo" => [("w0", 1.0)]))
    @test GW.route_replica(s1, "solo"; down = Set(["w0"])) == (String[], nothing)
    @test (@atomic s1.worker_load)["w0"][] == 0
end

@testset "round_robin: a worker the prober finds not ready gets no request" begin
    w0 = MockWorker("worker0", ["m"], false, false, false)
    w1 = MockWorker("worker1", ["m", "only1"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    u1 = _rurl(p1)
    gw = _retention_gateway([p0, p1]; env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.5"])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> _RGW.has_model(gw.routes, "only1") && length(something(_RGW.replicas(gw.routes, "m"), String[])) == 2, 30) === :ok

        w1.ready = false
        @test timedwait(() -> _hr_down(gw, u1), 10) === :ok      # one prober round
        w0.infer_calls = w1.infer_calls = 0
        for _ in 1:6
            @test _infer(port, "m").model_name == "worker0"
            @test _hr_call(port, _hr_shm_request("m")).model_name == "worker0"
        end
        # A model only worker1 serves: UNAVAILABLE, inline and shared memory, and no call to it.
        @test _rstatus(() -> _infer(port, "only1")) == gRPCClient.GRPC_UNAVAILABLE
        @test _rstatus(() -> _hr_call(port, _hr_shm_request("only1"))) == gRPCClient.GRPC_UNAVAILABLE
        @test w1.infer_calls == 0
        @test w0.infer_calls == 12

        # Ready again: routable from the round that sees it.
        w1.ready = true
        @test timedwait(() -> !_hr_down(gw, u1), 10) === :ok
        @test _infer(port, "only1").model_name == "worker1"
        @test Set(_infer(port, "m").model_name for _ in 1:4) == Set(["worker0", "worker1"])
    finally
        _RGW.stop!(gw)
        close(s0)
        close(s1)
    end
end

@testset "round_robin: a killed worker gets no request once the prober has seen it" begin
    w0 = MockWorker("worker0", ["m"], false, false, false)
    w1 = MockWorker("worker1", ["m", "only1"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    u1 = _rurl(p1)
    gw = _retention_gateway([p0, p1]; env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.5"])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> _RGW.has_model(gw.routes, "only1") && length(something(_RGW.replicas(gw.routes, "m"), String[])) == 2, 30) === :ok

        gRPCServer.stop!(s1; force = true)           # connections to it are now refused
        @test timedwait(() -> _hr_down(gw, u1), 10) === :ok
        before = _hr_attempts(gw, u1)
        for _ in 1:6
            @test _infer(port, "m").model_name == "worker0"
            @test _hr_call(port, _hr_shm_request("m")).model_name == "worker0"
        end
        @test _rstatus(() -> _infer(port, "only1")) == gRPCClient.GRPC_UNAVAILABLE
        @test _rstatus(() -> _hr_call(port, _hr_shm_request("only1"))) == gRPCClient.GRPC_UNAVAILABLE
        @test _hr_attempts(gw, u1) == before           # not one call was made to it

        # Back on its port: routable again once the prober sees it ready.
        s1, _ = _start_mock(w1; port = p1)
        @test timedwait(() -> _rstatus(() -> _infer(port, "only1")) === nothing, 30) === :ok
    finally
        _RGW.stop!(gw)
        close(s0)
        gRPCServer.stop!(s1; force = true)
    end
end

# An lpt_packing gateway over `ports` whose prober runs one round at startup and then only when a
# test calls `_check_once`, with compute budgets no test reaches, and `m` placed on two GPUs.
function _hr_lpt_gateway(ports)
    path = tempname() * ".yaml"
    eps = join(("  - \"127.0.0.1:$p\"" for p in ports), "\n")
    write(
        path, """
        listen:
          grpc: "127.0.0.1:0"
          metrics: "127.0.0.1:0"
        scheduling:
          mode: lpt_packing
          rebalance_compute_seconds: 1000000000
          first_rebalance_compute_seconds: 1000000000
          models:
            m:
              replicas: 2
        endpoints:
        $eps
        """
    )
    env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "3600", "REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS" => "0"]
    gw = try
        withenv(() -> _HGW.serve_gateway(path; blocking = false), env...)
    finally
        rm(path; force = true)
    end
    aff = gw.prober.scheduler
    timedwait(() -> (@atomic aff.tick).unix_seconds > 0, 30)   # the startup round has ticked
    return gw, aff
end

# Place n on `ub` alone and m on both workers, as a repack could have.
function _hr_place!(aff, ua, ub)
    asn = Dict{String, GW.Placement}("m" => sort([(ua, 0.5), (ub, 0.5)]), "n" => [(ub, 1.0)])
    @atomic aff.assignment = asn
    GW._swap_outstanding!(aff, asn)
    return asn
end

@testset "lpt_packing: a placed worker that is not ready gets no request, inline or shared memory" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        _hr_place!(aff, ua, ub)
        @test _aff_infer(port, "n").model_name == "workerB"   # n's one replica is B

        B.ready = false
        _HGW._check_once(gw.prober)
        @test _hr_down(gw, ub)
        b0 = B.calls
        for _ in 1:8
            @test _aff_infer(port, "n").model_name == "workerA"
            @test _aff_infer(port, "m").model_name == "workerA"
        end
        for _ in 1:4
            @test _hr_call(port, _hr_shm_request("n")).model_name == "workerA"
            @test _hr_call(port, _hr_shm_request("m")).model_name == "workerA"
        end
        @test B.calls == b0

        # Ready again: m's traffic reaches B again.
        B.ready = true
        _HGW._check_once(gw.prober)
        @test !_hr_down(gw, ub)
        @test timedwait(() -> (_aff_infer(port, "m"); B.calls > b0), 15) === :ok
    finally
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end

@testset "lpt_packing: a hung worker is marked down at the probe timeout and gets no request" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        _hr_place!(aff, ua, ub)
        @test _aff_infer(port, "n").model_name == "workerB"

        # B stops answering anything. The probe round waits for its probe's deadline (5 s, under the
        # prober's 8 s watchdog) and then marks it down.
        B.hang = true
        b0 = B.calls
        t0 = time()
        _HGW._check_once(gw.prober)
        waited = time() - t0
        @test _hr_down(gw, ub)
        @test 4.0 <= waited < GW.PROBE_TIMEOUT_SECONDS + 2.0
        # No request goes to it: n, placed only on B, is served by A at once instead of waiting for a
        # deadline at B.
        for _ in 1:4
            t1 = time()
            @test _rstatus(() -> _hr_call(port, GWInf.ModelInferRequest(; model_name = "n"); deadline = 3)) === nothing
            @test time() - t1 < 2.0
            @test _hr_call(port, _hr_shm_request("n"); deadline = 3).model_name == "workerA"
        end
        @test B.calls == b0
    finally
        B.hang = false
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end

# How many repacks `gateway_repacks_total` counts for `trigger`.
function _hr_repacks(gw, trigger)
    body = String(HTTP.get("http://127.0.0.1:$(_HGW.bound_ports(gw)[2])/metrics"; retry = false).body)
    m = match(Regex("gateway_repacks_total\\{trigger=\"$trigger\"\\} (\\S+)"), body)
    return m === nothing ? 0 : round(Int, parse(Float64, m.captures[1]))
end

@testset "lpt_packing: losing a worker repacks at once; its return repacks it back in" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        _hr_place!(aff, ua, ub)
        rep0 = @atomic aff.report
        @test _aff_infer(port, "n").model_name == "workerB"

        # B goes down. The next round repacks for it, with the compute budget nowhere near reached.
        B.ready = false
        _HGW._check_once(gw.prober)
        rep = @atomic aff.report
        @test rep.trigger === :membership
        @test rep.count == rep0.count + 1
        @test (@atomic aff.tick).active_threshold >= 1.0e9 && aff.compute_accum == 0.0
        @test !aff.did_first_tick_repack                # the first-repack budget stays armed
        @test _hr_repacks(gw, "membership") == 1
        asn = @atomic aff.assignment
        @test asn["n"] == [(ua, 1.0)]                   # the single replica moved to the survivor
        @test asn["m"] == [(ua, 1.0)]                   # two replicas, capped by one worker present
        @test all(pl -> all(wv -> first(wv) != ub, pl), values(asn))
        b0 = B.calls
        for _ in 1:6
            @test _aff_infer(port, "n").model_name == "workerA"
            @test _aff_infer(port, "m").model_name == "workerA"
            @test _hr_call(port, _hr_shm_request("n")).model_name == "workerA"
        end
        @test B.calls == b0

        # B is back. The round that sees it repacks again, and m gets its second replica (a move
        # onto a worker that does not hold it is staged first, so wait for the commit).
        B.ready = true
        _HGW._check_once(gw.prober)
        @test (@atomic aff.report).trigger === :membership
        @test _hr_repacks(gw, "membership") == 2
        @test timedwait(() -> aff.transition === nothing, 15) === :ok
        @test sort(first.((@atomic aff.assignment)["m"])) == sort([ua, ub])
        @test timedwait(() -> (_aff_infer(port, "m"); B.calls > b0), 15) === :ok
        # Settled: a round with the same healthy workers repacks nothing.
        n = (@atomic aff.report).count
        _HGW._check_once(gw.prober)
        @test (@atomic aff.report).count == n
    finally
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end

@testset "lpt_packing: a worker lost while moves stage gets no request, and the placement converges" begin
    models = ["hot", "warm", "cool"]
    A = AffMockWorker("workerA", copy(models))
    B = AffMockWorker("workerB", copy(models))
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        # hot and warm share A (0.9 busy) while B idles at 0.1, so a repack moves warm to B, staging
        # its weights there first (as in test_lpt_packing.jl).
        asn = Dict{String, GW.Placement}("hot" => [(ua, 1.0)], "warm" => [(ua, 1.0)], "cool" => [(ub, 1.0)])
        @atomic aff.assignment = asn
        GW._swap_outstanding!(aff, asn)
        for (m, r) in ("hot" => 6.0, "warm" => 3.0, "cool" => 1.0)
            aff.rate_ewma[m] = r
            aff.cost_ewma[m] = 0.1
        end
        B.stage_delay = 3.0
        GW._repack!(aff, GW.poll_workers(gw.pool, [ua, ub]), gw.metrics; trigger = :operator, pool = gw.pool)
        staging = aff.transition
        @test staging !== nothing
        @test (@atomic aff.assignment)["warm"] == [(ua, 1.0)]   # not switched while it stages

        # B goes down while warm stages there. The round repacks at once over A alone, abandoning the
        # staged move rather than waiting for it.
        B.ready = false
        _HGW._check_once(gw.prober)
        @test (@atomic aff.report).trigger === :membership
        @test aff.transition === nothing
        @test all(pl -> pl == [(ua, 1.0)], values(@atomic aff.assignment))
        b0 = B.calls
        for m in models
            @test _aff_infer(port, m).model_name == "workerA"
        end
        # The abandoned staging ends (B answered its read) and commits nothing.
        @test timedwait(() -> istaskdone(staging), 15) === :ok
        @test all(pl -> pl == [(ua, 1.0)], values(@atomic aff.assignment))
        @test aff.transition === nothing
        for m in models, _ in 1:3
            @test _aff_infer(port, m).model_name == "workerA"
            @test _hr_call(port, _hr_shm_request(m)).model_name == "workerA"
        end
        @test B.calls == b0

        # B is back: the placement converges over both workers, and stays put.
        B.ready = true
        B.stage_delay = 0.0
        _HGW._check_once(gw.prober)
        @test timedwait(() -> aff.transition === nothing, 15) === :ok
        rep = @atomic aff.report
        @test rep.trigger === :membership && rep.polled == sort([ua, ub]) && !rep.staging
        @test any(pl -> any(wv -> first(wv) == ub, pl), values(@atomic aff.assignment))
        n = rep.count
        _HGW._check_once(gw.prober)
        @test (@atomic aff.report).count == n
    finally
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end
