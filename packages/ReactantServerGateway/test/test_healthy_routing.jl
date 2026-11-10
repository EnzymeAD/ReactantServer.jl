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
# A request whose connection to a worker is refused marks that worker down at once, without waiting
# for the prober, and moves on to a healthy replica even when it uses shared memory (it never left
# the gateway). Before, the worker kept getting its share of requests until the next probe round, and
# a shared-memory request that met the refusal came back UNAVAILABLE.
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

# --- A refused connection marks the worker down at once ----------------------------------------

@testset "a ready probe sent before a refusal does not clear it" begin
    d = GW.DiscoveredRoutes()
    GW.record_answers!(d, Dict("w" => ["m"]))
    @test GW.note_worker_ready!(d, "w", false, 100.0)                       # refused at t = 100
    @test !GW.is_healthy(d, "w") && GW.replicas(d, "m") == String[]
    @test !GW.note_worker_ready!(d, "w", true, 101.0; observed_at = 99.5)   # that probe was sent before
    @test !GW.is_healthy(d, "w")
    @test GW.note_worker_ready!(d, "w", true, 111.0; observed_at = 110.0)   # the next one clears it
    @test GW.is_healthy(d, "w") && GW.replicas(d, "m") == ["w"]
end

# The refusal tests kill a worker the gateway has sent no request to, so the gateway holds no
# inference connection to it and the next request must connect, which is refused. A worker that dies
# while the gateway holds a connection to it fails the next call on that connection as a broken
# connection, not a refusal (the request may have been sent), and only the call after it, which
# connects afresh, is refused; the last testset below shows that case.
@testset "round_robin: a refused connection marks the worker down at once; its request moves on (shm = $shm)" for shm in (false, true)
    w0 = MockWorker("worker0", ["m"], false, false, false)
    w1 = MockWorker("worker1", ["m"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    u1 = _rurl(p1)
    request() = shm ? _hr_shm_request("m") : GWInf.ModelInferRequest(; model_name = "m")
    # One prober round at startup, then none unless the test runs one.
    gw = _retention_gateway([p0, p1]; env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "3600"])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> length(something(_RGW.replicas(gw.routes, "m"), String[])) == 2, 30) === :ok

        # worker1 dies. Round robin puts it first for one of the next two requests, which finds its
        # connection refused and is served by worker0, a shared-memory request too, since it never
        # left the gateway.
        gRPCServer.stop!(s1; force = true)
        for _ in 1:2
            @test _hr_call(port, request()).model_name == "worker0"
        end
        @test _hr_down(gw, u1)

        # From then on no attempt is made on it, not even once a worker answers on its port again: only
        # a probe clears the mark.
        s1, _ = _start_mock(w1; port = p1)
        w1.infer_calls = 0
        before = _hr_attempts(gw, u1)
        for _ in 1:6
            @test _infer(port, "m").model_name == "worker0"
            @test _hr_call(port, _hr_shm_request("m")).model_name == "worker0"
        end
        @test w1.infer_calls == 0
        @test _hr_attempts(gw, u1) == before

        # The prober's next round finds it ready: routable again.
        _RGW._check_once(gw.prober)
        @test !_hr_down(gw, u1)
        @test Set(_infer(port, "m").model_name for _ in 1:4) == Set(["worker0", "worker1"])
    finally
        _RGW.stop!(gw)
        close(s0)
        gRPCServer.stop!(s1; force = true)
    end
end

@testset "a worker that dies under an open connection: a broken call first, then the refusal marks it down" begin
    w0 = MockWorker("worker0", ["m"], false, false, false)
    w1 = MockWorker("worker1", ["m"], false, false, false)
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)
    u1 = _rurl(p1)
    gw = _retention_gateway([p0, p1]; env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "3600"])
    try
        port, _ = _RGW.bound_ports(gw)
        @test timedwait(() -> length(something(_RGW.replicas(gw.routes, "m"), String[])) == 2, 30) === :ok
        @test Set(_infer(port, "m").model_name for _ in 1:4) == Set(["worker0", "worker1"])   # connections open
        gRPCServer.stop!(s1; force = true)
        # Every inline request is still served; within a few, one has met the refusal on a fresh
        # connection and marked worker1 down.
        for _ in 1:6
            @test _infer(port, "m").model_name == "worker0"
        end
        @test _hr_down(gw, u1)
    finally
        _RGW.stop!(gw)
        close(s0)
    end
end

@testset "lpt_packing: a refused connection marks the worker down, and the next round repacks for it" begin
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

        # B dies (no request has been sent to it yet). A shared-memory request for n, placed on B
        # alone, finds B's connection refused and is served by A; B is marked down before any
        # prober round.
        gRPCServer.stop!(sb; force = true)
        @test _hr_call(port, _hr_shm_request("n")).model_name == "workerA"
        @test _hr_down(gw, ub)
        before = _hr_attempts(gw, ub)
        for _ in 1:4
            @test _aff_infer(port, "n").model_name == "workerA"
            @test _aff_infer(port, "m").model_name == "workerA"
            @test _hr_call(port, _hr_shm_request("m")).model_name == "workerA"
        end
        @test _hr_attempts(gw, ub) == before
        @test (@atomic aff.report).count == rep0.count          # no repack until the prober's round

        # The next round repacks for it.
        _HGW._check_once(gw.prober)
        rep = @atomic aff.report
        @test rep.trigger === :membership && rep.count == rep0.count + 1
        @test all(pl -> all(wv -> first(wv) != ub, pl), values(@atomic aff.assignment))

        # B is back on its port: the round that finds it ready makes it routable and repacks it in.
        sb = _aff_server(pb, B)
        gRPCServer.start!(sb)
        @test timedwait(15) do
            try
                AInf.GRPCInferenceService.ServerReady("127.0.0.1", pb, AInf.ServerReadyRequest()).ready
            catch
                false
            end
        end === :ok
        _HGW._check_once(gw.prober)
        @test !_hr_down(gw, ub)
        @test (@atomic aff.report).trigger === :membership
        @test timedwait(() -> aff.transition === nothing, 15) === :ok
        @test sort(first.((@atomic aff.assignment)["m"])) == sort([ua, ub])
    finally
        _HGW.stop!(gw)
        close(sa)
        gRPCServer.stop!(sb; force = true)
    end
end

# A worker that dies while the gateway holds connections to it, and comes back on its port, is a new
# process: the connections pooled for the old one are dead, and the first call on one fails. The
# prober therefore resets the worker's data handle when it comes back from an outage, before it is
# marked healthy. Before, the membership repack that gave the worker its replicas back staged their
# weights over the old connection; the call failed at once, the changes were dropped, and on an idle
# fleet nothing planned them again: the worker was healthy and placed nothing (CPU e2e step 11,
# phase C, and a single worker killed under lpt_packing every time).
@testset "lpt_packing: a worker killed under open connections and restarted gets fresh ones, and its placement back" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        _hr_place!(aff, ua, ub)
        # Requests reach B, so the gateway holds a data connection to it.
        @test _aff_infer(port, "n").model_name == "workerB"
        @test B.calls > 0

        # B dies under it. The next round finds it down and repacks without it.
        gRPCServer.stop!(sb; force = true)
        _HGW._check_once(gw.prober)
        @test _hr_down(gw, ub)
        @test all(pl -> all(wv -> first(wv) != ub, pl), values(@atomic aff.assignment))

        # B is back on its port. The round that finds it ready repacks it in; m's second replica on B
        # is staged there first, and the staging reaches the new B.
        sb = _aff_server(pb, B)
        gRPCServer.start!(sb)
        @test timedwait(15) do
            try
                AInf.GRPCInferenceService.ServerReady("127.0.0.1", pb, AInf.ServerReadyRequest()).ready
            catch
                false
            end
        end === :ok
        staged = length(B.staged)
        b0 = B.calls
        _HGW._check_once(gw.prober)
        @test !_hr_down(gw, ub)
        @test (@atomic aff.report).trigger === :membership
        @test timedwait(() -> aff.transition === nothing, 15) === :ok
        @test length(B.staged) == staged + 1
        rep = @atomic aff.report
        @test rep.changes_planned > 0 && rep.changes_committed == rep.changes_planned && rep.changes_dropped == 0
        @test sort(first.((@atomic aff.assignment)["m"])) == sort([ua, ub])
        @test timedwait(() -> (_aff_infer(port, "m"); B.calls > b0), 15) === :ok
    finally
        _HGW.stop!(gw)
        close(sa)
        gRPCServer.stop!(sb; force = true)
    end
end

# A worker restarted between two probe rounds is never seen down, so the prober never gives it fresh
# connections (above), and the data handle may still hold one to the old process. A StageWeights call
# that fails in the transport is therefore sent once more; libcurl dropped the dead connection with
# the failure, so the second goes out on a fresh one. Before, the staging failed and the change was
# dropped, and the underlying error was not logged.
@testset "lpt_packing: a staging call that fails on a connection to a replaced worker is sent once more" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        port = _HGW.bound_ports(gw)[1]
        # m on A alone (it is configured for two GPUs) and n on B; a request for n opens the gateway's
        # data connection to B.
        asn = Dict{String, GW.Placement}("m" => [(ua, 1.0)], "n" => [(ub, 1.0)])
        @atomic aff.assignment = asn
        GW._swap_outstanding!(aff, asn)
        @test _aff_infer(port, "n").model_name == "workerB"

        # B is replaced between prober rounds: no round sees it down. The prober's own connection to it
        # is a fresh one, as after a probe that hung (the prober resets it then); the data connection
        # still goes to the old process.
        gRPCServer.stop!(sb; force = true)
        sb = _aff_server(pb, B)
        gRPCServer.start!(sb)
        @test timedwait(15) do
            try
                AInf.GRPCInferenceService.ServerReady("127.0.0.1", pb, AInf.ServerReadyRequest()).ready
            catch
                false
            end
        end === :ok
        GW.reset_probe_connection!(GW.get_clients(gw.pool, ub))
        @test !_hr_down(gw, ub)

        # An operator repack gives m its second replica, on B, staging its weights there first. The
        # first StageWeights fails on the dead connection and is sent once more, on a fresh one.
        staged = length(B.staged)
        logs = Test.TestLogger(; min_level = Logging.Info)
        Logging.with_logger(logs) do
            GW._repack!(aff, GW.poll_workers(gw.pool, [ua, ub]), gw.metrics; trigger = :operator, pool = gw.pool)
            @test timedwait(() -> aff.transition === nothing, 15) === :ok
        end
        @test length(B.staged) == staged + 1
        rep = @atomic aff.report
        @test rep.changes_planned > 0 && rep.changes_committed == rep.changes_planned && rep.changes_dropped == 0
        @test sort(first.((@atomic aff.assignment)["m"])) == sort([ua, ub])
        # The first failure is logged with its cause, which used to be discarded.
        retried = filter(r -> occursin("sending it once more", r.message), logs.logs)
        @test length(retried) == 1
        @test all(r -> !isempty(string(get(r.kwargs, :error, ""))), retried)
        @test !any(r -> occursin("staging weights for a placement change failed", r.message), logs.logs)
    finally
        _HGW.stop!(gw)
        close(sa)
        gRPCServer.stop!(sb; force = true)
    end
end

@testset "invoke_infer reports a refused connection, and nothing else, as refused" begin
    body = UInt8[0x0a, 0x01, 0x61]
    function attempt(host, port; deadline = 5)
        grpc = gRPCClient.gRPCCURL(; sticky = true)
        chan = gRPCClient.gRPCChannel(host, port; grpc = grpc, deadline = 30)
        wc = GW.WorkerClients("$host:$port", grpc, chan, 5.0, 5.0)
        refused = Ref(false)
        status = try
            GW.invoke_infer(wc, body; deadline = deadline, refused = refused)
            nothing
        catch e
            e.grpc_status
        end
        gRPCClient.grpc_shutdown(grpc)
        return status, refused[]
    end
    # Nothing listens: refused.
    @test attempt("127.0.0.1", grpc_free_port()) == (gRPCClient.GRPC_UNAVAILABLE, true)
    # Accepted, then closed at once: a reset, after the request was sent. Not refused.
    srv = Sockets.listen(Sockets.localhost, 0)
    rport = Int(Sockets.getsockname(srv)[2])
    closer = @async try
        while isopen(srv)
            c = Sockets.accept(srv)
            readavailable(c)
            close(c)
        end
    catch
    end
    @test attempt("127.0.0.1", rport) == (gRPCClient.GRPC_UNAVAILABLE, false)
    close(srv)
    # Accepted and never answered: the call's deadline. Not refused.
    srv = Sockets.listen(Sockets.localhost, 0)
    hport = Int(Sockets.getsockname(srv)[2])
    held = Any[]
    acceptor = @async try
        while isopen(srv)
            push!(held, Sockets.accept(srv))
        end
    catch
    end
    @test attempt("127.0.0.1", hport; deadline = 1) == (gRPCClient.GRPC_DEADLINE_EXCEEDED, false)
    close(srv)
    foreach(close, held)
    # A name that does not resolve. Not refused.
    @test attempt("no-such-host.invalid", 9) == (gRPCClient.GRPC_UNAVAILABLE, false)
end

# --- Only the health check takes a worker out --------------------------------------------------

_hr_on(aff, url) = any(pl -> any(wv -> first(wv) == url, pl), values(@atomic aff.assignment))

@testset "lpt_packing: a healthy worker whose control-status poll fails keeps its placement and its traffic" begin
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

        # B passes its health check every round, but its ModelControlStatus times out once and then
        # fails outright. No round repacks, n stays placed on B alone, and B keeps serving it.
        for st in (:slow, :down, :down)
            B.status = st
            _HGW._check_once(gw.prober)
            @test !_hr_down(gw, ub)
            @test (@atomic aff.report).count == rep0.count
            @test (@atomic aff.assignment)["n"] == [(ub, 1.0)]
            b0 = B.calls
            @test _aff_infer(port, "n").model_name == "workerB"
            @test _hr_call(port, _hr_shm_request("n")).model_name == "workerB"
            @test B.calls == b0 + 2
        end
        @test _hr_repacks(gw, "membership") == 0

        # A repack due for another reason meanwhile (here an operator's) plans B from its last answer:
        # B is among the workers it places on, and keeps n.
        @atomic aff.repack_requested += 1
        _HGW._check_once(gw.prober)
        rep = @atomic aff.report
        @test rep.trigger === :operator && rep.count == rep0.count + 1
        @test ub in rep.polled
        @test (@atomic aff.assignment)["n"] == [(ub, 1.0)]
        @test _aff_infer(port, "n").model_name == "workerB"

        # Losing the health check still takes it out at once.
        B.ready = false
        _HGW._check_once(gw.prober)
        rep = @atomic aff.report
        @test rep.trigger === :membership && rep.count == rep0.count + 2
        @test !_hr_on(aff, ub)
        @test _aff_infer(port, "n").model_name == "workerA"
    finally
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end

@testset "lpt_packing: a healthy worker that has not answered a control-status poll gets no placement until it does" begin
    A = AffMockWorker("workerA", ["m", "n"])
    B = AffMockWorker("workerB", ["m", "n"])
    sa = _aff_server(0, A); pa = grpc_start_ephemeral!(sa)
    sb = _aff_server(0, B); pb = grpc_start_ephemeral!(sb)
    ua, ub = "127.0.0.1:$pa", "127.0.0.1:$pb"
    gw, aff = _hr_lpt_gateway([pa, pb])
    try
        # B fails its health check, which takes it out and forgets its answers.
        B.ready = false
        _HGW._check_once(gw.prober)
        @test !_hr_on(aff, ub)
        n1 = (@atomic aff.report).count

        # B is healthy again but has not answered a poll since: it is routable, but not placed, and
        # its return is no membership change yet.
        B.ready = true
        B.status = :down
        for _ in 1:2
            _HGW._check_once(gw.prober)
            @test !_hr_down(gw, ub)
            @test (@atomic aff.report).count == n1
            @test !_hr_on(aff, ub)
        end

        # Its first answer: it joins, and the repack places m's second replica on it.
        B.status = :ok
        _HGW._check_once(gw.prober)
        rep = @atomic aff.report
        @test rep.trigger === :membership && rep.count == n1 + 1
        @test timedwait(() -> aff.transition === nothing, 15) === :ok
        @test sort(first.((@atomic aff.assignment)["m"])) == sort([ua, ub])
    finally
        _HGW.stop!(gw)
        close(sa)
        close(sb)
    end
end

# --- The held-model log ------------------------------------------------------------------------

@testset "a held model's UNAVAILABLE logs at info once per outage, then at debug" begin
    a = "127.0.0.1:7001"
    pool = GW.ClientPool(_sched_cfg("round_robin"))
    routes = GW.DiscoveredRoutes()
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics)
    st = GW.GatewayState(pool, routes, GW.RegisterGate(), metrics, refresher, GW.RoundRobinScheduler())
    dispatch() = GW._dispatch_infer(st, "m", "id", UInt8[], Int64(0))
    # The levels of the held-model lines `f` logs.
    function levels(f)
        logger = Test.TestLogger(; min_level = Logging.Debug)
        Logging.with_logger(f, logger)
        return [l.level for l in logger.logs if occursin("no ready worker serves the model", string(l.message))]
    end
    try
        GW.record_answers!(routes, Dict(a => ["m"]))
        GW.note_worker_ready!(routes, a, false)              # an outage starts: m is held
        @test levels(() -> foreach(_ -> @test(dispatch()[2] == GW.STATUS_UNAVAILABLE), 1:3)) ==
            [Logging.Info, Logging.Debug, Logging.Debug]
        # Table rebuilds during the outage keep it reported: still debug.
        GW.record_answers!(routes, Dict(a => nothing))
        @test levels(dispatch) == [Logging.Debug]
        # The outage ends and a new one starts: info again, once.
        GW.note_worker_ready!(routes, a, true)
        GW.note_worker_ready!(routes, a, false)
        @test levels(() -> (dispatch(); dispatch())) == [Logging.Info, Logging.Debug]
    finally
        GW.close_pool!(pool)
    end
end
