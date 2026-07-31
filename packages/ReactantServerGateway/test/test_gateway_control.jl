# The gateway's own scheduling control plane (GatewayControlService): the pure state helpers, then the
# four RPCs end to end against a mock fleet.
#
# The mock-worker harness (AffMockWorker, _aff_router, _aff_gatewayfile, _aff_infer) comes from
# test_lpt_packing.jl, which runtests.jl includes first.

import ProtoBuf
import gRPCClient

const _GW_CONTROL = "/reactant_control.GatewayControlService"

_ctl_call(reqT, respT, rpc, port, req; deadline = 30) =
    grpc_call(reqT, respT, rpc, port, req; service = _GW_CONTROL, deadline = deadline)

_status(port) = _ctl_call(ACtl.GetSchedulingStatusRequest, ACtl.GetSchedulingStatusResponse,
                         "GetSchedulingStatus", port, ACtl.GetSchedulingStatusRequest())

# gRPC status of a failed call, for the negative paths.
_ctl_status(f) = try
    f()
    nothing
catch e
    e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : rethrow()
end

@testset "control proto: the gateway messages round-trip" begin
    # Cheap regression on the regenerate-and-split step: a field added to the proto but lost in the
    # split would fail here rather than at runtime.
    p = ACtl.SchedulingPolicy(; rebalance_compute_seconds = 42.0, routing_fill_mode = "spread",
                              default_replicas = Int64(-1), generation = UInt64(3))
    io = IOBuffer()
    e = ProtoBuf.ProtoEncoder(io)
    ProtoBuf.encode(e, p)
    d = ProtoBuf.ProtoDecoder(IOBuffer(take!(io)))
    rt = ProtoBuf.decode(d, ACtl.SchedulingPolicy)
    @test rt.rebalance_compute_seconds == 42.0
    @test rt.routing_fill_mode == "spread"
    @test rt.default_replicas == -1 && rt.generation == 3

    req = ACtl.SetModelPlacementRequest(; name = "m", replicas = Int64(2), fill_mode = "run",
                                        fill_factor = -1.0, allow_unknown_model = true)
    io = IOBuffer()
    ProtoBuf.encode(ProtoBuf.ProtoEncoder(io), req)
    rt2 = ProtoBuf.decode(ProtoBuf.ProtoDecoder(IOBuffer(take!(io))), ACtl.SetModelPlacementRequest)
    @test rt2.name == "m" && rt2.replicas == 2 && rt2.fill_mode == "run" && rt2.allow_unknown_model
end

@testset "repack handshake: coalesces and bypasses the compute budget" begin
    # No gRPC and no workers: the handshake is pure state, so it is tested directly.
    s = _pk_state()
    @test GW.request_repack!(s) == 1
    @test GW.request_repack!(s) == 2                     # two requests, one pending repack
    completed, waited = GW.await_repack(s, 2, 0.0)
    @test !completed && waited == 0.0
    completed, waited = GW.await_repack(s, 2, 0.2)       # nothing serves it: gives up at the budget
    @test !completed && waited >= 0.15

    # The prober is the only writer of `repack_completed`, and it stores `requested`, so both
    # coalesced requests are satisfied by the single repack.
    @atomic s.repack_completed = 2
    @test GW.await_repack(s, 1, 0.0)[1]
    @test GW.await_repack(s, 2, 0.0)[1]
    @test !GW.await_repack(s, 3, 0.0)[1]

    # The hook the prober polls between rounds so a forced repack does not wait out the interval.
    @test GW.scheduler_repack_seq(s) == 2
    @test GW.scheduler_repack_seq(GW.RoundRobinScheduler()) == 0
end

@testset "gateway control: status, knobs, placement, and forced repack" begin
    models = ["alpha", "beta"]
    w0 = AffMockWorker("worker0", copy(models))
    w1 = AffMockWorker("worker1", copy(models))
    p0, p1 = grpc_free_port(), grpc_free_port()
    s0 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p0; context = w0)
    s1 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p1; context = w1)

    gw_port, admin_port = grpc_free_port(), grpc_free_port()
    gatewayfile = _aff_gatewayfile(gw_port, admin_port, [p0, p1])
    # A short probe interval keeps the bounded-wait test fast: the prober is nudged awake by a forced
    # repack, and this bounds how long the nudge can take to be noticed.
    gw = withenv("REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.2") do
        GW.serve_gateway(gatewayfile; blocking = false)
    end
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
        for _ in 1:10
            _aff_infer(gw_port, "alpha")
            _aff_infer(gw_port, "beta")
        end
        aff = gw.prober.scheduler

        @testset "GetSchedulingStatus reports the live state" begin
            st = _status(gw_port)
            @test st.mode == "lpt_packing"
            @test st.policy !== nothing && st.repack !== nothing
            @test st.policy.routing_fill_mode == "run"          # the new default
            @test st.policy.generation == 0                     # untouched: exactly as configured
            @test st.policy.default_replicas == 1
            @test Set(m.name for m in st.models) == Set(models)
            for m in st.models
                @test m.effective_replicas == 1                 # default_replicas
                @test m.fill_quantum == 8                       # max_batch from the mock
                @test m.fill_mode == "run"
                @test m.placed
                @test sum(r.routed_total for r in m.replicas) > 0
            end
            @test length(st.workers) == 2
            @test all(w -> w.ready, st.workers)
            @test all(w -> w.weight_budget_bytes == 8 * 1024^3, st.workers)
        end

        @testset "SetSchedulingPolicy is all-or-nothing" begin
            before = GW.knobs(aff)
            resp = _ctl_call(ACtl.SetSchedulingPolicyRequest, ACtl.SetSchedulingPolicyResponse,
                "SetSchedulingPolicy", gw_port,
                ACtl.SetSchedulingPolicyRequest(;
                    update_mask = ["hysteresis", "routing_policy", "compaction_interval"],
                    policy = ACtl.SchedulingPolicy(; hysteresis = 0.25, routing_policy = "fill_least",
                                                   compaction_interval = Int64(4))))
            @test resp.applied.hysteresis == 0.25
            @test resp.applied.routing_policy == "fill_least"
            @test resp.applied.compaction_interval == 4
            @test resp.applied.generation == before.generation + 1
            @test !resp.persisted                                # in-memory only, by contract
            k = GW.knobs(aff)
            @test k.hysteresis == 0.25 && k.routing_policy == :fill_least
            @test k.rebalance_compute_seconds == before.rebalance_compute_seconds   # untouched

            # A rejected value leaves every knob alone, including the valid ones in the same request.
            held = GW.knobs(aff)
            @test _ctl_status(() -> _ctl_call(ACtl.SetSchedulingPolicyRequest,
                ACtl.SetSchedulingPolicyResponse, "SetSchedulingPolicy", gw_port,
                ACtl.SetSchedulingPolicyRequest(; update_mask = ["hysteresis", "routing_fill_factor"],
                    policy = ACtl.SchedulingPolicy(; hysteresis = 1.5, routing_fill_factor = 2.0)))) ==
                gRPCClient.GRPC_INVALID_ARGUMENT
            @test GW.knobs(aff) === held                          # nothing swapped in

            for mask in (String[], ["nope"], ["max_worker_share"], ["mode"])
                @test _ctl_status(() -> _ctl_call(ACtl.SetSchedulingPolicyRequest,
                    ACtl.SetSchedulingPolicyResponse, "SetSchedulingPolicy", gw_port,
                    ACtl.SetSchedulingPolicyRequest(; update_mask = mask,
                        policy = ACtl.SchedulingPolicy(; hysteresis = 0.1)))) ==
                    gRPCClient.GRPC_INVALID_ARGUMENT
            end

            # Changing the halflife warns that existing EWMA history is not recomputed.
            resp = _ctl_call(ACtl.SetSchedulingPolicyRequest, ACtl.SetSchedulingPolicyResponse,
                "SetSchedulingPolicy", gw_port,
                ACtl.SetSchedulingPolicyRequest(; update_mask = ["ema_halflife_compute_seconds"],
                    policy = ACtl.SchedulingPolicy(; ema_halflife_compute_seconds = 5.0)))
            @test resp.applied.ema_halflife_compute_seconds == 5.0
            @test any(w -> occursin("previous halflife", w), resp.warnings)

            # ...and 0 means "track the rebalance interval", exactly as in gateway.yml, rather than
            # being rejected as out of range (the stored knob is the resolved value).
            resp = _ctl_call(ACtl.SetSchedulingPolicyRequest, ACtl.SetSchedulingPolicyResponse,
                "SetSchedulingPolicy", gw_port,
                ACtl.SetSchedulingPolicyRequest(; update_mask = ["ema_halflife_compute_seconds"],
                    policy = ACtl.SchedulingPolicy(; ema_halflife_compute_seconds = 0.0)))
            @test resp.applied.ema_halflife_compute_seconds ==
                  GW.knobs(aff).rebalance_compute_seconds
        end

        @testset "SetModelPlacement promotes a model and reports the effect" begin
            resp = _ctl_call(ACtl.SetModelPlacementRequest, ACtl.SetModelPlacementResponse,
                "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "alpha", replicas = Int64(2),
                                              fill_factor = -1.0))
            @test resp.configured_replicas == 2 && resp.effective_replicas == 2
            @test resp.fill_mode == "run" && resp.fill_quantum == 8
            @test !resp.persisted
            @test any(w -> occursin("next repack", w), resp.warnings)
            # 2 x 256 MiB against an 8 GiB budget fits, so no oversubscription warning here.
            @test !any(w -> occursin("oversubscribe", w), resp.warnings)

            # Above the fleet size: accepted and clamped, with the clamp reported rather than an error.
            resp = _ctl_call(ACtl.SetModelPlacementRequest, ACtl.SetModelPlacementResponse,
                "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "beta", replicas = Int64(5),
                                              fill_factor = -1.0))
            @test resp.effective_replicas == 2
            @test any(w -> occursin("clamp", w), resp.warnings)

            # A typo is NOT_FOUND, unless the caller is deliberately pre-seeding.
            @test _ctl_status(() -> _ctl_call(ACtl.SetModelPlacementRequest,
                ACtl.SetModelPlacementResponse, "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "nope", replicas = Int64(2),
                                              fill_factor = -1.0))) == gRPCClient.GRPC_NOT_FOUND
            resp = _ctl_call(ACtl.SetModelPlacementRequest, ACtl.SetModelPlacementResponse,
                "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "nope", replicas = Int64(2),
                                              fill_factor = -1.0, allow_unknown_model = true))
            @test resp.configured_replicas == 2
            @test any(w -> occursin("not currently serve", w) || occursin("no worker", w), resp.warnings)

            # The per-model fill fields are independent of the replica count, and selecting the legacy
            # basis returns the parking warning.
            resp = _ctl_call(ACtl.SetModelPlacementRequest, ACtl.SetModelPlacementResponse,
                "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "alpha", replicas = Int64(2),
                                              fill_mode = "inflight", fill_factor = 0.5))
            @test resp.fill_mode == "inflight" && resp.fill_quantum == 4
            @test any(w -> occursin("ONE replica", w), resp.warnings)
            @test GW.knobs(aff).model_overrides["alpha"].fill_factor == 0.5

            # ...and clearing puts it back on the fleet default.
            resp = _ctl_call(ACtl.SetModelPlacementRequest, ACtl.SetModelPlacementResponse,
                "SetModelPlacement", gw_port,
                ACtl.SetModelPlacementRequest(; name = "alpha", replicas = Int64(2),
                                              fill_mode = "inherit", fill_factor = 0.0))
            @test resp.fill_mode == "run" && resp.fill_quantum == 8
        end

        @testset "Repack applies a promotion within the bounded wait" begin
            before = aff.last_rebalance
            # The steady-state budget is enormous, so only the forced path can fire this repack.
            GW.set_knobs!(aff; rebalance_compute_seconds = 1.0e9,
                          first_rebalance_compute_seconds = 0.0)
            resp = _ctl_call(ACtl.RepackRequest, ACtl.RepackResponse, "Repack", gw_port,
                             ACtl.RepackRequest(; wait_seconds = 10.0); deadline = 40)
            @test resp.completed
            @test resp.sequence == 1 && resp.completed_sequence >= 1
            @test resp.waited_seconds < 10.0
            @test aff.last_rebalance > before
            @test resp.repack !== nothing && resp.repack.last_trigger == "operator"
            # A forced repack must not consume the first-traffic-repack budget.
            @test aff.did_first_tick_repack == false

            # The promotion set above is now installed: alpha is on both workers.
            @test length((@atomic aff.assignment)["alpha"]) == 2
            st = _status(gw_port)
            alpha = only(m for m in st.models if m.name == "alpha")
            @test alpha.effective_replicas == 2
            @test length(alpha.replicas) == 2
            @test st.repack.operator_repacks_completed >= 1

            # wait_seconds = 0 queues without waiting.
            resp = _ctl_call(ACtl.RepackRequest, ACtl.RepackResponse, "Repack", gw_port,
                             ACtl.RepackRequest(; wait_seconds = 0.0))
            @test resp.sequence == 2 && !resp.completed
            @test _ctl_status(() -> _ctl_call(ACtl.RepackRequest, ACtl.RepackResponse, "Repack",
                gw_port, ACtl.RepackRequest(; wait_seconds = -1.0))) ==
                gRPCClient.GRPC_INVALID_ARGUMENT
        end

        @testset "the worker-facing ControlService still answers on the same router" begin
            # Two services on one router must stay path-namespaced: CompactMemory is the gateway's
            # pre-existing control RPC and fans out to every worker.
            resp = grpc_call(ACtl.CompactMemoryRequest, ACtl.CompactMemoryResponse, "CompactMemory",
                gw_port, ACtl.CompactMemoryRequest(; reload_models = String[]);
                service = "/reactant_control.ControlService")
            @test resp.reloaded_models == 0
        end
    finally
        GW.stop!(gw)
        close(s0)
        close(s1)
        rm(gatewayfile; force = true)
    end
end

@testset "gateway control: the mutators require lpt_packing" begin
    w0 = AffMockWorker("worker0", ["alpha"])
    p0 = grpc_free_port()
    s0 = gRPCServer.serve!(_aff_router(), "127.0.0.1", p0; context = w0)
    gw_port, admin_port = grpc_free_port(), grpc_free_port()
    path = tempname() * ".yaml"
    write(path, """
    listen:
      grpc: "127.0.0.1:$gw_port"
      metrics: "127.0.0.1:$admin_port"
    scheduling:
      mode: round_robin
    endpoints:
      - "127.0.0.1:$p0"
    """)
    gw = GW.serve_gateway(path; blocking = false)
    try
        for _ in 1:40
            try
                _aff_infer(gw_port, "alpha")
                break
            catch
                sleep(0.1)
            end
        end
        # The read answers in every mode: an operator's first call is a probe of what is running, and
        # the mode with no policy block beats an error that sends them to read gateway.yml.
        st = _status(gw_port)
        @test st.mode == "round_robin"
        @test st.policy === nothing && st.repack === nothing
        @test isempty(st.models)

        @test _ctl_status(() -> _ctl_call(ACtl.RepackRequest, ACtl.RepackResponse, "Repack",
            gw_port, ACtl.RepackRequest(; wait_seconds = 0.0))) ==
            gRPCClient.GRPC_FAILED_PRECONDITION
        @test _ctl_status(() -> _ctl_call(ACtl.SetModelPlacementRequest,
            ACtl.SetModelPlacementResponse, "SetModelPlacement", gw_port,
            ACtl.SetModelPlacementRequest(; name = "alpha", replicas = Int64(2),
                                          fill_factor = -1.0))) ==
            gRPCClient.GRPC_FAILED_PRECONDITION
        @test _ctl_status(() -> _ctl_call(ACtl.SetSchedulingPolicyRequest,
            ACtl.SetSchedulingPolicyResponse, "SetSchedulingPolicy", gw_port,
            ACtl.SetSchedulingPolicyRequest(; update_mask = ["hysteresis"],
                policy = ACtl.SchedulingPolicy(; hysteresis = 0.2)))) ==
            gRPCClient.GRPC_FAILED_PRECONDITION
    finally
        GW.stop!(gw)
        close(s0)
        rm(path; force = true)
    end
end
