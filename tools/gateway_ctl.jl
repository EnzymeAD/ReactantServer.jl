#!/usr/bin/env julia
#
# Operator CLI for the gateway's scheduling control plane (GatewayControlService). Read the live
# placement and knobs, promote a model onto more GPUs or change how its requests use them, retune the
# repack cadence, and force a repack.
#
# Run against the gateway's gRPC port (listen.grpc, default 8001), not the metrics port:
#
#   julia --project=packages/ReactantServerGateway tools/gateway_ctl.jl [--gateway host:port] <cmd>
#
#     status                                          the whole scheduling state
#     repack [--wait SECONDS]                         repack now (default waits 30s for it to land)
#     set-replicas MODEL <n|all|default> [--fill-mode run|spread|inflight|inherit]
#                                       [--fill-factor F] [--allow-unknown]
#     set-policy KEY=VALUE [KEY=VALUE ...]            any SchedulingPolicy field (see --help)
#
# Nothing set here is persisted: a gateway restart reverts every override to gateway.yml plus the
# environment. Every change is logged by the gateway, which is the only durable record of it.
#
# Zero-install alternative, if you would rather not run Julia:
#   grpcurl -plaintext -proto proto_src/reactant_control_v1.proto \
#       -d '{}' HOST:PORT reactant_control.GatewayControlService/GetSchedulingStatus

using Printf
using ReactantServerGateway
using ReactantServerCore.control
import gRPCClient

const GW = ReactantServerGateway

const POLICY_KEYS = ["rebalance_compute_seconds", "first_rebalance_compute_seconds", "hysteresis",
                     "ema_halflife_compute_seconds", "default_replicas", "routing_policy", "work_basis",
                     "routing_fill_factor", "routing_fill_mode", "compaction_mode",
                     "compaction_interval", "forbid_memory_oversubscription"]

usage() = print("""
Usage: gateway_ctl.jl [--gateway HOST:PORT] COMMAND

Commands:
  status
  repack [--wait SECONDS]
  set-replicas MODEL <n|all|default> [--fill-mode MODE] [--fill-factor F] [--allow-unknown]
  set-policy KEY=VALUE ...

set-policy keys:
  $(join(POLICY_KEYS, "\n  "))

Replica counts: a positive integer, "all" (every ready worker), or "default" (clear the override).
Fill modes: run (default), spread, inflight (parks a model on one replica: see the docs), or
            inherit (clear a per-model override).
""")

# --- transport ---------------------------------------------------------------------------------

function split_target(t)
    i = findlast(==(':'), t)
    i === nothing && error("--gateway must be host:port, got '$t'")
    return String(t[1:(i - 1)]), parse(Int, t[(i + 1):end])
end

# One client per call, all sharing the global libcurl handle: this is a short-lived CLI. The deadline
# must exceed any bounded wait the gateway is asked to do, or we would give up before it answers. The
# handle is shut down once in `main`, not per call (a service client is not itself closeable).
function call(ctor, host, port, req; deadline = 20)
    client = ctor(host, port; deadline = deadline)
    return gRPCClient.grpc_sync_request(client, req)
end

# --- formatting --------------------------------------------------------------------------------

fmt_replicas(n) = n == -1 ? "all" : (n == 0 ? "-" : string(n))
fmt_bytes(n) = n <= 0 ? "-" : Base.format_bytes(n)

function print_warnings(ws; prefix = "warning")
    for w in ws
        println("  $prefix: $w")
    end
end

function print_status(resp)
    println("mode: $(resp.mode)")
    if resp.policy === nothing
        println("(this scheduling mode has no runtime knobs)")
    else
        p = resp.policy
        println("\nknobs" * (p.generation == 0 ? " (as configured)" : " (generation $(p.generation), tuned at runtime)"))
        @printf("  rebalance_compute_seconds        %s\n", p.rebalance_compute_seconds)
        @printf("  first_rebalance_compute_seconds  %s\n", p.first_rebalance_compute_seconds)
        @printf("  hysteresis                       %s\n", p.hysteresis)
        @printf("  ema_halflife_compute_seconds     %s\n", p.ema_halflife_compute_seconds)
        @printf("  default_replicas                 %s\n", fmt_replicas(p.default_replicas))
        @printf("  routing_policy                   %s\n", p.routing_policy)
        @printf("  work_basis                       %s\n", p.work_basis)
        @printf("  routing_fill_mode                %s\n", p.routing_fill_mode)
        @printf("  routing_fill_factor              %s\n", p.routing_fill_factor)
        @printf("  compaction                       %s every %d repack(s)\n", p.compaction_mode, p.compaction_interval)
        @printf("  forbid_memory_oversubscription   %s\n", p.forbid_memory_oversubscription)
    end
    r = resp.repack
    if r !== nothing
        println("\nrepacks: $(r.repack_count) total, last trigger '$(r.last_trigger)' " *
                "($(round(r.last_models_placed)) models, $(r.last_models_moved) moved)")
        @printf("  compute since last repack        %.1f of %.1f GPU-seconds\n",
                r.compute_accumulated_seconds, r.active_threshold_seconds)
        @printf("  first-repack budget armed        %s\n", !r.first_tick_repack_done)
        @printf("  operator repacks                 %d requested, %d served\n",
                r.operator_repacks_requested, r.operator_repacks_completed)
    end
    if !isempty(resp.models)
        println("\nmodels")
        @printf("  %-28s %5s %5s %-9s %5s %9s %9s  %s\n",
                "name", "cfg", "eff", "fill", "quant", "util", "cost",
                "placement (requests routed / items in flight)")
        for m in resp.models
            # requests routed / items in flight: the two are deliberately different units.
            place = join(["$(r.worker) $(r.routed_total)req/$(r.outstanding)it" for r in m.replicas], "  ")
            flags = string(m.placed ? "" : " [unplaced]", m.drifted ? " [drifted]" : "")
            @printf("  %-28s %5s %5d %-9s %5d %9.4f %9.4f  %s%s\n",
                    m.name, fmt_replicas(m.configured_replicas), m.effective_replicas,
                    m.fill_mode, m.fill_quantum, m.utilization, m.cost_seconds, place, flags)
        end
    end
    if !isempty(resp.workers)
        println("\nworkers")
        @printf("  %-24s %6s %7s %11s %11s %11s\n",
                "worker", "ready", "models", "in-flight", "weights", "budget")
        for w in resp.workers
            @printf("  %-24s %6s %7d %11.3f %11s %11s%s\n",
                    w.worker, w.ready, w.models_placed, w.inflight_compute,
                    fmt_bytes(w.assigned_weight_bytes), fmt_bytes(w.weight_budget_bytes),
                    w.oversubscribed ? "  OVERSUBSCRIBED" : "")
        end
    end
    isempty(resp.warnings) || (println(); print_warnings(resp.warnings))
end

# --- commands ----------------------------------------------------------------------------------

parse_replicas(v) = v == "all" ? Int64(-1) : (v in ("default", "-") ? Int64(0) : Int64(parse(Int, v)))

# `KEY=VALUE` for a SchedulingPolicy field. The mask is built from the keys the caller passed, so this
# script needs no knowledge of which knobs exist beyond their value types.
function build_policy(pairs)
    mask = String[]
    kw = Dict{Symbol,Any}()
    for p in pairs
        k, _, v = partition_kv(p)
        k in POLICY_KEYS || error("unknown policy key '$k'; accepted: $(join(POLICY_KEYS, ", "))")
        push!(mask, k)
        kw[Symbol(k)] = coerce_policy_value(k, v)
    end
    return mask, SchedulingPolicy(; kw...)
end

function partition_kv(p)
    i = findfirst(==('='), p)
    i === nothing && error("expected KEY=VALUE, got '$p'")
    return String(p[1:(i - 1)]), '=', String(p[(i + 1):end])
end

function coerce_policy_value(k, v)
    k in ("routing_policy", "work_basis", "routing_fill_mode", "compaction_mode") && return v
    k == "forbid_memory_oversubscription" && return parse(Bool, v)
    k == "default_replicas" && return parse_replicas(v)
    k == "compaction_interval" && return Int64(parse(Int, v))
    return parse(Float64, v)
end

function main(args)
    host, port = "127.0.0.1", 8001
    rest = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--gateway"
            host, port = split_target(args[i + 1]); i += 2
        elseif a in ("-h", "--help")
            usage(); return 0
        else
            push!(rest, a); i += 1
        end
    end
    isempty(rest) && (usage(); return 2)
    # Release the shared libcurl handle on every exit path: an open handle keeps a libuv timer alive,
    # which can hold the process open past the last response. A rejected request is the operator's
    # own input error, so report the gateway's message rather than a Julia stacktrace.
    try
        return dispatch(popfirst!(rest), rest, host, port)
    catch e
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        println(stderr, "error: $(e.message)")
        println(stderr, "  ($(e.grpc_status) from $host:$port)")
        return 1
    finally
        gRPCClient.grpc_shutdown()
    end
end

function dispatch(cmd, rest, host, port)
    if cmd == "status"
        print_status(call(GatewayControlService_GetSchedulingStatus_Client, host, port,
                          GetSchedulingStatusRequest()))
    elseif cmd == "repack"
        wait = 30.0
        j = findfirst(==("--wait"), rest)
        j === nothing || (wait = parse(Float64, rest[j + 1]))
        resp = call(GatewayControlService_Repack_Client, host, port,
                    RepackRequest(; wait_seconds = wait); deadline = ceil(Int, wait) + 30)
        println(resp.completed ?
                "repack $(resp.sequence) completed in $(round(resp.waited_seconds; digits = 2))s" :
                "repack $(resp.sequence) queued (not observed within $(round(resp.waited_seconds; digits = 2))s)")
        r = resp.repack
        r === nothing || println("  now $(r.repack_count) repack(s) total, last trigger '$(r.last_trigger)', $(r.last_models_moved) model(s) moved")
        print_warnings(resp.warnings)
    elseif cmd == "set-replicas"
        length(rest) >= 2 || (usage(); return 2)
        model = popfirst!(rest)
        replicas = parse_replicas(popfirst!(rest))
        fill_mode = ""
        fill_factor = -1.0
        allow_unknown = false
        while !isempty(rest)
            a = popfirst!(rest)
            a == "--fill-mode" && (fill_mode = popfirst!(rest); continue)
            a == "--fill-factor" && (fill_factor = parse(Float64, popfirst!(rest)); continue)
            a == "--allow-unknown" && (allow_unknown = true; continue)
            error("unexpected argument '$a'")
        end
        resp = call(GatewayControlService_SetModelPlacement_Client, host, port,
                    SetModelPlacementRequest(; name = model, replicas = replicas,
                                             fill_mode = fill_mode, fill_factor = fill_factor,
                                             allow_unknown_model = allow_unknown))
        println("$(resp.name): replicas $(fmt_replicas(resp.configured_replicas)) " *
                "(effective $(resp.effective_replicas)), fill $(resp.fill_mode) quantum $(resp.fill_quantum)")
        print_warnings(resp.warnings)
    elseif cmd == "set-policy"
        isempty(rest) && (usage(); return 2)
        mask, policy = build_policy(rest)
        resp = call(GatewayControlService_SetSchedulingPolicy_Client, host, port,
                    SetSchedulingPolicyRequest(; update_mask = mask, policy = policy))
        println("applied $(join(mask, ", ")) (generation $(resp.applied.generation))")
        print_warnings(resp.warnings)
    else
        println("unknown command '$cmd'")
        usage()
        return 2
    end
    return 0
end

# The parentheses matter: a bare `@__FILE__` would slurp the rest of the line as macro arguments.
abspath(PROGRAM_FILE) == (@__FILE__) && exit(main(ARGS))
