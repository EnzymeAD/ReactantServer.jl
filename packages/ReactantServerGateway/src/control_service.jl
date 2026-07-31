# The gateway's own control plane: `GatewayControlService` (see proto_src/reactant_control_v1.proto).
#
# These RPCs are answered by the gateway process, not forwarded to a worker, and they drive the live
# lpt_packing scheduling state: read the placement and knobs, retune the knobs, promote a model onto
# more GPUs or change how it uses them, and force a repack. Workers do not implement this service.
#
# Nothing is persisted. A gateway restart reverts every override to gateway.yml plus the environment,
# and restart is not hypothetical: the gateway deliberately exits on a wedged client stack expecting a
# supervisor to bring it back. Every mutation is therefore logged at info level with its before and
# after values, because the log is the only record that a change happened, and each mutating response
# carries `persisted = false` plus a `generation` an operator can read back.

const _CTRL_GW = ReactantServerCore.control

# Bounded wait for a forced repack. The prober is nudged awake as soon as a request lands (see
# `scheduler_repack_seq` in scheduler.jl), so the wait normally resolves in well under a second; the
# cap exists so a wedged poll cannot hold a control call open indefinitely.
const _REPACK_WAIT_POLL_SECONDS = 0.05
const _REPACK_WAIT_MAX_SECONDS = 120.0

# --- error mapping ----------------------------------------------------------------------------

# Run a control action, mapping a rejected value to INVALID_ARGUMENT and anything unexpected to
# INTERNAL. Mirrors the worker's `_as_control` (packages/ReactantServer/src/transport/control_grpc.jl).
function _as_gateway_control(f)
    try
        return f()
    catch e
        e isa gRPCServer.gRPCServiceCallException && rethrow()
        e isa ReactantServerCore.ConfigError &&
            throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT, e.msg))
        @error "gateway control: handler failed" exception = (e, catch_backtrace())
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL,
            "gateway control call failed: $(sprint(showerror, e))"))
    end
end

_invalid_arg(msg) =
    throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT, msg))

# The scheduling mode name, derived from the scheduler object actually routing rather than from the
# config, so a reported mode can never disagree with reality.
_mode_name(::GatewayScheduler) = "unknown"
_mode_name(::RoundRobinScheduler) = "round_robin"
_mode_name(::LeastOutstandingScheduler) = "least_outstanding"
_mode_name(::LptPackingState) = "lpt_packing"

# The mutators need the packing scheduler; the read is answered in every mode. Dispatched on the type
# rather than a string compare on the config.
_require_packing(s::GatewayScheduler) =
    throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_FAILED_PRECONDITION,
        "gateway scheduling mode is '$(_mode_name(s))'; the scheduler control RPCs require lpt_packing"))
_require_packing(s::LptPackingState) = s

# --- the forced-repack handshake --------------------------------------------------------------

"""
    request_repack!(s::LptPackingState) -> Int

Ask the prober to repack on its next round regardless of the accumulated-compute budget, returning
the sequence number assigned to this request. Requests that arrive inside one tick window coalesce
into a single repack, and every one of them is satisfied by it.
"""
request_repack!(s::LptPackingState) = @atomic s.repack_requested += 1

"""
    await_repack(s::LptPackingState, seq, budget) -> (completed, waited)

Wait up to `budget` seconds for the prober to have served repack `seq`. Polls and yields rather than
blocking, so it never starves the prober task it is waiting on.
"""
function await_repack(s::LptPackingState, seq::Integer, budget::Real)
    (@atomic s.repack_completed) >= seq && return (true, 0.0)
    budget <= 0 && return (false, 0.0)
    t0 = time()
    while time() - t0 < budget
        sleep(_REPACK_WAIT_POLL_SECONDS)
        (@atomic s.repack_completed) >= seq && return (true, time() - t0)
    end
    return ((@atomic s.repack_completed) >= seq, time() - t0)
end

# --- wire conversions -------------------------------------------------------------------------

# Replica counts on the wire: > 0 a count, -1 every ready worker, 0 no override. `REPLICAS_ALL` is
# typemax(Int) internally, which does not survive a round trip as a count, hence the sentinel.
_replicas_to_wire(n::Integer) = n == REPLICAS_ALL ? Int64(-1) : Int64(n)

function _replicas_from_wire(v::Integer, key::AbstractString)
    v == 0 && return 0                       # clear the override
    v == -1 && return REPLICAS_ALL
    v < -1 && throw(ReactantServerCore.ConfigError(
        "$key must be a positive count, -1 for all ready workers, or 0 to clear the override, got $v"))
    return Int(v)
end

_policy_to_wire(k::PackingKnobs) = _CTRL_GW.SchedulingPolicy(;
    rebalance_compute_seconds = k.rebalance_compute_seconds,
    first_rebalance_compute_seconds = k.first_rebalance_compute_seconds,
    hysteresis = k.hysteresis,
    ema_halflife_compute_seconds = k.ema_halflife_compute,
    default_replicas = _replicas_to_wire(k.default_replicas),
    routing_policy = String(k.routing_policy),
    routing_fill_factor = k.routing_fill_factor,
    routing_fill_mode = String(k.routing_fill_mode),
    compaction_mode = String(k.compaction_mode),
    compaction_interval = Int64(k.compaction_interval),
    forbid_memory_oversubscription = k.forbid_memory_oversubscription,
    generation = UInt64(k.generation))

# `update_mask` entry -> the `PackingKnobs` field it writes and the value read off the wire message.
# Startup-only settings are absent on purpose: naming one is an error, not a silent no-op.
const _POLICY_WIRE_FIELDS = Dict{String,Symbol}(
    "rebalance_compute_seconds" => :rebalance_compute_seconds,
    "first_rebalance_compute_seconds" => :first_rebalance_compute_seconds,
    "hysteresis" => :hysteresis,
    "ema_halflife_compute_seconds" => :ema_halflife_compute,
    "default_replicas" => :default_replicas,
    "routing_policy" => :routing_policy,
    "routing_fill_factor" => :routing_fill_factor,
    "routing_fill_mode" => :routing_fill_mode,
    "compaction_mode" => :compaction_mode,
    "compaction_interval" => :compaction_interval,
    "forbid_memory_oversubscription" => :forbid_memory_oversubscription,
)

# Settings that exist but cannot change without a restart, named so the error can say why rather than
# reporting them as unknown.
const _STARTUP_ONLY_FIELDS = Dict{String,String}(
    "mode" => "the scheduler is a distinct type chosen at startup, and lpt_packing has hard startup preconditions",
    "scheduling_mode" => "the scheduler is a distinct type chosen at startup, and lpt_packing has hard startup preconditions",
    "max_worker_share" => "advisory only; no code reads it",
    "endpoints" => "the worker endpoint list is bound into the client pool at startup",
    "listen" => "listen addresses are bound at startup",
)

function _updates_from_mask(mask, policy)
    isempty(mask) && _invalid_arg("update_mask is empty; name at least one SchedulingPolicy field to apply")
    policy === nothing && _invalid_arg("policy is unset but update_mask names $(length(mask)) field(s) to apply")
    updates = Dict{Symbol,Any}()
    for name in mask
        n = String(name)
        haskey(_STARTUP_ONLY_FIELDS, n) &&
            _invalid_arg("'$n' cannot be changed at runtime: $(_STARTUP_ONLY_FIELDS[n]). Change it in gateway.yml and restart.")
        field = get(_POLICY_WIRE_FIELDS, n, nothing)
        field === nothing &&
            _invalid_arg("unknown scheduling knob '$n'; accepted: $(join(sort(collect(keys(_POLICY_WIRE_FIELDS))), ", "))")
        raw = getfield(policy, Symbol(n))
        # `default_replicas` needs the wire sentinel decoded before the shared validator sees it.
        updates[field] = field === :default_replicas ?
                         _replicas_from_wire(raw, "default_replicas") : raw
    end
    return updates
end

# `ema_halflife_compute_seconds: 0` means "track the rebalance interval" in gateway.yml, and it is
# resolved once at startup. Honor the same spelling over the wire (the stored knob is the resolved
# value and must stay positive), resolving against the rebalance budget this same request installs
# when it sets both.
function _resolve_halflife!(updates::Dict{Symbol,Any}, cur::PackingKnobs)
    haskey(updates, :ema_halflife_compute) || return updates
    updates[:ema_halflife_compute] == 0 || return updates
    updates[:ema_halflife_compute] =
        get(updates, :rebalance_compute_seconds, cur.rebalance_compute_seconds)
    return updates
end

# --- status -----------------------------------------------------------------------------------

function _model_rows(s::LptPackingState, k::PackingKnobs, rep::RepackReport)
    assignment = @atomic s.assignment
    out_snap = @atomic s.outstanding
    routed_snap = @atomic s.routed
    plan = @atomic s.fill_plan
    max_batch = @atomic s.max_batch
    rows = _CTRL_GW.ModelSchedulingStatus[]
    # Every model the packer knows about: placed models plus any with an override that has not landed.
    names = union(Set(keys(assignment)), Set(keys(k.model_overrides)), Set(keys(max_batch)))
    for m in sort(collect(names))
        placement = get(assignment, m, nothing)
        reps = _CTRL_GW.ReplicaPlacement[]
        if placement !== nothing
            for (w, weight) in placement
                a = get(out_snap, (m, w), nothing)
                r = get(routed_snap, (m, w), nothing)
                push!(reps, _CTRL_GW.ReplicaPlacement(; worker = w, weight = weight,
                                                      outstanding = Int64(a === nothing ? 0 : a[]),
                                                      routed_total = Int64(r === nothing ? 0 : r[])))
            end
        end
        mk = get(k.model_overrides, m, nothing)
        p = get(plan, m, nothing)
        push!(rows, _CTRL_GW.ModelSchedulingStatus(;
            name = m,
            configured_replicas = _replicas_to_wire(mk === nothing ? 0 : mk.replicas),
            effective_replicas = Int64(placement === nothing ? 0 : length(placement)),
            replicas = reps,
            utilization = get(rep.utilization, m, 0.0),
            arrival_rate = get(rep.rate, m, 0.0),
            cost_seconds = get(rep.cost, m, 0.0),
            max_batch = Int64(get(max_batch, m, 0)),
            fill_mode = String(p === nothing ? :unknown : p.mode),
            fill_quantum = Int64(p === nothing ? 0 : p.quantum),
            weight_nbytes = Int64(round(get(rep.mem, m, 0.0))),
            placed = placement !== nothing,
            drifted = m in rep.drifted))
    end
    return rows
end

function _worker_rows(s::LptPackingState, rep::RepackReport, tick::TickReport)
    assignment = @atomic s.assignment
    wload = @atomic s.worker_load
    # Readiness comes from the most recent evidence: the last prober tick, or, in the window between
    # the startup placement and the first tick, the workers that answered the startup poll. Otherwise
    # a status call right after startup would report an entire healthy fleet as not ready.
    ready = Set(tick.unix_seconds > 0 ? tick.ready_workers : rep.polled)
    assigned = Dict{String,Float64}()
    placed = Dict{String,Int}()
    for (m, placement) in assignment, (w, _) in placement
        assigned[w] = get(assigned, w, 0.0) + get(rep.mem, m, 0.0)
        placed[w] = get(placed, w, 0) + 1
    end
    rows = _CTRL_GW.WorkerSchedulingStatus[]
    for w in sort(collect(union(Set(keys(rep.mem_cap)), Set(keys(assigned)), ready)))
        cap = get(rep.mem_cap, w, 0.0)
        bytes = get(assigned, w, 0.0)
        a = get(wload, w, nothing)
        push!(rows, _CTRL_GW.WorkerSchedulingStatus(;
            worker = w, ready = w in ready, models_placed = Int64(get(placed, w, 0)),
            inflight_compute = a === nothing ? 0.0 : a[],
            assigned_weight_bytes = Int64(round(bytes)),
            weight_budget_bytes = Int64(round(cap)),
            oversubscribed = cap > 0 && bytes > cap))
    end
    return rows
end

_repack_status(s::LptPackingState) = begin
    rep = @atomic s.report
    tick = @atomic s.tick
    _CTRL_GW.RepackStatus(;
        last_repack_unix_seconds = rep.unix_seconds,
        last_repack_wall_elapsed_seconds = rep.wall_elapsed,
        last_repack_compute_seconds = rep.compute_elapsed,
        last_trigger = String(rep.trigger),
        last_models_placed = Int64(rep.models_placed),
        last_models_moved = Int64(rep.models_moved),
        repack_count = UInt64(rep.count),
        compute_accumulated_seconds = tick.compute_accum,
        active_threshold_seconds = tick.active_threshold,
        first_tick_repack_done = tick.first_tick_repack_done,
        last_tick_unix_seconds = tick.unix_seconds,
        repacks_since_compaction = Int64(s.repacks_since_compact),
        operator_repacks_requested = UInt64(@atomic s.repack_requested),
        operator_repacks_completed = UInt64(@atomic s.repack_completed))
end

# GetSchedulingStatus answers in every mode: an operator's first call is usually a probe of what is
# actually running, and reporting the mode with no policy block beats an error that sends them to read
# gateway.yml.
function _gw_scheduling_status(::_CTRL_GW.GetSchedulingStatusRequest, st::GatewayState)
    sched = st.scheduler
    sched isa LptPackingState || return _CTRL_GW.GetSchedulingStatusResponse(; mode = _mode_name(sched))
    return _as_gateway_control() do
        k = knobs(sched)
        rep = @atomic sched.report
        tick = @atomic sched.tick
        warnings = String[]
        k.routing_fill_mode === :inflight && push!(warnings, _INFLIGHT_PARKING_WARNING)
        for (m, mk) in k.model_overrides
            mk.fill_mode === :inflight && push!(warnings, "model '$m': $(_INFLIGHT_PARKING_WARNING)")
        end
        rep.count == 0 && push!(warnings, "no repack has run yet; per-model demand figures are unset")
        _CTRL_GW.GetSchedulingStatusResponse(; mode = "lpt_packing", policy = _policy_to_wire(k),
            repack = _repack_status(sched), models = _model_rows(sched, k, rep),
            workers = _worker_rows(sched, rep, tick), warnings = sort!(warnings))
    end
end

# --- mutators ---------------------------------------------------------------------------------

# Warn about the knob changes whose effect is easy to misread, rather than silently applying them.
function _policy_warnings(s::LptPackingState, before::PackingKnobs, after::PackingKnobs)
    w = String[]
    tick = @atomic s.tick
    if after.rebalance_compute_seconds != before.rebalance_compute_seconds &&
       tick.compute_accum >= after.rebalance_compute_seconds
        push!(w, "the fleet has already accumulated $(round(tick.compute_accum; digits = 1)) GPU-seconds, at or above the new $(after.rebalance_compute_seconds) second budget; a repack will fire on the next prober tick")
    end
    after.ema_halflife_compute != before.ema_halflife_compute &&
        push!(w, "existing arrival-rate and cost history was folded at the previous halflife ($(before.ema_halflife_compute)s) and is not recomputed; only future decay changes")
    after.first_rebalance_compute_seconds != before.first_rebalance_compute_seconds &&
        tick.first_tick_repack_done &&
        push!(w, "first_rebalance_compute_seconds no longer applies: the first traffic-driven repack has already run")
    after.routing_fill_mode === :inflight && before.routing_fill_mode !== :inflight &&
        push!(w, _INFLIGHT_PARKING_WARNING)
    return w
end

function _gw_set_scheduling_policy(req::_CTRL_GW.SetSchedulingPolicyRequest, st::GatewayState)
    s = _require_packing(st.scheduler)
    return _as_gateway_control() do
        before = knobs(s)
        updates = _resolve_halflife!(_updates_from_mask(req.update_mask, req.policy), before)
        after = set_knobs!(s; updates...)
        warnings = _policy_warnings(s, before, after)
        @info "gateway control: scheduling policy updated" fields = sort!(String[String(f) for f in keys(updates)]) generation = after.generation before = _knob_digest(before) after = _knob_digest(after)
        _CTRL_GW.SetSchedulingPolicyResponse(; applied = _policy_to_wire(after),
                                            warnings = warnings, persisted = false)
    end
end

# A compact one-line rendering of the knobs for the audit log (the full struct is noisy and the log is
# the only record a change happened).
_knob_digest(k::PackingKnobs) =
    "rebalance=$(k.rebalance_compute_seconds) first=$(k.first_rebalance_compute_seconds) hyst=$(k.hysteresis) halflife=$(k.ema_halflife_compute) replicas=$(k.default_replicas == REPLICAS_ALL ? "all" : k.default_replicas) policy=$(k.routing_policy) fill=$(k.routing_fill_mode)x$(k.routing_fill_factor) compaction=$(k.compaction_mode)/$(k.compaction_interval) noover=$(k.forbid_memory_oversubscription) gen=$(k.generation)"

# Project the memory effect of a placement change: the model's footprint lands on every replica, so
# promoting it can push a worker past its on-demand budget. The packer degrades gracefully (its LRU
# churns) and the operator may be doing this deliberately, so this warns and never rejects. The
# projection is an estimate: which workers the packer actually picks depends on the next poll.
function _placement_warnings(s::LptPackingState, model::AbstractString, k::Int)
    rep = @atomic s.report
    footprint = get(rep.mem, String(model), 0.0)
    (footprint <= 0 || isempty(rep.mem_cap)) && return String[]
    assignment = @atomic s.assignment
    assigned = Dict{String,Float64}(w => 0.0 for w in keys(rep.mem_cap))
    for (m, placement) in assignment, (w, _) in placement
        m == String(model) && continue                 # its current homes are re-chosen by the repack
        haskey(assigned, w) && (assigned[w] += get(rep.mem, m, 0.0))
    end
    # The packer prefers the least-loaded workers, so project onto those.
    targets = sort(collect(keys(assigned)); by = w -> (assigned[w], w))[1:min(k, length(assigned))]
    out = String[]
    for w in targets
        cap = get(rep.mem_cap, w, 0.0)
        cap > 0 || continue
        total = assigned[w] + footprint
        total > cap && push!(out,
            "estimated: placing '$model' on $k worker(s) would raise $w to $(Base.format_bytes(round(Int, total))) of weights against a $(Base.format_bytes(round(Int, cap))) budget, so the packer will oversubscribe and that worker's LRU will churn (the exact workers depend on the next poll)")
    end
    return out
end

function _gw_set_model_placement(req::_CTRL_GW.SetModelPlacementRequest, st::GatewayState)
    s = _require_packing(st.scheduler)
    return _as_gateway_control() do
        name = String(req.name)
        isempty(name) && _invalid_arg("name is empty")
        if !req.allow_unknown_model && !has_model(st.routes, name)
            throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND,
                "no worker currently serves model '$name'; set allow_unknown_model to pre-seed an override for a model that is about to be loaded"))
        end
        replicas = _replicas_from_wire(req.replicas, "replicas")
        cur = get(knobs(s).model_overrides, name, ModelKnobs())
        # Independent fields: "" / negative mean "leave unchanged", so one call can change the replica
        # count, the fill mode, the run length, or any combination.
        fill_mode = isempty(req.fill_mode) ? cur.fill_mode :
                    _parse_fill_mode(req.fill_mode; allow_inherit = true)
        fill_factor = req.fill_factor < 0 ? cur.fill_factor :
                      (req.fill_factor == 0 ? 0.0 :
                       _ck_positive("fill_factor", req.fill_factor))
        next = ModelKnobs(replicas, fill_mode, fill_factor)
        overrides = copy(knobs(s).model_overrides)      # copy-on-write: never mutate the live dict
        if next == ModelKnobs()
            delete!(overrides, name)
        else
            overrides[name] = next
        end
        after = set_knobs!(s; model_overrides = overrides)
        # Re-resolve now so the response reports the rule that will apply, and so a promoted model
        # routes by its new mode immediately rather than only after the next poll.
        plan = _publish_fill_plan!(s, after, @atomic(s.max_batch))
        p = get(plan, name, resolve_fill_plan(after, name, 0))

        nready = length((@atomic s.tick).ready_workers)
        effective = replicas == 0 ? after.default_replicas : replicas
        effective = nready > 0 ? clamp(effective, 1, nready) : effective
        warnings = String[]
        nready > 0 && replicas != 0 && replicas != REPLICAS_ALL && replicas > nready &&
            push!(warnings, "replicas=$replicas exceeds the $nready worker(s) ready at the last poll; the placement will clamp to $nready")
        req.allow_unknown_model && !has_model(st.routes, name) &&
            push!(warnings, "no worker currently serves '$name'; the override is stored and applies once it is loaded")
        p.mode === :inflight && push!(warnings, _INFLIGHT_PARKING_WARNING)
        append!(warnings, _placement_warnings(s, name, effective))
        push!(warnings, "takes effect at the next repack; call Repack to apply it now")

        @info "gateway control: model placement updated" model = name replicas = (replicas == 0 ? "cleared" : (replicas == REPLICAS_ALL ? "all" : replicas)) fill_mode = fill_mode fill_factor = fill_factor generation = after.generation
        _CTRL_GW.SetModelPlacementResponse(; name = name,
            configured_replicas = _replicas_to_wire(replicas),
            effective_replicas = Int64(effective == REPLICAS_ALL ? max(nready, 1) : effective),
            fill_mode = String(p.mode), fill_quantum = Int64(p.quantum),
            warnings = warnings, persisted = false)
    end
end

function _gw_repack(req::_CTRL_GW.RepackRequest, st::GatewayState)
    s = _require_packing(st.scheduler)
    return _as_gateway_control() do
        req.wait_seconds < 0 && _invalid_arg("wait_seconds must be non-negative")
        warnings = String[]
        budget = Float64(req.wait_seconds)
        if budget > _REPACK_WAIT_MAX_SECONDS
            push!(warnings, "wait_seconds clamped to the server maximum of $(_REPACK_WAIT_MAX_SECONDS)s")
            budget = _REPACK_WAIT_MAX_SECONDS
        end
        # A repack runs on the prober, which only ticks over workers that reported ready. Waiting on a
        # fleet with none would burn the whole budget to no purpose, so say so instead.
        if budget > 0 && isempty((@atomic s.tick).ready_workers)
            throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE,
                "no worker was ready at the last prober tick, so a repack cannot run; retry once a worker is ready (or pass wait_seconds=0 to queue it)"))
        end
        seq = request_repack!(s)
        completed, waited = await_repack(s, seq, budget)
        completed || push!(warnings, "repack queued but not observed within $(round(waited; digits = 2))s; it will run on the next prober round")
        @info "gateway control: repack requested" sequence = seq completed = completed waited_seconds = round(waited; digits = 2)
        _CTRL_GW.RepackResponse(; sequence = UInt64(seq),
            completed_sequence = UInt64(@atomic s.repack_completed),
            completed = completed, waited_seconds = waited,
            repack = _repack_status(s), warnings = warnings)
    end
end

# Register the four handlers on the gateway's router (see build_gateway_router).
register_gateway_control_service!(router, state::GatewayState) =
    register_GatewayControlService!(router;
        GetSchedulingStatus = (req, ctx) -> _gw_scheduling_status(req, ctx.payload),
        SetSchedulingPolicy = (req, ctx) -> _gw_set_scheduling_policy(req, ctx.payload),
        SetModelPlacement = (req, ctx) -> _gw_set_model_placement(req, ctx.payload),
        Repack = (req, ctx) -> _gw_repack(req, ctx.payload))
