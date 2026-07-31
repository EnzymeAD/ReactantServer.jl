# The lpt_packing scheduler's runtime-mutable knobs, plus the snapshots it publishes for readers on
# other tasks.
#
# Every knob an operator can change while the gateway runs lives in one immutable `PackingKnobs`
# struct, held behind a single `@atomic` field on `LptPackingState`. A reader takes ONE snapshot
# (`k = knobs(s)`) and then reads plain fields off it, so a repack can never observe half of an
# update, and the request hot path pays a single pointer load. A writer builds a whole new struct
# under the scheduler's `control_lock` and swaps it in (see `set_knobs!`).
#
# One `@atomic` field per knob was rejected. `@atomic` in Julia is not opt-in per access: a plain
# read or write of a field *declared* `@atomic` raises `ConcurrencyViolationError`, so every read
# site and every test assignment would have to change, with an error class that only appears at
# runtime. Worse, independent atomics would let a repack pair a new `hysteresis` with an old
# `default_replicas`, which is exactly the mixed-generation placement this design rules out.
#
# `model_overrides` is a Dict *reference*: it is replaced wholesale, never mutated in place. The type
# cannot enforce that; writers must respect it (the same copy-on-write contract as
# `LptPackingState.arrivals`).

# One model's scheduling overrides. Every field carries a "not set" sentinel, so a model can override
# its replica count, its fill basis, its run length, or any subset, and inherit the fleet default for
# the rest. Resolution against the defaults happens on the prober tick, never on the request path.
struct ModelKnobs
    replicas::Int          # 0 = inherit `default_replicas`; `REPLICAS_ALL` = every ready worker
    fill_mode::Symbol      # :inherit | :run | :spread | :inflight
    fill_factor::Float64   # 0.0 = inherit `routing_fill_factor`
end

ModelKnobs() = ModelKnobs(0, :inherit, 0.0)
ModelKnobs(mc::GatewayModelConfig) = ModelKnobs(mc.replicas, mc.fill_mode, mc.fill_factor)

"""
    PackingKnobs

The complete set of lpt_packing settings that can change without a gateway restart. Built from
[`GatewayConfig`](@ref) at startup and thereafter replaced wholesale by the control plane.

`generation` is 0 while every knob still matches `gateway.yml`, and is bumped by each accepted
mutation, so an operator inheriting a running gateway can tell a tuned one from a fresh one. Nothing
here is persisted: a restart reverts to the config file and the environment.
"""
struct PackingKnobs
    hysteresis::Float64
    # Already resolved: a configured 0 means "track the rebalance interval" and is folded in once
    # here, so this never reads 0 and a later change to `rebalance_compute_seconds` does not
    # retroactively move the halflife.
    ema_halflife_compute::Float64
    rebalance_compute_seconds::Float64
    first_rebalance_compute_seconds::Float64   # 0 = no separate first-repack budget
    default_replicas::Int
    model_overrides::Dict{String,ModelKnobs}
    routing_fill_factor::Float64
    routing_policy::Symbol                     # :fill_rr | :fill_least
    routing_fill_mode::Symbol                  # :run | :spread | :inflight
    forbid_memory_oversubscription::Bool
    compaction_mode::Symbol                    # :off | :eager | :scheduled
    compaction_interval::Int
    generation::Int
end

function PackingKnobs(cfg::GatewayConfig)
    return PackingKnobs(
        cfg.hysteresis,
        cfg.ema_halflife_compute_seconds > 0 ? cfg.ema_halflife_compute_seconds :
            cfg.rebalance_compute_seconds,
        cfg.rebalance_compute_seconds,
        cfg.first_rebalance_compute_seconds,
        cfg.default_replicas,
        Dict{String,ModelKnobs}(name => ModelKnobs(mc) for (name, mc) in cfg.models),
        cfg.routing_fill_factor,
        Symbol(cfg.routing_policy),
        Symbol(cfg.routing_fill_mode),
        cfg.forbid_memory_oversubscription,
        cfg.compaction_mode,
        cfg.compaction_interval,
        0,
    )
end

"""
    FillPlan

One model's resolved routing rule: what its fill quantum counts and how big that quantum is. Built on
the prober tick by [`resolve_fill_plan`](@ref) and read once per request, so the request path never
has to walk the three-level override chain or touch `max_batch`.
"""
struct FillPlan
    mode::Symbol      # :run | :spread | :inflight (never :inherit; resolved)
    quantum::Int      # >= 1
end

# The plan for a model with no entry in the published snapshot: a cold or unknown model, routed as if
# the fleet default applied with an unknown max batch. Never installed, only returned as a fallback.
const _DEFAULT_FILL_PLAN = FillPlan(:run, 1)

"""
    resolve_fill_plan(k::PackingKnobs, model, max_batch) -> FillPlan

Collapse the three configuration levels for one model: its own override, then the `scheduling:`
default, then the built-in. `max_batch` is the worker-reported effective max batch (0 when unknown),
which scales into the quantum through the fill factor.

An unknown max batch yields a quantum of 1, which is the honest degradation: with no batch size to
aim at, `run` becomes exact rotation and `spread` becomes least-in-flight, rather than pretending to
fill something.
"""
function resolve_fill_plan(k::PackingKnobs, model::AbstractString, max_batch::Int)
    mk = get(k.model_overrides, model, nothing)
    mode = (mk === nothing || mk.fill_mode === :inherit) ? k.routing_fill_mode : mk.fill_mode
    factor = (mk === nothing || mk.fill_factor <= 0) ? k.routing_fill_factor : mk.fill_factor
    quantum = max_batch <= 0 ? 1 : max(1, round(Int, factor * max_batch))
    return FillPlan(mode, quantum)
end

# The replica-count overrides in the shape `compute_assignment` wants. Only models carrying an actual
# override appear, so a model whose override was cleared (or that was never listed) falls back to
# `default_replicas` through `compute_assignment`'s own `get`. Rebuilt per repack; repacks are rare
# and the request path never touches this.
replica_overrides(k::PackingKnobs) =
    Dict{String,Int}(m => mk.replicas for (m, mk) in k.model_overrides if mk.replicas > 0)

# Validate and coerce one knob by field name, through the same checks `gateway.yml` uses (config.jl),
# so the config path and the control plane cannot disagree about what a knob accepts. Unknown names
# throw rather than being ignored, so a typo in a control request is reported instead of silently
# doing nothing.
function _validate_knob(field::Symbol, v)
    field === :hysteresis && return _ck_unit("scheduling.hysteresis", v)
    field === :ema_halflife_compute &&
        return _ck_positive("scheduling.ema_halflife_compute_seconds", v)
    field === :rebalance_compute_seconds &&
        return _ck_positive("scheduling.rebalance_compute_seconds", v)
    field === :first_rebalance_compute_seconds &&
        return _ck_nonneg("scheduling.first_rebalance_compute_seconds", v;
                          hint = "0 = use rebalance_compute_seconds")
    field === :default_replicas && return _parse_replicas(v, "scheduling.default_replicas")
    field === :routing_fill_factor && return _ck_positive("scheduling.routing_fill_factor", v)
    field === :routing_policy && return _parse_routing_policy(v)
    field === :routing_fill_mode && return _parse_fill_mode(v)
    field === :compaction_mode && return _parse_gateway_compaction_mode(v)
    field === :compaction_interval &&
        return _ck_nonneg_int("scheduling.compaction_interval", v; hint = "0 = disabled")
    field === :forbid_memory_oversubscription && return Bool(v)
    field === :model_overrides && return v::Dict{String,ModelKnobs}
    field === :generation && return Int(v)
    throw(ConfigError("unknown scheduling knob '$field'"))
end

"""
    apply_updates(k::PackingKnobs; kwargs...) -> PackingKnobs

A copy of `k` with the named fields replaced, each validated exactly as `gateway.yml` validates it.

Total and all-or-nothing: every update is validated before anything is constructed, so a rejected
value leaves the caller's live knobs untouched. `generation` is bumped unless it is passed
explicitly. Runs on a control-plane mutation only, never on the request path, so the dynamic field
walk here costs nothing that matters.
"""
function apply_updates(k::PackingKnobs; kwargs...)
    fields = fieldnames(PackingKnobs)
    vals = Any[getfield(k, f) for f in fields]
    bump = true
    for (key, v) in kwargs
        i = findfirst(==(key), fields)
        i === nothing && throw(ConfigError("unknown scheduling knob '$key'"))
        vals[i] = _validate_knob(key, v)
        key === :generation && (bump = false)
    end
    bump && (vals[findfirst(==(:generation), fields)] = k.generation + 1)
    return PackingKnobs(vals...)
end

# --- published snapshots ----------------------------------------------------------------------
#
# `_repack!` and `tick_packing!` own a set of plain (non-atomic) fields on `LptPackingState` that a
# control handler on another task must be able to read. Rather than making each of them atomic, the
# prober publishes an immutable summary at the end of every tick and every repack, and handlers read
# the summary. One atomic pointer swap per tick, and the prober's own accounting stays plain.

"""
    TickReport

What the prober observed on its most recent round: enough for a control handler to explain why a
repack has or has not fired without touching the prober's own fields.
"""
struct TickReport
    unix_seconds::Float64
    ready_workers::Vector{String}
    fleet_compute::Float64        # cumulative fleet GPU-seconds as of this tick
    compute_accum::Float64        # accumulated since the last repack
    active_threshold::Float64     # the budget the next tick compares `compute_accum` against
    first_tick_repack_done::Bool  # false = the first-repack budget is still armed
end

TickReport() = TickReport(0.0, String[], 0.0, 0.0, 0.0, false)

"""
    RepackReport

The last repack's inputs and outcome. `utilization` and `rate` are copies of the prober's in-place
EWMA dicts (for the same reason `cost_snapshot` is a copy): the next repack folds those dicts in
place and must not race a handler reading them.
"""
struct RepackReport
    unix_seconds::Float64
    wall_elapsed::Float64
    compute_elapsed::Float64
    trigger::Symbol               # :none | :startup | :compute | :operator
    count::Int                    # repacks since start, including the startup placement
    models_placed::Int
    models_moved::Int
    utilization::Dict{String,Float64}
    rate::Dict{String,Float64}
    cost::Dict{String,Float64}
    mem::Dict{String,Float64}     # per-model weight footprint seen by this repack
    mem_cap::Dict{String,Float64} # per-worker weight budget (0 = unconstrained)
    polled::Vector{String}
    drifted::Set{String}          # models not reported by every ready worker
end

RepackReport() = RepackReport(0.0, 0.0, 0.0, :none, 0, 0, 0,
                              Dict{String,Float64}(), Dict{String,Float64}(),
                              Dict{String,Float64}(), Dict{String,Float64}(),
                              Dict{String,Float64}(), String[], Set{String}())
