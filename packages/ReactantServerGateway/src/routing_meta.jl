# Per-model routing metadata, learned from the workers' control plane and owned by nothing in
# particular: the health prober polls once per round and publishes an immutable snapshot that any
# scheduler can read on the request hot path. This exists because "how much work is this request?"
# is a property of the MODEL, not of the packing policy, and every scheduler that routes by work
# needs the same three answers:
#
#   - where the batch axis is, so a request can be sized in ITEMS without decoding its payload
#     (the worker reports an input NAME and a 1-based axis; the position genuinely varies per bundle
#     and is not guessable, see `wire_batch_spec`)
#   - what one item costs in GPU-seconds, so items of a cheap model do not weigh the same as items
#     of an expensive one
#   - the effective max batch, the item count that makes a fill quantum dimensionally honest
#
# Before this, only `lpt_packing` polled `ModelControlStatus`, so it was the only scheduler that
# could route by anything but a raw request count. The poll now lives here and lpt_packing consumes
# the same result, so there is exactly one control-status round per prober tick no matter how many
# consumers there are.
#
# Cost per ITEM is the reason `ModelStatus.rows_served` is reported: cost per REQUEST
# (Δcompute / Δrequests) already embeds whatever batch sizes the clients happened to send, so it
# cannot be scaled by an item count without double-counting. Cost per item (Δcompute / Δrows) can.

# Watchdog ceiling for one control-status poll. The gateway's clients share one libcurl multi
# handle with a 16-slot request semaphore (GRPC_MAX_STREAMS); if its driving stalls, in-flight
# calls never release their slot and new calls block forever, so every poll is bounded and a worker
# that exceeds this is treated as not-polled.
const POLL_TIMEOUT_SECONDS = 8.0

# EWMA fold with halflife `h` over an interval `dt`. Shared by this cache and lpt_packing, which
# both age their estimates against fleet compute rather than wall-clock.
_ewma(old::Float64, sample::Float64, dt::Float64, h::Float64) =
    (alpha = 1 - 2.0^(-dt / h); (1 - alpha) * old + alpha * sample)

# One model's routing metadata. Costs are measured GPU-seconds, 0.0 until the fleet has served
# enough of the model to derive them (see `RoutingMetaSnapshot.mean_cost_per_item` for the
# cold-start stand-in).
struct ModelRoutingMeta
    max_batch::Int              # effective max batch the worker coalesces to, in items; 0 unknown
    batch_input::String         # wire input carrying the batch axis; "" = model declares none
    batch_axis::Int             # 1-based axis within that input; 0 = none
    cost_per_request::Float64   # Δtotal_compute / Δrequests_served
    cost_per_item::Float64      # Δtotal_compute / Δrows_served
end

ModelRoutingMeta() = ModelRoutingMeta(0, "", 0, 0.0, 0.0)

# The published snapshot: every model's metadata plus the fleet-mean cost per item, which stands in
# for a model with no measurement yet so a cold model weighs like an average one rather than
# vanishing (weight 0, attracting every request) or dominating.
struct RoutingMetaSnapshot
    models::Dict{String, ModelRoutingMeta}
    mean_cost_per_item::Float64
end

RoutingMetaSnapshot() = RoutingMetaSnapshot(Dict{String, ModelRoutingMeta}(), 0.0)

"""
    RoutingMeta(halflife_compute_seconds)

The gateway's per-model routing metadata cache. `refresh_routing_meta!` updates it from a control
plane poll on the prober task; readers take [`routing_meta`](@ref) once per request and read plain
fields off the immutable result.

Costs are smoothed with an EWMA that decays against **fleet compute consumed**, not wall-clock, the
same clock lpt_packing's repack cadence uses: a model's measured cost ages in proportion to how much
inference the fleet is doing, so an idle fleet coasts on its last good measurement instead of
decaying toward noise.
"""
mutable struct RoutingMeta
    @atomic snapshot::RoutingMetaSnapshot
    halflife::Float64                                    # in fleet GPU-seconds
    # Prober-task-owned: EWMA state and the cumulative baselines the per-tick deltas are taken
    # against. Never touched from a request task, so plain fields are correct.
    cost_req::Dict{String, Float64}
    cost_item::Dict{String, Float64}
    last_cum::Dict{String, Tuple{Float64, UInt64, UInt64}}  # model -> (compute, requests, rows)
    last_fleet_compute::Float64
end

RoutingMeta(halflife::Real = 30.0) =
    RoutingMeta(
    RoutingMetaSnapshot(), Float64(halflife), Dict{String, Float64}(),
    Dict{String, Float64}(), Dict{String, Tuple{Float64, UInt64, UInt64}}(), 0.0
)

# The cache sized from gateway.yml. The halflife mirrors `PackingKnobs`: `ema_halflife_compute_seconds`
# when set, else the rebalance budget, so the two age their cost estimates at the same rate.
RoutingMeta(cfg::GatewayConfig) =
    RoutingMeta(
    cfg.ema_halflife_compute_seconds > 0 ? cfg.ema_halflife_compute_seconds :
        (cfg.rebalance_compute_seconds > 0 ? cfg.rebalance_compute_seconds : 30.0)
)

"""
    routing_meta(rm::RoutingMeta) -> RoutingMetaSnapshot

The current snapshot. Take this **once** per request and read plain fields off the result; re-reading
the atomic mid-decision can mix generations.
"""
routing_meta(rm::RoutingMeta) = @atomic rm.snapshot

"""
    model_meta(snap::RoutingMetaSnapshot, model) -> ModelRoutingMeta

`model`'s metadata, or an empty record (no batch axis, no cost) for a model the gateway has not
polled yet. Never `nothing`, so callers do not branch on absence.
"""
model_meta(snap::RoutingMetaSnapshot, model::AbstractString) =
    get(snap.models, model, ModelRoutingMeta())

"""
    request_items(snap::RoutingMetaSnapshot, model, body) -> Int

How many items the request in `body` carries: the extent of `model`'s batch axis, read out of the
request without decoding its payload. Falls back to 1 for a model that declares no batch axis, one
the gateway has not polled yet, or a request missing that input, so the count degrades to a request
count rather than to zero.
"""
function request_items(snap::RoutingMetaSnapshot, model::AbstractString, body::AbstractVector{UInt8})
    m = model_meta(snap, model)
    m.batch_axis < 1 && return 1
    n = peek_batch_size(body, m.batch_input, m.batch_axis)
    return n > 0 ? n : 1
end

"""
    item_cost(snap::RoutingMetaSnapshot, model) -> Float64

The GPU-seconds one item of `model` costs: its own measurement, else the fleet mean as a cold-start
stand-in, else 1.0 before the fleet has measured anything. Never 0, so a work-weighted charge is
never silently free.
"""
function item_cost(snap::RoutingMetaSnapshot, model::AbstractString)
    c = model_meta(snap, model).cost_per_item
    c > 0 && return c
    return snap.mean_cost_per_item > 0 ? snap.mean_cost_per_item : 1.0
end

# Poll every ready worker's ModelControlStatus concurrently and aggregate: per-model cumulative
# (compute, requests, rows) summed across workers, the per-model weight footprint, effective max
# batch and batch axis, each worker's weight-memory budget, the workers that answered, and the
# fleet's total cumulative compute (the compute-trigger signal). I/O only; no state mutation, so
# both this cache and lpt_packing's repack can be driven from one call.
function poll_workers(pool::ClientPool, ready_urls::Vector{String})
    sums = Dict{String, Tuple{Float64, UInt64, UInt64}}()   # model -> (compute, requests, rows)
    permodel_workers = Dict{String, Vector{String}}()
    mem = Dict{String, Float64}()                 # model -> resident weight bytes
    mem_cap = Dict{String, Float64}()             # worker -> on-demand weight budget (0 = unconstrained)
    max_batch = Dict{String, Int}()               # model -> effective max batch (max across workers)
    batch_at = Dict{String, Tuple{String, Int}}()  # model -> (wire input name, 1-based batch axis)
    polled = String[]
    lk = ReentrantLock()
    @sync for url in ready_urls
        wc = get_clients(pool, url)
        wc === nothing && continue
        @async begin
            # Watchdog-bounded: a wedged client stack would otherwise hang the prober tick (and with
            # it route discovery and /readyz). A worker that does not answer is simply skipped this
            # round, as if not ready. A hung call (timed out, not a fast refuse) means a poisoned
            # connection; drop it so the next poll reconnects fresh.
            resp, to = _bounded(
                () -> fetch_control_status(wc), POLL_TIMEOUT_SECONDS, nothing,
                "ModelControlStatus poll", url
            )
            if resp === nothing
                to && reset_clients!(wc)
                return
            end
            lock(lk) do
                push!(polled, url)
                mem_cap[url] = Float64(resp.weight_cache_max_bytes)
                for ms in resp.models
                    tc, rq, rw = get(sums, ms.name, (0.0, UInt64(0), UInt64(0)))
                    sums[ms.name] = (
                        tc + ms.total_compute_seconds, rq + ms.requests_served,
                        rw + ms.rows_served,
                    )
                    mem[ms.name] = max(get(mem, ms.name, 0.0), Float64(ms.weight_nbytes))
                    max_batch[ms.name] = max(get(max_batch, ms.name, 0), Int(ms.max_batch_size))
                    # Every worker serves the same bundle, so the first non-empty report wins; a
                    # worker mid-reload can legitimately report nothing yet.
                    isempty(ms.batch_input_name) || haskey(batch_at, ms.name) ||
                        (batch_at[ms.name] = (ms.batch_input_name, Int(ms.batch_axis)))
                    push!(get!(permodel_workers, ms.name, String[]), url)
                end
            end
        end
    end
    fleet_compute = 0.0
    for (tc, _, _) in values(sums)
        fleet_compute += tc
    end
    return (; sums, permodel_workers, mem, mem_cap, max_batch, batch_at, polled, fleet_compute)
end

"""
    refresh_routing_meta!(rm::RoutingMeta, poll) -> RoutingMetaSnapshot

Fold one [`poll_workers`](@ref) result into the cache and publish a fresh snapshot. Prober task only.

Costs come from deltas against the previous poll, so each sample measures that interval rather than
the process's lifetime average. The exception is the first poll, whose baseline is zero: its sample is
the counters as they stand, which makes a cost available after one round instead of two (the same
opening prior lpt_packing takes on its first repack). A negative delta means a worker restarted (its
counters reset); that model is re-baselined and its cost left at the last good value rather than being
poisoned by a nonsense sample.
"""
function refresh_routing_meta!(rm::RoutingMeta, poll)
    # Decay clock: the fleet GPU-seconds consumed since the previous refresh. The first observation
    # only baselines (dt = 0 makes the fold a no-op, so the first real sample is adopted outright).
    dt = 0.0
    if rm.last_fleet_compute > 0.0
        delta = poll.fleet_compute - rm.last_fleet_compute
        delta > 0 && (dt = delta)         # negative = worker restart: re-baseline, no decay
    end
    rm.last_fleet_compute = poll.fleet_compute

    for (m, (tc, rq, rw)) in poll.sums
        prev_tc, prev_rq, prev_rw = get(rm.last_cum, m, (0.0, UInt64(0), UInt64(0)))
        dtc = tc - prev_tc
        drq = Int(rq) - Int(prev_rq)
        drw = Int(rw) - Int(prev_rw)
        rm.last_cum[m] = (tc, rq, rw)
        (dtc < 0 || drq < 0 || drw < 0) && continue        # worker restart: re-baseline only
        drq > 0 && (rm.cost_req[m] = _ewma(get(rm.cost_req, m, dtc / drq), dtc / drq, dt, rm.halflife))
        drw > 0 && (rm.cost_item[m] = _ewma(get(rm.cost_item, m, dtc / drw), dtc / drw, dt, rm.halflife))
    end

    # Publish. Models come from this poll's `max_batch` (every model any worker reported) unioned with
    # the models we hold a cost for, so a model missing from one round keeps its metadata rather than
    # blinking out of the cache mid-flight.
    models = Dict{String, ModelRoutingMeta}()
    names = union(keys(poll.max_batch), keys(rm.cost_item), keys(rm.cost_req))
    for m in names
        bi, ax = get(poll.batch_at, m, ("", 0))
        prev = model_meta(routing_meta(rm), m)
        # A worker mid-reload reports no batch axis; keep the last known one rather than degrading
        # the model to one item per request for a round.
        if isempty(bi)
            bi = prev.batch_input
            ax = prev.batch_axis
        end
        models[m] = ModelRoutingMeta(
            get(poll.max_batch, m, prev.max_batch), bi, ax,
            get(rm.cost_req, m, 0.0), get(rm.cost_item, m, 0.0)
        )
    end
    measured = Float64[c for c in (m.cost_per_item for m in values(models)) if c > 0]
    mean_item = isempty(measured) ? 0.0 : sum(measured) / length(measured)
    snap = RoutingMetaSnapshot(models, mean_item)
    @atomic rm.snapshot = snap
    return snap
end
