# The gateway scheduler interface. A scheduler decides, for each request, which worker(s) host the
# model and in what order to try them, and may run background work (placement, polling) on the
# prober tick. "Scheduling" here is strictly gateway-side: which worker a request goes to, not the
# worker's own request scheduler.
#
# The mode is chosen by `scheduling.mode` in gateway.yml and built by `make_scheduler`:
#   - round_robin       spread each model's requests uniformly across its replicas (RoundRobinScheduler)
#   - least_outstanding send each request to the replica with the least in-flight work, measured in
#                       GPU-seconds, items, or requests per `scheduling.work_basis`
#                       (LeastOutstandingScheduler)
#   - lpt_packing       concentrate each model on few GPUs to feed batch coalescing (LptPackingState, see lpt_packing.jl)
#
# Each scheduler owns its own data structure. The request hot path calls `select_replicas` with a
# `ScheduleContext` that carries the shared gateway resources (the worker pool, the discovered
# routing table, metrics, and the on-demand route refresher), so a scheduler can reach whatever it
# needs to make a decision; anything mode-specific lives in the scheduler struct itself.

abstract type GatewayScheduler end

# Everything a scheduler may consult to route one request. Parametric on the pool type so the
# request hot path stays type-stable (mirrors GatewayState). Extend this struct as future
# schedulers need more (e.g. the raw request body for content-based routing).
struct ScheduleContext{P<:ClientPool}
    model::String
    id::String
    # Items this request carries, as resolved by `request_units` (1 for a model that declares no
    # batch axis). Schedulers that route by work rather than by request count charge this.
    units::Int
    pool::P
    routes::DiscoveredRoutes
    metrics::GatewayMetrics
    refresher::RouteRefresher
end

"""
    select_replicas(s::GatewayScheduler, ctx::ScheduleContext) -> Union{Nothing,Tuple{Vector{String},Any}}

Order the worker URLs that should serve `ctx.model` (the chosen worker first, the rest as failover)
and return them with an opaque, scheduler-specific reservation to hand back to [`release!`](@ref)
when the request completes (or `nothing` if the scheduler tracks nothing). Return `nothing` when the
scheduler has no route for the model; the caller refreshes the routing table once and re-selects
before giving up. Required for every scheduler.
"""
function select_replicas end

# Optional lifecycle hooks; the no-op defaults let a scheduler implement only what it needs.

# Startup hook, run once after the pool is built and before serving. May verify preconditions
# (throwing to abort startup) and do initial work. lpt_packing overrides this.
scheduler_start!(::GatewayScheduler, ::ClientPool, metrics) = nothing

# Prober-tick hook, run each health round with the workers that reported ready and this round's
# control-status poll (see routing_meta.jl; `nothing` when no consumer asked for one). lpt_packing
# overrides this to fold costs and repack.
scheduler_tick!(::GatewayScheduler, ::ClientPool, ready_urls, metrics, poll) = nothing

# Whether this scheduler needs the prober to poll the workers' control plane each round. False keeps
# a fleet that routes by request count at exactly one probe round-trip per worker per tick, as before;
# the schedulers that route by work say so and get the shared poll (and with it the routing-metadata
# cache) at the cost of one extra RPC per worker per tick.
needs_routing_meta(::GatewayScheduler) = false

# Record a request arrival, for schedulers that estimate arrival rate. lpt_packing overrides this.
record_arrival!(::GatewayScheduler, model::AbstractString) = nothing

# How many items a request carries, for a scheduler that routes by work rather than by request
# count. The default never looks at the body; `lpt_packing` overrides it to peek the batch axis the
# worker reported for that model.
request_units(::GatewayScheduler, ::AbstractString, ::AbstractVector{UInt8}) = 1

# The scheduler's forced-work sequence number, polled by the prober between rounds so an
# operator-forced repack does not have to wait out the full probe interval. 0 means "never asks for an
# early round", which is every scheduler but lpt_packing.
scheduler_repack_seq(::GatewayScheduler) = 0

# Release a reservation returned by `select_replicas`, on every dispatch path. The default ignores
# it (covers schedulers that reserve nothing, and the `nothing` reservation of any scheduler).
release!(::GatewayScheduler, reservation) = nothing

# --- comparing in-flight work -------------------------------------------------------------------

# One nanosecond of GPU time, as the quantum for comparing two workers' loads. Far below the smallest
# charge that can exist (a single item of the cheapest model is microseconds) and far above the
# floating-point residue that accumulates in a load counter.
const _LOAD_QUANTUM_INV = 1.0e9

"""
    _load_key(load) -> Int

`load` reduced to an integer number of nanoseconds, for ORDERING two workers' in-flight work.

A load counter does not return to exactly 0.0 when a worker drains: its charges are added and
subtracted in different orders, and floating-point addition is not associative, so an idle worker
keeps a couple of ulp (netai02 sat at 4.44e-16 on two GPUs and exactly 0.0 on the other two). The
values are numerically meaningless, but comparing them raw is not: both `fill_least` and
`least_outstanding` are specified to treat equal loads as a TIE and let the rotation cursor (or the
URL order) break it, so that an idle fleet warms every replica. Residue turns every tie into a strict
inequality, the tie-break never runs, and the residue-free workers win every time. On netai02 that
skewed a 4-replica model's work 1.9x between its GPUs.

Quantizing to a nanosecond restores the tie without hiding any difference that could matter: two
workers whose in-flight work differs by less than a nanosecond of GPU time are, for routing purposes,
carrying the same load. Integer keys then compare exactly.
"""
_load_key(load::Real) = round(Int, Float64(load) * _LOAD_QUANTUM_INV)

# Snap a drained counter to exactly zero. Called after a release: if the counter has come back within
# one quantum of zero, store a true 0.0 so the residue cannot accumulate over the process's life and
# the exported gauge reads 0 when the worker is idle (a gauge showing 4.44e-16 reads as "busy" to a
# human and to a `> 0` alert). The compare-and-swap is what makes it safe: if a concurrent charge
# landed in between, the swap fails and the value it wrote stands, so no charge is ever lost.
function _settle_zero!(c::Threads.Atomic{Float64})
    v = c[]
    (v != 0.0 && abs(v) * _LOAD_QUANTUM_INV < 1.0) && Threads.atomic_cas!(c, v, 0.0)
    return nothing
end

# --- per-worker in-flight work ------------------------------------------------------------------

# The live per-worker work counters a work-routing scheduler compares, as `(basis, url => counter)`,
# or `nothing` for a scheduler that tracks none (round_robin). This is the number that decides which
# replica a request goes to, so exporting it is the difference between being able to explain a routing
# decision and guessing at it.
inflight_work(::GatewayScheduler) = nothing

# Exported at SCRAPE time from the scheduler's live counters, the same pattern as `RoutedCollector`:
# the series never lags a prober tick, and a worker that leaves the snapshot (a repack moved every
# model off it) simply stops being emitted instead of freezing at its last value.
struct WorkerWorkCollector{S<:GatewayScheduler} <: Prometheus.Collector
    scheduler::S
    worker_names::Dict{String,String}   # worker url -> friendly label, as GatewayMetrics maps them
end

Prometheus.metric_names(::WorkerWorkCollector) = ("gateway_worker_inflight_work",)

# The unit is `basis`-dependent, so it rides along as a label rather than being buried in the help
# text: a panel legend then states what it is reading, and a basis change is visible in the series
# rather than silently rescaling the old one.
const _WORK_GAUGE_HELP =
    "In-flight work on a worker, as the scheduler that routes to it counts it (see the `basis` " *
    "label and scheduling.work_basis): GPU-seconds under basis=compute, items under basis=items, " *
    "and under basis=requests either cost-weighted requests (lpt_packing fill_least) or a plain " *
    "request count (least_outstanding). This is the quantity fill_least and least_outstanding " *
    "compare when choosing a replica."

function Prometheus.collect!(out::Vector, c::WorkerWorkCollector)
    got = inflight_work(c.scheduler)
    got === nothing && return out
    basis, counters = got
    isempty(counters) && return out
    ln = Prometheus.LabelNames((:worker, :basis))
    b = String(basis)
    samples = Prometheus.Sample[]
    for (w, a) in counters
        push!(samples, Prometheus.Sample(nothing, ln,
                                        Prometheus.LabelValues((get(c.worker_names, w, w), b)),
                                        Float64(a[])))
    end
    # Deterministic order across scrapes; Prometheus.jl sorts family children for the same reason.
    sort!(samples; by = x -> x.label_values.label_values)
    push!(out, Prometheus.Metric("gauge", "gateway_worker_inflight_work", _WORK_GAUGE_HELP, samples))
    return out
end

# Register the gauge for a scheduler that has one. Called from each scheduler's `scheduler_start!`,
# so round_robin (which tracks no work) registers nothing and exports no empty family.
register_worker_work!(s::GatewayScheduler, m::GatewayMetrics) =
    inflight_work(s) === nothing ? nothing :
        Prometheus.register(m.registry, WorkerWorkCollector(s, m.worker_names))

"""
    make_scheduler(cfg::GatewayConfig, meta::RoutingMeta = RoutingMeta(cfg)) -> GatewayScheduler

Build the scheduler for the configured `scheduling.mode`. `meta` is the gateway's shared
routing-metadata cache (see routing_meta.jl), which a scheduler that routes by work reads on the
request path; it defaults to a fresh cache so a test can build a scheduler standalone.
"""
function make_scheduler(cfg::GatewayConfig, meta::RoutingMeta = RoutingMeta(cfg))
    cfg.scheduling_mode == "lpt_packing" && return LptPackingState(cfg, meta)
    cfg.scheduling_mode == "least_outstanding" &&
        return LeastOutstandingScheduler(meta; basis = Symbol(cfg.work_basis))
    return RoundRobinScheduler()
end

# --- round_robin ------------------------------------------------------------------------------

# Stateless: the round-robin cursor lives in the discovered routing table (see routing.jl), so this
# scheduler just delegates to `pick`, which rotates the replicas and returns them in failover order.
struct RoundRobinScheduler <: GatewayScheduler end

function select_replicas(::RoundRobinScheduler, ctx::ScheduleContext)
    urls = pick(ctx.routes, ctx.model)
    urls === nothing && return nothing
    return (urls, nothing)
end

# --- least_outstanding ------------------------------------------------------------------------

# Send each request to the replica carrying the least in-flight WORK, spreading load over a model's
# replicas without concentrating. Works over the autodiscovered routes like round_robin (no FIFO or
# all-models-on-all-workers preconditions). Its data structure is a per-worker in-flight load
# counter, grown copy-on-write under the lock so the hot path reads an immutable snapshot lock free
# (the same pattern as LptPackingState.arrivals).
#
# `basis` decides what "least" measures, per `scheduling.work_basis` (the same knob `lpt_packing`'s
# `fill_least` policy reads, so one setting describes what the whole fleet calls busy):
#
#   :compute   (default) in-flight GPU-seconds: each request charges its item count times the model's
#              measured cost per item. The only basis that is honest across heterogeneous models,
#              where 32 items of a cheap model are not 32 items of an expensive one.
#   :items     in-flight items. Needs no cost measurement, and is already correct for a fleet of
#              like-cost models; a client that pre-batches no longer looks like a single request.
#   :requests  in-flight requests, the original behavior. Blind to batch size: one request carrying
#              32 items counts the same as one carrying a single item.
#
# The ladder degrades rather than misreporting: `:compute` falls back to the fleet-mean cost per item
# for a model the gateway has not measured yet, and both work bases resolve to one unit per request
# for a model that declares no batch axis (a meta, an unbatched model) or that the gateway has not
# polled yet. So a cold fleet behaves exactly like `:requests` and converges to true work as
# measurements arrive, rather than routing on zeros.
const WORK_BASES = (:compute, :items, :requests)

mutable struct LeastOutstandingScheduler <: GatewayScheduler
    basis::Symbol
    meta::RoutingMeta                                    # shared routing metadata (batch axis + cost)
    @atomic inflight::Dict{String,Threads.Atomic{Float64}}
    lock::ReentrantLock
end

function LeastOutstandingScheduler(meta::RoutingMeta = RoutingMeta(); basis::Symbol = :compute)
    basis in WORK_BASES ||
        throw(ArgumentError("work basis must be one of $(WORK_BASES), got :$basis"))
    return LeastOutstandingScheduler(basis, meta, Dict{String,Threads.Atomic{Float64}}(), ReentrantLock())
end

# Only the work bases need the control plane; `:requests` counts what it can see in the request itself.
needs_routing_meta(s::LeastOutstandingScheduler) = s.basis !== :requests

# How many items a request carries. `:requests` never looks at the body (nor at the metadata cache),
# so that basis stays exactly as cheap as it was.
function request_units(s::LeastOutstandingScheduler, model::AbstractString, body::AbstractVector{UInt8})
    s.basis === :requests && return 1
    return request_items(routing_meta(s.meta), model, body)
end

# The load one request charges: its items, weighted by the model's cost per item under `:compute`.
# Captured at selection and the identical value released, so a cost that changes in between (the
# prober refolds the EWMA every tick) cannot leave the counters drifting.
function _charge(s::LeastOutstandingScheduler, ctx::ScheduleContext)
    units = Float64(max(ctx.units, 1))
    s.basis === :compute || return units
    return units * item_cost(routing_meta(s.meta), ctx.model)
end

# The in-flight load counter for `url`, creating it on first sight (lock + copy-on-write swap).
function _inflight_counter!(s::LeastOutstandingScheduler, url::AbstractString)
    cur = @atomic s.inflight
    c = get(cur, url, nothing)
    c === nothing || return c
    return lock(s.lock) do
        cur2 = @atomic s.inflight
        cc = get(cur2, url, nothing)
        if cc === nothing
            nxt = copy(cur2)
            cc = nxt[String(url)] = Threads.Atomic{Float64}(0.0)
            @atomic s.inflight = nxt
        end
        cc
    end
end

function select_replicas(s::LeastOutstandingScheduler, ctx::ScheduleContext)
    urls = pick(ctx.routes, ctx.model)
    urls === nothing && return nothing
    counters = [_inflight_counter!(s, u) for u in urls]
    best = 1
    for i in 2:length(urls)
        # Quantized, so that equally loaded replicas compare EQUAL and the URL order breaks the tie
        # as documented; raw float loads carry residue that would make every tie a strict inequality
        # and pin the model to whichever replica happens to hold the smaller crumb (see `_load_key`).
        ci, cb = _load_key(counters[i][]), _load_key(counters[best][])
        (ci < cb || (ci == cb && urls[i] < urls[best])) && (best = i)
    end
    charge = _charge(s, ctx)
    Threads.atomic_add!(counters[best], charge)          # reserve the chosen replica
    rest = String[urls[i] for i in eachindex(urls) if i != best]
    return (vcat(urls[best], rest), (counters[best], charge))
end

function release!(::LeastOutstandingScheduler, res::Tuple{Threads.Atomic{Float64},Float64})
    Threads.atomic_sub!(res[1], res[2])
    _settle_zero!(res[1])       # a drained worker reads exactly 0, not a few ulp of cancellation
    return nothing
end

inflight_work(s::LeastOutstandingScheduler) = (s.basis, @atomic s.inflight)

# No preconditions to verify and nothing to place; the hook exists only to export the load gauge.
function scheduler_start!(s::LeastOutstandingScheduler, ::ClientPool, metrics)
    metrics === nothing || register_worker_work!(s, metrics)
    return nothing
end
