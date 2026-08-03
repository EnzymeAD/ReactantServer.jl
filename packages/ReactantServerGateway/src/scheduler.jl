# The gateway scheduler interface. A scheduler decides, for each request, which worker(s) host the
# model and in what order to try them, and may run background work (placement, polling) on the
# prober tick. "Scheduling" here is strictly gateway-side: which worker a request goes to, not the
# worker's own request scheduler.
#
# The mode is chosen by `scheduling.mode` in gateway.yml and built by `make_scheduler`:
#   - round_robin       spread each model's requests uniformly across its replicas (RoundRobinScheduler)
#   - least_outstanding send each request to the replica with the least in-flight work, measured in
#                       GPU-seconds, items, or requests per `scheduling.least_outstanding_basis`
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

"""
    make_scheduler(cfg::GatewayConfig, meta::RoutingMeta = RoutingMeta(cfg)) -> GatewayScheduler

Build the scheduler for the configured `scheduling.mode`. `meta` is the gateway's shared
routing-metadata cache (see routing_meta.jl), which a scheduler that routes by work reads on the
request path; it defaults to a fresh cache so a test can build a scheduler standalone.
"""
function make_scheduler(cfg::GatewayConfig, meta::RoutingMeta = RoutingMeta(cfg))
    cfg.scheduling_mode == "lpt_packing" && return LptPackingState(cfg)
    cfg.scheduling_mode == "least_outstanding" &&
        return LeastOutstandingScheduler(meta; basis = Symbol(cfg.least_outstanding_basis))
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
# `basis` decides what "least" measures, per `scheduling.least_outstanding_basis`:
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
const LEAST_OUTSTANDING_BASES = (:compute, :items, :requests)

mutable struct LeastOutstandingScheduler <: GatewayScheduler
    basis::Symbol
    meta::RoutingMeta                                    # shared routing metadata (batch axis + cost)
    @atomic inflight::Dict{String,Threads.Atomic{Float64}}
    lock::ReentrantLock
end

function LeastOutstandingScheduler(meta::RoutingMeta = RoutingMeta(); basis::Symbol = :compute)
    basis in LEAST_OUTSTANDING_BASES ||
        throw(ArgumentError("least_outstanding basis must be one of $(LEAST_OUTSTANDING_BASES), got :$basis"))
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
        ci, cb = counters[i][], counters[best][]
        (ci < cb || (ci == cb && urls[i] < urls[best])) && (best = i)
    end
    charge = _charge(s, ctx)
    Threads.atomic_add!(counters[best], charge)          # reserve the chosen replica
    rest = String[urls[i] for i in eachindex(urls) if i != best]
    return (vcat(urls[best], rest), (counters[best], charge))
end

release!(::LeastOutstandingScheduler, res::Tuple{Threads.Atomic{Float64},Float64}) =
    (Threads.atomic_sub!(res[1], res[2]); nothing)
