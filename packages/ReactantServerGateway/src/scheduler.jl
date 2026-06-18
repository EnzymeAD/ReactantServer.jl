# The gateway scheduler interface. A scheduler decides, for each request, which worker(s) host the
# model and in what order to try them, and may run background work (placement, polling) on the
# prober tick. "Scheduling" here is strictly gateway-side: which worker a request goes to, not the
# worker's own request scheduler.
#
# The mode is chosen by `scheduling.mode` in gateway.yml and built by `make_scheduler`:
#   - round_robin       spread each model's requests uniformly across its replicas (RoundRobinScheduler)
#   - least_outstanding send each request to the replica with the least in-flight work (LeastOutstandingScheduler)
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

# Prober-tick hook, run each health round with the workers that reported ready. lpt_packing
# overrides this to poll costs and repack.
scheduler_tick!(::GatewayScheduler, ::ClientPool, ready_urls, metrics) = nothing

# Record a request arrival, for schedulers that estimate arrival rate. lpt_packing overrides this.
record_arrival!(::GatewayScheduler, model::AbstractString) = nothing

# Release a reservation returned by `select_replicas`, on every dispatch path. The default ignores
# it (covers schedulers that reserve nothing, and the `nothing` reservation of any scheduler).
release!(::GatewayScheduler, reservation) = nothing

"""
    make_scheduler(cfg::GatewayConfig) -> GatewayScheduler

Build the scheduler for the configured `scheduling.mode`.
"""
function make_scheduler(cfg::GatewayConfig)
    cfg.scheduling_mode == "lpt_packing" && return LptPackingState(cfg)
    cfg.scheduling_mode == "least_outstanding" && return LeastOutstandingScheduler()
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

# Send each request to the replica with the fewest in-flight requests, spreading load over a model's
# replicas without concentrating. Works over the autodiscovered routes like round_robin (no FIFO or
# all-models-on-all-workers preconditions). Its data structure is a per-worker in-flight counter,
# grown copy-on-write under the lock so the hot path reads an immutable snapshot lock free (the same
# pattern as LptPackingState.arrivals).
mutable struct LeastOutstandingScheduler <: GatewayScheduler
    @atomic inflight::Dict{String,Threads.Atomic{Int}}
    lock::ReentrantLock
end
LeastOutstandingScheduler() = LeastOutstandingScheduler(Dict{String,Threads.Atomic{Int}}(), ReentrantLock())

# The in-flight counter for `url`, creating it on first sight (lock + copy-on-write swap).
function _inflight_counter!(s::LeastOutstandingScheduler, url::AbstractString)
    cur = @atomic s.inflight
    c = get(cur, url, nothing)
    c === nothing || return c
    return lock(s.lock) do
        cur2 = @atomic s.inflight
        cc = get(cur2, url, nothing)
        if cc === nothing
            nxt = copy(cur2)
            cc = nxt[String(url)] = Threads.Atomic{Int}(0)
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
    Threads.atomic_add!(counters[best], 1)               # reserve the chosen replica
    rest = String[urls[i] for i in eachindex(urls) if i != best]
    return (vcat(urls[best], rest), counters[best])
end

release!(::LeastOutstandingScheduler, c::Threads.Atomic{Int}) = (Threads.atomic_sub!(c, 1); nothing)
