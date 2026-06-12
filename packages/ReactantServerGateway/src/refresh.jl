# On-demand route refresh.
#
# The health prober rediscovers routes on a fixed interval (see health.jl), so a worker that
# dynamically loads or unloads a model is picked up within one tick. This closes that staleness
# window on demand for two request-time events:
#   * a request arrives for a model the routing table does not know  -> refresh_now! (synchronous,
#     the client is waiting), then re-pick;
#   * a worker reports a routed model is gone (NOT_FOUND)             -> request_refresh! (async,
#     fire-and-forget; we are already failing over / returning the error).
#
# Both reuse the prober's `discover_routes` + atomic `swap_table!`. Refreshes are single-flight
# (concurrent callers coalesce onto one in-flight scan) and rate-limited by `min_interval`, so a
# flood of requests for a genuinely-absent model cannot storm the workers. Full-table swaps are
# atomic, so a refresher scan racing the prober's scan is safe (last writer wins, both reflect
# near-current ground truth).

const REFRESH_MIN_INTERVAL_SECONDS = 1.0

mutable struct RouteRefresher
    pool::ClientPool
    routes::DiscoveredRoutes
    metrics::GatewayMetrics
    lock::ReentrantLock
    inflight::Union{Task,Nothing}   # the in-progress scan, shared by concurrent callers
    last_refresh::Float64           # time() of the last completed scan (0.0 = never), for rate limiting
    min_interval::Float64           # floor between forced scans; storm guard
    scans::Int                      # total scans performed (observability / tests)
end

RouteRefresher(pool::ClientPool, routes::DiscoveredRoutes, metrics::GatewayMetrics;
               min_interval::Real = REFRESH_MIN_INTERVAL_SECONDS) =
    RouteRefresher(pool, routes, metrics, ReentrantLock(), nothing, 0.0, Float64(min_interval), 0)

# Spawn a discovery scan. The caller must hold `r.lock`. The scan runs `discover_routes`, swaps the
# fresh table in, and (always, even on failure) stamps `last_refresh`, bumps the counter, and clears
# `inflight` so a failed scan never wedges the refresher.
function _spawn_scan!(r::RouteRefresher)
    t = @async begin
        try
            table = discover_routes(r.pool)
            swap_table!(r.routes, table)
            set_routing_size!(r.metrics, nmodels(table))
        catch e
            @warn "refresh: route discovery failed" exception = e
        finally
            lock(r.lock) do
                r.last_refresh = time()
                r.scans += 1
                r.inflight = nothing
            end
        end
    end
    r.inflight = t
    return t
end

"""
    refresh_now!(r::RouteRefresher)

Synchronously refresh the routing table, for the path where a client is waiting on a model the
table does not yet know. If a scan is already in flight, wait for it; if a scan completed within
`min_interval`, return without scanning (the table is already fresh enough); otherwise start a scan
and wait for it. Concurrent callers coalesce onto a single scan.
"""
function refresh_now!(r::RouteRefresher)
    task = lock(r.lock) do
        if r.inflight !== nothing
            return r.inflight
        elseif time() - r.last_refresh < r.min_interval
            return nothing
        else
            return _spawn_scan!(r)
        end
    end
    task === nothing || wait(task)
    return nothing
end

"""
    request_refresh!(r::RouteRefresher)

Request an asynchronous, fire-and-forget refresh, for the path where a worker reported a routed
model is gone. Never blocks: starts a scan only if none is in flight and the rate limit allows,
otherwise does nothing (an in-flight or recent scan already covers it).
"""
function request_refresh!(r::RouteRefresher)
    lock(r.lock) do
        (r.inflight === nothing && time() - r.last_refresh >= r.min_interval) && _spawn_scan!(r)
    end
    return nothing
end
