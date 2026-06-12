# Periodic worker readiness probe and route autodiscovery. Each round probes every endpoint's
# ServerReady (driving /readyz: the aggregate is ready when at least one worker reports ready) and
# its RepositoryIndex (driving the routing table: the model -> ready-endpoints map is rebuilt and
# swapped in atomically). A control-plane pin/unpin or a node restart is picked up on the next
# round. Mirrors and extends the Go gateway's internal/health.

const HEALTH_INTERVAL_SECONDS = 10.0

mutable struct HealthProber
    pool::ClientPool
    metrics::GatewayMetrics
    admin::AdminServer
    routes::Union{DiscoveredRoutes,Nothing}
    packing::Union{LptPackingState,Nothing}   # packing-mode rebalancer, ticked on its own interval
    interval::Float64
    running::Threads.Atomic{Bool}
    task::Union{Task,Nothing}
end

HealthProber(pool::ClientPool, metrics::GatewayMetrics, admin::AdminServer,
             routes::Union{DiscoveredRoutes,Nothing} = nothing;
             packing::Union{LptPackingState,Nothing} = nothing,
             interval::Real = HEALTH_INTERVAL_SECONDS) =
    HealthProber(pool, metrics, admin, routes, packing, Float64(interval), Threads.Atomic{Bool}(true), nothing)

# Query every endpoint's ready models concurrently and build the model -> endpoints routing table.
# An unreachable endpoint is skipped (it contributes no routes this round and is picked up later).
function discover_routes(pool::ClientPool)
    workers = all_clients(pool)
    found = Dict{String,Vector{String}}()
    lk = ReentrantLock()
    @sync for wc in workers
        @async begin
            names = discover_models(wc)
            names === nothing && return
            lock(lk) do
                for n in names
                    push!(get!(found, n, String[]), wc.url)
                end
            end
        end
    end
    return RoutingTable(found)
end

function _check_once(p::HealthProber)
    workers = all_clients(p.pool)
    results = Vector{Bool}(undef, length(workers))
    @sync for (i, wc) in enumerate(workers)
        @async results[i] = probe_ready(wc)
    end
    any_ready = false
    for (i, wc) in enumerate(workers)
        set_worker_ready!(p.metrics, wc.url, results[i])
        any_ready |= results[i]
    end
    set_ready!(p.admin, any_ready)
    if p.routes !== nothing
        table = discover_routes(p.pool)
        swap_table!(p.routes, table)
        set_routing_size!(p.metrics, nmodels(table))
    end
    # LPT-packing rebalance on its own cadence (>= the probe interval). Placement is computed over
    # the workers that reported ready this round; a dead worker drops out until it recovers.
    if p.packing !== nothing &&
       time() - p.packing.last_rebalance >= p.packing.rebalance_seconds
        ready_urls = String[wc.url for (i, wc) in enumerate(workers) if results[i]]
        if isempty(ready_urls)
            @warn "lpt_packing: no ready workers this round; keeping the previous assignment"
        else
            try
                rebalance!(p.packing, p.pool, ready_urls, p.metrics)
            catch e
                @warn "lpt_packing: rebalance failed; keeping the previous assignment" exception = e
            end
        end
    end
    return any_ready
end

# Probe once immediately, then on the interval until stopped.
function start_prober!(p::HealthProber)
    p.task = @async begin
        try
            _check_once(p)
        catch e
            @warn "health: initial probe failed" exception = e
        end
        while p.running[]
            sleep(p.interval)
            p.running[] || break
            try
                _check_once(p)
            catch e
                @warn "health: probe round failed" exception = e
            end
        end
    end
    return p
end

stop_prober!(p::HealthProber) = (p.running[] = false; nothing)
