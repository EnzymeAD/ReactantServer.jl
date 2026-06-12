# The model -> worker URLs map that ModelInfer consults on every request. A model may be hosted
# on more than one worker (replication); the table round-robins across the replicas. The table is
# rebuilt by autodiscovery (see health.jl) and swapped atomically into a `DiscoveredRoutes`
# holder; each table is itself immutable apart from the per-entry round-robin cursor, which
# mutates via an atomic counter, so reads are lock free.

mutable struct RouteEntry
    const urls::Vector{String}
    next::Threads.Atomic{UInt64}
end

struct RoutingTable
    routes::Dict{String,RouteEntry}
end

function RoutingTable(routes::AbstractDict)
    d = Dict{String,RouteEntry}()
    for (model, urls) in routes
        cp = sort(String[String(u) for u in urls])
        d[String(model)] = RouteEntry(cp, Threads.Atomic{UInt64}(0))
    end
    return RoutingTable(d)
end

"""
    pick(table, model) -> Union{Vector{String},Nothing}

The worker URLs hosting `model`, rotated so the round-robin choice is first and the remaining
replicas follow in failover order. Each call advances the cursor. Returns `nothing` when the
model is not known.
"""
function pick(t::RoutingTable, model::AbstractString)
    e = get(t.routes, model, nothing)
    e === nothing && return nothing
    n = length(e.urls)
    n == 0 && return nothing
    # atomic_add! returns the previous value, so the first call yields start 0.
    start = Int(Threads.atomic_add!(e.next, UInt64(1)) % UInt64(n))
    return String[e.urls[(start + i) % n + 1] for i in 0:(n - 1)]
end

nmodels(t::RoutingTable) = length(t.routes)

# Holder for the current routing table. Autodiscovery swaps a freshly built table in atomically;
# `pick` reads the current table lock free. Starts empty until the first discovery round.
mutable struct DiscoveredRoutes
    @atomic current::RoutingTable
end
DiscoveredRoutes() = DiscoveredRoutes(RoutingTable(Dict{String,Vector{String}}()))

current_table(d::DiscoveredRoutes) = @atomic d.current
swap_table!(d::DiscoveredRoutes, t::RoutingTable) = (@atomic d.current = t; nothing)

pick(d::DiscoveredRoutes, model::AbstractString) = pick(current_table(d), model)
nmodels(d::DiscoveredRoutes) = nmodels(current_table(d))
