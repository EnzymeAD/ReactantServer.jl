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
    routes::Dict{String, RouteEntry}
end

# `prev`, when given, is the table this one replaces: a model it also routes keeps its round-robin
# cursor (the same atomic), so a rebuild every probe round does not send every model's next request
# to its first replica again (GW-20).
function RoutingTable(routes::AbstractDict; prev::Union{RoutingTable, Nothing} = nothing)
    d = Dict{String, RouteEntry}()
    for (model, urls) in routes
        cp = sort(String[String(u) for u in urls])
        old = prev === nothing ? nothing : get(prev.routes, model, nothing)
        d[String(model)] = RouteEntry(cp, old === nothing ? Threads.Atomic{UInt64}(0) : old.next)
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

# Whether a model is known, without the cursor side effect `pick` has. Used by the control plane to
# reject a typo'd model name instead of storing an override that could never take effect.
has_model(t::RoutingTable, model::AbstractString) = haskey(t.routes, model)

# The worker URLs hosting `model` in the table's order, or `nothing` when no worker routes it. Unlike
# `pick` it leaves the round-robin cursor alone, so the metadata RPCs never shift request routing.
function replicas(t::RoutingTable, model::AbstractString)
    e = get(t.routes, model, nothing)
    return e === nothing ? nothing : copy(e.urls)
end

"""
    DiscoveredRoutes(; retention = ROUTE_RETENTION_SECONDS)

Holder for the current routing table, and for what it is built from: each worker's last answer to
discovery, and when each worker's current outage began. Autodiscovery records the answers and swaps
a freshly built table in atomically (`discover_routes!`); `pick` reads the current table lock free.
Starts empty until the first discovery round.

A worker's routes are its last answer, kept until it answers again. An answer replaces them, so a
model the worker no longer lists is dropped at once, but a call that did not answer (it timed out,
was refused, or failed) changes nothing: one slow answer must not take a worker's models out of
the table, and with them, when every worker missed the same round, every model, which clients see
as NOT_FOUND and do not retry (GW-11). While the prober finds a worker unreachable or not ready, its
answers are not taken either (a draining or failing worker can list nothing, which says nothing
about what it serves), and its routes are used only for models no ready worker serves, so a
request for such a model goes to it and fails over or comes back UNAVAILABLE, which clients retry.
Once the outage has lasted `retention` seconds, its routes are dropped, and a model nobody else
serves becomes NOT_FOUND.
"""
mutable struct DiscoveredRoutes
    @atomic current::RoutingTable
    const lock::ReentrantLock
    const answers::Dict{String, Vector{String}}    # worker url -> the models it last reported READY
    const down_since::Dict{String, Float64}        # worker url -> when its current outage began
    retention::Float64
end
DiscoveredRoutes(; retention::Real = ROUTE_RETENTION_SECONDS) = DiscoveredRoutes(
    RoutingTable(Dict{String, Vector{String}}()), ReentrantLock(),
    Dict{String, Vector{String}}(), Dict{String, Float64}(), Float64(retention)
)

current_table(d::DiscoveredRoutes) = @atomic d.current
swap_table!(d::DiscoveredRoutes, t::RoutingTable) = (@atomic d.current = t; nothing)

# The prober's verdict on a worker this round: ready, or not (not ready, unreachable, or hung). An
# outage starts at the first round a worker is not ready and ends at the next round it is.
function note_worker_ready!(d::DiscoveredRoutes, url::AbstractString, ready::Bool, now::Real = time())
    lock(d.lock) do
        ready ? delete!(d.down_since, url) : get!(d.down_since, String(url), Float64(now))
    end
    return nothing
end

# Record one round of discovery answers (`nothing` for a worker that did not answer) and swap in the
# table they make. Returns the table.
function record_answers!(d::DiscoveredRoutes, answers::AbstractDict, now::Real = time())
    return lock(d.lock) do
        for (url, names) in answers
            (names === nothing || haskey(d.down_since, url)) && continue
            d.answers[String(url)] = collect(String, names)
        end
        rebuild_table!(d, now)
    end
end

# Build and swap in the table from the recorded answers: every model a ready worker reported is
# routed to the ready workers that reported it; a model only workers in an outage reported is routed
# to them; and a worker whose outage has lasted past `retention` is forgotten. The caller holds
# `d.lock`.
function rebuild_table!(d::DiscoveredRoutes, now::Real = time())
    live = Dict{String, Vector{String}}()
    standby = Dict{String, Vector{String}}()
    for url in collect(keys(d.answers))
        since = get(d.down_since, url, nothing)
        if since !== nothing && now - since > d.retention
            delete!(d.answers, url)
            continue
        end
        into = since === nothing ? live : standby
        for m in d.answers[url]
            push!(get!(into, m, String[]), url)
        end
    end
    for (m, urls) in standby
        haskey(live, m) || (live[m] = urls)
    end
    table = RoutingTable(live; prev = current_table(d))
    swap_table!(d, table)
    return table
end

pick(d::DiscoveredRoutes, model::AbstractString) = pick(current_table(d), model)
nmodels(d::DiscoveredRoutes) = nmodels(current_table(d))
has_model(d::DiscoveredRoutes, model::AbstractString) = has_model(current_table(d), model)
replicas(d::DiscoveredRoutes, model::AbstractString) = replicas(current_table(d), model)
