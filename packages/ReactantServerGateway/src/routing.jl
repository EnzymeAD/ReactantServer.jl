# The model -> worker URLs map that ModelInfer consults on every request. A model may be hosted
# on more than one worker (replication); the table round-robins across the replicas. The table is
# rebuilt by autodiscovery (see health.jl) and swapped atomically into a `DiscoveredRoutes`
# holder; each table is itself immutable apart from the per-entry round-robin cursor, which
# mutates via an atomic counter, so reads are lock free.
#
# The table also carries the gateway's one record of which workers are healthy: the set of workers
# marked down (`down`). A request is only ever sent to a worker outside it, so the routes hold
# healthy workers only, and a model that only down workers serve is `held`: known, but not routed.
# Every scheduler reads health from here, through one atomic read of the table, so the routes and
# the down set a request sees always belong together.

mutable struct RouteEntry
    const urls::Vector{String}
    next::Threads.Atomic{UInt64}
end

# A held model: the down workers that serve it, and whether a request has been answered for it
# (and logged at info) during this outage. The flag lives as long as the model stays held, carried
# across rebuilds like a route's cursor, so a new outage starts with a fresh one.
struct HeldEntry
    urls::Vector{String}
    answered::Threads.Atomic{Bool}
end

struct RoutingTable
    routes::Dict{String, RouteEntry}      # model -> the healthy workers that serve it
    held::Dict{String, HeldEntry}         # model -> its workers, for a model only down workers serve
    down::Set{String}                     # the workers marked down (see `note_worker_ready!`)
end

# `prev`, when given, is the table this one replaces: a model it also routes keeps its round-robin
# cursor (the same atomic), so a rebuild every probe round does not send every model's next request
# to its first replica again (GW-20), and a model it also holds keeps its outage flag.
function RoutingTable(
        routes::AbstractDict; prev::Union{RoutingTable, Nothing} = nothing,
        held::AbstractDict = Dict{String, Vector{String}}(), down = Set{String}()
    )
    d = Dict{String, RouteEntry}()
    for (model, urls) in routes
        cp = sort(String[String(u) for u in urls])
        old = prev === nothing ? nothing : get(prev.routes, model, nothing)
        d[String(model)] = RouteEntry(cp, old === nothing ? Threads.Atomic{UInt64}(0) : old.next)
    end
    h = Dict{String, HeldEntry}()
    for (model, urls) in held
        old = prev === nothing ? nothing : get(prev.held, model, nothing)
        h[String(model)] = HeldEntry(
            sort(String[String(u) for u in urls]), old === nothing ? Threads.Atomic{Bool}(false) : old.answered
        )
    end
    return RoutingTable(d, h, Set{String}(String(u) for u in down))
end

"""
    first_held_answer!(table, model) -> Bool

Whether this is the first request answered for the held `model` during its current outage, marking
it answered. False for every later request of the outage, and for a model that is not held. The
gateway logs the first at info and the rest at debug, so an outage is reported once per model
without a line per request.
"""
function first_held_answer!(t::RoutingTable, model::AbstractString)
    e = get(t.held, model, nothing)
    e === nothing && return false
    return !Threads.atomic_xchg!(e.answered, true)
end

"""
    pick(table, model) -> Union{Vector{String},Nothing}

The healthy worker URLs hosting `model`, rotated so the round-robin choice is first and the
remaining replicas follow in failover order. Each call advances the cursor. Returns `nothing` when
the model is not known, and an empty vector when it is known but every worker that serves it is
down (the caller answers UNAVAILABLE without calling any worker).
"""
function pick(t::RoutingTable, model::AbstractString)
    e = get(t.routes, model, nothing)
    e === nothing && return haskey(t.held, model) ? String[] : nothing
    n = length(e.urls)
    n == 0 && return String[]
    # atomic_add! returns the previous value, so the first call yields start 0.
    start = Int(Threads.atomic_add!(e.next, UInt64(1)) % UInt64(n))
    return String[e.urls[(start + i) % n + 1] for i in 0:(n - 1)]
end

# The models the table knows: routed, or held while every worker that serves them is down.
nmodels(t::RoutingTable) = length(t.routes) + length(t.held)

# Whether a model is known, without the cursor side effect `pick` has. Used by the control plane to
# reject a typo'd model name instead of storing an override that could never take effect, and by
# the request path to label metrics. A held model is known: a worker serves it, it is just not up.
has_model(t::RoutingTable, model::AbstractString) = haskey(t.routes, model) || haskey(t.held, model)

# The healthy worker URLs hosting `model` in the table's order, an empty vector for a held model, or
# `nothing` when no worker serves it. Unlike `pick` it leaves the round-robin cursor alone, so the
# metadata RPCs never shift request routing.
function replicas(t::RoutingTable, model::AbstractString)
    e = get(t.routes, model, nothing)
    e === nothing && return haskey(t.held, model) ? String[] : nothing
    return copy(e.urls)
end

# Whether the worker at `url` may be sent a request: it is not marked down.
is_healthy(t::RoutingTable, url::AbstractString) = !(url in t.down)

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
as NOT_FOUND and do not retry (GW-11).

A worker is down from the moment the prober finds it unreachable, hung or not ready, or a request
finds its connection refused, until the prober next finds it ready (`note_worker_ready!`, the one
place that changes it). A down worker is sent no request: its answers are not taken (a draining or
failing worker can list nothing, which says nothing about what it serves), its models go to the
healthy workers that serve them, and a model that only down workers serve is held, so its requests
are answered UNAVAILABLE, which clients retry, without calling any worker. Once the outage has
lasted `retention` seconds, its routes are dropped, and a model nobody else serves becomes
NOT_FOUND.
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

"""
    note_worker_ready!(d, url, ready, now = time(); observed_at = now) -> Bool

Record a verdict on the worker at `url`: ready, or not (not ready, unreachable, hung, or its
connection refused). This is the one place a worker's health changes: the prober marks a worker
here every round, and the request path marks one down here when its connection is refused. An
outage starts at the first verdict that it is not ready and ends at the next ready verdict observed
after the outage began (`observed_at`, when the probe was sent), so a probe answered ready just
before a request found the connection refused does not clear the refusal. Only the prober gives a
ready verdict. A change of health swaps in a rebuilt routing table at once, so the next request
already sees it. Returns whether the worker's health changed.
"""
function note_worker_ready!(
        d::DiscoveredRoutes, url::AbstractString, ready::Bool, now::Real = time();
        observed_at::Real = now
    )
    return lock(d.lock) do
        changed = false
        if ready
            since = get(d.down_since, url, nothing)
            if since !== nothing && since <= observed_at
                delete!(d.down_since, url)
                changed = true
            end
        elseif !haskey(d.down_since, url)
            d.down_since[String(url)] = Float64(now)
            changed = true
        end
        changed && rebuild_table!(d, now)
        changed
    end
end

# Whether the worker at `url` may be sent a request right now.
is_healthy(d::DiscoveredRoutes, url::AbstractString) = is_healthy(current_table(d), url)

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

# Build and swap in the table from the recorded answers: every model a healthy worker reported is
# routed to the healthy workers that reported it; a model only down workers reported is held; and a
# worker whose outage has lasted past `retention` is forgotten. The caller holds `d.lock`.
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
    held = Dict(m => urls for (m, urls) in standby if !haskey(live, m))
    table = RoutingTable(live; prev = current_table(d), held = held, down = keys(d.down_since))
    swap_table!(d, table)
    return table
end

pick(d::DiscoveredRoutes, model::AbstractString) = pick(current_table(d), model)
first_held_answer!(d::DiscoveredRoutes, model::AbstractString) = first_held_answer!(current_table(d), model)
nmodels(d::DiscoveredRoutes) = nmodels(current_table(d))
has_model(d::DiscoveredRoutes, model::AbstractString) = has_model(current_table(d), model)
replicas(d::DiscoveredRoutes, model::AbstractString) = replicas(current_table(d), model)
