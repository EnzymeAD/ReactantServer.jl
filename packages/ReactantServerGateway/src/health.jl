# Periodic worker readiness probe and route autodiscovery. Each round probes every endpoint's
# ServerReady (driving /readyz: the aggregate is ready when at least one worker reports ready) and
# its RepositoryIndex (driving the routing table: the model -> ready-endpoints map is rebuilt and
# swapped in atomically). A control-plane pin/unpin or a node restart is picked up on the next
# round. Mirrors and extends the Go gateway's internal/health.

const HEALTH_INTERVAL_SECONDS = 10.0

# Watchdog ceiling for one probe/discovery call. The gRPC clients carry their own deadline, but
# that deadline is enforced by the shared libcurl multi handle, and the multi's timer/socket
# driving can wedge after a burst of failed connects (e.g. the window where every worker is
# still compiling). A wedged multi hangs every call forever; an unbounded call inside the
# prober's @sync would then wedge the round, and with it /readyz, permanently. A timed-out call
# counts as not-ready / no-routes and is retried next round (the abandoned task is leaked,
# bounded to worker-churn windows).
const PROBE_TIMEOUT_SECONDS = 8.0

# A watchdog timeout is the wedge signature only as a *regression*: the client's own 5s deadline
# can be missed when the multi handle stops being driven, which a fresh process recovers. But an
# all-timeout round is NOT a wedge during warmup, when workers are simply slow or busy (e.g.
# compiling a large model set) and have never answered yet, nor when a heavily loaded host
# starves the prober's tasks. Restarting then is useless churn. So the gateway exits to be
# restarted only after it has talked to a worker at least once (a true regression) and then sees
# this many consecutive all-timeout rounds. Override with REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS;
# 0 disables.
const WEDGE_EXIT_ROUNDS = 3

# Returns (value, timed_out). On timeout the abandoned task keeps running detached; `default`
# is returned in its place. `level` sets the log level of the timeout message: the default `:warn`
# suits the runtime prober (a worker that was ready and stops answering is a real event), while the
# startup precondition wait passes `:debug` because timeouts there are the expected warmup window
# (workers compiling before their control plane answers) and are already summarized by the caller.
#
# A call that fails at its own gRPC deadline also counts as timed out. Since gRPCClient 1.1 a call
# on a stalled connection no longer hangs past the watchdog: the client's deadline watchdog fails it
# at the deadline (5s for the probes, under this 8s ceiling). Treating only the watchdog as a hang
# would then never reset the poisoned connection, and every later call would reuse it and fail the
# same way. `f` must let a deadline error escape (see `_deadline_exceeded`); other errors are a
# fast refuse and return `default` with timed_out false.
function _bounded(f, secs, default, what::String, url::String; level::Symbol = :warn)
    t = @async f()
    if timedwait(() -> istaskdone(t), secs) !== :ok
        _log_hung("health: $what timed out; treating as unavailable", level, url, secs)
        return default, true
    end
    try
        return fetch(t), false
    catch e
        if _deadline_exceeded(e isa TaskFailedException ? e.task.exception : e)
            _log_hung("health: $what hit its deadline; treating as unavailable", level, url, secs)
            return default, true
        end
        return default, false
    end
end

_log_hung(msg, level, url, secs) =
    level === :debug ? (@debug msg worker = url timeout_s = secs) :
    level === :info ? (@info msg worker = url timeout_s = secs) :
    (@warn msg worker = url timeout_s = secs)

mutable struct HealthProber
    pool::ClientPool
    metrics::GatewayMetrics
    admin::AdminServer
    routes::Union{DiscoveredRoutes, Nothing}
    scheduler::GatewayScheduler               # ticked each round via scheduler_tick! (no-op unless lpt_packing)
    # Per-model routing metadata (batch axis and measured costs), refreshed from this round's
    # control-status poll and read by whichever scheduler routes by work. See routing_meta.jl.
    meta::RoutingMeta
    interval::Float64
    running::Threads.Atomic{Bool}
    task::Union{Task, Nothing}
    wedged_rounds::Int                        # consecutive all-timeout rounds since last response
    wedge_exit_rounds::Int                    # exit(1) after this many; 0 disables
    ever_responsive::Bool                     # a worker has answered (non-timeout) at least once
    # The scheduler forced-work sequence this task has already consumed, so an operator-forced repack
    # wakes the prober early (see the wait in `start_prober!`) instead of waiting out `interval`.
    # Prober-task-owned; plain.
    last_repack_seq::Int
    # Per-worker outage bookkeeping for the recovery message, keyed by url and prober-task owned: the
    # workers that have reported ready at least once, and for each one currently not ready, when the
    # outage began and how many rounds it has missed.
    ever_ready::Set{String}
    outages::Dict{String, Tuple{Float64, Int}}
    # The gateway's shared-memory bookkeeping, for the region names a worker missed while it was
    # down (see `catch_up_shm!`), or nothing (tests that build a prober alone).
    gate::Union{RegisterGate, Nothing}
end

function HealthProber(
        pool::ClientPool, metrics::GatewayMetrics, admin::AdminServer,
        routes::Union{DiscoveredRoutes, Nothing} = nothing;
        scheduler::GatewayScheduler = RoundRobinScheduler(),
        meta::RoutingMeta = RoutingMeta(),
        gate::Union{RegisterGate, Nothing} = nothing,
        interval::Real = _health_interval_seconds(),
        wedge_exit_rounds::Integer =
            parse(
            Int, get(
                ENV, "REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS",
                string(WEDGE_EXIT_ROUNDS)
            )
        )
    )
    return HealthProber(
        pool, metrics, admin, routes, scheduler, meta, Float64(interval),
        Threads.Atomic{Bool}(true), nothing, 0, Int(wedge_exit_rounds), false, 0,
        Set{String}(), Dict{String, Tuple{Float64, Int}}(), gate
    )
end

# The probe interval, overridable so a test (or a deployment that wants a tighter control-plane
# latency) does not have to wait the full default between rounds.
function _health_interval_seconds()
    v = tryparse(Float64, get(ENV, "REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS", ""))
    return (v === nothing || v <= 0) ? HEALTH_INTERVAL_SECONDS : v
end

# Query every endpoint's ready models concurrently, record the answers in `routes`, and swap in the
# model -> endpoints routing table they make; returns it. An endpoint that does not answer keeps the
# routes it last reported (see `DiscoveredRoutes` for how long).
function discover_routes!(routes::DiscoveredRoutes, pool::ClientPool)
    workers = all_clients(pool)
    answers = Dict{String, Union{Nothing, Vector{String}}}()
    lk = ReentrantLock()
    @sync for wc in workers
        @async begin
            names, _ = _bounded(
                () -> discover_models(wc), PROBE_TIMEOUT_SECONDS, nothing,
                "RepositoryIndex discovery", wc.url
            )
            lock(() -> (answers[wc.url] = names), lk)
        end
    end
    return record_answers!(routes, answers)
end

function _check_once(p::HealthProber)
    workers = all_clients(p.pool)
    results = Vector{Bool}(undef, length(workers))
    timeouts = Vector{Bool}(undef, length(workers))
    elapsed = Vector{Float64}(undef, length(workers))
    sent = Vector{Float64}(undef, length(workers))
    @sync for (i, wc) in enumerate(workers)
        @async begin
            t0 = sent[i] = monotonic_time()
            results[i], timeouts[i] =
                _bounded(
                () -> probe_ready(wc), PROBE_TIMEOUT_SECONDS, false,
                "ServerReady probe", wc.url
            )
            elapsed[i] = monotonic_time() - t0
        end
    end
    # A worker coming back (marked down, now answering ready) first drops the shared-memory regions
    # it missed while it was down; until that succeeds it stays down (see `catch_up_shm!`).
    caught_up = catch_up_shm!(p, workers, results)
    any_ready = false
    any_responsive = false
    for (i, wc) in enumerate(workers)
        set_worker_ready!(p.metrics, wc.url, results[i])
        observe_probe!(p.metrics, wc.url, elapsed[i])
        any_ready |= results[i]
        any_responsive |= !timeouts[i]   # answered within the watchdog (ready or not)
        # A hung probe (timed out, not a fast refuse) means the worker was caught mid-stall, and the
        # probe connection may be one that never answered, which every later probe would wait behind
        # until its deadline (see `reset_if_parked!`); drop it so this round's discovery and the next
        # probe reconnect fresh. The probes have a handle of their own, so this touches no inference.
        timeouts[i] && reset_probe_connection!(wc)
        # A worker back from an outage may be a new process at the same address, whose predecessor's
        # pooled connections are dead; it gets fresh ones before it is marked healthy, so nothing is
        # routed or staged over them (see `reset_returning_connections!`). Otherwise, a worker that
        # answered ready on the probe handle while its inference calls keep reaching their deadline
        # unanswered has a parked inference connection; reset that one.
        if results[i] && caught_up[i] && _returning(p, wc.url)
            reset_returning_connections!(wc)
        else
            results[i] && reset_if_parked!(wc)
        end
        _track_outage!(p, wc.url, results[i])
        # A ready answer clears only an outage that began before the probe was sent: a request that
        # found the worker's connection refused while this probe was out keeps it down until the next.
        p.routes === nothing || note_worker_ready!(p.routes, wc.url, results[i] && caught_up[i]; observed_at = sent[i])
    end
    set_ready!(p.admin, any_ready)
    p.ever_responsive |= any_responsive
    _track_wedge!(p, !isempty(workers) && all(timeouts))
    if p.routes !== nothing
        table = discover_routes!(p.routes, p.pool)
        set_routing_size!(p.metrics, nmodels(table))
    end
    # Control-plane round, for the schedulers that route by work: ONE ModelControlStatus poll shared
    # by the routing-metadata cache (per-model batch axis and measured costs) and the scheduler tick,
    # so adding a consumer never adds an RPC. Skipped entirely when nothing needs it, which keeps a
    # round_robin fleet at exactly one probe round-trip per worker as before.
    #
    # Scheduler tick: lpt_packing folds the poll to refresh routing metadata and accumulate consumed
    # compute, repacking once the fleet has consumed the configured compute budget or the set of
    # healthy workers changed (other schedulers are no-ops). Placement is computed over the healthy
    # workers, read from the routing table's record of who is down, the one the request path reads,
    # so the two never disagree; a dead worker drops out until it recovers.
    ready_urls = String[
        wc.url for (i, wc) in enumerate(workers)
            if (p.routes === nothing ? results[i] : is_healthy(p.routes, wc.url))
    ]
    if !isempty(ready_urls)
        try
            poll = nothing
            if needs_routing_meta(p.scheduler)
                poll = poll_workers(p.pool, ready_urls)
                refresh_routing_meta!(p.meta, poll)
            end
            scheduler_tick!(p.scheduler, p.pool, ready_urls, p.metrics, poll)
        catch e
            @warn "scheduler tick failed; keeping the previous state" exception = e
        end
    end
    return any_ready
end

"""
    catch_up_shm!(prober, workers, ready) -> Vector{Bool}

For each worker that is marked down but answered this round's probe ready, unregister every
shared-memory region name it missed while it was down (`note_missed!`), all of them at once when it
missed an unregister of every region. Returns, per worker, whether it may be marked ready: false only
for a returning worker whose unregisters did not all succeed, which stays down and is tried again
next round.

Why: register and unregister fan out to the healthy workers only. A worker that was marked down but
stayed alive (hung for a while, then recovered without a restart) still holds what it had
registered. If the client meanwhile unregistered a name, unlinked its object and registered the name
again for a new object, a request under that name on the returning worker would read the old
object's inputs and write outputs the client never sees, and report success, until the worker's own
sweep (`shm_reap!`, every 60 s and at its next registration) dropped the stale registration. Dropping
the missed names before the worker takes requests closes that: it holds none of them, so a request
under one fails with FAILED_PRECONDITION and the client registers again, which now reaches it. A
restarted worker holds nothing, and the unregisters are no-ops. ReactantServerClient names each pool
uniquely and never reuses a name, so for it this only costs a register; it matters to a client that
reuses region names, as stock Triton clients may.
"""
function catch_up_shm!(p::HealthProber, workers, ready::AbstractVector{Bool})
    ok = fill(true, length(workers))
    (p.gate === nothing || p.routes === nothing) && return ok
    @sync for (i, wc) in enumerate(workers)
        (ready[i] && !is_healthy(p.routes, wc.url)) || continue
        names = missed_names(p.gate, wc.url)
        isempty(names) && continue
        @async begin
            todo = "" in names ? [""] : sort(names)
            err = nothing
            for name in todo
                err = _catch_up_call(wc, name)
                err === nothing || break
            end
            if err === nothing
                clear_missed!(p.gate, wc.url, names)
                @info "health: a returning worker dropped the shared-memory regions it missed while down" worker = wc.url regions = todo
            else
                ok[i] = false
                @warn "health: a returning worker could not drop the shared-memory regions it missed while down; it stays out this round" worker = wc.url regions = todo error = err
            end
        end
    end
    return ok
end

# One catch-up unregister, watchdog-bounded like a probe: `nothing` on success, else what went wrong.
# A call that hangs drops the probe connection, as a hung probe does.
function _catch_up_call(wc::WorkerClients, name::AbstractString)
    t = @async try
        invoke_shm_catch_up(wc, _encode_unregister(name))
        nothing
    catch e
        e
    end
    if timedwait(() -> istaskdone(t), PROBE_TIMEOUT_SECONDS) !== :ok
        reset_probe_connection!(wc)
        return "no answer within $(PROBE_TIMEOUT_SECONDS) s"
    end
    e = fetch(t)
    e === nothing && return nothing
    _deadline_exceeded(e) && reset_probe_connection!(wc)
    return sprint(showerror, e)
end

# Whether the worker at `url` is coming back from an outage: the routing table has it marked down (by
# the prober, or by a request that found its connection refused), or the prober has counted rounds it
# missed. Read before `_track_outage!` clears the count and `note_worker_ready!` the mark.
_returning(p::HealthProber, url::String) =
    haskey(p.outages, url) || (p.routes !== nothing && !is_healthy(p.routes, url))

# Outage accounting: a worker that was ready and then missed one or more rounds (a probe timeout,
# which already logs a warning, or a fast refuse or not-ready answer, which does not) logs one line
# when it reports ready again, so a reader of the log can see that the outage ended and how long it
# lasted. A worker that has never been ready is still warming up; its first ready is not a recovery.
# `p` is the prober, or anything with its `ever_ready` and `outages` fields (the tests pass a stand-in).
function _track_outage!(p, url::String, ready::Bool, now::Float64 = monotonic_time())
    if ready
        outage = pop!(p.outages, url, nothing)
        if outage !== nothing
            since, rounds = outage
            @info "health: worker ready again" worker = url down_s = round(now - since; digits = 1) missed_rounds = rounds
        end
        push!(p.ever_ready, url)
    elseif url in p.ever_ready
        since, rounds = get(p.outages, url, (now, 0))
        p.outages[url] = (since, rounds + 1)
    end
    return nothing
end

# Wedge accounting: a round where EVERY probe missed even the watchdog. This is treated as a
# wedged client stack (and the gateway exits for a fresh restart) ONLY after a worker has
# answered at least once (`ever_responsive`) — i.e. a genuine regression from working to wedged.
# Before that, an all-timeout round is just warmup (workers slow/busy/compiling) or a loaded
# host starving the prober; restarting then is useless churn, so it is ignored. Only the probes
# count, and they run on handles of their own (see `WorkerClients`), so inference queued at busy
# workers, however much, cannot make a round look wedged.
function _track_wedge!(p::HealthProber, all_timed_out::Bool)
    p.wedged_rounds = all_timed_out ? p.wedged_rounds + 1 : 0
    p.ever_responsive || return nothing
    if p.wedge_exit_rounds > 0 && p.wedged_rounds >= p.wedge_exit_rounds
        @error "health: every ServerReady probe timed out for $(p.wedged_rounds) consecutive rounds after previously reaching a worker; the gRPC client stack is wedged. Exiting so the supervisor restarts the gateway." rounds = p.wedged_rounds
        exit(1)
    end
    return nothing
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
            _await_next_round(p)
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

# Sleep until the next round, returning early when the scheduler has forced-work pending so an
# operator-triggered repack lands in about a poll interval rather than after the full probe interval.
#
# The sequence is consumed whether or not the repack can actually run: if no worker is ready the
# scheduler's completed counter never advances, and a predicate that kept firing on the same request
# would spin this loop at the poll rate forever. This way each request buys at most one extra round.
function _await_next_round(p::HealthProber)
    seq = scheduler_repack_seq(p.scheduler)
    if seq > p.last_repack_seq
        p.last_repack_seq = seq
        return nothing
    end
    timedwait(p.interval; pollint = 0.05) do
        !p.running[] || scheduler_repack_seq(p.scheduler) > p.last_repack_seq
    end
    return nothing
end

stop_prober!(p::HealthProber) = (p.running[] = false; nothing)
