# LPT-packing scheduling: concentrate each model's traffic on few workers so the workers' batch
# coalescing has deep same-model queues to draw from, while balancing expected utilization across
# workers to avoid contention. Uniform spreading (round robin) is the worst case for coalescing
# when every model is loaded on every worker; concentration is the point of this mode.
#
# Each model is placed on a fixed, operator-configured number of distinct GPUs (`replicas`,
# default 1, never grown automatically under load). The packer chooses which GPUs by balancing two
# dimensions: expected compute utilization u_m = lambda_m * c_m (lambda_m the gateway-measured
# arrival rate EWMA, c_m the worker-reported per-request compute cost, both smoothed) and resident
# weight memory (ModelStatus.weight_nbytes) against each worker's weight-memory budget
# (ModelControlStatusResponse.weight_cache_max_bytes). The placement score is the max of the
# normalized compute and memory loads, so the packer simultaneously minimizes weight
# eviction/loading (memory pressure) and keeps every GPU busy (compute balance).
#
# Within a model's replica set the gateway routes each request to fill one replica's batch before
# moving to the next (see route_replica), using the worker-reported effective max batch size as the
# fill target, so the workers receive favorable groupings to coalesce. Coalescing itself stays at
# the worker; the gateway only routes and tracks a small per-replica outstanding counter.
#
# Repacks are driven by accumulated fleet compute, not wall-clock: the prober polls the workers
# every tick and a repack fires once the fleet has consumed `rebalance_compute_seconds`
# GPU-seconds since the last one, subject to a `min_rebalance_seconds` wall-clock floor.
#
# LPT-packing mode requires worker FIFO discipline and all models on all workers, verified as a hard
# failure at gateway startup (see verify_lpt_packing_preconditions!). Runtime drift degrades
# gracefully: a dead worker is excluded from placement until ready; a model temporarily missing
# from some workers gets a uniform distribution over its actual replicas with a warning.

# One model's placement: worker URLs with weights (1/k across its k replicas), sorted by URL.
# The weights are exported as metrics; request routing uses outstanding-batch counts, not weights.
const Placement = Vector{Tuple{String,Float64}}

"""
    compute_assignment(u, workers, prev; mem=Dict(), mem_cap=Dict(),
                       replicas=Dict(), default_replicas=1, hysteresis=0.1)
        -> Dict{String,Placement}

Pure assignment math (no I/O): greedy two-dimensional (vector) bin packing onto a fixed number of
distinct GPUs per model. Every model demands compute time (`u[m]`, GPU-seconds/second) and resident
weight memory (`mem[m]`, bytes); every worker offers compute capacity 1.0 and a weight-memory
budget (`mem_cap[url]`, bytes; absent or `<= 0` means unconstrained, i.e. all weights resident).
Models are placed in descending compute-utilization order (memory-heavier first among ties), each
onto exactly `k = clamp(get(replicas, m, default_replicas), 1, length(workers))` distinct workers
chosen to minimize resulting pressure: the maximum of a worker's normalized compute and memory
loads. The max-norm balances the two competing concerns: a memory-full worker stops attracting
models even when compute-idle (avoiding eviction churn), and a compute-hot worker stops attracting
them even with memory free (avoiding idle GPUs).

A model's compute is charged `u[m]/k` to each of its `k` workers, its weights the full footprint
to every one (replication costs memory everywhere it serves). For `k == 1`, hysteresis keeps the
model's previous placement unless moving improves its resulting pressure by more than the threshold,
since placement stability is what coalescing buys from. For `k > 1`, the `k` lowest-pressure workers
are chosen, previous members winning ties for stability, and weights are even (`1/k`). Workers no
longer present are ignored in `prev`. Cold models (no traffic yet) carry `u[m] == 0` and are placed
like any other, packed by memory; models absent from `u` entirely are not placed and the caller
routes them uniformly. Load never changes a model's `k`; replica count is fixed by configuration.
"""
function compute_assignment(u::Dict{String,Float64}, workers::Vector{String},
                            prev::Dict{String,Placement};
                            mem::Dict{String,Float64}=Dict{String,Float64}(),
                            mem_cap::Dict{String,Float64}=Dict{String,Float64}(),
                            replicas::Dict{String,Int}=Dict{String,Int}(),
                            default_replicas::Int=1, hysteresis::Float64=0.1)
    out = Dict{String,Placement}()
    isempty(workers) && return out
    cload = Dict{String,Float64}(w => 0.0 for w in workers)   # compute, normalized (capacity 1.0)
    mload = Dict{String,Float64}(w => 0.0 for w in workers)   # memory, bytes
    capof(w) = (c = get(mem_cap, w, Inf); c <= 0 ? Inf : c)
    # Pressure of worker `w` after placing compute `uc` and memory `wm` on it.
    score_after(w, uc, wm) = max(cload[w] + uc,
                                 capof(w) == Inf ? 0.0 : (mload[w] + wm) / capof(w))
    # Descending compute demand, then descending memory (pack the bulky cold models early),
    # ties broken by name for determinism.
    order = sort!(collect(keys(u)); by = m -> (-u[m], -get(mem, m, 0.0), m))
    for m in order
        um = u[m]
        wm = get(mem, m, 0.0)
        prev_pl = get(prev, m, nothing)
        k = clamp(get(replicas, m, default_replicas), 1, length(workers))
        if k == 1
            best = argmin(w -> score_after(w, um, wm), workers)
            chosen = best
            # Hysteresis: stick with the previous single placement unless the move improves the
            # model's resulting pressure by more than the threshold.
            if prev_pl !== nothing && length(prev_pl) == 1 && haskey(cload, prev_pl[1][1])
                wp = prev_pl[1][1]
                score_after(wp, um, wm) <= score_after(best, um, wm) * (1 + hysteresis) && (chosen = wp)
            end
            out[m] = [(chosen, 1.0)]
            cload[chosen] += um
            mload[chosen] += wm
        else
            # Place `k` replicas on the `k` lowest-pressure distinct workers (previous members win
            # ties for stability). Even weights keep every share identical and the sum at 1.
            # Weights are resident on every member, so each is charged the full footprint.
            prevset = prev_pl === nothing ? Set{String}() : Set(first.(prev_pl))
            ranked = sort(workers; by = w -> (score_after(w, um / k, wm), w in prevset ? 0 : 1, w))
            chosen = ranked[1:k]
            wshare = 1.0 / k
            for w in chosen
                cload[w] += um * wshare
                mload[w] += wm
            end
            out[m] = sort!([(w, wshare) for w in chosen]; by = first)
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------------

mutable struct LptPackingState
    # knobs (from GatewayConfig)
    hysteresis::Float64
    rate_halflife::Float64
    rebalance_compute_seconds::Float64       # fleet GPU-seconds consumed that triggers a repack
    min_rebalance_seconds::Float64           # wall-clock floor between repacks (0 = none)
    default_replicas::Int
    model_replicas::Dict{String,Int}         # per-model replica overrides (immutable after build)
    routing_fill_factor::Float64
    routing_policy::String                    # "fill" | "least_outstanding"
    # arrival counting: a copy-on-write snapshot dict of per-model atomic counters. Reads (the
    # request hot path) touch only the immutable snapshot; insertion of a new model swaps in a
    # copy under the lock.
    @atomic arrivals::Dict{String,Threads.Atomic{Int}}
    lock::ReentrantLock
    # EWMAs and the cumulative baselines for worker-counter deltas (model -> (compute, requests)).
    # Touched only on the prober task (poll/repack), so no locking needed.
    rate_ewma::Dict{String,Float64}          # requests/sec
    cost_ewma::Dict{String,Float64}          # compute seconds/request
    last_cum::Dict{String,Tuple{Float64,UInt64}}
    last_rebalance::Float64
    # compute-trigger accounting (prober task only): fleet GPU-seconds since the last repack and the
    # previous fleet cumulative-compute total used to derive the per-tick delta.
    compute_accum::Float64
    last_fleet_compute::Float64
    # routing metadata, swapped atomically each tick; the request hot path reads the snapshot.
    @atomic max_batch::Dict{String,Int}      # model -> effective max batch (largest compiled, capped)
    # the live assignment, swapped atomically; readers never lock
    @atomic assignment::Dict{String,Placement}
    # outstanding (in-flight) request counters, swapped atomically at repack so the hot path reads a
    # stable snapshot and increments the shared atomics inside. Per (model, worker) drives the fill
    # policy; per worker (multi-replica routing only) drives least_outstanding.
    @atomic outstanding::Dict{Tuple{String,String},Threads.Atomic{Int}}
    @atomic worker_outstanding::Dict{String,Threads.Atomic{Int}}
    # per-model selection lock (multi-replica models only), so a pick-and-reserve is atomic and
    # concurrent requests do not stampede onto the same replica.
    @atomic sel_locks::Dict{String,ReentrantLock}
    # label pairs / models previously exported to the gauges, zeroed when dropped
    exported::Set{Tuple{String,String}}
    replicas_exported::Set{String}
end

LptPackingState(cfg::GatewayConfig) = LptPackingState(
    cfg.hysteresis, cfg.rate_halflife_seconds, cfg.rebalance_compute_seconds,
    cfg.min_rebalance_seconds, cfg.default_replicas,
    Dict{String,Int}(name => mc.replicas for (name, mc) in cfg.models),
    cfg.routing_fill_factor, cfg.routing_policy,
    Dict{String,Threads.Atomic{Int}}(), ReentrantLock(),
    Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Tuple{Float64,UInt64}}(),
    0.0, 0.0, 0.0,
    Dict{String,Int}(), Dict{String,Placement}(),
    Dict{Tuple{String,String},Threads.Atomic{Int}}(), Dict{String,Threads.Atomic{Int}}(),
    Dict{String,ReentrantLock}(),
    Set{Tuple{String,String}}(), Set{String}())

# Hot path: one dict lookup on an immutable snapshot plus an atomic increment. Insertion of a
# never-seen model takes the lock once to swap in an extended copy.
function record_arrival!(s::LptPackingState, model::AbstractString)
    counters = @atomic s.arrivals
    c = get(counters, model, nothing)
    if c === nothing
        c = lock(s.lock) do
            cur = @atomic s.arrivals
            cc = get(cur, model, nothing)
            if cc === nothing
                nxt = copy(cur)
                cc = nxt[String(model)] = Threads.Atomic{Int}(0)
                @atomic s.arrivals = nxt
            end
            cc
        end
    end
    Threads.atomic_add!(c, 1)
    return nothing
end

# EWMA fold with halflife `h` over an interval `dt`.
_ewma(old::Float64, sample::Float64, dt::Float64, h::Float64) =
    (alpha = 1 - 2.0^(-dt / h); (1 - alpha) * old + alpha * sample)

# Poll every ready worker's ModelControlStatus concurrently and aggregate: per-model cumulative
# (compute, requests) summed across workers, the per-model weight footprint and effective max batch,
# each worker's weight-memory budget, the workers that answered, and the fleet's total cumulative
# compute (the compute-trigger signal). I/O only; no state mutation.
function _poll_workers(pool::ClientPool, ready_urls::Vector{String})
    sums = Dict{String,Tuple{Float64,UInt64}}()
    permodel_workers = Dict{String,Vector{String}}()
    mem = Dict{String,Float64}()                 # model -> resident weight bytes
    mem_cap = Dict{String,Float64}()             # worker -> on-demand weight budget (0 = unconstrained)
    max_batch = Dict{String,Int}()               # model -> effective max batch (max across workers)
    polled = String[]
    lk = ReentrantLock()
    @sync for url in ready_urls
        wc = get_clients(pool, url)
        wc === nothing && continue
        @async begin
            resp = fetch_control_status(wc)
            resp === nothing && return
            lock(lk) do
                push!(polled, url)
                mem_cap[url] = Float64(resp.weight_cache_max_bytes)
                for ms in resp.models
                    tc, rq = get(sums, ms.name, (0.0, UInt64(0)))
                    sums[ms.name] = (tc + ms.total_compute_seconds, rq + ms.requests_served)
                    mem[ms.name] = max(get(mem, ms.name, 0.0), Float64(ms.weight_nbytes))
                    max_batch[ms.name] = max(get(max_batch, ms.name, 0), Int(ms.max_batch_size))
                    push!(get!(permodel_workers, ms.name, String[]), url)
                end
            end
        end
    end
    fleet_compute = 0.0
    for (tc, _) in values(sums)
        fleet_compute += tc
    end
    return (; sums, permodel_workers, mem, mem_cap, max_batch, polled, fleet_compute)
end

# Rebuild the outstanding/worker-total counters and per-model selection locks to cover the new
# assignment, carrying over the live atomics for placements that persist (so in-flight counts
# survive a repack) and dropping the rest, then swap the snapshots in atomically. Lock objects are
# reused per model so the same object guards a model regardless of which snapshot a request read.
function _swap_outstanding!(s::LptPackingState, next::Dict{String,Placement})
    prev_out = @atomic s.outstanding
    prev_wout = @atomic s.worker_outstanding
    prev_locks = @atomic s.sel_locks
    out = Dict{Tuple{String,String},Threads.Atomic{Int}}()
    wout = Dict{String,Threads.Atomic{Int}}()
    locks = Dict{String,ReentrantLock}()
    for (m, placement) in next
        length(placement) > 1 && (locks[m] = get(prev_locks, m, ReentrantLock()))
        for (w, _) in placement
            out[(m, w)] = get(prev_out, (m, w), Threads.Atomic{Int}(0))
            haskey(wout, w) || (wout[w] = get(prev_wout, w, Threads.Atomic{Int}(0)))
        end
    end
    @atomic s.outstanding = out
    @atomic s.worker_outstanding = wout
    @atomic s.sel_locks = locks
    return nothing
end

# Recompute and install a new assignment from a fresh poll: fold arrival-rate and compute-cost
# EWMAs, build expected utilization, run compute_assignment with the configured replica counts,
# swap in the assignment and outstanding counters, reset the compute accumulator, and export
# metrics. Runs on the prober task only. A model not reported by every ready worker (runtime drift)
# gets a uniform placement over the workers that do serve it, with a warning.
function _repack!(s::LptPackingState, poll, metrics::Union{GatewayMetrics,Nothing})
    now = time()
    dt = s.last_rebalance == 0.0 ? s.rate_halflife : max(now - s.last_rebalance, 1e-3)
    s.last_rebalance = now

    # Arrival rates.
    counters = @atomic s.arrivals
    for (m, c) in counters
        n = Threads.atomic_xchg!(c, 0)
        s.rate_ewma[m] = _ewma(get(s.rate_ewma, m, 0.0), n / dt, dt, s.rate_halflife)
    end

    # Worker-reported costs: delta the per-model cumulative (compute, requests) against the previous
    # repack. A negative delta means a worker restarted (counters reset); re-baseline and skip.
    for (m, (tc, rq)) in poll.sums
        prev_tc, prev_rq = get(s.last_cum, m, (0.0, UInt64(0)))
        dtc, drq = tc - prev_tc, Int(rq) - Int(prev_rq)
        s.last_cum[m] = (tc, rq)
        (dtc < 0 || drq < 0) && continue          # worker restart: re-baseline only
        drq > 0 && (s.cost_ewma[m] = _ewma(get(s.cost_ewma, m, dtc / drq), dtc / drq, dt, s.rate_halflife))
    end

    @atomic s.max_batch = poll.max_batch

    # Expected utilization. Every fully-replicated model is packed, including cold ones (no traffic
    # yet, u = 0): they still occupy weight memory, so the packer gives each a concentrated home
    # placed by the memory dimension. A model missing from some polled workers (runtime drift)
    # routes uniformly over its actual replicas until the fleet converges.
    full = Dict{String,Float64}()
    drifted = Dict{String,Placement}()
    nready = length(poll.polled)
    for (m, ws) in poll.permodel_workers
        if length(ws) == nready
            r = get(s.rate_ewma, m, 0.0)
            c = get(s.cost_ewma, m, 0.0)
            full[m] = (r > 0 && c > 0) ? r * c : 0.0
        else
            @warn "lpt_packing: model not on all ready workers; routing uniformly over its replicas" model = m replicas = length(ws) ready = nready
            drifted[m] = [(w, 1.0 / length(ws)) for w in sort(ws)]
        end
    end

    prev = @atomic s.assignment
    next = compute_assignment(full, sort(poll.polled), prev;
                              mem=poll.mem, mem_cap=poll.mem_cap,
                              replicas=s.model_replicas, default_replicas=s.default_replicas,
                              hysteresis=s.hysteresis)
    merge!(next, drifted)
    @atomic s.assignment = next
    _swap_outstanding!(s, next)

    s.compute_accum = 0.0
    s.last_fleet_compute = poll.fleet_compute

    # Memory oversubscription warning: when a worker's assigned weight footprint exceeds its
    # on-demand budget the packing is infeasible (total weights outgrew the fleet); the worker's
    # LRU cache degrades gracefully, but the operator should know.
    assigned_mem = Dict{String,Float64}(w => 0.0 for w in poll.polled)
    for (m, placement) in next, (w, _) in placement
        haskey(assigned_mem, w) && (assigned_mem[w] += get(poll.mem, m, 0.0))
    end
    for (w, bytes) in assigned_mem
        cap = get(poll.mem_cap, w, 0.0)
        cap > 0 && bytes > cap &&
            @warn "lpt_packing: assigned weight footprint exceeds the worker's on-demand budget; expect eviction churn" worker = w assigned = Base.format_bytes(round(Int, bytes)) budget = Base.format_bytes(round(Int, cap))
    end

    if metrics !== nothing
        out_snap = @atomic s.outstanding
        live = Set{Tuple{String,String}}()
        live_models = Set{String}()
        for (m, placement) in next
            push!(live_models, m)
            set_model_replicas!(metrics, m, length(placement))
            for (w, weight) in placement
                set_placement_weight!(metrics, m, w, weight)
                a = get(out_snap, (m, w), nothing)
                set_replica_outstanding!(metrics, m, w, a === nothing ? 0 : a[])
                push!(live, (m, w))
            end
        end
        for (m, w) in setdiff(s.exported, live)
            set_placement_weight!(metrics, m, w, 0.0)
            set_replica_outstanding!(metrics, m, w, 0)
        end
        for m in setdiff(s.replicas_exported, live_models)
            set_model_replicas!(metrics, m, 0)
        end
        s.exported = live
        s.replicas_exported = live_models
        for (m, um) in full
            set_model_utilization!(metrics, m, um)
        end
    end
    return nothing
end

"""
    rebalance!(s, pool, ready_urls, metrics) -> nothing

Force a repack now: poll the ready workers and recompute the assignment unconditionally. Used at
startup (so the first requests already route by packing) and by tests. The periodic, compute-driven
path is [`tick_packing!`](@ref).
"""
function rebalance!(s::LptPackingState, pool::ClientPool, ready_urls::Vector{String},
                    metrics::Union{GatewayMetrics,Nothing}=nothing)
    _repack!(s, _poll_workers(pool, ready_urls), metrics)
    return nothing
end

"""
    tick_packing!(s, pool, ready_urls, metrics) -> nothing

One prober tick: poll the ready workers, refresh the routing metadata, accumulate the fleet's
consumed compute, and repack only when the accumulated compute crosses `rebalance_compute_seconds`
and the `min_rebalance_seconds` wall-clock floor has elapsed. The cheap per-tick work is the poll
and the accumulator; the EWMA fold and `compute_assignment` run only on a triggered repack.
"""
function tick_packing!(s::LptPackingState, pool::ClientPool, ready_urls::Vector{String},
                       metrics::Union{GatewayMetrics,Nothing}=nothing)
    poll = _poll_workers(pool, ready_urls)
    @atomic s.max_batch = poll.max_batch         # keep routing metadata fresh between repacks
    if s.last_fleet_compute == 0.0
        s.last_fleet_compute = poll.fleet_compute   # first observation: baseline only
    else
        delta = poll.fleet_compute - s.last_fleet_compute
        s.last_fleet_compute = poll.fleet_compute
        delta > 0 && (s.compute_accum += delta)     # negative delta = worker restart: re-baseline
    end
    now = time()
    triggered = s.compute_accum >= s.rebalance_compute_seconds &&
                (s.last_rebalance == 0.0 || now - s.last_rebalance >= s.min_rebalance_seconds)
    triggered && _repack!(s, poll, metrics)
    return nothing
end

# The effective per-replica fill quantum for `model`: the worker-reported max batch scaled by the
# fill factor, at least 1. When the max batch is unknown (uncapped, no compiled batch reported) the
# quantum is 1, which degrades the fill policy to least-outstanding within the replica set.
function _fill_quantum(s::LptPackingState, model::AbstractString)
    maxB = get(@atomic(s.max_batch), model, 0)
    return maxB <= 0 ? 1 : max(1, round(Int, s.routing_fill_factor * maxB))
end

"""
    route_replica(s, model) -> Union{Nothing, Tuple{Vector{String}, Counters}}

Order a model's replicas for dispatch and reserve the chosen one. Returns `nothing` when the model
has no placement yet (cold or unknown); the caller falls back to round robin. Otherwise returns the
ordered replica URLs (the chosen worker first, the rest as failover) and `Counters`, the reserved
outstanding atomics to release when the request completes (`nothing` for a single-replica model,
which takes the fast path with no counter work). The chosen replica is the one that fills its batch
first: under the `fill` policy, the replica with the fewest full quanta of in-flight requests
(ties broken toward the replica closest to completing its current quantum, so one fills before the
next); under `least_outstanding`, the worker with the least total in-flight work. The reservation
(an atomic increment under the per-model selection lock) makes concurrent selections see the choice,
so requests do not stampede onto the same replica.
"""
function route_replica(s::LptPackingState, model::AbstractString)
    placement = get(@atomic(s.assignment), model, nothing)
    placement === nothing && return nothing
    n = length(placement)
    n == 0 && return nothing
    n == 1 && return (String[placement[1][1]], nothing)

    workers = String[p[1] for p in placement]
    lk = get(@atomic(s.sel_locks), model, nothing)
    lk === nothing && return (workers, nothing)   # mid-repack drift: route in order, untracked
    out_snap = @atomic s.outstanding
    wout_snap = @atomic s.worker_outstanding
    Q = _fill_quantum(s, model)

    res = lock(lk) do
        cobjs = Vector{Threads.Atomic{Int}}(undef, n)
        outs = Vector{Int}(undef, n)
        for i in 1:n
            a = get(out_snap, (model, workers[i]), nothing)
            a === nothing && return nothing       # snapshot drift: bail to untracked routing
            cobjs[i] = a
            outs[i] = a[]
        end
        order = collect(1:n)
        if s.routing_policy == "least_outstanding"
            wouts = Vector{Int}(undef, n)
            for i in 1:n
                wa = get(wout_snap, workers[i], nothing)
                wouts[i] = wa === nothing ? 0 : wa[]
            end
            sort!(order; by = i -> (wouts[i], outs[i], workers[i]))
        else
            sort!(order; by = i -> (fld(outs[i], Q), -outs[i], workers[i]))
        end
        chosen = order[1]
        Threads.atomic_add!(cobjs[chosen], 1)
        wobj = get(wout_snap, workers[chosen], nothing)
        wobj === nothing || Threads.atomic_add!(wobj, 1)
        ordered = String[workers[i] for i in order]
        return (ordered, cobjs[chosen], wobj)
    end
    res === nothing && return (workers, nothing)
    ordered, mwc, wobj = res
    return (ordered, (mwc, wobj))
end

# Release a reservation made by route_replica (decrement the per-replica and per-worker counters).
# Robust to the failure path: called once in a finally regardless of how the dispatch ended.
function _release_route!(counters)
    counters === nothing && return nothing
    mwc, wobj = counters
    Threads.atomic_sub!(mwc, 1)
    wobj === nothing || Threads.atomic_sub!(wobj, 1)
    return nothing
end

"""
    verify_lpt_packing_preconditions!(pool) -> nothing

LPT-packing mode's hard startup checks: every configured worker must be reachable over the control
plane, report FIFO scheduling discipline, and serve an identical model set (load-all). Any
violation raises with the offending worker named.
"""
function verify_lpt_packing_preconditions!(pool::ClientPool)
    statuses = Dict{String,Any}()
    for wc in all_clients(pool)
        resp = fetch_control_status(wc)
        resp === nothing &&
            error("lpt_packing scheduling: worker $(wc.url) is unreachable over the control plane; all workers must be up at startup")
        statuses[wc.url] = resp
    end
    for (url, resp) in statuses
        resp.discipline == "fifo" ||
            error("lpt_packing scheduling requires worker FIFO discipline; worker $url reports '$(resp.discipline)' (set scheduler.discipline: fifo in the node file)")
    end
    sets = Dict(url => sort([ms.name for ms in resp.models]) for (url, resp) in statuses)
    ref_url = first(keys(sets))
    for (url, names) in sets
        names == sets[ref_url] ||
            error("lpt_packing scheduling requires all models on all workers; $url serves $(length(names)) models but $ref_url serves $(length(sets[ref_url])) (model sets differ)")
    end
    return nothing
end
