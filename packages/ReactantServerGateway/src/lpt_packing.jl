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
# fill target, so the workers receive favorable groupings to coalesce. The routing_policy decides
# only which replica a fresh batch opens on: `fill_rr` round-robins it across the set and
# `fill_least` opens on the least compute-loaded GPU (so a model's batches avoid GPUs busy with
# other models). Coalescing itself stays at the worker; the gateway only routes and tracks small
# per-replica and per-worker outstanding/load counters. (Spreading without concentration is the
# separate `least_outstanding` scheduling mode, not an lpt_packing policy; see scheduler.jl.)
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

# Watchdog ceiling for one control-status poll. The gateway's clients share one libcurl multi
# handle with a 16-slot request semaphore (GRPC_MAX_STREAMS); if its driving stalls, in-flight
# calls never release their slot and new calls block forever, so every poll is bounded and a worker
# that exceeds this is treated as not-polled.
const POLL_TIMEOUT_SECONDS = 8.0

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
# Optional per-repack diagnostics filled by `compute_assignment` (single-replica hysteresis only).
# `held` counts models whose lowest-pressure worker differed from their current one but that stayed
# put because the improvement was under the hysteresis threshold. `max_improvement` is the largest
# available relative pressure reduction across single-replica models (taken or not), so a caller can
# see where the fleet sits relative to the threshold even when nothing moved.
mutable struct RepackStats
    held::Int
    max_improvement::Float64
end
RepackStats() = RepackStats(0, 0.0)

function compute_assignment(u::Dict{String,Float64}, workers::Vector{String},
                            prev::Dict{String,Placement};
                            mem::Dict{String,Float64}=Dict{String,Float64}(),
                            mem_cap::Dict{String,Float64}=Dict{String,Float64}(),
                            replicas::Dict{String,Int}=Dict{String,Int}(),
                            default_replicas::Int=1, hysteresis::Float64=0.1,
                            stats::Union{Nothing,RepackStats}=nothing)
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
                bestscore = score_after(best, um, wm)
                prevscore = score_after(wp, um, wm)
                if stats !== nothing
                    impr = bestscore > 0 ? (prevscore / bestscore - 1.0) : 0.0
                    impr > stats.max_improvement && (stats.max_improvement = impr)
                end
                if prevscore <= bestscore * (1 + hysteresis)
                    chosen = wp
                    stats !== nothing && best != wp && (stats.held += 1)
                end
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

mutable struct LptPackingState <: GatewayScheduler
    # knobs (from GatewayConfig)
    hysteresis::Float64
    rate_halflife::Float64
    rebalance_compute_seconds::Float64       # fleet GPU-seconds consumed that triggers a repack
    min_rebalance_seconds::Float64           # wall-clock floor between repacks (0 = none)
    default_replicas::Int
    model_replicas::Dict{String,Int}         # per-model replica overrides (immutable after build)
    routing_fill_factor::Float64
    routing_policy::String                    # "fill_rr" | "fill_least"
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
    # Per-model measured per-request compute cost (GPU-seconds/request), published from cost_ewma at
    # each repack so the request hot path can read it without racing the prober. Drives fill_least's
    # compute-weighted load.
    @atomic cost_snapshot::Dict{String,Float64}
    # the live assignment, swapped atomically; readers never lock
    @atomic assignment::Dict{String,Placement}
    # outstanding (in-flight) request counters, swapped atomically at repack so the hot path reads a
    # stable snapshot and increments the shared atomics inside. Per (model, worker) drives the fill
    # quantum.
    @atomic outstanding::Dict{Tuple{String,String},Threads.Atomic{Int}}
    # Per-worker in-flight compute load: the sum over a worker's in-flight requests of each routed
    # model's cost weight (GPU-seconds). Drives fill_least's least-loaded batch-start choice. Every
    # routed request, single- or multi-replica, contributes, so the load reflects all models.
    @atomic worker_load::Dict{String,Threads.Atomic{Float64}}
    # per-model selection lock (multi-replica models only), so a pick-and-reserve is atomic and
    # concurrent requests do not stampede onto the same replica.
    @atomic sel_locks::Dict{String,ReentrantLock}
    # per-model round-robin cursor (multi-replica models only): the next replica index a fresh batch
    # opens on under fill_rr. Advanced only when a batch actually starts (see route_replica).
    @atomic rr_cursor::Dict{String,Threads.Atomic{Int}}
    # label pairs / models previously exported to the gauges, zeroed when dropped
    exported::Set{Tuple{String,String}}
    replicas_exported::Set{String}
    # memory-compaction cadence (prober task only): mode, interval in repacks, and the number of
    # repacks since the last fan-out (so the first placement-changing repack at or after the interval
    # fires it). See `_maybe_compact_fleet!`.
    compaction_mode::Symbol
    compaction_interval::Int
    repacks_since_compact::Int
end

LptPackingState(cfg::GatewayConfig) = LptPackingState(
    cfg.hysteresis, cfg.rate_halflife_seconds, cfg.rebalance_compute_seconds,
    cfg.min_rebalance_seconds, cfg.default_replicas,
    Dict{String,Int}(name => mc.replicas for (name, mc) in cfg.models),
    cfg.routing_fill_factor, cfg.routing_policy,
    Dict{String,Threads.Atomic{Int}}(), ReentrantLock(),
    Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Tuple{Float64,UInt64}}(),
    0.0, 0.0, 0.0,
    Dict{String,Int}(), Dict{String,Float64}(), Dict{String,Placement}(),
    Dict{Tuple{String,String},Threads.Atomic{Int}}(),
    Dict{String,Threads.Atomic{Float64}}(),
    Dict{String,ReentrantLock}(), Dict{String,Threads.Atomic{Int}}(),
    Set{Tuple{String,String}}(), Set{String}(),
    cfg.compaction_mode, cfg.compaction_interval, 0)

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
            # Watchdog-bounded: a wedged client stack would otherwise hang the prober tick (and with
            # it route discovery and /readyz). A worker that does not answer is simply skipped this
            # round, as if not ready. A hung call (timed out, not a fast refuse) means a poisoned
            # connection; drop it so the next poll reconnects fresh.
            resp, to = _bounded(() -> fetch_control_status(wc), POLL_TIMEOUT_SECONDS, nothing,
                                "ModelControlStatus poll", url)
            if resp === nothing
                to && reset_clients!(wc)
                return
            end
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
    prev_wload = @atomic s.worker_load
    prev_locks = @atomic s.sel_locks
    prev_cursors = @atomic s.rr_cursor
    out = Dict{Tuple{String,String},Threads.Atomic{Int}}()
    wload = Dict{String,Threads.Atomic{Float64}}()
    locks = Dict{String,ReentrantLock}()
    cursors = Dict{String,Threads.Atomic{Int}}()
    for (m, placement) in next
        if length(placement) > 1
            locks[m] = get(prev_locks, m, ReentrantLock())
            cursors[m] = get(prev_cursors, m, Threads.Atomic{Int}(0))
        end
        for (w, _) in placement
            out[(m, w)] = get(prev_out, (m, w), Threads.Atomic{Int}(0))
            haskey(wload, w) || (wload[w] = get(prev_wload, w, Threads.Atomic{Float64}(0.0)))
        end
    end
    @atomic s.outstanding = out
    @atomic s.worker_load = wload
    @atomic s.sel_locks = locks
    @atomic s.rr_cursor = cursors
    return nothing
end

# Recompute and install a new assignment from a fresh poll: fold arrival-rate and compute-cost
# EWMAs, build expected utilization, run compute_assignment with the configured replica counts,
# swap in the assignment and outstanding counters, reset the compute accumulator, and export
# metrics. Runs on the prober task only. A model not reported by every ready worker (runtime drift)
# gets a uniform placement over the workers that do serve it, with a warning.
function _repack!(s::LptPackingState, poll, metrics::Union{GatewayMetrics,Nothing})
    now = time()
    # Elapsed since the last repack, for the repack log: wall time and the fleet GPU-seconds that
    # accumulated (the compute that triggered this repack). Captured before they are reset below.
    wall_elapsed = s.last_rebalance == 0.0 ? 0.0 : now - s.last_rebalance
    compute_elapsed = s.compute_accum
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
    # Publish a fresh per-model cost snapshot for the request hot path (fill_least). A copy, so the
    # next repack's in-place EWMA fold above cannot race a concurrent reader of the snapshot. The
    # reserved `_COST_DEFAULT_KEY` entry carries the fleet-mean measured cost, used as the cold-start
    # weight for models with no measured cost yet (same units, so a cold model counts like an average
    # one rather than dominating or vanishing).
    cs = copy(s.cost_ewma)
    isempty(cs) || (cs[_COST_DEFAULT_KEY] = sum(values(cs)) / length(cs))
    @atomic s.cost_snapshot = cs

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
    stats = RepackStats()
    next = compute_assignment(full, sort(poll.polled), prev;
                              mem=poll.mem, mem_cap=poll.mem_cap,
                              replicas=s.model_replicas, default_replicas=s.default_replicas,
                              hysteresis=s.hysteresis, stats=stats)
    merge!(next, drifted)
    @atomic s.assignment = next
    _swap_outstanding!(s, next)

    s.compute_accum = 0.0
    s.last_fleet_compute = poll.fleet_compute

    # Count models whose worker set changed from the previous assignment (a model new this repack is
    # an initial placement, not a move, so it is not counted), and collect the workers affected by a
    # move (gained or lost a model) so the caller can compact just those.
    moved = 0
    changed_workers = Set{String}()
    for (m, placement) in next
        prevpl = get(prev, m, nothing)
        prevpl === nothing && continue
        nextws = Set(first.(placement)); prevws = Set(first.(prevpl))
        nextws == prevws && continue
        moved += 1
        union!(changed_workers, symdiff(nextws, prevws))
    end
    for (m, prevpl) in prev          # models dropped entirely this repack: their old workers lose them
        haskey(next, m) && continue
        union!(changed_workers, Set(first.(prevpl)))
    end
    @info "lpt_packing: repack" models = length(next) moved = moved held_by_hysteresis = stats.held max_improvement = round(stats.max_improvement; digits = 3) hysteresis = s.hysteresis compute_seconds = round(compute_elapsed; digits = 2) wall_seconds = round(wall_elapsed; digits = 1)

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
    return changed_workers
end

# After a repack, fan a CompactMemory out to the workers whose assignment changed, on the gateway's
# compaction cadence. Counts every repack; once `compaction_interval` repacks have elapsed, the first
# one that actually moved a model (non-empty `changed`) fires the fan-out and resets the counter, so
# it can land later than exactly N. `:off` disables; `:eager` sends an empty reload list (the
# on-demand region refills lazily as requests arrive); `:scheduled` sends each changed worker the set
# of models this repack assigned to it, warming the new placement. Runs on the prober task.
function _maybe_compact_fleet!(s::LptPackingState, pool::ClientPool,
                               metrics::Union{GatewayMetrics,Nothing}, changed::Set{String})
    (s.compaction_mode == :off || s.compaction_interval <= 0) && return nothing
    s.repacks_since_compact += 1
    (s.repacks_since_compact >= s.compaction_interval && !isempty(changed)) || return nothing
    s.repacks_since_compact = 0

    perworker = Dict{String,Vector{String}}(w => String[] for w in changed)
    if s.compaction_mode == :scheduled
        for (m, placement) in @atomic(s.assignment), (w, _) in placement
            haskey(perworker, w) && push!(perworker[w], m)
        end
    end
    total, ok, failed = _compact_workers(pool, metrics, perworker)
    @info "lpt_packing: compaction" mode = s.compaction_mode workers = ok reloaded = total failed = failed
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
    if triggered
        changed = _repack!(s, poll, metrics)
        _maybe_compact_fleet!(s, pool, metrics, changed)
    end
    return nothing
end

# The effective per-replica fill quantum for `model`: the worker-reported max batch scaled by the
# fill factor, at least 1. When the max batch is unknown (uncapped, no compiled batch reported) the
# quantum is 1, which degrades the fill policy to least-outstanding within the replica set.
function _fill_quantum(s::LptPackingState, model::AbstractString)
    maxB = get(@atomic(s.max_batch), model, 0)
    return maxB <= 0 ? 1 : max(1, round(Int, s.routing_fill_factor * maxB))
end

# Reserved key in the cost snapshot for the fleet-mean cost (see _repack!). Never a real model name.
const _COST_DEFAULT_KEY = ""

# The compute weight charged for one in-flight request of `model`: its measured per-request cost, or
# the fleet-mean as a cold-start stand-in, or 1.0 before any cost is known. The value is captured at
# reservation and the identical value released, so an intervening repack (which changes the cost)
# never leaves the per-worker load drifting.
function _route_weight(s::LptPackingState, model::AbstractString)
    snap = @atomic s.cost_snapshot
    c = get(snap, model, 0.0)
    c > 0 && return c
    return get(snap, _COST_DEFAULT_KEY, 1.0)
end

# Reserve `model` on a single worker `w` (the n==1 fast path and the shared per-worker bookkeeping):
# bump the per-(model,worker) and per-worker compute-load counters, returning the reservation tuple
# to release later. Any counter missing from the live snapshot (mid-repack drift) is skipped and
# released as a no-op.
function _reserve_on!(out_snap, wload_snap, model, w, weight)
    mwc = get(out_snap, (model, w), nothing)
    wload = get(wload_snap, w, nothing)
    mwc === nothing || Threads.atomic_add!(mwc, 1)
    wload === nothing || Threads.atomic_add!(wload, weight)
    return (mwc, wload, weight)
end

"""
    route_replica(s, model) -> Union{Nothing, Tuple{Vector{String}, Counters}}

Order a model's replicas for dispatch and reserve the chosen one. Returns `nothing` when the model
has no placement yet (cold or unknown); the caller falls back to round robin. Otherwise returns the
ordered replica URLs (the chosen worker first, the rest as failover) and `Counters`, the reserved
counters to release when the request completes.

All `fill_*` policies concentrate: a replica part-way through filling its quantum keeps receiving the
model's requests (the `-outs` term wins), so batches coalesce. They differ only in which replica a
*fresh* batch opens on, when the replicas tie on fill progress:

  - `fill_rr`: round-robins the opening replica across the model's set; the per-model `rr_cursor`
    advances only on a genuine batch start (more than one replica tied at the best fill progress),
    so mid-fill concentration never rotates.
  - `fill_least`: opens on the replica whose worker has the least in-flight compute load (in-flight
    requests weighted by measured per-request cost), so a model's batches land on whichever GPU is
    least busy across all models; URL breaks exact ties.

Every routed request, single- or multi-replica, bumps the per-worker compute-load counter, so
`fill_least` sees load from all models. The reservation (atomic increments under the per-model
selection lock for multi-replica models) makes concurrent selections see the choice, so requests do
not stampede onto the same replica.
"""
function route_replica(s::LptPackingState, model::AbstractString)
    placement = get(@atomic(s.assignment), model, nothing)
    placement === nothing && return nothing
    n = length(placement)
    n == 0 && return nothing

    out_snap = @atomic s.outstanding
    wload_snap = @atomic s.worker_load
    weight = _route_weight(s, model)

    if n == 1
        w = placement[1][1]
        return (String[w], _reserve_on!(out_snap, wload_snap, model, w, weight))
    end

    workers = String[p[1] for p in placement]
    lk = get(@atomic(s.sel_locks), model, nothing)
    lk === nothing && return (workers, nothing)   # mid-repack drift: route in order, untracked
    Q = _fill_quantum(s, model)
    cursors = @atomic s.rr_cursor

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
        if s.routing_policy == "fill_least"
            wloads = Vector{Float64}(undef, n)
            for i in 1:n
                wa = get(wload_snap, workers[i], nothing)
                wloads[i] = wa === nothing ? 0.0 : wa[]
            end
            sort!(order; by = i -> (fld(outs[i], Q), -outs[i], wloads[i], workers[i]))
            chosen = order[1]
        else   # fill_rr
            # Rank by fill progress; among replicas tied at the best progress, pick the one next in
            # rotation from the cursor (offset distance, 0-based). The cursor advances only on a real
            # batch start (more than one replica tied), so a lone mid-fill winner keeps concentrating.
            cur = get(cursors, model, nothing)
            base = cur === nothing ? 0 : cur[]
            prog(i) = (fld(outs[i], Q), -outs[i])
            sort!(order; by = i -> (prog(i), mod(i - 1 - base, n)))
            chosen = order[1]
            best = prog(chosen)
            tied = count(i -> prog(i) == best, 1:n)
            tied > 1 && cur !== nothing && (cur[] = mod(chosen, n))
        end
        Threads.atomic_add!(cobjs[chosen], 1)
        wload = get(wload_snap, workers[chosen], nothing)
        wload === nothing || Threads.atomic_add!(wload, weight)
        ordered = String[workers[i] for i in order]
        return (ordered, cobjs[chosen], wload)
    end
    res === nothing && return (workers, nothing)
    ordered, mwc, wload = res
    return (ordered, (mwc, wload, weight))
end

# Release a reservation made by route_replica: decrement the per-(model,worker) request counter and
# subtract the captured compute weight from the per-worker load. Robust to the failure path: called
# once in a finally regardless of how the dispatch ended, and to any counter that was absent at
# reservation time (drift), which is stored as `nothing`.
function _release_route!(counters)
    counters === nothing && return nothing
    mwc, wload, weight = counters
    mwc === nothing || Threads.atomic_sub!(mwc, 1)
    wload === nothing || Threads.atomic_sub!(wload, weight)
    return nothing
end

# --- GatewayScheduler interface (see scheduler.jl) --------------------------------------------
# `record_arrival!(s::LptPackingState, model)` is defined above and is the specialization of the
# generic for this scheduler; the rest of the interface is adapted here.

release!(::LptPackingState, reservation) = _release_route!(reservation)

scheduler_tick!(s::LptPackingState, pool::ClientPool, ready_urls, metrics) =
    tick_packing!(s, pool, ready_urls, metrics)

# Hard startup preconditions (all workers reachable, FIFO discipline, identical model sets), then an
# initial rebalance so the first requests already route by packing rather than waiting a prober tick.
function scheduler_start!(s::LptPackingState, pool::ClientPool, metrics)
    verify_lpt_packing_preconditions!(pool; wait_seconds = _startup_wait_seconds())
    @info "gateway scheduling: lpt_packing" rebalance_compute_seconds = s.rebalance_compute_seconds min_rebalance_seconds = s.min_rebalance_seconds default_replicas = (s.default_replicas == REPLICAS_ALL ? "all" : s.default_replicas) routing_policy = s.routing_policy
    rebalance!(s, pool, copy(pool.order), metrics)
    return nothing
end

# Route to the placement replica that fills its batch first (route_replica reserves it), the rest of
# the placement following as failover, then any discovered replicas outside the placement as a
# last resort so a concentrated model survives its worker dying between repacks. A model without a
# placement yet (cold, or new since the last repack) falls back to round robin over discovered routes.
function select_replicas(s::LptPackingState, ctx::ScheduleContext)
    routed = route_replica(s, ctx.model)
    if routed === nothing
        urls = pick(ctx.routes, ctx.model)
        urls === nothing && return nothing
        return (urls, nothing)
    end
    urls, counters = routed
    rr = pick(ctx.routes, ctx.model)
    if rr !== nothing
        extra = String[u for u in rr if !(u in urls)]
        isempty(extra) || (urls = vcat(urls, extra))
    end
    return (urls, counters)
end

"""
    verify_lpt_packing_preconditions!(pool; wait_seconds=0, poll_interval=10.0,
                                      call_timeout=8.0, wedge_rounds=3) -> nothing

LPT-packing mode's startup checks: every configured worker must be reachable over the control
plane, report FIFO scheduling discipline, and serve an identical model set (load-all).

Reachability is gated rather than asserted: a worker compiles and warms up every model before its
control plane answers, so at startup the workers are usually not up yet. With `wait_seconds > 0`
(or `Inf` to wait indefinitely) the check polls every `poll_interval` seconds until all workers
answer, logging which are still pending; with `wait_seconds <= 0` (the default) it checks once and
fails fast. The supervisor sets this to wait for the workers it co-launches (see `gateway_spec`).
Once all workers are reachable, FIFO discipline and identical model sets are hard requirements and
a violation raises with the offending worker named.

Every poll is watchdog-bounded (`call_timeout`): the gateway's clients share one libcurl multi
handle whose request semaphore caps in-flight requests at GRPC_MAX_STREAMS (16). After the burst of
failed connects during warmup the handle's socket/timer driving can stall, so in-flight requests
never complete, never return their semaphore slot, and every new call blocks at acquire forever (the
"wedge"). A worker that is merely down refuses fast and releases its slot; but if every worker's
call exceeds the watchdog for `wedge_rounds` consecutive rounds (the wedge signature), the process
exits so the supervisor restarts the gateway with a fresh handle (16 free slots), which recovers.
"""
function verify_lpt_packing_preconditions!(pool::ClientPool; wait_seconds::Real=0,
                                           poll_interval::Real=10.0, call_timeout::Real=8.0,
                                           wedge_rounds::Integer=3)
    clients = all_clients(pool)
    forever = isinf(wait_seconds)
    deadline = (forever || wait_seconds <= 0) ? nothing : time() + Float64(wait_seconds)
    statuses = Dict{String,Any}()
    wedged_streak = 0
    while true
        statuses = Dict{String,Any}()
        pending = String[]
        timed_out = 0
        for wc in clients
            resp, to = _bounded(() -> fetch_control_status(wc), call_timeout, nothing,
                                "ModelControlStatus poll", wc.url)
            if resp === nothing
                push!(pending, wc.url)
                # A hung call (not a fast refuse) means the worker was caught mid-stall and its
                # connection is poisoned; drop it so the next poll reconnects fresh.
                to && (timed_out += 1; reset_clients!(wc))
            else
                statuses[wc.url] = resp
            end
        end
        isempty(pending) && break
        # Wedge signature: every worker's call exceeded the watchdog (the client stack stopped being
        # driven, not just refused). A fresh process recovers, so exit for a supervisor restart
        # rather than spin forever on a dead handle.
        wedged_streak = timed_out == length(clients) ? wedged_streak + 1 : 0
        if wedge_rounds > 0 && wedged_streak >= wedge_rounds
            @error "lpt_packing: control-plane calls timed out for $(wedged_streak) consecutive rounds; the gRPC client stack is wedged. Exiting so the supervisor restarts the gateway with a fresh stack." rounds = wedged_streak
            exit(1)
        end
        if !forever && (wait_seconds <= 0 || time() >= deadline)
            suffix = wait_seconds <= 0 ? "" : " after $(round(Int, wait_seconds))s"
            error("lpt_packing scheduling: worker(s) $(sort(pending)) unreachable over the control plane$(suffix); all workers must be up (set REACTANT_GATEWAY_STARTUP_WAIT_SECONDS to wait for slow-starting workers, or 'inf' to wait indefinitely)")
        end
        @info "lpt_packing: waiting for all workers before serving (workers compile before they answer the control plane)" ready = sort(collect(keys(statuses))) pending = sort(pending)
        sleep(poll_interval)
    end
    for (url, resp) in statuses
        # FIFO and EDF are both compatible with lpt_packing: neither imposes a competing per-model
        # fairness policy (EDF only reorders by request deadline, degrading to FIFO for equal
        # deadlines), so the gateway stays the placement/fairness authority. FAIR is rejected.
        resp.discipline in ("fifo", "edf") ||
            error("lpt_packing scheduling requires worker FIFO or EDF discipline; worker $url reports '$(resp.discipline)' (set scheduler.discipline: fifo or edf in the node file)")
    end
    sets = Dict(url => sort([ms.name for ms in resp.models]) for (url, resp) in statuses)
    ref_url = first(keys(sets))
    for (url, names) in sets
        names == sets[ref_url] ||
            error("lpt_packing scheduling requires all models on all workers; $url serves $(length(names)) models but $ref_url serves $(length(sets[ref_url])) (model sets differ)")
    end
    return nothing
end
