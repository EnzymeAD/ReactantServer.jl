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
# Within a model's replica set the gateway concentrates its requests on one replica at a time, so the
# workers receive deep same-model groupings to coalesce. `routing_fill_mode` decides what the fill
# quantum `Q` (the worker-reported max batch scaled by `routing_fill_factor`) counts, per model:
#
# The quantum and every in-flight counter are denominated in ITEMS, not requests: a request's item
# count is the leading dimension of its first input tensor, peeked out of the body without decoding
# the payload (see headers.jl), so one request of 32 items and 32 requests of one item are the same
# amount of work. `max_batch` is itself an item count, which is what makes `Q = factor * max_batch`
# dimensionally honest. Models the worker does not batch (max_batch <= 1: metas, unbatched models)
# charge one unit per request, since their leading dimension names an axis, not a count.
#
#   :run       (default) a run spends `Q` ITEMS on one replica, then the next run opens elsewhere, so
#              the model's GPUs serve it in turn: batches stay deep and the share is even to within one
#              run at any concurrency. A run also ends when the model has nothing in flight (no batch
#              to protect, so it rotates per request) or when its replica falls a whole quantum behind
#              the least-backed-up one.
#   :spread    equalize in-flight ITEMS across the set, so all k GPUs serve the model at once. Lowest
#              latency for a latency-bound model whose concurrency is well below its max batch, at the
#              cost of coalescing depth. This is the right mode for a client that already batches to
#              max_batch itself, where there is nothing left for the worker to coalesce.
#   :inflight  the legacy basis: `Q` counts items IN FLIGHT. A replica keeps receiving the model's
#              traffic until it holds a full quantum at once, which never happens below `n*(Q-1)`
#              in-flight, so a model whose concurrency sits between 2 and Q and never drains is served
#              by ONE replica indefinitely while the others get nothing. Retained only to reproduce
#              pre-:run behavior; see the warning on `GatewayConfig.routing_fill_mode`.
#
# `routing_policy` decides which replica each run opens on: `fill_rr` rotates (load-blind and exactly
# even) and `fill_least` opens on the least compute-loaded GPU, so a model's runs avoid GPUs busy with
# other models. Coalescing itself stays at the worker; the gateway only routes and tracks small
# per-replica and per-worker counters. (Spreading every request with no concentration at all is the
# separate `least_outstanding` scheduling mode, not an lpt_packing setting; see scheduler.jl.)
#
# Repacks are driven by accumulated fleet compute, not wall-clock: the prober polls the workers
# every tick and a repack fires once the fleet has consumed `rebalance_compute_seconds` GPU-seconds
# since the last one. The first tick-driven repack can use a separate, smaller
# `first_rebalance_compute_seconds` budget (a quick early rebalance), then steady-state uses the
# larger `rebalance_compute_seconds`; the startup cold placement does not consume the first budget.
# An operator can also force one out of band (Repack in control_service.jl), which the prober serves
# on its next round without consuming the first-repack budget.
#
# Every knob above is runtime-mutable through the gateway's own control plane
# (GatewayControlService, see control_service.jl), including a model's replica count and fill mode.
# The knobs live in one immutable snapshot swapped atomically (see knobs.jl), so a repack never
# observes a half-applied change; nothing is persisted, so a restart reverts to gateway.yml.
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
                       replicas=Dict(), default_replicas=1, hysteresis=0.1,
                       forbid_memory_oversubscription=false)
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

When `forbid_memory_oversubscription` is true the budget becomes a hard constraint: a model is
placed only on workers where its weights still fit (for `k == 1` the candidate set is filtered and
hysteresis only holds a previous home that still fits; for `k > 1` feasible workers are ranked
first), falling back to the unconstrained choice only when no worker can fit it. This is greedy
first-fit-decreasing, so it never oversubscribes when the greedy pass can avoid it, but it does not
guarantee finding a feasible packing that exists.
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
                            forbid_memory_oversubscription::Bool=false,
                            stats::Union{Nothing,RepackStats}=nothing)
    out = Dict{String,Placement}()
    isempty(workers) && return out
    cload = Dict{String,Float64}(w => 0.0 for w in workers)   # compute, normalized (capacity 1.0)
    mload = Dict{String,Float64}(w => 0.0 for w in workers)   # memory, bytes
    capof(w) = (c = get(mem_cap, w, Inf); c <= 0 ? Inf : c)
    # Pressure of worker `w` after placing compute `uc` and memory `wm` on it.
    score_after(w, uc, wm) = max(cload[w] + uc,
                                 capof(w) == Inf ? 0.0 : (mload[w] + wm) / capof(w))
    # Whether `w` can still hold a model of footprint `wm` within its budget (unconstrained always
    # fits). Used only when `forbid_memory_oversubscription` is set.
    fits(w, wm) = (c = capof(w); c == Inf || mload[w] + wm <= c)
    # Descending compute demand, then descending memory (pack the bulky cold models early),
    # ties broken by name for determinism.
    order = sort!(collect(keys(u)); by = m -> (-u[m], -get(mem, m, 0.0), m))
    for m in order
        um = u[m]
        wm = get(mem, m, 0.0)
        prev_pl = get(prev, m, nothing)
        k = clamp(get(replicas, m, default_replicas), 1, length(workers))
        if k == 1
            # Optionally restrict to workers where the weights still fit, so a feasible home is never
            # passed over for one that oversubscribes; fall back to all workers when none fit.
            cands = workers
            if forbid_memory_oversubscription
                feasible = filter(w -> fits(w, wm), workers)
                isempty(feasible) || (cands = feasible)
            end
            best = argmin(w -> score_after(w, um, wm), cands)
            chosen = best
            # Hysteresis: stick with the previous single placement unless the move improves the
            # model's resulting pressure by more than the threshold. Under the guarantee, only stay if
            # the previous home is still a candidate (still fits), so an over-budget home is vacated.
            if prev_pl !== nothing && length(prev_pl) == 1 && haskey(cload, prev_pl[1][1])
                wp = prev_pl[1][1]
                bestscore = score_after(best, um, wm)
                prevscore = score_after(wp, um, wm)
                if stats !== nothing
                    impr = bestscore > 0 ? (prevscore / bestscore - 1.0) : 0.0
                    impr > stats.max_improvement && (stats.max_improvement = impr)
                end
                if wp in cands && prevscore <= bestscore * (1 + hysteresis)
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
            # Under the guarantee, rank workers that still fit ahead of those that do not, so the `k`
            # replicas fill feasible workers first and only spill onto an infeasible one when fewer
            # than `k` can fit. With the guarantee off the lead term is a constant 0 (no reordering).
            fitrank(w) = (forbid_memory_oversubscription && !fits(w, wm)) ? 1 : 0
            ranked = sort(workers; by = w -> (fitrank(w), score_after(w, um / k, wm), w in prevset ? 0 : 1, w))
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

# One model's in-progress fill run (multi-replica models only). Every field is read and written
# exclusively under that model's `sel_locks` entry, so plain `Int`s are correct here and one dict
# lookup covers the whole routing decision.
#
# `target` is an index into the model's current placement, 0 when no run is open. `units` counts the
# ITEMS this run has sent (the sum of the requests' batch sizes), which is what the `:run` quantum
# compares against. `cursor` is the 0-based rotation position the next run opens from.
mutable struct FillRun
    cursor::Int
    target::Int
    units::Int
end

FillRun() = FillRun(0, 0, 0)

mutable struct LptPackingState <: GatewayScheduler
    # Every runtime-mutable knob, in one immutable snapshot swapped atomically (see knobs.jl). The
    # prober reads one snapshot per tick and the request hot path one per request, so neither can
    # observe a half-applied policy change. Written only by `set_knobs!` under `control_lock`.
    @atomic knobs::PackingKnobs
    control_lock::ReentrantLock              # serializes the read-modify-write of `knobs`
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
    # whether a tick-driven repack has fired yet. The startup cold placement does not count; the
    # first tick repack sets this, after which the steady-state `rebalance_compute_seconds` applies.
    did_first_tick_repack::Bool
    # routing metadata, swapped atomically each tick; the request hot path reads the snapshot.
    @atomic max_batch::Dict{String,Int}      # model -> effective max batch (largest compiled, capped)
    # Per-model measured per-request compute cost (GPU-seconds/request), published from cost_ewma at
    # each repack so the request hot path can read it without racing the prober. Drives fill_least's
    # compute-weighted load.
    @atomic cost_snapshot::Dict{String,Float64}
    # the live assignment, swapped atomically; readers never lock
    @atomic assignment::Dict{String,Placement}
    # In-flight ITEM counters (the sum of the batch sizes of the requests in flight), swapped
    # atomically at repack so the hot path reads a stable snapshot. Per (model, worker): backpressure
    # for every fill mode, and the quantum itself under `:spread` / `:inflight`. Items rather than
    # requests so the comparison against `Q` is dimensionally consistent; for a client with a fixed
    # request size this is just a scaled request count, which is why the item denomination degrades
    # gracefully and the reverse would not.
    @atomic outstanding::Dict{Tuple{String,String},Threads.Atomic{Int}}
    # Cumulative requests routed per (model, worker), carried across repacks like `outstanding`.
    # Exported as `gateway_replica_routed_total`, which is the only series that shows a starved
    # replica: its in-flight gauge reads 0 whether it is idle or never chosen.
    @atomic routed::Dict{Tuple{String,String},Threads.Atomic{Int}}
    # Per-worker in-flight compute load: the sum over a worker's in-flight requests of each routed
    # model's cost weight (GPU-seconds). Drives fill_least's least-loaded batch-start choice. Every
    # routed request, single- or multi-replica, contributes, so the load reflects all models.
    @atomic worker_load::Dict{String,Threads.Atomic{Float64}}
    # per-model selection lock (multi-replica models only), so a pick-and-reserve is atomic and
    # concurrent requests do not stampede onto the same replica.
    @atomic sel_locks::Dict{String,ReentrantLock}
    # per-model fill run (multi-replica models only): which replica is currently being filled, how
    # much of its quantum it has served, and the rotation cursor for opening the next one. Read and
    # written only under the model's `sel_locks` entry, so the fields are plain (see FillRun).
    @atomic fill_runs::Dict{String,FillRun}
    # Per-model resolved routing rule (mode + quantum), republished every prober tick from the knobs
    # and the worker-reported max batch. The request path reads this instead of walking the override
    # chain. See `_publish_fill_plan!`.
    @atomic fill_plan::Dict{String,FillPlan}
    # label pairs / models previously exported to the gauges, zeroed when dropped
    exported::Set{Tuple{String,String}}
    replicas_exported::Set{String}
    # Repacks since the last compaction fan-out, so the first placement-changing repack at or after
    # `knobs.compaction_interval` fires it. Prober task only. See `_maybe_compact_fleet!`.
    repacks_since_compact::Int
    # Prober-published summaries of the prober's own plain fields, so a control handler on another
    # task can report live scheduling state without racing it (see knobs.jl).
    @atomic tick::TickReport
    @atomic report::RepackReport
    # Operator-forced repack handshake. `repack_requested` is bumped only by a control handler;
    # `repack_completed` is stored only by the prober, and only after the new assignment and its
    # report are installed, so a caller that observes its own sequence is guaranteed to see the
    # results. Two monotone counters, so N requests arriving inside one tick window coalesce into one
    # repack and all N are satisfied by the single store, and neither side ever blocks the other.
    @atomic repack_requested::Int
    @atomic repack_completed::Int
end

LptPackingState(cfg::GatewayConfig) = LptPackingState(
    PackingKnobs(cfg), ReentrantLock(),
    Dict{String,Threads.Atomic{Int}}(), ReentrantLock(),
    Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Tuple{Float64,UInt64}}(),
    0.0, 0.0, 0.0, false,
    Dict{String,Int}(), Dict{String,Float64}(), Dict{String,Placement}(),
    Dict{Tuple{String,String},Threads.Atomic{Int}}(),
    Dict{Tuple{String,String},Threads.Atomic{Int}}(),
    Dict{String,Threads.Atomic{Float64}}(),
    Dict{String,ReentrantLock}(), Dict{String,FillRun}(), Dict{String,FillPlan}(),
    Set{Tuple{String,String}}(), Set{String}(), 0,
    TickReport(), RepackReport(), 0, 0)

"""
    knobs(s::LptPackingState) -> PackingKnobs

The scheduler's current knob snapshot. Take this **once** per tick or per request and read plain
fields off the result; re-reading the atomic mid-computation defeats the point of the snapshot,
which is that a policy change is all-or-nothing from any reader's perspective.
"""
knobs(s::LptPackingState) = @atomic s.knobs

"""
    set_knobs!(s::LptPackingState; kwargs...) -> PackingKnobs

Replace one or more runtime knobs, returning the snapshot now in force. Each keyword names a
[`PackingKnobs`](@ref) field; unnamed fields keep their value and `generation` is bumped.

The read-modify-write is serialized by `s.control_lock`, so concurrent callers cannot lose an
update, and installing it is a single atomic store, so the prober sees either all of the change or
none of it. Values are validated exactly as `gateway.yml` validates them, and validation happens
before anything is installed, so a rejected value leaves the live knobs untouched.
"""
function set_knobs!(s::LptPackingState; kwargs...)
    return lock(s.control_lock) do
        next = apply_updates(@atomic(s.knobs); kwargs...)
        @atomic s.knobs = next
        next
    end
end

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
    batch_at = Dict{String,Tuple{String,Int}}()  # model -> (wire input name, 1-based batch axis)
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
                    # Every worker serves the same bundle, so the first non-empty report wins; a
                    # worker mid-reload can legitimately report nothing yet.
                    isempty(ms.batch_input_name) || haskey(batch_at, ms.name) ||
                        (batch_at[ms.name] = (ms.batch_input_name, Int(ms.batch_axis)))
                    push!(get!(permodel_workers, ms.name, String[]), url)
                end
            end
        end
    end
    fleet_compute = 0.0
    for (tc, _) in values(sums)
        fleet_compute += tc
    end
    return (; sums, permodel_workers, mem, mem_cap, max_batch, batch_at, polled, fleet_compute)
end

# Rebuild the outstanding/worker-total counters and per-model selection locks to cover the new
# assignment, carrying over the live atomics for placements that persist (so in-flight counts
# survive a repack) and dropping the rest, then swap the snapshots in atomically. Lock objects are
# reused per model so the same object guards a model regardless of which snapshot a request read.
function _swap_outstanding!(s::LptPackingState, next::Dict{String,Placement})
    prev_out = @atomic s.outstanding
    prev_routed = @atomic s.routed
    prev_wload = @atomic s.worker_load
    prev_locks = @atomic s.sel_locks
    prev_runs = @atomic s.fill_runs
    out = Dict{Tuple{String,String},Threads.Atomic{Int}}()
    routed = Dict{Tuple{String,String},Threads.Atomic{Int}}()
    wload = Dict{String,Threads.Atomic{Float64}}()
    locks = Dict{String,ReentrantLock}()
    runs = Dict{String,FillRun}()
    for (m, placement) in next
        if length(placement) > 1
            locks[m] = get(prev_locks, m, ReentrantLock())
            # Carry the run record so the rotation cursor survives a repack, but end the run in
            # progress: its `target` indexed the OLD placement, which may have gained or lost workers.
            # The next request reopens against the new one.
            run = get(prev_runs, m, nothing)
            runs[m] = run === nothing ? FillRun() : FillRun(run.cursor, 0, 0)
        end
        for (w, _) in placement
            out[(m, w)] = get(prev_out, (m, w), Threads.Atomic{Int}(0))
            routed[(m, w)] = get(prev_routed, (m, w), Threads.Atomic{Int}(0))
            haskey(wload, w) || (wload[w] = get(prev_wload, w, Threads.Atomic{Float64}(0.0)))
        end
    end
    @atomic s.outstanding = out
    @atomic s.routed = routed
    @atomic s.worker_load = wload
    @atomic s.sel_locks = locks
    @atomic s.fill_runs = runs
    return nothing
end

# Recompute and install a new assignment from a fresh poll: fold arrival-rate and compute-cost
# EWMAs, build expected utilization, run compute_assignment with the configured replica counts,
# swap in the assignment and outstanding counters, reset the compute accumulator, and export
# metrics. Runs on the prober task only. A model not reported by every ready worker (runtime drift)
# gets a uniform placement over the workers that do serve it, with a warning.
function _repack!(s::LptPackingState, poll, metrics::Union{GatewayMetrics,Nothing};
                  trigger::Symbol=:compute, k::PackingKnobs=knobs(s))
    now = time()
    # Elapsed since the last repack, for the repack log: wall time and the fleet GPU-seconds that
    # accumulated (the compute that triggered this repack). Captured before they are reset below.
    wall_elapsed = s.last_rebalance == 0.0 ? 0.0 : now - s.last_rebalance
    compute_elapsed = s.compute_accum
    s.last_rebalance = now

    # Arrival rates. The sample is the wall-clock arrival rate over the interval (so u = rate * cost
    # stays in GPU-sec/wall-sec, comparable to a worker's capacity 1.0); the EWMA decays against the
    # fleet compute consumed this interval, with halflife `ema_halflife_compute`. On the startup
    # rebalance (no interval yet) `compute_elapsed` is 0, so the fold is a no-op and the EWMAs stay at
    # 0 until real traffic drives a tick repack; that is correct, since nothing has run yet.
    counters = @atomic s.arrivals
    for (m, c) in counters
        n = Threads.atomic_xchg!(c, 0)
        sample = wall_elapsed > 0 ? n / wall_elapsed : 0.0
        s.rate_ewma[m] = _ewma(get(s.rate_ewma, m, 0.0), sample, compute_elapsed, k.ema_halflife_compute)
    end

    # Worker-reported costs: delta the per-model cumulative (compute, requests) against the previous
    # repack. A negative delta means a worker restarted (counters reset); re-baseline and skip.
    for (m, (tc, rq)) in poll.sums
        prev_tc, prev_rq = get(s.last_cum, m, (0.0, UInt64(0)))
        dtc, drq = tc - prev_tc, Int(rq) - Int(prev_rq)
        s.last_cum[m] = (tc, rq)
        (dtc < 0 || drq < 0) && continue          # worker restart: re-baseline only
        drq > 0 && (s.cost_ewma[m] = _ewma(get(s.cost_ewma, m, dtc / drq), dtc / drq, compute_elapsed, k.ema_halflife_compute))
    end

    @atomic s.max_batch = poll.max_batch
    _publish_fill_plan!(s, k, poll.max_batch, poll.batch_at)
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
                              replicas=replica_overrides(k), default_replicas=k.default_replicas,
                              hysteresis=k.hysteresis,
                              forbid_memory_oversubscription=k.forbid_memory_oversubscription,
                              stats=stats)
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
    @info "lpt_packing: repack" models = length(next) moved = moved held_by_hysteresis = stats.held max_improvement = round(stats.max_improvement; digits = 3) hysteresis = k.hysteresis compute_seconds = round(compute_elapsed; digits = 2) wall_seconds = round(wall_elapsed; digits = 1)

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
        inc_repacks!(metrics, trigger)
        live = Set{Tuple{String,String}}()
        live_models = Set{String}()
        for (m, placement) in next
            push!(live_models, m)
            set_model_replicas!(metrics, m, length(placement))
            for (w, weight) in placement
                set_placement_weight!(metrics, m, w, weight)
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
    _refresh_live_gauges!(s, metrics)

    # Publish this repack for readers on other tasks (the control plane). The EWMA dicts are copied
    # for the same reason `cost_snapshot` is: the next repack folds them in place. `poll.mem` /
    # `poll.mem_cap` are freshly built by this poll and never mutated afterwards, so they can be
    # referenced directly.
    @atomic s.report = RepackReport(now, wall_elapsed, compute_elapsed, trigger,
                                    (@atomic s.report).count + 1, length(next), moved,
                                    copy(full), copy(s.rate_ewma), copy(s.cost_ewma),
                                    poll.mem, poll.mem_cap, sort(poll.polled),
                                    Set{String}(keys(drifted)))
    return changed_workers
end

# Refresh the gauges that describe live routing state on every prober tick, not only at a repack:
# in-flight requests per replica, and each model's effective fill quantum. Zeroing of label pairs that
# have disappeared stays at repack (see `s.exported`), so this only ever writes pairs that are in the
# current assignment. Prober task only; a nothing `metrics` (tests) is a no-op.
function _refresh_live_gauges!(s::LptPackingState, metrics::Union{GatewayMetrics,Nothing})
    metrics === nothing && return nothing
    out_snap = @atomic s.outstanding
    plan = @atomic s.fill_plan
    for (m, placement) in @atomic(s.assignment)
        p = get(plan, m, nothing)
        p === nothing || set_model_fill_quantum!(metrics, m, p.quantum)
        for (w, _) in placement
            a = get(out_snap, (m, w), nothing)
            set_replica_outstanding!(metrics, m, w, a === nothing ? 0 : a[])
        end
    end
    return nothing
end

# Exports the cumulative per-replica routed counts at scrape time from the scheduler's live snapshot,
# so the series never lags a repack. This is the only view that separates an idle replica from a
# starved one: both report zero in-flight, but only the starved one has a flat routed total.
struct RoutedCollector <: Prometheus.Collector
    state::LptPackingState
    worker_names::Dict{String,String}   # worker url -> friendly label, as GatewayMetrics maps them
end

Prometheus.metric_names(::RoutedCollector) = ("gateway_replica_routed_total",)

function Prometheus.collect!(metrics::Vector, c::RoutedCollector)
    snap = @atomic c.state.routed
    isempty(snap) && return metrics
    ln = Prometheus.LabelNames((:model, :worker))
    samples = Prometheus.Sample[]
    for ((m, w), a) in snap
        push!(samples, Prometheus.Sample(nothing, ln,
                                        Prometheus.LabelValues((m, get(c.worker_names, w, w))),
                                        Float64(a[])))
    end
    # Deterministic order across scrapes; Prometheus.jl sorts family children for the same reason.
    sort!(samples; by = x -> x.label_values.label_values)
    push!(metrics, Prometheus.Metric("counter", "gateway_replica_routed_total",
        "Cumulative REQUESTS routed to a model's replica on a worker (a count of routing " *
        "decisions; see gateway_replica_outstanding for the item-denominated in-flight work).",
        samples))
    return metrics
end

register_routed!(s::LptPackingState, m::GatewayMetrics) =
    Prometheus.register(m.registry, RoutedCollector(s, m.worker_names))

# After a repack, fan a CompactMemory out to the workers whose assignment changed, on the gateway's
# compaction cadence. Counts every repack; once `compaction_interval` repacks have elapsed, the first
# one that actually moved a model (non-empty `changed`) fires the fan-out and resets the counter, so
# it can land later than exactly N. `:off` disables; `:eager` sends an empty reload list (the
# on-demand region refills lazily as requests arrive); `:scheduled` sends each changed worker the set
# of models this repack assigned to it, warming the new placement. Runs on the prober task.
function _maybe_compact_fleet!(s::LptPackingState, pool::ClientPool,
                               metrics::Union{GatewayMetrics,Nothing}, changed::Set{String};
                               k::PackingKnobs=knobs(s))
    (k.compaction_mode == :off || k.compaction_interval <= 0) && return nothing
    s.repacks_since_compact += 1
    (s.repacks_since_compact >= k.compaction_interval && !isempty(changed)) || return nothing
    s.repacks_since_compact = 0

    perworker = Dict{String,Vector{String}}(w => String[] for w in changed)
    if k.compaction_mode == :scheduled
        for (m, placement) in @atomic(s.assignment), (w, _) in placement
            haskey(perworker, w) && push!(perworker[w], m)
        end
    end
    total, ok, failed = _compact_workers(pool, metrics, perworker)
    @info "lpt_packing: compaction" mode = k.compaction_mode workers = ok reloaded = total failed = failed
    return nothing
end

"""
    rebalance!(s, pool, ready_urls, metrics) -> nothing

Force a repack now: poll the ready workers and recompute the assignment unconditionally. Used at
startup (so the first requests already route by packing) and by tests. The periodic, compute-driven
path is [`tick_packing!`](@ref).
"""
function rebalance!(s::LptPackingState, pool::ClientPool, ready_urls::Vector{String},
                    metrics::Union{GatewayMetrics,Nothing}=nothing; trigger::Symbol=:startup)
    _repack!(s, _poll_workers(pool, ready_urls), metrics; trigger = trigger)
    return nothing
end

"""
    tick_packing!(s, pool, ready_urls, metrics) -> nothing

One prober tick: poll the ready workers, refresh the routing metadata, accumulate the fleet's
consumed compute, and repack only when the accumulated compute crosses the active budget. The first
tick-driven repack uses `first_rebalance_compute_seconds` (when set); every repack after uses
`rebalance_compute_seconds`. The cheap per-tick work is the poll and the accumulator; the EWMA fold
and `compute_assignment` run only on a triggered repack.
"""
function tick_packing!(s::LptPackingState, pool::ClientPool, ready_urls::Vector{String},
                       metrics::Union{GatewayMetrics,Nothing}=nothing)
    # Read the forced-repack request BEFORE the knobs: the acquire pairs with the release a handler
    # did when it swapped `knobs`, so any knob or override written before a request bump is visible to
    # the repack that serves it. `SetModelPlacement` followed by `Repack` always lands together.
    requested = @atomic s.repack_requested
    # One knob snapshot for the whole tick, so a policy change landing mid-tick cannot produce a
    # repack that mixes old and new settings.
    k = knobs(s)
    poll = _poll_workers(pool, ready_urls)
    @atomic s.max_batch = poll.max_batch         # keep routing metadata fresh between repacks
    _publish_fill_plan!(s, k, poll.max_batch, poll.batch_at)   # ...and re-resolve the fill rules
    _refresh_live_gauges!(s, metrics)            # in-flight and quantum, refreshed between repacks
    if s.last_fleet_compute == 0.0
        s.last_fleet_compute = poll.fleet_compute   # first observation: baseline only
    else
        delta = poll.fleet_compute - s.last_fleet_compute
        s.last_fleet_compute = poll.fleet_compute
        delta > 0 && (s.compute_accum += delta)     # negative delta = worker restart: re-baseline
    end
    # The first tick-driven repack may use a smaller budget so an early rebalance corrects the cold
    # placement quickly; steady-state then uses the larger budget. The flag is flipped here, not in
    # `_repack!`, because the startup `rebalance!` also calls `_repack!` and must not consume it.
    threshold = (!s.did_first_tick_repack && k.first_rebalance_compute_seconds > 0) ?
                k.first_rebalance_compute_seconds : k.rebalance_compute_seconds
    forced = requested > (@atomic s.repack_completed)
    if forced || s.compute_accum >= threshold
        # A forced repack does not consume the first-repack budget, for the same reason the startup
        # cold placement does not: that budget is about the first repack driven by real traffic.
        forced || (s.did_first_tick_repack = true)
        changed = _repack!(s, poll, metrics; trigger = forced ? :operator : :compute, k = k)
        _maybe_compact_fleet!(s, pool, metrics, changed; k = k)
        # Stored LAST, after the assignment swap and the report publish, so a handler that sees its
        # sequence is guaranteed to observe this repack's results.
        forced && @atomic s.repack_completed = requested
    end
    # Published after any repack, so `compute_accum` reflects the reset and a reader never sees the
    # pre-repack accumulator paired with a post-repack assignment.
    @atomic s.tick = TickReport(time(), sort(ready_urls), poll.fleet_compute, s.compute_accum,
                                threshold, s.did_first_tick_repack)
    return nothing
end

# Republish every model's resolved routing rule (fill mode and quantum) from the current knobs and the
# worker-reported max batches. Called wherever `max_batch` is refreshed, so both a knob change and a
# newly reported batch shape reach the request path on the next tick, and the request path never has
# to walk the per-model override chain. Prober task only; the swap is atomic.
function _publish_fill_plan!(s::LptPackingState, k::PackingKnobs, max_batch::Dict{String,Int},
                             batch_at::Dict{String,Tuple{String,Int}} = Dict{String,Tuple{String,Int}}())
    plan = Dict{String,FillPlan}()
    for (m, mb) in max_batch
        bi, ax = get(batch_at, m, ("", 0))
        plan[m] = resolve_fill_plan(k, m, mb, bi, ax)
    end
    # A model with an override but no reported max batch still gets a plan, so its configured mode
    # applies from its first request rather than only after a successful poll.
    for m in keys(k.model_overrides)
        haskey(plan, m) || (plan[m] = resolve_fill_plan(k, m, 0))
    end
    @atomic s.fill_plan = plan
    return plan
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
function _reserve_on!(out_snap, routed_snap, wload_snap, model, w, weight, units)
    mwc = get(out_snap, (model, w), nothing)
    wload = get(wload_snap, w, nothing)
    rtd = get(routed_snap, (model, w), nothing)
    mwc === nothing || Threads.atomic_add!(mwc, units)   # in-flight ITEMS
    wload === nothing || Threads.atomic_add!(wload, weight)
    rtd === nothing || Threads.atomic_add!(rtd, 1)       # cumulative REQUESTS: never released
    return (mwc, wload, weight, units)
end

"""
    route_replica(s, model) -> Union{Nothing, Tuple{Vector{String}, Counters}}

Order a model's replicas for dispatch and reserve the chosen one. Returns `nothing` when the model
has no placement yet (cold or unknown); the caller falls back to round robin. Otherwise returns the
ordered replica URLs (the chosen worker first, the rest as failover) and `Counters`, the reserved
counters to release when the request completes.

The model's `fill_mode` (see [`FillPlan`](@ref)) decides what the quantum `Q` counts, and
`routing_policy` decides which replica a fresh run opens on:

  - `:run` (default): `Q` counts requests **routed** to the current replica. A run of `Q` requests
    goes to one replica, then the next run opens elsewhere, so the model's GPUs serve it in turn:
    batches stay deep and the share is even to within one run length at any concurrency. A run also
    ends when the model has nothing in flight (no batch to protect, so it rotates per request) or when
    its replica falls a whole quantum behind the least-backed-up one (backpressure: a slow replica
    loses its turn rather than accumulating a queue).
  - `:spread`: equalize in-flight items across the set, so all `k` GPUs serve the model at once.
    Lowest latency for a latency-bound model whose concurrency is well below its max batch, at the
    cost of coalescing depth.
  - `:inflight` (legacy): `Q` counts requests **in flight**, so a replica keeps receiving the model's
    traffic until it holds a full quantum at once. A model whose in-flight concurrency stays between 2
    and `Q` and never drains is served by one replica indefinitely; see the warning on
    `GatewayConfig.routing_fill_mode`.

`routing_policy` applies at every run boundary, not only when replicas tie: `fill_rr` rotates
(load-blind and exactly even), `fill_least` opens on the replica whose worker carries the least
in-flight compute load across all models. Exact ties rotate rather than falling through to the URL,
so an idle fleet warms every replica instead of pinning to the lowest-named one.

The returned order is the chosen replica first, then the remaining replicas least-backed-up first, so
a failover after `NotFound`/`Unavailable` lands on the emptiest alternative.

Every routed request, single- or multi-replica, bumps the per-worker compute-load counter, so
`fill_least` sees load from all models. The reservation (atomic increments under the per-model
selection lock for multi-replica models) makes concurrent selections see the choice, so requests do
not stampede onto the same replica.
"""
function route_replica(s::LptPackingState, model::AbstractString, units::Integer=1)
    placement = get(@atomic(s.assignment), model, nothing)
    placement === nothing && return nothing
    n = length(placement)
    n == 0 && return nothing

    out_snap = @atomic s.outstanding
    routed_snap = @atomic s.routed
    wload_snap = @atomic s.worker_load
    weight = _route_weight(s, model)
    # One resolved plan per request: the mode and the quantum must come from the same generation.
    # `units` is how many items this request carries, already resolved by `request_units`.
    plan = get(@atomic(s.fill_plan), model, _DEFAULT_FILL_PLAN)
    units = max(Int(units), 1)

    if n == 1
        w = placement[1][1]
        return (String[w], _reserve_on!(out_snap, routed_snap, wload_snap, model, w, weight, units))
    end

    workers = String[p[1] for p in placement]
    lk = get(@atomic(s.sel_locks), model, nothing)
    lk === nothing && return (workers, nothing)   # mid-repack drift: route in order, untracked
    policy = knobs(s).routing_policy
    Q = plan.quantum
    all_runs = @atomic s.fill_runs

    res = lock(lk) do
        cobjs = Vector{Threads.Atomic{Int}}(undef, n)
        outs = Vector{Int}(undef, n)
        total = 0
        for i in 1:n
            a = get(out_snap, (model, workers[i]), nothing)
            a === nothing && return nothing       # snapshot drift: bail to untracked routing
            cobjs[i] = a
            outs[i] = a[]
            total += outs[i]
        end
        # Mid-repack drift: decide with a throwaway record rather than skipping the fill logic, so a
        # request between the swap and the next tick still routes sanely (it just does not persist).
        run = get(all_runs, model, nothing)
        run === nothing && (run = FillRun())
        # Read lazily: only the fill_least paths need it, and reading per candidate avoids the
        # Vector{Float64} this function used to allocate on every multi-replica request.
        wload_of(i) = (wa = get(wload_snap, workers[i], nothing); wa === nothing ? 0.0 : wa[])
        rot(i) = mod(i - 1 - run.cursor, n)       # 0-based distance from the rotation cursor

        chosen = 0
        if plan.mode === :spread
            # All replicas serve the model at once; ties go to the policy so an idle set still rotates.
            chosen = policy === :fill_least ?
                     argmin(i -> (outs[i], wload_of(i), rot(i)), 1:n) :
                     argmin(i -> (outs[i], rot(i)), 1:n)
            run.cursor = mod(chosen, n)
            run.target = chosen
            run.units = units
        elseif plan.mode === :run
            # `behind` is the backpressure guard: a replica a whole quantum deeper than the
            # least-backed-up one can neither keep nor open a run.
            floor_q = minimum(fld(outs[i], Q) for i in 1:n)
            behind(i) = fld(outs[i], Q) > floor_q
            if run.target < 1 || run.target > n || run.units >= Q || behind(run.target) || total == 0
                chosen = policy === :fill_least ?
                         argmin(i -> (behind(i), wload_of(i), rot(i)), 1:n) :
                         argmin(i -> (behind(i), rot(i)), 1:n)
                run.cursor = mod(chosen, n)
                run.target = chosen
                run.units = units
            else
                chosen = run.target
                # A run is spent by ITEMS, so one request carrying a full batch closes it just as 32
                # single-item requests would. This is the whole point of the item denomination: a
                # client that pre-batches to max_batch would otherwise hold a replica for Q batches.
                run.units += units
            end
        else   # :inflight (legacy)
            # Fill progress in whole quanta, then the deepest replica within that bucket, which is
            # what keeps a mid-fill replica winning until it holds a full quantum.
            prog(i) = (fld(outs[i], Q), -outs[i])
            if policy === :fill_least
                chosen = argmin(i -> (prog(i), wload_of(i), rot(i)), 1:n)
            else
                chosen = argmin(i -> (prog(i), rot(i)), 1:n)
            end
            # The cursor advances only on a genuine batch start (more than one replica tied at the
            # best progress), so a lone mid-fill winner keeps concentrating.
            best = prog(chosen)
            count(i -> prog(i) == best, 1:n) > 1 && (run.cursor = mod(chosen, n))
            run.target = chosen
            run.units = units
        end

        Threads.atomic_add!(cobjs[chosen], units)
        wload = get(wload_snap, workers[chosen], nothing)
        wload === nothing || Threads.atomic_add!(wload, weight)
        rtd = get(routed_snap, (model, workers[chosen]), nothing)
        rtd === nothing || Threads.atomic_add!(rtd, 1)
        # Chosen first, then least-backed-up, as the failover order.
        order = collect(1:n)
        sort!(order; by = i -> (i == chosen ? 0 : 1, fld(outs[i], Q), outs[i], workers[i]))
        ordered = String[workers[i] for i in order]
        return (ordered, cobjs[chosen], wload)
    end
    res === nothing && return (workers, nothing)
    ordered, mwc, wload = res
    return (ordered, (mwc, wload, weight, units))
end

# Release a reservation made by route_replica: decrement the per-(model,worker) request counter and
# subtract the captured compute weight from the per-worker load. Robust to the failure path: called
# once in a finally regardless of how the dispatch ended, and to any counter that was absent at
# reservation time (drift), which is stored as `nothing`.
function _release_route!(counters)
    counters === nothing && return nothing
    mwc, wload, weight, units = counters
    # The captured `units` and `weight` are subtracted, not recomputed: a repack between reservation
    # and release changes both, and recomputing would leave the counters drifting.
    mwc === nothing || Threads.atomic_sub!(mwc, units)
    wload === nothing || Threads.atomic_sub!(wload, weight)
    return nothing
end

# --- GatewayScheduler interface (see scheduler.jl) --------------------------------------------
# `record_arrival!(s::LptPackingState, model)` is defined above and is the specialization of the
# generic for this scheduler; the rest of the interface is adapted here.

release!(::LptPackingState, reservation) = _release_route!(reservation)

# How many items a request carries, for the item-denominated quantum and in-flight counters. The
# worker told us where to look (see `FillPlan`); a model that declares no batch axis, or a request
# missing that input, counts as one item. Only this scheduler pays the peek: the generic default in
# scheduler.jl returns 1 without touching the body.
function request_units(s::LptPackingState, model::AbstractString, body::AbstractVector{UInt8})
    plan = get(@atomic(s.fill_plan), model, nothing)
    (plan === nothing || plan.batch_axis < 1) && return 1
    n = peek_batch_size(body, plan.batch_input, plan.batch_axis)
    return n > 0 ? n : 1
end

scheduler_repack_seq(s::LptPackingState) = @atomic s.repack_requested

scheduler_tick!(s::LptPackingState, pool::ClientPool, ready_urls, metrics) =
    tick_packing!(s, pool, ready_urls, metrics)

# Hard startup preconditions (all workers reachable, FIFO discipline, identical model sets), then an
# initial rebalance so the first requests already route by packing rather than waiting a prober tick.
function scheduler_start!(s::LptPackingState, pool::ClientPool, metrics)
    verify_lpt_packing_preconditions!(pool; wait_seconds = _startup_wait_seconds())
    k = knobs(s)
    @info "gateway scheduling: lpt_packing" rebalance_compute_seconds = k.rebalance_compute_seconds first_rebalance_compute_seconds = k.first_rebalance_compute_seconds ema_halflife_compute = k.ema_halflife_compute default_replicas = (k.default_replicas == REPLICAS_ALL ? "all" : k.default_replicas) routing_policy = k.routing_policy routing_fill_mode = k.routing_fill_mode forbid_memory_oversubscription = k.forbid_memory_oversubscription
    _warn_inflight_fill(k)
    metrics === nothing || register_routed!(s, metrics)
    rebalance!(s, pool, copy(pool.order), metrics)
    return nothing
end

# `inflight` is supported but hazardous: it parks a model on one replica for any sustained concurrency
# below the quantum. An operator who selects it (fleet-wide or per model) is told once per gateway
# start rather than being left to discover it from an idle GPU.
const _INFLIGHT_PARKING_WARNING = "fill mode 'inflight' counts requests in flight, so a model whose in-flight concurrency stays between 2 and its quantum and never drains to zero is served by ONE replica indefinitely while the others receive no traffic. Diagnose with gateway_replica_routed_total (a flat series on one replica of a multi-replica model); prefer 'run', or 'spread' when a model needs every GPU at once."

function _warn_inflight_fill(k::PackingKnobs)
    per_model = sort!(String[m for (m, mk) in k.model_overrides if mk.fill_mode === :inflight])
    if k.routing_fill_mode === :inflight
        @warn "lpt_packing: $(_INFLIGHT_PARKING_WARNING)" scope = "fleet default (scheduling.routing_fill_mode)"
    end
    isempty(per_model) ||
        @warn "lpt_packing: $(_INFLIGHT_PARKING_WARNING)" scope = "per-model override" models = per_model
    return nothing
end

# Route to the placement replica that fills its batch first (route_replica reserves it), the rest of
# the placement following as failover, then any discovered replicas outside the placement as a
# last resort so a concentrated model survives its worker dying between repacks. A model without a
# placement yet (cold, or new since the last repack) falls back to round robin over discovered routes.
function select_replicas(s::LptPackingState, ctx::ScheduleContext)
    routed = route_replica(s, ctx.model, ctx.units)
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
            # level=:debug: at startup a worker not answering is the expected warmup window (it
            # compiles before its control plane answers), not a fault. The per-worker timeout is
            # quieted to debug; the once-per-round "waiting for all workers" @info below is the
            # operator-facing progress line. The runtime prober keeps the default :warn.
            resp, to = _bounded(() -> fetch_control_status(wc), call_timeout, nothing,
                                "ModelControlStatus poll", wc.url; level = :debug)
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
    # The loop only falls through here once every worker answered (its other exits are error()/exit(1)),
    # so log an explicit all-ready confirmation: the "waiting" line above never prints in the all-ready
    # state (it is past the `isempty(pending) && break`), and a fast start answers on the first poll
    # without ever logging "waiting".
    @info "lpt_packing: all workers ready" count = length(statuses) workers = sort(collect(keys(statuses)))
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
