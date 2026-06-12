# LPT-packing scheduling: concentrate each model's traffic on few workers so the workers' batch
# coalescing has deep same-model queues to draw from, while balancing expected utilization across
# workers to avoid contention. Uniform spreading (round robin) is the worst case for coalescing
# when every model is loaded on every worker; concentration is the point of this mode.
#
# Each model m gets a probability distribution p_m over workers (weights sum to 1). Per request
# the gateway samples a worker from p_m (failover order: remaining replicas by descending weight).
# Distributions are recomputed on a rebalance tick from per-model expected utilization
# u_m = lambda_m * c_m, where lambda_m is the gateway-measured arrival rate (EWMA) and c_m is the
# true per-request compute cost reported by the workers over the control plane
# (delta total_compute_seconds / delta requests_served between polls, EWMA-smoothed) - free of
# queueing effects, unlike gateway-observed latency.
#
# Packing is two-dimensional: besides compute time, every model occupies resident weight memory
# (ModelStatus.weight_nbytes) on each worker that serves it, and every worker has a weight-memory
# budget (ModelControlStatusResponse.weight_cache_max_bytes, its on-demand cache). The placement
# score is the max of the normalized compute and memory loads, so the packer simultaneously
# minimizes weight eviction/loading (memory pressure) and keeps every GPU busy (compute balance).
#
# LPT-packing mode requires worker FIFO discipline and all models on all workers, verified as a hard
# failure at gateway startup (see verify_lpt_packing_preconditions!). Runtime drift degrades
# gracefully: a dead worker is excluded from placement until ready; a model temporarily missing
# from some workers gets a uniform distribution over its actual replicas with a warning.
#
# v2 direction (not implemented): make a hot model's slice adapt to observed queue depth rather
# than the static max_worker_share; the share cap is the anti-starvation floor (no model can claim
# all of a worker's expected capacity, and the worker's per-model queue caps bound damage under
# FIFO).

# One model's placement: worker URLs with sampling weights, sorted by descending weight.
const Placement = Vector{Tuple{String,Float64}}

"""
    compute_assignment(u, workers, prev; mem=Dict(), mem_cap=Dict(), max_share=0.8, hysteresis=0.1)
        -> Dict{String,Placement}

Pure assignment math (no I/O): greedy concentration with two-dimensional (vector) bin packing.
Every model demands compute time (`u[m]`, GPU-seconds/second) and resident weight memory
(`mem[m]`, bytes); every worker offers compute capacity 1.0 and a weight-memory budget
(`mem_cap[url]`, bytes; absent or `<= 0` means unconstrained, i.e. all weights resident). Models
are placed in descending compute-utilization order (memory-heavier first among ties), each wholly
on the worker that minimizes the placement's resulting pressure: the maximum of the worker's
normalized compute load and normalized memory load. The max-norm balances the two competing
concerns: a memory-full worker stops attracting models even when compute-idle (avoiding eviction
churn), and a compute-hot worker stops attracting them even with memory free (avoiding idle GPUs).

A model whose compute demand exceeds `max_share` is split evenly across the minimum number of
lowest-pressure workers that brings each share under the cap; its weights are charged to every
member (spreading a model costs memory everywhere it serves, which is exactly why concentration
is memory-efficient). Hysteresis keeps a model's previous placement unless moving improves its
resulting pressure by more than the threshold - placement stability is what coalescing buys
from. Workers no longer present are ignored in `prev`. Cold models (no traffic yet) carry
`u[m] == 0` and are placed like any other, packed by memory; models absent from `u` entirely are
not placed and the caller routes them uniformly.
"""
function compute_assignment(u::Dict{String,Float64}, workers::Vector{String},
                            prev::Dict{String,Placement};
                            mem::Dict{String,Float64}=Dict{String,Float64}(),
                            mem_cap::Dict{String,Float64}=Dict{String,Float64}(),
                            max_share::Float64=0.8, hysteresis::Float64=0.1)
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
        if um <= max_share || length(workers) == 1
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
            # Split a model too hot for one worker across the minimum number of workers that
            # brings each share under the cap, lowest pressure first (previous members win ties
            # for stability). Even weights keep every share identical and the sum at 1. Weights
            # are resident on every member, so each is charged the full footprint.
            k = clamp(ceil(Int, um / max_share), 2, length(workers))
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
    max_share::Float64
    hysteresis::Float64
    rate_halflife::Float64
    rebalance_seconds::Float64
    # arrival counting: a copy-on-write snapshot dict of per-model atomic counters. Reads (the
    # request hot path) touch only the immutable snapshot; insertion of a new model swaps in a
    # copy under the lock.
    @atomic arrivals::Dict{String,Threads.Atomic{Int}}
    lock::ReentrantLock
    # EWMAs and the cumulative baselines for worker-counter deltas (model -> (compute, requests)).
    rate_ewma::Dict{String,Float64}          # requests/sec
    cost_ewma::Dict{String,Float64}          # compute seconds/request
    last_cum::Dict{String,Tuple{Float64,UInt64}}
    last_rebalance::Float64
    # the live assignment, swapped atomically; readers never lock
    @atomic assignment::Dict{String,Placement}
    # label pairs previously exported to the packing-weight gauge, zeroed when dropped
    exported::Set{Tuple{String,String}}
end

LptPackingState(cfg::GatewayConfig) = LptPackingState(
    cfg.max_worker_share, cfg.hysteresis, cfg.rate_halflife_seconds, cfg.rebalance_seconds,
    Dict{String,Threads.Atomic{Int}}(), ReentrantLock(),
    Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Tuple{Float64,UInt64}}(),
    0.0, Dict{String,Placement}(), Set{Tuple{String,String}}())

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

"""
    rebalance!(s, pool, ready_urls, metrics) -> nothing

One packing tick: drain the arrival counters into rate EWMAs, poll `ModelControlStatus` from the
ready workers and fold compute-cost deltas into cost EWMAs, recompute the assignment
(`compute_assignment`), atomically swap it in, and export the placement to the metrics gauges.
A model not reported by every ready worker (runtime drift, e.g. a dynamic load in progress on some
workers) gets a uniform placement over the workers that do serve it, with a warning.
"""
function rebalance!(s::LptPackingState, pool::ClientPool, ready_urls::Vector{String},
                    metrics::Union{GatewayMetrics,Nothing}=nothing)
    now = time()
    dt = s.last_rebalance == 0.0 ? s.rebalance_seconds : max(now - s.last_rebalance, 1e-3)
    s.last_rebalance = now

    # Arrival rates.
    counters = @atomic s.arrivals
    for (m, c) in counters
        n = Threads.atomic_xchg!(c, 0)
        s.rate_ewma[m] = _ewma(get(s.rate_ewma, m, 0.0), n / dt, dt, s.rate_halflife)
    end

    # Worker-reported costs: aggregate cumulative (compute, requests) per model across the ready
    # workers, then delta against the previous poll. A negative delta means a worker restarted
    # (counters reset); re-baseline and skip the cost sample this tick. The same poll carries
    # each model's weight footprint and each worker's weight-memory budget, the second packing
    # dimension.
    sums = Dict{String,Tuple{Float64,UInt64}}()
    permodel_workers = Dict{String,Vector{String}}()
    mem = Dict{String,Float64}()                 # model -> resident weight bytes
    mem_cap = Dict{String,Float64}()             # worker -> on-demand weight budget (0 = unconstrained)
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
                    push!(get!(permodel_workers, ms.name, String[]), url)
                end
            end
        end
    end
    for (m, (tc, rq)) in sums
        prev_tc, prev_rq = get(s.last_cum, m, (0.0, UInt64(0)))
        dtc, drq = tc - prev_tc, Int(rq) - Int(prev_rq)
        s.last_cum[m] = (tc, rq)
        (dtc < 0 || drq < 0) && continue          # worker restart: re-baseline only
        drq > 0 && (s.cost_ewma[m] = _ewma(get(s.cost_ewma, m, dtc / drq), dtc / drq, dt, s.rate_halflife))
    end

    # Expected utilization. Every fully-replicated model is packed, including cold ones (no
    # traffic yet, u = 0): they still occupy weight memory, so the packer gives each a
    # concentrated home placed by the memory dimension rather than spreading its weights across
    # every worker. A model missing from some polled workers (runtime drift, e.g. a dynamic load
    # in progress) routes uniformly over its actual replicas until the fleet converges.
    full = Dict{String,Float64}()
    drifted = Dict{String,Placement}()
    nready = length(polled)
    for (m, ws) in permodel_workers
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
    next = compute_assignment(full, sort(polled), prev;
                              mem=mem, mem_cap=mem_cap,
                              max_share=s.max_share, hysteresis=s.hysteresis)
    merge!(next, drifted)
    @atomic s.assignment = next

    # Memory oversubscription warning: when a worker's assigned weight footprint exceeds its
    # on-demand budget the packing is infeasible (total weights outgrew the fleet); the worker's
    # LRU cache degrades gracefully, but the operator should know.
    assigned_mem = Dict{String,Float64}(w => 0.0 for w in polled)
    for (m, placement) in next, (w, _) in placement
        haskey(assigned_mem, w) && (assigned_mem[w] += get(mem, m, 0.0))
    end
    for (w, bytes) in assigned_mem
        cap = get(mem_cap, w, 0.0)
        cap > 0 && bytes > cap &&
            @warn "lpt_packing: assigned weight footprint exceeds the worker's on-demand budget; expect eviction churn" worker = w assigned = Base.format_bytes(round(Int, bytes)) budget = Base.format_bytes(round(Int, cap))
    end

    if metrics !== nothing
        live = Set{Tuple{String,String}}()
        for (m, placement) in next, (w, weight) in placement
            set_placement_weight!(metrics, m, w, weight)
            push!(live, (m, w))
        end
        for (m, w) in setdiff(s.exported, live)
            set_placement_weight!(metrics, m, w, 0.0)
        end
        s.exported = live
        for (m, um) in full
            set_model_utilization!(metrics, m, um)
        end
    end
    return nothing
end

"""
    pick_placement(s, model) -> Union{Vector{String},Nothing}

Weighted-sample the model's placement: the sampled worker first, then the remaining replicas by
descending weight as failover targets (same contract as the round-robin `pick`). Returns `nothing`
when the model has no placement yet (cold or unknown); the caller falls back to round robin.
"""
function pick_placement(s::LptPackingState, model::AbstractString)
    placement = get(@atomic(s.assignment), model, nothing)
    placement === nothing && return nothing
    n = length(placement)
    n == 0 && return nothing
    n == 1 && return String[placement[1][1]]
    r = rand()
    acc = 0.0
    idx = n
    for (i, (_, w)) in enumerate(placement)
        acc += w
        if r < acc
            idx = i
            break
        end
    end
    urls = String[placement[idx][1]]
    order = sort([i for i in 1:n if i != idx]; by = i -> -placement[i][2])
    append!(urls, String[placement[i][1] for i in order])
    return urls
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
