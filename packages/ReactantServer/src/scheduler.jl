# The scheduler: a weighted-fair, coalescing dispatch policy.
#
# Requests arrive concurrently and are placed in per-model queues. Whenever the GPU is free
# the scheduler decides which model to dispatch next, coalesces that model's queued requests
# into a single execution at a compiled batch size, runs it, and splits the outputs back to
# each caller. The policy layer rides on the language task scheduler; it does not implement
# its own threading. Only one execution runs at a time by design.
#
# Decision order each time the GPU frees up:
#   1. Pick the model with a non-empty queue that has been charged the least device time
#      for its weight (or, under the FIFO discipline, the model with the oldest queued
#      request; under EDF, the soonest deadline).
#   2. Coalesce the selected model's queued requests (FIFO) into one compiled-batch-size
#      dispatch; the remainder stays queued.
#
# Per-model latency budgets (earliest-deadline-first escalation) are a planned extension;
# they are not implemented.
#
# A single Threads.Condition guards all mutable scheduler state (the per-model queues and the
# virtual-time, EMA and cost fields) and signals the dispatch loop when work arrives. The loop
# holds the lock only to select and dequeue; the actual execution runs outside the lock so
# producers keep enqueuing during GPU work, then the lock is re-taken briefly to record the
# dispatch.

const _WAIT_SAMPLE_CAP = 1024         # bounded ring of recent queue-wait samples per model

# How the learned cost of a compiled program moves (see `_update_cost!`). One measurement raises an
# estimate by at most `_COST_MAX_RISE` of itself. A new measurement's weight is `cost_ema_alpha` for
# back-to-back runs and grows toward 1 as the estimate ages, the estimate's own weight halving every
# `_COST_HALFLIFE_SECONDS` since it was last refreshed. Under EDF, an estimate not refreshed for
# `_COST_STALE_SECONDS` is measured again before the laxity check sheds a whole dispatch on it (see
# `execute_and_record!`).
const _COST_MAX_RISE = 0.25
const _COST_HALFLIFE_SECONDS = 2.0
const _COST_STALE_SECONDS = 10.0

# `ModelSchedState` (the per-model scheduling state) is defined in `runtime/model_types.jl` so
# that `ModelEntry` can hold it; its methods live here.

# A control command runs on the dispatch-loop thread (the sole mutator of residency). `op` takes
# the scheduler and returns a result; the result (or any thrown error) is delivered on `reply`.
struct ControlCommand
    op::Function
    reply::Channel{Any}
end

# Admits at most `capacity` meta orchestrations at once on this worker. The constructor defaults to 1,
# as does a bare `Scheduler`; a served worker passes REACTANT_META_CONCURRENCY (default 2, see
# server.jl `_meta_concurrency`). A compute-only meta (no `calls`) bypasses the gate. A meta holds a
# permit across its WHOLE run, including the CPU glue between stages, so at most `capacity` metas' GPU
# sub-calls ever compete with regular work; the GPU itself is not held during glue (sub-calls dispatch
# on the loop). `acquire_meta_gate!` is deadline-bounded: a meta that cannot get a permit before its
# deadline is shed before doing any GPU work, so a meta backlog sheds at admission rather than piling
# up.
mutable struct MetaGate
    cond::Threads.Condition   # guards `n_held`
    capacity::Int
    n_held::Int
end
MetaGate(capacity::Integer = 1) = MetaGate(Threads.Condition(), Int(capacity), 0)

function acquire_meta_gate!(g::MetaGate, who::AbstractString; deadline_ns::Integer = 0)
    dl = Int64(deadline_ns)
    lock(g.cond)
    timer = nothing   # one-shot wake at the deadline; armed on first park, closed in finally
    try
        while g.n_held >= g.capacity
            dl != 0 && Int64(time_ns()) >= dl && throw(DeadlineExceeded(String(who)))
            if dl != 0 && timer === nothing
                rem = (dl - Int64(time_ns())) / 1.0e9
                rem <= 0 && throw(DeadlineExceeded(String(who)))
                timer = Timer(
                    _ -> (
                        try
                            @lock g.cond notify(g.cond)
                        catch
                        end
                    ), rem
                )
            end
            wait(g.cond)   # releases the lock while parked, reacquires on wake
        end
        g.n_held += 1
    finally
        timer === nothing || close(timer)
        unlock(g.cond)
    end
    return nothing
end

function release_meta_gate!(g::MetaGate)
    @lock g.cond begin
        g.n_held -= 1
        notify(g.cond)
    end
    return nothing
end

# A background read of one model's weights into host memory, from its start until it returns,
# stalled or not (see "Staging" below). `kind` is what it reads for: `:stage`, the model's queued
# requests; `:prefetch`, a StageWeights hold; `:residency`, a residency change (`set_residency!`).
# Guarded by the scheduler's `cond`, except `progress`, which the reading task advances as bytes
# arrive.
mutable struct WeightRead
    entry::ModelEntry
    kind::Symbol
    prefetch::Any                     # the StageWeights hold (`Prefetched`) it fills, or nothing
    progress::ReadProgress
    task::Union{Task, Nothing}
    timer::Union{Timer, Nothing}      # the stall check, while it is watched
    started::Float64
    ended::Bool                       # it returned
    stalled::Bool                     # abandoned for want of progress: nothing waits for it any more
    close_weights::Bool               # its model was unloaded meanwhile: close the weights file once it returns
    outcome::Any                      # StageWeights: the model's row (state, read, seconds, error);
    # a residency change: `Some(result)`, or the exception it failed or stalled with
    done::Base.Event                  # StageWeights, a residency change: notified once `outcome` is set
end

# The most stalled reads a worker leaves outstanding before it starts no more: each holds a
# default-pool thread in a system call nothing can interrupt, and the memory it reads into. Four, or
# a quarter of the default pool when that is fewer, and at least one.
const _MAX_STALLED_READS = 4
_stalled_read_bound() = clamp(Threads.nthreads(:default) ÷ 4, 1, _MAX_STALLED_READS)

"""
    DISPATCH_STALL_SECONDS

The default of how long the dispatch loop may go without progress, with work in hand, before the
worker reports itself not live (and not ready): 600 s. A dispatch, a weight upload, a compaction
or a control command is one unit of work, and a legitimate one can take seconds to minutes (a large
model read from slow storage, a dynamic-mode compile), so the default is well beyond that and twenty
times the default request timeout (`scheduler.dispatch_timeout_seconds`, 30 s): a loop that has
finished nothing for that long while requests wait serves nobody.
`REACTANT_LIVENESS_DISPATCH_STALL_SECONDS` sets it (`0` turns the check off); the node supervisor
passes its `liveness.dispatch_stall_seconds`.
"""
const DISPATCH_STALL_SECONDS = 600.0

# Whether the dispatch loop is making progress. The loop stamps `since` when it takes up work and
# again after each unit of work it finishes, and clears it (0) while it waits with nothing to do, so
# `since` is the last sign of progress while it holds work. Read without the scheduler lock (which a
# stuck loop may hold) by the liveness and readiness checks, through `dispatch_stalled`.
mutable struct DispatchProgress
    const since::Threads.Atomic{Float64}
    const reported::Threads.Atomic{Bool}   # the current stall has been logged
    stall_seconds::Float64                 # no progress this long with work in hand is a stall; 0 = never
end
DispatchProgress(stall_seconds::Real = DISPATCH_STALL_SECONDS) =
    DispatchProgress(Threads.Atomic{Float64}(0.0), Threads.Atomic{Bool}(false), Float64(stall_seconds))

_mark_progress!(p::DispatchProgress) = (p.since[] = time(); p.reported[] = false; nothing)
_mark_idle!(p::DispatchProgress) = (p.since[] = 0.0; p.reported[] = false; nothing)

"""
    Scheduler

The weighted-fair, coalescing dispatch engine. Concurrent
requests land on per-model queues; a single dispatch loop runs one GPU execution at a time,
coalescing same-model requests into one batched execution at a compiled size and sharing GPU
time by relative model weight, charged in measured device seconds. It holds the
model registry, the backend, the device memory pool, the [`SchedulerConfig`](@ref),
per-model dispatch state, and an optional on-demand `WeightCache`. Submit work with
[`infer`](@ref) and read observability counters with [`scheduler_metrics`](@ref).
"""
mutable struct Scheduler
    registry::ModelRegistry          # single source of truth for which models exist; holds per-model sched state
    backend::AbstractBackend
    pool::MemoryPool
    cfg::SchedulerConfig
    cond::Threads.Condition          # guards the registry map + queues + control and signals the dispatch loop
    running::Bool
    task::Union{Task, Nothing}
    weight_cache::Union{WeightCache, Nothing}   # on-demand weight residency, or nothing when disabled
    scratch_pool::Union{BufferPool, Nothing}    # local reuse pool for meta intermediates (plain Memory); nothing disables
    control::Vector{ControlCommand}            # pending control commands, drained by the dispatch loop
    meta_gate::MetaGate                        # admits `capacity` GPU-using metas at once: 1 as constructed,
    # REACTANT_META_CONCURRENCY (default 2) on a served worker
    committed::Vector{QueuedRequest}           # in-flight metas' continuation sub-calls awaiting the next GPU slot, each
    # dispatched ahead of the discipline scan. One entry per in-flight meta, so
    # it is bounded by the gate: at most its capacity (REACTANT_META_CONCURRENCY on a served worker).
    compact_loads_mark::Int                    # weight-cache load count at the last automatic compaction; sole
    # writer is the dispatch thread. Drives load-driven compaction:
    # compact once `weight_cache.loads` advances by `cfg.compaction_interval`.
    autosize::Union{Nothing, NamedTuple{(:arena, :fraction, :wiggle), Tuple{Int, Float64, Float64}}}
    # the arena and knobs the startup auto-sizing ran with, kept so a hot-loaded
    # model can be probed and the weight budget re-resolved without a restart
    residency_lock::ReentrantLock              # one residency change at a time (`set_residency!`)
    prefetched::Dict{String, Any}              # weights staged ahead of a placement change, by model
    # (`stage_weights!`); guarded by `cond`
    failure::Any                               # the exception that killed the dispatch loop, or nothing
    # while it serves (see `_dispatch_loop_died!`); guarded by `cond`
    on_failure::Any                            # `f(err)` run once if the dispatch loop dies, or nothing
    vfloor::Float64                            # virtual time of the most recent fair dispatch; never
    # decreases (see `_select_fair!`); guarded by `cond`
    progress::DispatchProgress                 # the dispatch loop's last progress (see `dispatch_stalled`)
    reads::Vector{WeightRead}                  # background weight reads that have not returned, the
    # stalled ones included (see "Staging"); guarded by `cond`
    stalled_reads::Int                         # weight reads abandoned so far; guarded by `cond`
    stalled_read_bound::Int                    # with this many stalled reads outstanding, no read starts
end

function Scheduler(registry::ModelRegistry, backend::AbstractBackend, pool::MemoryPool, cfg::SchedulerConfig)
    return Scheduler(
        registry, backend, pool, cfg,
        Threads.Condition(), false, nothing, nothing, nothing, ControlCommand[],
        MetaGate(1), QueuedRequest[], 0, nothing, ReentrantLock(), Dict{String, Any}(),
        nothing, nothing, 0.0, DispatchProgress(), WeightRead[], 0, _stalled_read_bound()
    )
end

# A selected dispatch: the model entry, the chosen executable key (batch size; 0 = unbatched
# single module), and the requests coalesced into this execution (taken FIFO from the front).
struct Dispatch
    entry::ModelEntry
    size::Int
    taken::Vector{QueuedRequest}
end


# ---------------------------------------------------------------------------------------------
# Fair charging, EMA and cost-estimate updates
# ---------------------------------------------------------------------------------------------

# Charge `seconds` of dispatch-loop time to a model's fair-discipline virtual time, scaled by its
# weight: a model of weight 2 is charged half as much per second, so it gets twice the share.
_charge_fair!(st::ModelSchedState, seconds::Real) = (st.vtime += max(Float64(seconds), 0.0) / st.weight; nothing)

# Decay the recent-compute EMA to `now` by the configured half-life. Applied consistently
# before any read or write so the stored value is always current. The EMA is reported
# (`scheduler_metrics`); no discipline schedules by it.
function _decay_ema!(st::ModelSchedState, halflife::Float64, now::Float64)
    dt = now - st.ema_last_update
    if dt > 0
        st.recent_compute_ema *= exp(-dt * log(2) / halflife)
        st.ema_last_update = now
    end
    return st.recent_compute_ema
end

# Fold one measured run of a compiled program (shape variant, batch size) into its cost estimate.
#
# - The program's first run in this process is cold (lazy initialization, autotuning, caches) and is
#   not used: the second run seeds the estimate, at the smaller of the two times.
# - After that the estimate moves toward each measurement with weight
#   `w = 1 - (1 - alpha) * 2^(-age / _COST_HALFLIFE_SECONDS)`, where `age` is the time since the
#   estimate was last refreshed: `alpha` for back-to-back runs, approaching 1 for an estimate so old
#   that it says little about the program now. Weighting by elapsed time rather than by count keeps a
#   program measured once a minute from holding on to a value from many minutes ago.
# - A rise is capped at `_COST_MAX_RISE` of the estimate per measurement, however slow the run or old
#   the estimate, so one outlier (a GC pause, a cold cache, a co-tenant on the device) moves it by at
#   most that much. A fall is not capped: timing noise makes runs slower, not faster, so a fast run
#   is evidence of what the program costs, and the estimate never drops below the measurement.
#
# A model's estimates are written only on the dispatch thread, or at startup before it runs; a meta's
# own record is written under `s.cond`.
function _update_cost!(
        st::ModelSchedState, variant::VariantKey, B::Int, measured::Float64, alpha::Float64, now::Float64
    )
    isfinite(measured) || return nothing
    m = max(measured, 0.0)
    c = get!(CostEstimate, st.cost_estimate, (variant, B))
    c.runs += 1
    if c.runs == 1
        c.seconds = m                          # held, untrusted, until the second run
        return nothing
    elseif c.runs == 2
        c.seconds = min(c.seconds, m)
    else
        w = 1 - (1 - alpha) * 2.0^(-max(now - c.refreshed, 0.0) / _COST_HALFLIFE_SECONDS)
        c.seconds += min(w * (m - c.seconds), _COST_MAX_RISE * c.seconds)
    end
    c.refreshed = now
    return nothing
end

# The learned cost of one compiled program, or `nothing` until it has been measured warm.
function _cost_seconds(st::ModelSchedState, variant::VariantKey, B::Integer)
    c = get(st.cost_estimate, (variant, Int(B)), nothing)
    return (c === nothing || c.runs < 2) ? nothing : c.seconds
end

# Run one compiled program to measure its cost (warmup and the startup probe): twice for a program
# never run before, whose first run is cold and not used (see `_update_cost!`), once otherwise. A
# failed run propagates to the caller.
function _measure_program!(s::Scheduler, entry::ModelEntry, variant::VariantKey, sz::Int, inputs)
    st = entry.sched
    runs = (st !== nothing && !haskey(st.cost_estimate, (variant, sz))) ? 2 : 1
    for _ in 1:runs
        _, dt = _timed_run(s, entry.executable, inputs; offload = true)
        st === nothing || _update_cost!(st, variant, sz, dt, s.cfg.cost_ema_alpha, time())
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Batch contribution and coalescing plan
# ---------------------------------------------------------------------------------------------

# Rows present along the executable batch axis after preprocessing. Used to size output slices.
# Each executable input carries its OWN batch axis (per-input, batch-last), so read the count at
# each matched input's own `sp.batch_axis` rather than one shared position. The batch size comes
# from the first executable input that has a batch axis; a request whose batched inputs disagree on
# it is never coalesced or split (see `_batch_extents_agree`), so the count stands for every input
# of any request that shares a dispatch.
function _executable_rows(entry::ModelEntry, inputs::AbstractVector{NamedTensor})
    m = entry.manifest
    m.input_batch_dim === nothing && return 1
    for sp in m.executable_inputs
        sp.batch_axis === nothing && continue
        for t in inputs
            t.name == sp.name && return size(t.data, sp.batch_axis)
        end
    end
    return 1
end

# Whether a request's batched executable inputs all have the same extent, each read along its own
# batch axis. Only such a request may share a dispatch or be split. `_executable_rows` counts the
# first batched input alone, so a request whose inputs disagree (three rows of one input, two of
# another), merged with other requests, would pair its rows with a neighbor's while the merged
# shapes still match: the program runs and returns mixed rows to innocent callers. It dispatches
# alone instead and fails on its own. A missing batched input counts as a disagreement too, since
# merging it would fail every request in the dispatch. Validation cannot catch either: it sees the
# client's inputs, and preprocess may change an input's row count. Evaluated as the queue is
# planned rather than cached at enqueue; it reads a few sizes per request, beside the variant and
# row reads plan_batch already makes.
function _batch_extents_agree(entry::ModelEntry, inputs::AbstractVector{NamedTensor})
    rows = nothing
    for sp in entry.manifest.executable_inputs
        sp.batch_axis === nothing && continue
        k = findfirst(t -> t.name == sp.name, inputs)
        k === nothing && return false
        r = size(inputs[k].data, sp.batch_axis)
        rows === nothing && (rows = r)
        r == rows || return false
    end
    return true
end

# A model can coalesce multiple requests only when EVERY executable input carries a batch axis and
# every output does too (so inputs concatenate and outputs split per request). Otherwise it serves
# one request per dispatch. Single unbatched modules (batch key 0) are never coalesced. The strict
# all-or-nothing requirement mirrors the loader's manifest check, but is encoded here too so the
# runtime gate is self-contained (input_batch_dim !== nothing means at least one input is batched,
# not all, so it is not sufficient on its own).
function _coalescable(entry::ModelEntry)
    m = entry.manifest
    isempty(m.executable_inputs) && return false
    all(sp -> sp.batch_axis !== nothing, m.executable_inputs) || return false
    _has_unbatched(entry.executable) && return false
    return all(o -> o.batch_axis !== nothing, m.executable_outputs)
end

# The input-shape variant a queued request resolves to, read from its preprocessed (executable)
# inputs. Empty for single-shape models, so they all share the one default variant `Int[]`.
function _request_variant(entry::ModelEntry, prepared::Vector{NamedTensor})
    spec = entry.executable.sig.variant_spec
    isempty(spec) && return VariantKey()
    byname = Dict(t.name => t for t in prepared)
    return Int[size(byname[nm].data, ax) for (nm, ax) in spec]
end

_smallest_ge(sizes::Vector{Int}, n::Int) = (i = findfirst(>=(n), sizes); i === nothing ? nothing : sizes[i])

# Decide the dispatch batch for a model from its current queue (Step 3). Returns
# (chosen_size, taken) without mutating the queue, or nothing if the queue is empty. Coalescing
# uses only what is queued now (no look-ahead). Requests are taken FIFO; the remainder stays
# queued. For a partial fill the chosen compiled size exceeds the queued rows and the dispatch
# is padded. A per-model max_batch_size caps the rows coalesced into one dispatch (the compiled
# shape may still be larger and padded); a single request over the cap is never split, so it
# dispatches alone. For non-coalescable models exactly one request is taken at its own size, and so
# is it for every model when `coalesce` is false (`scheduler.coalescing`), so a request's rows always
# occupy the same slots of the same program whatever else is queued. A request whose batched inputs
# disagree on their batch extent is never coalesced either: it dispatches alone.
function plan_batch(entry::ModelEntry, st::ModelSchedState; coalesce::Bool = true)
    isempty(st.queue) && return nothing
    front = st.queue[1]
    # Size dispatches by the rows each request will occupy in the program, read from its preprocessed
    # inputs (preprocess runs before a request is queued). A hook can change the row count (a long text
    # split into several windows), and the client-facing count would then under- or overfill the batch.
    rows(qr) = _executable_rows(entry, qr.prepared)
    execs = entry.executable.execs

    # Pick the variant from the front request and only ever coalesce requests sharing it: different
    # input shapes cannot be concatenated along the batch axis. An uncompiled shape dispatches alone
    # so run_model raises the precise "no compiled program for input shape" error for that caller.
    variant = _request_variant(entry, front.prepared)
    haskey(execs, variant) || return (rows(front), QueuedRequest[front])
    inner = execs[variant]
    sizes = sort!(collect(keys(inner)))

    if !coalesce || !_coalescable(entry) || haskey(inner, 0)
        key = haskey(inner, 0) ? 0 :
            (
                _smallest_ge(sizes, rows(front)) === nothing ? maximum(sizes) :
                _smallest_ge(sizes, rows(front))
            )
        return (key, QueuedRequest[front])
    end

    # A front request whose batched inputs disagree on their batch extent never shares a dispatch
    # (see `_batch_extents_agree`). Planned at its own row count, it runs alone with its inputs
    # passed straight through, unpadded, so the execution fails for that caller only.
    _batch_extents_agree(entry, front.prepared) || return (rows(front), QueuedRequest[front])

    # The leading run of same-variant requests is the only window we may coalesce within (keeps
    # `taken` a contiguous front prefix so _finalize can deleteat! it, and preserves FIFO order).
    # A request whose batch extents disagree also ends it; it leads a later dispatch, alone.
    nprefix = 0
    for qr in st.queue
        _request_variant(entry, qr.prepared) == variant || break
        _batch_extents_agree(entry, qr.prepared) || break
        nprefix += 1
    end
    prefix = view(st.queue, 1:nprefix)

    cap = st.max_batch_size === nothing ? typemax(Int) : st.max_batch_size
    minB, maxB = minimum(sizes), maximum(sizes)
    R = min(sum(rows(qr) for qr in prefix), cap)
    # Largest compiled size that can be filled, else the smallest size for a partial fill.
    B = if R >= minB
        candidates = filter(<=(min(R, maxB)), sizes)
        isempty(candidates) ? minB : maximum(candidates)
    else
        minB
    end
    # A single request may itself exceed B; grow B to the smallest size that fits it.
    front_rows = rows(front)
    if B < front_rows
        grown = _smallest_ge(sizes, front_rows)
        B = grown === nothing ? maxB : grown
    end

    taken = QueuedRequest[]
    acc = 0
    for qr in prefix
        r = rows(qr)
        # B may exceed the cap on the partial-fill and front-oversize paths; bound by both.
        acc + r > min(B, cap) && break
        push!(taken, qr)
        acc += r
    end
    # Forward progress: an indivisible front request larger than B or the cap dispatches alone.
    isempty(taken) && push!(taken, front)
    return (B, taken)
end

# ---------------------------------------------------------------------------------------------
# Dispatch selection
# ---------------------------------------------------------------------------------------------

# A model is schedulable when it is prepared (sched + executable set) and has queued work. Metas are
# not scheduled here: they run on the request task under the meta gate (see `infer`), and their GPU
# sub-calls enter these queues as ordinary (committed) requests for the sub-models.
_schedulable(entry::ModelEntry) =
    entry.sched !== nothing && entry.executable !== nothing && !isempty(entry.sched.queue) &&
    !_awaiting_stage(entry)

# A model whose weights must be read from the weights file waits for a background task to read them
# (`:staging`), or for a free staging slot (`:pending`), instead of reading on the dispatch loop: a
# read there holds the device idle for as long as the disk takes, which no request on any model
# should pay for. Meanwhile the loop serves the other models. A model already on the device is never
# held back. See `_kick_staging!`.
_awaiting_stage(entry::ModelEntry) =
    (entry.sched.host_stage === :staging || entry.sched.host_stage === :pending) &&
    entry.executable.weights === nothing

# There is work when any model has a queued request. A committed meta sub-call is pushed onto its
# sub-model's queue, so it is covered by this scan (and `s.committed` is only ever set with that push).
_has_queued(s::Scheduler) = any(_schedulable, values(s.registry.by_name))

function _finalize(entry::ModelEntry, plan)
    B, taken = plan
    deleteat!(entry.sched.queue, 1:length(taken))     # taken are the front entries, in order
    return Dispatch(entry, B, taken)
end

# Select the next dispatch under the lock. An in-flight meta's committed continuation jumps the line:
# if one is pending, dispatch its sub-model now, ahead of the discipline scan. Otherwise dispatch on
# the configured discipline; each discipline coalesces the chosen model's queue via `plan_batch`.
# Returns a Dispatch or nothing if nothing is queued.
function select_dispatch!(s::Scheduler, now::Float64)
    # In-flight metas' committed sub-calls jump ahead of the discipline scan, one per in-flight meta (so
    # at most REACTANT_META_CONCURRENCY of them). Each was pushed to the front of its sub-model's queue,
    # so plan_batch draws it (leading the batch). Pop in arrival order; skip a stale entry whose request
    # is no longer queued (already coalesced/dispatched) or whose model is gone.
    i = 1
    while i <= length(s.committed)
        qr = s.committed[i]
        entry = get(s.registry.by_name, qr.req.model_name, nothing)
        # A sub-call whose model is still being read into host memory keeps its place in line.
        if entry !== nothing && entry.sched !== nothing && entry.executable !== nothing &&
                _awaiting_stage(entry) && any(x -> x === qr, entry.sched.queue)
            i += 1
            continue
        end
        deleteat!(s.committed, i)
        (entry === nothing || !_schedulable(entry)) && continue
        # `===` does not curry (it is a builtin, not a Fix2), so test identity with an explicit closure.
        any(x -> x === qr, entry.sched.queue) || continue
        return _finalize(entry, plan_batch(entry, entry.sched; coalesce = s.cfg.coalescing))
    end
    s.cfg.discipline == FIFO && return _select_fifo!(s)
    s.cfg.discipline == EDF && return _select_edf!(s)
    return _select_fair!(s)
end

# FIFO: serve the model whose oldest queued request is the oldest overall, then coalesce its queue.
# Ties break by model name for determinism. (An in-flight meta's committed sub-call bypasses this scan
# entirely; it is handled in `select_dispatch!` before any discipline runs.)
function _select_fifo!(s::Scheduler)
    chosen = nothing
    best_t = Inf
    for entry in values(s.registry.by_name)
        _schedulable(entry) || continue
        t = entry.sched.queue[1].enqueued_at
        if chosen === nothing || t < best_t || (t == best_t && entry.name < chosen.name)
            chosen, best_t = entry, t
        end
    end
    chosen === nothing && return nothing
    return _finalize(chosen, plan_batch(chosen, chosen.sched; coalesce = s.cfg.coalescing))
end

# The soonest deadline among a model's queued requests, or typemax (least urgent) when none carries
# a deadline. Requests with deadline_ns == 0 (no deadline) are treated as least urgent.
function _queue_min_deadline(st::ModelSchedState)
    best = typemax(Int64)
    for qr in st.queue
        dl = qr.req.deadline_ns
        dl != 0 && dl < best && (best = dl)
    end
    return best
end

# EDF: serve the model whose most-urgent queued request has the soonest deadline, then coalesce that
# model's queue (unchanged). Ties break by the model's oldest queued request and then by name, so when
# deadlines are equal (the common case: one deadline per client) this is exactly `_select_fifo!`. It
# diverges only to promote a model carrying a request with less budget left, so a meta's sub-call,
# which inherits the meta's deadline, is served ahead of fresher regular work. Queues stay
# arrival-ordered (never reordered), so `queue[1]` is always the model's oldest request and the FIFO
# tiebreak is exact. Coalescing still draws FIFO from the front, so a request deep behind the batch
# size advances over successive dispatches of its (preferentially selected) model rather than jumping
# the batch; that keeps coalescing semantics intact.
function _select_edf!(s::Scheduler)
    chosen = nothing
    best_dl = typemax(Int64)
    best_oldest = Inf
    for entry in values(s.registry.by_name)
        _schedulable(entry) || continue
        st = entry.sched
        dl = _queue_min_deadline(st)
        oldest = st.queue[1].enqueued_at
        if chosen === nothing || dl < best_dl ||
                (dl == best_dl && oldest < best_oldest) ||
                (dl == best_dl && oldest == best_oldest && entry.name < chosen.name)
            chosen, best_dl, best_oldest = entry, dl, oldest
        end
    end
    chosen === nothing && return nothing
    return _finalize(chosen, plan_batch(chosen, chosen.sched; coalesce = s.cfg.coalescing))
end

# Fair: start-time fair queueing over measured device seconds. Each dispatch charges its model the
# loop time it took over the model's weight (`_charge_fair!`), and the schedulable model with the
# least virtual time `vtime` goes next, so while models stay backlogged each one's device time
# tracks its weight share, whatever the models cost per dispatch. `s.vfloor` is the virtual time of
# the most recent fair dispatch, and a candidate below it is lifted to it before comparing: a model
# that was idle, newly loaded, or waiting for its weights to be read rejoins level with the busy
# ones, neither banking credit for the time it was away nor paying for capacity it used while it
# ran alone. Candidates are exactly the schedulable models with a plan, so the selector stays
# work-conserving: a model alone in the queue is served whatever its virtual time. Ties break by
# the oldest front request, then by name. (Metas are not scheduled here; they run under the meta
# gate and their sub-calls compete as ordinary requests for the sub-models.)
function _select_fair!(s::Scheduler)
    chosen = nothing
    chosen_plan = nothing
    for entry in values(s.registry.by_name)
        _schedulable(entry) || continue
        st = entry.sched
        plan = plan_batch(entry, st; coalesce = s.cfg.coalescing)
        plan === nothing && continue
        st.vtime = max(st.vtime, s.vfloor)
        if chosen === nothing || st.vtime < chosen.sched.vtime ||
                (st.vtime == chosen.sched.vtime && st.queue[1].enqueued_at < chosen.sched.queue[1].enqueued_at) ||
                (st.vtime == chosen.sched.vtime && st.queue[1].enqueued_at == chosen.sched.queue[1].enqueued_at && entry.name < chosen.name)
            chosen, chosen_plan = entry, plan
        end
    end
    chosen === nothing && return nothing
    s.vfloor = chosen.sched.vtime     # every candidate was lifted to the floor, so it never decreases
    return _finalize(chosen, chosen_plan)
end

# ---------------------------------------------------------------------------------------------
# Coalescing the tensors and splitting the outputs
# ---------------------------------------------------------------------------------------------

# Concatenate the per-request executable inputs along the batch axis into one set sized to B,
# padding with zeros when the real rows fall short of B. Inputs without a batch axis are taken
# from the first request (they do not vary across the batch). Every batched input must come to
# `total_rows`: plan_batch merges only requests whose inputs agree on their batch extent, and an
# input that did not would be padded out of step with the others, pairing one request's rows with
# another's, so it is refused here rather than run.
function _coalesce_inputs(entry::ModelEntry, pres::Vector{Vector{NamedTensor}}, total_rows::Int, B::Int)
    m = entry.manifest
    merged = NamedTensor[]
    for sp in m.executable_inputs
        arrays = Array[_named(pres[k], sp.name).data for k in eachindex(pres)]
        if sp.batch_axis === nothing
            push!(merged, NamedTensor(sp.name, arrays[1]))
            continue
        end
        axis = sp.batch_axis
        data = length(arrays) == 1 ? arrays[1] : cat(arrays...; dims = axis)
        size(data, axis) == total_rows ||
            error("input '$(sp.name)' has $(size(data, axis)) rows along its batch axis, expected $total_rows")
        if total_rows < B
            padshape = collect(size(data))
            padshape[axis] = B - total_rows
            data = cat(data, zeros(eltype(data), padshape...); dims = axis)
        end
        push!(merged, NamedTensor(sp.name, data))
    end
    return merged
end

function _named(ts::Vector{NamedTensor}, name::AbstractString)
    for t in ts
        t.name == name && return t
    end
    error("input '$name' not produced by preprocess")
end

# Slice each output to one request's row range, dropping padding. Outputs without a batch axis
# are passed through unchanged (only reached for non-coalescable single-request dispatches).
function _slice_outputs(entry::ModelEntry, out::Vector{NamedTensor}, offset::Int, rows::Int)
    specs = entry.manifest.executable_outputs
    sliced = NamedTensor[]
    for (i, t) in enumerate(out)
        sp = i <= length(specs) ? specs[i] : nothing
        if sp === nothing || sp.batch_axis === nothing
            push!(sliced, t)
        else
            idx = (offset + 1):(offset + rows)
            push!(sliced, NamedTensor(t.name, collect(selectdim(t.data, sp.batch_axis, idx))))
        end
    end
    return sliced
end

# Executions that run off the interactive thread. The dispatch loop runs on the interactive thread,
# which is also the thread Julia services its I/O on, so while an execution blocks it (the wait for a
# program's results is a gc_safe call that never yields) the worker reads no requests and answers no
# health checks. A long program therefore runs on a default-pool task the loop waits for
# (`off_interactive`); executions stay serialized exactly as before. The hand-off is not free: on a
# 128-core host it measured a 264 us median (347 us p99) with the default pool idle, the time to wake
# a sleeping thread, and with the pool saturated by request handlers a 14 us median but a 6.3 ms p99
# and a 152 ms worst case, waiting for a free thread. That is material next to a millisecond program,
# so only a program whose learned cost (for this shape variant and batch size) is at least
# `_OFFLOAD_EXEC_SECONDS`, or that has not been measured yet (it could be long), is handed off; a
# shorter one delays request intake by no more than its own run time.
const _OFFLOAD_EXEC_SECONDS = 0.05

_offload_execution(st::ModelSchedState, variant::VariantKey, B::Integer) =
    something(_cost_seconds(st, variant, B), Inf) >= _OFFLOAD_EXEC_SECONDS

# Run one execution, timed where it runs, so the measured compute excludes any task hand-off.
function _timed_run(s::Scheduler, model::LoadedModel, inputs; offload::Bool)
    run = function ()
        t0 = time()
        out = run_model(s.backend, s.pool, model, inputs)
        return (out, time() - t0)
    end
    return offload ? off_interactive(run) : run()
end

function _record_wait!(st::ModelSchedState, now::Float64, taken::Vector{QueuedRequest})
    for qr in taken
        push!(st.wait_samples, now - qr.enqueued_at)
    end
    while length(st.wait_samples) > _WAIT_SAMPLE_CAP
        popfirst!(st.wait_samples)
    end
    return nothing
end

# Run one selected dispatch. However it ends (run, failed before or after the upload, every request
# dropped at its deadline, withdrawn or cancelled, or an error that escapes it), it ends here, so this
# is where a staged model that no longer needs what it read gives back its staging slot (see
# `_release_idle_stage!`). A dispatch that uploaded has nothing staged left, and a model with requests
# still queued keeps its weights for the dispatch that will run them.
function execute_and_record!(s::Scheduler, d::Dispatch)
    try
        return _execute_dispatch!(s, d)
    finally
        _release_idle_stage!(s, d.entry)
    end
end

# Run the coalesced dispatch outside the lock, deliver per-request results, then take the lock
# briefly to record compute time once for the whole dispatch (never once per request).
function _execute_dispatch!(s::Scheduler, d::Dispatch)
    entry, B, taken = d.entry, d.size, d.taken
    st = entry.sched
    # Deadline admission: drop any request that cannot meet its deadline before we begin GPU work.
    # This is the only place a request is cancelled, and it never interrupts a running PJRT/GPU call —
    # it only refuses to START work that will not finish in time. The base check drops requests whose
    # deadline has already passed (all disciplines). Under EDF we add a laxity margin: a request is also
    # dropped if it cannot finish within this dispatch's learned compute cost, so the scheduler does not
    # burn GPU on work that will miss anyway (the classic EDF overload failure mode). Dropped requests
    # get a DeadlineExceeded reply (mapped to gRPC DEADLINE_EXCEEDED upstream); only feasible ones run,
    # and when none are feasible we skip run_model entirely. The cost estimate is learned from prior
    # dispatches of this program; until it has been measured warm the laxity drop is a no-op.
    # A request its caller has withdrawn (`_withdraw!`) is dropped here too, without a reply: nobody
    # waits for one.
    #
    # A drop leaves nothing measured, so an inflated estimate would shed every request whose budget it
    # exceeds for good. When the laxity margin alone would drop every request of the dispatch and the
    # estimate has not been refreshed for `_COST_STALE_SECONDS`, the oldest such request runs anyway
    # to measure the program again, at most once per that interval. A withdrawn request is never that
    # run.
    now_ns = Int64(time_ns())
    # Every taken request shares the front request's shape variant (plan_batch coalesces within one),
    # and the cost of the program that runs is that variant's at B.
    variant = _request_variant(entry, taken[1].prepared)
    est = s.cfg.discipline == EDF ? _cost_seconds(st, variant, B) : nothing
    laxity_ns = est === nothing ? Int64(0) : round(Int64, est * 1.0e9)
    live = QueuedRequest[]
    remeasure = nothing          # the oldest request only the laxity margin drops
    for qr in taken
        # A committed request is an in-flight meta's continuation: skip the laxity drop (we already
        # spent GPU on its earlier stages, so we do not abandon it on a prediction), but still honor
        # the base deadline-passed check, so a meta that has genuinely run out of time stops here.
        margin_ns = qr.committed ? Int64(0) : laxity_ns
        if _withdrawn(qr)
            continue          # its caller stopped waiting (dispatch timeout, cancellation): never run it
        elseif request_cancelled(qr.req)
            # Its client gave up its shared-memory slot (CORE-3). Checked before the deadline branch so
            # such a request is answered here, once, and is never picked as the re-measure run.
            put!(qr.reply, RequestCancelled(qr.req.model_name, "its client gave up the shared-memory slot"))
        elseif qr.req.deadline_ns != 0 && now_ns + margin_ns >= qr.req.deadline_ns
            if remeasure === nothing && now_ns < qr.req.deadline_ns
                remeasure = qr
            else
                put!(qr.reply, DeadlineExceeded(qr.req.model_name))
            end
        else
            push!(live, qr)
        end
    end
    # Withdrawal is set from the caller's task, so check it again before the re-measure decision: a
    # request withdrawn since the scan is neither run to measure the program nor answered. A slot
    # its client gave up on since the scan (CORE-3) is answered CANCELLED instead of being run.
    if remeasure !== nothing && !_withdrawn(remeasure)
        c = st.cost_estimate[(variant, B)]       # exists: only a measured estimate has a margin
        if request_cancelled(remeasure.req)
            put!(remeasure.reply, RequestCancelled(remeasure.req.model_name, "its client gave up the shared-memory slot"))
        elseif isempty(live) && time() - max(c.refreshed, c.remeasured) >= _COST_STALE_SECONDS
            c.remeasured = time()
            push!(live, remeasure)
            @debug "EDF: running a request its stale cost estimate would shed, to measure the program again" model = entry.name variant = variant size = B estimate = est
        else
            put!(remeasure.reply, DeadlineExceeded(remeasure.req.model_name))
        end
    end
    if length(live) != length(taken)
        taken = live          # the catch below and the slicing loop operate only on live requests
        isempty(taken) && return nothing    # `execute_and_record!` gives back a slot it staged
    end
    try
        # Inputs were already run through the model's preprocess hook on each caller's task before
        # the request was queued (see `infer`); the loop runs no model.jl code, only coalesce +
        # execute + slice, so user hooks never serialize against the GPU.
        pres = Vector{NamedTensor}[qr.prepared for qr in taken]
        rows = Int[_executable_rows(entry, p) for p in pres]
        total = sum(rows)
        # A lone request that already fills the dispatch needs neither concatenation nor
        # padding, so pass its inputs straight through (this also covers non-batched models
        # whose manifest does not enumerate executable_inputs).
        merged = (length(taken) == 1 && (!_coalescable(entry) || rows[1] == B)) ?
            pres[1] : _coalesce_inputs(entry, pres, total, B)

        # Ensure the model's weights are resident before running (loads on demand and evicts
        # LRU under budget pressure). Load time is excluded from compute_time, which feeds the
        # cost estimate, so the estimate reflects steady-state (resident) execution. The load
        # still held the dispatch loop, so the fair discipline charges it to the model.
        t_load = time()
        _acquire_for_dispatch!(s, entry)
        load_time = time() - t_load
        out, compute_time = _timed_run(
            s, entry.executable, merged; offload = _offload_execution(st, variant, B)
        )

        # Slice every per-request result before delivering any. Slicing can throw; doing it up
        # front means the catch below has not yet filled any reply channel, so it can deliver the
        # error exactly once per request without blocking on a full slot. Each result is the raw
        # device output for that request; the caller's task runs the postprocess hook on it (see
        # `infer`), off this loop.
        results = Vector{Any}(undef, length(taken))
        offset = 0
        for i in eachindex(taken)
            results[i] = _slice_outputs(entry, out, offset, rows[i])
            offset += rows[i]
        end

        lock(s.cond) do
            now = time()
            _decay_ema!(st, s.cfg.ema_halflife_seconds, now)
            st.recent_compute_ema += compute_time
            _charge_fair!(st, max(compute_time, 0.0) + max(load_time, 0.0))
            _update_cost!(st, variant, B, compute_time, s.cfg.cost_ema_alpha, now)
            st.dispatch_count += 1
            st.requests_served += length(taken)
            st.rows_served += total
            st.total_compute += compute_time
            st.batch_size_hist[B] = get(st.batch_size_hist, B, 0) + 1
            _record_wait!(st, now, taken)
        end

        # Deliver last: each reply is a fresh single-slot channel, so these puts never block.
        for (i, qr) in enumerate(taken)
            put!(qr.reply, results[i])
        end
    catch e
        for qr in taken
            put!(qr.reply, e)
        end
    end
    return nothing
end

# Run a meta model's orchestration on the calling (gRPC request) task, off the GPU dispatch loop. A
# per-worker gate admits a bounded number of GPU-using metas at a time, held across the whole run
# including the CPU glue between stages, so only that many metas' GPU sub-calls ever compete with
# regular work. The GPU itself is not held during the glue: each sub-call (via `QueueingCaller`)
# re-enters the scheduler as a committed request that dispatches on the loop, so while a meta computes
# between stages the loop serves other models. A meta whose deadline passes before it can take a permit
# is shed before any GPU work. A COMPUTE-ONLY meta (empty `calls`) issues no sub-calls and so contends
# for nothing the gate protects; it bypasses the gate entirely, so a heavy pure-Julia meta never blocks
# a GPU meta from a permit. The meta's recorded compute is the time spent INSIDE its sub-calls (its
# GPU/model-call cost), NOT its wall time: the data-dependent Julia glue between stages is excluded, so
# the control plane (lpt_packing) balances on the meta's real device work rather than its CPU glue. A
# compute-only meta therefore reports ~0 compute, which is correct — it consumes no GPU.
# `scratch_out` goes to `run_meta`: when given, the meta's scratch slots outlive this call (see `infer`).
function _run_meta_request(
        s::Scheduler, meta::MetaEntry, req::InferRequest;
        scratch_out::Union{Vector{PoolSlot}, Nothing} = nothing
    )
    # Base deadline check first: no point starting already-expired work (no GPU spent yet).
    req.deadline_ns != 0 && Int64(time_ns()) >= req.deadline_ns && throw(DeadlineExceeded(meta.name))
    gated = !isempty(meta.calls)   # compute-only metas (no sub-calls) skip the gate
    gated && acquire_meta_gate!(s.meta_gate, meta.name; deadline_ns = req.deadline_ns)
    caller = QueueingCaller(s, s.scratch_pool)
    call_ns = Ref(Int64(0))
    try
        out = run_meta(
            meta, caller, req.inputs;
            deadline_ns = req.deadline_ns, call_ns_out = call_ns, scratch_out = scratch_out
        )
        compute = call_ns[] / 1.0e9     # sub-call (model) time only; the Julia glue between stages is excluded
        st = meta.sched
        lock(s.cond) do
            now = time()
            _decay_ema!(st, s.cfg.ema_halflife_seconds, now)
            st.recent_compute_ema += compute
            _update_cost!(st, VariantKey(), 0, compute, s.cfg.cost_ema_alpha, now)
            st.dispatch_count += 1
            st.requests_served += 1
            st.rows_served += 1
            st.total_compute += compute
        end
        return out
    finally
        gated && release_meta_gate!(s.meta_gate)
    end
end

# ---------------------------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------------------------

# Initialize an entry's scheduling state from the config (or defaults). Works for a regular model
# (which also needs `executable` compiled) and for a meta (which has no executable but is dispatched).
function init_sched_state!(s::Scheduler, entry::AbstractDispatchEntry, now::Float64)
    mc = get(s.cfg.models, entry.name, ModelSchedConfig(1.0))
    entry.sched = ModelSchedState(entry.name, mc, now)
    return entry
end

# Measure the worst-case execution scratch (transient device memory beyond resident weights) by
# probing each model in ISOLATION: free all non-pinned device weights so only pinned + the probed
# model are resident, run every compiled (variant,size), and read the allocator's peak. The
# per-model run also seeds the cost estimates. Returns max scratch in bytes (0 when
# the backend reports no device stats, e.g. CPU/mock).
#
# Two measurement modes, chosen by whether the backend can reset the allocator's high-water mark
# (`clear_memory_stats!`):
#
# * Resettable (the PJRT C API backend, and the Reactant backend on a CUDA client): before each model
#   the peak is reset, so `peak - in_use_before` is exactly that model's transient, whatever ran
#   earlier. Compile-time autotuning scratch, which can be many times a model's real run scratch and
#   used to force an operator restart before a clean measurement, cannot leak in, and pinned models
#   are measured too.
# * Monotone peak (fallback): the peak only grows, so a naive `peak - (pinned + this_model.nbytes)`
#   leaks the heaviest model's weights into every later model's "scratch" and depends on iteration
#   order. Probing in ascending weight order and subtracting the largest weight set resident so far
#   makes `max_i (peak_i - pinned - maxweight_i)` provably exactly `max_i scratch_i` (peak_i >=
#   pinned + W_i + S_i gives the >= bound, W_j <= W_i for j <= i the <= bound), but anything that
#   moved the peak before the probe (autotuning) still inflates it.
function _probe_max_scratch!(s::Scheduler, pinned_bytes::Int)
    (s.weight_cache === nothing || device_memory_stats(s.backend, s.pool) === nothing) && return 0
    resettable = clear_memory_stats!(s.backend, s.pool)
    resettable || @info "memory probe: allocator peak is not resettable on this backend; using the ordering-based estimate (pinned models unmeasured)"
    probe = [
        e for e in values(s.registry.by_name)
            if e.executable !== nothing && !isempty(e.manifest.executable_inputs) &&
            (resettable || !is_device_pinned(e.executable))
    ]          # need synthesizable inputs to measure
    sort!(probe; by = e -> e.executable.nbytes)                  # ascending weight
    max_scratch = 0
    max_weight = 0
    for entry in probe
        model = entry.executable
        if !is_device_pinned(model)
            _free_nonpinned!(s.weight_cache)                     # isolate: only pinned + M will be resident
            try
                probe_acquire!(s.weight_cache, entry)
            catch err
                err isa NotResidentError && continue             # externally-managed: cannot load to measure
                rethrow()
            end
        end
        max_weight = max(max_weight, model.nbytes)               # == model.nbytes given ascending order
        scratch = _probe_entry_scratch!(s, entry, resettable, pinned_bytes + max_weight)
        max_scratch = max(max_scratch, scratch)
        # Per-model S for admission (exact with a resettable peak, an upper bound otherwise, since
        # the monotone peak can carry an earlier model's transient).
        model_footprint(s.weight_cache, entry).scratch_bytes = scratch
    end
    return max(0, max_scratch)
end

# Run every compiled (variant, size) of one resident model, seeding its cost estimates, and
# return its scratch estimate in bytes. Resettable: the peak is reset first and the result is
# `peak - in_use_before`, exact for this model. Otherwise `peak - resident_floor`, the monotone-peak
# formula (`resident_floor` = pinned + the heaviest weight set resident so far). When the model
# cannot be run (no synthesizable inputs, or the run failed) the compiler's static `temp + outputs`
# accounting, padded by half (it undershoots the allocator's view by 7 to 50 percent in
# measurements), stands in so the model still reserves its declared scratch.
function _probe_entry_scratch!(s::Scheduler, entry::ModelEntry, resettable::Bool, resident_floor::Int)
    model = entry.executable
    baseline = 0
    if resettable
        clear_memory_stats!(s.backend, s.pool)
        dm0 = device_memory_stats(s.backend, s.pool)
        baseline = dm0 === nothing ? 0 : dm0.in_use
    end
    static = 0
    ran = false
    for (variant, inner) in model.execs
        for (sz, exec) in inner
            cms = compiled_memory_stats(s.backend, exec)
            cms === nothing || (static = max(static, Int(cms.temp) + Int(cms.outputs)))
            inputs = _zero_inputs(entry, variant, sz)
            inputs === nothing && continue
            try
                _measure_program!(s, entry, variant, sz, inputs)
                ran = true
            catch err
                @warn "memory probe / warmup run failed; the program stays unmeasured until it dispatches" model = entry.name variant = variant size = sz exception = err
            end
        end
    end
    ran || return ceil(Int, 1.5 * static)
    dm = device_memory_stats(s.backend, s.pool)
    dm === nothing && return ceil(Int, 1.5 * static)
    measured = resettable ? dm.peak_in_use - baseline : dm.peak_in_use - resident_floor
    return max(0, measured)
end

# Size the on-demand cache from the unified memory model: pinned reserve their footprint, the
# isolation probe measures worst-case scratch, and the cache gets whatever the arena has left after
# pinned + scratch + the wiggle slack (`weight_budget`). Runs on the startup thread before the
# dispatch loop spawns (so mutating `max_bytes` does not race the loop).
function _autosize_weight_cache!(s::Scheduler, arena::Int, fraction::Float64, wiggle::Float64)
    s.weight_cache === nothing && return nothing
    s.autosize = (arena = arena, fraction = fraction, wiggle = wiggle)
    pinned = pinned_weight_bytes(s.registry)
    max_scratch = _probe_max_scratch!(s, pinned)
    b = weight_budget(;
        arena = arena, fraction = fraction, wiggle = wiggle,
        max_scratch = max_scratch, pinned_bytes = pinned
    )
    lock(s.weight_cache.lock) do
        s.weight_cache.max_bytes = b.on_demand_budget
        s.weight_cache.pinned_bytes = pinned
        s.weight_cache.max_scratch = max_scratch
        s.weight_cache.weight_pool = b.weight_pool
    end
    has_ondemand = any(
        e -> e.executable !== nothing && !is_device_pinned(e.executable),
        values(s.registry.by_name)
    )
    if b.pinned_over_commit
        @warn "weight budget: pinned weights plus execution scratch exceed the arena minus wiggle; pinned models may not fit" arena = arena pinned_bytes = pinned max_scratch = max_scratch weight_pool = b.weight_pool wiggle = wiggle
    elseif b.on_demand_budget == 0 && has_ondemand
        @warn "weight budget: no room for the on-demand cache after pinned weights + scratch + wiggle; non-pinned models reload on every dispatch" arena = arena pinned_bytes = pinned max_scratch = max_scratch
    else
        @info "weight budget resolved" arena = arena fraction = fraction wiggle = wiggle pinned_bytes = pinned max_scratch = max_scratch weight_pool = b.weight_pool on_demand_budget = b.on_demand_budget
    end
    return nothing
end

# Prepare every registered entry's scheduling state, then spawn the dispatch loop. Entries are
# already in the registry (loaded and compiled by `serve`). When `autosize_arena > 0` (set by the
# worker when the on-demand cache is enabled), the isolation probe runs every model regardless of
# discipline, doubling as the cost warmup and the scratch measurement, and the cache is sized
# before serving begins.
function start!(
        s::Scheduler; autosize_arena::Integer = 0, autosize_fraction::Real = 0.0,
        autosize_wiggle::Real = 0.0
    )
    now = time()
    for entry in values(s.registry.by_name)
        init_sched_state!(s, entry, now)
    end
    for entry in values(s.registry.meta)   # metas are dispatched too, so they get sched state as well
        init_sched_state!(s, entry, now)
    end
    s.weight_cache === nothing || preload_pinned!(s.weight_cache, s.registry)
    if autosize_arena > 0 && s.weight_cache !== nothing
        # The probe runs every model (seeding the cost estimates too) and sizes the cache.
        _autosize_weight_cache!(s, Int(autosize_arena), Float64(autosize_fraction), Float64(autosize_wiggle))
        _fill_device!(s)
    elseif s.cfg.discipline == FAIR
        # No auto-size (cache off, or no device stats): just seed the cost estimates.
        for entry in values(s.registry.by_name)
            _warmup_entry!(s, entry)
        end
    end
    s.weight_cache === nothing || (s.weight_cache.serving = true)
    s.running = true
    # Run the dispatch loop on the interactive threadpool when one exists (the worker is started
    # with `--threads=auto,1`), so the GPU dispatch is scheduled promptly and is never starved by
    # the per-request preprocess/postprocess tasks that saturate the default pool. With no
    # interactive thread the default pool is used; correctness is unaffected, only the overlap.
    if Threads.nthreads(:interactive) > 0
        s.task = Threads.@spawn :interactive dispatch_loop(s)
    else
        s.task = Threads.@spawn dispatch_loop(s)
    end
    return s
end

# After the startup probe (which leaves only the last model it measured on the device), load the
# on-demand models that fit the device budget, largest first, the order that packs without gaps. A
# deployment whose models all fit then starts with every one resident and never reads a weights file
# after startup, whatever the host weight budget. Self-managed only: under externally managed
# residency the control plane decides what is resident. Returns the number of models loaded.
function _fill_device!(s::Scheduler)
    wc = s.weight_cache
    (wc === nothing || wc.mode == EXTERNALLY_MANAGED) && return 0
    names = String[
        e.name for e in values(s.registry.by_name)
            if e.executable !== nothing && !is_device_pinned(e.executable) && e.executable.weights === nothing
    ]
    loaded = 0
    for name in _largest_first(wc, s.registry, names)
        entry = s.registry.by_name[name]
        wc.resident_bytes + entry.executable.nbytes <= wc.max_bytes || continue
        acquire!(wc, entry)
        loaded += 1
    end
    loaded > 0 && @info "startup: on-demand models loaded onto the device" loaded = loaded of = length(names) resident_bytes = wc.resident_bytes on_demand_budget = wc.max_bytes
    return loaded
end

function shutdown!(s::Scheduler)
    lock(s.cond) do
        s.running = false
        notify(s.cond)
    end
    return nothing
end

# Reject everything still pending at shutdown so no caller stays blocked on its reply channel:
# queued requests across all models and any pending control commands, with `err` (UNAVAILABLE at
# the transport). Every caller is answered before any held weights are released, since a release
# does shared-store I/O that can fail. Caller holds `s.cond`.
function _reject_pending_locked!(s::Scheduler, err::Exception = _unavailable(s))
    for entry in values(s.registry.by_name)
        entry.sched === nothing && continue
        for qr in entry.sched.queue
            put!(qr.reply, err)
        end
        empty!(entry.sched.queue)
    end
    for entry in values(s.registry.meta)
        entry.sched === nothing && continue
        for qr in entry.sched.queue
            put!(qr.reply, err)
        end
        empty!(entry.sched.queue)
    end
    empty!(s.committed)   # references into the (now drained) model queues; clear the stale priority list
    for c in s.control
        put!(c.reply, err)
    end
    empty!(s.control)
    for (name, pf) in s.prefetched
        entry = get(s.registry.by_name, name, nothing)
        (pf.staged === nothing || entry === nothing || s.weight_cache === nothing) ||
            release_staged!(s.weight_cache, entry, pf.staged; uploaded = false)
    end
    empty!(s.prefetched)
    for entry in values(s.registry.by_name)
        st = entry.sched
        if st !== nothing && st.host_stage === :staged && s.weight_cache !== nothing
            release_staged!(s.weight_cache, entry, st.staged; uploaded = false)
            st.staged = nothing
            st.host_stage = :none
        end
    end
    return nothing
end

# The rejection a request or control command meets once the dispatch loop no longer runs: the
# worker is shutting down, or the loop died.
_unavailable(s::Scheduler) = WorkerUnavailableError(
    s.failure === nothing ? "server is shutting down" :
        "the worker's dispatch loop failed, so this worker serves no more requests until it restarts"
)

"""
    dispatch_alive(scheduler) -> Bool

Whether the dispatch loop can still serve: `false` once it has died with an unexpected error
(the worker then reports not live and not ready, and rejects every request with `UNAVAILABLE`),
and `false` while it is stalled (`dispatch_stalled`), which ServerLive, ServerReady,
ModelReady, `/healthz` and `/readyz` all report until it makes progress again.
"""
function dispatch_alive(s::Scheduler)
    s.failure === nothing || return false
    t = s.task
    # A loop task that ended while the scheduler still runs died some way the guard did not see.
    s.running && t !== nothing && istaskdone(t) && return false
    return !dispatch_stalled(s)
end

"""
    dispatch_stalled(scheduler; now=time()) -> Bool

Whether the dispatch loop holds work (a dispatch, a control command, or more requests queued
behind the one it finished) and has made no progress for longer than its stall limit
(`DISPATCH_STALL_SECONDS` by default): stuck in a GPU call that never returns, say, while
the threads that answer health checks are still free. An idle loop is never stalled, however long
it waits. Lock-free, so a loop stuck while holding the scheduler lock cannot block it.
"""
function dispatch_stalled(s::Scheduler; now::Real = time())
    p = s.progress
    p.stall_seconds > 0 || return false
    t = p.since[]
    (t > 0 && now - t > p.stall_seconds) || return false
    Threads.atomic_cas!(p.reported, false, true) ||
        @error "the dispatch loop has made no progress for $(round(now - t; digits = 1))s with work in hand; reporting this worker not live and not ready until it does" stall_seconds = p.stall_seconds
    return true
end

# Build zero-filled executable inputs at batch size `sz` (0 = unbatched) and input-shape `variant`
# for warmup. The variant supplies the variable-axis sizes in the same (input, axis) order they were
# collected in (see ModelSignature.variant_spec); returns nothing if it runs short.
function _zero_inputs(entry::ModelEntry, variant::VariantKey, sz::Int)
    inputs = NamedTensor[]
    vi = 0
    for sp in entry.manifest.executable_inputs
        dims = Int[]
        for d in sp.shape
            if d.kind == BATCH
                push!(dims, sz == 0 ? 1 : sz)
            elseif d.kind == FIXED
                push!(dims, d.size)
            else                          # variable axis: take the next size from the variant
                vi += 1
                vi <= length(variant) || return nothing
                push!(dims, variant[vi])
            end
        end
        push!(inputs, NamedTensor(sp.name, zeros(julia_type(sp.dtype), dims...)))
    end
    return inputs
end

# Seed one entry's cost_estimate by running each compiled (variant, size) twice, the first run being
# cold (see `_measure_program!`). Failures are non-fatal: a program left unmeasured is measured by its
# first real dispatches, and until then the EDF laxity check does not shed on it.
function _warmup_entry!(s::Scheduler, entry::ModelEntry)
    (entry.executable === nothing || entry.sched === nothing) && return nothing
    isempty(entry.manifest.executable_inputs) && return nothing   # cannot synthesize inputs
    for (variant, inner) in entry.executable.execs
        for sz in keys(inner)
            inputs = _zero_inputs(entry, variant, sz)
            inputs === nothing && continue
            try
                s.weight_cache === nothing || acquire!(s.weight_cache, entry)
                _measure_program!(s, entry, variant, sz, inputs)
            catch err
                @warn "scheduler cost warmup failed; the program stays unmeasured until it dispatches" model = entry.name variant = variant size = sz exception = err
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Staging: reading weights into host memory ahead of the dispatch
# ---------------------------------------------------------------------------------------------
#
# Loading a model off the device has a host half (the shared store's lock waits, reserving its
# region, reading the weights file) and a device half (eviction and upload). Run on the dispatch
# loop, the host half holds the GPU idle for as long as a disk read takes. Instead, a model with
# queued requests and no weights on the device gets a background task that does the host half
# (`stage_host_weights`); while it runs the model is not selectable (`_awaiting_stage`) and the loop
# serves everything else. When the read lands the model becomes selectable again, and its dispatch
# evicts and uploads in order, as any load does, so a staged model never takes device memory ahead
# of its turn. Staging is an optimization only: a model that was never staged, or whose read failed,
# loads on the dispatch as before.
#
# The read is host memory only, so at most `_MAX_HOST_STAGES` models per worker are read ahead
# (counting those read and waiting for their dispatch), oldest queued request first. Each one
# waiting holds its shared-store region against eviction until it is uploaded. A model past the cap
# is `:pending`: it waits for a slot rather than reading on the dispatch loop. Its requests wait for
# the read either way; on the loop they would also hold up every other model's.
#
# A model is `:staged` only while it has requests queued or its dispatch is running, so the slot it
# holds always comes back. A read that lands for a model with nothing queued is dropped (`_run_stage!`).
# Every dispatch ends in `_release_idle_stage!` (see `execute_and_record!`): one that uploaded has
# nothing staged left, and one whose requests all failed before the upload, or were dropped at their
# deadline, withdrawn or cancelled, releases the read once nothing more is queued. Unloading the model
# (`_evict_entry!`) or stopping the worker (`_reject_pending_locked!`) releases it as well. A slot
# kept past that would be kept for good, and with every slot gone each cold model would wait
# `:pending` forever.
#
# This runs after every pass of the dispatch loop and on every submission, so whatever left a queued
# model off the device (an eviction, a compaction, a residency change) has marked it before the next
# selection.
#
# A read that stops making progress is abandoned. Storage that stops answering (a hard NFS mount, a
# shared-store lock held by a stuck worker) leaves a read in a system call that never returns and
# cannot be interrupted, holding its slot: two of them would leave every other cold model `:pending`
# for good while the worker still reports ready. So a read counts the bytes it reads as it goes
# (`ReadProgress`), and one that goes `scheduler.weight_read_stall_seconds` without any is abandoned
# (`_check_stall!`): its model's queued requests fail with `UNAVAILABLE` (`WeightsReadStalledError`,
# which a gateway retries on another replica), its slot passes to the next model, and nothing waits
# for it any more, the dispatch loop included. It stays in `s.reads` until it returns, if it ever does,
# and then what it read is released and used by nothing. Until then its model is not read again (a
# second read would hang the same way and hold another thread), so requests for it fail at once. Each
# stalled read holds a thread, so with `stalled_read_bound` of them outstanding the worker starts no
# read of storage at all and every request for a model off the device fails at once, except for a
# model with a copy in this worker's host cache, whose read takes it from RAM. Models on the device
# serve throughout.

const _MAX_HOST_STAGES = 2

# Why no read of `e`'s weights may start now, or nothing: an earlier read of it stalled and has not
# returned, or so many reads have that none starts. At that bound a model whose weights are in a
# private host copy is still read, since its read touches no storage. Caller holds `s.cond`.
function _stall_refusal(s::Scheduler, e::ModelEntry)
    name = e.name
    n = 0
    own = false
    for r in s.reads
        r.stalled || continue
        n += 1
        r.entry.name == name && (own = true)
    end
    own && return "model '$name' is not loaded on this worker now: an earlier read of its weights " *
        "stalled and has not returned, and it is not read again until it does (the weights storage " *
        "may not be answering)"
    (n < s.stalled_read_bound || _private_host_copy(s, e)) && return nothing
    stalled = n == 1 ? "a read of model weights stalled and has" : "$n reads of model weights stalled and have"
    return "model '$name' is not loaded on this worker now: $stalled not returned, so no new read of " *
        "storage starts until one does (the weights storage may not be answering)"
end

# Whether `e`'s weights are in this worker's private host cache, so reading them takes them from
# RAM. A region in the shared store does not count: taking it waits on a lock another worker may hold.
function _private_host_copy(s::Scheduler, e::ModelEntry)
    wc = s.weight_cache
    return wc !== nothing && !(wc.store isa SharedWeightStore) && has_host_copy(wc, e)
end

# Answer every request queued for `e` with `err`. Caller holds `s.cond`.
function _fail_queued!(e::ModelEntry, err)
    for qr in e.sched.queue
        put!(qr.reply, err)
    end
    empty!(e.sched.queue)
    return nothing
end

# Start background reads for models that need one, up to the cap, and mark the rest `:pending`. A
# model that may not be read now (`_stall_refusal`) has its requests failed instead. Caller holds
# `s.cond`.
function _kick_staging!(s::Scheduler)
    wc = s.weight_cache
    (wc === nothing || !s.running) && return nothing
    busy = 0
    cands = ModelEntry[]
    for e in values(s.registry.by_name)
        st = e.sched
        (st === nothing || e.executable === nothing) && continue
        if st.host_stage === :staging || st.host_stage === :staged
            busy += 1
        elseif st.host_stage === :none || st.host_stage === :pending
            if !isempty(st.queue) && needs_host_stage(wc, e)
                push!(cands, e)
            else
                st.host_stage = :none             # loaded meanwhile, or nothing left to read for
            end
        end
    end
    sort!(cands; by = e -> e.sched.queue[1].enqueued_at)
    now = time()
    for e in cands
        # Weights read ahead of a placement change (`stage_weights!`) need no second read.
        pf = get(s.prefetched, e.name, nothing)
        if pf !== nothing && pf.staged !== nothing && pf.expires > now
            delete!(s.prefetched, e.name)
            e.sched.staged = pf.staged
            e.sched.host_stage = :staged
            continue
        elseif pf !== nothing && pf.staged === nothing
            e.sched.host_stage = :pending         # its read is under way and kicks again when it lands
            continue
        end
        why = _stall_refusal(s, e)
        if why !== nothing
            _fail_queued!(e, WeightsReadStalledError(e.name, why))
            e.sched.host_stage = :none
            continue
        end
        if busy < _MAX_HOST_STAGES
            e.sched.host_stage = :staging
            e.sched.stage_task = _start_read!(r -> _run_stage!(s, r), s, e; kind = :stage).task
            busy += 1
        else
            e.sched.host_stage = :pending
        end
    end
    return nothing
end

# Start a background read of `e`'s weights for `kind` (see `WeightRead`): `body(r)` on a
# default-pool task, with a timer that checks it for progress when `weight_read_stall_seconds` is on.
# `prefetch` is the StageWeights hold it fills. Caller holds `s.cond`, which `body` takes to record the
# result, so `r.task` is set first.
function _start_read!(body, s::Scheduler, e::ModelEntry; kind::Symbol, prefetch = nothing)
    r = WeightRead(e, kind, prefetch, ReadProgress(), nothing, nothing, time(), false, false, false, nothing, Base.Event())
    push!(s.reads, r)
    limit = s.cfg.weight_read_stall_seconds
    if limit > 0
        tick = clamp(limit / 8, 0.01, 5.0)
        r.timer = Timer(_ -> _check_stall!(s, r), tick; interval = tick, spawn = true)
    end
    r.task = Threads.@spawn :default body(r)
    return r
end

# Read `r`'s model into host memory, counting the bytes as progress: the StagedWeights, or nothing
# and the failure with its backtrace.
function _watched_read(wc::WeightCache, r::WeightRead)
    try
        return with_read_progress(() -> stage_host_weights(wc, r.entry), r.progress), nothing
    catch err
        return nothing, (err, catch_backtrace())
    end
end

# A read's progress check, run by its timer: abandon it once it has gone the stall limit without
# reading a byte.
function _check_stall!(s::Scheduler, r::WeightRead)
    limit = s.cfg.weight_read_stall_seconds
    report = lock(s.cond) do
        (r.ended || r.stalled) && return nothing
        idle = max(Int64(time_ns()) - (@atomic r.progress.last_ns), 0) / 1.0e9
        idle < limit && return nothing
        _abandon_read!(s, r, idle)
        return (idle = idle, outstanding = count(x -> x.stalled, s.reads))
    end
    report === nothing && return nothing
    @warn "a weights read made no progress for the stall limit and was abandoned; the model is not read again on this worker until it returns" model = r.entry.name purpose = _READ_PURPOSE[r.kind] seconds_without_progress = round(report.idle; digits = 1) bytes_read = (@atomic r.progress.bytes) stalled_reads = report.outstanding no_reads_at = s.stalled_read_bound
    return nothing
end

const _READ_PURPOSE = Dict(:stage => "queued requests", :prefetch => "StageWeights", :residency => "a residency change")

# Abandon a read that stalled: fail the requests waiting for it, or answer the StageWeights call or
# the residency change waiting for it, and give its slot to the next model. The read itself goes on
# until it returns. Caller holds `s.cond`.
function _abandon_read!(s::Scheduler, r::WeightRead, idle::Float64)
    r.stalled = true
    s.stalled_reads += 1
    r.timer === nothing || close(r.timer)
    e = r.entry
    msg = "reading the weights of model '$(e.name)' stalled: no progress for $(round(idle; digits = 1)) s, " *
        "so the read was abandoned (the weights storage may not be answering); this worker does not read " *
        "the model again until that read returns"
    if r.kind === :stage
        st = e.sched
        if st !== nothing && st.host_stage === :staging && st.stage_task === r.task
            _fail_queued!(e, WeightsReadStalledError(e.name, msg))
            st.host_stage = :none
            st.stage_task = nothing
        end
    elseif r.kind === :prefetch
        # The hold goes now: left with nothing in it, it would keep the model `:pending`.
        pf = r.prefetch
        get(s.prefetched, e.name, nothing) === pf && delete!(s.prefetched, e.name)
        r.outcome = (state = "failed", read = false, seconds = time() - r.started, error = msg)
        notify(r.done)
    else
        r.outcome = WeightsReadStalledError(e.name, msg)
        notify(r.done)
    end
    _kick_staging!(s)
    notify(s.cond)
    return nothing
end

# The bookkeeping of a read that returned, stalled or not: it is no longer outstanding. Returns
# whether its model was unloaded meanwhile, so the caller closes the weights file once it has
# released what the read holds. Caller holds `s.cond`.
function _read_ended!(s::Scheduler, r::WeightRead)
    r.ended = true
    r.timer === nothing || close(r.timer)
    i = findfirst(x -> x === r, s.reads)
    i === nothing || deleteat!(s.reads, i)
    return r.close_weights
end

# How to let go of the StagedWeights `drop` a read returned with, or nothing to let go of.
_releaser(wc::WeightCache, r::WeightRead, drop) =
    drop === nothing ? nothing : () -> release_staged!(wc, r.entry, drop; uploaded = false)

# Release what a read returned with (`release()`, when it is not nothing), then close its model's
# weights file if the model was unloaded while it ran; log a stalled read's return.
function _after_read!(r::WeightRead, release, close_file::Bool, stalled::Bool, failure)
    try
        release === nothing || release()
    finally
        close_file && close_weights!(r.entry.weights)
    end
    stalled && @info "a stalled weights read returned; what it read was released, unused" model = r.entry.name seconds_since_start = round(time() - r.started; digits = 1) bytes_read = (@atomic r.progress.bytes) outcome = (failure === nothing ? "read" : sprint(showerror, failure[1]))
    return nothing
end

# The background read for one model's queued requests. On success the model waits, staged, for its
# dispatch; if it was unloaded, loaded some other way, or the read failed, the read is dropped and
# the dispatch loads the model itself. It is dropped too when nothing is queued for the model any
# more, as happens when something else loaded the model while the read ran (a compaction told to
# reload it) and the model then served its requests and was evicted again: held for a dispatch that
# never comes, the read would keep its slot for good. A read abandoned as stalled is dropped
# whenever it returns. Either way the loop is woken to reconsider the model.
function _run_stage!(s::Scheduler, r::WeightRead)
    entry = r.entry
    wc = s.weight_cache::WeightCache
    staged, failure = _watched_read(wc, r)
    drop = nothing
    close_file = false
    stalled = false
    lock(s.cond) do
        close_file = _read_ended!(s, r)
        stalled = r.stalled
        st = entry.sched
        live = s.running && get(s.registry.by_name, entry.name, nothing) === entry
        if stalled
            drop = staged                         # its requests were answered when it was abandoned
        elseif staged !== nothing && live && st.host_stage === :staging && entry.executable.weights === nothing &&
                !isempty(st.queue)
            st.staged = staged
            st.host_stage = :staged
            st.stage_task = nothing
        else
            drop = staged
            # A failed read is not repeated on the dispatch loop: the requests waiting for it get the
            # error, as they would have from a load that failed there, and the next request for the
            # model starts a fresh read.
            if failure !== nothing && st.host_stage === :staging
                _fail_queued!(entry, ErrorException("reading the weights of model '$(entry.name)' failed: $(sprint(showerror, failure[1]))"))
            end
            st.host_stage === :staging && (st.host_stage = :none)
            st.stage_task = nothing
        end
        _kick_staging!(s)
        notify(s.cond)
    end
    failure === nothing || stalled ||
        @warn "reading weights ahead of the dispatch failed; failing the model's queued requests" model = entry.name exception = failure
    _after_read!(r, _releaser(wc, r, drop), close_file, stalled, failure)
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Staging ahead of a placement change (the StageWeights RPC)
# ---------------------------------------------------------------------------------------------
#
# A gateway that moves a model onto this worker first asks for its weights to be read into host
# memory, and switches routing only once they are, so the model's first request here waits for an
# upload from RAM and never for a read from disk. The read runs on a default-pool task, never on the
# dispatch loop. What it read is held (a shared-store region is kept from eviction) until the
# model's first dispatch takes it, its hold expires, or the model is unloaded. The held bytes are
# capped so a large plan cannot crowd out the host cache it relies on.

const _PREFETCH_HOLD_MAX_SECONDS = 600.0
const _PREFETCH_UNBUDGETED_CAP = 4 * 2^30     # held bytes with no host budget to size from

mutable struct Prefetched
    staged::Any                       # StagedWeights once read, nothing while reading
    task::Union{Task, Nothing}        # the read while it runs
    expires::Float64
    nbytes::Int
end

# The most host memory held for reads ahead of placement changes: half the host budget, or a fixed
# cap without one.
function _prefetch_cap(wc::WeightCache)
    b = host_budget(wc)
    return b === nothing || b <= 0 ? _PREFETCH_UNBUDGETED_CAP : b ÷ 2
end

# Release holds that expired. Takes `s.cond` briefly; the releases run outside it.
function _sweep_prefetched!(s::Scheduler; now::Float64 = time())
    wc = s.weight_cache
    expired = lock(s.cond) do
        out = Tuple{String, Any}[]
        for (name, pf) in s.prefetched
            (pf.staged !== nothing && pf.expires <= now) && push!(out, (name, pf.staged))
        end
        for (name, _) in out
            delete!(s.prefetched, name)
        end
        return out
    end
    for (name, staged) in expired
        entry = get(s.registry.by_name, name, nothing)
        (entry === nothing || wc === nothing) || release_staged!(wc, entry, staged; uploaded = false)
    end
    return nothing
end

"""
    stage_weights!(scheduler, names; hold_seconds=60) -> Vector{NamedTuple}

Read the weights of each model in `names` (each sub-model of a meta) into host memory ahead of a
placement change, and hold them for `hold_seconds` (at most $(_PREFETCH_HOLD_MAX_SECONDS) s) or
until the model's first dispatch uploads them. Runs the reads on default-pool tasks, never on the
dispatch loop, and returns once they end, or once a read that stopped making progress is abandoned
(`scheduler.weight_read_stall_seconds`). One row per requested name (a meta's combines its
sub-models', taking the worst state):

- `state`: `"resident"` (on the device, or with a host floor: nothing to read), `"staged"` (held
  in host memory now), `"deferred"` (the held bytes would pass the cap; ask again later),
  `"skipped"` (a CPU or externally-managed worker, where the gateway does not place weights), or
  `"failed"` (the read failed or stalled, reads of model weights on this worker have stalled, or
  the model was unloaded or replaced before its weights could be held; `error` says why)
- `read`: whether the weights came from the weights file rather than a host copy
- `nbytes`, `seconds`: the model's weight bytes, and how long the read took
"""
function stage_weights!(s::Scheduler, names::Vector{String}; hold_seconds::Real = 60.0)
    wc = s.weight_cache
    hold = clamp(Float64(hold_seconds), 0.0, _PREFETCH_HOLD_MAX_SECONDS)
    wc === nothing || _sweep_prefetched!(s)
    # The models each requested name stands for: itself, or a meta's sub-models.
    groups = lock(s.cond) do
        out = Pair{String, Vector{ModelEntry}}[]
        for name in names
            meta = get(s.registry.meta, name, nothing)
            if meta !== nothing
                push!(out, name => ModelEntry[s.registry.by_name[c] for c in meta.calls if haskey(s.registry.by_name, c)])
            else
                e = get(s.registry.by_name, name, nothing)
                e === nothing && throw(ModelNotFoundError(name))
                push!(out, name => ModelEntry[e])
            end
        end
        return out
    end
    entries = unique!(ModelEntry[e for (_, es) in groups for e in es])
    rows = _stage_entries!(s, wc, entries, hold)
    byname = Dict(r.name => r for r in rows)
    # One row per requested name; a meta's is the worst of its sub-models' states.
    rank = Dict("failed" => 5, "deferred" => 4, "staged" => 3, "resident" => 2, "skipped" => 1)
    return map(groups) do (name, es)
        rs = [byname[e.name] for e in es]
        isempty(rs) && return (name = name, state = "resident", read = false, nbytes = 0, seconds = 0.0, error = "")
        worst = rs[argmax([rank[r.state] for r in rs])]
        return (
            name = name, state = worst.state, read = any(r -> r.read, rs),
            nbytes = sum(r -> r.nbytes, rs), seconds = maximum(r -> r.seconds, rs),
            error = join(filter(!isempty, [r.error for r in rs]), "; "),
        )
    end
end

function _stage_entries!(s::Scheduler, wc, entries::Vector{ModelEntry}, hold::Float64)
    row(e, state; read = false, seconds = 0.0, error = "") = (
        name = e.name, state = state, read = read,
        nbytes = e.executable === nothing ? 0 : Int(e.executable.nbytes), seconds = seconds, error = error,
    )
    rows = NamedTuple[]
    pending = Tuple{ModelEntry, WeightRead}[]
    for e in entries
        if wc === nothing || wc.mode != SELF_MANAGED
            push!(rows, row(e, "skipped"))
            continue
        end
        m = e.executable
        if m === nothing || is_device_pinned(m) || m.weights !== nothing || m.host_weights !== nothing
            push!(rows, row(e, "resident"))
            continue
        end
        state, msg = lock(s.cond) do
            # `stage_weights!` took the entries under an earlier hold of the lock, and a reload or
            # unload may have run since. Such an entry's weights file is closed (a read could get
            # another file's bytes through a reused descriptor, or fill a shared region under the old
            # version's digest), and a hold made for it would never be claimed: the version now
            # registered would wait behind it, `:pending`, for good (`_kick_staging!`).
            get(s.registry.by_name, e.name, nothing) === e ||
                return ("failed", "model '$(e.name)' was unloaded or replaced before its weights were read; ask again")
            pf = get(s.prefetched, e.name, nothing)
            if pf !== nothing
                pf.expires = max(pf.expires, time() + hold)      # an earlier request's read covers it
                return ("staged", "")
            end
            e.sched !== nothing && e.sched.host_stage in (:staging, :staged) && return ("staged", "")
            why = _stall_refusal(s, e)
            why === nothing || return ("failed", why)
            held = sum((p.nbytes for p in values(s.prefetched)); init = 0)
            held + m.nbytes > _prefetch_cap(wc) && return ("deferred", "")
            pf = Prefetched(nothing, nothing, Inf, Int(m.nbytes))
            s.prefetched[e.name] = pf
            r = _start_read!(r -> _run_prefetch!(s, r, hold), s, e; kind = :prefetch, prefetch = pf)
            pf.task = r.task
            push!(pending, (e, r))
            return ("", "")
        end
        isempty(state) || push!(rows, row(e, state; error = msg))
    end
    # Each read sets its row when it returns, or when it is abandoned as stalled, so this wait ends
    # within the stall limit even when a read never returns.
    for (e, r) in pending
        wait(r.done)
        o = r.outcome
        push!(rows, row(e, o.state; read = o.read, seconds = o.seconds, error = o.error))
    end
    return rows
end

# The read behind a StageWeights hold. What it read is held for the model's first dispatch, and the
# call that asked for it is answered with the model's row (`r.outcome`, then `r.done`). A read
# abandoned as stalled has had its call answered and its hold dropped already; whatever it returns
# with is released.
function _run_prefetch!(s::Scheduler, r::WeightRead, hold::Float64)
    e = r.entry
    pf = r.prefetch::Prefetched
    wc = s.weight_cache::WeightCache
    staged, failure = _watched_read(wc, r)
    keep = false
    close_file = false
    stalled = false
    lock(s.cond) do
        close_file = _read_ended!(s, r)
        stalled = r.stalled
        if !stalled
            pf.task = nothing
            mine = get(s.prefetched, e.name, nothing) === pf
            if staged !== nothing && mine && get(s.registry.by_name, e.name, nothing) === e
                pf.staged = staged
                pf.expires = time() + hold
                keep = true
            else
                # Not kept, so the hold goes too, whatever became of the model: left behind with
                # nothing in it, it would keep a version registered since `:pending` for good.
                mine && delete!(s.prefetched, e.name)
            end
            r.outcome = if staged === nothing
                (state = "failed", read = false, seconds = time() - r.started, error = sprint(showerror, failure[1]))
            elseif keep
                (state = "staged", read = !staged.hit, seconds = staged.host_seconds, error = "")
            else
                (
                    state = "failed", read = !staged.hit, seconds = staged.host_seconds,
                    error = "model '$(e.name)' was unloaded or replaced while its weights were read; ask again",
                )
            end
            notify(r.done)
        end
        # The dispatch loop may already have requests waiting for this model.
        _kick_staging!(s)
        notify(s.cond)
    end
    failure === nothing || stalled ||
        @warn "reading weights ahead of a placement change failed" model = e.name exception = failure
    _after_read!(r, _releaser(wc, r, keep ? nothing : staged), close_file, stalled, failure)
    return nothing
end

# Make the dispatched model's weights resident, uploading its staged weights when it has them.
# Evictions here may leave other queued models off the device, so staging is reconsidered after.
function _acquire_for_dispatch!(s::Scheduler, entry::ModelEntry)
    wc = s.weight_cache
    wc === nothing && return nothing
    model = entry.executable
    st = entry.sched
    # Already on the device with nothing staged (the common case): a hit, nothing to reconsider.
    (model.weights !== nothing && st.host_stage === :none) && return acquire!(wc, entry)
    staged = nothing
    lock(s.cond) do
        if st.host_stage === :staged
            staged = st.staged
            st.staged = nothing
            st.host_stage = :loading
        elseif st.host_stage === :none
            st.host_stage = :loading          # keeps a background read from starting for it now
        end
    end
    # Selection never picks a model whose weights still have to be read (`_awaiting_stage`), so a
    # load here uploads from host memory. Refuse rather than read the file on the dispatch loop.
    if staged === nothing && model.weights === nothing && needs_host_stage(wc, entry)
        lock(() -> (st.host_stage === :loading && (st.host_stage = :none)), s.cond)
        throw(ErrorException("internal: model '$(entry.name)' was dispatched before its weights were read into host memory"))
    end
    try
        acquire!(wc, entry; staged = staged)
    finally
        lock(s.cond) do
            st.host_stage === :loading && (st.host_stage = :none)
            _kick_staging!(s)
        end
    end
    return nothing
end

# Release a staged model's weights when it has nothing left to run, so its region and its staging
# slot do not stay held: its dispatch answered every request it took without uploading (they failed
# before the upload, or were dropped at their deadline, withdrawn or cancelled) and no other request
# has been queued since. A model with requests still queued keeps its weights for its next dispatch.
# Run at the end of every dispatch (`execute_and_record!`). Caller does not hold `s.cond`.
function _release_idle_stage!(s::Scheduler, entry::ModelEntry)
    wc = s.weight_cache
    wc === nothing && return nothing
    staged = lock(s.cond) do
        st = entry.sched
        (st.host_stage === :staged && isempty(st.queue)) || return nothing
        h = st.staged
        st.staged = nothing
        st.host_stage = :none
        _kick_staging!(s)
        return h
    end
    staged === nothing || release_staged!(wc, entry, staged; uploaded = false)
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Submission and the dispatch loop
# ---------------------------------------------------------------------------------------------

# Enqueue a request onto its model's queue and wake the dispatch loop. Unknown models
# (`ModelNotFoundError`), a stopped scheduler (`WorkerUnavailableError`), and a full queue
# (`QueueFullError`) are reported immediately through the reply channel (the caller sees the
# exception). `max_queue_depth` caps each model's queue independently, so one backlogged model
# cannot starve admission for the others. `expected` is the entry whose preprocess made
# `qr.prepared` (see `infer`); when another entry holds the name by now, the request is refused
# rather than run on a version it was not prepared for.
function submit!(s::Scheduler, qr::QueuedRequest; expected::Union{ModelEntry, Nothing} = nothing)
    lock(s.cond) do
        if !s.running
            put!(qr.reply, _unavailable(s))
            return
        end
        # Targets are regular models only: a meta runs on the request task (see `infer`), and its
        # GPU sub-calls arrive here as committed requests for the sub-models, which live in `by_name`.
        entry = get(s.registry.by_name, qr.req.model_name, nothing)
        if entry === nothing || entry.sched === nothing
            put!(qr.reply, ModelNotFoundError(qr.req.model_name))
            return
        end
        # A reload replaced the model while this request was in preprocess. Run here, its inputs
        # would meet the new program and its outputs the old postprocess, so it fails instead, as a
        # `ModelNotFoundError`: the version it was prepared for is unloaded, which the transport
        # maps to the retryable NOT_FOUND.
        if expected !== nothing && entry !== expected
            name = qr.req.model_name
            put!(
                qr.reply, ModelNotFoundError(
                    name, "model '$name' was replaced during the request (the version that preprocessed it was unloaded); retry"
                )
            )
            return
        end
        if length(entry.sched.queue) >= s.cfg.max_queue_depth
            put!(qr.reply, QueueFullError(qr.req.model_name, s.cfg.max_queue_depth))
            return
        end
        if qr.committed
            # An in-flight meta's continuation jumps the line: push to the front of its sub-model's
            # queue and record it so `select_dispatch!` dispatches this model ahead of the discipline
            # scan. One such request per in-flight meta (sub-models are meta-exclusive, so distinct metas
            # never contend for the same sub-model's front); the gate bounds the count to the concurrency.
            pushfirst!(entry.sched.queue, qr)
            push!(s.committed, qr)
        else
            push!(entry.sched.queue, qr)
        end
        _kick_staging!(s)
        notify(s.cond)
    end
    return nothing
end

# The dispatch loop, guarded. A failure one request or one dispatch can be held to is answered to
# just those callers (the execution's own catch, the guard around `execute_and_record!` below, and
# each control command's), and serving goes on. An exception that escapes anyway (one in selecting
# the next dispatch, say) means nothing will ever be dispatched again, so it is fatal to the worker
# rather than a silent stop: see `_dispatch_loop_died!`.
function dispatch_loop(s::Scheduler)
    try
        _dispatch_loop(s)
    catch err
        _dispatch_loop_died!(s, err, catch_backtrace())
    end
    return nothing
end

function _dispatch_loop(s::Scheduler)
    while true
        cmds, chosen = lock(s.cond) do
            while s.running && isempty(s.control) && !_has_queued(s)
                _mark_idle!(s.progress)   # nothing to do: waiting is not a stall
                wait(s.cond)
            end
            if !s.running
                _mark_idle!(s.progress)
                _reject_pending_locked!(s)
                return (ControlCommand[], nothing)
            end
            _mark_progress!(s.progress)   # work in hand: progress is measured from here
            # Drain control commands first; defer selecting a dispatch to the next iteration so
            # residency transitions take effect before the dispatch that may depend on them.
            if !isempty(s.control)
                cs = s.control
                s.control = ControlCommand[]
                return (cs, nothing)
            end
            return (ControlCommand[], select_dispatch!(s, time()))
        end
        # Control commands run on this (the dispatch) thread, the sole mutator of residency. The
        # slow host/device work inside `op` runs outside `s.cond`.
        for c in cmds
            try
                put!(c.reply, c.op(s))
            catch e
                put!(c.reply, e)
            end
            _mark_progress!(s.progress)
        end
        (isempty(cmds) && chosen === nothing) && break
        if chosen !== nothing
            try
                execute_and_record!(s, chosen)
            catch err
                # A failure outside the execution's own catch (the deadline admission, releasing an
                # idle staged model) belongs to this dispatch alone: fail its requests, keep serving.
                @error "a dispatch failed outside its execution; failing its requests" model = chosen.entry.name exception = (err, catch_backtrace())
                _fail_unanswered!(chosen.taken, err)
            end
            _mark_progress!(s.progress)
            _maybe_compact!(s)
            _mark_progress!(s.progress)
        end
        # A residency change or a compaction can leave queued models off the device.
        lock(() -> _kick_staging!(s), s.cond)
        isempty(s.prefetched) || _sweep_prefetched!(s)
    end
    return nothing
end

# Answer with `err` each request that has no reply waiting. Only the dispatch loop replies to a taken
# request, so an empty channel means it was never answered (or its caller has already taken the
# reply, and the extra one is never read).
function _fail_unanswered!(qrs::Vector{QueuedRequest}, err)
    for qr in qrs
        isready(qr.reply) || put!(qr.reply, err)
    end
    return nothing
end

# The dispatch loop died with `err`, which no single request can be held to, and nothing will be
# dispatched again. Log it with its backtrace; mark the scheduler failed, so ServerLive, ServerReady,
# ModelReady and `/readyz` say no from now on and every later request or control command is rejected
# with `WorkerUnavailableError` (UNAVAILABLE); answer everything already queued the same way; then run
# the `on_failure` hook once (a worker process exits nonzero there so its supervisor restarts it, see
# `on_dispatch_failure!`). A failure during a shutdown already under way only logs.
function _dispatch_loop_died!(s::Scheduler, err, bt)
    @error "the dispatch loop failed; this worker no longer serves and reports not live" exception = (err, bt)
    stopping = false
    hook = lock(s.cond) do
        stopping = !s.running
        s.failure = err
        s.running = false
        try
            _reject_pending_locked!(s)
        catch e
            @error "releasing held weights after the dispatch loop failed" exception = (e, catch_backtrace())
        end
        notify(s.cond)
        return s.on_failure
    end
    (hook === nothing || stopping) && return nothing
    try
        hook(err)
    catch e
        @error "the dispatch-failure hook threw" exception = (e, catch_backtrace())
    end
    return nothing
end

"""
    on_dispatch_failure!(scheduler, f) -> scheduler

Run `f(err)` once if the dispatch loop dies with the unexpected error `err`, or right away if it
already has. A blocking [`serve`](@ref) (a worker process) installs a hook that exits the process
nonzero, so the node supervisor restarts it; an embedded server keeps running and reports not live.
"""
function on_dispatch_failure!(s::Scheduler, f)
    failed = lock(s.cond) do
        s.on_failure = f
        return s.failure
    end
    failed === nothing || f(failed)
    return s
end

# ---------------------------------------------------------------------------------------------
# Control plane: residency transitions and live policy, executed on the dispatch thread
# ---------------------------------------------------------------------------------------------

# Enqueue a control command and block until the dispatch loop runs it. Re-raises any error.
# Raises immediately when the dispatch loop is not running (a command pushed then would never
# be drained and its caller would block forever).
function _run_control(s::Scheduler, op::Function)
    reply = Channel{Any}(1)
    lock(s.cond) do
        s.running || throw(_unavailable(s))
        push!(s.control, ControlCommand(op, reply))
        notify(s.cond)
    end
    result = take!(reply)
    result isa Exception && throw(result)
    return result
end

"""
    set_residency!(scheduler, name, target::ResidencyState) -> ResidencyState

Move a model to the `target` residency floor. Only meaningful in externally-managed mode; the
worker rejects it otherwise. Runs on the dispatch thread (the sole residency mutator) and blocks
until applied. Raises on an unknown model or in self-managed mode, and with a
[`WeightsReadStalledError`](@ref) (UNAVAILABLE) when reading the model's weights makes no progress
for `scheduler.weight_read_stall_seconds`, or reads of model weights have stalled on this worker;
the next residency change does not wait for such a read.
"""
function set_residency!(s::Scheduler, name::AbstractString, target::ResidencyState)
    # One residency change at a time, so the weights read ahead below are still what the op needs.
    return lock(s.residency_lock) do
        pre = _preread_for_residency(s, String(name), target)
        try
            _set_residency_op!(s, name, target, pre)
        catch
            for (nm, p) in pre
                p.floor === nothing || host_release!(s.weight_cache.store, nm)
            end
            rethrow()
        finally
            for (_, p) in pre
                p.staged === nothing || release_staged!(s.weight_cache, p.entry, p.staged; uploaded = false)
            end
        end
    end
end

# Read, off the dispatch loop, what moving `name` (each sub-model, for a meta) to `target` would
# otherwise read on it: a host floor for PINNED_SYSTEM, the weights to upload for PINNED_DEVICE.
# A model with a host copy to take them from needs nothing read. Each read is watched like a
# staging read (`_watched_preread`): one that stalls fails the change with
# `WeightsReadStalledError`, and what was read for earlier sub-models is let go.
function _preread_for_residency(s::Scheduler, name::String, target::ResidencyState)
    pre = Dict{String, NamedTuple{(:entry, :floor, :staged), Tuple{ModelEntry, Any, Any}}}()
    wc = s.weight_cache
    (wc === nothing || wc.mode != EXTERNALLY_MANAGED || target == UNPINNED) && return pre
    entries = lock(s.cond) do
        meta = get(s.registry.meta, name, nothing)
        meta === nothing && return ModelEntry[e for e in (get(s.registry.by_name, name, nothing),) if e !== nothing]
        return ModelEntry[s.registry.by_name[c] for c in meta.calls if haskey(s.registry.by_name, c)]
    end
    try
        for e in entries
            m = e.executable
            (m === nothing || m.state == target || m.host_weights !== nothing) && continue
            if target == PINNED_SYSTEM
                has_host_copy(wc, e) && !(wc.store isa SharedWeightStore) && continue   # taken from the private cache
                floor = _watched_preread(
                    s, e, () -> host_materialize(wc.store, e, m.sig.weight_names),
                    _ -> host_release!(wc.store, e.name)
                )
                pre[e.name] = (entry = e, floor = floor, staged = nothing)
            elseif target == PINNED_DEVICE && m.weights === nothing
                staged = _watched_preread(
                    s, e, () -> stage_host_weights(wc, e), st -> release_staged!(wc, e, st; uploaded = false)
                )
                pre[e.name] = (entry = e, floor = nothing, staged = staged)
            end
        end
    catch
        for (nm, p) in pre
            p.floor === nothing || host_release!(wc.store, nm)
            p.staged === nothing || release_staged!(wc, p.entry, p.staged; uploaded = false)
        end
        rethrow()
    end
    return pre
end

# Run `read()` for a residency change as a watched background read of `e`'s weights, and wait for
# it: its result, or the error it failed with. One that stops making progress is abandoned: this
# throws `WeightsReadStalledError` at once, so the change fails (UNAVAILABLE) and the caller lets go
# of `residency_lock`, and if the read ever returns, `release(result)` lets go of what it read. No
# read starts for a model whose earlier read stalled, nor while reads are stalled at the bound
# (`_stall_refusal`).
function _watched_preread(s::Scheduler, e::ModelEntry, read::Function, release::Function)
    r = lock(s.cond) do
        why = _stall_refusal(s, e)
        why === nothing || throw(WeightsReadStalledError(e.name, why))
        return _start_read!(r -> _run_preread!(s, r, read, release), s, e; kind = :residency)
    end
    wait(r.done)
    o = r.outcome
    o isa Some || throw(o)
    return something(o)
end

# The body of a residency change's watched read: run it, then hand the result to the waiting change,
# or, when the read was abandoned meanwhile, release it.
function _run_preread!(s::Scheduler, r::WeightRead, read::Function, release::Function)
    result = nothing
    failure = nothing
    try
        result = with_read_progress(read, r.progress)
    catch err
        failure = (err, catch_backtrace())
    end
    close_file = false
    stalled = false
    lock(s.cond) do
        close_file = _read_ended!(s, r)
        stalled = r.stalled
        if !stalled
            r.outcome = failure === nothing ? Some(result) : failure[1]
            notify(r.done)
        end
        _kick_staging!(s)
        notify(s.cond)
    end
    _after_read!(r, stalled && failure === nothing ? () -> release(result) : nothing, close_file, stalled, failure)
    return nothing
end

function _set_residency_op!(s::Scheduler, name::AbstractString, target::ResidencyState, pre)
    floor_of(e) = (p = get(pre, e.name, nothing); p === nothing ? nothing : p.floor)
    staged_of(e) = (p = get(pre, e.name, nothing); p === nothing ? nothing : p.staged)
    return _run_control(
        s, function (sch)
            sch.weight_cache === nothing &&
                throw(ErrorException("residency control requires a GPU worker (a CPU worker keeps every model resident)"))
            sch.weight_cache.mode == EXTERNALLY_MANAGED ||
                throw(ErrorException("worker is self-managed; residency is not externally controllable"))
            nm = String(name)
            # A meta owns no weights: its residency is its sub-models' residency (the group is the unit).
            # Pinning a meta pins each sub. When unpinning, leave a sub resident if another meta also
            # declares it (conservative — avoids prematurely unpinning a shared stage that a sibling meta
            # still needs; the cost is a little extra resident memory, never a wrong eviction).
            meta = get(sch.registry.meta, nm, nothing)
            if meta !== nothing
                last = target
                for sub in meta.calls
                    e = get(sch.registry.by_name, sub, nothing)
                    e === nothing && continue
                    if target == UNPINNED && any(
                            other.name != nm && sub in other.calls
                                for other in values(sch.registry.meta)
                        )
                        continue
                    end
                    last = set_residency_state!(sch.weight_cache, e, target; floor = floor_of(e), staged = staged_of(e))
                end
                return last
            end
            entry = get(sch.registry.by_name, nm, nothing)
            entry === nothing && throw(ModelNotFoundError(name))
            return set_residency_state!(sch.weight_cache, entry, target; floor = floor_of(entry), staged = staged_of(entry))
        end
    )
end

"""
    set_policy!(scheduler, name; weight=nothing) -> nothing

Update a model's live scheduler policy. `weight` is consulted only by the fair discipline and must
be a positive finite number. Available in both residency modes. Raises a `ModelNotFoundError` on an
unknown model, and an [`InvalidInputError`](@ref) for a weight that is zero, negative, infinite or
NaN (the `SetModelPolicy` RPC answers it with `INVALID_ARGUMENT`).
"""
function set_policy!(s::Scheduler, name::AbstractString; weight::Union{Real, Nothing} = nothing)
    # The fair discipline charges a dispatch as its seconds over the weight, so only a positive,
    # finite weight orders models: Inf charges nothing, and NaN fails every comparison.
    weight === nothing || (isfinite(weight) && weight > 0) ||
        throw(InvalidInputError("weight must be a positive finite number, got $weight"))
    lock(s.cond) do
        entry = get(s.registry.by_name, String(name), nothing)
        (entry === nothing || entry.sched === nothing) && throw(ModelNotFoundError(name))
        weight === nothing || (entry.sched.weight = Float64(weight))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Memory compaction: free all resident device weights so the allocator coalesces its free list,
# then reload the working set. Defragments the device arena (see `compact!` on the WeightCache).
# ---------------------------------------------------------------------------------------------

# Body assuming it already runs on the dispatch thread (mirrors `_admit_entry!`/`_evict_entry!`).
# A no-op without the on-demand cache: there, every model is permanently device-resident from
# startup (nothing churns), so there is no fragmenting working set to compact.
function _compact_entry!(sch::Scheduler; reload_models::Vector{String} = String[])
    sch.weight_cache === nothing && return 0
    return compact!(sch.weight_cache, sch.registry; reload = reload_models)
end

"""
    compact!(scheduler; reload_models=String[]) -> Int

Free the resident non-pinned device weights so the allocator coalesces its free list, then reload
each model named in `reload_models`. An empty list frees only; non-pinned models not listed reload
lazily on their next dispatch (self-managed mode). Device-pinned models are left in place (loaded
once at startup, never reloaded from disk). A no-op when the on-demand weight cache is disabled.
Runs on the dispatch thread (the sole residency mutator) and blocks until applied. Returns the
number of models reloaded.
"""
compact!(s::Scheduler; reload_models::Vector{String} = String[]) =
    _run_control(s, sch -> _compact_entry!(sch; reload_models = reload_models))

# Automatic, worker-local compaction driven by `cfg.compaction_interval` (0 disables), counting
# on-demand weight-cache loads, not dispatches: a load is what places a variable-size weight block
# and so drives fragmentation, while a dispatch to an already-resident model does not. This is the
# standalone (no-gateway) trigger and is off by default, so a gateway-fronted worker never
# self-compacts (the gateway owns compaction there, via the CompactMemory RPC). It refills lazily:
# the on-demand region is freed and models come back as requests arrive.
#
# Called from the dispatch loop after a dispatch, so it is already on the dispatch thread and must
# NOT go through `_run_control` (nesting it would deadlock). A failure must not kill the loop.
function _maybe_compact!(s::Scheduler)
    n = s.cfg.compaction_interval
    (n > 0 && s.weight_cache !== nothing) || return nothing
    s.weight_cache.loads - s.compact_loads_mark >= n || return nothing
    s.compact_loads_mark = s.weight_cache.loads
    try
        _compact_entry!(s)
    catch e
        @warn "automatic compaction failed" exception = (e, catch_backtrace())
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Model lifecycle: admit/evict a model from the live system on the dispatch thread.
#
# These are the reusable seam for runtime model load/unload: the RepositoryModelLoad path (the
# dynamic directory watcher, see watcher.jl) compiles a bundle (reusing `load_bundles` +
# `build_loaded_model`) and calls `admit!`/`load_model!`; the unload path calls `evict!`. They run
# on the dispatch thread via the control queue, so they are the sole mutator of residency; the fast
# registry/queue mutation is done under `s.cond` (consistent with `submit!`), and the slow work
# (compile, warmup, freeing device buffers) runs outside it.
#
# `_admit_entry!`/`_evict_entry!` are the bodies that assume they are *already* on the dispatch
# thread; the public `admit!`/`evict!` wrap them in `_run_control`. `load_model!` composes the two
# in a single control op so a reload is one stop-the-world swap (evict the old, freeing its device
# memory, before compiling the new). The watcher calls `load_model!`/`evict!`; do not call the
# public wrappers from inside a control op (nesting `_run_control` would deadlock).
# ---------------------------------------------------------------------------------------------

# Insert a compiled entry into the live registry and prepare it for dispatch. Runs on the dispatch
# thread (warmup executes the model). Raises on a duplicate name. Returns the model name. `staged` is
# the model's weights read into host memory before the control op (see `_prestage_for_admit`): the
# first upload uses them, so warmup and the budget re-probe never read the weights file here.
function _admit_entry!(sch::Scheduler, entry::ModelEntry; staged = nothing)
    lock(sch.cond) do
        haskey(sch.registry.by_name, entry.name) &&
            throw(ErrorException("model '$(entry.name)' is already registered"))
        sch.registry.by_name[entry.name] = entry
        init_sched_state!(sch, entry, time())
    end
    staged === nothing || acquire!(sch.weight_cache, entry; staged = staged)
    # Warmup runs the model on this (the dispatch) thread, outside the lock.
    sch.cfg.discipline == FAIR && _warmup_entry!(sch, entry)
    return entry.name
end

# Whether admitting `entry` runs it, and so loads its weights, inside the control op: warmup under
# the fair discipline, or the budget re-probe after a hot load.
_admit_runs_model(sch::Scheduler) = sch.cfg.discipline == FAIR || sch.autosize !== nothing

# Read an unpinned (or not yet floored) model's weights into host memory before its admission's
# control op, when that op will run the model, so the op only uploads them. `nothing` when nothing
# would be read there. Runs on the caller's task, off the dispatch loop.
function _prestage_for_admit(
        sch::Scheduler, entry::ModelEntry, names::Vector{String}; resident::Bool = false
    )
    wc = sch.weight_cache
    (wc === nothing || wc.mode != SELF_MANAGED || resident || !_admit_runs_model(sch)) && return nothing
    return stage_host_weights(wc, entry, names)
end

# Remove a model from the live registry, rejecting any queued requests, and release its residency.
# Runs on the dispatch thread. Returns the removed entry, or `nothing` if it was not registered.
function _evict_entry!(sch::Scheduler, name::AbstractString)
    entry, staged, pf, reading = lock(sch.cond) do
        e = pop!(sch.registry.by_name, String(name), nothing)
        h = nothing
        p = e === nothing ? nothing : pop!(sch.prefetched, e.name, nothing)
        if e !== nothing && e.sched !== nothing
            for qr in e.sched.queue
                put!(qr.reply, ModelNotFoundError(name, "model '$name' was unloaded"))
            end
            empty!(e.sched.queue)
            h = e.sched.staged
            e.sched.staged = nothing
            e.sched.host_stage = :none
        end
        # A background read of this model's weights that has not returned (stalled or not) keeps the
        # weights file open: a descriptor closed under a running pread can be reused by another
        # open, and the read would then copy the wrong file's bytes. The read closes it once it
        # returns, and drops what it read, since the model is gone. Nothing here waits for it: this
        # runs on the dispatch loop, and a read on storage that stopped answering never returns.
        reading = false
        if e !== nothing
            for r in sch.reads
                r.entry === e || continue
                r.close_weights = true
                reading = true
            end
        end
        return (e, h, p, reading)
    end
    entry === nothing && return nothing
    # A hold for a placement change: what it read is released now, and a read still filling it finds
    # the hold gone when it returns and releases what it read itself.
    pf === nothing || pf.staged === nothing || sch.weight_cache === nothing ||
        release_staged!(sch.weight_cache, entry, pf.staged; uploaded = false)
    staged === nothing || sch.weight_cache === nothing ||
        release_staged!(sch.weight_cache, entry, staged; uploaded = false)
    # Release residency outside the lock (device frees can be slow).
    sch.weight_cache === nothing || release_all!(sch.weight_cache, entry)
    # Close the weights file: it holds the loaded version's inode, which a replaced file keeps alive
    # (and keeps on disk) until then. A read still running closes it when it returns.
    reading || close_weights!(entry.weights)
    # Eagerly free every compiled executable (all variants and batch sizes). Waiting for GC would
    # leave the executables' command buffers resident on the device indefinitely (they live outside
    # the BFC arena and only the finalizer reclaims them). Safe here: this runs on the dispatch
    # thread, the sole executor, and the model's queue was just drained, so nothing is in flight.
    if entry.executable !== nothing
        for inner in values(entry.executable.execs), exec in values(inner)
            free_executable!(sch.backend, exec)
        end
    end
    nbytes = entry.executable === nothing ? 0 : entry.executable.nbytes
    log_model_unloaded(
        String(name), nbytes;
        memory = memory_report(sch.backend, sch.pool; registry = sch.registry, weight_cache = sch.weight_cache)
    )
    return entry
end

"""
    admit!(scheduler, entry::ModelEntry) -> String

Register a fully compiled `ModelEntry` (its `executable` already built) into the live system:
insert it under the lock, initialize its scheduling state, and seed its costs (fair discipline).
Raises on a duplicate name or an uncompiled entry. Returns the model name.
"""
function admit!(s::Scheduler, entry::ModelEntry)
    entry.executable === nothing &&
        throw(ErrorException("model '$(entry.name)' has no compiled executable; compile before admit!"))
    m = entry.executable
    staged = _prestage_for_admit(
        s, entry, m.sig.weight_names;
        resident = m.weights !== nothing || m.host_weights !== nothing || is_device_pinned(m)
    )
    try
        return _run_control(s, sch -> _admit_entry!(sch, entry; staged = staged))
    finally
        staged === nothing || release_staged!(s.weight_cache, entry, staged; uploaded = false)
    end
end

"""
    evict!(scheduler, name) -> Union{ModelEntry,Nothing}

Remove a model from the live system: drop it from the registry, reject any queued requests (so
callers unblock), release its residency (device buffers and any shared host floor), and free its
compiled executables (reclaiming their device-side command buffers). Returns the removed entry,
or `nothing` if it was not registered; the entry's executables are dead and must not be re-run.
"""
function evict!(s::Scheduler, name::AbstractString)
    return _run_control(s, sch -> _evict_entry!(sch, String(name)))
end

"""
    load_model!(scheduler, backend, pool, entry; state, on_demand, store, unchanged) -> Union{String, Nothing}

Compile and admit a freshly loaded (uncompiled) bundle `entry`, replacing any existing model of
the same name. Runs entirely on the dispatch thread inside one control op (a stop-the-world swap):
any existing model is evicted first so its device memory is freed before the new artifact is
compiled, then the new entry is compiled (`build_loaded_model`) and admitted. This is the dynamic
directory watcher's load/reload path. Returns the model name.

`unchanged()` says whether the bundle's files are still the ones `entry` was read from. It is asked
before the control op, where `false` leaves any existing model serving, and again once the new
version is compiled, before it is admitted, where `false` discards it (the existing model is gone by
then). Either way the entry is never admitted and `nothing` is returned.
"""
function load_model!(
        s::Scheduler, backend::AbstractBackend, pool::MemoryPool, entry::ModelEntry;
        state::ResidencyState, on_demand::Bool, store::WeightStore = PrivateWeightStore(),
        executable_cache::Bool = false, unchanged::Function = () -> true
    )
    # Read the weights before the control op, which runs on the dispatch loop: a device-resident
    # model's host arrays (and a CPU worker's, which keeps every model resident), or an on-demand
    # model's staged weights when admission will run it. A system-pinned model's host floor is
    # materialized after the op instead (`_attach_floor!`): the old version's floor, which shares
    # its name in the shared store, is released inside the op.
    names = weight_order(entry.weights)
    host_arrays = nothing
    staged = nothing
    if !on_demand || state == PINNED_DEVICE
        host_arrays = materialize_host_weights(entry.weights, names)
        _check_unmodified(entry)
    else
        staged = _prestage_for_admit(s, entry, names)
    end
    defer_floor = on_demand && state == PINNED_SYSTEM
    name = try
        # The files changed while the weights were read: keep the running version, try again later.
        unchanged() || (close_weights!(entry.weights); return nothing)
        _run_control(
            s, function (sch)
                _evict_entry!(sch, entry.name)          # no-op on a fresh load; frees device memory on reload
                entry.executable = build_loaded_model(
                    backend, pool, entry;
                    state = state, on_demand = on_demand, store = store, source = :dynamic,
                    executable_cache = executable_cache, host_arrays = host_arrays,
                    defer_floor = defer_floor
                )
                # Changed during the compile: this version may mix old and new files. Never admit it.
                unchanged() || (_discard_unadmitted!(sch, entry); return nothing)
                n = _admit_entry!(sch, entry; staged = staged)
                _reprobe_after_load!(sch, entry)
                return n
            end
        )
    finally
        staged === nothing || release_staged!(s.weight_cache, entry, staged; uploaded = false)
    end
    name === nothing && return nothing
    defer_floor && _attach_floor!(s, entry, store, names)
    return name
end

# Free what `build_loaded_model` gave an entry that is then not admitted (see `load_model!`): the
# releases `_evict_entry!` makes, for an entry that never reached the registry. Dispatch thread.
function _discard_unadmitted!(sch::Scheduler, entry::ModelEntry)
    sch.weight_cache === nothing || release_all!(sch.weight_cache, entry)
    close_weights!(entry.weights)
    for inner in values(entry.executable.execs), exec in values(inner)
        free_executable!(sch.backend, exec)
    end
    return nothing
end

# Give a hot-loaded system-pinned model its host floor: read it off the dispatch loop, then commit
# it in a control op. Until then the model serves like an unpinned one (staged on demand).
function _attach_floor!(s::Scheduler, entry::ModelEntry, store::WeightStore, names::Vector{String})
    floor = host_materialize(store, entry, names)
    attached = _run_control(
        s, function (sch)
            live = get(sch.registry.by_name, entry.name, nothing) === entry
            (live && entry.executable.host_weights === nothing) || return false
            sch.weight_cache === nothing ? (entry.executable.host_weights = floor) :
                attach_host_floor!(sch.weight_cache, entry, floor)
            return true
        end
    )
    attached || host_release!(store, entry.name)
    return nothing
end

# After a hot load, fold the new model into the weight budget without a restart: reset the
# allocator peak the compile left behind, measure this one model's scratch in isolation of that
# peak, and re-resolve the budget with the arena and knobs the startup auto-sizing used. Only the
# max scratch can grow here (a reload of a lighter version keeps the old ceiling until restart, the
# safe direction). No-op when the cache or auto-sizing is off, or the backend has no device stats.
function _reprobe_after_load!(sch::Scheduler, entry::ModelEntry)
    (sch.weight_cache === nothing || sch.autosize === nothing || entry.executable === nothing) && return nothing
    device_memory_stats(sch.backend, sch.pool) === nothing && return nothing
    isempty(entry.manifest.executable_inputs) && return nothing
    resettable = clear_memory_stats!(sch.backend, sch.pool)
    if !is_device_pinned(entry.executable)
        try
            acquire!(sch.weight_cache, entry)
        catch err
            err isa NotResidentError && return nothing
            rethrow()
        end
    end
    pinned = pinned_weight_bytes(sch.registry)
    scratch = _probe_entry_scratch!(sch, entry, resettable, pinned + entry.executable.nbytes)
    model_footprint(sch.weight_cache, entry).scratch_bytes = scratch
    a = sch.autosize
    lock(sch.weight_cache.lock) do
        max_scratch = max(sch.weight_cache.max_scratch, scratch)
        b = weight_budget(;
            arena = a.arena, fraction = a.fraction, wiggle = a.wiggle,
            max_scratch = max_scratch, pinned_bytes = pinned
        )
        sch.weight_cache.max_bytes = b.on_demand_budget
        sch.weight_cache.pinned_bytes = pinned
        sch.weight_cache.max_scratch = max_scratch
        sch.weight_cache.weight_pool = b.weight_pool
        @info "weight budget re-resolved after model load" model = entry.name model_scratch = scratch max_scratch = max_scratch pinned_bytes = pinned weight_pool = b.weight_pool on_demand_budget = b.on_demand_budget resettable_peak = resettable
    end
    return nothing
end

"""
    unload_model!(scheduler, name) -> Union{ModelEntry,Nothing}

Alias for [`evict!`](@ref): remove a model from the live system. Provided for symmetry with
[`load_model!`](@ref) on the dynamic directory-watch path.
"""
unload_model!(s::Scheduler, name::AbstractString) = evict!(s, name)

"""
    put_meta!(scheduler, entry::MetaEntry) -> String

Register (or replace) a meta model. Meta models carry no executable or scheduling state and the
dispatch loop never touches `registry.meta`, so this mutates the registry directly under the
scheduler lock rather than through the control queue. Returns the model name.
"""
function put_meta!(s::Scheduler, entry::MetaEntry)
    lock(s.cond) do
        init_sched_state!(s, entry, time())   # give it a queue + sched so the dispatch loop can run it
        s.registry.meta[entry.name] = entry
    end
    return entry.name
end

"""
    remove_meta!(scheduler, name) -> Nothing

Remove a meta model from the registry (the dynamic watcher's unload path for meta bundles).
"""
function remove_meta!(s::Scheduler, name::AbstractString)
    lock(s.cond) do
        e = pop!(s.registry.meta, String(name), nothing)
        if e !== nothing && e.sched !== nothing
            for qr in e.sched.queue       # unblock any callers waiting on an in-flight meta request
                put!(qr.reply, ModelNotFoundError(name, "model '$name' was unloaded"))
            end
            empty!(e.sched.queue)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Waiting for a reply: the dispatch timeout and cancellation
# ---------------------------------------------------------------------------------------------
#
# A caller waits on its request's reply channel, bounded by `scheduler.dispatch_timeout_seconds`
# from submission and, for a gRPC call, by the client cancelling it. Either way out WITHDRAWS the
# request: the dispatch loop drops a withdrawn request where it drops expired ones
# (`execute_and_record!`), so one still queued never runs, and the result of one already executing
# is put on a channel nobody reads. The caller's wake-up is a sentinel its own timer puts on the reply
# channel, which has room for it beside the scheduler's one reply, so neither side ever blocks.

# How often a waiting caller's timer checks its dispatch timeout and its cancellation probe.
const _WAIT_POLL_SECONDS = 0.1

# The cancellation probe of the request being served: a zero-argument function that returns true
# once the client has cancelled the call, or nothing. `infer` sets it around a meta's orchestration,
# so the meta's sub-calls (which re-enter `infer`, on the same task or tasks it spawns) inherit it.
const _CANCEL_PROBE = Base.ScopedValues.ScopedValue{Any}(nothing)

# Put on a request's reply channel by its own caller's timer, never by the scheduler.
struct _StopWaiting
    err::Exception
end

# Withdraw a request its caller no longer waits for; see `_withdrawn`.
_withdraw!(qr::QueuedRequest) = (qr.withdrawn[] = true; nothing)

# Whether the dispatch loop must drop `qr` rather than run it: its caller stopped waiting.
_withdrawn(qr::QueuedRequest) = qr.withdrawn[]

_is_cancelled(probe) = probe !== nothing && (
    try
        probe() === true
    catch
        false
    end
)

# Wait for the reply to each of `qrs` (one request, or the pieces of a split one) and return them in
# order. Past the dispatch timeout, or once `cancelled` reports the call cancelled, every piece is
# withdrawn and `DispatchTimeoutError` or `RequestCancelledError` is thrown.
function _await_replies(s::Scheduler, qrs::Vector{QueuedRequest}, cancelled)
    timeout = s.cfg.dispatch_timeout_seconds
    timed = timeout > 0 && isfinite(timeout)
    (timed || cancelled !== nothing) || return Any[take!(qr.reply) for qr in qrs]
    model = qrs[1].req.model_name
    t0 = time()
    poll = timed ? min(_WAIT_POLL_SECONDS, timeout) : _WAIT_POLL_SECONDS
    # A Timer runs its callback on one task, one firing at a time, and this one closes itself when it
    # fires, so each channel gets at most one wake-up.
    timer = Timer(poll; interval = poll) do t
        err = if timed && time() - t0 >= timeout
            DispatchTimeoutError(model, timeout)
        elseif _is_cancelled(cancelled)
            RequestCancelledError(model)
        else
            nothing
        end
        err === nothing && return
        close(t)
        for qr in qrs
            put!(qr.reply, _StopWaiting(err))
        end
    end
    try
        replies = Vector{Any}(undef, length(qrs))
        for (i, qr) in enumerate(qrs)
            r = take!(qr.reply)
            if r isa _StopWaiting
                foreach(_withdraw!, qrs)
                throw(r.err)
            end
            replies[i] = r
        end
        return replies
    finally
        close(timer)
    end
end

# A preprocess that leaves out an input the shape variant is read from would fail the variant lookup
# on the dispatch loop, where the failure cannot be pinned on this one request; fail it here instead.
function _check_variant_inputs(entry::ModelEntry, prepared::Vector{NamedTensor})
    entry.executable === nothing && return nothing
    for (nm, _) in entry.executable.sig.variant_spec
        any(t -> t.name == nm, prepared) ||
            throw(ErrorException("preprocess of model '$(entry.name)' did not produce executable input '$nm'"))
    end
    return nothing
end

"""
    infer(scheduler, request; cancelled=nothing) -> Vector{NamedTensor}

Submit a request and block until the scheduler returns the result. Re-raises any error
captured during dispatch.

Runs the model's `preprocess`/`postprocess` hooks here, on the caller's task, rather than on the
dispatch loop: preprocess before the request is queued, postprocess on the raw device outputs the
loop hands back. Because each gRPC request runs on its own task, many requests' hook work
proceeds in parallel and overlaps the single, serialized GPU execution. The dispatch loop coalesces
and runs `qr.prepared` and never executes a request whose preprocess has not finished, since a
request is only enqueued (made visible to the loop) after preprocess returns. A request runs on
the version of the model whose preprocess it went through: if a reload replaces the model in the
meantime, the request fails (NOT_FOUND on the wire, which a client or gateway retries).

The wait is bounded by `scheduler.dispatch_timeout_seconds` (0 waits indefinitely): past it the
request is withdrawn (it never runs if still queued; a running one's result is discarded) and
[`DispatchTimeoutError`](@ref) is thrown. `cancelled`, a zero-argument function that returns `true`
once the caller's client has cancelled the call, is polled while waiting; it withdraws the request
the same way and throws [`RequestCancelledError`](@ref). A meta's sub-calls inherit it.

`scratch_out` applies to a meta only (a regular model acquires no scratch). When given, the pool slots
the meta's `call.scratch` acquires are collected there and stay held after `infer` returns or throws,
because the outputs may alias them; the caller releases them with `release_scratch!` once it no longer
reads the outputs (the gRPC handler does, after encoding the response). Without it, an output that
aliases a slot is copied before the slot is released.
"""
function infer(
        s::Scheduler, req::InferRequest; committed::Bool = false, cancelled = _CANCEL_PROBE[],
        scratch_out::Union{Vector{PoolSlot}, Nothing} = nothing
    )
    # Under the lock: the dispatch loop and the watcher insert and remove entries concurrently.
    entry, meta = lock(s.cond) do
        e = get(s.registry.by_name, req.model_name, nothing)
        return (e, e === nothing ? get(s.registry.meta, req.model_name, nothing) : nothing)
    end
    if entry === nothing
        # A meta has no executable: it runs its orchestration on this task under the meta gate, and its
        # sub-calls re-enter `infer` (committed) for the sub-models. A meta is never itself a committed
        # sub-call (metas cannot call metas), so `committed` is irrelevant on this branch.
        (meta === nothing || meta.sched === nothing) && throw(ModelNotFoundError(req.model_name))
        return Base.ScopedValues.with(
            () -> _run_meta_request(s, meta, req; scratch_out = scratch_out), _CANCEL_PROBE => cancelled
        )
    end
    entry.sched === nothing && throw(ModelNotFoundError(req.model_name))
    # A call whose client is already gone is neither preprocessed nor queued.
    _is_cancelled(cancelled) && throw(RequestCancelledError(req.model_name))
    # preprocess/postprocess come from a bundle's model.jl, defined in a newer world age;
    # invokelatest crosses that boundary (harmless for identity).
    prepared = Base.invokelatest(entry.preprocess, req.inputs)
    _check_variant_inputs(entry, prepared)
    # A request larger than the largest compiled batch is split into pieces the model can run and
    # the pieces' outputs are joined back, so a client that batches 32 against a bundle compiled for
    # `[1]` gets an answer rather than an INTERNAL error. The inverse of coalescing: the same batch
    # axes, the same slicing, run in the other direction. The split cuts the preprocessed inputs and
    # postprocess sees the joined outputs once, so a hook that changes the row count (one long text
    # into several windows) splits like any other request.
    maxB = _max_compiled_rows(entry, prepared)
    rows = _executable_rows(entry, prepared)
    if maxB !== nothing && rows > maxB
        return _infer_split(s, entry, req, prepared, rows, maxB; committed, cancelled)
    end
    qr = QueuedRequest(req, prepared; committed = committed)
    submit!(s, qr; expected = entry)        # only onto the version whose preprocess ran
    raw = only(_await_replies(s, [qr], cancelled))
    raw isa Exception && throw(raw)
    return Base.invokelatest(entry.postprocess, raw)
end

# ---------------------------------------------------------------------------------------------
# Oversize requests: split, run, join
# ---------------------------------------------------------------------------------------------
#
# The largest compiled batch for this request's shape variant, or `nothing` when the variant is
# uncompiled (then the ordinary path raises the precise error) or the model is unbatched (key 0).
# Only a coalescable model splits: every executable output then carries a batch axis, so the join
# has an axis to join along rather than silently taking an unbatched output from the first piece.
# A request whose batched inputs disagree on their batch extent is never split either: there is no
# one row count to cut every input by, so it runs whole and alone (see `_batch_extents_agree`).
function _max_compiled_rows(entry::ModelEntry, prepared::Vector{NamedTensor})
    _coalescable(entry) || return nothing
    _batch_extents_agree(entry, prepared) || return nothing
    execs = entry.executable.execs
    variant = _request_variant(entry, prepared)
    haskey(execs, variant) || return nothing
    inner = execs[variant]
    haskey(inner, 0) && return nothing
    isempty(inner) && return nothing
    return maximum(keys(inner))
end

# `[1:B, B+1:2B, ...]` covering `rows`, each piece at most `maxB` rows.
_split_ranges(rows::Int, maxB::Int) = [i:min(i + maxB - 1, rows) for i in 1:maxB:rows]

# The `rng` rows of every tensor that has a batch axis in `specs`; tensors without one are shared.
function _take_rows(tensors::AbstractVector{NamedTensor}, specs, rng::UnitRange{Int})
    axes = Dict{String, Int}(sp.name => sp.batch_axis for sp in specs if sp.batch_axis !== nothing)
    return NamedTensor[
        haskey(axes, t.name) ? NamedTensor(t.name, collect(selectdim(t.data, axes[t.name], rng))) : t
            for t in tensors
    ]
end

# Join the pieces' outputs along each output's batch axis, in piece order.
function _join_outputs(entry::ModelEntry, pieces::Vector{Vector{NamedTensor}})
    specs = entry.manifest.executable_outputs
    first_piece = pieces[1]
    joined = NamedTensor[]
    for (i, t) in enumerate(first_piece)
        sp = i <= length(specs) ? specs[i] : nothing
        if sp === nothing || sp.batch_axis === nothing
            push!(joined, t)
        else
            data = cat((pc[i].data for pc in pieces)...; dims = sp.batch_axis)
            push!(joined, NamedTensor(t.name, data))
        end
    end
    return joined
end

function _infer_split(
        s::Scheduler, entry::ModelEntry, req::InferRequest, prepared::Vector{NamedTensor},
        rows::Int, maxB::Int; committed::Bool, cancelled = nothing
    )
    in_specs = entry.manifest.executable_inputs
    ranges = _split_ranges(rows, maxB)
    # Submit every piece before waiting on any, so the pieces coalesce with each other and with
    # other traffic exactly as separate requests would; the deadline and the requested outputs are
    # the whole request's.
    qrs = QueuedRequest[]
    for rng in ranges
        piece = _take_rows(prepared, in_specs, rng)
        qr = QueuedRequest(req, piece; committed = committed)
        submit!(s, qr; expected = entry)    # every piece onto the version that preprocessed them
        push!(qrs, qr)
    end
    replies = _await_replies(s, qrs, cancelled)
    raws = Vector{Vector{NamedTensor}}(undef, length(qrs))
    failure = nothing
    for (i, r) in enumerate(replies)
        if r isa Exception
            failure === nothing && (failure = r)
        else
            raws[i] = r
        end
    end
    failure === nothing || throw(failure)
    return Base.invokelatest(entry.postprocess, _join_outputs(entry, raws))
end

# ---------------------------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------------------------

function _percentile(xs::Vector{Float64}, q::Float64)
    isempty(xs) && return 0.0
    s = sort(xs)
    idx = clamp(ceil(Int, q * length(s)), 1, length(s))
    return s[idx]
end

"""
    scheduler_metrics(scheduler) -> Dict{String,NamedTuple}

Snapshot per-model observability: dispatch count, total compute consumed, current recent-compute
EMA, queue-wait P50/P99, the histogram of dispatch batch sizes, and residency.
"""
function scheduler_metrics(s::Scheduler)
    return lock(s.cond) do
        now = time()
        return Dict(
            entry.name => (
                dispatch_count = entry.sched.dispatch_count,
                requests_served = entry.sched.requests_served,
                rows_served = entry.sched.rows_served,
                total_compute = entry.sched.total_compute,
                recent_compute_ema = _decay_ema!(entry.sched, s.cfg.ema_halflife_seconds, now),
                queue_depth = length(entry.sched.queue),
                wait_p50 = _percentile(entry.sched.wait_samples, 0.5),
                wait_p99 = _percentile(entry.sched.wait_samples, 0.99),
                batch_size_hist = copy(entry.sched.batch_size_hist),
                max_batch_size = _effective_max_batch(entry),
                state = entry.executable.state,
                pinned = is_device_pinned(entry.executable),
                resident = entry.executable.weights !== nothing,
                weight_nbytes = entry.executable.nbytes,
                host_stage = entry.sched.host_stage,
            ) for entry in values(s.registry.by_name) if entry.sched !== nothing && entry.executable !== nothing
        )
    end
end

# The wire-facing batch axis as ("input name", 1-based axis), or ("", 0) when the model declares
# none. Reported over the control plane so a gateway can size a request in items rather than
# guessing at a position that genuinely varies between bundles. See `wire_batch_spec`.
function _wire_batch(entry::ModelEntry)
    spec = ReactantServerCore.wire_batch_spec(entry.manifest)
    return spec === nothing ? ("", 0) : (spec[1], spec[2])
end

# The largest compiled batch shape a model can serve, limited by its configured coalescing cap;
# 0 when no batched shape is compiled. Reported over the control plane as routing metadata.
function _effective_max_batch(entry::ModelEntry)
    largest = 0
    for inner in values(entry.executable.execs), k in keys(inner)
        k > largest && (largest = k)
    end
    cap = entry.sched.max_batch_size
    return cap === nothing ? largest : min(Int(cap), largest)
end

# A meta's control-plane status: its OWN serving counters and queue, but a weight footprint and
# residency AGGREGATED over its sub-models (the meta owns no weights). The gateway packs this as one
# unit; internal sub-models are not reported separately. A meta is non-coalescable (max batch 0).
function _meta_group_status(s::Scheduler, meta::MetaEntry)
    nbytes = 0
    dev = true
    host = true
    floor = PINNED_DEVICE        # least-resident state across subs; start at the most-resident
    any_sub = false
    for sub in meta.calls
        e = get(s.registry.by_name, sub, nothing)
        (e === nothing || e.executable === nothing) && continue
        any_sub = true
        nbytes += e.executable.nbytes
        dev &= (e.executable.weights !== nothing)
        host &= (e.executable.host_weights !== nothing)
        Integer(e.executable.state) < Integer(floor) && (floor = e.executable.state)
    end
    any_sub || (dev = false; host = false; floor = UNPINNED)   # compute-only meta: no weights
    return (
        state = floor, device_resident = dev, host_resident = host, weight_nbytes = nbytes,
        weight = meta.sched.weight, queue_depth = length(meta.sched.queue),
        total_compute = meta.sched.total_compute, requests_served = meta.sched.requests_served,
        rows_served = meta.sched.rows_served,
        dispatch_count = meta.sched.dispatch_count, max_batch_size = 0,
        # A meta is non-coalescable, so it declares no wire batch axis: one item per request.
        # `rows_served` is therefore incremented once per request too, which makes a meta's cost
        # per item equal its cost per request. That is the honest figure for it.
        batch_input_name = "", batch_axis = 0,
    )
end

"""
    describe_models(scheduler, names) -> Vector{NamedTuple}

What the control plane's DescribeModels reports, as `(name, manifest, max_batch_size)` per model,
sorted by name. An empty `names` describes every model requests are routed to, which leaves out the
internal sub-models a meta model calls; naming one describes it. Throws naming every model the
worker does not serve, which the handler maps to NOT_FOUND. `max_batch_size` is the effective
coalescing limit (0 for a meta model, and for a model not yet compiled).
"""
function describe_models(s::Scheduler, names::AbstractVector{<:AbstractString})
    return lock(s.cond) do
        reg = s.registry
        wanted = isempty(names) ? routable_model_names(reg) : sort!(unique(String.(names)))
        unknown = [n for n in wanted if get_model(reg, n) === nothing && get_meta(reg, n) === nothing]
        isempty(unknown) || throw(ModelNotFoundError(join(unknown, ", "), "unknown model: $(join(unknown, ", "))"))
        return map(wanted) do n
            e = get_model(reg, n)
            e === nothing && return (name = n, manifest = get_meta(reg, n).manifest, max_batch_size = 0)
            compiled = e.sched !== nothing && e.executable !== nothing
            return (name = n, manifest = e.manifest, max_batch_size = compiled ? _effective_max_batch(e) : 0)
        end
    end
end

"""
    load_costs(scheduler, names) -> Dict{String, NamedTuple}

What loading each named model (a meta: its sub-models together) onto the device costs, for a
gateway pricing placement changes: whether a host copy exists to upload from, the upload's device
seconds, the weights-file read's seconds (off the device), and the model's device loads so far (not
counting the startup probe's). Empty on a worker with no on-demand weight cache. Called outside
`s.cond`: `has_host_copy` stats the shared store.
"""
function load_costs(s::Scheduler, names)
    wc = s.weight_cache
    out = Dict{String, NamedTuple{(:host_copy, :upload_seconds, :read_seconds, :loads), Tuple{Bool, Float64, Float64, Int}}}()
    wc === nothing && return out
    for name in names
        entries = lock(s.cond) do
            meta = get(s.registry.meta, name, nothing)
            meta === nothing && return ModelEntry[e for e in (get(s.registry.by_name, name, nothing),) if e !== nothing]
            return ModelEntry[s.registry.by_name[c] for c in meta.calls if haskey(s.registry.by_name, c)]
        end
        host, up, rd, loads = true, 0.0, 0.0, 0
        any_weights = false
        for e in entries
            e.executable === nothing && continue
            any_weights = true
            fp = model_footprint(wc, e)
            up += reload_seconds(wc, fp)
            rd += read_seconds(wc, fp)
            loads += fp.loads - fp.probe_loads
            host &= has_host_copy(wc, e)
        end
        out[name] = (host_copy = any_weights && host, upload_seconds = up, read_seconds = rd, loads = loads)
    end
    return out
end

"""
    control_status(scheduler) -> NamedTuple

A control-plane snapshot of the worker: its residency mode and scheduling discipline, plus a
per-model view. A meta is reported as a single model whose footprint is the sum of its sub-models'
weights; the internal sub-models are folded into it and not reported on their own, so the gateway
packs and routes the group as one unit and never sees the stages.
"""
function control_status(s::Scheduler)
    # Host-cache figures first, outside `s.cond`: with the shared store they read /dev/shm.
    hb = s.weight_cache === nothing ? nothing : host_budget(s.weight_cache)
    hcached = s.weight_cache === nothing ? 0 : host_cached_bytes(s.weight_cache)
    return lock(s.cond) do
        mode = s.weight_cache === nothing ? SELF_MANAGED : s.weight_cache.mode
        subs = internal_submodels(s.registry)
        models = Dict{String, Any}()
        for entry in values(s.registry.by_name)
            (entry.sched === nothing || entry.executable === nothing) && continue
            entry.name in subs && continue   # internal stage of a meta: folded into the meta, hidden here
            models[entry.name] = (
                state = entry.executable.state,
                device_resident = entry.executable.weights !== nothing,
                host_resident = entry.executable.host_weights !== nothing,
                weight_nbytes = entry.executable.nbytes,
                weight = entry.sched.weight,
                queue_depth = length(entry.sched.queue),
                # Cumulative serving counters: a gateway derives true per-request compute cost from
                # deltas of total_compute / requests_served between polls (lpt_packing scheduling),
                # and cost per ITEM from deltas of total_compute / rows_served, which is the unit a
                # gateway needs to weigh in-flight work when clients pre-batch.
                total_compute = entry.sched.total_compute,
                requests_served = entry.sched.requests_served,
                rows_served = entry.sched.rows_served,
                dispatch_count = entry.sched.dispatch_count,
                # Effective max batch the worker coalesces to, capped by max_batch_size.
                max_batch_size = _effective_max_batch(entry),
                # How a gateway counts the items in a request without decoding it: the name of the
                # wire-facing input carrying the batch axis and that tensor's 1-based axis. The
                # position varies per bundle and is not guessable from the request alone.
                batch_input_name = _wire_batch(entry)[1],
                batch_axis = _wire_batch(entry)[2],
            )
        end
        for meta in values(s.registry.meta)
            meta.sched === nothing && continue
            models[meta.name] = _meta_group_status(s, meta)
        end
        # The on-demand weight budget is the memory capacity a gateway packs weight footprints
        # against; 0 (cache disabled, all weights resident) means memory is not a constraint.
        cache_max = s.weight_cache === nothing ? 0 : s.weight_cache.max_bytes
        return (
            residency_mode = mode, discipline = s.cfg.discipline, models = models,
            weight_cache_max_bytes = cache_max,
            host_weight_budget_bytes = hb === nothing ? 0 : hb,
            host_cached_bytes = hcached,
        )
    end
end

"""
    weight_cache_metrics(scheduler) -> Union{NamedTuple,Nothing}

Snapshot the on-demand weight cache (resident bytes/budget, currently resident models, and
load/evict counters) and the weight reads that stalled (`stalled_reads`, abandoned so far, and
`stalled_reads_outstanding`, abandoned and not yet returned), or `nothing` when on-demand loading
is disabled.
"""
function weight_cache_metrics(s::Scheduler)
    s.weight_cache === nothing && return nothing
    reads = lock(s.cond) do
        (stalled_reads = s.stalled_reads, stalled_reads_outstanding = count(r -> r.stalled, s.reads))
    end
    return merge(weight_cache_stats(s.weight_cache), reads)
end
