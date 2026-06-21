# The scheduler: a deficit-weighted, cost-aware, coalescing dispatch policy.
#
# Requests arrive concurrently and are placed in per-model queues. Whenever the GPU is free
# the scheduler decides which model to dispatch next, coalesces that model's queued requests
# into a single execution at a compiled batch size, runs it, and splits the outputs back to
# each caller. The policy layer rides on the language task scheduler; it does not implement
# its own threading. Only one execution runs at a time by design.
#
# Decision order each time the GPU frees up:
#   1. Score every model with a non-empty queue by a deficit-weighted, cost-aware
#      priority and pick the highest (or, under the FIFO discipline, pick the model
#      with the oldest queued request).
#   2. Coalesce the selected model's queued requests (FIFO) into one compiled-batch-size
#      dispatch; the remainder stays queued.
#
# Per-model latency budgets (earliest-deadline-first escalation) are a planned extension;
# they are not implemented.
#
# A single Threads.Condition guards all mutable scheduler state (the per-model queues and the
# EMA/cost fields) and signals the dispatch loop when work arrives. The loop holds the lock
# only to select and dequeue; the actual execution runs outside the lock so producers keep
# enqueuing during GPU work, then the lock is re-taken briefly to record the dispatch.

# Effective cost used for scoring when a (model, batch size) pair has no measurement yet. The
# absolute value does not matter: while every candidate is unseeded they share it, so the
# deficit term decides. Warmup normally seeds real values before traffic arrives.
const _DEFAULT_COST = 0.01            # seconds
const _WAIT_SAMPLE_CAP = 1024         # bounded ring of recent queue-wait samples per model

# `ModelSchedState` (the per-model scheduling state) is defined in `runtime/model_types.jl` so
# that `ModelEntry` can hold it; its methods live here.

# A control command runs on the dispatch-loop thread (the sole mutator of residency). `op` takes
# the scheduler and returns a result; the result (or any thrown error) is delivered on `reply`.
struct ControlCommand
    op::Function
    reply::Channel{Any}
end

# Admits at most `capacity` meta orchestrations at once on this worker (default 1). A meta holds a
# permit across its WHOLE run, including the CPU glue between stages, so only one meta's GPU sub-calls
# ever compete with regular work; the GPU itself is not held during glue (sub-calls dispatch on the
# loop). `acquire_meta_gate!` is deadline-bounded: a meta that cannot get a permit before its deadline
# is shed before doing any GPU work, so a meta backlog sheds at admission rather than piling up.
mutable struct MetaGate
    cond::Threads.Condition   # guards `n_held`
    capacity::Int
    n_held::Int
end
MetaGate(capacity::Integer=1) = MetaGate(Threads.Condition(), Int(capacity), 0)

function acquire_meta_gate!(g::MetaGate, who::AbstractString; deadline_ns::Integer=0)
    dl = Int64(deadline_ns)
    lock(g.cond)
    timer = nothing   # one-shot wake at the deadline; armed on first park, closed in finally
    try
        while g.n_held >= g.capacity
            dl != 0 && Int64(time_ns()) >= dl && throw(DeadlineExceeded(String(who)))
            if dl != 0 && timer === nothing
                rem = (dl - Int64(time_ns())) / 1e9
                rem <= 0 && throw(DeadlineExceeded(String(who)))
                timer = Timer(_ -> (try; @lock g.cond notify(g.cond); catch; end), rem)
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

"""
    Scheduler

The deficit-weighted, cost-aware, coalescing dispatch engine. Concurrent
requests land on per-model queues; a single dispatch loop runs one GPU execution at a time,
coalescing same-model requests into one batched execution at a compiled size and sharing GPU
time by relative model weight and a learned per-batch-size cost estimate. It holds the
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
    task::Union{Task,Nothing}
    weight_cache::Union{WeightCache,Nothing}   # on-demand weight residency, or nothing when disabled
    scratch_pool::Union{BufferPool,Nothing}    # local reuse pool for meta intermediates (plain Memory); nothing disables
    control::Vector{ControlCommand}            # pending control commands, drained by the dispatch loop
    meta_gate::MetaGate                        # admits one meta orchestration at a time (capacity configurable)
    committed::Union{QueuedRequest,Nothing}    # an in-flight meta's continuation sub-call awaiting the next GPU slot;
                                               # selected ahead of the discipline scan. At most one (one meta at a time).
end

function Scheduler(registry::ModelRegistry, backend::AbstractBackend, pool::MemoryPool, cfg::SchedulerConfig)
    return Scheduler(registry, backend, pool, cfg,
                     Threads.Condition(), false, nothing, nothing, nothing, ControlCommand[],
                     MetaGate(1), nothing)
end

# A selected dispatch: the model entry, the chosen executable key (batch size; 0 = unbatched
# single module), and the requests coalesced into this execution (taken FIFO from the front).
struct Dispatch
    entry::ModelEntry
    size::Int
    taken::Vector{QueuedRequest}
end


# ---------------------------------------------------------------------------------------------
# EMA and cost-estimate updates
# ---------------------------------------------------------------------------------------------

# Decay the recent-compute EMA to `now` by the configured half-life. Applied consistently
# before any read or write so the stored value is always current.
function _decay_ema!(st::ModelSchedState, halflife::Float64, now::Float64)
    dt = now - st.ema_last_update
    if dt > 0
        st.recent_compute_ema *= exp(-dt * log(2) / halflife)
        st.ema_last_update = now
    end
    return st.recent_compute_ema
end

# Refine the per-batch-size cost estimate toward the measured time. Initializes to the first
# measurement so the estimate converges immediately rather than drifting up from zero.
function _update_cost!(st::ModelSchedState, B::Int, measured::Float64, alpha::Float64)
    old = get(st.cost_estimate, B, measured)
    st.cost_estimate[B] = alpha * measured + (1 - alpha) * old
    return nothing
end

effective_cost(st::ModelSchedState, B::Int, discount::Float64) =
    get(st.cost_estimate, B, _DEFAULT_COST) * (1 - discount)

# Deficit-weighted, cost-aware priority. A model that has consumed less than its share recently
# scores higher; dividing by cost stops an expensive model from blocking cheaper ones on a
# marginal fairness edge. The clamp bounds both lockout and domination.
priority(share::Float64, normalized_ema::Float64, cap::Float64, eff_cost::Float64) =
    clamp(share - normalized_ema, -cap, cap) / eff_cost

# ---------------------------------------------------------------------------------------------
# Batch contribution and coalescing plan
# ---------------------------------------------------------------------------------------------

# Rows a request contributes along its batch axis, read from the client-facing input spec
# (what the caller sent). Models with no batch axis contribute a single row.
function _request_rows(entry::ModelEntry, req::InferRequest)
    for sp in client_input_spec(entry.manifest)
        sp.batch_axis === nothing && continue
        for t in req.inputs
            t.name == sp.name && return size(t.data, sp.batch_axis)
        end
    end
    return 1
end

# Rows present along the executable batch axis after preprocessing. Used to size output slices.
function _executable_rows(entry::ModelEntry, inputs::AbstractVector{NamedTensor})
    m = entry.manifest
    m.input_batch_dim === nothing && return 1
    axis = m.input_batch_dim + 1
    for sp in m.executable_inputs
        sp.batch_axis === nothing && continue
        for t in inputs
            t.name == sp.name && return size(t.data, axis)
        end
    end
    return 1
end

# A model can coalesce multiple requests only when its inputs carry a batch axis and every
# output does too (so outputs can be split per request). Otherwise it serves one request per
# dispatch. Single unbatched modules (batch key 0) are never coalesced.
function _coalescable(entry::ModelEntry)
    m = entry.manifest
    m.input_batch_dim === nothing && return false
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
# dispatches alone. For non-coalescable models exactly one request is taken at its own size.
function plan_batch(entry::ModelEntry, st::ModelSchedState)
    isempty(st.queue) && return nothing
    front = st.queue[1]
    execs = entry.executable.execs

    # Pick the variant from the front request and only ever coalesce requests sharing it: different
    # input shapes cannot be concatenated along the batch axis. An uncompiled shape dispatches alone
    # so run_model raises the precise "no compiled program for input shape" error for that caller.
    variant = _request_variant(entry, front.prepared)
    haskey(execs, variant) || return (_request_rows(entry, front.req), QueuedRequest[front])
    inner = execs[variant]
    sizes = sort!(collect(keys(inner)))

    if !_coalescable(entry) || haskey(inner, 0)
        key = haskey(inner, 0) ? 0 :
              (_smallest_ge(sizes, _request_rows(entry, front.req)) === nothing ? maximum(sizes) :
               _smallest_ge(sizes, _request_rows(entry, front.req)))
        return (key, QueuedRequest[front])
    end

    # The leading run of same-variant requests is the only window we may coalesce within (keeps
    # `taken` a contiguous front prefix so _finalize can deleteat! it, and preserves FIFO order).
    nprefix = 0
    for qr in st.queue
        _request_variant(entry, qr.prepared) == variant || break
        nprefix += 1
    end
    prefix = view(st.queue, 1:nprefix)

    cap = st.max_batch_size === nothing ? typemax(Int) : st.max_batch_size
    minB, maxB = minimum(sizes), maximum(sizes)
    R = min(sum(_request_rows(entry, qr.req) for qr in prefix), cap)
    # Largest compiled size that can be filled, else the smallest size for a partial fill.
    B = if R >= minB
        candidates = filter(<=(min(R, maxB)), sizes)
        isempty(candidates) ? minB : maximum(candidates)
    else
        minB
    end
    # A single request may itself exceed B; grow B to the smallest size that fits it.
    front_rows = _request_rows(entry, front.req)
    if B < front_rows
        grown = _smallest_ge(sizes, front_rows)
        B = grown === nothing ? maxB : grown
    end

    taken = QueuedRequest[]
    acc = 0
    for qr in prefix
        r = _request_rows(entry, qr.req)
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
    entry.sched !== nothing && entry.executable !== nothing && !isempty(entry.sched.queue)

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
    if s.committed !== nothing
        qr = s.committed
        s.committed = nothing
        # The committed request was pushed to the front of its sub-model's queue. Dispatch that model
        # now (plan_batch draws from the front, so the committed request leads the batch). If the model
        # is gone or already drained, fall through to a normal selection.
        entry = get(s.registry.by_name, qr.req.model_name, nothing)
        entry !== nothing && _schedulable(entry) && return _finalize(entry, plan_batch(entry, entry.sched))
    end
    s.cfg.discipline == FIFO && return _select_fifo!(s)
    s.cfg.discipline == EDF && return _select_edf!(s)
    return _select_fair!(s, now)
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
    return _finalize(chosen, plan_batch(chosen, chosen.sched))
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
    return _finalize(chosen, plan_batch(chosen, chosen.sched))
end

# Fair: deficit-weighted, cost-aware priority. Decay every EMA to now before reading, then normalize.
# (Metas are not scheduled here; they run under the meta gate and their sub-calls compete as ordinary
# requests for the sub-models.)
function _select_fair!(s::Scheduler, now::Float64)
    for entry in values(s.registry.by_name)
        entry.sched === nothing || _decay_ema!(entry.sched, s.cfg.ema_halflife_seconds, now)
    end
    total_ema = sum(e.sched.recent_compute_ema for e in values(s.registry.by_name) if e.sched !== nothing; init=0.0)
    total_w = sum(e.sched.weight for e in values(s.registry.by_name) if e.sched !== nothing; init=0.0)

    chosen = nothing
    chosen_plan = nothing
    best_p = -Inf
    for entry in values(s.registry.by_name)
        _schedulable(entry) || continue
        st = entry.sched
        plan = plan_batch(entry, st)
        plan === nothing && continue
        B = plan[1]
        share = total_w == 0 ? 0.0 : st.weight / total_w
        # All-zero case: every model normalizes to zero, so the deficit equals the share and
        # quiet or newly loaded models stay schedulable.
        nema = total_ema == 0 ? 0.0 : st.recent_compute_ema / total_ema
        p = priority(share, nema, s.cfg.recency_penalty_cap, effective_cost(st, B, s.cfg.coalescing_discount))
        if chosen === nothing || p > best_p ||
           (p == best_p && st.queue[1].enqueued_at < chosen.sched.queue[1].enqueued_at) ||
           (p == best_p && st.queue[1].enqueued_at == chosen.sched.queue[1].enqueued_at && entry.name < chosen.name)
            chosen, chosen_plan, best_p = entry, plan, p
        end
    end
    chosen === nothing && return nothing
    return _finalize(chosen, chosen_plan)
end

# ---------------------------------------------------------------------------------------------
# Coalescing the tensors and splitting the outputs
# ---------------------------------------------------------------------------------------------

# Concatenate the per-request executable inputs along the batch axis into one set sized to B,
# padding with zeros when the real rows fall short of B. Inputs without a batch axis are taken
# from the first request (they do not vary across the batch).
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
        data = length(arrays) == 1 ? arrays[1] : cat(arrays...; dims=axis)
        if total_rows < B
            padshape = collect(size(data))
            padshape[axis] = B - total_rows
            data = cat(data, zeros(eltype(data), padshape...); dims=axis)
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

function _record_wait!(st::ModelSchedState, now::Float64, taken::Vector{QueuedRequest})
    for qr in taken
        push!(st.wait_samples, now - qr.enqueued_at)
    end
    while length(st.wait_samples) > _WAIT_SAMPLE_CAP
        popfirst!(st.wait_samples)
    end
    return nothing
end

# Run the coalesced dispatch outside the lock, deliver per-request results, then take the lock
# briefly to record compute time once for the whole dispatch (never once per request).
function execute_and_record!(s::Scheduler, d::Dispatch)
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
    # dispatches; unseeded it defaults small, so the laxity drop is a no-op until the model has run once.
    now_ns = Int64(time_ns())
    laxity_ns = s.cfg.discipline == EDF ?
        round(Int64, get(st.cost_estimate, B, _DEFAULT_COST) * 1e9) : Int64(0)
    live = QueuedRequest[]
    for qr in taken
        # A committed request is an in-flight meta's continuation: skip the laxity drop (we already
        # spent GPU on its earlier stages, so we do not abandon it on a prediction), but still honor
        # the base deadline-passed check, so a meta that has genuinely run out of time stops here.
        margin_ns = qr.committed ? Int64(0) : laxity_ns
        if qr.req.deadline_ns != 0 && now_ns + margin_ns >= qr.req.deadline_ns
            put!(qr.reply, DeadlineExceeded(qr.req.model_name))
        else
            push!(live, qr)
        end
    end
    if length(live) != length(taken)
        taken = live          # the catch below and the slicing loop operate only on live requests
        isempty(taken) && return nothing
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
        # cost estimate, so the estimate reflects steady-state (resident) execution.
        s.weight_cache === nothing || acquire!(s.weight_cache, entry)
        t0 = time()
        out = run_model(s.backend, s.pool, entry.executable, merged)
        compute_time = time() - t0

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
            _update_cost!(st, B, compute_time, s.cfg.cost_ema_alpha)
            st.dispatch_count += 1
            st.requests_served += length(taken)
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
function _run_meta_request(s::Scheduler, meta::MetaEntry, req::InferRequest)
    # Base deadline check first: no point starting already-expired work (no GPU spent yet).
    req.deadline_ns != 0 && Int64(time_ns()) >= req.deadline_ns && throw(DeadlineExceeded(meta.name))
    gated = !isempty(meta.calls)   # compute-only metas (no sub-calls) skip the gate
    gated && acquire_meta_gate!(s.meta_gate, meta.name; deadline_ns=req.deadline_ns)
    caller = QueueingCaller(s, s.scratch_pool)
    call_ns = Ref(Int64(0))
    try
        out = run_meta(meta, caller, req.inputs; deadline_ns=req.deadline_ns, call_ns_out=call_ns)
        compute = call_ns[] / 1e9     # sub-call (model) time only; the Julia glue between stages is excluded
        st = meta.sched
        lock(s.cond) do
            now = time()
            _decay_ema!(st, s.cfg.ema_halflife_seconds, now)
            st.recent_compute_ema += compute
            _update_cost!(st, 0, compute, s.cfg.cost_ema_alpha)
            st.dispatch_count += 1
            st.requests_served += 1
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

# Prepare every registered entry's scheduling state, seed cost estimates from a warmup pass, then
# spawn the dispatch loop. Entries are already in the registry (loaded and compiled by `serve`).
function start!(s::Scheduler)
    now = time()
    for entry in values(s.registry.by_name)
        init_sched_state!(s, entry, now)
    end
    for entry in values(s.registry.meta)   # metas are dispatched too, so they get sched state as well
        init_sched_state!(s, entry, now)
    end
    s.weight_cache === nothing || preload_pinned!(s.weight_cache, s.registry)
    # Cost learning only feeds the fair discipline; FIFO needs no per-batch-size estimates.
    if s.cfg.discipline == FAIR
        for entry in values(s.registry.by_name)
            _warmup_entry!(s, entry)
        end
    end
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

function shutdown!(s::Scheduler)
    lock(s.cond) do
        s.running = false
        notify(s.cond)
    end
    return nothing
end

# Reject everything still pending at shutdown so no caller stays blocked on its reply channel:
# queued requests across all models and any pending control commands. Caller holds `s.cond`.
function _reject_pending_locked!(s::Scheduler)
    err = ErrorException("server is shutting down")
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
    for c in s.control
        put!(c.reply, err)
    end
    empty!(s.control)
    return nothing
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

# Seed one entry's cost_estimate by running each compiled size once. Failures are non-fatal: the
# default cost covers any unseeded pair until the first real dispatch measures it.
function _warmup_entry!(s::Scheduler, entry::ModelEntry)
    (entry.executable === nothing || entry.sched === nothing) && return nothing
    isempty(entry.manifest.executable_inputs) && return nothing   # cannot synthesize inputs
    for (variant, inner) in entry.executable.execs
        for sz in keys(inner)
            inputs = _zero_inputs(entry, variant, sz)
            inputs === nothing && continue
            try
                s.weight_cache === nothing || acquire!(s.weight_cache, entry)
                t0 = time()
                run_model(s.backend, s.pool, entry.executable, inputs)
                entry.sched.cost_estimate[sz] = max(time() - t0, eps())
            catch err
                @warn "scheduler cost warmup failed; using default cost" model = entry.name variant = variant size = sz exception = err
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Submission and the dispatch loop
# ---------------------------------------------------------------------------------------------

# Enqueue a request onto its model's queue and wake the dispatch loop. Unknown models, a
# stopped scheduler, and a full queue are reported immediately through the reply channel
# (the caller sees an exception). `max_queue_depth` caps each model's queue independently,
# so one backlogged model cannot starve admission for the others.
function submit!(s::Scheduler, qr::QueuedRequest)
    lock(s.cond) do
        if !s.running
            put!(qr.reply, ErrorException("server is shutting down"))
            return
        end
        # Targets are regular models only: a meta runs on the request task (see `infer`), and its
        # GPU sub-calls arrive here as committed requests for the sub-models, which live in `by_name`.
        entry = get(s.registry.by_name, qr.req.model_name, nothing)
        if entry === nothing || entry.sched === nothing
            put!(qr.reply, ErrorException("unknown model: $(qr.req.model_name)"))
            return
        end
        if length(entry.sched.queue) >= s.cfg.max_queue_depth
            put!(qr.reply, ErrorException("queue for model '$(qr.req.model_name)' is full ($(s.cfg.max_queue_depth)); request rejected"))
            return
        end
        if qr.committed
            # An in-flight meta's continuation jumps the line: push to the front of its sub-model's
            # queue and flag it so `select_dispatch!` dispatches this model next, ahead of the
            # discipline scan. Only one meta runs at a time, so at most one committed request exists.
            pushfirst!(entry.sched.queue, qr)
            s.committed = qr
        else
            push!(entry.sched.queue, qr)
        end
        notify(s.cond)
    end
    return nothing
end

function dispatch_loop(s::Scheduler)
    while true
        cmds, chosen = lock(s.cond) do
            while s.running && isempty(s.control) && !_has_queued(s)
                wait(s.cond)
            end
            if !s.running
                _reject_pending_locked!(s)
                return (ControlCommand[], nothing)
            end
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
        end
        (isempty(cmds) && chosen === nothing) && break
        chosen === nothing || execute_and_record!(s, chosen)
    end
    return nothing
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
        s.running || throw(ErrorException("server is shutting down"))
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
until applied. Raises on an unknown model or in self-managed mode.
"""
function set_residency!(s::Scheduler, name::AbstractString, target::ResidencyState)
    return _run_control(s, function (sch)
        sch.weight_cache === nothing &&
            throw(ErrorException("residency control requires the on-demand weight cache (set runtime.weight_cache_bytes > 0)"))
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
                if target == UNPINNED && any(other.name != nm && sub in other.calls
                                             for other in values(sch.registry.meta))
                    continue
                end
                last = set_residency_state!(sch.weight_cache, e, target)
            end
            return last
        end
        entry = get(sch.registry.by_name, nm, nothing)
        entry === nothing && throw(ErrorException("unknown model: $name"))
        return set_residency_state!(sch.weight_cache, entry, target)
    end)
end

"""
    set_policy!(scheduler, name; weight=nothing) -> nothing

Update a model's live scheduler policy. `weight` is consulted only by the fair discipline.
Available in both residency modes. Raises on an unknown model.
"""
function set_policy!(s::Scheduler, name::AbstractString; weight::Union{Real,Nothing}=nothing)
    lock(s.cond) do
        entry = get(s.registry.by_name, String(name), nothing)
        (entry === nothing || entry.sched === nothing) && throw(ErrorException("unknown model: $name"))
        weight === nothing || (entry.sched.weight = Float64(weight))
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
# thread (warmup executes the model). Raises on a duplicate name. Returns the model name.
function _admit_entry!(sch::Scheduler, entry::ModelEntry)
    lock(sch.cond) do
        haskey(sch.registry.by_name, entry.name) &&
            throw(ErrorException("model '$(entry.name)' is already registered"))
        sch.registry.by_name[entry.name] = entry
        init_sched_state!(sch, entry, time())
    end
    # Warmup runs the model on this (the dispatch) thread, outside the lock.
    sch.cfg.discipline == FAIR && _warmup_entry!(sch, entry)
    return entry.name
end

# Remove a model from the live registry, rejecting any queued requests, and release its residency.
# Runs on the dispatch thread. Returns the removed entry, or `nothing` if it was not registered.
function _evict_entry!(sch::Scheduler, name::AbstractString)
    entry = lock(sch.cond) do
        e = pop!(sch.registry.by_name, String(name), nothing)
        if e !== nothing && e.sched !== nothing
            for qr in e.sched.queue
                put!(qr.reply, ErrorException("model '$name' was unloaded"))
            end
            empty!(e.sched.queue)
        end
        return e
    end
    entry === nothing && return nothing
    # Release residency outside the lock (device frees can be slow).
    sch.weight_cache === nothing || release_all!(sch.weight_cache, entry)
    nbytes = entry.executable === nothing ? 0 : entry.executable.nbytes
    log_model_unloaded(String(name), nbytes;
        memory=memory_report(sch.backend, sch.pool; registry=sch.registry, weight_cache=sch.weight_cache))
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
    return _run_control(s, sch -> _admit_entry!(sch, entry))
end

"""
    evict!(scheduler, name) -> Union{ModelEntry,Nothing}

Remove a model from the live system: drop it from the registry, reject any queued requests (so
callers unblock), and release its residency (device buffers and any shared host floor). Returns
the removed entry, or `nothing` if it was not registered.
"""
function evict!(s::Scheduler, name::AbstractString)
    return _run_control(s, sch -> _evict_entry!(sch, String(name)))
end

"""
    load_model!(scheduler, backend, pool, entry; state, on_demand, store) -> String

Compile and admit a freshly loaded (uncompiled) bundle `entry`, replacing any existing model of
the same name. Runs entirely on the dispatch thread inside one control op (a stop-the-world swap):
any existing model is evicted first so its device memory is freed before the new artifact is
compiled, then the new entry is compiled (`build_loaded_model`) and admitted. This is the dynamic
directory watcher's load/reload path. Returns the model name.
"""
function load_model!(s::Scheduler, backend::AbstractBackend, pool::MemoryPool, entry::ModelEntry;
                     state::ResidencyState, on_demand::Bool, store::WeightStore=PrivateWeightStore())
    return _run_control(s, function (sch)
        _evict_entry!(sch, entry.name)          # no-op on a fresh load; frees device memory on reload
        entry.executable = build_loaded_model(backend, pool, entry;
                                              state=state, on_demand=on_demand, store=store, source=:dynamic)
        return _admit_entry!(sch, entry)
    end)
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
                put!(qr.reply, ErrorException("model '$name' was unloaded"))
            end
            empty!(e.sched.queue)
        end
    end
    return nothing
end

"""
    infer(scheduler, request) -> Vector{NamedTensor}

Submit a request and block until the scheduler returns the result. Re-raises any error
captured during dispatch.

Runs the model's `preprocess`/`postprocess` hooks here, on the caller's task, rather than on the
dispatch loop: preprocess before the request is queued, postprocess on the raw device outputs the
loop hands back. Because each gRPC request runs on its own task, many requests' hook work
proceeds in parallel and overlaps the single, serialized GPU execution. The dispatch loop coalesces
and runs `qr.prepared` and never executes a request whose preprocess has not finished, since a
request is only enqueued (made visible to the loop) after preprocess returns.
"""
function infer(s::Scheduler, req::InferRequest; committed::Bool=false)
    entry = get(s.registry.by_name, req.model_name, nothing)
    if entry === nothing
        # A meta has no executable: it runs its orchestration on this task under the meta gate, and its
        # sub-calls re-enter `infer` (committed) for the sub-models. A meta is never itself a committed
        # sub-call (metas cannot call metas), so `committed` is irrelevant on this branch.
        meta = get(s.registry.meta, req.model_name, nothing)
        (meta === nothing || meta.sched === nothing) &&
            throw(ErrorException("unknown model: $(req.model_name)"))
        return _run_meta_request(s, meta, req)
    end
    entry.sched === nothing && throw(ErrorException("unknown model: $(req.model_name)"))
    # preprocess/postprocess come from a bundle's model.jl, defined in a newer world age;
    # invokelatest crosses that boundary (harmless for identity).
    prepared = Base.invokelatest(entry.preprocess, req.inputs)
    qr = QueuedRequest(req, prepared; committed=committed)
    submit!(s, qr)
    raw = take!(qr.reply)
    raw isa Exception && throw(raw)
    return Base.invokelatest(entry.postprocess, raw)
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
    lock(s.cond) do
        now = time()
        return Dict(entry.name => (
            dispatch_count = entry.sched.dispatch_count,
            total_compute = entry.sched.total_compute,
            recent_compute_ema = _decay_ema!(entry.sched, s.cfg.ema_halflife_seconds, now),
            queue_depth = length(entry.sched.queue),
            wait_p50 = _percentile(entry.sched.wait_samples, 0.5),
            wait_p99 = _percentile(entry.sched.wait_samples, 0.99),
            batch_size_hist = copy(entry.sched.batch_size_hist),
            state = entry.executable.state,
            pinned = is_device_pinned(entry.executable),
            resident = entry.executable.weights !== nothing,
            weight_nbytes = entry.executable.nbytes,
        ) for entry in values(s.registry.by_name) if entry.sched !== nothing && entry.executable !== nothing)
    end
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
    return (state = floor, device_resident = dev, host_resident = host, weight_nbytes = nbytes,
            weight = meta.sched.weight, queue_depth = length(meta.sched.queue),
            total_compute = meta.sched.total_compute, requests_served = meta.sched.requests_served,
            dispatch_count = meta.sched.dispatch_count, max_batch_size = 0)
end

"""
    control_status(scheduler) -> NamedTuple

A control-plane snapshot of the worker: its residency mode and scheduling discipline, plus a
per-model view. A meta is reported as a single model whose footprint is the sum of its sub-models'
weights; the internal sub-models are folded into it and not reported on their own, so the gateway
packs and routes the group as one unit and never sees the stages.
"""
function control_status(s::Scheduler)
    lock(s.cond) do
        mode = s.weight_cache === nothing ? SELF_MANAGED : s.weight_cache.mode
        subs = internal_submodels(s.registry)
        models = Dict{String,Any}()
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
                # deltas of total_compute / requests_served between polls (lpt_packing scheduling).
                total_compute = entry.sched.total_compute,
                requests_served = entry.sched.requests_served,
                dispatch_count = entry.sched.dispatch_count,
                # Effective max batch the worker coalesces to, capped by max_batch_size.
                max_batch_size = _effective_max_batch(entry),
            )
        end
        for meta in values(s.registry.meta)
            meta.sched === nothing && continue
            models[meta.name] = _meta_group_status(s, meta)
        end
        # The on-demand weight budget is the memory capacity a gateway packs weight footprints
        # against; 0 (cache disabled, all weights resident) means memory is not a constraint.
        cache_max = s.weight_cache === nothing ? 0 : s.weight_cache.max_bytes
        return (residency_mode = mode, discipline = s.cfg.discipline, models = models,
                weight_cache_max_bytes = cache_max)
    end
end

"""
    weight_cache_metrics(scheduler) -> Union{NamedTuple,Nothing}

Snapshot the on-demand weight cache (resident bytes/budget, currently resident models, and
load/evict counters), or `nothing` when on-demand loading is disabled.
"""
function weight_cache_metrics(s::Scheduler)
    s.weight_cache === nothing && return nothing
    return weight_cache_stats(s.weight_cache)
end
