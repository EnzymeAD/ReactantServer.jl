# Deterministic unit tests for the scheduler policy: priority math, EMA decay, cost learning,
# coalescing batch selection, fairness selection (the spec's worked example), and the FIFO
# discipline. These exercise the pure helpers and select_dispatch! directly, with no spawned
# dispatch loop and no real clock, so they are fully deterministic.

using ReactantServer: ModelSchedState, ModelSchedConfig, ModelEntry, ModelRegistry, ModelSignature,
    LoadedModel, Manifest, TensorSpec, Dim, BatchingSpec, Provenance, BATCH, FIXED, F32,
    Scheduler, SchedulerConfig, MockBackend, MockClient, MockDevice, MemoryPool, QueuedRequest,
    InferRequest, NamedTensor, priority, effective_cost, plan_batch, select_dispatch!,
    _decay_ema!, _update_cost!

# A manifest with no batch axis (not coalescable): plan_batch serves one request per dispatch.
_sched_trivial_manifest(name) = Manifest("2.0", name, "", TensorSpec[], TensorSpec[], nothing,
    nothing, BatchingSpec(Int[]), Provenance(Dict{String,Any}()), nothing)

# A coalescable manifest: input "x" and output "y" both shaped (features, n) with the batch axis
# last (1-based axis 2, i.e. 0-based input_batch_dim 1).
function _sched_batched_manifest(name; features=2)
    inx = TensorSpec("x", F32, Dim[Dim(FIXED, features), Dim(BATCH)], 2)
    outy = TensorSpec("y", F32, Dim[Dim(FIXED, features), Dim(BATCH)], 2)
    return Manifest("2.0", name, "", TensorSpec[inx], TensorSpec[outy], nothing, nothing,
        BatchingSpec(Int[]), Provenance(Dict{String,Any}()), 1)
end

_sched_sig() = ModelSignature(String[], DataType[], String[], 0, String[], DataType[], 0)

# A registry entry whose executable exposes the given compiled-size keys; the executables
# themselves are placeholders since selection never runs them.
function _sched_entry(name, sizes::Vector{Int}; coalescable=false, features=2)
    execs = Dict{Int,Any}(sz => nothing for sz in sizes)
    model = LoadedModel(_sched_sig(), execs, Any[])
    manifest = coalescable ? _sched_batched_manifest(name; features=features) : _sched_trivial_manifest(name)
    return ModelEntry(name, manifest, Dict{Int,Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
end

# Register an entry plus its scheduling state into a scheduler (the consolidated `entry.sched`).
function _install!(s, entry, st)
    entry.sched = st
    s.registry.by_name[entry.name] = entry
    return entry
end

function _empty_sched(; discipline=ReactantServer.FAIR)
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    return Scheduler(ModelRegistry(), backend, pool,
        SchedulerConfig(30.0, 1024, 30.0; discipline=discipline))
end

function _qr(model; rows=1, arrival=0.0, deadline_ns=0)
    req = InferRequest(model, ["y"], [NamedTensor("x", zeros(Float32, 2, rows))], Int64(deadline_ns))
    return QueuedRequest(req, req.inputs, arrival, Channel{Any}(1))
end

@testset "priority: spec worked example" begin
    # Three equal-weight models, EMAs 0.7/0.2/0.1, cap 0.25, discount 0.10, batch-1 costs in ms.
    share = 1 / 3
    pa = priority(share, 0.7, 0.25, 10 * 0.9)
    pb = priority(share, 0.2, 0.25, 30 * 0.9)
    pc = priority(share, 0.1, 0.25, 100 * 0.9)
    @test pa ≈ -0.0278 atol = 1e-4
    @test pb ≈ 0.00493 atol = 1e-4
    @test pc ≈ 0.00259 atol = 1e-4
    @test pb > pc > pa            # B wins; the recently-busy A is penalized to negative
    # the deficit is clamped: A's raw deficit is -0.367 but the penalty caps at -0.25
    @test priority(share, 0.7, 0.25, 1.0) == -0.25
    @test priority(share, 0.0, 0.25, 1.0) == 0.25
end

@testset "EMA decay halves over a half-life" begin
    st = ModelSchedState("m", ModelSchedConfig(1.0), 0.0)
    st.recent_compute_ema = 1.0
    _decay_ema!(st, 30.0, 30.0)               # one half-life elapsed
    @test st.recent_compute_ema ≈ 0.5 atol = 1e-9
    _decay_ema!(st, 30.0, 60.0)               # a second half-life
    @test st.recent_compute_ema ≈ 0.25 atol = 1e-9
end

@testset "cost estimate converges toward measurement" begin
    st = ModelSchedState("m", ModelSchedConfig(1.0), 0.0)
    _update_cost!(st, 1, 0.040, 0.2)          # first measurement seeds the estimate
    @test st.cost_estimate[1] == 0.040
    for _ in 1:50
        _update_cost!(st, 1, 0.010, 0.2)
    end
    @test st.cost_estimate[1] ≈ 0.010 atol = 1e-3
    @test effective_cost(st, 1, 0.10) ≈ 0.010 * 0.9 atol = 1e-4
end

@testset "plan_batch coalescing rules" begin
    # full fill: rows 2+2+2 with sizes [1,4,8] picks B=4 and takes two requests; the third stays
    e = _sched_entry("c", [1, 4, 8]; coalescable=true)
    st = ModelSchedState("c", ModelSchedConfig(1.0), 0.0)
    push!(st.queue, _qr("c"; rows=2), _qr("c"; rows=2), _qr("c"; rows=2))
    B, taken = plan_batch(e, st)
    @test B == 4
    @test length(taken) == 2

    # partial fill: a single 1-row request with sizes [4,8] pads up to B=4
    e2 = _sched_entry("p", [4, 8]; coalescable=true)
    st2 = ModelSchedState("p", ModelSchedConfig(1.0), 0.0)
    push!(st2.queue, _qr("p"; rows=1))
    B2, taken2 = plan_batch(e2, st2)
    @test B2 == 4
    @test length(taken2) == 1

    # oversized front request bumps B up to the smallest size that fits it
    e3 = _sched_entry("g", [1, 4, 8]; coalescable=true)
    st3 = ModelSchedState("g", ModelSchedConfig(1.0), 0.0)
    push!(st3.queue, _qr("g"; rows=6), _qr("g"; rows=1))
    B3, taken3 = plan_batch(e3, st3)
    @test B3 == 8
    @test length(taken3) == 2          # 6 + 1 both fit in 8

    # non-coalescable model serves exactly one request per dispatch
    e4 = _sched_entry("s", [1]; coalescable=false)
    st4 = ModelSchedState("s", ModelSchedConfig(1.0), 0.0)
    push!(st4.queue, _qr("s"), _qr("s"))
    _, taken4 = plan_batch(e4, st4)
    @test length(taken4) == 1
end

@testset "plan_batch respects per-model max_batch_size" begin
    # the cap limits coalescing: four 2-row requests with sizes [1,4,8] and cap 4 pick B=4 and
    # take two requests (uncapped this queue would fill B=8 with all four)
    e = _sched_entry("c", [1, 4, 8]; coalescable=true)
    st = ModelSchedState("c", ModelSchedConfig(1.0; max_batch_size=4), 0.0)
    push!(st.queue, _qr("c"; rows=2), _qr("c"; rows=2), _qr("c"; rows=2), _qr("c"; rows=2))
    B, taken = plan_batch(e, st)
    @test B == 4
    @test length(taken) == 2

    # a cap below the smallest compiled size still pads to that size but takes <= cap rows
    e2 = _sched_entry("p", [4, 8]; coalescable=true)
    st2 = ModelSchedState("p", ModelSchedConfig(1.0; max_batch_size=2), 0.0)
    push!(st2.queue, _qr("p"; rows=1), _qr("p"; rows=1), _qr("p"; rows=1))
    B2, taken2 = plan_batch(e2, st2)
    @test B2 == 4
    @test length(taken2) == 2

    # a single request over the cap cannot be split: it dispatches alone at a size that fits
    e3 = _sched_entry("g", [1, 4, 8]; coalescable=true)
    st3 = ModelSchedState("g", ModelSchedConfig(1.0; max_batch_size=2), 0.0)
    push!(st3.queue, _qr("g"; rows=6), _qr("g"; rows=1))
    B3, taken3 = plan_batch(e3, st3)
    @test B3 == 8
    @test length(taken3) == 1
end

@testset "select_dispatch! picks the fair winner (worked example)" begin
    s = _empty_sched()
    now = 1000.0
    specs = (("A", 0.7, 0.010), ("B", 0.2, 0.030), ("C", 0.1, 0.100))
    for (name, ema, cost) in specs
        st = ModelSchedState(name, ModelSchedConfig(1.0), now)
        st.recent_compute_ema = ema
        st.cost_estimate[1] = cost
        push!(st.queue, _qr(name; arrival=now))
        _install!(s, _sched_entry(name, [1]; coalescable=false), st)
    end
    d = select_dispatch!(s, now)
    @test d.entry.name == "B"
    @test length(d.taken) == 1
    @test length(s.registry.by_name["B"].sched.queue) == 0      # the taken request was dequeued
    @test length(s.registry.by_name["A"].sched.queue) == 1
end

@testset "FIFO discipline serves the globally oldest request, still coalescing" begin
    s = _empty_sched(; discipline=ReactantServer.FIFO)
    now = 1000.0
    # "hot" would dominate on fairness, but FIFO ignores fairness and serves the oldest arrival.
    hot = ModelSchedState("hot", ModelSchedConfig(1.0), now)
    hot.recent_compute_ema = 0.0
    hot.cost_estimate[1] = 0.001
    push!(hot.queue, _qr("hot"; arrival=now))                        # newer
    _install!(s, _sched_entry("hot", [1]; coalescable=false), hot)

    old = ModelSchedState("old", ModelSchedConfig(1.0), now)
    old.recent_compute_ema = 100.0
    old.cost_estimate[1] = 1.0
    push!(old.queue, _qr("old"; arrival=now - 1.0))                  # older arrival wins under FIFO
    _install!(s, _sched_entry("old", [1]; coalescable=false), old)

    d = select_dispatch!(s, now)
    @test d.entry.name == "old"

    # FIFO still coalesces the chosen model's queued requests into one dispatch.
    s2 = _empty_sched(; discipline=ReactantServer.FIFO)
    stc = ModelSchedState("c", ModelSchedConfig(1.0), now)
    push!(stc.queue, _qr("c"; rows=2, arrival=now), _qr("c"; rows=2, arrival=now + 0.1))
    _install!(s2, _sched_entry("c", [1, 4, 8]; coalescable=true), stc)
    d2 = select_dispatch!(s2, now)
    @test d2.size == 4
    @test length(d2.taken) == 2
end

@testset "EDF serves the soonest-deadline model, degrading to FIFO on ties" begin
    s = _empty_sched(; discipline=ReactantServer.EDF)
    now = 1000.0
    # "old" arrived first but carries a far deadline; "urgent" arrived later with a sooner deadline.
    # EDF picks the soonest deadline regardless of arrival (the in-flight meta sub-call case).
    old = ModelSchedState("old", ModelSchedConfig(1.0), now)
    push!(old.queue, _qr("old"; arrival=now, deadline_ns=Int64(5_000_000_000)))
    _install!(s, _sched_entry("old", [1]; coalescable=false), old)
    urgent = ModelSchedState("urgent", ModelSchedConfig(1.0), now)
    push!(urgent.queue, _qr("urgent"; arrival=now + 1.0, deadline_ns=Int64(1_000_000_000)))
    _install!(s, _sched_entry("urgent", [1]; coalescable=false), urgent)
    @test select_dispatch!(s, now).entry.name == "urgent"

    # With equal deadlines (the uniform-deadline case) EDF degrades to FIFO: oldest arrival wins.
    s2 = _empty_sched(; discipline=ReactantServer.EDF)
    a = ModelSchedState("a", ModelSchedConfig(1.0), now)
    push!(a.queue, _qr("a"; arrival=now - 1.0, deadline_ns=Int64(9_000_000_000)))   # older
    _install!(s2, _sched_entry("a", [1]; coalescable=false), a)
    b = ModelSchedState("b", ModelSchedConfig(1.0), now)
    push!(b.queue, _qr("b"; arrival=now, deadline_ns=Int64(9_000_000_000)))         # same deadline, newer
    _install!(s2, _sched_entry("b", [1]; coalescable=false), b)
    @test select_dispatch!(s2, now).entry.name == "a"

    # Requests with no deadline (0) are least urgent: a deadline'd model is served first.
    s3 = _empty_sched(; discipline=ReactantServer.EDF)
    nod = ModelSchedState("nodeadline", ModelSchedConfig(1.0), now)
    push!(nod.queue, _qr("nodeadline"; arrival=now - 5.0))                          # oldest, but no deadline
    _install!(s3, _sched_entry("nodeadline", [1]; coalescable=false), nod)
    dl = ModelSchedState("hasdeadline", ModelSchedConfig(1.0), now)
    push!(dl.queue, _qr("hasdeadline"; arrival=now, deadline_ns=Int64(2_000_000_000)))
    _install!(s3, _sched_entry("hasdeadline", [1]; coalescable=false), dl)
    @test select_dispatch!(s3, now).entry.name == "hasdeadline"
end

@testset "submit! caps each model's queue independently and rejects after shutdown" begin
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    s = Scheduler(ModelRegistry(), backend, pool, SchedulerConfig(30.0, 2, 30.0))
    now = 1000.0
    for name in ("a", "b")
        _install!(s, _sched_entry(name, [1]), ModelSchedState(name, ModelSchedConfig(1.0), now))
    end
    s.running = true

    # Fill model a to its cap; the next submit for a is rejected, but b still admits.
    ok1, ok2, full = _qr("a"), _qr("a"), _qr("a")
    ReactantServer.submit!(s, ok1)
    ReactantServer.submit!(s, ok2)
    ReactantServer.submit!(s, full)
    err = take!(full.reply)
    @test err isa Exception
    @test occursin("full", sprint(showerror, err))
    bqr = _qr("b")
    ReactantServer.submit!(s, bqr)
    @test length(s.registry.by_name["b"].sched.queue) == 1
    @test length(s.registry.by_name["a"].sched.queue) == 2

    # After shutdown, new submissions are rejected immediately.
    s.running = false
    late = _qr("a")
    ReactantServer.submit!(s, late)
    late_err = take!(late.reply)
    @test late_err isa Exception
    @test occursin("shutting down", sprint(showerror, late_err))
end

@testset "shutdown! rejects queued requests so blocked callers unblock" begin
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    s = Scheduler(ModelRegistry(), backend, pool, SchedulerConfig(30.0, 16, 30.0))
    entry = _install!(s, _sched_entry("m", [1]), ModelSchedState("m", ModelSchedConfig(1.0), 0.0))
    # No executable: the entry accepts submissions but is never schedulable, so a queued
    # request can only be released by the shutdown path.
    entry.executable = nothing

    ReactantServer.start!(s)
    qr = _qr("m")
    ReactantServer.submit!(s, qr)
    @test length(entry.sched.queue) == 1
    ReactantServer.shutdown!(s)
    e = take!(qr.reply)                              # would hang forever before the fix
    @test e isa Exception
    @test occursin("shutting down", sprint(showerror, e))
    @test isempty(entry.sched.queue)

    # Control commands after shutdown fail fast instead of blocking forever.
    @test_throws Exception ReactantServer.set_residency!(s, "m", ReactantServer.PINNED_DEVICE)
end
