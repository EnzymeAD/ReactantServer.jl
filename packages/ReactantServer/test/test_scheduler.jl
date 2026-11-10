# Deterministic unit tests for the scheduler policy: EMA decay, cost learning, coalescing batch
# selection, the fair discipline (selection rules, simulated scenarios and randomized checks of its
# guarantees), and the FIFO and EDF disciplines. These exercise the pure helpers and
# select_dispatch! directly, with no spawned dispatch loop and no real clock, so they are fully
# deterministic.

using ReactantServer: ModelSchedState, ModelSchedConfig, ModelEntry, ModelRegistry, ModelSignature,
    LoadedModel, Manifest, TensorSpec, Dim, BatchingSpec, Provenance, BATCH, FIXED, F32,
    Scheduler, SchedulerConfig, MockBackend, MockClient, MockDevice, MemoryPool, QueuedRequest,
    InferRequest, NamedTensor, plan_batch, select_dispatch!, _decay_ema!, _update_cost!,
    _charge_fair!
using Random: MersenneTwister, randexp

# A manifest with no batch axis (not coalescable): plan_batch serves one request per dispatch.
_sched_trivial_manifest(name) = Manifest(
    "2.0", name, "", TensorSpec[], TensorSpec[], nothing,
    nothing, BatchingSpec(Int[]), Provenance(Dict{String, Any}()), nothing
)

# A coalescable manifest: input "x" and output "y" both shaped (features, n) with the batch axis
# last (1-based axis 2, i.e. 0-based input_batch_dim 1).
function _sched_batched_manifest(name; features = 2)
    inx = TensorSpec("x", F32, Dim[Dim(FIXED, features), Dim(BATCH)], 2)
    outy = TensorSpec("y", F32, Dim[Dim(FIXED, features), Dim(BATCH)], 2)
    return Manifest(
        "2.0", name, "", TensorSpec[inx], TensorSpec[outy], nothing, nothing,
        BatchingSpec(Int[]), Provenance(Dict{String, Any}()), 1
    )
end

_sched_sig() = ModelSignature(String[], DataType[], String[], 0, String[], DataType[], 0)

# A registry entry whose executable exposes the given compiled-size keys; the executables
# themselves are placeholders since selection never runs them.
function _sched_entry(name, sizes::Vector{Int}; coalescable = false, features = 2)
    execs = Dict{Int, Any}(sz => nothing for sz in sizes)
    model = LoadedModel(_sched_sig(), execs, Any[])
    manifest = coalescable ? _sched_batched_manifest(name; features = features) : _sched_trivial_manifest(name)
    return ModelEntry(name, manifest, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
end

# Register an entry plus its scheduling state into a scheduler (the consolidated `entry.sched`).
function _install!(s, entry, st)
    entry.sched = st
    s.registry.by_name[entry.name] = entry
    return entry
end

function _empty_sched(; discipline = ReactantServer.FAIR)
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    return Scheduler(
        ModelRegistry(), backend, pool,
        SchedulerConfig(30.0, 1024, 30.0; discipline = discipline)
    )
end

function _qr(model; rows = 1, arrival = 0.0, deadline_ns = 0)
    req = InferRequest(model, ["y"], [NamedTensor("x", zeros(Float32, 2, rows))], Int64(deadline_ns))
    return QueuedRequest(req, req.inputs, arrival, Channel{Any}(1))
end

@testset "EMA decay halves over a half-life" begin
    st = ModelSchedState("m", ModelSchedConfig(1.0), 0.0)
    st.recent_compute_ema = 1.0
    _decay_ema!(st, 30.0, 30.0)               # one half-life elapsed
    @test st.recent_compute_ema ≈ 0.5 atol = 1.0e-9
    _decay_ema!(st, 30.0, 60.0)               # a second half-life
    @test st.recent_compute_ema ≈ 0.25 atol = 1.0e-9
end

@testset "cost estimate: cold run unused, bounded rise, time-weighted fall" begin
    st = ModelSchedState("m", ModelSchedConfig(1.0), 0.0)
    v = ReactantServer.VariantKey()
    est() = ReactantServer._cost_seconds(st, v, 1)
    # The first run is cold and does not count; the second seeds, at the smaller of the two (SCH-4).
    _update_cost!(st, v, 1, 0.5, 0.2, 0.0)
    @test est() === nothing
    _update_cost!(st, v, 1, 0.04, 0.2, 0.1)
    @test est() == 0.04
    # Back-to-back measurements converge.
    for i in 1:50
        _update_cost!(st, v, 1, 0.01, 0.2, 0.1 + i * 1.0e-3)
    end
    @test est() ≈ 0.01 atol = 1.0e-3
    # One outlier raises the estimate by a quarter at most, however slow and however old the estimate.
    e0 = est()
    _update_cost!(st, v, 1, 10.0, 0.2, 1000.0)
    @test est() ≈ 1.25 * e0
    # A faster run pulls it down with weight alpha when back to back...
    st.cost_estimate[(v, 1)] = ReactantServer.CostEstimate(1.0, 5, 100.0, -Inf)
    _update_cost!(st, v, 1, 0.5, 0.2, 100.0)
    @test est() ≈ 0.9
    # ...and almost all the way once the estimate is ten half-lives old.
    st.cost_estimate[(v, 1)] = ReactantServer.CostEstimate(1.0, 5, 100.0, -Inf)
    _update_cost!(st, v, 1, 0.1, 0.2, 100.0 + 10 * ReactantServer._COST_HALFLIFE_SECONDS)
    @test est() ≈ 0.1 + 0.9 * 0.8 * 2.0^-10
    # A non-finite measurement (a clock step) is ignored.
    before = est()
    _update_cost!(st, v, 1, NaN, 0.2, 200.0)
    @test est() == before
end

# A model compiled at batch size 1 for two input shapes: input "img" is (w, h, n) with w and h
# variable, so the variant key is [w, h]. Each variant's program sleeps for its own time, so the two
# cost different amounts at the same batch size, as a detector's aspect ratios do.
function _sched_ms_entry(name, seconds::Dict{Vector{Int}, Float64})
    dims = Dim[Dim(ReactantServer.VARIABLE), Dim(ReactantServer.VARIABLE), Dim(BATCH)]
    manifest = Manifest(
        "2.0", name, "", TensorSpec[TensorSpec("img", F32, dims, 3)], TensorSpec[TensorSpec("y", F32, dims, 3)],
        nothing, nothing, BatchingSpec(Int[1]), Provenance(Dict{String, Any}()),
        2, "model", String[], collect(keys(seconds))
    )
    sig = ModelSignature(["img"], DataType[Float32], String[], 1, ["y"], DataType[Float32], 2, [("img", 1), ("img", 2)])
    execs = Dict{ReactantServer.VariantKey, Dict{Int, Any}}(
        v => Dict{Int, Any}(1 => ReactantServer.MockExecutable(args -> (sleep(c); [args[1]]), 1))
            for (v, c) in seconds
    )
    return ModelEntry(name, manifest, Dict{Int, Vector{UInt8}}(), "", nothing, LoadedModel(sig, execs, Any[]), nothing, identity, identity)
end

function _ms_qr(name, w, h; deadline_ns = 0)
    req = InferRequest(name, ["y"], [NamedTensor("img", zeros(Float32, w, h, 1))], Int64(deadline_ns))
    return QueuedRequest(req, req.inputs, monotonic_time(), Channel{Any}(1))
end

@testset "cost estimates are kept per (shape variant, batch size)" begin
    # Before (SCH-10) the estimate was keyed by batch size alone: warmup let the last variant it ran
    # overwrite the others, and one variant's dispatches set the cost every variant was judged by.
    small, large = [2, 3], [4, 2]
    e = _sched_ms_entry("ms", Dict(small => 0.0, large => 0.25))
    s = _empty_sched()
    _install!(s, e, ModelSchedState("ms", ModelSchedConfig(1.0), 0.0))
    st = e.sched
    ReactantServer._warmup_entry!(s, e)
    @test ReactantServer._cost_seconds(st, large, 1) >= 0.25
    @test ReactantServer._cost_seconds(st, small, 1) < 0.25
    @test !haskey(st.cost_estimate, (ReactantServer.VariantKey(), 1))

    # The choice to run a program off the interactive thread reads that program's own estimate.
    st.cost_estimate[(small, 1)] = ReactantServer.CostEstimate(0.001, 2, monotonic_time(), -Inf)
    st.cost_estimate[(large, 1)] = ReactantServer.CostEstimate(1.0, 2, monotonic_time(), -Inf)
    @test !ReactantServer._offload_execution(st, small, 1)
    @test ReactantServer._offload_execution(st, large, 1)
    @test ReactantServer._offload_execution(st, [9, 9], 1)        # never measured: could be long

    # Under EDF, a small-shape request is judged by the small program's cost: two dispatches of the
    # slow shape (0.25 s each) do not make a 0.1 s budget look infeasible for the fast one.
    s2 = _empty_sched(; discipline = ReactantServer.EDF)
    e2 = _sched_ms_entry("ms2", Dict(small => 0.0, large => 0.25))
    _install!(s2, e2, ModelSchedState("ms2", ModelSchedConfig(1.0), 0.0))
    for _ in 1:2
        qr = _ms_qr("ms2", 4, 2)
        push!(e2.sched.queue, qr)
        ReactantServer.execute_and_record!(s2, select_dispatch!(s2, 0.0))
        @test !(take!(qr.reply) isa Exception)
    end
    @test ReactantServer._cost_seconds(e2.sched, large, 1) >= 0.25
    qr = _ms_qr("ms2", 2, 3; deadline_ns = Int64(time_ns()) + 100_000_000)
    push!(e2.sched.queue, qr)
    ReactantServer.execute_and_record!(s2, select_dispatch!(s2, 0.0))
    out = take!(qr.reply)
    @test !(out isa Exception)
    @test size(only(out).data) == (2, 3, 1)
end

# A coalescable model compiled only at batch size 4, whose program is `run` (args -> outputs).
function _sched_run_entry(name, run)
    sig = ModelSignature(["x"], DataType[Float32], String[], 1, ["y"], DataType[Float32], 1)
    model = LoadedModel(sig, Dict{Int, Any}(4 => ReactantServer.MockExecutable(run, 1)), Any[])
    return ModelEntry(name, _sched_batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
end

@testset "warmup seeds a program's cost from a warm run, not its cold first run" begin
    # SCH-4: the first execution pays one-time costs, and seeding from it alone left EDF an estimate
    # far above what the program takes.
    calls = Threads.Atomic{Int}(0)
    e = _sched_run_entry("cold", args -> (Threads.atomic_add!(calls, 1) == 0 && sleep(0.3); [args[1]]))
    s = _empty_sched()
    _install!(s, e, ModelSchedState("cold", ModelSchedConfig(1.0), 0.0))
    ReactantServer._warmup_entry!(s, e)
    @test calls[] == 2
    @test ReactantServer._cost_seconds(e.sched, ReactantServer.VariantKey(), 4) < 0.1
end

@testset "EDF: an inflated cost estimate is measured again instead of shedding a model for good" begin
    # SCH-4: the laxity check drops a request that cannot finish within the learned cost, and a drop
    # runs nothing, so an estimate above every client's budget was never measured again: the model
    # was shed permanently. Here the program takes microseconds, the estimate says 0.5 s, and every
    # client allows 0.1 s.
    fail = Ref(false)
    e = _sched_run_entry("m", args -> (fail[] && error("boom"); [args[1]]))
    s = _empty_sched(; discipline = ReactantServer.EDF)
    _install!(s, e, ModelSchedState("m", ModelSchedConfig(1.0), 0.0))
    st = e.sched
    key = (ReactantServer.VariantKey(), 4)
    st.cost_estimate[key] = ReactantServer.CostEstimate(0.5, 2, monotonic_time(), -Inf)
    function dispatch3()
        qrs = [_qr("m"; deadline_ns = Int64(time_ns()) + 100_000_000) for _ in 1:3]
        append!(st.queue, qrs)
        d = select_dispatch!(s, 0.0)
        @test length(d.taken) == 3                 # the three coalesce into one batch-4 dispatch
        ReactantServer.execute_and_record!(s, d)
        return [take!(qr.reply) for qr in qrs]
    end
    shed(r) = r isa ReactantServer.DeadlineExceeded

    # While the estimate is fresh it is believed: every request is shed and nothing runs.
    for _ in 1:2
        @test all(shed, dispatch3())
    end
    @test st.dispatch_count == 0
    @test ReactantServer._cost_seconds(st, key...) == 0.5

    # Once it has not been refreshed for the stale interval, the oldest request runs to measure the
    # program again; the others are still shed. The measurement replaces the old estimate...
    st.cost_estimate[key].refreshed -= ReactantServer._COST_STALE_SECONDS + 1
    r = dispatch3()
    @test !(r[1] isa Exception)
    @test all(shed, r[2:3])
    @test ReactantServer._cost_seconds(st, key...) < 0.05
    # ...and the model serves again.
    @test !any(r -> r isa Exception, dispatch3())

    # At most one such run per interval, even when it fails and so refreshes nothing.
    st.cost_estimate[key] = ReactantServer.CostEstimate(0.5, 5, monotonic_time() - ReactantServer._COST_STALE_SECONDS - 1, -Inf)
    fail[] = true
    r = dispatch3()
    @test r[1] isa ErrorException
    @test all(shed, r[2:3])
    fail[] = false
    @test all(shed, dispatch3())
    st.cost_estimate[key].remeasured -= ReactantServer._COST_STALE_SECONDS + 1
    r = dispatch3()
    @test !(r[1] isa Exception)
end

@testset "EDF: a withdrawn request is never the run that measures a stale estimate" begin
    # A withdrawn request's caller has stopped waiting (dispatch timeout, cancellation), so it never
    # runs and gets no reply, also when it is the oldest request the stale-estimate rule would run.
    e = _sched_run_entry("w", args -> [args[1]])
    s = _empty_sched(; discipline = ReactantServer.EDF)
    _install!(s, e, ModelSchedState("w", ModelSchedConfig(1.0), 0.0))
    st = e.sched
    key = (ReactantServer.VariantKey(), 4)
    st.cost_estimate[key] = ReactantServer.CostEstimate(0.5, 5, monotonic_time() - ReactantServer._COST_STALE_SECONDS - 1, -Inf)
    qrs = [_qr("w"; deadline_ns = Int64(time_ns()) + 100_000_000) for _ in 1:3]
    ReactantServer._withdraw!(qrs[1])
    append!(st.queue, qrs)
    ReactantServer.execute_and_record!(s, select_dispatch!(s, 0.0))
    @test !isready(qrs[1].reply)                   # withdrawn: neither run nor answered
    @test !(take!(qrs[2].reply) isa Exception)     # the oldest live request measures instead
    @test take!(qrs[3].reply) isa ReactantServer.DeadlineExceeded
    @test st.dispatch_count == 1

    # With every request withdrawn nothing runs, and the interval is not spent.
    st.cost_estimate[key] = ReactantServer.CostEstimate(0.5, 5, monotonic_time() - ReactantServer._COST_STALE_SECONDS - 1, -Inf)
    qrs = [_qr("w"; deadline_ns = Int64(time_ns()) + 100_000_000) for _ in 1:2]
    foreach(ReactantServer._withdraw!, qrs)
    append!(st.queue, qrs)
    ReactantServer.execute_and_record!(s, select_dispatch!(s, 0.0))
    @test !any(qr -> isready(qr.reply), qrs)
    @test st.dispatch_count == 1
    @test st.cost_estimate[key].remeasured == -Inf
end

@testset "plan_batch coalescing rules" begin
    # full fill: rows 2+2+2 with sizes [1,4,8] picks B=4 and takes two requests; the third stays
    e = _sched_entry("c", [1, 4, 8]; coalescable = true)
    st = ModelSchedState("c", ModelSchedConfig(1.0), 0.0)
    push!(st.queue, _qr("c"; rows = 2), _qr("c"; rows = 2), _qr("c"; rows = 2))
    B, taken = plan_batch(e, st)
    @test B == 4
    @test length(taken) == 2

    # partial fill: a single 1-row request with sizes [4,8] pads up to B=4
    e2 = _sched_entry("p", [4, 8]; coalescable = true)
    st2 = ModelSchedState("p", ModelSchedConfig(1.0), 0.0)
    push!(st2.queue, _qr("p"; rows = 1))
    B2, taken2 = plan_batch(e2, st2)
    @test B2 == 4
    @test length(taken2) == 1

    # oversized front request bumps B up to the smallest size that fits it
    e3 = _sched_entry("g", [1, 4, 8]; coalescable = true)
    st3 = ModelSchedState("g", ModelSchedConfig(1.0), 0.0)
    push!(st3.queue, _qr("g"; rows = 6), _qr("g"; rows = 1))
    B3, taken3 = plan_batch(e3, st3)
    @test B3 == 8
    @test length(taken3) == 2          # 6 + 1 both fit in 8

    # non-coalescable model serves exactly one request per dispatch
    e4 = _sched_entry("s", [1]; coalescable = false)
    st4 = ModelSchedState("s", ModelSchedConfig(1.0), 0.0)
    push!(st4.queue, _qr("s"), _qr("s"))
    _, taken4 = plan_batch(e4, st4)
    @test length(taken4) == 1
end

@testset "an oversize request is split into compiled-size pieces and joined back" begin
    # A client batching 32 against a bundle compiled for [1] used to get INTERNAL on every call; two
    # rounds of n=1 verification could not see it. The split is coalescing run backwards.
    e = _sched_entry("o", [1, 4]; coalescable = true)
    x = NamedTensor("x", reshape(Float32.(1:20), 2, 10))          # 10 rows, batch axis 2
    req = InferRequest("o", ["y"], [x], Int64(0))
    prepared = req.inputs
    @test ReactantServer._max_compiled_rows(e, prepared) == 4
    @test ReactantServer._executable_rows(e, prepared) == 10
    @test ReactantServer._split_ranges(10, 4) == [1:4, 5:8, 9:10]
    @test ReactantServer._split_ranges(4, 4) == [1:4]
    pieces = [
        ReactantServer._take_rows(prepared, e.manifest.executable_inputs, r)
            for r in ReactantServer._split_ranges(10, 4)
    ]
    @test [size(only(pc).data, 2) for pc in pieces] == [4, 4, 2]
    @test only(pieces[3]).data == x.data[:, 9:10]
    # Pretend each piece produced its own rows as output; the join restores the full batch in order.
    outs = [NamedTensor[NamedTensor("y", only(pc).data .* 2)] for pc in pieces]
    joined = ReactantServer._join_outputs(e, outs)
    @test only(joined).data == x.data .* 2
    # Unbatched (key 0) and uncompiled variants are never split.
    e0 = _sched_entry("u", [0]; coalescable = true)
    @test ReactantServer._max_compiled_rows(e0, prepared) === nothing
end

@testset "a preprocess that changes the row count still splits an oversize request" begin
    # One client row becomes two executable rows (a text cut into two windows, say). Three client
    # rows are six executable rows against a bundle compiled for [1, 4]: before the split read the
    # preprocessed rows it refused (3 != 6), dispatched all six whole, and found no program for 6.
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    reg = ModelRegistry()
    sizes_run = Int[]
    run_lock = ReentrantLock()
    double(args) = (lock(() -> push!(sizes_run, size(args[1], 2)), run_lock); [args[1] .* 2])
    sig = ModelSignature(["x"], DataType[Float32], String[], 1, ["y"], DataType[Float32], 1)
    execs = Dict{Int, Any}(1 => ReactantServer.MockExecutable(double, 1), 4 => ReactantServer.MockExecutable(double, 1))
    pre(inputs) = [NamedTensor("x", repeat(inputs[1].data; inner = (1, 2)))]
    post(outputs) = (y = outputs[1].data; [NamedTensor("y", y[:, 1:2:end] .+ y[:, 2:2:end])])
    reg.by_name["w"] = ModelEntry(
        "w", _sched_batched_manifest("w"), Dict{Int, Vector{UInt8}}(), "", nothing,
        LoadedModel(sig, execs, Any[]), nothing, pre, post
    )
    s = Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
    ReactantServer.start!(s)
    lock(() -> empty!(sizes_run), run_lock)    # start! warms each compiled size; count only the request
    try
        x = reshape(Float32.(1:6), 2, 3)
        out = ReactantServer.infer(s, InferRequest("w", ["y"], [NamedTensor("x", x)], Int64(0)))
        # Each window doubles, and postprocess sums a row's two windows back together: 4x.
        @test only(out).data == x .* 4
        # The six rows ran as a full b4 piece and a two-row piece padded to b4; no b6 was asked for.
        @test sort(sizes_run) == [4, 4]
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "a request whose model is replaced during its preprocess never runs on the new version" begin
    # A reload swaps the entry under the same name while a request is in the old version's
    # preprocess. Enqueued on the new entry it would run the new program on inputs prepared for the
    # old one, and its outputs would go through the old postprocess; it must fail instead, and so
    # must every piece of an oversize request.
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    reg = ModelRegistry()
    sig = ModelSignature(["x"], DataType[Float32], String[], 1, ["y"], DataType[Float32], 1)
    v2_rows = Ref(0)
    function model_version(scale; pre = identity, count = false)
        scaled(args) = (count && (v2_rows[] += size(args[1], 2)); [args[1] .* scale])
        execs = Dict{Int, Any}(1 => ReactantServer.MockExecutable(scaled, 1))
        return ModelEntry(
            "m", _sched_batched_manifest("m"), Dict{Int, Vector{UInt8}}(), "", nothing,
            LoadedModel(sig, execs, Any[]), nothing, pre, identity
        )
    end
    s = Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
    ReactantServer.start!(s)
    try
        for rows in (1, 3)                       # one piece, and three split pieces
            entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
            blocking(inputs) = (put!(entered, nothing); take!(release); inputs)
            ReactantServer.admit!(s, model_version(2.0f0; pre = blocking))
            req = InferRequest("m", ["y"], [NamedTensor("x", ones(Float32, 2, rows))], Int64(0))
            t = Threads.@spawn ReactantServer.infer(s, req)
            take!(entered)                       # in v1's preprocess
            ReactantServer.evict!(s, "m")        # the reload: v1 out, v2 in under the same name
            v2 = model_version(3.0f0; count = true)
            ReactantServer.admit!(s, v2)
            v2_rows[] = 0                        # admission's warmup ran v2; count requests only
            put!(release, nothing)
            err = try
                fetch(t)
                nothing
            catch e
                e isa TaskFailedException ? e.task.result : e
            end
            @test err isa ReactantServer.ModelNotFoundError          # the transport's NOT_FOUND
            @test ReactantServer._grpc_status(err) == ReactantServer.gRPCServer.StatusCode.NOT_FOUND
            @test occursin("was replaced during the request", sprint(showerror, err))
            @test v2_rows[] == 0                 # nothing of it ran on v2
            @test isempty(v2.sched.queue)
            # A fresh request runs on v2.
            out = ReactantServer.infer(s, InferRequest("m", ["y"], [NamedTensor("x", ones(Float32, 2, 1))], Int64(0)))
            @test only(out).data == fill(3.0f0, 2, 1)
            ReactantServer.evict!(s, "m")
        end
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "plan_batch respects per-model max_batch_size" begin
    # the cap limits coalescing: four 2-row requests with sizes [1,4,8] and cap 4 pick B=4 and
    # take two requests (uncapped this queue would fill B=8 with all four)
    e = _sched_entry("c", [1, 4, 8]; coalescable = true)
    st = ModelSchedState("c", ModelSchedConfig(1.0; max_batch_size = 4), 0.0)
    push!(st.queue, _qr("c"; rows = 2), _qr("c"; rows = 2), _qr("c"; rows = 2), _qr("c"; rows = 2))
    B, taken = plan_batch(e, st)
    @test B == 4
    @test length(taken) == 2

    # a cap below the smallest compiled size still pads to that size but takes <= cap rows
    e2 = _sched_entry("p", [4, 8]; coalescable = true)
    st2 = ModelSchedState("p", ModelSchedConfig(1.0; max_batch_size = 2), 0.0)
    push!(st2.queue, _qr("p"; rows = 1), _qr("p"; rows = 1), _qr("p"; rows = 1))
    B2, taken2 = plan_batch(e2, st2)
    @test B2 == 4
    @test length(taken2) == 2

    # a single request over the cap cannot be split: it dispatches alone at a size that fits
    e3 = _sched_entry("g", [1, 4, 8]; coalescable = true)
    st3 = ModelSchedState("g", ModelSchedConfig(1.0; max_batch_size = 2), 0.0)
    push!(st3.queue, _qr("g"; rows = 6), _qr("g"; rows = 1))
    B3, taken3 = plan_batch(e3, st3)
    @test B3 == 8
    @test length(taken3) == 1
end

@testset "plan_batch with coalescing off takes only the front request" begin
    # Queued rows that would fill B=4 (see the coalescing rules above) dispatch one request at a
    # time, each at the smallest compiled size that fits it.
    e = _sched_entry("c", [1, 4, 8]; coalescable = true)
    st = ModelSchedState("c", ModelSchedConfig(1.0), 0.0)
    front = _qr("c"; rows = 2)
    push!(st.queue, front, _qr("c"; rows = 2), _qr("c"; rows = 2))
    B, taken = plan_batch(e, st; coalesce = false)
    @test B == 4
    @test taken == [front]

    # One row on [1, 4, 8] runs the batch-1 program; under `largest` only [8] exists and it pads.
    st1 = ModelSchedState("c", ModelSchedConfig(1.0), 0.0)
    push!(st1.queue, _qr("c"), _qr("c"))
    @test plan_batch(e, st1; coalesce = false)[1] == 1
    e8 = _sched_entry("l", [8]; coalescable = true)
    st8 = ModelSchedState("l", ModelSchedConfig(1.0), 0.0)
    push!(st8.queue, _qr("l"), _qr("l"))
    B8, taken8 = plan_batch(e8, st8; coalesce = false)
    @test B8 == 8
    @test length(taken8) == 1
end

@testset "scheduler.coalescing reaches every discipline" begin
    for discipline in (ReactantServer.FAIR, ReactantServer.FIFO, ReactantServer.EDF)
        backend = MockBackend()
        pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
        s = Scheduler(
            ModelRegistry(), backend, pool,
            SchedulerConfig(30.0, 1024, 30.0; discipline = discipline, coalescing = false)
        )
        st = ModelSchedState("c", ModelSchedConfig(1.0), 0.0)
        push!(st.queue, _qr("c"), _qr("c"), _qr("c"))
        _install!(s, _sched_entry("c", [1, 4]; coalescable = true), st)
        d = select_dispatch!(s, 0.0)
        @test length(d.taken) == 1
        @test d.size == 1
        @test length(st.queue) == 2
    end
end

@testset "fair: the least virtual time goes next, and a returning model is lifted to the floor" begin
    s = _empty_sched()
    now = 1000.0
    for (name, vt) in (("A", 0.3), ("B", 0.1), ("C", 0.2))
        st = ModelSchedState(name, ModelSchedConfig(1.0), now)
        st.vtime = vt
        push!(st.queue, _qr(name; arrival = now))
        _install!(s, _sched_entry(name, [1]; coalescable = false), st)
    end
    d = select_dispatch!(s, now)
    @test d.entry.name == "B"
    @test length(d.taken) == 1
    @test isempty(s.registry.by_name["B"].sched.queue)            # the taken request was dequeued
    @test length(s.registry.by_name["A"].sched.queue) == 1
    @test s.vfloor == 0.1

    # A charge is the seconds over the weight; a negative measurement (a clock step) charges nothing.
    stw = ModelSchedState("w", ModelSchedConfig(2.0), now)
    _charge_fair!(stw, 0.5)
    @test stw.vtime == 0.25
    _charge_fair!(stw, -1.0)
    @test stw.vtime == 0.25

    # A model that was idle while the others ran rejoins at the floor: it banked no credit, so it
    # does not get a run of dispatches, and its old virtual time is gone.
    s2 = _empty_sched()
    s2.vfloor = 5.0
    busy = ModelSchedState("busy", ModelSchedConfig(1.0), now)
    busy.vtime = 5.0
    push!(busy.queue, _qr("busy"; arrival = now - 1.0))         # older front request
    _install!(s2, _sched_entry("busy", [1]), busy)
    back = ModelSchedState("back", ModelSchedConfig(1.0), now)
    back.vtime = 0.5                                             # last ran long ago
    push!(back.queue, _qr("back"; arrival = now))
    _install!(s2, _sched_entry("back", [1]), back)
    @test select_dispatch!(s2, now).entry.name == "busy"         # tie at the floor: oldest request
    @test back.vtime == 5.0
    _charge_fair!(busy, 0.01)
    push!(busy.queue, _qr("busy"; arrival = now))
    @test select_dispatch!(s2, now).entry.name == "back"

    # Equal virtual time and arrival break by name, for determinism.
    s3 = _empty_sched()
    for name in ("y", "x")
        st = ModelSchedState(name, ModelSchedConfig(1.0), now)
        push!(st.queue, _qr(name; arrival = now))
        _install!(s3, _sched_entry(name, [1]), st)
    end
    @test select_dispatch!(s3, now).entry.name == "x"

    # Work-conserving: a model alone in the queue is served however far ahead it is.
    s4 = _empty_sched()
    solo = ModelSchedState("solo", ModelSchedConfig(1.0), now)
    solo.vtime = 1.0e6
    push!(solo.queue, _qr("solo"; arrival = now))
    _install!(s4, _sched_entry("solo", [1]), solo)
    _install!(s4, _sched_entry("quiet", [1]), ModelSchedState("quiet", ModelSchedConfig(1.0), now))
    @test select_dispatch!(s4, now).entry.name == "solo"
    @test s4.vfloor == 1.0e6
end

# A simulated fair run: the real `select_dispatch!` chooses every dispatch, which is recorded as
# `execute_and_record!` records it (the virtual-time charge and the recent-compute EMA) on a
# simulated clock, so the run is deterministic and no time passes. Each model is
# `(name, weight, seconds per dispatch, on, off)`: saturated, one request always queued, while
# `on <= t < off`, and idle otherwise. Over the window `[w0, w1]` it returns each model's share of
# the device time and its longest wait between dispatches, measured as the audit's simulation
# (`sims/fair_fix_sim.jl`) measures them.
function _fair_scenario(models; T, w0 = 0.0, w1 = T)
    s = _empty_sched()
    for (name, w, _, _, _) in models
        _install!(s, _sched_entry(name, [1]), ModelSchedState(name, ModelSchedConfig(w), 0.0))
    end
    spec = Dict(m[1] => m for m in models)
    served = Dict(m[1] => 0.0 for m in models)
    last = Dict(m[1] => NaN for m in models)
    gap = Dict(m[1] => 0.0 for m in models)
    t = 0.0
    while t < T
        for (name, _, _, on, off) in models
            q = s.registry.by_name[name].sched.queue
            if on <= t < off
                isempty(q) && push!(q, _qr(name; arrival = t))
            else
                empty!(q)
            end
        end
        d = select_dispatch!(s, t)
        d === nothing && (t += 0.001; continue)
        name, st = d.entry.name, d.entry.sched
        c = spec[name][3]
        t += c
        _decay_ema!(st, s.cfg.ema_halflife_seconds, t)
        st.recent_compute_ema += c
        _charge_fair!(st, c)
        if w0 <= t <= w1
            served[name] += c
            start = isnan(last[name]) ? max(w0, spec[name][4]) : last[name]
            gap[name] = max(gap[name], t - c - start)
            last[name] = t
        end
    end
    total = sum(values(served))
    return Dict(name => (share = served[name] / total, gap = gap[name]) for name in keys(served))
end

@testset "fair: no lockout from history, idle models, cost or a model rejoining (simulated)" begin
    # The scenarios the old EMA-deficit ranking failed (SCH-3); its result is in each comment. The
    # wait bounds are the discipline's guarantee: at most one dispatch of each other model plus
    # the weight-scaled cost of this model's own.
    idle8 = [("i$i", 1.0, 0.01, Inf, Inf) for i in 1:8]

    # 1: A alone for 200 s, then B. Was: A got nothing for 29.8 s while B caught up.
    r = _fair_scenario([("A", 1.0, 0.01, 0.0, Inf), ("B", 1.0, 0.01, 200.0, Inf)]; T = 400.0, w0 = 200.0)
    @test r["A"].share ≈ 0.5 atol = 0.01
    @test r["B"].share ≈ 0.5 atol = 0.01
    @test r["A"].gap <= 0.02 + 1.0e-9
    @test r["B"].gap <= 0.02 + 1.0e-9

    # 2: A at 1 ms and B at 100 ms, both busy, with 8 idle models loaded. Was: B took 65% of the
    # device over this minute (90% over 300 s), the cost division ranking the cheap model last
    # once both were over their share.
    r = _fair_scenario(vcat([("A", 1.0, 0.001, 0.0, Inf), ("B", 1.0, 0.1, 0.0, Inf)], idle8); T = 60.0)
    @test r["A"].share ≈ 0.5 atol = 0.01
    @test r["B"].share ≈ 0.5 atol = 0.01
    @test r["A"].gap <= 0.101 + 1.0e-9
    @test r["B"].gap <= 0.101 + 1.0e-9

    # 3: as 2, but A alone first and B joining at 50 s. Was: A locked out for 13.6 s (98 s after
    # 200 s alone).
    r = _fair_scenario(
        vcat([("A", 1.0, 0.001, 0.0, Inf), ("B", 1.0, 0.1, 50.0, Inf)], idle8); T = 110.0, w0 = 50.0
    )
    @test r["A"].share ≈ 0.5 atol = 0.01
    @test r["B"].share ≈ 0.5 atol = 0.01
    @test r["A"].gap <= 0.101 + 1.0e-9

    # 4: weights 2:1 at equal cost split the device 2:1.
    r = _fair_scenario([("A", 2.0, 0.01, 0.0, Inf), ("B", 1.0, 0.01, 0.0, Inf)]; T = 300.0)
    @test r["A"].share ≈ 2 / 3 atol = 0.01
    @test r["B"].share ≈ 1 / 3 atol = 0.01
    @test r["A"].gap <= 0.015 + 1.0e-9
    @test r["B"].gap <= 0.03 + 1.0e-9

    # 5: A and B busy; C idle until 200 s, then busy. Was: C took 72% for the next 30 s and A and B
    # each went 17.4 s without a dispatch. An idle model must not hoard credit.
    r = _fair_scenario(
        [("A", 1.0, 0.01, 0.0, Inf), ("B", 1.0, 0.01, 0.0, Inf), ("C", 1.0, 0.01, 200.0, Inf)];
        T = 300.0, w0 = 200.0, w1 = 230.0
    )
    for m in ("A", "B", "C")
        @test r[m].share ≈ 1 / 3 atol = 0.01
        @test r[m].gap <= 0.04 + 1.0e-9
    end
end

# One randomized fair run, ported from the audit's bounds check (`sims/fair_bounds_check.jl`) to
# the real selector: random weights, a random longest dispatch C_i per model, each dispatch costing
# uniform(0.2, 1.0) * C_i, and models switching between backlogged and idle at random (model 1
# stays backlogged throughout). Returns the worst margin of each guarantee, each <= 0 when it holds:
#   inv    floor <= vtime_j <= floor + C_j / w_j for every backlogged model j, at every decision
#   wait   a model's wait for its next dispatch over sum_{j != i} (C_j + (w_j / w_i) C_i), minus 1
#   lag    |dW_i / w_i - dW_j / w_j| - (C_i / w_i + C_j / w_j), for the device time dW each of a
#          pair got while both stayed backlogged
#   share  the device time guaranteed to model 1 over the run, minus the time it got:
#          (w_1 / S) T - (w_1 / S) sum_{j != 1} ((w_j / w_1) C_1 + C_j), S the total weight of the
#          models that were ever backlogged
function _fair_property_run(rng; T = 100.0)
    n = rand(rng, 2:8)
    w = rand(rng, n) .* 3.5 .+ 0.5
    C = rand(rng, n) .* 0.199 .+ 0.001
    names = ["m$i" for i in 1:n]
    s = _empty_sched()
    states = [ModelSchedState(names[i], ModelSchedConfig(w[i]), 0.0) for i in 1:n]
    for i in 1:n
        _install!(s, _sched_entry(names[i], [1]), states[i])
    end
    index = Dict(names[i] => i for i in 1:n)
    busy = [i == 1 || rand(rng, Bool) for i in 1:n]
    next_flip = [i == 1 ? Inf : randexp(rng) * 5 for i in 1:n]
    ever = copy(busy)
    wait_start = [busy[i] ? 0.0 : NaN for i in 1:n]
    run_start = zeros(Int, n)              # decision at which the current backlogged run began
    served = zeros(n)
    history = Vector{Float64}[]            # `served` as it stood at each decision
    worst_inv, worst_wait, worst_lag = -Inf, -Inf, -Inf
    t = 0.0
    while t < T
        for i in 1:n                       # backlog changes take effect between dispatches
            if t >= next_flip[i]
                busy[i] = !busy[i]
                next_flip[i] = t + randexp(rng) * 5
                busy[i] && (ever[i] = true; wait_start[i] = t)
                busy[i] || (wait_start[i] = NaN)
            end
            q = states[i].queue
            busy[i] ? (isempty(q) && push!(q, _qr(names[i]; arrival = t))) : empty!(q)
        end
        push!(history, copy(served))
        k = length(history)
        for i in 1:n
            run_start[i] = busy[i] ? (run_start[i] == 0 ? k : run_start[i]) : 0
        end
        d = select_dispatch!(s, t)
        c = index[d.entry.name]
        F = s.vfloor
        for j in 1:n
            busy[j] || continue
            worst_inv = max(worst_inv, F - states[j].vtime, states[j].vtime - F - C[j] / w[j])
        end
        bound = sum(C[j] + (w[j] / w[c]) * C[c] for j in 1:n if j != c; init = 0.0)
        worst_wait = max(worst_wait, (t - wait_start[c]) / bound - 1)
        for i in 1:n, j in (i + 1):n
            (busy[i] && busy[j]) || continue
            k0 = max(run_start[i], run_start[j])
            dWi, dWj = served[i] - history[k0][i], served[j] - history[k0][j]
            worst_lag = max(worst_lag, abs(dWi / w[i] - dWj / w[j]) - (C[i] / w[i] + C[j] / w[j]))
        end
        cost = (0.2 + 0.8 * rand(rng)) * C[c]
        _charge_fair!(d.entry.sched, cost)
        t += cost
        served[c] += cost
        wait_start[c] = busy[c] ? t : NaN
    end
    S = sum(w[j] for j in 1:n if ever[j])
    guaranteed = (w[1] / S) * t - (w[1] / S) * sum((w[j] / w[1]) * C[1] + C[j] for j in 2:n if ever[j]; init = 0.0)
    return (inv = worst_inv, wait = worst_wait, lag = worst_lag, share = guaranteed - served[1])
end

@testset "fair: the wait bound, the share floor and the pairwise lag hold (randomized)" begin
    rng = MersenneTwister(20261105)
    worst = (inv = -Inf, wait = -Inf, lag = -Inf, share = -Inf)
    for _ in 1:40
        r = _fair_property_run(rng)
        worst = map(max, worst, r)
    end
    @test worst.inv <= 1.0e-9
    @test worst.wait <= 1.0e-9
    @test worst.lag <= 1.0e-9
    @test worst.share <= 1.0e-9
end

@testset "fair: set_policy! and SetModelPolicy refuse a weight that is not positive and finite" begin
    # A charge divides by the weight: 0 or Inf would stop or freeze a model's virtual time, a
    # negative weight would run it backwards, and NaN fails every comparison (SCH-18).
    s = _empty_sched()
    _install!(s, _sched_entry("m", [1]), ModelSchedState("m", ModelSchedConfig(1.0), 0.0))
    st = s.registry.by_name["m"].sched
    ReactantServer.set_policy!(s, "m"; weight = 2.5)
    @test st.weight == 2.5
    ctx = ReactantServer.InferContext(s, s.registry, ReactantServer.SharedMemoryRegistry(), "mock")
    GE = ReactantServer.gRPCServer.GRPCError
    for bad in (0.0, -1.0, NaN, Inf, -Inf)
        @test_throws ReactantServer.InvalidInputError ReactantServer.set_policy!(s, "m"; weight = bad)
        err = try
            ReactantServer._handle_set_model_policy(
                ctx, ReactantServer.control.SetModelPolicyRequest(; name = "m", has_weight = true, weight = bad)
            )
            nothing
        catch e
            e
        end
        @test err isa GE && err.code == ReactantServer.gRPCServer.StatusCode.INVALID_ARGUMENT
        @test st.weight == 2.5
    end
    # Without has_weight the weight field is not read, so its zero default is not an error.
    ReactantServer._handle_set_model_policy(ctx, ReactantServer.control.SetModelPolicyRequest(; name = "m"))
    @test st.weight == 2.5
end

@testset "FIFO discipline serves the globally oldest request, still coalescing" begin
    s = _empty_sched(; discipline = ReactantServer.FIFO)
    now = 1000.0
    # "hot" would dominate on fairness, but FIFO ignores fairness and serves the oldest arrival.
    hot = ModelSchedState("hot", ModelSchedConfig(1.0), now)
    hot.vtime = 0.0
    push!(hot.queue, _qr("hot"; arrival = now))                        # newer
    _install!(s, _sched_entry("hot", [1]; coalescable = false), hot)

    old = ModelSchedState("old", ModelSchedConfig(1.0), now)
    old.vtime = 100.0
    push!(old.queue, _qr("old"; arrival = now - 1.0))                  # older arrival wins under FIFO
    _install!(s, _sched_entry("old", [1]; coalescable = false), old)

    d = select_dispatch!(s, now)
    @test d.entry.name == "old"

    # FIFO still coalesces the chosen model's queued requests into one dispatch.
    s2 = _empty_sched(; discipline = ReactantServer.FIFO)
    stc = ModelSchedState("c", ModelSchedConfig(1.0), now)
    push!(stc.queue, _qr("c"; rows = 2, arrival = now), _qr("c"; rows = 2, arrival = now + 0.1))
    _install!(s2, _sched_entry("c", [1, 4, 8]; coalescable = true), stc)
    d2 = select_dispatch!(s2, now)
    @test d2.size == 4
    @test length(d2.taken) == 2
end

@testset "EDF serves the soonest-deadline model, degrading to FIFO on ties" begin
    s = _empty_sched(; discipline = ReactantServer.EDF)
    now = 1000.0
    # "old" arrived first but carries a far deadline; "urgent" arrived later with a sooner deadline.
    # EDF picks the soonest deadline regardless of arrival (the in-flight meta sub-call case).
    old = ModelSchedState("old", ModelSchedConfig(1.0), now)
    push!(old.queue, _qr("old"; arrival = now, deadline_ns = Int64(5_000_000_000)))
    _install!(s, _sched_entry("old", [1]; coalescable = false), old)
    urgent = ModelSchedState("urgent", ModelSchedConfig(1.0), now)
    push!(urgent.queue, _qr("urgent"; arrival = now + 1.0, deadline_ns = Int64(1_000_000_000)))
    _install!(s, _sched_entry("urgent", [1]; coalescable = false), urgent)
    @test select_dispatch!(s, now).entry.name == "urgent"

    # With equal deadlines (the uniform-deadline case) EDF degrades to FIFO: oldest arrival wins.
    s2 = _empty_sched(; discipline = ReactantServer.EDF)
    a = ModelSchedState("a", ModelSchedConfig(1.0), now)
    push!(a.queue, _qr("a"; arrival = now - 1.0, deadline_ns = Int64(9_000_000_000)))   # older
    _install!(s2, _sched_entry("a", [1]; coalescable = false), a)
    b = ModelSchedState("b", ModelSchedConfig(1.0), now)
    push!(b.queue, _qr("b"; arrival = now, deadline_ns = Int64(9_000_000_000)))         # same deadline, newer
    _install!(s2, _sched_entry("b", [1]; coalescable = false), b)
    @test select_dispatch!(s2, now).entry.name == "a"

    # Requests with no deadline (0) are least urgent: a deadline'd model is served first.
    s3 = _empty_sched(; discipline = ReactantServer.EDF)
    nod = ModelSchedState("nodeadline", ModelSchedConfig(1.0), now)
    push!(nod.queue, _qr("nodeadline"; arrival = now - 5.0))                          # oldest, but no deadline
    _install!(s3, _sched_entry("nodeadline", [1]; coalescable = false), nod)
    dl = ModelSchedState("hasdeadline", ModelSchedConfig(1.0), now)
    push!(dl.queue, _qr("hasdeadline"; arrival = now, deadline_ns = Int64(2_000_000_000)))
    _install!(s3, _sched_entry("hasdeadline", [1]; coalescable = false), dl)
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
