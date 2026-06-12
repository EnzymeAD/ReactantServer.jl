# run_model and the scheduler against the Reactant-free MockBackend.

function _trivial_manifest(name)
    ReactantServer.Manifest("2.0", name, "",
        ReactantServer.TensorSpec[], ReactantServer.TensorSpec[], nothing, nothing,
        ReactantServer.BatchingSpec(Int[]), ReactantServer.Provenance(Dict{String,Any}()), nothing)
end

function _scale_model()
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 0)
    weights = Any[ReactantServer.MockBuffer(Float32[2, 2, 2, 2])]
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)   # x .* w
    return ReactantServer.LoadedModel(sig, Dict{Int,Any}(0 => exec), weights)
end

@testset "mock run_model" begin
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    model = _scale_model()
    out = ReactantServer.run_model(backend, pool, model, [ReactantServer.NamedTensor("x", Float32[1, 2, 3, 4])])
    @test length(out) == 1
    @test out[1].name == "y"
    @test out[1].data == Float32[2, 4, 6, 8]

    # missing input is an error
    @test_throws ErrorException ReactantServer.run_model(backend, pool, model,
        [ReactantServer.NamedTensor("wrong", Float32[1, 2, 3, 4])])
end

@testset "mock scheduler" begin
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    entry = ReactantServer.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(), "", nothing,
                                 _scale_model(), nothing, identity, identity)
    reg.by_name["scale"] = entry

    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
    ReactantServer.start!(sched)

    req = ReactantServer.InferRequest("scale", ["y"], [ReactantServer.NamedTensor("x", Float32[1, 2, 3, 4])])
    @test ReactantServer.infer(sched, req)[1].data == Float32[2, 4, 6, 8]

    # several in a row are serialized through the single dispatch task
    for k in 1:5
        r = ReactantServer.InferRequest("scale", ["y"], [ReactantServer.NamedTensor("x", fill(Float32(k), 4))])
        @test ReactantServer.infer(sched, r)[1].data == fill(Float32(2k), 4)
    end

    # unknown model surfaces as a thrown error
    bad = ReactantServer.InferRequest("nope", String[], [ReactantServer.NamedTensor("x", Float32[1])])
    @test_throws Exception ReactantServer.infer(sched, bad)

    # The control-plane snapshot carries the cumulative serving counters (packing cost source):
    # 6 requests above, each dispatched alone (non-coalescable trivial manifest).
    snap = ReactantServer.control_status(sched)
    @test snap.models["scale"].requests_served == 6
    @test snap.models["scale"].dispatch_count == 6
    @test snap.models["scale"].total_compute > 0
end

# A batched mock model: x and y are (features=2, n) with the batch axis last; y = x .* w. The
# same broadcast executable serves every compiled size, so one fn covers both keys.
function _batched_scale_model(sizes::Vector{Int})
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    weights = Any[ReactantServer.MockBuffer(Float32[10, 100])]
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    return ReactantServer.LoadedModel(sig, Dict{Int,Any}(sz => exec for sz in sizes), weights)
end

function _batched_manifest(name)
    inx = ReactantServer.TensorSpec("x", ReactantServer.F32,
        ReactantServer.Dim[ReactantServer.Dim(ReactantServer.FIXED, 2), ReactantServer.Dim(ReactantServer.BATCH)], 2)
    outy = ReactantServer.TensorSpec("y", ReactantServer.F32,
        ReactantServer.Dim[ReactantServer.Dim(ReactantServer.FIXED, 2), ReactantServer.Dim(ReactantServer.BATCH)], 2)
    return ReactantServer.Manifest("2.0", name, "", ReactantServer.TensorSpec[inx], ReactantServer.TensorSpec[outy],
        nothing, nothing, ReactantServer.BatchingSpec(Int[]), ReactantServer.Provenance(Dict{String,Any}()), 1)
end

function _batched_scheduler(name, sizes)
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    reg.by_name[name] = ReactantServer.ModelEntry(name, _batched_manifest(name), Dict{Int,Vector{UInt8}}(),
        "", nothing, _batched_scale_model(sizes), nothing, identity, identity)
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    reg.by_name[name].sched = ReactantServer.ModelSchedState(name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    return sched
end

# Drive select_dispatch! + execute_and_record! directly (no spawned loop) so coalescing is
# deterministic rather than dependent on arrival timing.
@testset "mock scheduler coalesces queued requests into one dispatch" begin
    sched = _batched_scheduler("bscale", [1, 4])
    qrs = [ReactantServer.QueuedRequest(
               ReactantServer.InferRequest("bscale", ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[k, k], 2, 1))]),
               0.0, Channel{Any}(1)) for k in 1:4]
    for qr in qrs
        push!(sched.registry.by_name["bscale"].sched.queue, qr)
    end

    d = ReactantServer.select_dispatch!(sched, 0.0)
    @test d.size == 4                       # four single rows fill the batch-4 executable
    @test length(d.taken) == 4
    ReactantServer.execute_and_record!(sched, d)

    for (k, qr) in enumerate(qrs)
        res = take!(qr.reply)
        @test res isa Vector{ReactantServer.NamedTensor}
        @test res[1].data == reshape(Float32[10k, 100k], 2, 1)   # each caller gets its own slice
    end
    @test sched.registry.by_name["bscale"].sched.dispatch_count == 1             # one execution served all four
    @test sched.registry.by_name["bscale"].sched.batch_size_hist[4] == 1
end

@testset "mock scheduler pads a partial-fill coalesced dispatch" begin
    sched = _batched_scheduler("pad", [4])                       # only a batch-4 executable
    qr = ReactantServer.QueuedRequest(
        ReactantServer.InferRequest("pad", ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[3, 5], 2, 1))]),
        0.0, Channel{Any}(1))
    push!(sched.registry.by_name["pad"].sched.queue, qr)

    d = ReactantServer.select_dispatch!(sched, 0.0)
    @test d.size == 4                       # one row padded up to the smallest compiled size
    @test length(d.taken) == 1
    ReactantServer.execute_and_record!(sched, d)

    res = take!(qr.reply)
    @test res[1].data == reshape(Float32[30, 500], 2, 1)         # padding rows dropped from the result
    @test sched.registry.by_name["pad"].sched.batch_size_hist[4] == 1
end

@testset "_validate_inputs rejects malformed requests before dispatch" begin
    entry = ReactantServer.ModelEntry("v", _batched_manifest("v"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _batched_scale_model([1, 4]), nothing, identity, identity)
    _req(tensors) = ReactantServer.InferRequest("v", ["y"], tensors)
    _ok(rows) = ReactantServer.NamedTensor("x", zeros(Float32, 2, rows))
    GE = ReactantServer.gRPCServer.gRPCServiceCallException

    # Valid request passes; any batch extent is accepted on the batch axis.
    @test ReactantServer._validate_inputs(entry, _req([_ok(1)])) === nothing
    @test ReactantServer._validate_inputs(entry, _req([_ok(7)])) === nothing

    # Undeclared input name.
    @test_throws GE ReactantServer._validate_inputs(entry,
        _req([ReactantServer.NamedTensor("bogus", zeros(Float32, 2, 1))]))
    # Wrong dtype.
    @test_throws GE ReactantServer._validate_inputs(entry,
        _req([ReactantServer.NamedTensor("x", zeros(Float64, 2, 1))]))
    # Wrong rank.
    @test_throws GE ReactantServer._validate_inputs(entry,
        _req([ReactantServer.NamedTensor("x", zeros(Float32, 2))]))
    # Wrong fixed-axis extent.
    @test_throws GE ReactantServer._validate_inputs(entry,
        _req([ReactantServer.NamedTensor("x", zeros(Float32, 3, 1))]))
    # Missing required input.
    @test_throws GE ReactantServer._validate_inputs(entry, _req(ReactantServer.NamedTensor[]))
end
