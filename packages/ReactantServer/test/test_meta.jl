# Meta models: a Julia orchestration (a bundle's model.jl that calls register_meta_model) chains
# other models with data-dependent logic. A meta runs its orchestration on the request task under a
# one-meta-at-a-time gate, and each sub-call re-enters the local scheduler in-process via a
# QueueingCaller (no gateway, no shared-memory transport): the sub-model dispatches on the loop as a
# committed request. These tests exercise that path: manifest/bundle loading, the injected caller, the
# declared-call guard, the deadline bail, and the cross-bundle recursion check.

const _RS = ReactantServer

# y = x .* w (w = 2), unbatched — the backbone stand-in the meta models call.
function _meta_scale_model()
    sig = _RS.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 0)
    weights = Any[_RS.MockBuffer(Float32[2, 2, 2, 2])]
    exec = _RS.MockExecutable(args -> [args[1] .* args[2]], 1)
    return _RS.LoadedModel(sig, Dict{Int,Any}(0 => exec), weights)
end

_meta_manifest(name, calls) = _RS.parse_manifest(Dict{String,Any}(
    "format_version" => "2.0", "name" => name, "kind" => "meta",
    "meta" => Dict("calls" => collect(String, calls)),
    "client_inputs" => [Dict("name" => "x", "dtype" => "f32", "shape" => "c", "dims" => Dict("c" => 4))],
    "client_outputs" => [Dict("name" => "OUT", "dtype" => "f32", "shape" => "c", "dims" => Dict("c" => 4))]))

# A registry hosting the "scale" backbone, a started scheduler over it, and a QueueingCaller a meta
# uses to run sub-models: each sub-call re-enters the started scheduler in-process (the loop dispatches
# the committed sub-model request). `nothing` for the scratch pool -> `call.scratch` returns plain arrays.
function _meta_local_caller()
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    reg.by_name["scale"] = _RS.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    _RS.start!(sched)
    return sched, _RS.QueueingCaller(sched, nothing)
end

@testset "meta manifest parse + validate" begin
    m = _meta_manifest("detector", ["scale", "refine"])
    @test _RS.is_meta(m)
    @test m.kind == "meta"
    @test m.meta_calls == ["scale", "refine"]
    @test _RS.validate_manifest(m, "/tmp/detector", true) === m

    # A meta model requires model.jl.
    @test_throws _RS.ManifestError _RS.validate_manifest(m, "/tmp/detector", false)
    # client_inputs/outputs are mandatory for a meta model (no executable fallback).
    @test_throws _RS.ManifestError _RS.parse_manifest(Dict{String,Any}(
        "format_version" => "2.0", "name" => "d", "kind" => "meta",
        "meta" => Dict("calls" => ["scale"]))) |> (m -> _RS.validate_manifest(m, "/tmp/d", true))
    # A meta model may not list itself.
    @test_throws _RS.ManifestError _RS.validate_manifest(_meta_manifest("d", ["d"]), "/tmp/d", true)
    # A compute-only meta model may declare an empty calls list (does all work in Julia).
    let mc = _meta_manifest("compute_only", String[])
        @test mc.meta_calls == String[]
        @test _RS.validate_manifest(mc, "/tmp/compute_only", true) === mc
    end
end

@testset "meta model runs its sub-models in-process via QueueingCaller" begin
    sched, caller = _meta_local_caller()

    # The orchestration calls the backbone (x .* 2), then branches on the data: scale again when the
    # sum is small, otherwise add one. This is the data-dependent step torch.export cannot trace.
    run = function (inputs, call)
        y = call("scale", inputs)[1].data            # x .* 2
        out = sum(y) > 100 ? y .+ 1 : y .* 2
        return [_RS.NamedTensor("OUT", out)]
    end
    meta = _RS.MetaEntry("detector", _meta_manifest("detector", ["scale"]), ["scale"], run)

    # small branch: scale([1,2,3,4]) = [2,4,6,8] (sum 20 <= 100) -> *2 = [4,8,12,16]
    out = _RS.run_meta(meta, caller, [_RS.NamedTensor("x", Float32[1, 2, 3, 4])])
    @test out isa Vector{_RS.NamedTensor}
    @test out[1].name == "OUT"
    @test out[1].data == Float32[4, 8, 12, 16]

    # large branch: scale([40,40,40,40]) = [80,80,80,80] (sum 320 > 100) -> +1 = [81,...]
    out2 = _RS.run_meta(meta, caller, [_RS.NamedTensor("x", fill(Float32(40), 4))])
    @test out2[1].data == fill(Float32(81), 4)

    _RS.shutdown!(sched)
end

@testset "meta call guard rejects undeclared sub-calls" begin
    sched, caller = _meta_local_caller()
    run = (inputs, call) -> call("not_declared", inputs)   # callee absent from meta.calls
    meta = _RS.MetaEntry("detector", _meta_manifest("detector", ["scale"]), ["scale"], run)
    err = try
        _RS.run_meta(meta, caller, [_RS.NamedTensor("x", Float32[1, 2, 3, 4])])
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("undeclared", sprint(showerror, err))

    # A meta run that returns the wrong type is rejected.
    bad = _RS.MetaEntry("d2", _meta_manifest("d2", ["scale"]), ["scale"], (i, c) -> "nope")
    @test_throws ErrorException _RS.run_meta(bad, caller, [_RS.NamedTensor("x", Float32[1])])

    _RS.shutdown!(sched)
end

@testset "scheduler drops a request whose deadline has already passed" begin
    sched, _ = _meta_local_caller()
    x = [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]
    # An absolute deadline in the past is dropped at admission (before any GPU work) and surfaces as
    # DeadlineExceeded; the dispatch loop never runs the executable for it.
    expired = _RS.InferRequest("scale", String[], x, Int64(time_ns()) - Int64(1_000_000))
    @test_throws _RS.DeadlineExceeded _RS.infer(sched, expired)
    # A generous future deadline runs normally (x .* 2).
    live = _RS.InferRequest("scale", String[], x, Int64(time_ns()) + Int64(60_000_000_000))
    @test _RS.infer(sched, live)[1].data == Float32[2, 4, 6, 8]
    # No deadline (0) is unaffected.
    @test _RS.infer(sched, _RS.InferRequest("scale", String[], x))[1].data == Float32[2, 4, 6, 8]
    _RS.shutdown!(sched)
end

@testset "meta bails before a sub-call once its deadline has passed" begin
    sched, caller = _meta_local_caller()
    started = Ref(false)
    run = function (inputs, call)
        started[] = true
        return call("scale", inputs)        # the bail happens here, at the call boundary
    end
    meta = _RS.MetaEntry("detector", _meta_manifest("detector", ["scale"]), ["scale"], run)
    x = [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]
    past = Int64(time_ns()) - Int64(1_000_000)
    @test_throws _RS.DeadlineExceeded _RS.run_meta(meta, caller, x; deadline_ns = past)
    @test started[]                          # the orchestration ran; only the sub-call was refused
    # A generous deadline completes normally.
    fut = Int64(time_ns()) + Int64(60_000_000_000)
    @test _RS.run_meta(meta, caller, x; deadline_ns = fut)[1].data == Float32[2, 4, 6, 8]
    _RS.shutdown!(sched)
end

# Write a minimal meta bundle (manifest.yaml + model.jl) under `root/name`.
function _write_meta_bundle(root, name, calls; with_model_jl=true, run_body=nothing)
    dir = joinpath(root, name)
    mkpath(dir)
    calls_yaml = "[" * join(calls, ", ") * "]"
    write(joinpath(dir, "manifest.yaml"), """
    format_version: "2.0"
    name: $name
    kind: meta
    meta:
      calls: $calls_yaml
    client_inputs:
      - {name: x, dtype: f32, shape: c, dims: {c: 4}}
    client_outputs:
      - {name: OUT, dtype: f32, shape: c, dims: {c: 4}}
    """)
    if with_model_jl
        body = run_body === nothing ?
            "_run(inputs, call) = [ReactantServer.NamedTensor(\"OUT\", call(\"scale\", inputs)[1].data .* 2)]" :
            run_body
        write(joinpath(dir, "model.jl"), """
        $body
        register_meta_model("$name"; run=_run)
        """)
    end
    return dir
end

@testset "meta bundle loads into a MetaEntry" begin
    root = mktempdir()
    dir = _write_meta_bundle(root, "detector", ["scale"])
    entry = _RS.load_bundle_entry(dir)
    @test entry isa _RS.MetaEntry
    @test entry.name == "detector"
    @test entry.calls == ["scale"]

    # A meta bundle with no model.jl is rejected at load (validate_manifest enforces it).
    bad = _write_meta_bundle(root, "no_jl", ["scale"]; with_model_jl=false)
    @test_throws _RS.ManifestError _RS.load_bundle_entry(bad)

    # A meta model.jl that calls register_model (not register_meta_model) is rejected.
    wrongreg = _write_meta_bundle(root, "wrongreg", ["scale"];
        run_body="register_model(\"wrongreg\")")
    # The model.jl above never defines _run; register_model is called instead of register_meta_model.
    write(joinpath(wrongreg, "model.jl"), "register_model(\"wrongreg\")\n")
    @test_throws _RS.BundleError _RS.load_bundle_entry(wrongreg)
end

@testset "load_bundles rejects a meta model calling another meta model" begin
    root = mktempdir()
    _write_meta_bundle(root, "a", ["b"])
    _write_meta_bundle(root, "b", ["scale"])
    @test_throws _RS.BundleError _RS.load_bundles([root])
end

@testset "meta runs on the request task with in-process sub-calls" begin
    # End-to-end through the scheduler: register the backbone and a meta, start the loop, and submit
    # the meta via `infer`. `infer` runs the orchestration on the calling task under the meta gate; its
    # sub-call re-enters the scheduler and dispatches the backbone on the loop. Confirms a meta returns
    # its assembled outputs and is shed at admission when its deadline has already passed.
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    reg.by_name["scale"] = _RS.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    run = (inputs, call) -> [_RS.NamedTensor("OUT", call("scale", inputs)[1].data .+ 1)]
    reg.meta["det"] = _RS.MetaEntry("det", _meta_manifest("det", ["scale"]), ["scale"], run)
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    _RS.start!(sched)
    try
        out = _RS.infer(sched, _RS.InferRequest("det", String[], [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]))
        @test out isa Vector{_RS.NamedTensor}
        @test out[1].name == "OUT"
        @test out[1].data == Float32[3, 5, 7, 9]   # (x .* 2) .+ 1
        # A past-deadline meta request is dropped at admission (before taking a gate permit or running
        # the backbone) and surfaces as DeadlineExceeded.
        past = _RS.InferRequest("det", String[], [_RS.NamedTensor("x", Float32[1, 2, 3, 4])],
                                Int64(time_ns()) - Int64(1_000_000))
        @test_throws _RS.DeadlineExceeded _RS.infer(sched, past)
        # The meta's serving counters are recorded for the control plane (one successful run above).
        @test sched.registry.meta["det"].sched.requests_served == 1
        @test sched.registry.meta["det"].sched.total_compute >= 0.0
    finally
        _RS.shutdown!(sched)
    end
end

@testset "meta gate admits one meta at a time; a slow meta does not block regular models" begin
    # While a meta is mid-orchestration (paused in its CPU glue, holding the gate but NOT the GPU), a
    # concurrently submitted regular request to a different model is still served. This is the core win:
    # the GPU is free during the meta's glue.
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    reg.by_name["scale"] = _RS.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    reg.by_name["other"] = _RS.ModelEntry("other", _trivial_manifest("other"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    gate_open = Channel{Nothing}(1)   # released by the test once the regular request has been served
    glue_entered = Channel{Nothing}(1)
    run = function (inputs, call)
        y = call("scale", inputs)[1].data    # one GPU stage, then "glue" that parks
        put!(glue_entered, nothing)
        take!(gate_open)                     # hold the gate here, off the GPU, until the test signals
        return [_RS.NamedTensor("OUT", y .+ 1)]
    end
    reg.meta["det"] = _RS.MetaEntry("det", _meta_manifest("det", ["scale"]), ["scale"], run)
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    _RS.start!(sched)
    try
        meta_task = Threads.@spawn _RS.infer(sched,
            _RS.InferRequest("det", String[], [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]))
        take!(glue_entered)                  # the meta is now parked in its glue, holding the gate
        # A regular request to another model is served while the meta holds the gate (GPU is free).
        ot = _RS.infer(sched, _RS.InferRequest("other", String[], [_RS.NamedTensor("x", Float32[5, 6, 7, 8])]))
        @test ot[1].data == Float32[10, 12, 14, 16]
        put!(gate_open, nothing)             # let the meta finish
        out = fetch(meta_task)
        @test out[1].data == Float32[3, 5, 7, 9]
    finally
        _RS.shutdown!(sched)
    end
end

@testset "compute-only meta bypasses the gate" begin
    # A compute-only meta (empty calls) issues no sub-calls, so it must NOT take a gate permit: it runs
    # even while every gate permit is held by a GPU meta parked in its glue. Capacity-1 gate makes the
    # contention unambiguous.
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    reg.by_name["scale"] = _RS.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    hold = Channel{Nothing}(1)
    entered = Channel{Nothing}(1)
    gpu_run = function (inputs, call)
        call("scale", inputs)
        put!(entered, nothing)
        take!(hold)                                 # park while holding the single gate permit
        return [_RS.NamedTensor("OUT", inputs[1].data)]
    end
    reg.meta["gpu_meta"] = _RS.MetaEntry("gpu_meta", _meta_manifest("gpu_meta", ["scale"]), ["scale"], gpu_run)
    # Empty calls -> compute-only; does all work in Julia and touches no sub-model.
    co_run = (inputs, call) -> [_RS.NamedTensor("OUT", inputs[1].data .* 3)]
    reg.meta["compute_only"] = _RS.MetaEntry("compute_only", _meta_manifest("compute_only", String[]), String[], co_run)
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    sched.meta_gate = _RS.MetaGate(1)
    _RS.start!(sched)
    try
        gpu_task = Threads.@spawn _RS.infer(sched,
            _RS.InferRequest("gpu_meta", String[], [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]))
        take!(entered)                              # the GPU meta now holds the only permit
        # The compute-only meta runs to completion despite the gate being fully held.
        co = _RS.infer(sched, _RS.InferRequest("compute_only", String[], [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]))
        @test co[1].data == Float32[3, 6, 9, 12]
        put!(hold, nothing)
        @test fetch(gpu_task)[1].data == Float32[1, 2, 3, 4]
    finally
        _RS.shutdown!(sched)
    end
end

@testset "committed sub-calls track the gate (each in-flight meta cuts the line)" begin
    # The committed set is sized to the gate: with two metas in flight, both their sub-calls are
    # committed and both cut the line (no single-slot overwrite). Drive selection by hand (loop not
    # started) so the ordering is deterministic.
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    for nm in ("subA", "subB")
        reg.by_name[nm] = _RS.ModelEntry(nm, _trivial_manifest(nm), Dict{Int,Vector{UInt8}}(),
            "", nothing, _meta_scale_model(), nothing, identity, identity)
    end
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    now = time()
    for e in values(reg.by_name)
        _RS.init_sched_state!(sched, e, now)
    end
    sched.running = true     # allow submit!; the dispatch loop is NOT started, so we select by hand
    x = [_RS.NamedTensor("x", Float32[1, 2, 3, 4])]
    qrA = _RS.QueuedRequest(_RS.InferRequest("subA", String[], x); committed=true)
    qrB = _RS.QueuedRequest(_RS.InferRequest("subB", String[], x); committed=true)
    _RS.submit!(sched, qrA)
    _RS.submit!(sched, qrB)
    @test length(sched.committed) == 2          # both retained; no single-slot overwrite
    d1 = _RS.select_dispatch!(sched, time())
    d2 = _RS.select_dispatch!(sched, time())
    @test d1 isa _RS.Dispatch && d2 isa _RS.Dispatch
    @test Set([d1.entry.name, d2.entry.name]) == Set(["subA", "subB"])   # both jumped the line
    @test isempty(sched.committed)
    @test _RS.select_dispatch!(sched, time()) === nothing                # nothing left to dispatch
end
