# Meta models: a Julia orchestration (a bundle's model.jl that calls register_meta_model) chains
# other models with data-dependent logic. These tests exercise the worker-mode path (LocalCaller,
# no gateway): manifest/bundle loading, the injected caller, the declared-call guard, and the
# cross-bundle recursion check. The multi-worker GatewayCaller path is covered by the e2e suite.

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

# A scheduler hosting the "scale" backbone, plus a LocalCaller over it.
function _meta_local_caller()
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    reg = _RS.ModelRegistry()
    reg.by_name["scale"] = _RS.ModelEntry("scale", _trivial_manifest("scale"), Dict{Int,Vector{UInt8}}(),
        "", nothing, _meta_scale_model(), nothing, identity, identity)
    sched = _RS.Scheduler(reg, backend, pool, _RS.SchedulerConfig(30.0, 64, 30.0))
    _RS.start!(sched)
    return sched, _RS.LocalCaller(sched)
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

@testset "meta model runs in worker mode via LocalCaller" begin
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

@testset "build_caller defaults to LocalCaller with no loopback configured" begin
    sched, _ = _meta_local_caller()
    withenv(_RS.LOOPBACK_ENV => nothing) do
        @test _RS.build_caller(sched) isa _RS.LocalCaller
    end
    _RS.shutdown!(sched)
end
