# Multi-shape input support: one model compiled for several input shapes (e.g. detector aspect
# ratios) keyed by variant, all sharing one weight set. Exercises the variant key derivation,
# executable selection, bundle discovery, and variant-grouped coalescing against the MockBackend.

const _RS = ReactantServer

# A model whose input "img" has shape "whn" (w=axis 1 and h=axis 2 variable, n=axis 3 batch) and a
# matching output "y". variant_spec = [("img",1),("img",2)] so the variant key is [w, h].
_ms_sig() = _RS.ModelSignature(["img"], DataType[Float32], String[], 1, ["y"], DataType[Float32],
    2, [("img", 1), ("img", 2)])

function _ms_manifest(name)
    whn(letters_var) = _RS.TensorSpec(letters_var[1], _RS.F32,
        _RS.Dim[_RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.BATCH)], 3)
    inimg = whn(["img"])
    outy = _RS.TensorSpec("y", _RS.F32,
        _RS.Dim[_RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.BATCH)], 3)
    # input_batch_dim 2 (0-based axis of n), two declared variants.
    return _RS.Manifest("2.0", name, "", _RS.TensorSpec[inimg], _RS.TensorSpec[outy],
        nothing, nothing, _RS.BatchingSpec(Int[1]), _RS.Provenance(Dict{String,Any}()),
        2, "model", String[], [[2, 3], [4, 2]])
end

function _ms_named(w, h, b)
    t = _RS.NamedTensor("img", zeros(Float32, w, h, b))
    return Dict(t.name => t)
end

@testset "multishape _select_exec routes by input shape" begin
    execA = _RS.MockExecutable(args -> [args[1]], 1)
    execA2 = _RS.MockExecutable(args -> [args[1]], 1)
    execB = _RS.MockExecutable(args -> [args[1]], 1)
    execs = Dict{_RS.VariantKey,Dict{Int,Any}}(
        [2, 3] => Dict{Int,Any}(1 => execA, 2 => execA2),
        [4, 2] => Dict{Int,Any}(1 => execB))
    model = _RS.LoadedModel(_ms_sig(), execs, Any[])

    @test _RS._select_exec(model, _ms_named(2, 3, 1)) === execA          # variant [2,3], batch 1
    @test _RS._select_exec(model, _ms_named(2, 3, 2)) === execA2         # variant [2,3], batch 2
    @test _RS._select_exec(model, _ms_named(4, 2, 1)) === execB          # variant [4,2]

    # An uncompiled shape is reported clearly, naming the compiled variants.
    err = try
        _RS._select_exec(model, _ms_named(5, 5, 1)); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("input shape variant [5, 5]", err.msg)

    # An uncompiled batch size within a multi-batch variant is also reported ([2,3] has only 1,2).
    @test_throws ErrorException _RS._select_exec(model, _ms_named(2, 3, 3))
end

@testset "multishape run_model executes the routed variant and shares weights" begin
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    shared = Any[_RS.MockBuffer(Float32[7])]                 # the one weight set, shared by both variants
    # Each variant's executable tags its output with a distinct constant, so the output proves
    # which program ran; both receive the same `shared` weight buffer as their trailing argument.
    execA = _RS.MockExecutable(args -> [fill(Float32(args[2][1]) + 1, size(args[1]))], 1)
    execB = _RS.MockExecutable(args -> [fill(Float32(args[2][1]) + 2, size(args[1]))], 1)
    execs = Dict{_RS.VariantKey,Dict{Int,Any}}(
        [2, 3] => Dict{Int,Any}(1 => execA),
        [4, 2] => Dict{Int,Any}(1 => execB))
    model = _RS.LoadedModel(_ms_sig(), execs, shared)

    outA = _RS.run_model(backend, pool, model, [_RS.NamedTensor("img", zeros(Float32, 2, 3, 1))])
    @test size(outA[1].data) == (2, 3, 1)
    @test all(==(8f0), outA[1].data)                         # 7 (shared weight) + 1 (variant A)

    outB = _RS.run_model(backend, pool, model, [_RS.NamedTensor("img", zeros(Float32, 4, 2, 1))])
    @test size(outB[1].data) == (4, 2, 1)
    @test all(==(9f0), outB[1].data)                         # 7 (same shared weight) + 2 (variant B)
end

@testset "multishape _discover_modules finds per-variant files" begin
    dir = mktempdir()
    write(joinpath(dir, "model.v0.b1.mlir"), UInt8[0x01])
    write(joinpath(dir, "model.v1.b1.mlir"), UInt8[0x02])
    m = _ms_manifest(basename(dir))

    mods = _RS._discover_modules(dir, m)
    @test Set(keys(mods)) == Set([[2, 3], [4, 2]])
    @test mods[[2, 3]][1] == UInt8[0x01]
    @test mods[[4, 2]][1] == UInt8[0x02]

    # A declared variant with no module file is rejected.
    rm(joinpath(dir, "model.v1.b1.mlir"))
    @test_throws _RS.BundleError _RS._discover_modules(dir, m)
end

@testset "multishape plan_batch groups coalescing by variant" begin
    backend = _RS.MockBackend()
    pool = _RS.MemoryPool(backend, _RS.MockClient(), _RS.MockDevice(0), "mock", nothing)
    echo = _RS.MockExecutable(args -> [args[1]], 1)
    execs = Dict{_RS.VariantKey,Dict{Int,Any}}(
        [2, 3] => Dict{Int,Any}(1 => echo, 2 => echo),       # variant A compiled for batch 1 and 2
        [4, 2] => Dict{Int,Any}(1 => echo))                  # variant B compiled for batch 1
    model = _RS.LoadedModel(_ms_sig(), execs, Any[])
    st = _RS.ModelSchedState("ms", _RS.ModelSchedConfig(1.0), 0.0)
    entry = _RS.ModelEntry("ms", _ms_manifest("ms"), Dict{Int,Vector{UInt8}}(),
        "", nothing, model, st, identity, identity)

    mk(w, h) = let req = _RS.InferRequest("ms", ["y"], [_RS.NamedTensor("img", zeros(Float32, w, h, 1))])
        _RS.QueuedRequest(req, req.inputs, 0.0, Channel{Any}(1))
    end
    # Queue: A, A, B. Only the leading same-variant run (the two A's) coalesces; B waits.
    qa1, qa2, qb = mk(2, 3), mk(2, 3), mk(4, 2)
    append!(st.queue, [qa1, qa2, qb])

    B, taken = _RS.plan_batch(entry, st)
    @test B == 2                                             # two A rows fill the batch-2 program
    @test taken == [qa1, qa2]

    # Finalizing removes the taken front entries; the differently-shaped B request remains and
    # then plans on its own variant at batch 1.
    _RS._finalize(entry, (B, taken))
    @test st.queue == [qb]
    B2, taken2 = _RS.plan_batch(entry, st)
    @test B2 == 1
    @test taken2 == [qb]
end

# A two-input model whose inputs carry the batch axis at DIFFERENT Julia positions (per-input,
# natural batch-last across ranks), compiled for two shape variants. Exercises the full coalescing
# data plane (plan_batch grouping by variant, _coalesce_inputs concatenating each input along its
# OWN axis, _executable_rows, and _slice_outputs splitting per request) and pins that coalescing
# stops at the variant boundary.
#   img: "whn" (w,h variable, batch 'n' at Julia axis 3)   -> drives the variant key [w, h]
#   vec: "dn"  (d fixed = 2, batch 'n' at Julia axis 2)     -> batch axis at a DIFFERENT position
#   y:   "dn"  (d fixed = 2, batch 'n' at Julia axis 2)
_pi_sig() = _RS.ModelSignature(["img", "vec"], DataType[Float32, Float32], String[], 1,
    ["y"], DataType[Float32], 2, [("img", 1), ("img", 2)])

function _pi_manifest(name)
    inimg = _RS.TensorSpec("img", _RS.F32,
        _RS.Dim[_RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.VARIABLE), _RS.Dim(_RS.BATCH)], 3)
    invec = _RS.TensorSpec("vec", _RS.F32, _RS.Dim[_RS.Dim(_RS.FIXED, 2), _RS.Dim(_RS.BATCH)], 2)
    outy = _RS.TensorSpec("y", _RS.F32, _RS.Dim[_RS.Dim(_RS.FIXED, 2), _RS.Dim(_RS.BATCH)], 2)
    # input_batch_dim 2 (0-based; img's 'n' at Julia axis 3), two declared variants.
    return _RS.Manifest("2.0", name, "", _RS.TensorSpec[inimg, invec], _RS.TensorSpec[outy],
        nothing, nothing, _RS.BatchingSpec(Int[1, 2]), _RS.Provenance(Dict{String,Any}()),
        2, "model", String[], [[2, 3], [4, 2]])
end

@testset "per-input batch axis: coalesce within a variant, split at the boundary" begin
    echo = _RS.MockExecutable(args -> [args[1]], 1)
    execs = Dict{_RS.VariantKey,Dict{Int,Any}}(
        [2, 3] => Dict{Int,Any}(1 => echo, 2 => echo),       # variant A: batch 1 and 2
        [4, 2] => Dict{Int,Any}(1 => echo))                  # variant B: batch 1
    model = _RS.LoadedModel(_pi_sig(), execs, Any[])
    st = _RS.ModelSchedState("pi", _RS.ModelSchedConfig(1.0), 0.0)
    entry = _RS.ModelEntry("pi", _pi_manifest("pi"), Dict{Int,Vector{UInt8}}(),
        "", nothing, model, st, identity, identity)

    @test _RS._coalescable(entry)                            # both inputs and the output are batched

    mk(w, h) = let req = _RS.InferRequest("pi", ["y"],
            [_RS.NamedTensor("img", zeros(Float32, w, h, 1)),
             _RS.NamedTensor("vec", zeros(Float32, 2, 1))])
        _RS.QueuedRequest(req, req.inputs, 0.0, Channel{Any}(1))
    end
    # Queue A, A, B: only the leading same-variant run (the two A's) coalesces; B stops the window.
    qa1, qa2, qb = mk(2, 3), mk(2, 3), mk(4, 2)
    append!(st.queue, [qa1, qa2, qb])
    B, taken = _RS.plan_batch(entry, st)
    @test B == 2
    @test taken == [qa1, qa2]                                # variant boundary excludes qb

    # _executable_rows reads img's OWN axis (3), not a shared position: one row per request.
    @test _RS._executable_rows(entry, qa1.prepared) == 1

    # _coalesce_inputs concatenates each input along its own batch axis (img axis 3, vec axis 2).
    pres = Vector{_RS.NamedTensor}[qa1.prepared, qa2.prepared]
    merged = _RS._coalesce_inputs(entry, pres, 2, 2)
    @test size(_RS._named(merged, "img").data) == (2, 3, 2)  # concatenated along axis 3
    @test size(_RS._named(merged, "vec").data) == (2, 2)     # concatenated along axis 2

    # _slice_outputs splits a batch-2 output back per request along the output's own axis (2).
    out = [_RS.NamedTensor("y", Float32[1 2; 10 20])]        # column j is request j's row
    s1 = _RS._slice_outputs(entry, out, 0, 1)
    s2 = _RS._slice_outputs(entry, out, 1, 1)
    @test vec(_RS._named(s1, "y").data) == Float32[1, 10]
    @test vec(_RS._named(s2, "y").data) == Float32[2, 20]

    # The differently-shaped B request remains and then plans alone on its own variant at batch 1.
    _RS._finalize(entry, (B, taken))
    @test st.queue == [qb]
    B2, taken2 = _RS.plan_batch(entry, st)
    @test B2 == 1
    @test taken2 == [qb]
end
