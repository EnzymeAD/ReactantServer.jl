# ============================================================================
# Two-stage detectors as one program
# ============================================================================

"""
    stablehlo_text(path_or_bytes) -> String

The textual StableHLO of a serialized portable artifact (a bundle's `model*.mlir`, given as a file
path or its bytes), for `Ops.hlo_call`.
"""
stablehlo_text(path::AbstractString) = stablehlo_text(read(String(path)))
function stablehlo_text(bytes::AbstractVector{UInt8})
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    try
        mref = MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(bytes)), ctx)
        return string(MLIR.IR.Module(mref))
    finally
        MLIR.IR.deactivate(ctx)
    end
end

"""
    read_stage_bundle(dir) -> (; texts, weights, input_shapes)

Read a stage bundle's programs and weights back for fusing.

Returns the pieces [`export_two_stage_detector`](@ref) takes for one stage: `texts` maps each
input-shape variant key (`Int[]` for a single-shape bundle, otherwise the manifest `input_shapes`
entry in (input, axis) order of the variable axes) to the StableHLO text of its program, `weights`
is the `name => array` list in `argument_order` with each array in the axis order of the program's
StableHLO signature (for a torch export, torch order), which is what `Ops.hlo_call` takes, and
`input_shapes` lists the variant keys in manifest order. Each variant must carry exactly one module
(one batch size).
"""
function read_stage_bundle(dir::AbstractString)
    d = String(dir)
    manifest = YAML.load_file(joinpath(d, "manifest.yaml"))
    vletters = String[]
    for spec in manifest["executable_inputs"], c in spec["shape"]
        c in ('n', 'b') && continue
        get(spec["dims"], string(c), 0) == -1 && push!(vletters, string(c))
    end
    shapes = [Int[v[l] for l in vletters] for v in get(manifest, "input_shapes", Any[])]
    files = filter(f -> endswith(f, ".mlir"), readdir(d))
    one_module(pat) = begin
        hits = filter(f -> occursin(pat, f), files)
        length(hits) == 1 ||
            error("ReactantServerExport: expected one module matching $pat in `$d`, found $(hits)")
        joinpath(d, only(hits))
    end
    texts = Dict{Vector{Int}, String}()
    if isempty(shapes)
        texts[Int[]] = stablehlo_text(one_module(r"^model(\.b\d+)?\.mlir$"))
    else
        for (i, v) in enumerate(shapes)
            texts[v] = stablehlo_text(one_module(Regex("^model\\.v$(i - 1)(\\.b\\d+)?\\.mlir\$")))
        end
    end
    st = SafeTensors.deserialize(joinpath(d, "weights.safetensors"); mmap = false)
    order = JSON3.read(st.metadata["argument_order"], Vector{String})
    # A bundle stores each weight as the Julia array the server hands to the executable, whose
    # StableHLO entry signature lists the axes in reverse (row-major) order. `Ops.hlo_call` on the
    # program text takes arrays in the signature's order, so reverse the axes.
    unview(a) = ndims(a) <= 1 ? collect(a) : permutedims(collect(a), ndims(a):-1:1)
    return (; texts, weights = Pair{String, Any}[n => unview(st[n]) for n in order], input_shapes = shapes)
end

const _TRIM_MODEL_JL = """
# Two-stage detector exported as one StableHLO program: stage1 (backbone + RPN head), RPN proposal
# selection + NMS, ROIAlign, stage2 (box head), and per-class NMS. The program returns a fixed-size,
# zero-padded DETECTIONS buffer plus NUM_DETECTIONS; this hook trims the buffer to the detections
# found and returns it as OUTPUT__0.
function _trim(outputs)
    by = Dict(t.name => t.data for t in outputs)
    n = Int(by["NUM_DETECTIONS"][1])
    d = by["DETECTIONS"]
    out = ndims(d) == 2 ? d[:, 1:n] : d[:, 1:n, :]
    return [ReactantServer.NamedTensor("OUTPUT__0", Array{Float32}(out))]
end

register_model(basename(@__DIR__); postprocess = _trim)
"""

"""
    export_two_stage_detector(dir; name, input, stage1, stage1_weights, stage2, stage2_weights, cfg,
                              input_shapes=nothing, output_columns=6, output_layout=:cn,
                              glue_precision=Float64, provenance=Dict()) -> dir

Export a two-stage detector as one plain bundle.

Stage1, the traced `ReactantServerExport.Detection` glue, stage2, and the per-class NMS compile to a
single StableHLO program per input shape, and a `model.jl` postprocess hook trims the program's
fixed-size detection buffer.

- `input`: the client image [`IOSpec`](@ref), for example
  `IOSpec("INPUT__0", UInt8, [512, 512, 1]; letters = ['w', 'h', 'c'])`. Its first two Julia axes
  are the image width and height (the glue clips boxes to them); the reversed array is what stage1
  takes. With `input_shapes`, the variable axes are `-1`. Batched inputs are not supported (one
  program handles one image).
- `stage1`: `Dict` from variant key to stage1 StableHLO text: `Int[]` for a single-shape detector,
  otherwise one entry per `input_shapes` element. Stage1 returns `(feats..., objectness...,
  deltas...)` in torch layout, see `ReactantServerExport.Detection`.
- `stage2`: stage2 StableHLO text, `(K, C, pooled, pooled)` ROI features to
  `(cls_logits, bbox_deltas)`, with `K == cfg.post_nms_topk`.
- `stage1_weights`, `stage2_weights`: `name => array` pairs in each program's argument order
  ([`read_stage_bundle`](@ref) returns both halves from an existing stage bundle). They are written
  to `weights.safetensors` prefixed `stage1.` and `stage2.` and stay program ARGUMENTS; the entry
  arity is checked against them before anything is written.
- `output_columns`: 6 for rows `[x1, y1, x2, y2, score, class]`, 5 for `[x1, y1, x2, y2, score]`.
- `output_layout`: `:cn` returns `OUTPUT__0` as `(columns, N)`; `:cn1` as `(columns, N, 1)`.

The executable returns `DETECTIONS` (`(columns, detections_per_img)` in the chosen layout, zero
past the count) and `NUM_DETECTIONS` (`Int64`, length 1); `client_outputs` declares the trimmed
`OUTPUT__0` with a variable detection axis. `dir` should not already hold a bundle: files from an
earlier export with more variants would be left behind.
"""
function export_two_stage_detector(
        dir::AbstractString; name::AbstractString, input::IOSpec,
        stage1::AbstractDict, stage1_weights, stage2::AbstractString, stage2_weights,
        cfg::Detection.DetectorConfig, input_shapes::Union{Nothing, AbstractVector} = nothing,
        output_columns::Integer = 6, output_layout::Symbol = :cn,
        glue_precision::Type = Float64, provenance = Dict{String, Any}()
    )
    output_columns in (5, 6) || error("ReactantServerExport: output_columns must be 5 or 6, got $output_columns")
    output_layout in (:cn, :cn1) || error("ReactantServerExport: output_layout must be :cn or :cn1, got $output_layout")
    input.batch_axis === nothing ||
        error("ReactantServerExport: export_two_stage_detector handles one image per call; `input` must be unbatched")
    length(input.shape) >= 2 || error("ReactantServerExport: `input` needs at least (W, H) axes")
    variants = input_shapes === nothing ? [Int[]] : [Int[Int(x) for x in v] for v in input_shapes]
    nvar = count(==(-1), input.shape)
    input_shapes === nothing ?
        (nvar == 0 || error("ReactantServerExport: `input` has variable axes but no `input_shapes` were given")) :
        (nvar > 0 || error("ReactantServerExport: `input_shapes` given but `input` has no variable (-1) axes"))
    for v in variants
        haskey(stage1, v) || error("ReactantServerExport: no stage1 program for input shape variant $v")
    end

    w1 = Pair{String, Any}["stage1." * String(first(p)) => last(p) for p in stage1_weights]
    w2 = Pair{String, Any}["stage2." * String(first(p)) => last(p) for p in stage2_weights]
    weights = vcat(w1, w2)
    warrays = Any[last(p) for p in weights]
    n1 = length(w1)
    topk = cfg.detections_per_img
    C = Int(output_columns)

    ctxs = Any[]                                   # keep contexts alive through serialization
    modules = Dict{Any, Any}()
    for v in variants
        k = 0
        shp = [d == -1 ? v[k += 1] : d for d in input.shape]
        t1 = stage1[v]
        f = function (img, ws...)
            s1 = Reactant.Ops.hlo_call(t1, permutedims(img, ndims(img):-1:1), ws[1:n1]...)
            st2(roi) = Reactant.Ops.hlo_call(stage2, roi, ws[(n1 + 1):end]...)
            x1, y1, x2, y2, s, c, n, _ = Detection.detect(s1, st2, cfg, size(img, 1), size(img, 2); T = glue_precision)
            rows = vcat((reshape(r, 1, :) for r in (x1, y1, x2, y2, s, c)[1:C])...)
            dets = output_layout === :cn ? rows : reshape(rows, C, topk, 1)
            # a returned reshape wrapper would be emitted with its parent's shape: materialize
            return Detection._mat(dets), n .+ Reactant.Ops.constant(zeros(Int64, 1))
        end
        ctx = Reactant.ReactantContext()
        push!(ctxs, ctx)
        args = (Reactant.to_rarray(zeros(input.dtype, shp...)), map(Reactant.to_rarray, warrays)...)
        mod, fn_res = Compiler.compile_mlir(ctx, f, args; drop_unsupported_attributes = true)
        _check_entry_arity(fn_res, 1, length(warrays), isempty(v) ? "detector" : "detector variant $v")
        modules[v] = Dict(0 => mod)
    end

    dets_shape, letters, client_shape = output_layout === :cn ?
        ([C, topk], ['c', 'd'], [C, -1]) : ([C, topk, 1], ['d', 'u', 'c'], [C, -1, 1])
    prov = merge(
        _reactant_base_provenance(), Dict{String, Any}(
            "two_stage_detector" => "stage1 -> RPN proposals + NMS -> ROIAlign -> stage2 -> per-class NMS",
            "detections_per_img" => topk, "score_thresh" => cfg.score_thresh, "nms_thresh" => cfg.nms_thresh,
        ), Dict{String, Any}(string(k) => v for (k, v) in provenance)
    )
    GC.@preserve ctxs begin
        write_bundle(
            dir; name,
            executable_inputs = [input],
            executable_outputs = [
                IOSpec("DETECTIONS", Float32, dets_shape; letters),
                IOSpec("NUM_DETECTIONS", Int64, [1]; letters = ['a']),
            ],
            modules = input_shapes === nothing ? modules[Int[]] : modules,
            weights, client_outputs = [IOSpec("OUTPUT__0", Float32, client_shape; letters)],
            input_shapes, provenance = prov
        )
    end
    write(joinpath(dir, "model.jl"), _TRIM_MODEL_JL)
    assert_bundle_arity(dir)
    return dir
end
