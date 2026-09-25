# Two-stage detector tests: the traced glue (ReactantServerExport.Detection) against the host oracle
# (detection_glue.jl) on synthetic stage outputs, for both convention sets, and the single-program
# export (export_two_stage_detector) through the server's own load/compile/run/postprocess path.
# Included from runtests.jl, which provides `run_bundle`, Test, Random, Reactant, and the packages.

include("detection_glue.jl")
const _G = DetectionGlue
const _D = ReactantServerExport.Detection
const _Ops = Reactant.Ops

# ── oracle self-checks (reference values captured from torchvision.ops.roi_align, torch 2.12) ──

@testset "DetectionGlue oracle" begin
    @testset "roi_align aligned flag vs torchvision" begin
        # feat: torch [1,8,8] with f[r,c] = r*8 + c (0-based). Julia feat[c=1, r+1, col+1].
        H = W = 8
        feat = Array{Float64}(undef, 1, H, W)
        for r in 0:(H - 1), c in 0:(W - 1)
            feat[1, r + 1, c + 1] = r * 8 + c
        end
        boxes = reshape(Float64[1.3, 2.1, 6.7, 5.9], 1, 4)
        scale, pooled, ratio = 0.5, 2, 2

        exp_aligned = [9.025 10.375; 16.625 17.975]    # torch aligned=True  (half-pixel offset)
        exp_unaligned = [13.525 14.875; 21.125 22.475]   # torch aligned=False (torchvision detection)

        out = Array{Float64}(undef, 1, 1, pooled, pooled)
        _G.roi_align!(out, feat, boxes, scale; pooled = pooled, ratio = ratio, aligned = true)
        got_aligned = [out[1, 1, ph, pw] for ph in 1:pooled, pw in 1:pooled]
        _G.roi_align!(out, feat, boxes, scale; pooled = pooled, ratio = ratio, aligned = false)
        got_unaligned = [out[1, 1, ph, pw] for ph in 1:pooled, pw in 1:pooled]

        # wire-layout variant (pw,ph,C,K) must agree with roi_align!
        wire = Array{Float32}(undef, pooled, pooled, 1, 1)
        _G.roi_align_wire!(wire, feat, boxes, scale; pooled = pooled, ratio = ratio, aligned = false)
        got_wire = [Float64(wire[pw, ph, 1, 1]) for ph in 1:pooled, pw in 1:pooled]

        @test maximum(abs.(got_aligned .- exp_aligned)) < 1.0e-6
        @test maximum(abs.(got_unaligned .- exp_unaligned)) < 1.0e-6
        @test maximum(abs.(got_wire .- exp_unaligned)) < 1.0e-4
    end

    @testset "fast_rcnn_inference bg_first column selection" begin
        proposals = Float64[10 10 50 50; 20 20 60 60]
        deltas = zeros(Float64, 2, 3 * 4)        # zero deltas -> decoded == proposals
        # bg_first=true (torchvision): bg is col1; foreground cols 2,3 -> class ids 1,2.
        cls_first = Float64[0 5 0; 0 0 5]
        _, _, c1 = _G.fast_rcnn_inference(
            cls_first, deltas, proposals, 100, 100;
            score_thresh = 0.05, nms_thresh = 0.5, topk = 100, bg_first = true
        )
        @test sort(c1) == [1, 2]
        # bg_first=false: bg is last col; foreground cols 1,2 -> class ids 0,1.
        cls_last = Float64[5 0 0; 0 5 0]
        _, _, c0 = _G.fast_rcnn_inference(
            cls_last, deltas, proposals, 100, 100;
            score_thresh = 0.05, nms_thresh = 0.5, topk = 100, bg_first = false
        )
        @test sort(c0) == [0, 1]
    end

    @testset "fast_rcnn_inference min_size drops sub-pixel boxes" begin
        # proposal 1 is a sub-pixel box (0.005 px); zero deltas keep decoded == proposal.
        proposals = Float64[10 10 10.005 10.005; 20 20 60 60]
        deltas = zeros(Float64, 2, 3 * 4)
        cls = Float64[0 5 0; 0 5 0]   # both rows -> foreground class 1 (bg_first)
        _, _, c_keep = _G.fast_rcnn_inference(
            cls, deltas, proposals, 100, 100;
            score_thresh = 0.05, nms_thresh = 0.5, topk = 100, bg_first = true, min_size = 0.0
        )
        _, _, c_drop = _G.fast_rcnn_inference(
            cls, deltas, proposals, 100, 100;
            score_thresh = 0.05, nms_thresh = 0.5, topk = 100, bg_first = true, min_size = 1.0e-2
        )
        @test length(c_keep) == 2     # both kept without a size filter
        @test length(c_drop) == 1     # the sub-pixel box is dropped (torchvision min_size=1e-2)
    end
end

# ── synthetic detector ──────────────────────────────────────────────────────────────────────────

const _STRIDES = [4, 8, 16, 32, 64]
const _NF = 4                       # ROI-pooled levels P2..P5
const _CH = 8                       # feature channels
const _A = 3                        # anchors per location
const _NC = 4                       # class columns including background

# Cell anchors for sizes 8..128 at aspect ratios 0.5, 1, 2 (xyxy, centred on the origin).
_cell(s) = vcat(([-s / sqrt(r) / 2 -s * sqrt(r) / 2 s / sqrt(r) / 2 s * sqrt(r) / 2] for r in (0.5, 1.0, 2.0))...)

# Small enough to compile quickly on CPU, and chosen so every path runs: pre_nms_topk exceeds the P6
# grid (padding), a canonical size of 16 spreads boxes over all four ROI levels, and far more
# candidates survive the score threshold than detections_per_img.
function _cfg(; kw...)
    return DetectorConfig(;
        strides = _STRIDES, scales = [1 / 4, 1 / 8, 1 / 16, 1 / 32],
        cell_anchors = [_cell(s) for s in (8, 16, 32, 64, 128)],
        pre_nms_topk = 300, post_nms_topk = 64, detections_per_img = 20, pooled = 4,
        canonical_size = 16.0, kw...
    )
end
const _DETECTRON = _cfg()
const _TORCHVISION = _cfg(aligned = false, bg_first = true, min_size = 1.0e-2, sampling_ratio = 2)

# Stage1 from a torch-layout image (1, H, W): each level's maps are sin of a per-channel affine of the
# image average-pooled to that level's grid, so the outputs depend on the image, the weights are
# small per-channel vectors shared by every input shape, and the same code runs on host and traced
# arrays. `big` is a dead-weight tensor that makes the weights file dwarf the program, for the
# constant-folding check below.
function _pool(img, s)
    _, H, W = size(img)
    return dropdims(sum(reshape(img, s, H ÷ s, s, W ÷ s); dims = (1, 3)); dims = (1, 3)) ./ Float32(s * s * 255)
end
function _stage1(img, ws...)
    wf, bf, wo, bo, wd, bd, big = ws
    mp(p) = reshape(p, 1, 1, size(p)...)
    ch(v, n) = reshape(v, 1, n, 1, 1)
    ps = [mp(_pool(img, s)) for s in _STRIDES]
    feats = [sin.(ch(wf[:, l], _CH) .* ps[l] .+ ch(bf[:, l], _CH)) for l in 1:_NF]
    feats[1] = feats[1] .+ 0.0f0 * sum(big)
    objs = [3.0f0 .* sin.(ch(wo[:, l], _A) .* ps[l] .+ ch(bo[:, l], _A)) for l in 1:5]
    dels = [0.3f0 .* sin.(ch(wd[:, l], 4 * _A) .* ps[l] .+ ch(bd[:, l], 4 * _A)) for l in 1:5]
    return (feats..., objs..., dels...)
end
function _stage1_weights(rng)
    return Pair{String, Any}[
        "wf" => 40 .* randn(rng, Float32, _CH, _NF), "bf" => 6 .* rand(rng, Float32, _CH, _NF),
        "wo" => 60 .* randn(rng, Float32, _A, 5), "bo" => 6 .* rand(rng, Float32, _A, 5),
        "wd" => 60 .* randn(rng, Float32, 4 * _A, 5), "bd" => 6 .* rand(rng, Float32, 4 * _A, 5),
        "big" => randn(rng, Float32, 1024, 1024),
    ]
end

# Stage2: a linear box head on the flattened (K, C, P, P) ROI features, on host or traced arrays.
_flat(roi) = reshape(roi, size(roi, 1), :)
_mm(x::Array, w::Array) = x * w
_mm(x, w) = _Ops.dot_general(Reactant.ReactantCore.materialize_traced_array(x), w; contracting_dimensions = ([2], [1]))
_stage2(roi, wc, wd) = (x = _flat(roi); (_mm(x, wc), _mm(x, wd)))
function _stage2_weights(rng, cfg)
    nrc = cfg.bg_first ? _NC : _NC - 1           # torchvision carries a background delta group
    nin = _CH * cfg.pooled^2
    return Pair{String, Any}["wc" => 0.3f0 .* randn(rng, Float32, nin, _NC), "wd" => 0.2f0 .* randn(rng, Float32, nin, 4nrc)]
end

# The meta model.jl pipeline on host arrays: DetectionGlue in Float64 with loops and findall.
function _oracle(s1, stage2, cfg, iw, ih)
    wire(a) = permutedims(a, ndims(a):-1:1)
    nl = length(cfg.strides)
    feats = [_G.feature_chw(wire(s1[i])) for i in 1:_NF]
    bl = Matrix{Float64}[]; sl = Vector{Float64}[]
    for i in 1:nl
        O = wire(s1[_NF + i]); Dd = wire(s1[_NF + nl + i])
        anc = _G.generate_anchors(size(O, 2), size(O, 1), cfg.strides[i], cfg.cell_anchors[i])
        push!(bl, _G.decode_boxes(_G.deltas_matrix(Dd), anc, cfg.rpn_weights)); push!(sl, _G.objectness_flat(O))
    end
    pb = _G.select_rpn_proposals(bl, sl, ih, iw; pre = cfg.pre_nms_topk, post = cfg.post_nms_topk, nms_thresh = cfg.rpn_nms_thresh)
    Kp = size(pb, 1); P = cfg.pooled
    roi = zeros(Float32, P, P, _CH, cfg.post_nms_topk)
    lv = [
        _G.assign_level(
            view(pb, k, :); canon_level = cfg.canonical_level, canon_size = cfg.canonical_size,
            min_level = cfg.min_level, max_level = cfg.max_level
        ) for k in 1:Kp
    ]
    for l in 0:(_NF - 1)
        sel = findall(==(l), lv); isempty(sel) && continue
        _G.roi_align_wire!(
            view(roi, :, :, :, sel), feats[l + 1], pb[sel, :], cfg.scales[l + 1];
            pooled = P, ratio = cfg.sampling_ratio, aligned = cfg.aligned
        )
    end
    roi_t = permutedims(roi, (4, 3, 2, 1))
    cls, dl = stage2(roi_t)
    bx, sc, cl = _G.fast_rcnn_inference(
        Float64.(cls[1:Kp, :]), Float64.(dl[1:Kp, :]), pb, ih, iw;
        score_thresh = cfg.score_thresh, nms_thresh = cfg.nms_thresh, topk = cfg.detections_per_img,
        weights = cfg.roi_weights, bg_first = cfg.bg_first, min_size = cfg.min_size
    )
    return (; boxes = bx, scores = sc, classes = cl, proposals = pb, levels = lv, roi = roi_t)
end

# Every oracle detection has a traced one with the same class, box within `tol` px, and score within
# 1e-4 (set match: two detections whose scores tie to ~1e-6 may legitimately swap), and vice versa.
function _same_detections(boxes, scores, classes, ref; tol = 1.0e-3)
    length(scores) == length(ref.scores) || return false
    m(i, j) = classes[i] == ref.classes[j] && maximum(abs.(boxes[i, :] .- ref.boxes[j, :])) < tol &&
        abs(scores[i] - ref.scores[j]) < 1.0e-4
    return all(j -> any(i -> m(i, j), eachindex(scores)), eachindex(ref.scores)) &&
        all(i -> any(j -> m(i, j), eachindex(ref.scores)), eachindex(scores))
end

@testset "traced detection glue vs host oracle" begin
    for (label, cfg) in (("detectron2 conventions", _DETECTRON), ("torchvision conventions", _TORCHVISION))
        @testset "$label" begin
            rng = Xoshiro(20260924)
            W, H = 192, 128                      # non-square: clipping uses the right edge per axis
            img = 255 .* rand(rng, Float32, 1, H, W)
            w1 = [last(p) for p in _stage1_weights(rng)]
            w2 = [last(p) for p in _stage2_weights(rng, cfg)]
            s1 = _stage1(img, w1...)
            ref = _oracle(s1, r -> _stage2(r, w2...), cfg, W, H)
            @test length(ref.scores) == cfg.detections_per_img       # enough candidates to truncate
            @test length(unique(ref.levels)) == _NF                   # every ROI level is pooled from

            glue = (s1, wc, wd) -> begin
                x1, y1, x2, y2, s, c, n, ex = _D.detect(s1, r -> _stage2(r, wc, wd), cfg, W, H)
                return (x1, y1, x2, y2, s, c, n, ex.proposals..., ex.valid, ex.roi)
            end
            rs1 = map(Reactant.to_rarray, s1)
            rw = map(Reactant.to_rarray, w2)
            compiled = Reactant.@compile glue(rs1, rw...)
            out = map(o -> o isa Number ? convert(Int, o) : Array(o), compiled(rs1, rw...))
            x1, y1, x2, y2, s, c = out[1:6]
            n = out[7]; P = hcat(out[8:11]...); pv = out[12]; roi = out[13]

            Kp = size(ref.proposals, 1)
            @test count(pv) == Kp
            @test all(pv[1:Kp])
            @test maximum(abs.(P[1:Kp, :] .- ref.proposals)) < 1.0e-6
            @test maximum(abs.(roi[1:Kp, :, :, :] .- ref.roi[1:Kp, :, :, :])) < 1.0e-4
            @test all(iszero, roi[(Kp + 1):end, :, :, :])
            @test n == length(ref.scores)
            boxes = hcat(x1, y1, x2, y2)[1:n, :]
            @test _same_detections(boxes, s[1:n], Int.(c[1:n]), ref)
            @test all(iszero, s[(n + 1):end])                        # padding past the count is zero
        end
    end
end

# ── single-program export through the server ────────────────────────────────────────────────────

# Write a stage bundle from a function of torch-order arrays, so the export reads it back like a
# real stage bundle. Reactant emits a program's entry signature in reversed (row-major) axis order,
# while a torch.export'ed stage has its signature in torch order, which is what the glue expects;
# wrapping `f` in axis reversals gives the synthetic stage that torch-order signature.
_rev(a) = permutedims(a, ndims(a):-1:1)
function _write_stage(root, name, f, inputs_by_variant, weights; input_shapes = nothing)
    g = (x, ws...) -> map(o -> Reactant.ReactantCore.materialize_traced_array(_rev(o)), f(_rev(x), map(_rev, ws)...))
    wrev = Pair{String, Any}[first(p) => _rev(last(p)) for p in weights]
    ctxs = Any[]; modules = Dict{Any, Any}(); outs = nothing
    for (key, x) in inputs_by_variant
        ctx = Reactant.ReactantContext(); push!(ctxs, ctx)
        mod, _ = Reactant.Compiler.compile_mlir(ctx, g, (Reactant.to_rarray(_rev(x)), map(Reactant.to_rarray, last.(wrev))...))
        modules[key] = Dict(0 => mod)
        outs = f(x, last.(weights)...)
    end
    x0 = _rev(last(first(inputs_by_variant)))
    shape = input_shapes === nothing ? collect(size(x0)) : [-1, -1, collect(size(x0))[3:end]...]
    dir = joinpath(root, name)
    GC.@preserve ctxs begin
        write_bundle(
            dir; name,
            executable_inputs = [IOSpec("x", eltype(x0), shape)],
            executable_outputs = [IOSpec("y$i", Float32, collect(size(_rev(o)))) for (i, o) in enumerate(outs)],
            modules = input_shapes === nothing ? modules[Int[]] : modules, weights = wrev, input_shapes
        )
    end
    return dir
end

# Load one bundle and run it the way the server does, model.jl hooks included.
function _serve_one(root, name)
    backend = ReactantServer.ReactantBackend()
    pool = ReactantServer.resolve_client(backend, ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
    reg = ReactantServer.load_bundles([root]; include = [name])
    entry = ReactantServer.get_model(reg, name)
    entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)
    return (x) -> begin
        raw = ReactantServer.run_model(backend, pool, entry.executable, Base.invokelatest(entry.preprocess, [ReactantServer.NamedTensor("INPUT__0", x)]))
        only(Base.invokelatest(entry.postprocess, raw)).data
    end
end

@testset "export_two_stage_detector -> bundle -> server" begin
    @testset "single shape, u8 input, (6, N) rows with class" begin
        rng = Xoshiro(7)
        cfg = _DETECTRON
        w1 = _stage1_weights(rng); w2 = _stage2_weights(rng, cfg)
        mktempdir() do root
            W = H = 128
            s1dir = _write_stage(root, "det_stage1", (x, ws...) -> _stage1(Float32.(x), ws...), [Int[] => zeros(UInt8, 1, H, W)], w1)
            s2dir = _write_stage(root, "det_stage2", _stage2, [Int[] => zeros(Float32, cfg.post_nms_topk, _CH, cfg.pooled, cfg.pooled)], w2)
            S1 = read_stage_bundle(s1dir); S2 = read_stage_bundle(s2dir)
            @test [first(p) for p in S1.weights] == [first(p) for p in w1]
            @test all(last(a) == last(b) for (a, b) in zip(S1.weights, w1))      # in signature (torch) order
            @test isempty(S1.input_shapes)

            out_root = joinpath(root, "served"); mkpath(out_root)
            dir = export_two_stage_detector(
                joinpath(out_root, "det"); name = "det",
                input = IOSpec("INPUT__0", UInt8, [W, H, 1]; letters = ['w', 'h', 'c']),
                stage1 = S1.texts, stage1_weights = S1.weights,
                stage2 = S2.texts[Int[]], stage2_weights = S2.weights, cfg
            )
            @test isfile(joinpath(dir, "model.jl"))
            # weights stay program arguments: the program is a small fraction of the weights file
            wbytes = filesize(joinpath(dir, "weights.safetensors"))
            @test all(f -> filesize(joinpath(dir, f)) < wbytes ÷ 4, filter(endswith(".mlir"), readdir(dir)))
            @test assert_bundle_arity(dir).servable

            serve = _serve_one(out_root, "det")
            for _ in 1:2
                x = rand(rng, UInt8, W, H, 1)
                got = serve(x)
                ref = _oracle(_stage1(Float32.(permutedims(x, (3, 2, 1))), last.(w1)...), r -> _stage2(r, last.(w2)...), cfg, W, H)
                @test size(got) == (6, length(ref.scores))
                @test length(ref.scores) > 0
                @test _same_detections(permutedims(got[1:4, :]), got[5, :], Int.(got[6, :]), ref)
            end
        end
    end

    @testset "input_shapes variants, f32 input, (5, N, 1) rows" begin
        rng = Xoshiro(11)
        cfg = _TORCHVISION
        w1 = _stage1_weights(rng); w2 = _stage2_weights(rng, cfg)
        shapes = [[192, 128], [128, 192]]
        mktempdir() do root
            s1dir = _write_stage(
                root, "tv_stage1", _stage1, [s => zeros(Float32, 1, s[2], s[1]) for s in shapes], w1;
                input_shapes = shapes
            )
            s2dir = _write_stage(root, "tv_stage2", _stage2, [Int[] => zeros(Float32, cfg.post_nms_topk, _CH, cfg.pooled, cfg.pooled)], w2)
            S1 = read_stage_bundle(s1dir); S2 = read_stage_bundle(s2dir)
            @test S1.input_shapes == shapes

            out_root = joinpath(root, "served"); mkpath(out_root)
            export_two_stage_detector(
                joinpath(out_root, "tv"); name = "tv",
                input = IOSpec("INPUT__0", Float32, [-1, -1, 1]; letters = ['w', 'h', 'c']),
                stage1 = S1.texts, stage1_weights = S1.weights, input_shapes = S1.input_shapes,
                stage2 = S2.texts[Int[]], stage2_weights = S2.weights, cfg,
                output_columns = 5, output_layout = :cn1
            )
            serve = _serve_one(out_root, "tv")
            for (W, H) in shapes
                x = 255 .* rand(rng, Float32, W, H, 1)
                got = serve(x)
                ref = _oracle(_stage1(permutedims(x, (3, 2, 1)), last.(w1)...), r -> _stage2(r, last.(w2)...), cfg, W, H)
                @test size(got) == (5, length(ref.scores), 1)
                g = got[:, :, 1]
                # no class column: match on box and score only
                @test _same_detections(permutedims(g[1:4, :]), g[5, :], zeros(Int, size(g, 2)), (; ref..., classes = zeros(Int, length(ref.scores))))
            end
        end
    end
end
