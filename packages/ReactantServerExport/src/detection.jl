"""
    ReactantServerExport.Detection

Reactant-traceable second half of a two-stage (FPN + RPN + RoIHeads) detector: RPN proposal
selection, level-aware NMS, multi-level ROIAlign, box decode, and per-class final NMS, all with
static shapes, so a whole detector (stage1 -> glue -> stage2 -> glue) compiles to one StableHLO
program. [`export_two_stage_detector`](@ref ReactantServerExport.export_two_stage_detector) is the
high-level entry point; the functions here are its building blocks.

Every glue step has a fixed-shape form because the shapes of a detector's intermediates are
static even though their values are data-dependent: the stage2 batch is a fixed `K`, variable
selections are masks over fixed-size buffers rather than `findall`, sorts are stable with an index
operand, greedy NMS is the fixed point of a batched mat-vec recurrence, and adaptive ROIAlign loops
over the sample index up to the largest per-box grid in the batch. Every variable-length result is
returned as a fixed-size buffer plus a count.

Conventions: stage outputs are in torch (row-major) indexing, as `Ops.hlo_call` returns them for a
program exported from torch: feature, objectness, and delta maps are `(1, C, H, W)`, the ROI tensor
handed to stage2 is `(K, C, pooled, pooled)`, and stage2 returns `cls_logits (K, NC)` and
`bbox_deltas (K, 4 * nrc)`. The two framework conventions that differ (detectron2 vs torchvision)
are [`DetectorConfig`](@ref) fields.
"""
module Detection

using Reactant
using Reactant: @trace, Ops

export DetectorConfig, detect

"""
    DetectorConfig(; strides, scales, cell_anchors, ...)

Everything the glue needs besides the two stages.

- `strides`: per RPN level (e.g. `[4, 8, 16, 32, 64]` for P2 to P6)
- `scales`: per ROI-pooled level (e.g. `[1/4, 1/8, 1/16, 1/32]` for P2 to P5)
- `cell_anchors`: per RPN level, an `[A, 4]` xyxy matrix centred on the origin

The defaults are the detectron2 conventions (`aligned=true`, background in the last class column,
adaptive sampling). A torchvision Faster R-CNN uses `aligned=false, bg_first=true, min_size=1e-2,
sampling_ratio=2`.
"""
Base.@kwdef struct DetectorConfig
    strides::Vector{Int}
    scales::Vector{Float64}
    cell_anchors::Vector{Matrix{Float64}}
    rpn_weights::NTuple{4, Float64} = (1.0, 1.0, 1.0, 1.0)
    roi_weights::NTuple{4, Float64} = (10.0, 10.0, 5.0, 5.0)
    pre_nms_topk::Int = 1000                   # per RPN level
    post_nms_topk::Int = 1000                  # = the stage2 batch size K
    rpn_nms_thresh::Float64 = 0.7
    score_thresh::Float64 = 0.05
    nms_thresh::Float64 = 0.5
    detections_per_img::Int = 100
    pooled::Int = 7
    sampling_ratio::Int = 0                    # 0 = adaptive, ceil(roi / pooled) per axis
    aligned::Bool = true
    bg_first::Bool = false
    min_size::Float64 = 0.0                    # final boxes narrower than this are dropped
    canonical_level::Int = 4
    canonical_size::Float64 = 224.0
    min_level::Int = 2
    max_level::Int = 5
end

const SCALE_CLAMP = log(1000.0 / 16.0)

_iota(n) = Ops.constant(collect(1:n))
_mat(x) = Reactant.ReactantCore.materialize_traced_array(x)
_int(x) = _mat(vec(Ops.convert(Reactant.TracedRArray{Int64, ndims(x)}, _mat(x))))   # float -> Int64 index vector

# Stable descending argsort along dim 1 (each column independently for a matrix). Greedy NMS and
# top-k selection must break score ties by input order, as a stable sortperm(rev=true) does;
# `Base.sortperm` on a traced array is not stable.
function argsort_desc(v::AbstractVector)
    _, p = Ops.sort(_mat(v), _iota(length(v)); comparator = (a, b, _, _) -> a > b, is_stable = true)
    return p
end
function argsort_desc(v::AbstractMatrix)
    M, G = size(v)
    idx = Ops.constant(repeat(collect(1:M), 1, G))
    _, p = Ops.sort(_mat(v), idx; comparator = (a, b, _, _) -> a > b, dimension = 1, is_stable = true)
    return p
end
argsort_asc_perm(p::AbstractMatrix) = argsort_desc(-p)          # inverse of a column permutation

# Gather column-wise by per-column row indices `p` (M, G).
function colgather(x::AbstractMatrix, p::AbstractMatrix)
    M, G = size(x)
    lin = p .+ Ops.constant(repeat(reshape(collect(0:(G - 1)) .* M, 1, G), M, 1))
    return reshape(vec(x)[vec(lin)], M, G)
end

"""
    decode(dx, dy, dw, dh, x1, y1, x2, y2, weights, T) -> (x1, y1, x2, y2)

torchvision `BoxCoder.decode`, elementwise over equal-shape arrays.
"""
function decode(dx, dy, dw, dh, ax1, ay1, ax2, ay2, wts, ::Type{T}) where {T}
    wx, wy, ww, wh = T.(wts)
    w = ax2 .- ax1; h = ay2 .- ay1
    cx = ax1 .+ T(0.5) .* w; cy = ay1 .+ T(0.5) .* h
    dw = min.(dw ./ ww, T(SCALE_CLAMP)); dh = min.(dh ./ wh, T(SCALE_CLAMP))
    pcx = (dx ./ wx) .* w .+ cx; pcy = (dy ./ wy) .* h .+ cy
    pw = exp.(dw) .* w; ph = exp.(dh) .* h
    return (pcx .- T(0.5) .* pw, pcy .- T(0.5) .* ph, pcx .+ T(0.5) .* pw, pcy .+ T(0.5) .* ph)
end

"""
    batched_nms(x1, y1, x2, y2, s, valid, thresh, T) -> keep

Greedy NMS run independently in each column of `(M, G)` inputs. A column is one NMS group (an FPN
level or a class); groups never suppress each other, which is what torchvision's coordinate offset
trick achieves. `keep` is `(M, G)` Bool in the ORIGINAL row order.

Greedy NMS in descending-score order is the unique solution of `keep[i] = valid[i] && no kept
earlier j overlaps i above thresh`. That recurrence is a DAG in sort order, so iterating it from
`keep = valid` reaches the fixed point, which is exactly the greedy result. Each iteration is one
batched (M x M) mat-vec, and the loop typically settles in a handful of iterations.
"""
function batched_nms(x1, y1, x2, y2, s, valid, thresh, ::Type{T}) where {T}
    M, G = size(s)
    o = argsort_desc(ifelse.(valid, s, T(-Inf)))
    x1, y1, x2, y2 = colgather(x1, o), colgather(y1, o), colgather(x2, o), colgather(y2, o)
    vs = colgather(valid, o)
    c(v) = reshape(v, M, 1, G); r(v) = reshape(v, 1, M, G)
    iw = max.(zero(T), min.(c(x2), r(x2)) .- max.(c(x1), r(x1)))
    ih = max.(zero(T), min.(c(y2), r(y2)) .- max.(c(y1), r(y1)))
    inter = iw .* ih
    area = max.(zero(T), x2 .- x1) .* max.(zero(T), y2 .- y1)
    u = c(area) .+ r(area) .- inter
    iou = ifelse.(u .<= zero(T), zero(T), inter ./ u)
    pos = _iota(M)
    earlier = reshape(pos, 1, M, 1) .< reshape(pos, M, 1, 1)                 # j before i
    supf = ifelse.((iou .> T(thresh)) .& earlier, 1.0f0, 0.0f0)               # (i, j, g)
    vf = ifelse.(vs, 1.0f0, 0.0f0)
    keep = copy(vf)                    # loop-carried values must be distinct arrays
    changed = sum(vf) >= 0.0f0
    @trace while changed
        hit = Ops.dot_general(
            supf, _mat(keep); contracting_dimensions = ([2], [1]),
            batching_dimensions = ([3], [2])
        )                                                                      # (G, M)
        newk = ifelse.(permutedims(hit, (2, 1)) .> 0.0f0, 0.0f0, vf)
        changed = sum(abs.(newk .- keep)) > 0.0f0
        keep = newk
    end
    return colgather(keep, argsort_asc_perm(o)) .> 0.0f0                      # back to input order
end

# The first `n` kept entries of `s` (flattened in its base order) by descending score, and how many
# of them are real. Matches a stable global sort of the kept set, then truncation.
function top_kept(s, keep, n, ::Type{T}) where {T}
    sv = vec(ifelse.(keep, s, T(-Inf)))
    sel = argsort_desc(sv)[1:n]
    nk = min(sum(ifelse.(vec(keep), 1, 0)), n)
    return sel, nk
end

"""
    traced_anchors(H, W, stride, cell, T) -> (x1, y1, x2, y2)

The anchor grid for an `H x W` map in (h, w, a) order, built in-program from the `[A, 4]` cell
anchors and two iotas, so the bundle carries no per-shape anchor constants (baking them as Float64
constants added about 200 MB to a 15-variant bundle). Values are `cell + shift`, exactly as
detectron2's and torchvision's `grid_anchors` compute them.
"""
function traced_anchors(H::Int, W::Int, stride::Real, cell::AbstractMatrix, ::Type{T}) where {T}
    A = size(cell, 1)
    sx = Ops.constant(reshape(T.(collect(0:(W - 1)) .* stride), 1, W, 1))
    sy = Ops.constant(reshape(T.(collect(0:(H - 1)) .* stride), 1, 1, H))
    c(j) = Ops.constant(reshape(T.(cell[:, j]), A, 1, 1))
    zy = zero(T) .* sy; zx = zero(T) .* sx          # broadcast partners (exact zeros), no W*H constant
    return (vec(c(1) .+ sx .+ zy), vec(c(2) .+ sy .+ zx), vec(c(3) .+ sx .+ zy), vec(c(4) .+ sy .+ zx))
end

"""
    rpn_proposals(objs, dels, cfg, img_w, img_h, T) -> (x1, y1, x2, y2, valid)

`find_top_rpn_proposals`: per-level top-`pre_nms_topk` by objectness, decode against the grid
anchors, clip, drop empty boxes, level-aware NMS, keep the top `post_nms_topk`. Returns fixed
`post_nms_topk`-length vectors plus a validity mask.
"""
function rpn_proposals(objs, dels, cfg::DetectorConfig, img_w, img_h, ::Type{T}) where {T}
    L = length(objs); P = cfg.pre_nms_topk
    cols = [Any[] for _ in 1:6]
    for l in 1:L
        O = objs[l]; D = dels[l]
        _, A, H, W = size(O)
        sc = T.(vec(permutedims(reshape(O, A, H, W), (1, 3, 2))))                 # (h, w, a) order
        dd = T.(reshape(permutedims(reshape(D, 4, A, H, W), (1, 2, 4, 3)), 4, A * W * H))
        ac = traced_anchors(H, W, cfg.strides[l], cfg.cell_anchors[l], T)
        bx = decode(dd[1, :], dd[2, :], dd[3, :], dd[4, :], ac..., cfg.rpn_weights, T)
        n = min(length(sc), P)
        idx = argsort_desc(sc)[1:n]
        pad(v, fillv) = n == P ? v : vcat(v, Ops.constant(fill(fillv, P - n)))
        push!(cols[1], pad(bx[1][idx], zero(T))); push!(cols[2], pad(bx[2][idx], zero(T)))
        push!(cols[3], pad(bx[3][idx], zero(T))); push!(cols[4], pad(bx[4][idx], zero(T)))
        push!(cols[5], pad(sc[idx], T(-Inf)))
        push!(cols[6], Ops.constant(vcat(fill(true, n), fill(false, P - n))))
    end
    m(i) = hcat((reshape(v, P, 1) for v in cols[i])...)                         # (P, L)
    x1 = clamp.(m(1), zero(T), T(img_w)); x2 = clamp.(m(3), zero(T), T(img_w))
    y1 = clamp.(m(2), zero(T), T(img_h)); y2 = clamp.(m(4), zero(T), T(img_h))
    s = m(5)
    valid = m(6) .& (x2 .> x1) .& (y2 .> y1)
    keep = batched_nms(x1, y1, x2, y2, s, valid, cfg.rpn_nms_thresh, T)
    sel, kp = top_kept(s, keep, cfg.post_nms_topk, T)
    pv = _iota(cfg.post_nms_topk) .<= kp
    return vec(x1)[sel], vec(y1)[sel], vec(x2)[sel], vec(y2)[sel], pv
end

"""
    roi_align_fpn(feats, x1, y1, x2, y2, valid, cfg, T) -> (K, C, pooled, pooled) Float32

Multi-level ROIAlign: each box is pooled from the FPN level the ROIPooler level rule picks, with
torchvision's sampling rules (`aligned`, fixed or adaptive `sampling_ratio`, out-of-range samples
contribute 0, edge clamping). With adaptive sampling the per-box grid is data-dependent
(`ceil(roi / pooled)` per axis), so the loop runs over the sample index up to the largest grid in
this batch; each iteration bilinearly gathers one sample for every (box, bin) at once from the
concatenated levels. The accumulator is Float32. Invalid (padding) boxes produce zeros.
"""
function roi_align_fpn(feats, x1, y1, x2, y2, valid, cfg::DetectorConfig, ::Type{T}) where {T}
    K = length(x1); Pd = cfg.pooled; NL = length(feats)
    C = size(feats[1], 2)
    Hs = [size(F, 3) for F in feats]; Ws = [size(F, 4) for F in feats]
    offs = cumsum(vcat(0, Hs .* Ws))[1:NL]
    Fcat = hcat((_mat(reshape(F, C, size(F, 3) * size(F, 4))) for F in feats)...)   # (C, sum HW)

    area = max.(zero(T), (x2 .- x1) .* (y2 .- y1))
    lvl = clamp.(
        floor.(log2.(sqrt.(area) ./ T(cfg.canonical_size) .+ T(1.0e-8)) .+ T(cfg.canonical_level)),
        T(cfg.min_level), T(cfg.max_level)
    ) .- T(cfg.min_level)
    pick(vals) = sum(ifelse.(lvl .== T(l - 1), T(vals[l]), zero(T)) for l in 1:NL)
    Hk = pick(Hs); Wk = pick(Ws); Sk = pick(cfg.scales); Ok = pick(offs)

    off = cfg.aligned ? T(0.5) : zero(T)
    sw = x1 .* Sk .- off; sh = y1 .* Sk .- off
    rw = x2 .* Sk .- off .- sw; rh = y2 .* Sk .- off .- sh
    if !cfg.aligned
        rw = max.(rw, one(T)); rh = max.(rh, one(T))
    end
    bw = rw ./ T(Pd); bh = rh ./ T(Pd)
    if cfg.sampling_ratio > 0
        gh = zero(T) .* bh .+ T(cfg.sampling_ratio); gw = zero(T) .* bw .+ T(cfg.sampling_ratio)
    else
        gh = max.(one(T), ceil.(bh)); gw = max.(one(T), ceil.(bw))
    end
    cnt = gh .* gw
    nmax = maximum(ifelse.(valid, cnt, one(T)))

    # (pw, ph, k) layouts
    k3(v) = _mat(reshape(v, 1, 1, K))
    PW = Ops.constant(reshape(T.(0:(Pd - 1)), Pd, 1, 1))
    PH = Ops.constant(reshape(T.(0:(Pd - 1)), 1, Pd, 1))
    acc = Ops.constant(zeros(Float32, C, Pd * Pd * K))
    # constants hoisted out of the traced loop body (the loop macro must not see the type `T`)
    z0 = zero(T); o1 = one(T); h5 = T(0.5); m1 = T(-1)
    H3 = k3(Hk); W3 = k3(Wk); H1 = H3 .- o1; W1 = W3 .- o1; base = k3(Ok) .+ o1
    si = z0 * nmax
    @trace while si < nmax
        iy = floor.(si ./ gw); ix = si .- iy .* gw
        y = k3(sh) .+ PH .* k3(bh) .+ k3((iy .+ h5) .* bh ./ gh)
        x = k3(sw) .+ PW .* k3(bw) .+ k3((ix .+ h5) .* bw ./ gw)
        act = k3(valid .& (si .< cnt)) .& .!((y .< m1) .| (y .> H3) .| (x .< m1) .| (x .> W3))
        yy = max.(y, z0); xx = max.(x, z0)
        yl = floor.(yy); xl = floor.(xx)
        ytop = yl .>= H1; xtop = xl .>= W1
        yl = ifelse.(ytop, H1, yl); yh = ifelse.(ytop, yl, yl .+ o1); yy = ifelse.(ytop, yl, yy)
        xl = ifelse.(xtop, W1, xl); xh = ifelse.(xtop, xl, xl .+ o1); xx = ifelse.(xtop, xl, xx)
        ly = yy .- yl; lx = xx .- xl; hy = o1 .- ly; hx = o1 .- lx
        i00 = _int(ifelse.(act, base .+ yl .+ H3 .* xl, o1)); w00 = reshape(Float32.(vec(ifelse.(act, hy .* hx, z0))), 1, :)
        i01 = _int(ifelse.(act, base .+ yl .+ H3 .* xh, o1)); w01 = reshape(Float32.(vec(ifelse.(act, hy .* lx, z0))), 1, :)
        i10 = _int(ifelse.(act, base .+ yh .+ H3 .* xl, o1)); w10 = reshape(Float32.(vec(ifelse.(act, ly .* hx, z0))), 1, :)
        i11 = _int(ifelse.(act, base .+ yh .+ H3 .* xh, o1)); w11 = reshape(Float32.(vec(ifelse.(act, ly .* lx, z0))), 1, :)
        acc = acc .+ (w00 .* Fcat[:, i00] .+ w01 .* Fcat[:, i01] .+ w10 .* Fcat[:, i10] .+ w11 .* Fcat[:, i11])
        si = si + o1
    end
    cntf = reshape(Float32.(vec(PW .* zero(T) .+ PH .* zero(T) .+ k3(cnt))), 1, :)
    out = reshape(acc ./ cntf, C, Pd, Pd, K)                                       # (c, pw, ph, k)
    return permutedims(out, (4, 1, 3, 2))                                          # (k, c, ph, pw)
end

"""
    fast_rcnn_inference(cls, bd, x1, y1, x2, y2, valid, cfg, img_w, img_h, T)
        -> (x1, y1, x2, y2, score, class, n)  each of length detections_per_img

Softmax, class-specific decode, clip, score threshold, optional minimum box size, per-class NMS,
top detections. Class ids are the emitted ids (foreground column index minus 1, in 1-based columns).
"""
function fast_rcnn_inference(cls, bd, px1, py1, px2, py2, pvalid, cfg::DetectorConfig, img_w, img_h, ::Type{T}) where {T}
    K, NC = size(cls); nrc = size(bd, 2) ÷ 4
    fg = cfg.bg_first ? (2:NC) : (1:(NC - 1)); nfg = length(fg)
    c = T.(cls)
    p = exp.(c .- maximum(c; dims = 2)); p = p ./ sum(p; dims = 2)
    bdT = T.(bd)
    g(j) = nrc == 1 ? 0 : (fg[j] - 1)                  # delta group of foreground column j
    dx = hcat((reshape(bdT[:, 4g(j) + 1], K, 1) for j in 1:nfg)...)
    dy = hcat((reshape(bdT[:, 4g(j) + 2], K, 1) for j in 1:nfg)...)
    dw = hcat((reshape(bdT[:, 4g(j) + 3], K, 1) for j in 1:nfg)...)
    dh = hcat((reshape(bdT[:, 4g(j) + 4], K, 1) for j in 1:nfg)...)
    b = decode(dx, dy, dw, dh, px1, py1, px2, py2, cfg.roi_weights, T)             # (K, nfg) each
    bx1 = clamp.(b[1], zero(T), T(img_w)); bx2 = clamp.(b[3], zero(T), T(img_w))
    by1 = clamp.(b[2], zero(T), T(img_h)); by2 = clamp.(b[4], zero(T), T(img_h))
    s = p[:, fg]
    valid = reshape(pvalid, K, 1) .& (s .> T(cfg.score_thresh))
    if cfg.min_size > 0
        valid = valid .& ((bx2 .- bx1) .>= T(cfg.min_size)) .& ((by2 .- by1) .>= T(cfg.min_size))
    end
    keep = batched_nms(bx1, by1, bx2, by2, s, valid, cfg.nms_thresh, T)
    # candidates are ordered (proposal outer, class inner) before the stable sort, as a host
    # implementation that loops proposals then classes would push them
    t(v) = permutedims(v, (2, 1))
    sel, nd = top_kept(t(s), t(keep), cfg.detections_per_img, T)
    cid = Ops.constant(T.(repeat(reshape(collect(fg) .- 1, 1, nfg), K, 1)))
    f(v) = vec(t(v))[sel]
    return f(bx1), f(by1), f(bx2), f(by2), f(s), f(cid), nd
end

"""
    detect(s1, stage2, cfg, img_w, img_h; T=Float64) -> (x1, y1, x2, y2, score, class, n, extras)

The full glue given stage1's outputs `s1 = (feats..., objectness..., deltas...)` (4 + 5 + 5 maps
for a P2 to P6 FPN) and `stage2(roi) -> (cls_logits, bbox_deltas)`. The outputs are Float32 vectors
of length `detections_per_img`; entries past `n` are zero. `extras` carries the proposals, their
validity mask, and the ROI tensor, for parity checks. `T` is the precision of the box arithmetic.
"""
function detect(s1, stage2, cfg::DetectorConfig, img_w, img_h; T = Float64)
    nf = length(cfg.scales); nl = length(cfg.strides)
    length(s1) == nf + 2nl ||
        error("Detection.detect: stage1 returned $(length(s1)) tensors, expected $nf feature maps + $nl objectness + $nl delta maps")
    feats = s1[1:nf]; objs = s1[(nf + 1):(nf + nl)]; dels = s1[(nf + nl + 1):(nf + 2nl)]
    px1, py1, px2, py2, pv = rpn_proposals(objs, dels, cfg, img_w, img_h, T)
    roi = roi_align_fpn(feats, px1, py1, px2, py2, pv, cfg, T)
    cls, bd = stage2(roi)
    x1, y1, x2, y2, s, c, n = fast_rcnn_inference(cls, bd, px1, py1, px2, py2, pv, cfg, img_w, img_h, T)
    ok = _iota(cfg.detections_per_img) .<= n
    z(v) = Float32.(ifelse.(ok, v, zero(T)))
    return z(x1), z(y1), z(x2), z(y2), z(s), z(c), n, (; proposals = (px1, py1, px2, py2), valid = pv, roi)
end

end # module Detection
