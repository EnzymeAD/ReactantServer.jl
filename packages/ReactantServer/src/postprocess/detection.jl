# Julia detection glue for two-stage object detectors (FPN backbone + RPN + RoIHeads, e.g. a
# torchvision Faster R-CNN). Ports the data-dependent pipeline that sits between the two StableHLO
# executables: anchor generation, box decode, RPN proposal NMS, ROIAlign, and per-class final NMS.
# Pure Julia (no torch at serve time). Conventions: boxes are xyxy rows [N,4]; feature maps are passed
# channel-first as A[c, h, w] (0-based torch coords map to 1-based Julia indices internally). Pixel/class
# conventions that differ across frameworks are kwargs: `aligned` on roi_align, `bg_first` on
# fast_rcnn_inference (see those functions).

module DetectionGlue

# --- IoU + greedy NMS (torchvision batched_nms semantics: per-class offset, descending score) ---

@inline function _iou(a::AbstractVector, b::AbstractVector)
    ix1 = max(a[1], b[1]); iy1 = max(a[2], b[2])
    ix2 = min(a[3], b[3]); iy2 = min(a[4], b[4])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[3] - a[1]) * max(0.0, a[4] - a[2])
    ab = max(0.0, b[3] - b[1]) * max(0.0, b[4] - b[2])
    u = aa + ab - inter
    return u <= 0 ? 0.0 : inter / u
end

"""
    nms(boxes, scores, thresh) -> kept indices (descending score)

Greedy IoU suppression on xyxy `boxes` [N,4] by `scores` [N]. Returns kept row indices in
descending-score order, matching torchvision.ops.nms.
"""
function nms(boxes::AbstractMatrix, scores::AbstractVector, thresh::Real)
    order = sortperm(scores; rev=true)
    kept = Int[]
    suppressed = falses(length(order))
    @inbounds for ii in eachindex(order)
        suppressed[ii] && continue
        i = order[ii]
        push!(kept, i)
        bi = @view boxes[i, :]
        for jj in (ii + 1):length(order)
            suppressed[jj] && continue
            if _iou(bi, @view boxes[order[jj], :]) > thresh
                suppressed[jj] = true
            end
        end
    end
    return kept
end

"""
    batched_nms(boxes, scores, idxs, thresh) -> kept indices

Class/level-aware NMS: offsets boxes by `idxs` into disjoint coordinate bands so different groups
never suppress each other, then a single NMS. Mirrors torchvision.ops.batched_nms.
"""
function batched_nms(boxes::AbstractMatrix, scores::AbstractVector, idxs::AbstractVector, thresh::Real)
    isempty(scores) && return Int[]
    maxc = maximum(@view boxes[:, 3:4]) - minimum(@view boxes[:, 1:2])
    off = (maxc + 1) .* Float64.(idxs)
    shifted = copy(Float64.(boxes))
    shifted[:, 1] .+= off; shifted[:, 2] .+= off; shifted[:, 3] .+= off; shifted[:, 4] .+= off
    return nms(shifted, scores, thresh)
end

# --- box decode (Faster R-CNN box-to-box transform; torchvision BoxCoder.decode) ---

const SCALE_CLAMP = log(1000.0 / 16.0)

"""
    decode_boxes(deltas, boxes, weights) -> [M, B]

`deltas` [M,B] (B a multiple of 4, class-specific groups), `boxes` [M,4] anchors/proposals, `weights`
(wx,wy,ww,wh). Output xyxy per group, layout [x1,y1,x2,y2] interleaved per class.
"""
function decode_boxes(deltas::AbstractMatrix, boxes::AbstractMatrix, weights::NTuple{4,<:Real})
    M, B = size(deltas); nc = B ÷ 4
    out = Matrix{Float64}(undef, M, B)
    wx, wy, ww, wh = weights
    @inbounds for m in 1:M
        w = boxes[m, 3] - boxes[m, 1]; h = boxes[m, 4] - boxes[m, 2]
        cx = boxes[m, 1] + 0.5w; cy = boxes[m, 2] + 0.5h
        for c in 0:(nc - 1)
            dx = deltas[m, 4c + 1] / wx; dy = deltas[m, 4c + 2] / wy
            dw = min(deltas[m, 4c + 3] / ww, SCALE_CLAMP); dh = min(deltas[m, 4c + 4] / wh, SCALE_CLAMP)
            pcx = dx * w + cx; pcy = dy * h + cy
            pw = exp(dw) * w; ph = exp(dh) * h
            out[m, 4c + 1] = pcx - 0.5pw; out[m, 4c + 2] = pcy - 0.5ph
            out[m, 4c + 3] = pcx + 0.5pw; out[m, 4c + 4] = pcy + 0.5ph
        end
    end
    return out
end

# --- anchor generation (DefaultAnchorGenerator grid), order (h, w, a) ---

"""
    generate_anchors(H, W, stride, cell) -> [H*W*A, 4]

Grid anchors for a feature map of size H×W: shift the per-location base `cell` [A,4] by the stride
across the grid, in (h, w, a) order to match `objectness.permute(0,2,3,1).flatten`.
"""
function generate_anchors(H::Int, W::Int, stride::Real, cell::AbstractMatrix)
    A = size(cell, 1)
    out = Matrix{Float64}(undef, H * W * A, 4)
    r = 0
    @inbounds for h in 0:(H - 1), w in 0:(W - 1)
        sx = w * stride; sy = h * stride
        for a in 1:A
            r += 1
            out[r, 1] = cell[a, 1] + sx; out[r, 2] = cell[a, 2] + sy
            out[r, 3] = cell[a, 3] + sx; out[r, 4] = cell[a, 4] + sy
        end
    end
    return out
end

# --- ROIAlign (torchvision semantics: aligned=True, sampling_ratio=0 -> adaptive) ---

# ROIAlign for one output bin, accumulated across every channel into `acc` (length C, Float64).
# The sampling geometry (sample coordinates, the four neighbour indices, the four bilinear weights) is
# channel-independent, so it is computed once per sample point; the channel sweep then runs over `c`,
# the stride-1 (contiguous) dimension of `feat` ([C,H,W]) in column-major Julia, so each of the four
# neighbour reads is a contiguous C-length run and every loaded cache line is fully used (rather than
# loading a fresh line per channel, as a per-channel bilinear call did). The per-channel arithmetic
# and the per-channel accumulation order over samples match the previous per-channel formulation
# exactly, so the result is bit-for-bit unchanged. `acc` is reset here for this bin.
function _roi_bin_accum!(acc::AbstractVector{Float64}, feat::AbstractArray{<:Real,3},
                         C::Int, H::Int, W::Int, sh::Float64, sw::Float64, bh::Float64, bw::Float64,
                         ph::Int, pw::Int, gh::Int, gw::Int)
    @inbounds for c in 1:C
        acc[c] = 0.0
    end
    @inbounds for iy in 0:(gh - 1)
        y = sh + ph * bh + (iy + 0.5) * bh / gh
        for ix in 0:(gw - 1)
            x = sw + pw * bw + (ix + 0.5) * bw / gw
            # Out-of-range sample contributes 0 to every channel (the old `_bilinear` early-returned 0;
            # skipping the add is the same no-op for every channel).
            (y < -1.0 || y > H || x < -1.0 || x > W) && continue
            yy = y <= 0 ? 0.0 : y
            xx = x <= 0 ? 0.0 : x
            yl = floor(Int, yy); xl = floor(Int, xx)
            if yl >= H - 1
                yl = H - 1; yh = H - 1; yy = Float64(yl)
            else
                yh = yl + 1
            end
            if xl >= W - 1
                xl = W - 1; xh = W - 1; xx = Float64(xl)
            else
                xh = xl + 1
            end
            ly = yy - yl; lx = xx - xl; hy = 1 - ly; hx = 1 - lx
            w00 = hy * hx; w01 = hy * lx; w10 = ly * hx; w11 = ly * lx
            # +1 for 1-based Julia (feat[c, row, col]); c is the contiguous axis.
            r0 = yl + 1; r1 = yh + 1; q0 = xl + 1; q1 = xh + 1
            for c in 1:C
                acc[c] += w00 * feat[c, r0, q0] + w01 * feat[c, r0, q1] +
                          w10 * feat[c, r1, q0] + w11 * feat[c, r1, q1]
            end
        end
    end
    return acc
end

"""
    roi_align!(out, feat, boxes, scale; pooled=7, ratio=0, aligned=true)

ROIAlign of `feat` [C,H,W] at `boxes` [K,4] (xyxy, input-image coords) into `out` [K,C,pooled,pooled].
`scale` is the feature/image spatial scale; `ratio`=0 means adaptive sampling (ceil(roi/pooled)).
`aligned` selects the pixel-coordinate convention of `torchvision.ops.roi_align`: `true` shifts
sampling by a half pixel (`box*scale - 0.5`); `false` (the torchvision detection MultiScaleRoIAlign
default) uses no offset and clamps each ROI's width/height to at least one pixel (the legacy
malformed-ROI guard).
"""
function roi_align!(out::AbstractArray{<:Real,4}, feat::AbstractArray{<:Real,3},
                    boxes::AbstractMatrix, scale::Real; pooled::Int=7, ratio::Int=0, aligned::Bool=true)
    C, H, W = size(feat)
    K = size(boxes, 1)
    off = aligned ? 0.5 : 0.0
    acc = Vector{Float64}(undef, C)
    @inbounds for k in 1:K
        sw = boxes[k, 1] * scale - off; sh = boxes[k, 2] * scale - off
        ew = boxes[k, 3] * scale - off; eh = boxes[k, 4] * scale - off
        rw = ew - sw; rh = eh - sh
        aligned || (rw = max(rw, 1.0); rh = max(rh, 1.0))
        bw = rw / pooled; bh = rh / pooled
        gh = ratio > 0 ? ratio : max(1, ceil(Int, rh / pooled))
        gw = ratio > 0 ? ratio : max(1, ceil(Int, rw / pooled))
        cnt = gh * gw
        for ph in 0:(pooled - 1), pw in 0:(pooled - 1)
            _roi_bin_accum!(acc, feat, C, H, W, sh, sw, bh, bw, ph, pw, gh, gw)
            for c in 1:C
                out[k, c, ph + 1, pw + 1] = acc[c] / cnt
            end
        end
    end
    return out
end

"""
    roi_align_wire!(out, feat, boxes, scale; pooled=7, ratio=0, aligned=true)

ROIAlign writing directly into the executable's WIRE layout `out` [pooled,pooled,C,K] =
(pw,ph,C,K) (the Julia col-major reverse of torch [K,C,7,7]), instead of [K,C,7,7]+permutedims.
Lets a meta stage ROI features straight into a shared-memory scratch slot. Same per-element math as
`roi_align!` (the value is computed in Float64 and rounded once on store), so an `out::Array{Float32}`
is bit-identical to the old `Float64` roi_align + final `Float32` convert. See `roi_align!` for the
`aligned` convention (half-pixel offset `true` vs torchvision detection `false`).
"""
function roi_align_wire!(out::AbstractArray{<:Real,4}, feat::AbstractArray{<:Real,3},
                         boxes::AbstractMatrix, scale::Real; pooled::Int=7, ratio::Int=0, aligned::Bool=true)
    C, H, W = size(feat)
    K = size(boxes, 1)
    off = aligned ? 0.5 : 0.0
    acc = Vector{Float64}(undef, C)
    @inbounds for k in 1:K
        sw = boxes[k, 1] * scale - off; sh = boxes[k, 2] * scale - off
        ew = boxes[k, 3] * scale - off; eh = boxes[k, 4] * scale - off
        rw = ew - sw; rh = eh - sh
        aligned || (rw = max(rw, 1.0); rh = max(rh, 1.0))
        bw = rw / pooled; bh = rh / pooled
        gh = ratio > 0 ? ratio : max(1, ceil(Int, rh / pooled))
        gw = ratio > 0 ? ratio : max(1, ceil(Int, rw / pooled))
        cnt = gh * gw
        for ph in 0:(pooled - 1), pw in 0:(pooled - 1)
            _roi_bin_accum!(acc, feat, C, H, W, sh, sw, bh, bw, ph, pw, gh, gw)
            for c in 1:C
                out[pw + 1, ph + 1, c, k] = acc[c] / cnt
            end
        end
    end
    return out
end

# --- orchestration: RPN proposal selection + fast-rcnn final inference ---

"""
    select_rpn_proposals(boxes_levels, scores_levels, imgH, imgW; pre, post, nms_thresh) -> [Kp,4]

find_top_rpn_proposals: per-level top-`pre` by objectness, clip to image, drop empty, level-aware
NMS, then top-`post`. `boxes_levels[i]`/`scores_levels[i]` are the decoded boxes/objectness for level i.
"""
function select_rpn_proposals(boxes_levels::Vector{<:AbstractMatrix}, scores_levels::Vector{<:AbstractVector},
                              imgH::Real, imgW::Real; pre::Int=1000, post::Int=1000, nms_thresh::Real=0.7)
    bs = Matrix{Float64}[]; ss = Vector{Float64}[]; lv = Vector{Int}[]
    for (lid, (b, s)) in enumerate(zip(boxes_levels, scores_levels))
        n = min(length(s), pre)
        idx = partialsortperm(s, 1:n; rev=true)
        push!(bs, b[idx, :]); push!(ss, Float64.(s[idx])); push!(lv, fill(lid, n))
    end
    boxes = reduce(vcat, bs); scores = reduce(vcat, ss); lvl = reduce(vcat, lv)
    @views boxes[:, 1] .= clamp.(boxes[:, 1], 0, imgW); boxes[:, 3] .= clamp.(boxes[:, 3], 0, imgW)
    @views boxes[:, 2] .= clamp.(boxes[:, 2], 0, imgH); boxes[:, 4] .= clamp.(boxes[:, 4], 0, imgH)
    keep0 = findall(i -> boxes[i, 3] > boxes[i, 1] && boxes[i, 4] > boxes[i, 2], 1:size(boxes, 1))
    boxes, scores, lvl = boxes[keep0, :], scores[keep0], lvl[keep0]
    k = batched_nms(boxes, scores, lvl, nms_thresh)
    k = k[1:min(post, length(k))]
    return boxes[k, :]
end

_softmax_row(v) = (e = exp.(v .- maximum(v)); e ./ sum(e))

"""
    fast_rcnn_inference(cls, deltas, proposals, imgH, imgW; score_thresh, nms_thresh, topk, weights,
                        bg_first=false) -> (boxes [N,4], scores [N], classes [N])

Second-stage decode + per-class NMS. `cls` [K, C] softmax logits, `deltas` [K, nrc*4]
(nrc = number of foreground classes, or `C` when the background also has a delta group, or 1 for
class-agnostic), `proposals` [K,4]. Mirrors fast_rcnn_inference_single_image.

`bg_first` selects where the background score sits: `false` treats the **last** column as background
and columns `1:end-1` as foreground; `true` (torchvision FastRCNNPredictor) treats column **1** as
background and columns `2:end` as foreground. In both cases the per-class delta group for foreground
column `c` is `c-1` (0-based), and the emitted class id is `c-1`.

`min_size` drops decoded boxes whose width or height is below it before the final NMS (torchvision's
`postprocess_detections` uses `1e-2`); the default `0.0` keeps every box.
"""
function fast_rcnn_inference(cls::AbstractMatrix, deltas::AbstractMatrix, proposals::AbstractMatrix,
                             imgH::Real, imgW::Real; score_thresh::Real, nms_thresh::Real,
                             topk::Int, weights::NTuple{4,<:Real}=(10.0, 10.0, 5.0, 5.0),
                             bg_first::Bool=false, min_size::Real=0.0)
    K = size(cls, 1); nrc = size(deltas, 2) ÷ 4
    dec = decode_boxes(deltas, proposals, weights)            # [K, nrc*4]
    @inbounds for m in 1:K, c in 0:(nrc - 1)
        dec[m, 4c + 1] = clamp(dec[m, 4c + 1], 0, imgW); dec[m, 4c + 3] = clamp(dec[m, 4c + 3], 0, imgW)
        dec[m, 4c + 2] = clamp(dec[m, 4c + 2], 0, imgH); dec[m, 4c + 4] = clamp(dec[m, 4c + 4], 0, imgH)
    end
    fg_cols = bg_first ? (2:size(cls, 2)) : (1:(size(cls, 2) - 1))
    cb = Vector{Float64}[]; cs = Float64[]; cc = Int[]
    @inbounds for k in 1:K
        p = _softmax_row(@view cls[k, :])
        for c in fg_cols
            if p[c] > score_thresh
                ci = nrc == 1 ? 0 : (c - 1)
                push!(cb, dec[k, (4ci + 1):(4ci + 4)]); push!(cs, p[c]); push!(cc, c - 1)
            end
        end
    end
    isempty(cs) && return (Matrix{Float64}(undef, 0, 4), Float64[], Int[])
    boxes = reduce(vcat, (b' for b in cb))
    if min_size > 0
        ks = findall(i -> (boxes[i, 3] - boxes[i, 1]) >= min_size && (boxes[i, 4] - boxes[i, 2]) >= min_size,
                     1:size(boxes, 1))
        isempty(ks) && return (Matrix{Float64}(undef, 0, 4), Float64[], Int[])
        boxes, cs, cc = boxes[ks, :], cs[ks], cc[ks]
    end
    keep = batched_nms(boxes, cs, cc, nms_thresh)
    topk >= 0 && (keep = keep[1:min(topk, length(keep))])
    return (boxes[keep, :], cs[keep], cc[keep])
end

# --- layout helpers: bundle outputs are Julia column-major (W,H,C,1) = reverse of torch (1,C,H,W) ---

"""
    objectness_flat(O) -> Vector

RPN objectness `O` (W,H,A,1) flattened to (h,w,a) order to match `generate_anchors` and
`obj.permute(0,2,3,1).flatten`.
"""
function objectness_flat(O::AbstractArray)
    Wd, Hd, A = size(O, 1), size(O, 2), size(O, 3)
    v = Vector{Float64}(undef, Hd * Wd * A); r = 0
    @inbounds for h in 1:Hd, w in 1:Wd, a in 1:A
        r += 1; v[r] = O[w, h, a, 1]
    end
    return v
end

"""
    deltas_matrix(D) -> [H*W*A, 4]

RPN anchor deltas `D` (W,H,4A,1) reshaped to [H*W*A,4] in (h,w,a) order (a inner), matching the
RPN forward's view/permute/flatten.
"""
function deltas_matrix(D::AbstractArray)
    Wd, Hd = size(D, 1), size(D, 2); A = size(D, 3) ÷ 4
    m = Matrix{Float64}(undef, Hd * Wd * A, 4); r = 0
    @inbounds for h in 1:Hd, w in 1:Wd, a in 0:(A - 1)
        r += 1
        m[r, 1] = D[w, h, 4a + 1, 1]; m[r, 2] = D[w, h, 4a + 2, 1]
        m[r, 3] = D[w, h, 4a + 3, 1]; m[r, 4] = D[w, h, 4a + 4, 1]
    end
    return m
end

"feature map (W,H,C,1) -> [C,H,W] for roi_align!"
feature_chw(F::AbstractArray) = permutedims(dropdims(F; dims=4), (3, 2, 1))

"FPN ROIPooler level for a box (FPN paper Eqn.1): clamp(floor(log2(sqrt(area)/canon_size+1e-8)+canon_level),min,max)-min"
function assign_level(box::AbstractVector; canon_level::Int=4, canon_size::Real=224,
                      min_level::Int=2, max_level::Int=5)
    area = max(0.0, (box[3] - box[1]) * (box[4] - box[2]))
    return Int(clamp(floor(log2(sqrt(area) / canon_size + 1e-8) + canon_level), min_level, max_level)) - min_level
end

end # module DetectionGlue
