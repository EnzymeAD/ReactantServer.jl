# Converter handler: a standard torchvision Faster R-CNN object detector
# (`torchvision.models.detection.fasterrcnn_resnet50_fpn`, an FPN backbone + RPN + RoIHeads
# GeneralizedRCNN). Like any two-stage detector its middle (RPN proposal NMS, ROIAlign, box decode,
# per-class NMS) is data-dependent and cannot live in a static StableHLO graph, so one source model
# is emitted as THREE bundles under the output root:
#
#   <bundle>_stage1   backbone + RPN head             (StableHLO; 14 outputs: 4 ROI feats + 5 obj + 5 deltas)
#   <bundle>_stage2   box_head + box_predictor         (StableHLO; cls_logits + bbox_deltas, fixed K rois)
#   <bundle>          kind: meta bundle that chains stage1 -> DetectionGlue -> stage2 -> final NMS
#
# This works on a live `nn.Module`: it builds the torchvision model, wraps `backbone`+`rpn.head` and
# `box_head`+`box_predictor` as two small modules, and exports them with `torch.export` (via
# `export_bundle(:pytorch, ...)`). There is no `torch.jit` load and no reaching into a scripted graph's
# frozen internals. The meta bundle's model.jl bakes the per-model config read
# from the model (cell_anchors, box-coder weights, score/NMS/topk) and reproduces the GeneralizedRCNN
# detection pipeline in Julia via `ReactantServer.DetectionGlue`.
#
# Options:
#   weights       ("DEFAULT" | "none" | <path>) how to populate the model. "DEFAULT" downloads the
#                   pretrained COCO weights (the runnable demo path); "none" leaves it random
#                   (structure only); a path loads a state_dict .pth into the architecture.
#   num_classes   (int, default 91) classes INCLUDING background; set this when loading a custom head
#   image_size    (int, default 640) canonical square input edge; must be divisible by 64 (FPN p6/pool
#                   stride). The served wrapper letterboxes the image to this; the model itself is not
#                   resized at run time.
#   input_shapes  (optional list of [W,H]) compile stage1 for several input shapes (aspect-ratio
#                   variants) sharing one weight set, each edge divisible by 64; the meta routes each
#                   request to the matching variant by its input shape. stage2 is shared (7x7 ROI input).
#   input_dtype   ("u8" default | "f32") client image dtype. u8 is divided by 255 then ImageNet-normalized
#                   inside stage1; f32 is assumed already in [0,1] and only normalized.
#   output_cols   (int, default 6) per-detection width: 5 = [box4, score]; 6 = [box4, score, class].
#                   torchvision class ids are 1..num_classes-1 (background is class 0, dropped).
#
# The handler runs after torch/torchax/triton import and `using ReactantServerExport`, so PythonCall
# is available. It writes files directly (no ReactantServer dependency at convert time).

using PythonCall
using ReactantServerExport
const RSE = ReactantServerExport

pyimport("torchvision")  # registers the custom ops (nms, roi_align) referenced by the detector

# Stage wraps. stage1 takes a batched NCHW image, bakes /255 (u8) + ImageNet normalize, and returns
# the 4 ROI feature maps ('0'..'3') plus the 5 per-level objectness and box-delta maps. stage2 runs the
# box head + predictor on fixed-K ROI features. Both are plain nn.Modules that torch.export cleanly.
pyexec("""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def _build_frcnn(weights, num_classes, ckpt_path):
    if weights == "DEFAULT":
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        if ckpt_path is not None:
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval()

class _Stage1(torch.nn.Module):
    def __init__(self, model, u8):
        super().__init__()
        self.backbone = model.backbone
        self.rpn_head = model.rpn.head
        self.u8 = bool(u8)
        self.register_buffer("mean", torch.tensor(model.transform.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(model.transform.image_std).view(1, 3, 1, 1))
        self.roi_keys = ["0", "1", "2", "3"]
    def forward(self, image):                          # [1,3,H,W]
        x = image.to(torch.float32)
        if self.u8:
            x = x / 255.0
        x = (x - self.mean) / self.std
        feats = self.backbone(x)                       # OrderedDict 0,1,2,3,pool
        objs, deltas = self.rpn_head(list(feats.values()))
        return tuple([feats[k] for k in self.roi_keys] + list(objs) + list(deltas))

class _Stage2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.box_head = model.roi_heads.box_head
        self.box_predictor = model.roi_heads.box_predictor
    def forward(self, roi_feats):                      # [K,256,7,7]
        h = self.box_head(roi_feats)
        return self.box_predictor(h)
""", @__MODULE__)

const _JULIA_DTYPE = Dict("u8" => UInt8, "f32" => Float32)

_matrix_literal(m) = "[" * join([join(string.(m[r, :]), " ") for r in 1:size(m, 1)], "; ") * "]"

function handler(ctx)
    out_root = dirname(ctx.out_dir)
    s1_name = ctx.bundle_name * "_stage1"
    s2_name = ctx.bundle_name * "_stage2"
    s1_dir = joinpath(out_root, s1_name)
    s2_dir = joinpath(out_root, s2_name)

    img = Int(get(ctx.options, "image_size", 640))
    img % 64 == 0 || error("image_size=$img must be divisible by 64 (FPN p6/pool stride)")
    dtype_tok = String(get(ctx.options, "input_dtype", "u8"))
    haskey(_JULIA_DTYPE, dtype_tok) || error("input_dtype must be u8 or f32, got $dtype_tok")
    T = _JULIA_DTYPE[dtype_tok]
    ncol = Int(get(ctx.options, "output_cols", 6))
    ncol in (5, 6) || error("output_cols must be 5 or 6, got $ncol")
    num_classes = Int(get(ctx.options, "num_classes", 91))
    weights_tok = String(get(ctx.options, "weights", "DEFAULT"))

    # Optional multi-shape: [W, H] pairs each divisible by 64, sharing one weight set.
    shapes_opt = get(ctx.options, "input_shapes", nothing)
    shapes = Tuple{Int,Int}[]
    if shapes_opt !== nothing
        shapes_opt isa AbstractVector || error("input_shapes must be a list of [W, H] pairs")
        for (i, p) in enumerate(shapes_opt)
            (p isa AbstractVector && length(p) == 2) || error("input_shapes[$i] must be a [W, H] pair")
            (Int(p[1]) % 64 == 0 && Int(p[2]) % 64 == 0) ||
                error("input_shapes[$i]=$(Int(p[1]))x$(Int(p[2])) must have both edges divisible by 64")
            push!(shapes, (Int(p[1]), Int(p[2])))
        end
    end
    multishape = !isempty(shapes)

    # Build the torchvision model. "DEFAULT" pulls pretrained COCO weights; "none" is random; any other
    # value is a state_dict path loaded into the architecture (num_classes must match the saved head).
    ckpt = (weights_tok == "DEFAULT" || weights_tok == "none") ? nothing : weights_tok
    model = pyeval("_build_frcnn", @__MODULE__)(weights_tok, num_classes, ckpt)
    u8 = dtype_tok == "u8"
    stage1 = pyeval("_Stage1", @__MODULE__)(model, u8)
    stage2 = pyeval("_Stage2", @__MODULE__)(model)

    s1_outs = ["feat_0", "feat_1", "feat_2", "feat_3",
               "obj_0", "obj_1", "obj_2", "obj_3", "obj_4",
               "delta_0", "delta_1", "delta_2", "delta_3", "delta_4"]
    # stage1 input is a batched NCHW image: julia (W,H,3,1) -> torch [1,3,H,W]. The backbone is pure
    # conv, so a zero trace at each declared shape is valid; multi-shape variants share one weight set.
    isdir(s1_dir) && rm(s1_dir; recursive=true)
    ex1 = zeros(T, img, img, 3, 1)
    s1_variants = multishape ? [(zeros(T, W, H, 3, 1),) for (W, H) in shapes] : nothing
    RSE.export_bundle(Val(:pytorch), stage1, multishape ? first(s1_variants) : (ex1,);
        dir=s1_dir, name=s1_name, input_names=["INPUT__0"], output_names=s1_outs,
        batch_sizes=[1], shape_variants=s1_variants, matmul_precision="highest",
        # image input (W,H,3,1) -> "whcn"; the 14 feature/obj/delta maps (W,H,C,1) -> "whcn".
        axis_letters=merge(Dict("INPUT__0" => ['w', 'h', 'c']), Dict(n => ['w', 'h', 'c'] for n in s1_outs)))

    # stage2: box head + predictor on fixed-K roi features. torch [K,256,7,7] == julia (7,7,256,K).
    roi_k = Int(pyconvert(Int, pyeval("int", @__MODULE__)(model.rpn._post_nms_top_n["testing"])))
    isdir(s2_dir) && rm(s2_dir; recursive=true)
    ex2 = zeros(Float32, 7, 7, 256, roi_k)
    RSE.export_bundle(Val(:pytorch), stage2, (ex2,);
        dir=s2_dir, name=s2_name, input_names=["ROI_FEATS"],
        output_names=["cls_logits", "bbox_deltas"], batch_sizes=[roi_k], matmul_precision="highest")

    # Per-model config read from the live model: per-level cell_anchors [3,4], box-coder weights,
    # RPN pre-NMS top-k + NMS threshold, and the final score/NMS/topk.
    ag = model.rpn.anchor_generator
    cells = [pyconvert(Matrix{Float64}, ag.cell_anchors[i - 1].numpy().astype("float64")) for i in 1:5]
    rpnw = pyconvert(NTuple{4,Float64}, pyeval("tuple", @__MODULE__)(model.rpn.box_coder.weights))
    roiw = pyconvert(NTuple{4,Float64}, pyeval("tuple", @__MODULE__)(model.roi_heads.box_coder.weights))
    rpn_pre = pyconvert(Int, pyeval("int", @__MODULE__)(model.rpn._pre_nms_top_n["testing"]))
    rpn_nms = pyconvert(Float64, pyeval("float", @__MODULE__)(model.rpn.nms_thresh))
    score = pyconvert(Float64, pyeval("float", @__MODULE__)(model.roi_heads.score_thresh))
    nms = pyconvert(Float64, pyeval("float", @__MODULE__)(model.roi_heads.nms_thresh))
    topk = pyconvert(Int, pyeval("int", @__MODULE__)(model.roi_heads.detections_per_img))

    out_line = ncol == 5 ?
        "isempty(sc) ? zeros(Float32,5,0) : Array{Float32}(permutedims(hcat(bx,sc),(2,1)))" :
        "isempty(sc) ? zeros(Float32,6,0) : Array{Float32}(permutedims(hcat(bx,sc,Float64.(cl)),(2,1)))"

    cell_lits = join(["  " * _matrix_literal(c) for c in cells], ",\n")
    rpnw_lit = "(" * join(rpnw, ",") * ")"
    roiw_lit = "(" * join(roiw, ",") * ")"
    # Built flush-left (no source indentation) so the emitted file is clean Julia.
    model_jl = """
const _G = ReactantServer.DetectionGlue
const _STR=[4,8,16,32,64]; const _SCALES=[0.25,0.125,0.0625,0.03125]
const _CELL = Matrix{Float64}[
$cell_lits
]
const _IMG=$img; const _K=$roi_k; const _RPNW=$rpnw_lit; const _ROIW=$roiw_lit
const _PRE=$rpn_pre; const _RPNNMS=$rpn_nms; const _SCORE=$score; const _NMS=$nms; const _TOPK=$topk
const _OBJ=["obj_0","obj_1","obj_2","obj_3","obj_4"]
const _DEL=["delta_0","delta_1","delta_2","delta_3","delta_4"]
const _FEAT=["feat_0","feat_1","feat_2","feat_3"]
function _run(inputs, call)
    # The wrapper letterboxes to one of the compiled shapes; the image's own W,H bound the boxes.
    iw=size(inputs[1].data,1); ih=size(inputs[1].data,2)
    s1 = call("$s1_name", inputs)
    d = Dict(t.name=>t.data for t in s1)
    bl=Matrix{Float64}[]; sl=Vector{Float64}[]
    for i in 1:5
        O=d[_OBJ[i]]; D=d[_DEL[i]]
        anc=_G.generate_anchors(size(O,2),size(O,1),_STR[i],_CELL[i])
        push!(bl,_G.decode_boxes(_G.deltas_matrix(D),anc,_RPNW)); push!(sl,_G.objectness_flat(O))
    end
    pb=_G.select_rpn_proposals(bl,sl,ih,iw;pre=_PRE,post=_K,nms_thresh=_RPNNMS); Kp=size(pb,1)
    feats=[_G.feature_chw(d[f]) for f in _FEAT]
    # torchvision detection ROIAlign is aligned=false, sampling_ratio=2. Stage straight into a
    # Float32 wire-layout (W,H,C,K) scratch buffer (shared-memory-backed in multi-worker mode).
    roi=call.scratch((7,7,256,_K),Float32); fill!(roi,0f0)
    lv=[_G.assign_level(@view pb[k,:]) for k in 1:Kp]
    for l in 0:3
        sel=findall(==(l),lv); isempty(sel)&&continue
        _G.roi_align_wire!(view(roi,:,:,:,sel),feats[l+1],pb[sel,:],_SCALES[l+1];ratio=2,aligned=false)
    end
    s2=call("$s2_name",[ReactantServer.NamedTensor("ROI_FEATS",roi)])
    d2=Dict(t.name=>t.data for t in s2)
    cls=permutedims(d2["cls_logits"],(2,1))[1:Kp,:]; dl=permutedims(d2["bbox_deltas"],(2,1))[1:Kp,:]
    # torchvision FastRCNNPredictor puts background at class 0 -> bg_first=true; postprocess drops
    # final boxes smaller than 1e-2 px (remove_small_boxes) -> min_size=1e-2.
    bx,sc,cl=_G.fast_rcnn_inference(cls,dl,pb,ih,iw;score_thresh=_SCORE,nms_thresh=_NMS,topk=_TOPK,weights=_ROIW,bg_first=true,min_size=1e-2)
    D = $out_line
    return [ReactantServer.NamedTensor("OUTPUT__0", D)]
end
register_meta_model("$(ctx.bundle_name)"; run=_run)
"""

    # Client input is a batched NCHW image: julia (W,H,3,1), batch axis last (n). Single-shape bakes
    # the canonical square; multi-shape leaves w,h variable (-1) so any compiled aspect ratio is accepted.
    wdim = multishape ? -1 : img
    hdim = multishape ? -1 : img
    manifest = """
format_version: "2.0"
name: $(ctx.bundle_name)
kind: meta
meta:
  calls: [$s1_name, $s2_name]
client_inputs:
  - {name: INPUT__0, dtype: $dtype_tok, shape: whcn, dims: {w: $wdim, h: $hdim, c: 3}}
client_outputs:
  - {name: OUTPUT__0, dtype: f32, shape: dc, dims: {d: -1, c: $ncol}}
"""

    isdir(ctx.out_dir) && rm(ctx.out_dir; recursive=true)
    mkpath(ctx.out_dir)
    write(joinpath(ctx.out_dir, "model.jl"), model_jl)
    write(joinpath(ctx.out_dir, "manifest.yaml"), manifest)

    return [1]
end

handler
