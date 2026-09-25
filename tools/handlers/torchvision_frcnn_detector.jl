# Converter handler: a standard torchvision Faster R-CNN object detector
# (`torchvision.models.detection.fasterrcnn_resnet50_fpn`, an FPN backbone + RPN + RoIHeads
# GeneralizedRCNN), exported as ONE plain bundle `<bundle>` whose single StableHLO program runs the
# whole detector: backbone + RPN head, RPN proposal selection + NMS, ROIAlign, box head, and
# per-class NMS. The data-dependent middle has a fixed-shape form (static K proposals, masks, stable
# sorts), see `ReactantServerExport.Detection`; the program returns a fixed-size detection buffer
# plus a count, and the bundle's `model.jl` trims it to the detections found.
#
# This works on a live `nn.Module`: it builds the torchvision model, wraps `backbone`+`rpn.head` and
# `box_head`+`box_predictor` as two small modules, exports each with `torch.export` (via
# `export_bundle(:pytorch, ...)`) into a scratch directory, and fuses them with the traced glue via
# `export_two_stage_detector`. There is no `torch.jit` load and no reaching into a scripted graph's
# frozen internals. The glue's configuration (cell anchors, box-coder weights, RPN and final
# thresholds, top-k) is read off the live model.
#
# Options:
#   weights       ("DEFAULT" | "none" | <path>) how to populate the model. "DEFAULT" downloads the
#                   pretrained COCO weights (the runnable demo path); "none" leaves it random
#                   (structure only); a path loads a state_dict .pth into the architecture.
#   num_classes   (int, default 91) classes INCLUDING background; set this when loading a custom head
#   image_size    (int, default 640) canonical square input edge; must be divisible by 64 (FPN p6/pool
#                   stride). The client resizes or letterboxes the image to this; the model itself is
#                   not resized at run time.
#   input_shapes  (optional list of [W,H]) compile the detector for several input shapes
#                   (aspect-ratio variants) sharing one weight set, each edge divisible by 64; the
#                   server routes each request to the matching variant by its input shape.
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
pyexec(
    """
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
    """, @__MODULE__
)

const _JULIA_DTYPE = Dict("u8" => UInt8, "f32" => Float32)

function handler(ctx)
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
    shapes = Tuple{Int, Int}[]
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

    s1_outs = [
        "feat_0", "feat_1", "feat_2", "feat_3",
        "obj_0", "obj_1", "obj_2", "obj_3", "obj_4",
        "delta_0", "delta_1", "delta_2", "delta_3", "delta_4",
    ]
    roi_k = pyconvert(Int, pyeval("int", @__MODULE__)(model.rpn._post_nms_top_n["testing"]))

    # Export each stage on its own into a scratch directory, then read the programs and weights back
    # for the fused export. stage1 input is a batched NCHW image: julia (W,H,3,1) -> torch [1,3,H,W].
    # The backbone is pure conv, so a zero trace at each declared shape is valid; multi-shape variants
    # share one weight set. stage2 is the box head on fixed-K ROI features: torch [K,256,7,7].
    S1, S2 = mktempdir() do scratch
        s1_dir = joinpath(scratch, "stage1"); s2_dir = joinpath(scratch, "stage2")
        s1_variants = multishape ? [(zeros(T, W, H, 3, 1),) for (W, H) in shapes] : nothing
        RSE.export_bundle(
            Val(:pytorch), stage1, multishape ? first(s1_variants) : (zeros(T, img, img, 3, 1),);
            dir = s1_dir, name = "stage1", input_names = ["INPUT__0"], output_names = s1_outs,
            batch_sizes = [1], shape_variants = s1_variants, matmul_precision = "highest",
            axis_letters = merge(Dict("INPUT__0" => ['w', 'h', 'c']), Dict(n => ['w', 'h', 'c'] for n in s1_outs))
        )
        RSE.export_bundle(
            Val(:pytorch), stage2, (zeros(Float32, 7, 7, 256, roi_k),);
            dir = s2_dir, name = "stage2", input_names = ["ROI_FEATS"],
            output_names = ["cls_logits", "bbox_deltas"], batch_sizes = [roi_k], matmul_precision = "highest"
        )
        (RSE.read_stage_bundle(s1_dir), RSE.read_stage_bundle(s2_dir))
    end

    # The glue configuration, read from the live model: per-level cell_anchors [3,4], box-coder
    # weights, RPN pre-NMS top-k + NMS threshold, and the final score/NMS/topk. The rest are
    # torchvision conventions: ROIAlign is aligned=false with sampling_ratio=2 (MultiScaleRoIAlign),
    # FastRCNNPredictor puts background in class column 0 (bg_first), and postprocess_detections drops
    # final boxes smaller than 1e-2 px (remove_small_boxes).
    ag = model.rpn.anchor_generator
    cfg = RSE.DetectorConfig(;
        strides = [4, 8, 16, 32, 64], scales = [0.25, 0.125, 0.0625, 0.03125],
        cell_anchors = [pyconvert(Matrix{Float64}, ag.cell_anchors[i - 1].numpy().astype("float64")) for i in 1:5],
        rpn_weights = pyconvert(NTuple{4, Float64}, pyeval("tuple", @__MODULE__)(model.rpn.box_coder.weights)),
        roi_weights = pyconvert(NTuple{4, Float64}, pyeval("tuple", @__MODULE__)(model.roi_heads.box_coder.weights)),
        pre_nms_topk = pyconvert(Int, pyeval("int", @__MODULE__)(model.rpn._pre_nms_top_n["testing"])),
        post_nms_topk = roi_k,
        rpn_nms_thresh = pyconvert(Float64, pyeval("float", @__MODULE__)(model.rpn.nms_thresh)),
        score_thresh = pyconvert(Float64, pyeval("float", @__MODULE__)(model.roi_heads.score_thresh)),
        nms_thresh = pyconvert(Float64, pyeval("float", @__MODULE__)(model.roi_heads.nms_thresh)),
        detections_per_img = pyconvert(Int, pyeval("int", @__MODULE__)(model.roi_heads.detections_per_img)),
        pooled = 7, sampling_ratio = 2, aligned = false, bg_first = true, min_size = 1.0e-2,
    )

    # Client input is one NCHW image: julia (W,H,3,1). Single-shape bakes the canonical square;
    # multi-shape leaves w,h variable (-1) so any compiled aspect ratio is accepted.
    wdim = multishape ? -1 : img
    hdim = multishape ? -1 : img
    isdir(ctx.out_dir) && rm(ctx.out_dir; recursive = true)
    RSE.export_two_stage_detector(
        ctx.out_dir; name = ctx.bundle_name,
        input = RSE.IOSpec("INPUT__0", T, [wdim, hdim, 3, 1]; letters = ['w', 'h', 'c', 'a']),
        stage1 = S1.texts, stage1_weights = S1.weights,
        stage2 = S2.texts[Int[]], stage2_weights = S2.weights, cfg,
        input_shapes = multishape ? [[W, H] for (W, H) in shapes] : nothing,
        output_columns = ncol, output_layout = :cn,
        provenance = Dict("source" => "torchvision fasterrcnn_resnet50_fpn", "weights" => weights_tok),
    )
    return [1]
end

handler
