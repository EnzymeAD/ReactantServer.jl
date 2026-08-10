# Object Detection (GeneralizedRCNN)

A two-stage object detector is the canonical case for a [meta model](meta_models.md). A torchvision
Faster R-CNN (`torchvision.models.detection.fasterrcnn_resnet50_fpn`, an FPN backbone + RPN +
RoIHeads GeneralizedRCNN) is mostly dense tensor math, but the middle of the pipeline is not:
selecting RPN proposals, running NMS, pooling ROI features with `roi_align`, decoding boxes, and the
final per-class NMS are all data-dependent and cannot be captured by `torch.export` as a single
static graph. The shape of the work changes with the contents of the tensors, which is exactly
what a static StableHLO program cannot express.

The package ships a converter that handles this split for you. It produces two dense StableHLO
bundles for the parts that trace cleanly and a meta bundle whose `model.jl` runs the data-dependent
glue in Julia between them. The reusable detection math (NMS, `roi_align`, box decode, anchor
generation) lives in the `ReactantServer.DetectionGlue` module and is already part of the package,
so the generated `model.jl` is small and the conversion is config-driven.

This page walks through converting a standard torchvision detector end to end. For the
underlying execution model (the meta gate, committed sub-calls, placement) see
[Meta Models](meta_models.md).

## The shape of the conversion

One source model is emitted as three bundles under the output root:

| Bundle | Contents | Kind |
| --- | --- | --- |
| `<name>_stage1` | Backbone + RPN head | StableHLO |
| `<name>_stage2` | Box head + box predictor over a fixed number of ROIs | StableHLO |
| `<name>` | Chains stage1, then `DetectionGlue`, then stage2, then final NMS | `meta` |

`stage1` takes the preprocessed image and returns 14 dense tensors: the four ROI-pooling feature
maps (`feat_0`-`feat_3`), the five per-level objectness maps (`obj_0`-`obj_4`), and the five
per-level box deltas (`delta_0`-`delta_4`). `stage2` takes ROI-pooled features for a fixed `K`
proposals (`[K, 256, 7, 7]` in torch, `(7, 7, 256, K)` in the Julia column-major wire layout) and
returns `cls_logits` (`[K, num_classes]`) and `bbox_deltas` (`[K, num_classes*4]`). The meta bundle
owns no weights of its own; it is placed as a group with its two stages and routed by the gateway as
a single unit (see the placement section of [Meta Models](meta_models.md)).

Only `<name>` is addressable by clients. The two stages are internal to the meta and never appear in
the gateway's routing table.

Crucially, both stages are plain `nn.Module`s that `torch.export` traces directly: the converter
builds the torchvision model, wraps `backbone` + `rpn.head` and `box_head` + `box_predictor`, and
exports them. There is no `torch.jit` load and no reaching into a scripted graph's frozen internals.

## Running the converter

The converter is `tools/convert_to_stablehlo.jl`, driven by a YAML config. The torchvision detector
is a *handler* (a special-case builder), shipped at
`tools/handlers/torchvision_frcnn_detector.jl`. Reference it from the `handlers:` block of your
config, keyed by a model name:

```yaml
output_root: /docker/reactantserver/models

handlers:
  - file: handlers/torchvision_frcnn_detector.jl
    models: [my_detector]
    options:
      weights: DEFAULT      # DEFAULT = pretrained COCO; "none" = random; or a .pth path
      num_classes: 91       # classes INCLUDING background; set when loading a custom head
      image_size: 640       # canonical square input edge, divisible by 64
      input_dtype: u8       # client image dtype (u8 | f32)
      output_cols: 6        # 5 = [box4, score]; 6 = [box4, score, class]
```

Relative paths (including `file:` and any option key ending in `_dir`/`_path`) resolve against the
config file's directory. With `weights: DEFAULT` the converter builds the pretrained COCO model, so
no source artifact is needed (the runnable demo path); to convert your own trained detector, point
`weights` at a saved `state_dict` and set `num_classes` to match its head. A handler runs after the
torch/torchax/triton imports, so it may freely call `pyexec`/`pyimport`.

Run it from the repository root, instantiating against an environment that has torch, torchvision,
torchax, and `ReactantServerExport`:

```text
julia tools/convert_to_stablehlo.jl <config>.yaml --only my_detector
```

Use `--dry-run` to validate the config and handler load without paying torch startup, and `--force`
to rebuild a bundle that already exists. The run emits `my_detector_stage1`,
`my_detector_stage2`, and the `my_detector` meta bundle.

## The generated meta `model.jl`

The handler bakes the per-model config it reads from the live model (the per-level `cell_anchors`,
the RPN/ROI box-coder weights, the RPN pre-NMS top-k and NMS threshold, and the final
`score`/`nms`/`detections_per_img`) into the meta bundle's `model.jl`, then registers the
orchestration with [`register_meta_model`](@ref). The emitted function, lightly abridged, is:

```julia
const _G = ReactantServer.DetectionGlue

function _run(inputs, call)
    iw = size(inputs[1].data, 1); ih = size(inputs[1].data, 2)

    # Stage 1: backbone + RPN head. 14 dense outputs.
    s1 = call("my_detector_stage1", inputs)
    d  = Dict(t.name => t.data for t in s1)

    # Per-level: generate anchors, decode RPN deltas to boxes, flatten objectness.
    bl = Matrix{Float64}[]; sl = Vector{Float64}[]
    for i in 1:5
        O = d[_OBJ[i]]; D = d[_DEL[i]]
        anc = _G.generate_anchors(size(O, 2), size(O, 1), _STR[i], _CELL[i])
        push!(bl, _G.decode_boxes(_G.deltas_matrix(D), anc, _RPNW))
        push!(sl, _G.objectness_flat(O))
    end

    # Select the top-K proposals (NMS across levels), then ROIAlign the feature maps.
    pb = _G.select_rpn_proposals(bl, sl, ih, iw; pre=_PRE, post=_K, nms_thresh=_RPNNMS)
    Kp = size(pb, 1)
    feats = [_G.feature_chw(d[f]) for f in _FEAT]
    roi = call.scratch((7, 7, 256, _K), Float32); fill!(roi, 0f0)
    lv = [_G.assign_level(@view pb[k, :]) for k in 1:Kp]
    for l in 0:3
        sel = findall(==(l), lv); isempty(sel) && continue
        _G.roi_align_wire!(view(roi, :, :, :, sel), feats[l+1], pb[sel, :], _SCALES[l+1];
                           ratio=2, aligned=false)
    end

    # Stage 2: box head + predictor on the pooled ROIs, then the final per-class NMS.
    s2 = call("my_detector_stage2", [ReactantServer.NamedTensor("ROI_FEATS", roi)])
    d2 = Dict(t.name => t.data for t in s2)
    cls = permutedims(d2["cls_logits"], (2, 1))[1:Kp, :]
    dl  = permutedims(d2["bbox_deltas"], (2, 1))[1:Kp, :]
    bx, sc, cl = _G.fast_rcnn_inference(cls, dl, pb, ih, iw;
        score_thresh=_SCORE, nms_thresh=_NMS, topk=_TOPK, weights=_ROIW,
        bg_first=true, min_size=1e-2)

    out = isempty(sc) ? zeros(Float32, 6, 0) :
        Array{Float32}(permutedims(hcat(bx, sc, Float64.(cl)), (2, 1)))
    return [ReactantServer.NamedTensor("OUTPUT__0", out)]
end

register_meta_model("my_detector"; run = _run)
```

Above `_run`, the handler emits the per-model constants the function references: the FPN strides and
scales (`_STR`, `_SCALES`), the per-level cell anchors (`_CELL`), the RPN and ROI box-coder weights
(`_RPNW`, `_ROIW`), the RPN pre-NMS top-k (`_PRE`) and NMS threshold (`_RPNNMS`), and the final
`score`/`nms`/`detections_per_img` (`_SCORE`, `_NMS`, `_TOPK`). With `output_cols: 5` the output
assembly line instead emits `[box4, score]` rows.

Every data-dependent step is a plain function in `ReactantServer.DetectionGlue`:
`generate_anchors`, `decode_boxes`, `select_rpn_proposals`, `roi_align_wire!`, `assign_level`, and
`fast_rcnn_inference`. A few `DetectionGlue` knobs select the torchvision conventions:
`roi_align_wire!(...; aligned=false)` uses torchvision's ROIAlign offset and malformed-ROI clamp,
and `fast_rcnn_inference(...; bg_first=true, min_size=1e-2)` treats class 0 as background and
drops sub-pixel final boxes the way torchvision's `postprocess_detections` does. The ROI feature
tensor is the large intermediate handed between stages, so it is allocated from the worker's reuse
pool with `call.scratch`; in a fleet that buffer is backed by a shared-memory slot so the sub-call
sends it by reference instead of serializing it (see the `call.scratch` section of
[Meta Models](meta_models.md)).

## Options and assumptions

| Option | Default | Meaning |
| --- | --- | --- |
| `weights` | `DEFAULT` | `DEFAULT` = pretrained COCO; `none` = random; or a `state_dict` `.pth`. |
| `num_classes` | `91` | Classes including background; set this when loading a custom head. |
| `image_size` | `640` | Canonical square input edge, divisible by 64 (FPN p6/pool stride). |
| `input_dtype` | `u8` | Client image dtype: u8 (/255 then normalized) or f32 (assumed in [0,1]). |
| `output_cols` | `6` | Per-detection width: `5` = `[box4, score]`, `6` = `[box4, score, class]`. |
| `input_shapes` | none | Optional `[W, H]` pairs for extra aspect ratios, one weight set. |

Each `input_shapes` edge must be divisible by 64. The meta routes each request to the matching
variant by its input shape; stage2 is shared, because its ROI input is always the fixed 7x7 grid.

Two assumptions are worth calling out:

- **Input is a batched RGB image.** The client sends an NCHW image (`[1, 3, H, W]`) at the compiled
  size; stage1 bakes the ImageNet normalization (and the `/255` for `u8`) so the client sends a
  raw image. The served wrapper letterboxes to one of the compiled `image_size`/`input_shapes`;
  the model is not resized at run time.
- **Class ids follow torchvision.** With `output_cols: 6` the emitted class is the torchvision label
  (`1..num_classes-1`); background (class 0) is dropped.

## Runnable example

`examples/object_detection/` in the repository is a complete, runnable version of this walkthrough:
export a torchvision Faster R-CNN pretrained on COCO into StableHLO bundles, serve them on a single
GPU, send an image, and draw the predicted boxes + COCO labels back onto it with CairoMakie
(`detections.jpg`). The demo model is named `object_detector`, configured by
`examples/object_detection/detector.convert.yaml`:

```yaml
output_root: bundles
report_path: bundles/conversion_report.md

handlers:
  - file: ../../tools/handlers/torchvision_frcnn_detector.jl
    models: [object_detector]
    options:
      weights: DEFAULT     # pretrained COCO weights (downloaded by torchvision on first export)
      num_classes: 91      # COCO classes including background
      image_size: 640      # canonical square input edge (divisible by 64)
      input_dtype: u8      # client sends a uint8 image; stage1 bakes /255 + ImageNet normalize
      output_cols: 6       # per detection: [x1, y1, x2, y2, score, class]
```

The example is split into three single-purpose Julia environments so each loads only what it needs
(and they stop invalidating each other's precompilation): **export** is the only one with PythonCall
+ torch, **server** is the only one with Reactant, and **client** has CairoMakie but no Reactant.
Each environment resolves independently the first time you use it:

```text
for env in export server client; do
  julia --project=examples/object_detection/$env -e 'using Pkg; Pkg.instantiate()'
done
```

Then run the three steps in order (the server stays running; drive it from a second terminal):

```text
# 1. Export the bundles (first time only; writes ./bundles/). Needs network for the COCO weights.
julia --project=examples/object_detection/export \
    examples/object_detection/export/export.jl

# 2. Serve on a single GPU (blocks; Ctrl-C to stop). Add --cpu for a GPU-free smoke test.
CUDA_VISIBLE_DEVICES=0 julia --project=examples/object_detection/server \
    examples/object_detection/server/serve.jl

# 3. In another terminal: send an image and draw the result -> ./detections.jpg
julia --project=examples/object_detection/client \
    examples/object_detection/client/detect.jl
```

Pass your own image to step 3 as the first argument (a local path). The server port defaults to
8080; set `OD_PORT` (and `OD_HOST` for the client) to change it on both step 2 and step 3.

### Export

`export/export.jl` drives the shared converter in-process, so torch imports before Reactant (the
converter's required order), and writes `examples/object_detection/bundles/` with the
`object_detector`, `object_detector_stage1`, and `object_detector_stage2` bundles. It skips
conversion when the bundles already exist (delete the `bundles/` dir to re-export). The pretrained
COCO weights are downloaded by torchvision on the first run, so export needs network. On corporate
networks `export.jl` points Python's TLS at the OS CA bundle (`SSL_CERT_FILE`, defaulting from
`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/`JULIA_SSL_CA_ROOTS_PATH` or
`/etc/ssl/certs/ca-certificates.crt`) so the weight download trusts a MitM proxy's CA. The Python
dependencies are export-only: torch/torchax/jax come from `ReactantServerExport`'s CondaPkg and
`torchvision` from `export/CondaPkg.toml`; CondaPkg resolves and installs them on the first export.

### Serve

`server/serve.jl` serves the `object_detector` meta bundle on `127.0.0.1:$OD_PORT` (default 8080)
and blocks until Ctrl-C. Pass `--cpu` to use `ReactantServer.CPU_BACKEND` for a GPU-free smoke test;
otherwise it uses `ReactantServer.CUDA_BACKEND`.

### Client

`client/detect.jl` connects to the running server with the ReactantServerClient library (no Reactant
and no PythonCall, so it loads fast). It resizes the image to 640x640 (matching the export
`image_size`), sends it as `INPUT__0` through `infer_sync`, reads `OUTPUT__0` as Float32, draws the
boxes and labels with CairoMakie, and writes `examples/object_detection/detections.jpg`. With no
image argument it downloads a default object-rich test photo
(`https://ultralytics.com/images/bus.jpg`). Each output row is `[x1, y1, x2, y2, score, class]`,
boxes in the 640x640 input pixel space, `class` a COCO id mapped to a name through the COCO
`categories` table in `detect.jl`. The model bakes a 0.05 score threshold; the client
additionally only draws detections scoring at least 0.5.

On the first true end-to-end run, watch the client's `Raw output size=...` log line: it confirms the
`OUTPUT__0` orientation (`parse_detections` handles either), and sanity-check that the drawn boxes
land on the right objects.

## See also

- [Meta Models](meta_models.md) for the execution model, gating, deadlines, and placement
- [Bundles & model.jl](bundles.md) for the plain bundle path and the manifest encoding
- [Client Usage](client.md) for the client library the demo uses
- [`register_meta_model`](@ref) in the API reference
