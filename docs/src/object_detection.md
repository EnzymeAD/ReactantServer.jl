# Object Detection (GeneralizedRCNN)

A two-stage object detector (an FPN backbone + RPN + RoIHeads GeneralizedRCNN, such as a
torchvision `fasterrcnn_resnet50_fpn`) is mostly dense tensor math with a data-dependent middle:
selecting RPN proposals, running NMS, pooling ROI features with ROIAlign, decoding boxes, and the
final per-class NMS. `torch.export` cannot capture that middle, because the host implementations
use `findall`, variable-length lists, and loops whose trip count depends on the data.

For inference none of that needs to be variable-shape. The number of proposals handed to the box
head is a fixed `K` (1000 in torchvision), every other intermediate is bounded by a configuration
value, and only the *values* depend on the image. `ReactantServerExport` therefore traces the
whole detector (stage1, the glue, stage2, and the final NMS) into **one** static StableHLO
program and serves it as a plain bundle. The glue lives in `ReactantServerExport.Detection`; the
high-level entry point is `export_two_stage_detector`.

## Why the glue has a fixed-shape form

Each data-dependent step is rewritten over fixed-size buffers with a validity mask:

- **Proposal selection.** Per FPN level, the top `pre_nms_topk` anchors by objectness come from a
  stable sort; a level with fewer anchors is padded to `pre_nms_topk` with invalid entries. Clipping
  and the empty-box filter update the mask instead of dropping rows.
- **NMS.** Greedy NMS in descending-score order is the unique solution of the recurrence
  `keep[i] = valid[i] && no kept j before i overlaps i above the threshold`. That recurrence is a
  DAG in sort order, so iterating it from `keep = valid` reaches exactly the greedy result. Each
  iteration is one batched `(M x M)` mat-vec, one column per NMS group (an FPN level or a class),
  inside a traced `while` loop that typically settles in a handful of iterations.
- **Top-k with ties.** Sorts are stable with an index operand (`Ops.sort(...; is_stable=true)`),
  so equal scores keep their input order, as the host implementation's stable `sortperm` does.
  `Base.sortperm` on a traced array is not stable.
- **ROIAlign.** Each box picks its FPN level from its area, and all levels are concatenated so a
  single gather serves every box. With adaptive sampling (`sampling_ratio = 0`) the per-box
  sample grid is `ceil(roi / pooled)` per axis, which is data-dependent; the program loops over the
  sample index up to the largest grid in the batch and masks out samples a box does not have.
- **Anchors.** The anchor grid is built in the program from the cell anchors and two iotas, so
  a multi-shape bundle carries no per-shape anchor constants.

The program returns a fixed-size, zero-padded `DETECTIONS` buffer (`detections_per_img` rows) plus
`NUM_DETECTIONS`, and the bundle's `model.jl` trims the buffer to the detections found (see
[Variable-length results](bundles.md#Variable-length-results)).

Weights stay program **arguments**: each stage's weights are written to `weights.safetensors`
prefixed `stage1.` and `stage2.`, and the export refuses a program whose entry arity does not equal
inputs plus weights. The programs are small next to their weights (the served detectors' programs
are 104 KB to 9 MB against 159 to 402 MB of weights), which is also how a constant-folding
regression would show up.

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
no source artifact is needed; to convert your own trained detector, point `weights` at a saved
`state_dict` and set `num_classes` to match its head.

Run it from the repository root, in an environment that has torch, torchvision, torchax, and
`ReactantServerExport`:

```text
julia tools/convert_to_stablehlo.jl <config>.yaml --only my_detector
```

Use `--dry-run` to validate the config and handler load without paying torch startup, and `--force`
to rebuild a bundle that already exists. The run emits one bundle, `my_detector`.

## What the handler does

The handler builds the torchvision model and wraps `backbone` + `rpn.head` (stage1) and
`box_head` + `box_predictor` (stage2) as two small `nn.Module`s that `torch.export` traces directly.
It exports each into a scratch directory, reads the programs and weights back with
`read_stage_bundle`, and fuses them:

```julia
using ReactantServerExport

cfg = DetectorConfig(;
    strides = [4, 8, 16, 32, 64], scales = [0.25, 0.125, 0.0625, 0.03125],
    cell_anchors = cells,                     # per level, [A, 4] xyxy, read off the live model
    rpn_weights, roi_weights,                 # box-coder weights
    pre_nms_topk = 1000, post_nms_topk = 1000, rpn_nms_thresh = 0.7,
    score_thresh = 0.05, nms_thresh = 0.5, detections_per_img = 100,
    # torchvision conventions (the defaults are detectron2's)
    aligned = false, sampling_ratio = 2, bg_first = true, min_size = 1e-2,
)
S1 = read_stage_bundle(stage1_dir); S2 = read_stage_bundle(stage2_dir)
export_two_stage_detector("models/my_detector"; name = "my_detector",
    input = IOSpec("INPUT__0", UInt8, [640, 640, 3, 1]; letters = ['w', 'h', 'c', 'a']),
    stage1 = S1.texts, stage1_weights = S1.weights,
    stage2 = S2.texts[Int[]], stage2_weights = S2.weights, cfg,
    output_columns = 6)
```

Stage1 takes the image and returns 14 dense tensors: the four ROI-pooling feature maps, the five
per-level objectness maps, and the five per-level box deltas. Stage2 takes ROI-pooled features for
`K` proposals (`[K, 256, 7, 7]` in torch) and returns `cls_logits` and `bbox_deltas`. The
`DetectorConfig` fields that differ between frameworks:

| Field | detectron2 (default) | torchvision |
| --- | --- | --- |
| `aligned` | `true` (half-pixel ROIAlign offset) | `false` (and malformed ROIs clamp to 1 px) |
| `sampling_ratio` | `0` (adaptive) | `2` |
| `bg_first` | `false` (background is the last class column) | `true` (background is column 0) |
| `min_size` | `0.0` | `1e-2` (drop sub-pixel final boxes) |

## Options and assumptions

| Option | Default | Meaning |
| --- | --- | --- |
| `weights` | `DEFAULT` | `DEFAULT` = pretrained COCO; `none` = random; or a `state_dict` `.pth`. |
| `num_classes` | `91` | Classes including background; set this when loading a custom head. |
| `image_size` | `640` | Canonical square input edge, divisible by 64 (FPN p6/pool stride). |
| `input_dtype` | `u8` | Client image dtype: u8 (/255 then normalized) or f32 (assumed in [0,1]). |
| `output_cols` | `6` | Per-detection width: `5` = `[box4, score]`, `6` = `[box4, score, class]`. |
| `input_shapes` | none | Optional `[W, H]` pairs for extra aspect ratios, one weight set. |

Each `input_shapes` edge must be divisible by 64. Every variant is its own program in the same
bundle, sharing one weight set, and the server routes each request to the variant matching its
input shape. Boxes are clipped to the input's own width and height, which also covers letterboxed
inputs.

Two assumptions are worth calling out:

- **Input is one RGB image.** The client sends an NCHW image (`[1, 3, H, W]`, Julia `(W, H, 3, 1)`)
  at a compiled size; stage1 bakes the ImageNet normalization (and the `/255` for `u8`), so the
  client sends a raw image, resized or letterboxed to a compiled size. The model is not resized at
  run time.
- **Class ids follow torchvision.** With `output_cols: 6` the emitted class is the torchvision label
  (`1..num_classes-1`); background (class 0) is dropped.

## Validating a conversion

Compare against a reference as **detection sets**, not as ordered lists. When two overlapping
detections score within about `1e-6` of each other (a saturated softmax, for example 0.9999964 vs
0.9999958), a last-digit difference in the ROI features decides which one NMS keeps. The traced
ROIAlign accumulates in Float32, so its features can differ from a Float64 host implementation by
a few `1e-6`, which is far smaller than the difference between a GPU run (TF32) and any CPU run.
Inlining stage1 into the larger program also lets XLA compile it slightly differently, which can
reorder near-tied proposals without changing the final detections. A good acceptance check: every
reference detection has a served one with the same class, a box within half a pixel, and a score
within `1e-4`, and the counts match; report any unmatched detection together with its score gap to
its nearest neighbour.

The `ReactantServerExport` test suite checks the traced glue this way against a host
implementation, for both convention sets, on synthetic stage outputs.

## Runnable example

`examples/object_detection/` in the repository is a complete, runnable version of this walkthrough:
export a torchvision Faster R-CNN pretrained on COCO into one StableHLO bundle, serve it on a single
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
# 1. Export the bundle (first time only; writes ./bundles/). Needs network for the COCO weights.
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
converter's required order), and writes the `object_detector` bundle under
`examples/object_detection/bundles/`. It skips conversion when the bundle already exists (delete the
`bundles/` dir to re-export). The pretrained COCO weights are downloaded by torchvision on the first
run, so export needs network. On corporate networks `export.jl` points Python's TLS at the OS CA
bundle (`SSL_CERT_FILE`, defaulting from `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/
`JULIA_SSL_CA_ROOTS_PATH` or `/etc/ssl/certs/ca-certificates.crt`) so the weight download trusts a
MitM proxy's CA. The Python dependencies are export-only: torch/torchax/jax come from
`ReactantServerExport`'s CondaPkg and `torchvision` from `export/CondaPkg.toml`; CondaPkg resolves
and installs them on the first export.

### Serve

`server/serve.jl` serves the `object_detector` bundle on `127.0.0.1:$OD_PORT` (default 8080) and
blocks until Ctrl-C. Pass `--cpu` to use `ReactantServer.CPU_BACKEND` for a GPU-free smoke test;
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

## Reactant tracing pitfalls

Each of these cost an iteration while building the traced glue, and applies to any hand-written
traced export:

- **A returned `reshape` wrapper is emitted with its parent's shape.** `reshape(x, 2, 4, 6, 1)` as a
  function result produced an `(8, 6)` program output, and the server trusts the executable's shape,
  not the manifest. Materialize outputs with `Reactant.ReactantCore.materialize_traced_array`.
- **`@trace while` bodies.** Loop-carried values must be distinct arrays (`copy` before the loop),
  and captured values must be materialized traced arrays, not `reshape` wrappers. Referencing a type
  parameter such as `T(0.5)` inside the body breaks the macro; hoist constants out of the loop.
- **`map(1:n) do ... end` inside a trace failed;** plain `for` loops work.
- **Matrix products.** A traced `A * B` fell back to generic LinearAlgebra in one context; use
  `Ops.dot_general`. Number constructors on traced scalars (`Float32(x)`, `Int64(x)`, `sign(x)`) are
  not traceable; use `Ops.convert(TracedRArray{Int64, N}, x)` and `ifelse`.
- **Sorting.** `Base.sortperm` on traced arrays is not stable; use `Ops.sort(...; is_stable=true)`
  with an index operand where tie order matters.
- **Gathers.** `A[:, idx]` needs `idx` to be a materialized traced `Int64` vector.
- **Per-shape constants add up.** Baking anchor grids as Float64 constants added about 200 MB to a
  15-variant bundle; build such grids in the program from iotas instead.

## See also

- [Bundles & model.jl](bundles.md) for the plain bundle path, the manifest encoding, and the
  fixed-size buffer + trim pattern
- [Client Usage](client.md) for the client library the demo uses
- `export_two_stage_detector`, `DetectorConfig`, and `read_stage_bundle` in the
  [API reference](api.md)
