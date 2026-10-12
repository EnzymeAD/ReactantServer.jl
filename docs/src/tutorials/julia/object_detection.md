# Serve an Object Detector

This tutorial runs a real model end to end: it exports a torchvision Faster R-CNN, pretrained on
COCO, into one ReactantServer bundle, serves it, sends it a photo, and draws the detected objects
back onto the image. It runs on the CPU, so it needs no GPU, and the same steps serve on a GPU by
dropping one flag. How the detector's data-dependent middle becomes one static program is covered
in the [Object Detection](../../object_detection.md) reference.

You need a clone of the repository (the example's scripts live in `examples/object_detection/`)
and network access for the first export, which downloads the pretrained weights and the Python
packages the PyTorch export uses.

The demo model is named `object_detector`, configured by
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
(and they stop invalidating each other's precompilation): **export** holds the export tooling with
PythonCall and torch, **server** holds `ReactantServer`, and **client** holds
`ReactantServerClient` and CairoMakie, with neither Reactant nor torch.
Each environment resolves independently the first time you use it:

```sh
for env in export server client; do
  julia --project=examples/object_detection/$env -e 'using Pkg; Pkg.instantiate()'
done
```

Then run the three steps in order (the server stays running; drive it from a second terminal):

```sh
# 1. Export the bundle (first time only; writes ./bundles/). Needs network for the COCO weights.
julia --project=examples/object_detection/export \
    examples/object_detection/export/export.jl

# 2. Serve it on the CPU (blocks; Ctrl-C to stop). On a GPU host, drop --cpu and pick a device
#    with CUDA_VISIBLE_DEVICES=0.
julia --project=examples/object_detection/server \
    examples/object_detection/server/serve.jl --cpu

# 3. In another terminal: send an image and draw the result -> ./detections.jpg
julia --project=examples/object_detection/client \
    examples/object_detection/client/detect.jl
```

Pass your own image to step 3 as the first argument (a local path). The server port defaults to
8080; set `OD_PORT` (and `OD_HOST` for the client) to change it on both step 2 and step 3.

### Export

The export writes the `object_detector` bundle to `examples/object_detection/bundles/`, and skips
the work when the bundle already exists; delete `bundles/` to export again. The first run
downloads the pretrained COCO weights and, through CondaPkg, the Python packages the export uses.
Behind a proxy that intercepts TLS, set `SSL_CERT_FILE` to a CA bundle that trusts it; by default
the script uses the system bundle.

### Serve

The server compiles the detector before it accepts requests: about half a minute on the CPU of a
large workstation, and longer on a laptop.

### Client

The client resizes the image to 640x640, sends it, and writes
`examples/object_detection/detections.jpg` with the boxes and labels drawn. Without an image
argument it downloads a test photo (`https://ultralytics.com/images/bus.jpg`).

Each output row is `[x1, y1, x2, y2, score, class]`, with boxes in the 640x640 input's pixels and
`class` a COCO category id. The model drops detections scoring below 0.05, and the client draws
those scoring 0.5 or more.
