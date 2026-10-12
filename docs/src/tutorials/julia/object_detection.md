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
blocks until Ctrl-C. `--cpu` selects `ReactantServer.CPU_BACKEND`; without it the server uses
`ReactantServer.CUDA_BACKEND`. The server compiles the detector before it accepts requests: on the
CPU of a large workstation that took about half a minute, and longer on a laptop.

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
