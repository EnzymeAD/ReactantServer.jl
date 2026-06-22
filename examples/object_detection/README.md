# Object detection demo

Serving an object detector with ReactantServer, end to end: export a torchvision Faster R-CNN
(`fasterrcnn_resnet50_fpn`, pretrained on COCO) into StableHLO bundles, serve them on a single GPU,
send an image, and draw the predicted boxes + COCO labels back onto it with CairoMakie.

It is split into three single-purpose Julia environments so each loads only what it needs (and they
stop invalidating each other's precompilation): **export** is the only one with PythonCall + torch,
**server** is the only one with Reactant, and **client** has CairoMakie but no Reactant.

Each environment resolves independently the first time you use it:

```sh
for env in export server client; do
  julia --project=examples/object_detection/$env -e 'using Pkg; Pkg.instantiate()'
done
```

Then run the three steps in order (the server stays running; drive it from a second terminal):

```sh
# 1. Export the bundles (first time only; writes ./bundles/). Needs network for the COCO weights.
julia --project=examples/object_detection/export examples/object_detection/export/export.jl

# 2. Serve on a single GPU (blocks; Ctrl-C to stop). Add --cpu for a GPU-free smoke test.
CUDA_VISIBLE_DEVICES=0 julia --project=examples/object_detection/server examples/object_detection/server/serve.jl

# 3. In another terminal: send an image and draw the result -> ./detections.jpg
julia --project=examples/object_detection/client examples/object_detection/client/detect.jl
```

Pass your own image to step 3 as the first argument (a local path). The server port defaults to 8080;
set `OD_PORT` (and `OD_HOST` for the client) to change it on both step 2 and step 3.

## Notes

- **What it predicts:** COCO's 80 everyday object classes (person, bicycle, car, bus, dog, cat, ...).
  Each output row is `[x1, y1, x2, y2, score, class]`, boxes in the 640×640 input pixel space, `class`
  a COCO id mapped to a name in `client/detect.jl`. The model bakes a 0.05 score threshold; the client
  additionally only draws detections scoring at least 0.5.
- **Python deps (export only):** torch/torchax/jax come from `ReactantServerExport`'s CondaPkg and
  `torchvision` from `export/CondaPkg.toml`; CondaPkg resolves and installs them on the first export
  (needs network).
- **Corporate SSL:** `export.jl` points Python's TLS at the OS CA bundle (`SSL_CERT_FILE`,
  defaulting from `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/`JULIA_SSL_CA_ROOTS_PATH` or
  `/etc/ssl/certs/ca-certificates.crt`) so the torchvision weight download trusts a MitM proxy's CA.
- **First true end-to-end run:** watch the client's `Raw output size=...` line — it confirms the
  `OUTPUT__0` orientation (`parse_detections` handles either) — and sanity-check that the drawn boxes
  land on the right objects.
