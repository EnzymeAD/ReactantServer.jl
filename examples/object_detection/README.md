# Object detection demo

Serving an object detector with ReactantServer, end to end: export a torchvision Faster R-CNN
(`fasterrcnn_resnet50_fpn`, pretrained on COCO) into one StableHLO bundle, serve it, send an
image, and draw the predicted boxes and COCO labels back onto it with CairoMakie. It runs on the CPU
or a GPU.

The example is three single-purpose Julia environments, each loading only what it needs:
`export/` is the only one with PythonCall and torch, `server/` the only one with Reactant, and
`client/` has CairoMakie but no Reactant. `detector.convert.yaml` configures the conversion.

- [Serve an Object Detector](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/julia/object_detection/) walks through running it.
- [Object Detection](https://enzymead.github.io/ReactantServer.jl/dev/object_detection/) explains how the detector's data-dependent middle
  becomes one static program, and the tracing pitfalls met along the way.
