# Object detection demo

Serving an object detector with ReactantServer, end to end: export a torchvision Faster R-CNN
(`fasterrcnn_resnet50_fpn`, pretrained on COCO) into one StableHLO bundle, serve it, send an
image, and draw the predicted boxes and COCO labels back onto it with CairoMakie. It runs on the CPU
or a GPU.

The example is three single-purpose Julia environments, each loading only what it needs:
`export/` holds the export tooling with PythonCall and torch, `server/` holds the server, and
`client/` holds the client and CairoMakie, with neither Reactant nor torch. `detector.convert.yaml` configures the conversion.

- [Serve an Object Detector](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/examples/object_detection/) walks through running it.
- [Object Detection](https://enzymead.github.io/ReactantServer.jl/dev/object_detection/) explains how the detector's data-dependent middle
  becomes one static program, and the tracing pitfalls met along the way.
