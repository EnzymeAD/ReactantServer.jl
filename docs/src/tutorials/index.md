# Tutorials

The Basics show how to export a model and how to run inference against a server. Both work the
same however the server is run. The other two sections show how to run it: from a Julia session,
or with the container image.

Each page stands on its own, so you can start with any of them.

## Basics

- [Export a Model](basics/export.md): turn a Lux model into a bundle, the directory of compiled
  program, weights, and manifest that every server loads.
- [Run Inference](basics/inference.md): send a dataset through `ReactantServerClient`, with
  concurrent batched requests and shared memory when the client shares a host with the server.

## Serve from Julia

Add the packages to a Julia environment and run the server from a Julia session, on any machine,
with or without a GPU. This is the quickest way to try the server and the natural way to develop
against it. These pages run on the CPU; a GPU needs one line of configuration changed.

- [Serve a Model](julia/serve.md): one worker serving a model repository.
- [Run a Node](julia/node.md): the node supervisor, running one worker per device behind an
  embedded gateway.
- [Front Workers with a Gateway](julia/gateway.md): a standalone gateway over workers you start
  yourself, placing models with `lpt_packing`.
- [Serve an Object Detector](julia/object_detection.md) and
  [Serve Transformer Text Models](julia/transformers.md): real models end to end, exported from
  PyTorch.

## Serve with the Container

Run the container image on a GPU host. It is the supported deployment: the host needs only an
NVIDIA driver and a container runtime, and the image carries Julia, every package precompiled,
and the CUDA libraries.

- [Serve on One GPU](container/single_gpu.md): run the image over a model repository and query it.
- [Scale to Many GPUs](container/multi_gpu.md): one worker per GPU behind the embedded gateway,
  with models replicated on every GPU or distributed across them.

The basics and the first three Julia serving pages run in CI on Linux and macOS exactly as
written. The two worked examples run in CI on request, since they download PyTorch and pretrained
weights, and the container pages need a GPU.
