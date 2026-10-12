# Tutorials

The tutorials follow a model from export to inference: export it into a bundle, serve the bundle
from a Julia session or with the container image, then send it requests from a client. Each page
stands on its own, so you can start with any of them.

## Export

- [Export a Model](export.md): turn a Lux model into a bundle, the directory of compiled program,
  weights, and manifest that a server loads.

## Serve from Julia

Add the packages to a Julia environment and run the server from a Julia session, on any machine,
with or without a GPU. This is the natural way to develop against the server. These pages run on
the CPU; a GPU needs one line of configuration changed.

- [Serve a Model](julia/serve.md): one worker serving a model repository.
- [Run a Node](julia/node.md): the node supervisor, running one worker per device behind an
  embedded gateway.
- [Front Workers with a Gateway](julia/gateway.md): a standalone gateway over workers you start
  yourself, placing models with `lpt_packing`.

## Serve with the Container

Run the container image on a GPU host. It is the supported deployment: the host needs only an
NVIDIA driver and a container runtime, and the image carries Julia, every package precompiled,
and the CUDA libraries.

- [Serve on One GPU](container/single_gpu.md): run the image over a model repository and query it.
- [Scale to Many GPUs](container/multi_gpu.md): one worker per GPU behind the embedded gateway,
  with models replicated on every GPU or distributed across them.

## Inference

- [Run Inference](inference.md): send a dataset through `ReactantServerClient`, with concurrent
  batched requests and shared memory when the client shares a host with the server. The client
  works the same against either way of serving.

## End-to-End Examples

Real models exported from PyTorch, served, and queried, using the scripts in the repository's
`examples/` directory. Each runs on the CPU, and serves on a GPU by dropping one flag.

- [Serve an Object Detector](examples/object_detection.md): a pretrained Faster R-CNN that finds
  and labels the objects in a photo.
- [Serve Transformer Text Models](examples/transformers.md): four HuggingFace BERT-family models,
  queried with raw text, for sparse and dense embeddings, reranking, and sentiment.

Export, Run Inference, and the Julia serving pages run in CI on Linux and macOS
exactly as written. The end-to-end examples run in CI on request, since they download PyTorch and
pretrained weights, and the container pages need a GPU.
