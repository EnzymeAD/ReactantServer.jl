```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: ReactantServer.jl
  text: Production inference for Reactant-compiled models
  tagline: KServe V2 over gRPC from one GPU to many, with compiled XLA models, Julia-first pre and postprocessing, and the most models per card.
  actions:
    - theme: brand
      text: Tutorials
      link: /tutorials/native/export/
    - theme: alt
      text: API Reference
      link: /api/
    - theme: alt
      text: View on GitHub
      link: https://github.com/EnzymeAD/ReactantServer.jl
  image:
    src: /logo.svg
    alt: ReactantServer.jl

features:
  - icon: ⚡
    title: KServe V2, natively
    details: Speaks the KServe V2 inference API over gRPC, so standard Triton and KServe clients connect unchanged.
    link: /client/
  - icon: 🚀
    title: XLA under the hood
    details: Models compile ahead of time through Reactant and XLA into device executables; the runtime is device agnostic, CUDA today with CPU for development.
    link: /tutorials/native/export/
  - icon: 🧩
    title: Julia-first
    details: A bundle's model.jl registers pre and postprocessing in plain Julia, and every convention follows Julia's, column-major with the batch axis last.
    link: /bundles/
  - icon: 💾
    title: On-demand weights
    details: Weights stay in host RAM and stream to the GPU under an LRU byte budget, so a card serves more models than fit in VRAM.
    link: /on_demand_weights/
  - icon: 🔀
    title: A coalescing scheduler
    details: A deficit-weighted, cost-aware scheduler merges same-model requests into one execution at a compiled batch size.
    link: /scheduling/
  - icon: 🔁
    title: Hot reload
    details: In dynamic mode the server watches the model repository and reloads bundles online, with no restart.
    link: /bundles/
---
```

## What it is

ReactantServer.jl is a production inference server for XLA-accelerated models, compiled through
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (StableHLO via XLA today). It speaks the
KServe V2 inference API natively over gRPC, so standard Triton and KServe clients connect
unchanged; it scales from a single GPU to many from one container; and it balances model **memory**
against **compute** to squeeze the most models out of each GPU. It is Julia-first throughout:
custom pre and postprocessing is plain Julia in a bundle's `model.jl`, and every convention
follows Julia's, column-major with the batch axis last.

The system is a workspace of packages, split so that talking to a server never pulls in the heavy
Reactant/XLA stack: `ReactantServerCore` is the shared, Reactant-free substrate (dtypes,
protobufs, the manifest parser, node config, the codec, shared memory, the staging `BufferPool`);
`ReactantServer` is the worker, the only package that depends on Reactant; `ReactantServerGateway`
is the multi-GPU reverse proxy; `ReactantServerClient` is a Reactant-free client;
`ReactantServerNode` is the one-container supervisor. Offline export lives in
`ReactantServerExport`. Every package is on the [API](api.md) page.

## Why ReactantServer?

The server is built for compiled models, such as computer vision and scientific computing
workloads, where many models share a GPU and one model executes at a time. Local serving of large
language models, by splitting a model into compiled stages and streaming their weights from
storage, is a goal of the project. That shape rewards a server that is compiled rather
than interpreted: models are compiled ahead of time into device executables through Reactant's
PJRT bindings, so inference is a single batched kernel launch rather than an interpreter loop, and
the runtime is device-agnostic, CUDA today with CPU for development and fallback.

The design balances two resources that pull against each other. Compute is managed by a
deficit-weighted, cost-aware [scheduler](scheduling.md) that coalesces concurrent same-model
requests into one execution at a compiled batch size. Memory is managed by the
[on-demand weights](on_demand_weights.md) cache, which keeps weights in host RAM and streams them
onto the GPU under an LRU byte budget. Both are retunable at runtime, on a worker or across GPUs,
through a gRPC control plane, without a restart.

## Get started

There are two ways in, and two tracks of tutorials to match:

- **From Julia.** Every package is in the General registry, so a Julia environment is all you
  need: add the packages, export a model, serve it, and query it, on any machine, with or without a
  GPU. Start with [Export a Model](tutorials/native/export.md).
- **The container image.** The supported deployment: a host needs only the NVIDIA driver and a
  container runtime, and the image carries Julia, every package (precompiled), and the CUDA
  libraries. Start with [Serve on One GPU](tutorials/container_single_gpu.md).

Clients speak KServe V2 gRPC to `:8001`; health and metrics are on `:8002`. The first server
startup is slow, because every model compiles before the gRPC plane accepts traffic. The server is
designed for a trusted network, so read [Security](security.md) before exposing an endpoint.

## Start here

- Tutorials, native Julia: [Export a Model](tutorials/native/export.md),
  [Serve a Model](tutorials/native/serve.md), [Query a Server](tutorials/native/client.md),
  [Run a Node](tutorials/native/node.md), [Front Workers with a Gateway](tutorials/native/gateway.md),
  and the worked examples [Serve an Object Detector](tutorials/object_detection.md) and
  [Serve Transformer Text Models](tutorials/transformers.md).
- Tutorials, container: [Serve on One GPU](tutorials/container_single_gpu.md) and
  [Scale to Many GPUs](tutorials/container_multi_gpu.md).
- [Bundles](bundles.md): the bundle format, the manifest's named-axis notation, and the export
  frontends.
- [Node Configuration](node_config.md): the one typed YAML file that describes a machine.
- [Scheduling](scheduling.md): the cost-aware worker scheduler and batch coalescing.
- [On-demand Weights](on_demand_weights.md): host-RAM weights and the LRU byte budget.
- [Multi-GPU Gateway](gateway.md): the reverse proxy and its scheduling modes.
- [Client Usage](client.md): the Reactant-free client and its shared-memory transport.
- [Meta Models](meta_models.md) (experimental): chaining models with data-dependent Julia between
  stages.
- [Deployment](deployment.md): how a node behaves, the container image and its settings, and the
  multi-node contract.
- The [API](api.md): every documented name, collected automatically.
