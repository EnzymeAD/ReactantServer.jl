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
      link: /tutorials/
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
    link: /tutorials/export/
  - icon: 🧩
    title: Julia-first
    details: A bundle's model.jl registers pre and postprocessing in plain Julia, and every convention follows Julia's, column-major with the batch axis last.
    link: /bundles/
  - icon: 💾
    title: On-demand weights
    details: Weights stay in host RAM and stream to the GPU under a byte budget with cost-aware eviction, so a card serves more models than fit in VRAM.
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

## Why ReactantServer?

The server is built for workloads where many models share a GPU, such as computer vision and
scientific computing. Every model is compiled ahead of time through Reactant and XLA into an
optimized executable for the target device, and compiled models are fast. The runtime is
device-agnostic: CUDA today, with CPU for development.

The design balances two resources that pull against each other. Compute is managed by a
cost-aware [scheduler](scheduling.md) that coalesces concurrent requests for the same model into
one execution. Memory is managed by the [on-demand weights](on_demand_weights.md) cache, which
keeps weights in host RAM and moves them onto the GPU within a byte budget. Scheduling policy,
pinned models, and multi-GPU placement can all be changed at runtime, without a restart.

The server is a set of packages, split so that only the worker and the export tooling load Reactant
and XLA; the client, the gateway, and the supervisor install without them.
[Architecture](design/architecture.md) describes each package.

## Get started

The [Tutorials](tutorials/index.md) follow a model from [export](tutorials/export.md) to
[inference](tutorials/inference.md), with two ways to serve it in between:

- **From Julia.** Every package is in the General registry, so a Julia environment is all you
  need, on any machine, with or without a GPU. Start with [Serve a Model](tutorials/julia/serve.md).
- **With the container image.** The supported deployment: a host needs only the NVIDIA driver and
  a container runtime. Start with [Serve on One GPU](tutorials/container/single_gpu.md).

To see real models, [Serve an Object Detector](tutorials/examples/object_detection.md) and
[Serve Transformer Text Models](tutorials/examples/transformers.md) take PyTorch models through
export, serving, and inference.

The server is designed for a trusted network, so read [Security](security.md) before exposing an
endpoint.

## Reference

- [Bundles](bundles.md): the bundle format, the manifest's named-axis notation, and the export
  frontends.
- [Node Configuration](node_config.md): the one typed YAML file that describes a machine.
- [Scheduling](scheduling.md): the cost-aware worker scheduler and batch coalescing.
- [On-demand Weights](on_demand_weights.md): host-RAM weights, the byte budget, and fragmentation-aware admission.
- [Multi-GPU Gateway](gateway.md): the reverse proxy and its scheduling modes.
- [Client Usage](client.md): the Reactant-free client and its shared-memory transport.
- [Meta Models](meta_models.md) (experimental): chaining models with data-dependent Julia between
  stages.
- [Deployment](deployment.md): how a node behaves, the container image and its settings, and the
  multi-node contract.
- The [API](api.md): every documented name, collected automatically.
