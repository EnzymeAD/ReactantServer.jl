```@meta
CurrentModule = ReactantServer
```

# ReactantServer.jl

A production inference server that serves models compiled through Reactant.jl — StableHLO via
XLA today — from a single Julia process, built on Reactant's PJRT bindings. It targets static-graph workloads (computer vision,
scientific computing) where many models share one GPU and only one model executes at a time.
To serve more models than fit in GPU memory at once, it keeps every model's weights resident
in host RAM and transfers them onto the GPU on demand, evicting cold models under a memory
budget. It speaks the KServe V2 inference API natively over gRPC, so standard Triton and KServe
clients connect to it directly.

## What works today

- StableHLO bundle loading, manifest parsing and validation, and a typed YAML configuration
  with environment-variable overrides.
- The Reactant/PJRT runtime: deserialize a portable artifact, compile with weights bound as
  explicit arguments, execute, and read results back. A single shared memory pool backs all
  models.
- On-demand GPU weight loading with a host-RAM weight cache (see
  [On-demand Weights](manual/on_demand_weights.md)), which decouples the number of servable
  models from GPU memory capacity.
- A deficit-weighted, cost-aware, coalescing [`Scheduler`](@ref) that runs one
  GPU execution at a time and batches same-model requests at compiled sizes.
- The KServe V2 control plane over gRPC: liveness/readiness, model and server metadata,
  inference, a `RepositoryIndex`, and the Triton-compatible system shared-memory data plane.
- Model lifecycle control: the default `dynamic` mode watches the model repository and
  loads/unloads/reloads bundles online (weights, MLIR, and `model.jl` changes alike); `static`
  fixes the startup set; `explicit` cedes lifecycle and residency to an external control plane
  over the worker control RPCs.
- Multi-GPU scheduling in the gateway: `round_robin` or `lpt_packing`, which places each model on a
  fixed, operator-configured number of GPUs and routes its requests to fill one replica's batch
  before the next, preserving batch coalescing (see
  [Multi-GPU Gateway](manual/multi_gpu_gateway.md)).
- Multiple compiled batch sizes per model and custom per-model pre/post-processing via a
  bundle's `model.jl` (see [Bundles & model.jl](manual/bundles.md)).
- Meta models: `kind: meta` bundles whose `model.jl` chains several models with data-dependent
  Julia between stages, running off the dispatch loop and re-entering the scheduler for each
  sub-call (see [Meta Models](manual/meta_models.md)). The
  [Object Detection](manual/object_detection.md) guide is a worked end-to-end example, a
  torchvision Faster R-CNN split into two StableHLO stages chained by Julia detection glue.

ReactantServer.jl is a Julia workspace of five packages: **ReactantServerCore** (the shared,
Reactant-free substrate), **ReactantServer** (the worker, the only package that loads Reactant),
**ReactantServerGateway** (the multi-GPU reverse proxy), **ReactantServerClient** (a
Reactant-free inference client), and **ReactantServerNode** (the node
supervisor), plus the non-member **ReactantServerExport** (offline bundle export). See
[Architecture](design/architecture.md) for the split.

## Where to go next

- New here? Start with [Getting Started](manual/getting_started.md).
- Choosing a deployment shape (single GPU, multi-GPU, multi-node) with example configs:
  [Common Use Cases](manual/common_use_cases.md).
- Calling a server from your code: [Client Usage](manual/client_usage.md).
- Configuring a deployment: [Node Configuration](manual/node_config.md).
- Scaling to multiple GPUs: [Scaling to Multiple GPUs](manual/scaling.md).
- Packaging a model: [Bundles & model.jl](manual/bundles.md).
- Chaining models with data-dependent Julia: [Meta Models](manual/meta_models.md), with a worked
  [Object Detection](manual/object_detection.md) example.
- Serving more models than fit on the GPU: [On-demand Weights](manual/on_demand_weights.md).
- Scaling across GPUs: [Multi-GPU Gateway](manual/multi_gpu_gateway.md) and
  [Deployment](manual/deployment.md).
- The why and the how: [Philosophy](design/philosophy.md) and
  [Architecture](design/architecture.md).
- The full [API Reference](api/server.md).

## Quick start

```julia
using ReactantServer
ReactantServer.serve("config/node.yaml")                    # single worker: name optional
ReactantServer.serve("config/node.yaml"; worker="worker0")  # multi-worker: name the worker
```

All tests run on CPU and need no GPU. Each package is tested in its own environment:

```
julia --project=packages/ReactantServerCore   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer        -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient  -e 'using Pkg; Pkg.test()'
```
