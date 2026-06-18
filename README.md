# ReactantServer.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzymead.github.io/ReactantServer.jl/dev/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://enzymead.github.io/ReactantServer.jl/stable/)
[![CI](https://img.shields.io/github/checks-status/enzymead/ReactantServer.jl/main?label=CI)](https://github.com/enzymead/ReactantServer.jl/commits/main)

A production inference server for XLA-accelerated models, compiled through Reactant.jl
(StableHLO via XLA today). It speaks the KServe V2 inference API natively over gRPC, so standard
Triton and KServe clients connect unchanged; it scales from a single GPU to many from one
container; and it squeezes the most models out of each GPU by balancing model **memory** against
**compute**. It is Julia-first throughout — custom pre/postprocessing is plain Julia, and every
convention follows Julia's (column-major, batch-last axes).

It targets static-graph workloads — computer vision, scientific computing — where many models
share a GPU and one model executes at a time.

Not sure which setup fits you? [Common Use Cases](docs/src/manual/common_use_cases.md) walks
through the deployment shapes (single GPU, multi-GPU distributed or replicated, multi-node) with
an example configuration for each.

## Highlights

- **XLA-accelerated, Reactant-compiled.** Models are compiled ahead of time into device
  executables through Reactant's PJRT bindings. The runtime is device-agnostic (CUDA today, CPU
  for dev/fallback); supporting more accelerators is a goal, not a redesign. → [Architecture](docs/src/design/architecture.md), [Philosophy](docs/src/design/philosophy.md)
- **Julia-first pre/postprocessing.** A bundle's `model.jl` registers `preprocess`/`postprocess`
  hooks in plain Julia; they run per request, in parallel and overlapped with GPU execution. → [Bundles & model.jl](docs/src/manual/bundles.md)
- **Julia-aligned conventions.** Shapes are column-major with the batch axis last, the way Julia
  and Lux write them; the codec converts to KServe's row-major wire at the boundary, so Triton
  clients are unchanged and you never reason about row-major order. → [Getting Started](docs/src/manual/getting_started.md)
- **Elegant configuration.** One typed YAML node file (with environment-variable overrides)
  describes a machine; manifests declare tensors with an einsum-style named-axis notation. → [Node Configuration](docs/src/manual/node_config.md), [Bundles & model.jl](docs/src/manual/bundles.md)
- **Standard inference protocol.** KServe V2 over gRPC. Tensor data travels inline or through the
  Triton-compatible system-shared-memory extension for zero-copy local clients. → [Client Usage](docs/src/manual/client_usage.md)
- **One container, single or multi-GPU.** A node supervisor runs one worker per visible GPU: a
  single worker serves the public ports directly; two or more get an embedded gateway behind one
  endpoint. The external interface (`:8001` gRPC, `:8002` metrics/health) is identical either way.
  → [Docker Deployment](docs/src/manual/docker.md), [Scaling to Multiple GPUs](docs/src/manual/scaling.md)
- **Balances memory and compute.** Every model's weights stay resident in host RAM and stream
  onto the GPU on demand, evicted LRU under a byte budget — so a card serves far more models than
  fit in VRAM, paying a single host-to-device transfer on a cold call. → [On-demand Weights](docs/src/manual/on_demand_weights.md)
- **Batch coalescing.** Concurrent same-model requests are merged into one execution at a compiled
  batch size, amortizing per-launch overhead and the one-time weight transfer across the batch.
  → [Architecture](docs/src/design/architecture.md)
- **Scheduling modes for single and multi-GPU.** On a worker, `fair` (deficit-weighted,
  cost-aware) or `fifo`; across GPUs, the gateway offers `round_robin` or memory-aware
  `lpt_packing` that concentrates each model's traffic to fill batches. → [Architecture](docs/src/design/architecture.md), [Multi-GPU Gateway](docs/src/manual/multi_gpu_gateway.md)
- **Fast iteration.** In `dynamic` mode the server watches the model repository and hot-loads,
  unloads, and reloads bundles online — weights, MLIR, manifest, and `model.jl` alike — with no
  restart (`static` and `explicit` control modes are also available). → [Node Configuration](docs/src/manual/node_config.md)

## Quick start

The image is built locally (it is not published to a registry), so build it once and then serve a
directory of model bundles from the container (it scales to all visible GPUs):

```
git submodule update --init --recursive   # fetch the vendored lib/ forks the build needs
make image                                 # build reactantserver:latest (or: docker compose build)

docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
```

The build is large and the first server startup is slow, since every model compiles before the
gRPC plane accepts traffic. See [Docker Deployment](docs/src/manual/docker.md) for the
`docker compose` workflow and configuration.

Or from pure Julia:

```julia
using ReactantServerNode
ReactantServerNode.supervise("docker/node.yaml")   # one worker per GPU (+ gateway if >1)
```

Clients speak KServe V2 gRPC to `:8001`; health and metrics are on `:8002`. Walk through exporting
a model, configuring a node, and querying it in [Getting Started](docs/src/manual/getting_started.md).
ReactantServer is designed for a trusted network — read
[Security](docs/src/manual/docker.md#security) before exposing an endpoint.

## Status

The full serving path is implemented end to end: export a bundle, compile it through Reactant/PJRT,
schedule and coalesce requests, and serve over the KServe V2 gRPC control plane. The cost-aware
scheduler, the on-demand GPU weight cache, dynamic model lifecycle, and the single- and multi-GPU
deployment paths all work today on CUDA (with CPU for development and fallback); broader accelerator
support is intended to follow. Deferred to later milestones: dynamic-batch export with server-side
`stablehlo-refine` specialization, the compiled-executable disk cache, multi-model orchestrators,
and full StableHLO-signature validation of manifests. See
[Architecture](docs/src/design/architecture.md) for the full picture.

## Repository layout

A Julia 1.11+ workspace of five packages under `packages/`, split so that talking to a server never
pulls in the heavy Reactant/XLA stack:

- **`ReactantServerCore`** — shared, Reactant-free substrate: dtypes, the KServe V2 protobuf
  messages, boundary types, the manifest parser, node config, the wire codec, the shared-memory
  registry, and the staging `BufferPool`.
- **`ReactantServer`** — the inference worker (registry, runtime, scheduler, KServe V2 gRPC server);
  the **only** package that depends on Reactant. Exports `serve`, `stop!`, `register_model`.
- **`ReactantServerGateway`** — the multi-GPU KServe gRPC reverse proxy (`serve_gateway`). No Reactant.
- **`ReactantServerClient`** — a Reactant-free inference client (`KServeModel`, `infer_sync`,
  `infer_async`, `InferInput`, `InferOutput`).
- **`ReactantServerNode`** — the single-container node supervisor (`supervise`): detects GPUs, runs
  one worker per device, embeds the gateway when there is more than one, and multiplexes their logs.

Offline model export lives in `packages/ReactantServerExport` (a Reactant tracing frontend plus a
PythonCall-triggered PyTorch extension); it is deliberately **not** a workspace member, so its
Lux/PythonCall weakdeps stay out of the server images. The vendored forks/unregistered deps
(`Reactant`, `gRPCServer`, `gRPCClient`, `HTTP`) are git submodules under `lib/`.

## Documentation

- [Getting Started](docs/src/manual/getting_started.md) — export a model, configure a node, serve and query it.
- [Common Use Cases](docs/src/manual/common_use_cases.md) — choosing a deployment shape, with an example configuration for each.
- [Scaling to Multiple GPUs](docs/src/manual/scaling.md) — add GPUs and how the supervisor decides what to run.
- [Client Usage](docs/src/manual/client_usage.md) — the Julia client, batched inference, and IO validation.
- [Node Configuration](docs/src/manual/node_config.md) — the full node-file surface and environment overrides.
- [Bundles & model.jl](docs/src/manual/bundles.md) — the bundle format, manifest shapes, and pre/postprocessing.
- [On-demand Weights](docs/src/manual/on_demand_weights.md) — serving more models than fit in GPU memory.
- [Multi-GPU Gateway](docs/src/manual/multi_gpu_gateway.md) — routing, scheduling modes, and operations.
- [Docker Deployment](docs/src/manual/docker.md) — the image, roles, health, metrics, and [Security](docs/src/manual/docker.md#security).
- Design: [Philosophy](docs/src/design/philosophy.md) (mission, audience, non-goals) and [Architecture](docs/src/design/architecture.md) (how it fits together, the scheduler, the compiler advantage).
- [`docker/README.md`](docker/README.md) — the container deployment reference.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — running the tests and regenerating the protobuf bindings.
