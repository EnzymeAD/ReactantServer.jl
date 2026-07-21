# ReactantServer.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzymead.github.io/ReactantServer.jl/dev/)
[![CI](https://img.shields.io/github/checks-status/enzymead/ReactantServer.jl/main?label=CI)](https://github.com/enzymead/ReactantServer.jl/commits/main)

A production inference server for XLA-accelerated models, compiled through Reactant.jl
(StableHLO via XLA today). It speaks the KServe V2 inference API natively over gRPC, so standard
Triton and KServe clients connect unchanged; it scales from a single GPU to many from one
container; and it squeezes the most models out of each GPU by balancing model **memory** against
**compute**. It is Julia-first throughout — custom pre/postprocessing is plain Julia, and every
convention follows Julia's (column-major, batch-last axes).

It targets static-graph workloads — computer vision, scientific computing — where many models
share a GPU and one model executes at a time.

Not sure which setup fits you? [Common Use Cases](https://enzymead.github.io/ReactantServer.jl/dev/manual/common_use_cases/) walks
through the deployment shapes (single GPU, multi-GPU distributed or replicated, multi-node) with
an example configuration for each.

## Highlights

- **XLA-accelerated, Reactant-compiled.** Models are compiled ahead of time into device
  executables through Reactant's PJRT bindings. The runtime is device-agnostic (CUDA today, CPU
  for dev/fallback); supporting more accelerators is a goal, not a redesign. → [Architecture](https://enzymead.github.io/ReactantServer.jl/dev/design/architecture/), [Philosophy](https://enzymead.github.io/ReactantServer.jl/dev/design/philosophy/)
- **Julia-first pre/postprocessing.** A bundle's `model.jl` registers `preprocess`/`postprocess`
  hooks in plain Julia; they run per request, in parallel and overlapped with GPU execution. → [Bundles & model.jl](https://enzymead.github.io/ReactantServer.jl/dev/manual/bundles/)
- **Julia-aligned conventions.** Shapes are column-major with the batch axis last, the way Julia
  and Lux write them; the codec converts to KServe's row-major wire at the boundary, so Triton
  clients are unchanged and you never reason about row-major order. → [Getting Started](https://enzymead.github.io/ReactantServer.jl/dev/manual/getting_started/)
- **Elegant configuration.** One typed YAML node file (with environment-variable overrides)
  describes a machine; manifests declare tensors with an einsum-style named-axis notation. → [Node Configuration](https://enzymead.github.io/ReactantServer.jl/dev/manual/node_config/), [Bundles & model.jl](https://enzymead.github.io/ReactantServer.jl/dev/manual/bundles/)
- **Standard inference protocol.** KServe V2 over gRPC. Tensor data travels inline or through the
  Triton-compatible system-shared-memory extension for zero-copy local clients. → [Client Usage](https://enzymead.github.io/ReactantServer.jl/dev/manual/client_usage/)
- **One process, single or multi-GPU.** A node supervisor runs one worker per visible GPU: a
  single worker serves the public ports directly; two or more get an embedded gateway behind one
  endpoint. The external interface (`:8001` gRPC, `:8002` metrics/health) is identical either way.
  → [Deployment](https://enzymead.github.io/ReactantServer.jl/dev/manual/deployment/), [Scaling to Multiple GPUs](https://enzymead.github.io/ReactantServer.jl/dev/manual/scaling/)
- **Balances memory and compute.** Every model's weights stay resident in host RAM and stream
  onto the GPU on demand, evicted LRU under a byte budget — so a card serves far more models than
  fit in VRAM, paying a single host-to-device transfer on a cold call. → [On-demand Weights](https://enzymead.github.io/ReactantServer.jl/dev/manual/on_demand_weights/)
- **Batch coalescing.** Concurrent same-model requests are merged into one execution at a compiled
  batch size, amortizing per-launch overhead and the one-time weight transfer across the batch.
  → [Architecture](https://enzymead.github.io/ReactantServer.jl/dev/design/architecture/)
- **Scheduling modes for single and multi-GPU.** On a worker, `fair` (deficit-weighted,
  cost-aware) or `fifo`; across GPUs, the gateway offers `round_robin` or memory-aware
  `lpt_packing` that concentrates each model's traffic to fill batches. → [Architecture](https://enzymead.github.io/ReactantServer.jl/dev/design/architecture/), [Multi-GPU Gateway](https://enzymead.github.io/ReactantServer.jl/dev/manual/multi_gpu_gateway/)
- **Fast iteration.** In `dynamic` mode the server watches the model repository and hot-loads,
  unloads, and reloads bundles online — weights, MLIR, manifest, and `model.jl` alike — with no
  restart (`static` and `explicit` control modes are also available). A model is named by its
  bundle directory, and a renamed directory (a registry promotion such as `-staging` to
  `-production`) renames the live model in place with no recompile. → [Node Configuration](https://enzymead.github.io/ReactantServer.jl/dev/manual/node_config/)
- **Meta models.** A `kind: meta` bundle chains several models with data-dependent Julia between
  stages: its `model.jl` registers a `run` hook that calls sub-models, runs off the GPU dispatch
  loop, and re-enters the scheduler for each sub-call. → [Meta Models](https://enzymead.github.io/ReactantServer.jl/dev/manual/meta_models/)

## Quick start

The node runs natively (no containers): run the supervisor over a directory of model bundles and it
scales to all visible GPUs.

```
git submodule update --init lib/gRPCServer.jl   # the one vendored fork the build needs
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 julia --project=. -e 'using Pkg; Pkg.instantiate()'

CUDA_VISIBLE_DEVICES=0,1,2,3 INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles \
REACTANT_NODE_FILE=config/node.default.yaml \
  julia --handle-signals=no --project=packages/ReactantServerNode \
    -e 'using ReactantServerNode; ReactantServerNode.main()'
```

The first server startup is slow, since every model compiles before the gRPC plane accepts
traffic. See [Deployment](https://enzymead.github.io/ReactantServer.jl/dev/manual/deployment/) for
configuration and running it as a service.

Or from pure Julia:

```julia
using ReactantServerNode
ReactantServerNode.supervise("config/node.yaml")   # one worker per GPU (+ gateway if >1)
```

Clients speak KServe V2 gRPC to `:8001`; health and metrics are on `:8002`. Walk through exporting
a model, configuring a node, and querying it in [Getting Started](https://enzymead.github.io/ReactantServer.jl/dev/manual/getting_started/).
ReactantServer is designed for a trusted network; read
[Security](https://enzymead.github.io/ReactantServer.jl/dev/manual/deployment/#Security) before exposing an endpoint.

## Status

The full serving path is implemented end to end: export a bundle, compile it through Reactant/PJRT,
schedule and coalesce requests, and serve over the KServe V2 gRPC control plane. The cost-aware
scheduler, the on-demand GPU weight cache, dynamic model lifecycle, meta-model orchestration (with
a worked object detection example), and the single- and multi-GPU deployment paths all work today
on CUDA (with CPU for development and fallback); broader accelerator support is intended to follow. Deferred to later milestones: dynamic-batch export with server-side
`stablehlo-refine` specialization, the compiled-executable disk cache, multi-model orchestrators,
and full StableHLO-signature validation of manifests. See
[Architecture](https://enzymead.github.io/ReactantServer.jl/dev/design/architecture/) for the full picture.

## Repository layout

A Julia 1.12+ workspace of five packages under `packages/`, split so that talking to a server never
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
Lux/PythonCall weakdeps stay out of the server images. The vendored `gRPCServer` fork is the only
remaining git submodule under `lib/`.

## Acknowledgments

Development of ReactantServer.jl is sponsored by [Medical Metrics, Inc.](https://medicalmetrics.com/)
