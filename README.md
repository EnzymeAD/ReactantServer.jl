# ReactantServer.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzymead.github.io/ReactantServer.jl/dev/)
[![CI](https://github.com/EnzymeAD/ReactantServer.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzymeAD/ReactantServer.jl/commits/main)
[![Coverage](https://codecov.io/gh/EnzymeAD/ReactantServer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EnzymeAD/ReactantServer.jl)
[![Julia 1.12](https://img.shields.io/badge/Julia-1.12-9558b2)](https://julialang.org)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

A production inference server for models compiled through Reactant.jl (StableHLO via XLA today).
It speaks the KServe V2 inference API over gRPC, so standard Triton and KServe clients connect
unchanged; it scales from one GPU to many from one container; and it fits more models on each GPU
by balancing model **memory** against **compute**. It is built for workloads where many models
share a GPU, such as computer vision and scientific computing.

## Highlights

- **XLA-accelerated, Reactant-compiled.** Models are compiled ahead of time into device
  executables through Reactant's PJRT bindings. The runtime is device-agnostic (CUDA today, CPU
  for dev/fallback); supporting more accelerators is a goal, not a redesign. → [Architecture](https://enzymead.github.io/ReactantServer.jl/dev/design/architecture/), [Philosophy](https://enzymead.github.io/ReactantServer.jl/dev/design/philosophy/)
- **Julia-first pre/postprocessing.** A bundle's `model.jl` registers `preprocess`/`postprocess`
  hooks in plain Julia; they run per request, in parallel and overlapped with GPU execution. → [Bundles & model.jl](https://enzymead.github.io/ReactantServer.jl/dev/bundles/)
- **Julia-aligned conventions.** Shapes are column-major with the batch axis last, the way Julia
  and Lux write them; the codec converts to KServe's row-major wire at the boundary, so Triton
  clients are unchanged and you never reason about row-major order. → [Export a Model](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/export/)
- **One configuration file.** One typed YAML node file (with environment-variable overrides)
  describes a machine; manifests declare tensors with an einsum-style named-axis notation. → [Node Configuration](https://enzymead.github.io/ReactantServer.jl/dev/node_config/), [Bundles & model.jl](https://enzymead.github.io/ReactantServer.jl/dev/bundles/)
- **Standard inference protocol.** KServe V2 over gRPC. Tensor data travels inline
  (`raw_input_contents` / `raw_output_contents`) or through the
  Triton-compatible system-shared-memory extension for zero-copy local clients. → [Client Usage](https://enzymead.github.io/ReactantServer.jl/dev/client/)
- **One container, one GPU or many.** A node supervisor runs one worker per visible GPU, behind
  one endpoint that stays the same as the node grows.
  → [Deployment](https://enzymead.github.io/ReactantServer.jl/dev/deployment/), [Multi-GPU Gateway](https://enzymead.github.io/ReactantServer.jl/dev/gateway/)
- **Balances memory and compute.** Every model's weights stay resident in host RAM and stream
  onto the GPU on demand, evicted LRU under a byte budget — so a card serves far more models than
  fit in VRAM, paying a single host-to-device transfer on a cold call. → [On-demand Weights](https://enzymead.github.io/ReactantServer.jl/dev/on_demand_weights/)
- **Batch coalescing.** Concurrent same-model requests are merged into one execution at a compiled
  batch size, amortizing per-launch overhead and the one-time weight transfer across the batch.
  → [Scheduling](https://enzymead.github.io/ReactantServer.jl/dev/scheduling/)
- **Scheduling modes for single and multi-GPU.** On a worker, `fair` (deficit-weighted,
  cost-aware), `fifo`, or `edf`; across GPUs, the gateway offers `round_robin`,
  `least_outstanding`, or memory-aware `lpt_packing` that concentrates each model's traffic to fill
  batches, with a gRPC control plane for retuning placement without a restart.
  → [Scheduling](https://enzymead.github.io/ReactantServer.jl/dev/scheduling/), [Multi-GPU Gateway](https://enzymead.github.io/ReactantServer.jl/dev/gateway/)
- **Fast iteration.** In `dynamic` mode the server watches the model repository and hot-loads,
  unloads, and reloads bundles online — weights, MLIR, manifest, and `model.jl` alike — with no
  restart (`static` and `explicit` control modes are also available). A model is named by its
  bundle directory, and a renamed directory (a registry promotion such as `-staging` to
  `-production`) renames the live model in place with no recompile. → [Node Configuration](https://enzymead.github.io/ReactantServer.jl/dev/node_config/)
- **Meta models (experimental).** A `kind: meta` bundle chains several models with data-dependent
  Julia between stages: its `model.jl` registers a `run` hook that calls sub-models, runs off the
  GPU dispatch loop, and re-enters the scheduler for each sub-call. The design is subject to change;
  do not build production pipelines on it. → [Meta Models](https://enzymead.github.io/ReactantServer.jl/dev/meta_models/)

## Get started

The [tutorials](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/) follow a model from [export](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/export/) to
[inference](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/inference/), with two ways to serve it in between:

- **From Julia**, on any machine with or without a GPU: add the packages to an environment and
  [serve a model](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/julia/serve/) from a Julia session. These tutorials run in CI exactly
  as written.
- **With the container image**, the supported deployment: a host needs only the NVIDIA driver and
  a container runtime. [Serve on one GPU](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/container/single_gpu/), then
  [scale to many](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/container/multi_gpu/).

ReactantServer is designed for a trusted network; read [Security](https://enzymead.github.io/ReactantServer.jl/dev/security/) before exposing an
endpoint.

## Repository layout

A Julia 1.12+ workspace of five packages under `packages/`, split so that only the worker loads
Reactant and XLA among them:

- **`ReactantServerCore`**: the shared, Reactant-free code the other packages build on.
- **`ReactantServer`**: the worker, which runs models through Reactant.
- **`ReactantServerGateway`**: the reverse proxy that puts several workers behind one endpoint.
- **`ReactantServerClient`**: the client.
- **`ReactantServerNode`**: the node supervisor.

Offline model export lives in `packages/ReactantServerExport`, which also uses Reactant, to trace
models. It is not a workspace member, so its export dependencies stay out of the server image. All dependencies, including
`gRPCServer`, resolve from the package registries; there are no git submodules.

## Acknowledgments

Development of ReactantServer.jl is sponsored by [Medical Metrics, Inc.](https://medicalmetrics.com/)
