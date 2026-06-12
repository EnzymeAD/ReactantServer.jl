# ReactantServer: Project Overview

## What it is

ReactantServer is a production inference server that serves many compiled XLA models from a single
Julia process on one GPU. Models are delivered as self-contained bundles (a StableHLO program,
its weights, and a manifest), compiled once through Reactant.jl's PJRT bindings, and served over
the KServe V2 inference API via gRPC, so standard Triton and KServe clients connect without
changes. It targets static-graph workloads such as computer vision and scientific computing,
where many models share a GPU and one model executes at a time.

## Package layout

The project is a Julia workspace of four packages (plus the non-member `ReactantServerExport`
for offline bundle export), split so that talking to a server never requires the heavy
Reactant/XLA stack:

- **ReactantServerCore** — the shared, Reactant-free substrate: the dtype vocabulary, the
  KServe V2 protobuf messages, the transport-agnostic boundary types, the manifest parser,
  server/cluster config, the wire codec, the shared-memory registry, and the
  concurrency-safe staging `BufferPool`. The other three packages all build on it.
- **ReactantServer** — the inference worker. It owns the model registry, the runtime, the
  scheduler, and the KServe V2 gRPC server, and is the **only** package that loads Reactant.
  Exports `serve`, `serve_worker`, `stop!`, `register_model`.
- **ReactantServerGateway** — the multi-GPU reverse proxy (`serve_gateway`,
  `probe_worker_ready`). Builds only on Core and the gRPC layer; no Reactant.
- **ReactantServerClient** — the inference client (`KServeModel`, `infer_sync`,
  `infer_async`, `InferInput`, `InferOutput`). Also Reactant-free, so it installs on a plain
  client machine. See [Client Usage](../manual/client_usage.md).

The offline model-export tooling (`packages/ReactantServerExport`, with a PythonCall-triggered
PyTorch extension) is not a workspace member; it produces bundles the worker consumes.

## What it aims to achieve

The project has one economic goal: serve the largest possible number of models per GPU at a
given quality of service, while running each model as fast as the hardware allows. GPU memory
is the dominant cost in inference infrastructure, so serving more models per card directly
lowers cost per inference. Two technical levers deliver this:

1. **Fit more models on a GPU.** Weights do not all have to live on the GPU at once. The server
   keeps every model's weights resident in host RAM and moves a model's weights onto the GPU
   only when it is needed, evicting cold models to stay within a memory budget. This decouples
   the number of servable models from GPU memory capacity.
2. **Run each model faster with the XLA compiler.** Each model is compiled ahead of time into an
   optimized executable. XLA performs whole-program optimization, fuses kernels across
   operations, and plans memory layout, which makes execution significantly faster than running
   the equivalent eager-mode graph. Small teams get compiler-grade performance without building
   it themselves.

The broader mission, the target audience, and the explicit non-goals are on the
[Philosophy](philosophy.md) page. Operational and format details are in the
[Getting Started](../manual/getting_started.md) and
[Cluster Configuration](../manual/cluster_config.md) guides.

## How it fits together

- **Bundles.** A model is a directory with `manifest.yaml`, one or more StableHLO modules
  (`model.b{N}.mlir`, one per compiled batch size), a shared `weights.safetensors`, and an
  optional `model.jl` for pre/post-processing. Bundles are produced by offline conversion
  tooling from Lux or PyTorch models and loaded at server startup.
- **One process per GPU.** Each worker drives a single GPU, holds a single shared memory pool,
  and serves the full KServe V2 gRPC surface on its own. For multi-GPU deployments a thin
  gateway gives clients one endpoint and schedules each model's traffic across workers, either
  uniformly (round robin) or by adaptive memory-aware placement (`lpt_packing`); see the
  [Multi-GPU Gateway](../manual/multi_gpu_gateway.md) guide. A single-GPU deployment needs no
  gateway.
- **Weights as explicit arguments.** Weights are passed to the compiled executable as arguments
  rather than baked into it. This is what makes on-demand loading and weight sharing across
  batch sizes possible: compilation is independent of where the weights currently live.

## Fitting more models on the GPU

Because only one model executes at a time, the GPU does not need every model's weights resident
simultaneously. The server organizes weights in two tiers:

- **Host RAM (resident by default).** With the on-demand cache enabled, every model's weights
  default to system-pinned: materialized once from disk into host RAM at startup and kept
  there (`residency: unpinned` opts a model out). Host memory is plentiful and cheap relative
  to GPU memory, so this costs little and removes disk from the hot path.
- **GPU (managed working set).** Models marked as pinned keep their weights on the GPU for the
  server's lifetime. Every other model is loaded onto the GPU on demand when a request arrives,
  kept resident afterward so repeat requests are free, and evicted under a configured GPU byte
  budget using a least-recently-used policy. Eviction frees the device memory immediately
  through an explicit PJRT buffer release.

Because the weights are already in RAM, an on-demand GPU load is a single host-to-device
transfer rather than a reload from disk. In practice this is tens of milliseconds even for the
largest models, which is the same order of magnitude as a single inference. The effect is that a
GPU sized for a handful of models can serve a much larger catalog, paying a small transfer cost
only when a cold model is first called.

## Model lifecycle

How the set of loaded models changes over a worker's lifetime is set by `model_control_mode`
(mirroring Triton's model control modes):

- **`dynamic`** (the default) watches the model repository and reconciles continuously: new
  bundles are loaded, removed bundles are unloaded, and changed bundles are reloaded. Change
  detection covers the weights, the MLIR, the manifest, and the bundle's `model.jl`, so
  updating a model's weights or its Julia pre/post-processing is just writing the files. A
  two-poll debounce keeps half-written bundles from loading, and a reload swaps one model
  atomically while every other model keeps serving.
- **`static`** loads the startup set once and never changes it.
- **`explicit`** cedes the lifecycle to an external control plane: the worker takes no
  autonomous action, and model residency is driven entirely over a small gRPC control surface
  (`ModelControlStatus` to observe, `SetModelResidency` to pin and unpin, `SetModelPolicy` to
  adjust scheduling weight). A model serves only while the control plane holds it resident.
  This is the integration seam for organizations that run their own placement logic.

## The scheduler

The scheduler is the component that decides, each time the GPU is free, which model to run next
and how many queued requests to serve in one execution. It is a deficit-weighted, cost-aware,
coalescing dispatch policy that runs on top of the language task scheduler and
adds no threading of its own. Exactly one GPU execution runs at a time by design, which keeps
memory and concurrency reasoning simple and is what makes the single-resident-model strategy
above safe.

### Structure

Requests arrive concurrently over gRPC and are placed on per-model FIFO queues. A single
dispatch loop selects the next model, coalesces that model's queued requests into one execution
at a compiled batch size, runs it, and splits the results back to each caller. One lock guards
the queues and the per-model statistics and wakes the loop when work arrives. The loop holds the
lock only to select and dequeue; the GPU execution runs outside the lock, so new requests keep
arriving during a running inference.

### Decision order

Each time the GPU frees up, the scheduler decides in this order:

1. **Fairness.** Each model has a relative weight that defines its fair share of
   compute. The scheduler tracks a decaying exponential moving average of how much GPU time each
   model has recently consumed, and scores every model with a non-empty queue by how far below
   its fair share it is, divided by its estimated cost. A model that has used less than its share
   recently scores higher, and dividing by cost stops an expensive model from blocking cheaper
   ones on a marginal fairness edge. A clamp bounds both lockout and domination. Under the
   alternative `fifo` discipline, the model with the oldest queued request wins instead; choose
   it when workers sit behind a gateway running `lpt_packing`, which moves the placement and
   fairness decisions upstream (see the
   [Multi-GPU Gateway](../manual/multi_gpu_gateway.md) guide).
2. **Coalesce the winner.** The selected model's queued requests are taken in FIFO order and
   packed into the largest compiled batch size that fits, padding a partial batch up to the
   smallest size when needed and always making forward progress on at least the oldest request.
   The remainder stays queued for the next round.

Per-model latency budgets (earliest-deadline-first escalation ahead of fairness) are a planned
extension; they are not implemented today.

### Cost learning and coalescing

The scheduler measures the real GPU time of each execution and refines a per-batch-size cost
estimate with an exponential moving average, seeded by a warmup pass at startup so the first
real requests are scheduled sensibly. Coalescing concatenates the inputs of several same-model
requests along the batch axis into one execution, then slices the outputs back per caller. Only
models whose inputs and outputs carry a batch axis are coalesced; others serve one request per
dispatch.

Coalescing is the throughput lever. Packing many requests into one execution amortizes the fixed
per-launch overhead and, for a model that had to be loaded on demand, the one-time weight
transfer, across every image in the batch. On a representative compute-heavy model, per-image
latency drops by roughly three times from batch 1 to batch 32 while images per second more than
triples.

### Configuration and observability

Per model, operators set a relative weight, the residency floor, and whether the model is
pinned to the GPU. Globally they set the recent-compute half-life, the cost estimate smoothing,
a coalescing cost discount, a per-model maximum queue depth, and the GPU weight-cache byte
budget. Configuration is typed YAML with environment-variable overrides. The scheduler exposes
per-model metrics: dispatch count, total compute, current recent-compute load,
queue depth, queue-wait percentiles, and the histogram of coalesced batch sizes, plus weight
cache residency and load/evict counters.

## The XLA compiler advantage

Serving static graphs through an ahead-of-time compiler is the project's second source of
leverage. Each StableHLO module is compiled once into a device executable. XLA optimizes the
whole program rather than one operation at a time: it fuses adjacent operations into single
kernels, eliminates redundant work, and plans tensor layouts for the target device. The result
executes substantially faster than the same model run eagerly, and the cost is paid once at
startup rather than on every request. A bundle may carry several compiled batch sizes that share
one set of weights, so the scheduler can pick the executable that matches each coalesced batch
without duplicating parameters.

## Scope

ReactantServer is opinionated and deliberately narrow. It is for small and mid-size engineering
organizations that need efficient inference on static-graph models and that measure their own
systems. It is XLA-centric and is not a general multi-framework server, not an LLM serving
stack, and not a managed service. The reasoning behind these boundaries is on the
[Philosophy](philosophy.md) page.
