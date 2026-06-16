# ReactantServer.jl

A production inference server that serves compiled XLA models from a single Julia process,
built on Reactant.jl's PJRT bindings. It targets static-graph workloads (computer vision,
scientific computing) where many models share one GPU and only one model executes at a time.
To serve more models than fit in GPU memory at once, it keeps every model's weights resident
in host RAM and transfers them onto the GPU on demand, evicting cold models under a memory
budget. It speaks the KServe V2 inference API natively over gRPC, so standard Triton/KServe
clients connect to it directly.

For a high-level overview of the goals and the scheduler, see
[docs/src/design/architecture.md](docs/src/design/architecture.md); for the mission and
non-goals, [docs/src/design/philosophy.md](docs/src/design/philosophy.md).

## Repository layout

A Julia 1.11 workspace of five packages under `packages/`, plus the non-member
`ReactantServerExport`:

- **`ReactantServerCore`** — the shared, Reactant-free substrate: dtypes, the KServe V2 protobuf
  messages, boundary types, the manifest parser, server/node config, the wire codec, the
  shared-memory registry, and the concurrency-safe staging `BufferPool`.
- **`ReactantServer`** — the inference worker: registry, runtime, scheduler, and KServe V2 gRPC
  server. The **only** package that depends on Reactant. Exports `serve`, `serve_worker`,
  `stop!`, `register_model`.
- **`ReactantServerGateway`** — the multi-GPU KServe gRPC reverse proxy (`serve_gateway`,
  `probe_worker_ready`). No Reactant.
- **`ReactantServerClient`** — a Reactant-free inference client (`KServeModel`,
  `infer_sync`, `infer_async`, `InferInput`, `InferOutput`).
- **`ReactantServerNode`** — the single-container node supervisor (`supervise`): detects the
  visible GPUs, runs one worker subprocess per device (plus an embedded gateway when there is
  more than one worker; a single worker serves the public ports directly), multiplexes their
  logs with `[name]` line prefixes, and restarts children that die. No Reactant.

Offline model-export tooling is `packages/ReactantServerExport` (the bundle writer plus the
Reactant tracing frontend; PyTorch support is a package extension that loads when `PythonCall`
is present). It is deliberately not a workspace member, so its Lux/PythonCall weakdeps stay out
of the server images. The vendored forks/unregistered deps (`Reactant`, `gRPCServer`,
`gRPCClient`, `HTTP`) are git submodules under `lib/`.

## Status: core serving path implemented

This repository implements a complete path through every layer: load a StableHLO model bundle
from disk, compile it through Reactant/PJRT, schedule and serve it over the KServe V2 gRPC
control plane, and return a result. The cost-aware coalescing scheduler and the on-demand GPU
weight cache are both implemented (see below). The runtime is device agnostic; it defaults to
CPU PJRT and selects CUDA through configuration, with CPU fallback.

What works today:

- StableHLO bundle loading, manifest parsing and validation, and a typed YAML configuration
  with environment-variable overrides.
- The Reactant/PJRT runtime: deserialize a portable artifact, compile with weights bound as
  explicit arguments, execute, and read results back. A single shared memory pool backs all
  models.
- On-demand GPU weight loading with a host-RAM weight cache. With the cache enabled
  (`runtime.weight_cache_bytes > 0`), every model's weights default to system-pinned: they are
  materialized into host RAM at startup, transferred to the GPU on first request, kept resident
  for reuse, and evicted under the configured GPU byte budget (least-recently-used) with an
  immediate PJRT buffer free. Models can be pinned to stay GPU-resident
  (`scheduler.models.<name>.residency: device`, or the back-compat alias `pin_to_gpu: true`)
  or opted out of the host floor (`residency: unpinned`, which re-materializes from the mmap on
  each load). Because the weights are already in RAM, an on-demand load is a single
  host-to-device transfer, tens of milliseconds even for the largest models. Setting
  `weight_cache_bytes` to 0 keeps every model resident (the original behavior). This decouples
  the number of servable models from GPU memory capacity.
- The scheduler: a deficit-weighted, cost-aware, coalescing dispatch policy. Concurrent
  requests land on per-model queues; a single dispatch loop runs one execution at a
  time, coalesces same-model requests into one batched execution at a compiled size, and shares
  GPU time across models by relative weight and a learned per-batch-size cost estimate.
  Per-model latency budgets (earliest-deadline-first escalation) are planned but not yet
  implemented. Per-model metrics (dispatch count, compute, queue-wait percentiles, coalesced
  batch-size histogram) and weight cache residency counters are exposed.
- Model lifecycle control (`model_control_mode`): `dynamic` (the default) watches the model
  repository and loads new bundles, unloads removed ones, and reloads changed ones online,
  covering weights, MLIR, manifest, and `model.jl` changes with a two-poll debounce; `static`
  fixes the startup set; `explicit` cedes lifecycle and residency to an external control plane
  via the worker control RPCs (`ModelControlStatus`, `SetModelResidency`, `SetModelPolicy`).
- Gateway scheduling (`scheduling.mode`): `round_robin` spreads each model's requests uniformly
  across replicas; `lpt_packing` places models on workers adaptively with memory-aware LPT bin
  packing, concentrating each model's traffic for batch coalescing while balancing measured
  compute demand and resident weight footprint per GPU, rebalanced from live measurements
  (requires `fifo` workers; verified at startup).
- The KServe V2 control plane over gRPC: server and model liveness/readiness, model and server
  metadata, inference, and a `RepositoryIndex` that lists the models a worker has loaded (for
  direct client introspection). Tensor data travels either
  inline (`raw_input_contents` / `raw_output_contents`) or through the Triton-compatible system
  shared-memory extension. The gRPC transport is provided by gRPCServer.jl (HTTP/2); the codec,
  scheduler, and runtime are transport-agnostic.
- The system shared-memory data plane: clients register a POSIX region
  (`SystemSharedMemoryRegister` / `Unregister` / `Status`) and reference it from input and
  output tensors via the `shared_memory_region` / `shared_memory_offset` /
  `shared_memory_byte_size` parameters. The server attaches the region with
  InterProcessCommunication.jl, copies inputs into host arrays, and writes outputs back into
  client-provided regions.
- Multiple compiled batch sizes per model. A bundle may carry one static module per size
  (`model.b{N}.mlir`) alongside a single shared `weights.safetensors`; the server compiles
  each and selects the matching executable per request. A lone `model.mlir` (single size)
  remains supported.
- Custom per-model pre/post-processing via a bundle's `model.jl`, which calls `register_model`
  with `preprocess` and `postprocess` hooks. The scheduler runs them around each dispatch
  (crossing the world-age boundary with `invokelatest`); omitted hooks default to identity.
- Offline conversion tooling in `packages/ReactantServerExport` (not part of the server
  runtime): `export_bundle` traces a Reactant model (or any Reactant-traceable function) into a
  bundle, the PythonCall-triggered extension adds `export_bundle`/`export_torchscript_bundle` for
  a `torch.nn.Module` via `torch.export` + torchax, and `write_bundle` is the shared bundle-format
  writer. All frontends re-trace once per requested batch size.

Deferred to later milestones (each has a seam in the current code): dynamic-batch export
with server-side `stablehlo-refine` specialization (so a single MLIR module covers many
sizes); the compiled-executable disk cache; CUDA (device) shared memory; multi-model
orchestrators; and full StableHLO-signature validation of manifests.

## Deployment: one container, any number of GPUs

The recommended deployment is the unified `reactantserver` image, whose entrypoint is the node
supervisor (`ReactantServerNode`). It detects every GPU granted to the container, spawns one
single-GPU worker subprocess per device, multiplexes all logs onto the container's stdout with
`[worker0]` / `[gateway]` line prefixes, and restarts children that die. With two or more
workers it also runs an embedded gateway on the public ports; with a single worker it skips the
gateway and binds that worker to the public ports directly. A multi-GPU deployment is therefore
a single container with no per-GPU configuration:

```
docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
```

Clients speak KServe V2 gRPC to `:8001`; health and metrics are on `:8002`. See `docker/README.md`
for configuration, multi-node roles (`REACTANT_ROLE=all|workers|gateway`), and metrics.

Underneath, each worker is a single Julia process that drives one GPU and serves the complete
KServe V2 gRPC surface on its own. For a single-GPU bare-metal deployment that is all you need:
point your KServe V2 gRPC client straight at the worker's host and port.

The gateway reverse proxy exists for **multi-GPU configurations**. Because a worker hosts one
GPU and executes one model at a time, scaling across several GPUs means running one worker per
GPU and spreading models across them. The gateway gives clients a single gRPC endpoint and
routes each `ModelInferRequest` to a worker serving the requested model, forwarding the protobuf
bytes unchanged. Replica scheduling is set by `scheduling.mode`: `round_robin` (the default)
spreads requests uniformly, while `lpt_packing` derives an adaptive memory-aware placement from
live measurements, concentrating each model's traffic so worker-side batch coalescing fills (it
requires workers running the `fifo` scheduler discipline). A standalone gateway is configured by
its own `gateway.yml` (a flat list of worker endpoints across one or more nodes) and
autodiscovers which models each endpoint serves via `RepositoryIndex`, refreshing its routing
table periodically; the supervisor's embedded gateway is configured automatically from the node
file. The gateway is pure Julia, in its own `ReactantServerGateway` package
(`ReactantServerGateway.serve_gateway`); see `docs/src/manual/multi_gpu_gateway.md`.

## Shape convention (Julia-centric, zero-copy interop)

Shape declarations in `manifest.yaml` and the server's internal `NamedTensor.data` use the
Julia column-major convention (Lux convention: the batch dimension is the last axis). A
Lux `Dense(4 => 8)` input thus appears as `shape: "cn", dims: {c: 4}`, which is the reverse
of how Python/XLA writes the same tensor. The codec advertises and accepts KServe V2 wire
shapes in their canonical row-major form, so Triton-style clients are unchanged; the
underlying tensor bytes are the same memory under either view, and the codec converts by
reshaping rather than permuting. This is the standard column-major / row-major interop
trick, applied at the wire boundary.

The Julia client (`ReactantServerClient`) presents Julia column-major shapes throughout, so a
Julia user never reasons about the row-major wire order. You build and read arrays in their
natural Julia shape, you declare input and output shapes in Julia order, and introspection
(`model_io_spec` / `manifest_io_spec`) reports shapes in Julia order too, matching the manifest's
einsum letters. The reversal to KServe's row-major wire happens internally, only at the wire
boundary.

Shapes use an einsum-style notation: each axis is a single ASCII letter and the companion
`dims:` map gives the size of every non-batch letter. Letters `n` and `b` are reserved for
the batch axis (at most one occurrence per tensor); other letter names are tensor-scoped
and carry no implicit equality across tensors. A size of `-1` in `dims` marks a variable
non-batch axis (used today only in `client_outputs` that pass through `model.jl`). The
per-input batch axis is derived from the position of `n` (or `b`) in each input's shape
string; at inference the request's size along that axis must equal one of
`batching.compiled_batch_sizes`.

## Conversion tooling (ReactantServerExport)

`ReactantServerExport` produces bundles offline and is kept out of the server's dependency
graph. A project that owns a Lux model (or any Reactant-traceable function) adds
`ReactantServerExport` (`Pkg.develop`) and calls `export_bundle`; Lux is not required by the
package:

```julia
using ReactantServerExport
export_bundle(:lux, model, ps, st, example_input;
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
```

A PyTorch project additionally loads `PythonCall`, which triggers the package extension that
drives `torch.export.export` and torchax:

```julia
using ReactantServerExport, PythonCall
export_bundle(:pytorch, model, (example_input,);
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
# TorchScript artifacts: export_torchscript_bundle(pt_path, (example_input,); ...)
```

Both frontends trace once per batch size and write a server-loadable bundle. The batch
dimension is the last Julia axis (i.e. the leading PyTorch axis after row-major / column-major
reversal). Run the round-trip tests with
`julia --project=packages/ReactantServerExport/test packages/ReactantServerExport/test/runtests.jl`;
the PyTorch portion skips gracefully when `torch` / `torchax` are unavailable.

## Running

```julia
using ReactantServer
ReactantServer.serve("docker/node.yaml")                      # single worker: name optional
ReactantServer.serve("docker/node.yaml"; worker="worker0")    # multi-worker: name the worker
```

Configuration is a single node file (see `docker/node.yaml`) that describes one machine's
single-GPU workers, a shared model repository, and a base port. `serve` resolves this process's
worker from that file: global settings are merged with the worker's overrides, the listen port
is derived from `base_port`, and the device ordinal from the worker's `gpu`. Every worker loads
(and can serve) every bundle in the repository; the optional top-level `models:` map only pins
the named models to device memory on the listed workers. The `worker` keyword selects the
entry; it may be omitted when the node has exactly one worker.

`serve` loads the configuration, brings up the runtime client, compiles the bundles assigned
to the worker, starts the scheduler, and then starts the gRPC server. Pass `blocking=false` to
run it in the background and receive a handle that `stop!` shuts down. Pass
`backend=ReactantServer.ReactantBackend()` (the default) for real execution. Any KServe V2 gRPC
client can then call the server directly at the configured host and port.

For a multi-GPU deployment, the supervisor does the fan-out for you:

```julia
using ReactantServerNode
ReactantServerNode.supervise("docker/node.yaml")   # one worker per visible GPU (+ gateway if >1)
```

It synthesizes the worker list when the node file has none (`gpus: auto`) and spawns each worker
as a subprocess pinned to its device via `CUDA_VISIBLE_DEVICES`. With more than one worker it
also runs the embedded gateway, which routes by model name and load-balances replicated models;
a single worker serves clients directly. Equivalently, run one worker per
GPU yourself (each pointed at the same node file with a distinct `worker`) behind a standalone
gateway: `using ReactantServerGateway; ReactantServerGateway.serve_gateway(gateway_path)` reads
its own `gateway.yml` and serves the KServe V2 gRPC proxy. To call a server from Julia, use the
Reactant-free `ReactantServerClient` package (`KServeModel` + `infer_sync`). See `docker/`
for the container setup.

A bundle is a directory containing `manifest.yaml`, `model.mlir` (a serialized StableHLO
portable artifact), `weights.safetensors`, and an optional `model.jl`. Bundles are produced by
offline conversion tooling; the test suite builds small bundles directly (see
`packages/ReactantServer/test/stablehlo_fixtures.jl`).

## Security posture

ReactantServer is designed to run on a trusted network behind your own perimeter. Be aware of
the following before exposing any endpoint:

- All gRPC traffic (worker and gateway) is cleartext h2c. TLS settings are parsed by the
  gateway config but not yet enforced; a configured cert triggers a startup warning.
- There is no authentication or authorization on the KServe data plane, the worker
  control-plane RPCs (residency and policy), or the gateway's Prometheus metrics listener
  (which binds `0.0.0.0:8002` by default).
- Model bundles are trusted input: a bundle's optional `model.jl` executes arbitrary Julia in
  the server process. Only serve bundles you built or audited.
- POSIX shared memory is a local trust boundary. Client-registered regions and the optional
  node-shared host-weight store live in `/dev/shm`; the shared weight regions default to mode
  `666` (world-writable) for cross-container friction-free sharing. Set
  `runtime.shared_host_weights_mode: "660"` on production or multi-user systems.

## Testing

Each package is tested in its own environment; all tests run on CPU and need no GPU:

```
julia --project=packages/ReactantServerCore   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer        -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient  -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerNode    -e 'using Pkg; Pkg.test()'
```

`packages/ReactantServer/test/spike_reactant.jl` is a standalone script that exercises the
Reactant runtime path in isolation.

## Protobuf bindings

The KServe V2 messages and gRPC service stubs in
`packages/ReactantServerCore/src/proto/inference/` are generated from
`proto_src/grpc_predict_v2.proto` with ProtoBuf.jl. Load gRPCServer and gRPCClient alongside
ProtoBuf so both the server method builders / `register_GRPCInferenceService!` and the client
constructors are emitted, and keep `add_kwarg_constructors=true` (the handlers and codec build
messages with keyword arguments):

```julia
using ProtoBuf, gRPCServer, gRPCClient
ProtoBuf.protojl("grpc_predict_v2.proto", "proto_src", "packages/ReactantServerCore/src/proto";
    always_use_modules=true, add_kwarg_constructors=true)
```

The generated file is then split so `ReactantServerCore` compiles only the messages (no gRPC
dependency): the messages stay in `grpc_predict_v2_pb.jl`, while the gRPCClient and gRPCServer
service stubs are extracted into `grpc_client_stubs.jl` and `grpc_server_stubs.jl`, which
`ReactantServerCore` ships but does not compile. Each consumer includes the stub file it needs
(client stubs in the client and gateway; server stubs in the worker and gateway) via
`ReactantServerCore.inference_client_stubs_path()` / `inference_server_stubs_path()`.
