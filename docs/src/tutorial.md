# Tutorial: Export to Inference, End to End

This page walks one model through the whole system: export a small Lux MLP into a bundle with
`export_bundle(:lux, ...)`, write the minimal single-GPU node file, run the server with the
supervisor (the container launcher) or from pure Julia, and query it with `ReactantServerClient`
over KServe V2 gRPC. The commands target a CUDA GPU host (`backend: cuda`, a visible GPU). The
runtime is device agnostic, so the same steps work on CPU by setting `backend: cpu` and dropping
the GPU flags; that is handy for following along without a GPU.

## How the pieces fit

ReactantServer is a Julia workspace of packages, each with one job. A served model is a *bundle*:
a directory with a `manifest.yaml`, a compiled StableHLO program (one `model.b{N}.mlir` per
compiled batch size), and a shared `weights.safetensors`; see [Bundles](bundles.md) for the
format. The worker serves bundles over the KServe V2 gRPC API, one worker per GPU, and the node
supervisor runs a whole machine from one process: it detects the visible GPUs, spawns one worker
subprocess per device, and embeds a thin gateway when there are two or more workers so clients
keep a single endpoint. The client talks to a worker or the gateway from anywhere.

| Package | Role |
|---|---|
| `ReactantServerCore` | Shared, Reactant-free substrate: dtype vocabulary, KServe V2 protobufs, node config, wire codec, shared-memory staging. |
| `ReactantServer` | The inference worker: model registry, runtime, scheduler, KServe V2 gRPC server. The only package that loads Reactant. |
| `ReactantServerNode` | The node supervisor: detects GPUs, spawns one worker subprocess per device, embeds the gateway when needed. |
| `ReactantServerGateway` | Multi-GPU reverse proxy that fronts two or more workers with one endpoint. |
| `ReactantServerClient` | The inference client (`KServeModel`, `infer_sync`, `InferInput`, `InferOutput`). Reactant-free. |
| `ReactantServerExport` | Offline bundle production from Lux or PyTorch models. Not a workspace member; the server never loads it. |

The split is deliberate: only the export step needs the model framework, only `ReactantServer`
needs Reactant, and the client is just another KServe V2 gRPC client.

## Three environments, one flow

Because the packages are split that way, this tutorial runs in three separate Julia environments
that are never loaded together. Export, serve, and query each use their own project, and they talk
to one another only through files and the gRPC wire protocol.

1. **Export environment.** `ReactantServerExport` is not a workspace member; it carries Lux and
   PythonCall weak dependencies the server image should not. Use its test project, which has them
   ready: `julia --project=packages/ReactantServerExport/test`.
2. **Server environment.** The workspace under `packages/` (`ReactantServerCore`,
   `ReactantServer`, `ReactantServerNode`, `ReactantServerGateway`), with gRPCServer sourced from
   its `s-celles-merge` branch on GitHub through the workspace `[sources]`. After cloning,
   instantiate once:

```text
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

3. **Client environment.** `ReactantServerClient` depends only on `ReactantServerCore` and the
   gRPC layer, so add it to any project on a plain client machine and `using ReactantServerClient`.

## Step 1: Export a model into a bundle

Export happens offline, in the export environment. Start it and trace a tiny Lux MLP. The batch
dimension is the last Julia axis (the Lux convention), so a 4-feature input is `(4, batch)`:

```julia
using Lux, ReactantServerExport, Random

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)

example = randn(Float32, 4, 1)            # (features, batch)
ReactantServerExport.export_bundle(:lux, model, ps, st, example;
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1])
```

`export_bundle(:lux, model, ps, st, example; ...)` traces `model(x, ps, st)` at each batch size
and writes `models/mlp/`: a `manifest.yaml`, one compiled StableHLO module per batch size (here
`model.b1.mlir`), and `weights.safetensors`. The input tensor is named `input` and the output
`output` by default (override with `input_name` / `output_name`); the client snippets below use
those names. The `:lux` frontend works for any Reactant-traceable `model(x, ps, st)`; Lux itself
is not required by the frontend, only by this example that builds the model.

`models/` is now a model repository: every immediate subdirectory with a `manifest.yaml` is a
servable model, keyed by its directory name (`mlp` here). See [Bundles](bundles.md) for the
manifest format and custom pre/post-processing via a `model.jl`.

## Step 2: Configure a single-GPU node

A deployment is described by one *node file*. The minimal single-GPU node needs only the model
repository, a base port, the runtime backend, and one worker:

```yaml
# node.yaml
model_repo: /var/lib/reactantserver/models
base_port: 8080
metrics_base_port: 9100

global:
  runtime:
    backend: cuda         # use "cpu" to follow along without a GPU
  endpoints:
    host: 0.0.0.0

workers:
  - { name: worker0 }     # one worker on the (single) GPU
```

The explicit one-entry `workers:` list works with every run path below, including a bare
`ReactantServer.serve`, which expects the workers list. Under the supervisor you can instead omit
`workers:` and write `gpus: auto`, and it synthesizes one worker per detected GPU; two or more
workers bring up the embedded [Gateway](gateway.md). See [Node Configuration](node_config.md) for
the full surface: scheduler tuning, on-demand weights, per-model pinning, environment overrides.

## Step 3: Run the node supervisor

Run the supervisor, pointing it at the repository from Step 1. `INFERENCE_SERVER_MODEL_DIRS`
overrides the node file's `model_repo`, and `REACTANT_NODE_FILE` names the node file:

```text
CUDA_VISIBLE_DEVICES=0 INFERENCE_SERVER_MODEL_DIRS=$PWD/models REACTANT_NODE_FILE=node.yaml \
  julia --handle-signals=no --project=packages/ReactantServerNode \
    -e 'using ReactantServerNode; ReactantServerNode.main()'
```

This is the pure-Julia form of the container launcher: in the image, `docker/entrypoint.node.sh`
runs the same entry point, detects the visible GPUs, spawns one worker subprocess per device, and
restarts children that die.

With a single GPU the node runs one worker and no gateway: the worker serves the KServe V2 gRPC
API on `localhost:8001` and metrics/health on `localhost:8002` (`/readyz`, `/healthz`,
`/metrics`). The first start compiles every model before accepting traffic, so give it a moment;
`curl localhost:8002/readyz` returns 200 once it is serving.

## Step 4: Or serve from pure Julia

Two entry points, differing only in which ports are exposed. First the supervisor, which behaves
exactly like the container launcher `docker/entrypoint.node.sh`: one worker (no gateway) on the
public ports 8001 (gRPC) and 8002 (metrics):

```julia
using ReactantServerNode
ReactantServerNode.supervise("node.yaml")
```

Second, a single bare worker with no supervisor, which serves on the node file's own ports
(gRPC on `base_port`, 8080, metrics on `metrics_base_port`, 9100):

```julia
using ReactantServer
ReactantServer.serve("node.yaml")          # blocks; Ctrl-C to stop
```

[`serve`](@ref) loads the node file, brings up the runtime, compiles the worker's bundles, starts
the `ReactantServer.Scheduler`, and finally starts the gRPC server so traffic is accepted only once
models are live. Pass `blocking=false` to get a `ReactantServer.RunningServer` you can
[`stop!`](@ref):

```julia
server = ReactantServer.serve("node.yaml"; blocking = false)
# ... issue requests ...
ReactantServer.stop!(server)
```

`supervise` is the right choice for deployment (it is what the launcher runs, and it scales to
many GPUs unchanged); a bare `serve` is convenient for a quick single-worker REPL session.

Start Julia multithreaded so per-request `preprocess`/`postprocess` hooks overlap the GPU
execution: `julia --threads=auto,1` gives a default pool for the hooks plus one interactive
thread for the GPU dispatch loop. Set this yourself only for a bare `serve`; under the
supervisor each worker is instead sized to its share of the host (`min(cores / workers, 16)`
compute threads plus the interactive one, overridable via `REACTANT_WORKER_THREADS`), so
co-located workers do not oversubscribe the CPU. With a single thread the server still works,
just without the overlap.

## Step 5: Query it with the ReactantServerClient

The server speaks KServe V2 over gRPC, so any Triton/KServe client works; this repository ships
the Reactant-free `ReactantServerClient`. From the client environment, point it at the port your
server is using (8001 for the supervisor, 8080 for a bare `serve`) and use the bundle's tensor
names `input` / `output`:

```julia
using ReactantServerClient

kserve_init()
try
    model = KServeModel("grpc://127.0.0.1:8001", "mlp"; max_batch_size = 1)
    x = Float32[1, 2, 3, 4]                       # one 4-feature item
    response = infer_sync(model, [InferInput("input", x)])
    y = InferOutput("output", response, Float32)  # length-3 output
    @show vec(collect(y))
finally
    kserve_shutdown()
end
```

[`kserve_init`](@ref) prepares the client transport (a staging pool plus dispatch slots) and
[`kserve_shutdown`](@ref) tears it down. [`KServeModel`](@ref)`(url, model_name)` binds a model
on the server; [`infer_sync`](@ref) sends one request and returns the raw response;
[`InferInput`](@ref) builds a named input tensor and [`InferOutput`](@ref) reads a named tensor
back as `Float32`. See [Client Usage](client.md) for batched inference over a dataset, IO
validation, and the shared-memory data path.

## Next steps

- [Bundles](bundles.md): the bundle format, manifest fields, and custom pre/post-processing.
- [Node Configuration](node_config.md): the full node.yaml surface and environment overrides.
- [Scheduling](scheduling.md): how the worker picks the next model and coalesces requests.
- [On-demand Weights](on_demand_weights.md): serving more models than fit in GPU memory.
- [Gateway](gateway.md): multi-GPU deployments behind one client endpoint.
- [Client](client.md): the full client API and data paths.
- [Meta Models](meta_models.md) and the worked examples [Object Detection](object_detection.md)
  and [Transformers](transformers.md).
- [Deployment](deployment.md): systemd, Docker, health, and metrics.
- [API](api.md): every documented name, collected automatically.
