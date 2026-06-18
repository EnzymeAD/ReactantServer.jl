```@meta
CurrentModule = ReactantServer
```

# Getting Started

This is a complete walkthrough: export a small Lux model into a bundle, configure a single-GPU
node, run it (both with `docker compose` and from pure Julia), and query it with the Julia
client. When you are ready for more than one GPU, continue to
[Scaling to Multiple GPUs](scaling.md).

The commands below target a CUDA GPU host (`backend: cuda`, `--gpus all`). The runtime is device
agnostic, so the same steps work on CPU by setting `backend: cpu` and dropping the GPU flags;
that is handy for following along without a GPU.

## Installation

ReactantServer is a Julia workspace of five packages under `packages/` (`ReactantServerCore`,
`ReactantServer`, `ReactantServerGateway`, `ReactantServerClient`, `ReactantServerNode`), plus the
non-member `ReactantServerExport` for offline bundle export (see
[Architecture](../design/architecture.md)). It vendors its forked/unregistered dependencies
(Reactant, gRPCServer, gRPCClient, HTTP) as git submodules under `lib/` and wires them in through
the workspace `[sources]`. After cloning, populate the submodules and instantiate the workspace:

```
git submodule update --init --recursive
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To work in a single package, activate its project instead, e.g.
`julia --project=packages/ReactantServer`.

## Step 1: Export a model into a bundle

A served model is a *bundle*: a directory with a `manifest.yaml`, a compiled StableHLO program
(`model.mlir`), and its `weights.safetensors`. Bundles are produced offline by
`ReactantServerExport`, which is not a workspace member (it carries Lux/PythonCall weak
dependencies that the server image should not), so use its ready test environment:

```
julia --project=packages/ReactantServerExport/test
```

Export a tiny Lux MLP. The batch dimension is the last Julia axis (the Lux convention), so a
4-feature input is `(4, batch)`:

```julia
using Lux, ReactantServerExport, Random

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)

example = randn(Float32, 4, 1)            # (features, batch)
ReactantServerExport.export_bundle(:lux, model, ps, st, example;
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1])
```

This writes `models/mlp/`: a `manifest.yaml`, the compiled StableHLO module (one per batch size,
here `model.b1.mlir`), and `weights.safetensors`. The input tensor is named `input` and the
output `output` by default (override with `input_name` / `output_name`); the client snippets
below use those names. `models/` is now a model repository: every immediate subdirectory with a
`manifest.yaml` is a servable model, keyed by its directory name (`mlp` here). See
[Bundles & model.jl](bundles.md) for the manifest format and custom pre/post-processing.

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
`ReactantServer.serve` (which expects the workers list). Under the supervisor you can instead
omit `workers:` and write `gpus: auto`, and it synthesizes one worker per detected GPU; that is
how the image's baked default and [Scaling to Multiple GPUs](scaling.md) work. See
[Node Configuration](node_config.md) for the full surface (scheduler, on-demand weights,
per-model pinning, environment overrides).

## Step 3: Run it with docker compose

Build the image and start the node, pointing `REACTANTSERVER_MODELS` at the repository from
Step 1:

```
make image
REACTANTSERVER_MODELS=$PWD/models docker compose up
```

With a single GPU the node runs one worker and **no gateway**: the worker serves the KServe V2
gRPC API on `localhost:8001` and metrics/health on `localhost:8002` (`/readyz`, `/healthz`,
`/metrics`). The first start compiles every model before accepting traffic, so give it a moment;
`curl localhost:8002/readyz` returns 200 once it is serving. See [Docker Deployment](docker.md)
for the image, healthcheck, and metrics details.

## Step 4: Or run it from pure Julia

Two entry points, differing only in which ports are exposed:

```julia
# The supervisor: same behavior as the container. One worker (no gateway) on the public
# ports 8001 (gRPC) and 8002 (metrics), just like `docker compose up`.
using ReactantServerNode
ReactantServerNode.supervise("node.yaml")
```

```julia
# A single bare worker, no supervisor: serves on the node file's own port (base_port, 8080).
using ReactantServer
ReactantServer.serve("node.yaml")          # blocks; Ctrl-C to stop
```

[`serve`](@ref) loads the node file, brings up the runtime, compiles the worker's bundles, starts
the [`Scheduler`](@ref), and finally starts the gRPC server so traffic is accepted only once
models are live. Pass `blocking=false` to get a [`RunningServer`](@ref) you can [`stop!`](@ref):

```julia
server = ReactantServer.serve("node.yaml"; blocking=false)
# ... issue requests ...
ReactantServer.stop!(server)
```

`supervise` is the right choice for deployment (it is what the container runs and it scales to
many GPUs unchanged); a bare `serve` is convenient for a quick single-worker REPL session.

Start Julia multithreaded so per-request `preprocess`/`postprocess` overlap the GPU execution
(`julia --threads=auto,1`: a default pool for the hooks plus one interactive thread for the GPU
dispatch loop). Set this yourself only for a bare `serve`; under the supervisor (the container
image) each worker is instead sized to its share of the host, `min(cores ÷ workers, 16)` threads
plus the interactive one, so co-located workers do not oversubscribe the CPU (see
[Scaling to Multiple GPUs](scaling.md)). With a single thread the server still works, just without
the overlap.

## Step 5: Query it

The server speaks KServe V2 over gRPC, so any Triton/KServe client works; this repository ships
the Reactant-free `ReactantServerClient`. Point it at the port your server is using (8001 for the
supervisor or compose, 8080 for a bare `serve`), and use the bundle's tensor names `input` /
`output`:

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

See [Client Usage](client_usage.md) for batched inference over a dataset, IO validation, and the
shared-memory data path.

## Next steps

- [Common Use Cases](common_use_cases.md): choose a deployment shape (single GPU, multi-GPU
  distributed or replicated, multi-node) with an example configuration for each.
- [Scaling to Multiple GPUs](scaling.md): add GPUs and how the supervisor decides what to start.
- [Node Configuration](node_config.md): the full config surface and environment overrides.
- [Bundles & model.jl](bundles.md): the bundle format and custom pre/post-processing.
- [On-demand Weights](on_demand_weights.md): serving more models than fit in GPU memory.
- [Docker Deployment](docker.md): the image, roles, health, and metrics.

## Testing

Each package is tested in its own environment; all tests run on CPU and need no GPU:

```
julia --project=packages/ReactantServerCore   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer        -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient  -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerNode    -e 'using Pkg; Pkg.test()'
```
