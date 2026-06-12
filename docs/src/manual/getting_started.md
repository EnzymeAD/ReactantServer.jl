```@meta
CurrentModule = ReactantServer
```

# Getting Started

This guide brings up a single worker, points a client at it, and shuts it down. A single GPU
is just a one-worker cluster, so no gateway is involved.

## Installation

ReactantServer is a Julia workspace of four packages under `packages/`, plus the non-member
`ReactantServerExport` for offline bundle export (see
[Architecture](../design/architecture.md)). It vendors its forked/unregistered
dependencies (Reactant, gRPCServer, gRPCClient, HTTP) as git submodules under `lib/` and wires
them in through the workspace `[sources]`. After cloning, populate the submodules and
instantiate the workspace (this resolves all four members against one shared manifest):

```
git submodule update --init --recursive
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To work in a single package, activate its project instead, e.g.
`julia --project=packages/ReactantServer`. The worker (`serve`) lives in `ReactantServer`; a
client uses `ReactantServerClient` (see [Client Usage](client_usage.md)); the multi-GPU proxy
(`serve_gateway`) lives in `ReactantServerGateway`.

The runtime is device agnostic. It defaults to CPU PJRT and selects CUDA through configuration,
with CPU fallback, so you can develop and test without a GPU.

## Starting a worker

Configuration is a single node file (see [Cluster Configuration](cluster_config.md)). The
shipped `docker/node.yaml` describes the layout. To serve:

```julia
using ReactantServer

# Single-worker node: the worker name may be omitted.
ReactantServer.serve("docker/node.yaml")

# Multi-worker node: name the worker this process should serve.
ReactantServer.serve("docker/node.yaml"; worker="worker0")
```

[`serve`](@ref) loads the configuration, brings up the runtime client, compiles the bundles
assigned to the worker, starts the [`Scheduler`](@ref), and finally starts the gRPC server so
traffic is accepted only once models and the scheduler are live. By default it blocks, serving
until the process is stopped.

## Running in the background

Pass `blocking=false` to get a [`RunningServer`](@ref) handle and keep control of the REPL.
Shut it down with [`stop!`](@ref):

```julia
server = ReactantServer.serve("docker/node.yaml"; blocking=false)
# ... issue requests against the configured host and port ...
ReactantServer.stop!(server)
```

The default backend is `ReactantServer.ReactantBackend()`, which performs real compilation and
execution. Any KServe V2 gRPC client can then call the server directly at the configured host
and port.

## Connecting a client

The server speaks the KServe V2 (Open Inference Protocol) gRPC API natively, so a standard
Triton or KServe gRPC client connects without changes. This repository ships one such client,
the Reactant-free `ReactantServerClient` package; see [Client Usage](client_usage.md) for a
worked example. Tensor data travels either inline (`raw_input_contents` / `raw_output_contents`)
or through the Triton-compatible system shared-memory extension.

### Shape convention

Shape declarations in `manifest.yaml` and the server's internal [`NamedTensor`](@ref) data use
the Julia column-major convention (the batch dimension is the last axis), which is the reverse
of how Python/XLA writes the same tensor. The codec advertises and accepts KServe V2 wire
shapes in their canonical row-major form, so Triton-style clients are unchanged; the underlying
tensor bytes are identical under either view, and the codec converts by reshaping rather than
permuting. See [Bundles & model.jl](bundles.md) for the einsum-style shape notation used in the
manifest.

## Testing

Each package is tested in its own environment; all tests run on CPU and need no GPU:

```
julia --project=packages/ReactantServerCore   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer        -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient  -e 'using Pkg; Pkg.test()'
```

## Next steps

- [Cluster Configuration](cluster_config.md): the full config surface and overrides.
- [Bundles & model.jl](bundles.md): how to package a model and add pre/post-processing.
- [On-demand Weights](on_demand_weights.md): serving more models than fit in GPU memory.
- [Multi-GPU Gateway](multi_gpu_gateway.md) and [Docker Deployment](docker.md): scaling out.
