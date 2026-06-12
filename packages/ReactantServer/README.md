# ReactantServer.jl

The inference **worker**: it loads StableHLO model bundles, compiles them through Reactant/PJRT,
schedules and batches requests, and serves the KServe V2 gRPC control plane for one GPU. This is
the **only** package in the workspace that depends on Reactant.

Public entry points: `serve`, `serve_worker`, `stop!`, `register_model`. It re-exports the
shared API from [`ReactantServerCore`](../ReactantServerCore), so the substrate is reachable as
`ReactantServer.X`.

```julia
using ReactantServer
ReactantServer.serve("examples/node.yaml")                    # single worker
ReactantServer.serve("examples/node.yaml"; worker="worker0")  # multi-worker
```

Part of the [ReactantServer.jl](../../README.md) workspace. See the docs for
[Getting Started](../../docs/src/manual/getting_started.md).
