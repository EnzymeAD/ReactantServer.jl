# ReactantServer.jl

The inference **worker**: it loads StableHLO model bundles, compiles them through Reactant/PJRT,
schedules and batches requests, and serves the KServe V2 gRPC inference API for one GPU. This is
the **only** package in the workspace that depends on Reactant.

Public entry points: `serve`, `serve_worker`, `stop!`, `register_model`. It re-exports the
shared API from [`ReactantServerCore`](../ReactantServerCore), so the substrate is reachable as
`ReactantServer.X`.

Part of the [ReactantServer.jl](../../README.md) workspace. To get started, follow the
[Serve a Model](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/julia/serve/) tutorial; the [Node
Configuration](https://enzymead.github.io/ReactantServer.jl/dev/node_config/) and
[Deployment](https://enzymead.github.io/ReactantServer.jl/dev/deployment/) pages cover how a worker
is configured and run.
