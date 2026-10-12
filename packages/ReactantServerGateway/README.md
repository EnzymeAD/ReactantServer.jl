# ReactantServerGateway.jl

The pure-Julia KServe V2 gRPC **reverse proxy** that fronts a multi-GPU cluster of
[`ReactantServer`](../ReactantServer) workers behind one endpoint. It reads model name from each
`ModelInferRequest` and forwards the raw protobuf bytes to the worker that hosts the model, with
placement and load balancing across the workers that serve it (`round_robin`,
`least_outstanding`, or `lpt_packing`), failover, and fans shared-memory register/unregister out to all
workers.

It builds only on [`ReactantServerCore`](../ReactantServerCore) and the gRPC/HTTP layer, so it
carries **no Reactant dependency** and deploys as a small, fast-starting image.

Public entry points: `serve_gateway`, `probe_worker_ready`.

Part of the [ReactantServer.jl](../../README.md) workspace. The
[Front Workers with a Gateway](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/julia/gateway/) tutorial runs it in
front of two workers; [Multi-GPU Gateway](https://enzymead.github.io/ReactantServer.jl/dev/gateway/) is the reference for what the gateway does,
its scheduling modes, and its `gateway.yml` configuration.
