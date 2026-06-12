```@meta
CurrentModule = ReactantServerGateway
```

# Gateway

`ReactantServerGateway` is the standalone KServe V2 gRPC reverse proxy that fronts a multi-GPU
cluster of `ReactantServer` workers. It depends only on `ReactantServerCore` and the gRPC/HTTP
layer, never on Reactant, so it deploys as a small, fast-starting image. See
[Multi-GPU Gateway](../manual/multi_gpu_gateway.md) for the operational view.

```@docs
serve_gateway
probe_worker_ready
RunningGateway
stop!
```
