# API reference

Every exported name that carries a docstring across the five documented packages appears below,
collected automatically. `ReactantServer` re-exports the whole `ReactantServerCore` substrate, so
the shared API (dtypes, protobuf messages, boundary types, manifest, config, codec, shared memory,
the staging `BufferPool`) is documented under its original bindings once. The guide pages give the
worked context: [Tutorial](tutorial.md), [Bundles](bundles.md),
[Node Configuration](node_config.md), [Scheduling](scheduling.md),
[On-demand Weights](on_demand_weights.md), [Multi-GPU Gateway](gateway.md),
[Client Usage](client.md), [Meta Models](meta_models.md),
[Object Detection](object_detection.md), [Transformer Text Models](transformers.md), and
[Deployment](deployment.md).

```@autodocs
Modules = [
    ReactantServer,
    ReactantServerClient,
    ReactantServerExport,
    ReactantServerGateway,
    ReactantServerNode,
    ReactantServerCore,
]
Public = true
Private = false
```

## Operational internals

A few names the guides and docstrings reach for are deliberately unexported; they are documented
here so their docstrings render and the qualified references in the guides are backed by the
manual. Reach them from the REPL fully qualified, for example `ReactantServer.scheduler_metrics()`.
They are implementation details and may change without a breaking release.

```@docs
ReactantServer.Scheduler
ReactantServer.infer
ReactantServer.scheduler_metrics
ReactantServer.set_policy!
ReactantServer.set_residency!
ReactantServer.control_status
ReactantServer.compact!
ReactantServer.acquire!
ReactantServer.NotResidentError
ReactantServer.RunningServer
ReactantServer.weight_budget
ReactantServer.weight_cache_stats
ReactantServer.free_weights!
ReactantServer.transfer_to_device
ReactantServerClient.RetryPolicy
ReactantServerGateway.RunningGateway
```
