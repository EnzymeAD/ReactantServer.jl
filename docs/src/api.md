# API reference

The exported API is documented one package per page, collected automatically from the docstrings:

- [Core](api_core.md): the shared substrate (dtypes, protobuf messages, boundary types, manifest,
  config, codec, shared memory, the weight store, the staging `BufferPool`). `ReactantServer`
  re-exports all of it, so it is documented once, under its original bindings.
- [Worker](api_worker.md): `ReactantServer`, the inference worker.
- [Gateway](api_gateway.md): `ReactantServerGateway`.
- [Node](api_node.md): `ReactantServerNode`, the single-container supervisor.
- [Client](api_client.md): `ReactantServerClient`.
- [Export](api_export.md): `ReactantServerExport`, the offline export tooling.

The guide pages give the worked context: [Tutorials](tutorials/export.md), [Bundles](bundles.md),
[Node Configuration](node_config.md), [Scheduling](scheduling.md),
[On-demand Weights](on_demand_weights.md), [Multi-GPU Gateway](gateway.md),
[Client Usage](client.md), [Meta Models](meta_models.md),
[Object Detection](object_detection.md), [Transformer Text Models](transformers.md), and
[Deployment](deployment.md).

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
ReactantServer.admit_weights!
ReactantServer.AdmissionError
ReactantServer.migrate_bundle_weights!
ReactantServer.verify_transposed
ReactantServer.resolve_weights_path
ReactantServer.NotResidentError
ReactantServer.QueueFullError
ReactantServer.WorkerUnavailableError
ReactantServer.WeightsReadStalledError
ReactantServer.ModelNotFoundError
ReactantServer.DispatchTimeoutError
ReactantServer.RequestCancelledError
ReactantServer.RunningServer
ReactantServer.weight_budget
ReactantServer.weight_cache_stats
ReactantServer.free_weights!
ReactantServer.transfer_to_device
ReactantServerClient.RetryPolicy
ReactantServerGateway.RunningGateway
```
