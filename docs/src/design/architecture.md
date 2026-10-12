# Architecture

This page describes how the pieces of ReactantServer fit together: the packages, the processes
they run as, the path a request takes, and how weights reach a compiled program. Each component
has its own page for the detail, linked from the section that introduces it.

## Packages

The project is a Julia workspace of packages, split so that talking to a server never requires
the Reactant and XLA stack:

- **ReactantServerCore**: the shared, Reactant-free code the other packages build on: the KServe
  V2 messages and wire format, and the manifest and node file formats.
- **ReactantServer**: the worker, and the only package that loads Reactant.
- **ReactantServerGateway**: the reverse proxy that puts several workers behind one endpoint.
- **ReactantServerClient**: the client. It is Reactant-free, so it installs on any machine that
  sends requests.
- **ReactantServerNode**: the node supervisor, which runs a machine's workers and gateway as
  subprocesses.
- **ReactantServerExport**: offline export of Lux and PyTorch models into bundles. It is not a
  workspace member; the worker reads the bundles it writes, never the package.

The [API](../api.md) page lists every package's documented names.

## Processes

A worker drives one device. It serves the full KServe V2 surface on its own, so a single worker
needs nothing else.

A node with several devices runs one worker process per device, started and restarted by the node
supervisor. With two or more workers the supervisor also starts a gateway process, which gives
clients one endpoint and routes each model's requests to the workers that serve it. The gateway
forwards inference requests as raw bytes and never decodes a tensor.
[Deployment](../deployment.md) describes the supervisor, and [Multi-GPU Gateway](../gateway.md)
describes how the gateway places models and routes requests.

## A request's path

1. A client sends a KServe V2 `ModelInfer` request. Tensors travel inline in the request or, when
   client and worker share a host, in a shared-memory region the client registered beforehand
   ([Client Usage](../client.md)).
2. On a multi-worker node, the gateway picks a worker for the model and forwards the request.
3. The worker decodes the request on the request's own task and runs the bundle's `preprocess`
   hook there, if it has one, then queues the request with the scheduler.
4. A single dispatch loop picks the next model to run, coalesces that model's queued requests into
   one execution at a compiled batch size, and runs it on the device. The device runs one
   execution at a time ([Scheduling](../scheduling.md)).
5. Each request's task receives its slice of the output, runs the bundle's `postprocess` hook,
   and encodes the response.

Pre and postprocessing run on the request tasks rather than in the dispatch loop, so the CPU work
of many requests runs in parallel with the device execution.

A meta model replaces steps 3 to 5 with a Julia `run` hook that calls other models; each call
re-enters the scheduler as an ordinary request ([Meta Models](../meta_models.md)).

## Weights as arguments

A bundle holds one compiled program per batch size and one set of weights
([Bundles](../bundles.md)). The weights are passed to each program as arguments rather than
compiled into it. That is what lets every batch size share one copy of the weights, and lets a
worker move a model's weights onto the device when the model is needed and evict them when it is
not, without recompiling ([On-demand Weights](../on_demand_weights.md)).

## Model lifecycle

The set of models a worker serves can change while it runs. By default the worker watches the
model repository and loads, reloads, and unloads bundles as they change; the `model_control_mode`
key in [Node Configuration](../node_config.md) chooses between that, a fixed startup set, and
control by an external system over gRPC.

The project's goals, audience, and non-goals are on the [Philosophy](philosophy.md) page.
