# Multi-GPU Gateway

`reactant-gateway` is a gRPC reverse proxy that fronts several ReactantServer.jl workers behind one
KServe V2 gRPC endpoint. It is pure Julia, lives in its own package `ReactantServerGateway`
(`ReactantServerGateway.serve_gateway`), and reuses `ReactantServerCore`'s cluster/config
parsing and the generated KServe protobuf. Because it builds only on `ReactantServerCore` and
the gRPC layer, the gateway carries no Reactant dependency.

In the standard single-node deployment you do not start the gateway yourself: the node
supervisor ([`ReactantServerNode.supervise`](@ref ReactantServerNode.supervise), the container's
default entry point) runs it as an embedded child and synthesizes its worker endpoint list from
the node file. This page describes what that gateway does and how to run it standalone, which is
needed only when workers and the gateway live on different hosts (see the `gateway` role in
[Docker Deployment](@ref)). A single worker already serves the full KServe V2 gRPC API for its
GPU, so a bare single-GPU deployment without the supervisor can also be addressed directly.

Clients connect to a single gRPC endpoint. The gateway extracts the model name from each
`ModelInferRequest` and forwards the raw protobuf bytes over gRPC to the worker that hosts that
model. The KServe V2 protobuf wire format is identical end to end; the gateway is a
gRPC-to-gRPC pass-through that never re-marshals the body.

## What the gateway does

- **Single endpoint:** clients reach all workers through one gRPC listener.
- **Model-name routing by autodiscovery:** the gateway is given a flat list of worker
  `endpoints:` in its own `gateway.yml` and queries each worker's `RepositoryIndex` RPC
  (every 10s) to learn which models it currently serves. The discovered model-to-workers
  routing table is rebuilt and swapped in atomically on each probe, so a control-plane
  pin/unpin or a worker restart flips routing on the next probe.
- **Replica scheduling:** a model served by more than one worker is load-balanced across those
  workers, either uniformly (`round_robin`, the default) or by adaptive placement
  (`lpt_packing`); see [Scheduling modes](#scheduling-modes) below. Either way, a request fails
  over to the remaining replicas when a worker returns `NotFound` or `Unavailable`.
- **Readiness probe:** a background loop calls each worker's KServe `ServerReady` RPC; `/readyz`
  is ready when at least one worker reports ready.
- **Raw passthrough:** the `ModelInfer` hot path never decodes or re-marshals the protobuf body.
  The request and response types are `Vector{UInt8}` end to end (gRPCServer.jl and gRPCClient.jl
  support raw byte messages natively). To route, the gateway decodes a partial schema that
  declares only `model_name` (field 1) and `id` (field 3); ProtoBuf skips the tensor payload.
- **SHM broadcast:** `SystemSharedMemoryRegister` / `Unregister` are fanned out to every worker.
  POSIX SHM regions are host-local; every worker attaches via `shm_open` independently. Register
  succeeds only if all workers succeed (it rolls back partial success); unregister succeeds if
  any worker does.
- **Observability:** structured logs, Prometheus metrics, `/healthz`, and `/readyz` on a
  separate admin HTTP port.

## Scheduling modes

The gateway routes each model's requests across its replicas according to `scheduling.mode` in
`gateway.yml`:

```yaml
scheduling:
  mode: lpt_packing        # round_robin (default) | lpt_packing
  rebalance_seconds: 15    # placement recomputation cadence
  rate_halflife_seconds: 30
  max_worker_share: 0.8    # cap on one worker's capacity a single model may claim
  hysteresis: 0.1          # minimum improvement before a model moves workers
```

**`round_robin`** (the default) spreads each model's requests uniformly across its replicas.
It is fully predictable from the config file and needs no measurements, at the cost of thin
per-worker queues: when every model is on every worker, each worker sees a slice of every
model's traffic, so coalesced batches rarely fill.

**`lpt_packing`** places models on workers automatically and adaptively. Each model gets a
sampling distribution over workers, recomputed every `rebalance_seconds` from two live
measurements: its compute demand (the gateway-measured arrival rate times the true per-request
compute cost the workers report over the control plane) and its resident weight footprint
against each worker's weight-memory budget. The packer places models heaviest-first, each
wholly on the least pressured worker, where pressure is whichever of compute or memory is
closer to full. Concentrating a model's traffic on one worker is what lets the worker's batch
coalescing fill compiled batch sizes, and packing by memory keeps each GPU's resident weight
set bounded so evictions stay rare.

Three safeguards apply. A model whose demand exceeds `max_worker_share` of one GPU is split
evenly across the minimum number of workers that brings each share under the cap, so a hot
model cannot starve its neighbors. Placements are sticky: a model moves only when the move
improves its resulting pressure by more than `hysteresis`, because batching depends on traffic
staying where the queues are. And a worker that drops out is excluded from placement, its
traffic failing over to the remaining replicas immediately.

`lpt_packing` has two preconditions, verified as a hard failure at gateway startup: every
worker must run the `fifo` scheduler discipline (placement and fairness decisions move to the
gateway, so workers should not re-order against it; see `scheduler.discipline` in
[Cluster Configuration](cluster_config.md)), and every worker must serve the identical model
set. Runtime drift degrades gracefully: a model temporarily missing from some workers is
routed uniformly over its actual replicas with a warning until the fleet converges.

The placement is observable: `gateway_placement_weight` reports each model's current sampling
weight per worker, and `gateway_model_utilization` reports its estimated demand in GPU-seconds
per second.

## What the gateway does not do

- Streaming RPCs.
- The repository / model-config / statistics / trace / log RPCs in the Triton spec, plus
  `ServerLive`, `ServerReady`, `ModelMetadata`, and `RepositoryIndex` for clients (only
  `ModelInfer` and the two SHM RPCs are proxied; everything else returns `UNIMPLEMENTED`).
- TLS: parsed but not yet enforced; the listener and the worker back-hop are cleartext h2c.
- CUDA shared memory.
- Dynamic worker membership: the worker endpoint list is fixed at startup (from `gateway.yml`
  or `REACTANT_GATEWAY_WORKERS`). Which models each worker serves is rediscovered continuously,
  but adding or removing workers requires a gateway restart.

## Build

The standalone gateway image is built from the repository root (podman by default; Docker works
too):

```
make gateway    # build the slim reactant-gateway image
```

The image is produced by `docker/Dockerfile.gateway` (see [Docker Deployment](docker.md)). It is
built from the `ReactantServerGateway` member alone, so it pulls no Reactant/XLA stack. The
unified node image (`make image`) also contains the gateway and runs it in the `gateway` role.

## Run

The supervisor starts the embedded gateway for you. To run a gateway standalone (the multi-node
case), start the Julia workers first (one per GPU, each pointed at the same node file with a
distinct `worker`, or a worker-role node container per host). Then run the gateway against its
own `gateway.yml`:

```julia
using ReactantServerGateway
ReactantServerGateway.serve_gateway("docker/gateway.yml")
```

The gateway is decoupled from the node files: `gateway.yml` (see `docker/gateway.yml` and
[Cluster Configuration](cluster_config.md)) carries only the gateway's own settings plus a flat
`endpoints:` list of worker `host:port` addresses, which may span any number of nodes. The
gateway autodiscovers which models each endpoint serves via `RepositoryIndex`; nothing about
model placement is configured on the gateway.

Environment variables override the resolved config using the prefix `REACTANT_GATEWAY_` and the
dotted path uppercased with underscores, for example `REACTANT_GATEWAY_LOGGING_LEVEL=debug` or
`REACTANT_GATEWAY_LISTEN_GRPC=0.0.0.0:8001`. `REACTANT_GATEWAY_WORKERS` (comma separated)
replaces the endpoint list.

## Operational notes

- The gateway is a single point of failure. Each Julia worker stays reachable on its own
  KServe V2 gRPC endpoint during a gateway outage, so a client can fall back to addressing a
  worker directly.
- The routing table is rebuilt every 10s from each worker's `RepositoryIndex` and swapped in
  atomically. If a worker dies, its routes persist until the next successful probe (up to
  ~10s); in the gap, requests to its models fail over to the remaining replicas (on `NotFound`
  or `Unavailable`), and a model with no live replica returns `NotFound`. The worker-side
  readiness probe (`ServerReady`, same 10s loop) drives `/readyz` and the
  `gateway_worker_ready` metric.
- Under `lpt_packing`, placement rebalancing runs on its own cadence
  (`scheduling.rebalance_seconds`, default 15s), separate from the 10s discovery and readiness
  probes.
- Every `ModelInfer` is logged with the model name, the client-supplied request id (KServe `id`
  field), worker URL, request and response byte counts, worker latency, and gRPC status. Logs
  contain no tensor data.
