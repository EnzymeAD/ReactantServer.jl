# Multi-GPU Gateway

The gateway is a gRPC reverse proxy that fronts several ReactantServer.jl workers behind one
KServe V2 gRPC endpoint. It is pure Julia, lives in its own package `ReactantServerGateway`
(`ReactantServerGateway.serve_gateway`), and reuses `ReactantServerCore`'s node/config parsing
and the generated KServe protobuf. Because it builds only on `ReactantServerCore` and the gRPC
layer, the gateway carries no Reactant dependency.

You do not start the gateway yourself. When a node has two or more workers, the supervisor
([`ReactantServerNode.supervise`](@ref ReactantServerNode.supervise), the container's default
entry point) runs the gateway as an embedded child and synthesizes its worker endpoint list from
the node file; a single-worker node skips it entirely (one worker already serves the full KServe
V2 API). This page describes what that embedded gateway does. See
[Scaling to Multiple GPUs](scaling.md) for when it appears and [Docker Deployment](@ref) for the
container.

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
  workers, either uniformly (`round_robin`, the default) or by packing each model onto a fixed,
  operator-configured number of GPUs with coalescing-aware routing (`lpt_packing`); see
  [Scheduling modes](#scheduling-modes) below. Either way, a request fails over to the remaining
  replicas when a worker returns `NotFound` or `Unavailable`.
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

For guidance on choosing `round_robin` versus `lpt_packing` and setting replica counts for your
situation, with an example configuration for each shape, see
[Common Use Cases](common_use_cases.md).

The gateway routes each model's requests across its replicas according to `scheduling.mode` in
`gateway.yml`:

```yaml
scheduling:
  mode: lpt_packing             # round_robin (default) | least_outstanding | lpt_packing
  rebalance_compute_seconds: 30 # fleet GPU-seconds consumed that triggers a repack
  min_rebalance_seconds: 0      # wall-clock floor between repacks (0 = none)
  rate_halflife_seconds: 30
  hysteresis: 0.1               # minimum improvement before a model moves workers
  default_replicas: 1           # GPUs per model unless overridden below (a number, or "all")
  routing_fill_factor: 1.0      # per-replica fill target as a multiple of max batch size (lpt_packing only)
  routing_policy: fill_rr       # fill_rr (default) | fill_least  (lpt_packing only)
  compaction_mode: off          # off (default) | eager | scheduled  (defragment workers after a repack)
  compaction_interval: 0        # repacks between compactions; 0 disables  (see On-demand Weights)
  models:
    big-model:
      replicas: 2               # this model is placed on 2 distinct GPUs (a number, or "all")
```

**`round_robin`** (the default) spreads each model's requests uniformly across its replicas.
It is fully predictable from the config file and needs no measurements, at the cost of thin
per-worker queues: when every model is on every worker, each worker sees a slice of every
model's traffic, so coalesced batches rarely fill.

**`least_outstanding`** sends each request to the replica with the fewest in-flight requests,
spreading by live occupancy rather than blindly. Like `round_robin` it needs no measurements and no
preconditions and does not concentrate traffic, so it favors even spreading over batch coalescing;
prefer it over `round_robin` when a model's replicas have uneven or unpredictable per-request
latency, so a slow replica stops attracting new work instead of accumulating a backlog.

**`lpt_packing`** places each model on a fixed number of distinct GPUs and routes its requests
to preserve batch fill. A model's replica count is operator-controlled: `default_replicas`
(default 1, the single-GPU case that coalesces best), overridable per model under
`scheduling.models.<name>.replicas`. Both accept a positive integer or `all`, which places the
model on every ready worker (so `default_replicas: all` replicates the whole model set across all
GPUs without listing each model, and tracks the fleet as workers come and go). The count is set at
startup and never grows automatically under load; a hot model relies on its worker's queue and
coalescing rather than fanning out.

!!! warning "Replication is the operator's responsibility"
    The gateway does not check that a replica count is feasible for your hardware. Replicating a
    model charges its full weight footprint to every GPU it lands on, so `replicas: 2` (or
    `default_replicas: all`) only makes sense when those weights actually fit on each card
    alongside everything else placed there. If the assigned footprint exceeds a worker's
    on-demand weight budget, the weights cannot all stay resident and the worker thrashes,
    loading and evicting weights on nearly every request, which destroys throughput. Size replica
    counts against your GPU memory. The gateway logs a `weight footprint exceeds the worker's
    on-demand budget` warning at each repack when a placement is oversubscribed, so watch for it. The
packer chooses which GPUs host each model's replicas by balancing two live measurements: compute
demand (the gateway-measured arrival rate times the true per-request compute cost the workers
report over the control plane) and resident weight footprint against each worker's weight-memory
budget, placing models heaviest-first onto the least pressured workers, where pressure is
whichever of compute or memory is closer to full. Packing by memory keeps each GPU's resident
weight set bounded so evictions stay rare. Placements are sticky: a single-replica model moves
only when the move improves its resulting pressure by more than `hysteresis`, because batching
depends on traffic staying where the queues are. (`max_worker_share` is accepted but advisory
only; load no longer determines a model's GPU count.)

Repacks are driven by accumulated compute, not wall-clock: the gateway polls the workers every
probe round and recomputes the placement once the fleet has consumed `rebalance_compute_seconds`
GPU-seconds since the last repack, subject to the `min_rebalance_seconds` wall-clock floor. An
idle fleet does not repack until traffic resumes.

For a model with more than one replica, the gateway routes to fill one replica's batch before
moving to the next, so the workers receive favorable groupings to coalesce (the coalescing itself
stays at the worker). It tracks the in-flight request count per replica and keeps sending a model's
requests to the replica it is currently filling until that replica holds about `routing_fill_factor`
times the model's max batch size, then opens a fresh batch on another replica. Set
`routing_fill_factor` above 1.0 to keep the next batch queued so a worker does not go idle between
dispatches.

`routing_policy` (lpt_packing only) decides only *which* replica a fresh batch opens on (both
variants preserve the fill-one-replica-first behavior above; they differ only at the batch
boundary):

- **`fill_rr`** (default) round-robins the opening replica across the model's set, so successive
  batches of the same model spread evenly over its GPUs.
- **`fill_least`** opens each batch on the replica whose GPU currently carries the least in-flight
  compute load, measured across *all* models as in-flight requests weighted by each model's
  measured per-request compute cost. Prefer this when replicas share GPUs with other models, so a
  model's batches open on whichever of its GPUs is least busy rather than always the same one.

Spreading every request without concentrating it is the separate `least_outstanding` scheduling
mode above, not a routing policy.

A single-replica model is the degenerate case: all its requests go to its one GPU (and still count
toward that GPU's load for the `fill_least` decisions of models that share it).

`lpt_packing` has two preconditions, verified at gateway startup: every worker must run the `fifo`
scheduler discipline (placement decisions move to the gateway, so workers should not re-order
against it; see `scheduler.discipline` in [Node Configuration](node_config.md)), and every worker
must serve the identical model set. Because a worker compiles and warms up every model before its
control plane answers, the workers are usually not up when the gateway starts, so the gateway waits
for all of them before serving rather than failing, logging which workers are still pending. Under
the node supervisor (the embedded gateway) this wait is enabled automatically; for a standalone
gateway set `REACTANT_GATEWAY_STARTUP_WAIT_SECONDS` (a number of seconds, or `inf` to wait
indefinitely; the default `0` fails fast). Each poll is watchdog-bounded, and if the gRPC client
stack wedges during the long warmup (a known libcurl failure mode) the gateway exits so the
supervisor restarts it with a fresh stack; this self-heals and you may see one such restart before
it serves. Once all workers are up, a wrong discipline or differing model set is a hard error. Runtime drift after startup degrades gracefully: a model temporarily
missing from some workers is routed uniformly over its actual replicas with a warning until the
fleet converges, and a worker that drops out is excluded from placement, its traffic failing over
to the remaining replicas.

The placement is observable: `gateway_model_replicas` reports each model's replica count,
`gateway_placement_weight` reports its per-worker weight, `gateway_replica_outstanding` reports
the in-flight requests per replica sampled at the last repack, and `gateway_model_utilization`
reports its estimated demand in GPU-seconds per second.

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

## Configuration

The supervisor configures the embedded gateway for you: it synthesizes the worker endpoint list
(and the worker metrics list) from the node file into `REACTANT_GATEWAY_WORKERS` /
`REACTANT_GATEWAY_WORKER_METRICS`, and binds the gateway to the public ports (8001/8002). Nothing
about model placement is configured on the gateway; it autodiscovers which models each worker
serves via `RepositoryIndex` and refreshes its routing table periodically.

To tune the gateway, mount a `docker/gateway.yml` over `/etc/reactantserver/gateway.yml` (or set
`REACTANT_GATEWAY_FILE`); it carries the gateway's own settings (listen addresses, message
limits, logging, and the `scheduling:` block above). Settings can also be overridden by
environment with the prefix `REACTANT_GATEWAY_` and the dotted path uppercased with underscores,
e.g. `REACTANT_GATEWAY_LOGGING_LEVEL=debug` or `REACTANT_GATEWAY_SCHEDULING_MODE=lpt_packing`.

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
- Under `lpt_packing`, the gateway polls the workers on every 10s probe round to refresh routing
  metadata and accumulate consumed compute, but recomputes the placement only once the fleet has
  consumed `scheduling.rebalance_compute_seconds` GPU-seconds (subject to the
  `scheduling.min_rebalance_seconds` floor). Each repack logs a `lpt_packing: repack` line with the
  number of models placed, how many `moved` workers, how many were `held_by_hysteresis`, the largest
  available `max_improvement` against the `hysteresis` threshold, and the `compute_seconds`/
  `wall_seconds` since the last repack — useful for watching placement churn and the trigger cadence.
- If a probe to a worker hangs (times out, rather than failing fast), the gateway drops and
  recreates that worker's gRPC connection before the next attempt. This recovers from a half-open
  connection (e.g. caught during a worker's brief silent-accept window at startup) that would
  otherwise be reused and stall every later request to that worker — the per-worker equivalent of a
  restart, without dropping HTTP/2 multiplexing for healthy workers.
- Successful `ModelInfer` requests are not logged (to keep the hot path quiet); worker errors and a
  model with no live replica are logged, and per-request latency and gRPC status are exported as
  Prometheus metrics. Logs contain no tensor data.
