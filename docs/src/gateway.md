# Multi-GPU Gateway

The gateway is a gRPC reverse proxy that fronts several ReactantServer.jl workers behind one
KServe V2 gRPC endpoint. It is pure Julia, lives in its own package `ReactantServerGateway`
([`serve_gateway`](@ref)), and reuses `ReactantServerCore`'s node and config parsing plus the
generated KServe protobuf. Because it builds only on `ReactantServerCore` and the gRPC layer,
the gateway carries no Reactant dependency. The documented surface, [`serve_gateway`](@ref) and
[`probe_worker_ready`](@ref) plus the `ReactantServerGateway.RunningGateway` handle
`serve_gateway(blocking=false)` returns, is collected on the [API](api.md) page.

You do not start the gateway yourself. When a node has two or more workers, the supervisor
([`ReactantServerNode.supervise`](@ref), the node's default entry point) runs the gateway as an
embedded child and synthesizes its worker endpoint list from the node file; a single-worker node
skips it entirely, because one worker already serves the full KServe V2 API. See
[Deployment](deployment.md) for the node; this page describes what its embedded gateway does.

Clients connect to a single gRPC endpoint. The gateway extracts the model name from each
`ModelInferRequest` and forwards the raw protobuf bytes over gRPC to the worker that hosts that
model. The KServe V2 protobuf wire format is identical end to end, so the gateway is a
gRPC-to-gRPC pass-through that never re-marshals the body.

## One endpoint whether you run 1 GPU or 8

The key scaling fact: nothing about your model, client, or external interface changes as you add
GPUs. You run the same launcher (or the same `supervise` call), the supervisor puts one worker
on each GPU, and it fronts them with the embedded gateway, so clients still send one request to
one endpoint and it is routed to a worker that serves the model. `gpus: auto` in the node file
already means "one worker per visible GPU", so the single-GPU node file scales as-is; you only
edit it to do something non-default (see [Node Configuration](node_config.md)).

The supervisor decides whether a gateway exists at all: a single-worker node serves the full
KServe V2 API directly on the public ports with no gateway, and a node with two or more workers
embeds the gateway in front of them, each worker binding `base_port + i` and the gateway binding
the public 8001/8002. The supervisor synthesizes the gateway's worker list from the node file,
so the gateway needs no configuration of its own; [Deployment](deployment.md) owns that decision.
The external interface (8001 for gRPC, 8002 for metrics/health) is therefore identical whether
you run 1 GPU or 8, which is why your client and the public ports never change as you scale; see
[Client Usage](client.md) for the client side. The `REACTANT_ROLE` environment variable can
split the default `all` role into `workers` only or `gateway` only for deployments that span
machines; that topology is beyond the scope of these guides.

One scrape on 8002 still covers the whole node: the embedded gateway serves its own `gateway_*`
series and merges in every worker's `/metrics`. Each worker tags its series with `worker` and
`gpu` labels itself (the `gpu` value is the physical device behind its `CUDA_VISIBLE_DEVICES`),
so per-GPU and per-worker breakdowns need no Prometheus relabeling, for example
`sum by (gpu) (rate(worker_dispatch_total[1m]))`. See [Deployment](deployment.md) for the scrape
config.

## What the gateway does

- **Single endpoint:** clients reach all workers through one gRPC listener.
- **Model-name routing by autodiscovery:** the gateway is given a flat list of worker
  `endpoints:` in its own `gateway.yml` and queries each worker's `RepositoryIndex` RPC (every
  10s) to learn which models it currently serves. The discovered model-to-workers routing table
  is rebuilt and swapped in atomically on each probe, so a control-plane pin/unpin or a worker
  restart flips routing on the next probe.
- **Replica scheduling:** a model served by more than one worker is load-balanced across those
  workers, either uniformly (`round_robin`, the default) or by packing each model onto a fixed,
  operator-configured number of GPUs with coalescing-aware routing (`lpt_packing`); see
  [Scheduling modes](#scheduling-modes) below. A third mode, `least_outstanding`, spreads by
  live occupancy. Either way, a request fails over to the remaining replicas when a worker
  returns `NotFound` or `Unavailable`.
- **Readiness probe:** a background loop calls each worker's KServe `ServerReady` RPC; `/readyz`
  is ready when at least one worker reports ready.
- **Raw passthrough:** the `ModelInfer` hot path never decodes or re-marshals the protobuf body.
  The request and response types are `Vector{UInt8}` end to end (gRPCServer.jl and gRPCClient.jl
  support raw byte messages natively). To route, the gateway decodes a partial schema declaring
  only `model_name` (field 1) and `id` (field 3). Under `lpt_packing` a second partial decode
  reads one named tensor's shape to size the request in items (see the fill quantum below).
  ProtoBuf seeks past everything else, including every tensor payload; shapes precede the data on
  the wire, so neither decode touches it. Requests from this repo's client make that property
  stronger: their inputs ship as `raw_input_contents`, so every `InferInputTensor` descriptor
  stays contiguous in field 5 and the whole payload moves to field 7, letting a prefix scan read
  every input's name, datatype, and shape. That is a property of these requests, not a protocol
  guarantee: a Triton client may still inline typed contents, and the gateway keeps handling
  that.
- **SHM broadcast:** `SystemSharedMemoryRegister` / `Unregister` are fanned out to every worker.
  POSIX SHM regions are host-local; every worker attaches via `shm_open` independently. Register
  succeeds only if all workers succeed (it rolls back partial success); unregister succeeds if
  any worker does.
- **SHM namespace probe:** `IsSameIPCNamespace` is fanned out to every worker and the gateway
  returns `true` only if all of them can see the client's region (any worker may service a later
  `ModelInfer`). A worker that errors or does not implement the RPC counts as `false`.
- **Observability:** structured logs, Prometheus metrics, `/healthz`, and `/readyz` on a
  separate admin HTTP port.

## Scheduling modes

The gateway routes each model's requests across its replicas according to `scheduling.mode` in
`gateway.yml`. `round_robin` is the predictable default and needs no measurements;
`least_outstanding` spreads by live occupancy; `lpt_packing` concentrates each model's traffic
on few GPUs for deep coalescing batches and is the mode whose placement the gateway manages.
Every setting in the block below except `mode` can also be changed while the gateway runs,
through its own gRPC control plane; see
[Runtime scheduling control](#runtime-scheduling-control).

```yaml
scheduling:
  mode: lpt_packing             # round_robin (default) | least_outstanding | lpt_packing
  work_basis: compute           # compute (default) | items | requests  (what "least busy" measures,
                                #   for least_outstanding and for fill_least; see below)
  rebalance_compute_seconds: 300 # fleet GPU-seconds consumed that triggers a repack
  first_rebalance_compute_seconds: 60 # smaller budget for the first repack (0 = use rebalance_compute_seconds)
  ema_halflife_compute_seconds: 0 # demand-EMA halflife in fleet compute-seconds (0 = track rebalance_compute_seconds)
  hysteresis: 0.0               # extra improvement required before a model moves workers (0 = move on any gain)
  default_replicas: 1           # GPUs per model unless overridden below (a number, or "all")
  routing_fill_factor: 1.0      # fill quantum as a multiple of max batch size (lpt_packing only)
  routing_policy: fill_least    # fill_least (default) | fill_rr  (lpt_packing only)
  routing_fill_mode: run        # run (default) | spread | inflight  (what the quantum counts;
                                #   `inflight` can park a model on one GPU, see the warning below)
  forbid_memory_oversubscription: true # never strand a model on-demand when it could fit resident (default on)
  compaction_mode: eager        # eager (default) | off | scheduled  (defragment workers after a repack)
  compaction_interval: 1        # repacks between compactions; 0 disables  (see On-demand Weights)
  models:
    big-model:
      replicas: 2               # this model is placed on 2 distinct GPUs (a number, or "all")
      fill_mode: spread         # optional per-model override of routing_fill_mode
      fill_factor: 0.5          # optional per-model override of routing_fill_factor
```

### round_robin

`round_robin` (the default) spreads each model's requests uniformly across its replicas. It is
fully predictable from the config file and needs no measurements, at the cost of thin per-worker
queues: when every model is on every worker, each worker sees a slice of every model's traffic,
so coalesced batches rarely fill.

### least_outstanding and work_basis

`least_outstanding` sends each request to the replica carrying the least in-flight work, spreading
by live occupancy rather than blindly. Like `round_robin` it needs no preconditions and does not
concentrate traffic, so it favors even spreading over batch coalescing; prefer it over
`round_robin` when a model's replicas have uneven or unpredictable per-request latency, so a slow
replica stops attracting new work instead of accumulating a backlog.

`scheduling.work_basis` decides what "work" means. It is fleet-wide: the same setting denominates
`least_outstanding`'s per-replica counter and `lpt_packing`'s `fill_least` per-worker load, so
there is one answer to "what does this fleet call busy".

- **`compute`** (the default) counts in-flight GPU-seconds: each request charges its item count
  times the model's worker-measured cost per item. This is the only basis that is comparable
  across models, where 32 items of a cheap model are not 32 items of an expensive one.
- **`items`** counts in-flight items, so a client that pre-batches is weighed by how much it
  sent. Correct without any cost measurement when a fleet's models cost about the same per item.
  Weakest under `fill_least`, which compares one worker's load across *all* models, so dropping
  the cost weighting makes a cheap model's items look as expensive as a costly model's.
- **`requests`** is the per-request denominator each mechanism used before this knob existed: a
  raw count for `least_outstanding`, and the model's measured cost per *request* for `fill_least`
  (which keeps its cross-model cost weighting). Blind to batch size either way, so a pre-batching
  client defeats the balancing. Set this to pin the pre-knob behavior exactly.

`compute` and `items` need to know where each model's batch axis is and what an item costs, which
the gateway learns from a `ModelControlStatus` poll on each health-probe round. Under
`least_outstanding` that is one extra RPC per worker per tick (`requests` polls nothing);
`lpt_packing` already polls.

Under `lpt_packing` the basis is also runtime-mutable, like every other knob in the block: change
it with `tools/gateway_ctl.jl set-policy work_basis=items` and read it back from
`GetSchedulingStatus`, so a fleet can be switched and rolled back without a restart. Under
`least_outstanding` it is fixed at startup, and the startup log line states which basis is in
force.

### Models with no batch axis, and metas

Both work bases degrade to one unit per request rather than misreporting, so no model ever routes
on zeros:

- **A model that declares no batch axis** (`wire_batch_spec` finds none: an unbatched bundle, a
  model whose inputs are all fixed-shape) counts as one item per request. Its leading dimension
  names an axis, not a count, so there is nothing to read; under `compute` it still charges its
  own measured cost, which for such a model is a cost per request.
- **A model the gateway has not polled yet** counts as one item until the first successful poll,
  and under `compute` borrows the fleet-mean cost per item, so a cold model weighs like an
  average one instead of looking free and attracting every request.
- **A meta** (`kind: meta`) deliberately reports no batch axis and counts one row per request,
  even when its own `client_inputs` declare a batch axis. The two halves go together: rows equal
  requests, so a meta's cost per item equals its cost per request, and a work-routing gateway
  charges a meta its whole measured cost. Reporting an axis without also counting rows would
  charge N times the per-request cost; counting rows without reporting an axis would divide the
  cost by N and then charge one of them.

A meta is also the clearest argument for `compute` over `items`. Its work is data-dependent (a
detection meta's ROI count varies per image, and its Julia glue is not GPU time at all), so no
item count predicts it, while under `items` the most expensive thing on the fleet would charge
one unit and a cheap batched model would charge 32. Note that a meta's `total_compute` counts its
sub-call GPU time only, and its sub-models are folded into it and hidden from the gateway, so
that time is counted once rather than double counted.

### lpt_packing

`lpt_packing` places each model on a fixed number of distinct GPUs and routes its requests to
preserve batch fill. A model's replica count is operator-controlled: `default_replicas` (default
1, the single-GPU case that coalesces best), overridable per model under
`scheduling.models.<name>.replicas`. Both accept a positive integer or `all`, which places the
model on every ready worker (so `default_replicas: all` replicates the whole model set across all
GPUs without listing each model, and tracks the fleet as workers come and go). The count never
grows *automatically* under load, so a hot model relies on its worker's queue and coalescing
rather than fanning out on its own; an operator can promote one at any time with
`SetModelPlacement`, effective at the next repack.

!!! warning "Replication is the operator's responsibility"
    The gateway does not check that a replica count is feasible for your hardware. Replicating a
    model charges its full weight footprint to every GPU it lands on, so `replicas: 2` (or
    `default_replicas: all`) only makes sense when those weights actually fit on each card
    alongside everything else placed there. If the assigned footprint exceeds a worker's
    on-demand weight budget, the weights cannot all stay resident and the worker thrashes,
    loading and evicting weights on nearly every request, which destroys throughput. Size replica
    counts against your GPU memory. The gateway logs a `weight footprint exceeds the worker's
    on-demand budget` warning at each repack when a placement is oversubscribed, so watch for it.

The packer chooses which GPUs host each model's replicas by balancing two live measurements:
compute demand (the gateway-measured arrival rate times the true per-request compute cost the
workers report over the control plane) and resident weight footprint against each worker's
weight-memory budget, placing models heaviest-first onto the least pressured workers, where
pressure is whichever of compute or memory is closer to full. Packing by memory keeps each GPU's
resident weight set bounded so evictions stay rare. With `forbid_memory_oversubscription` enabled
(the default), this becomes a hard guarantee: a model is placed only on a worker where its
weights still fit whenever some worker can hold it, so a feasible fully-resident placement is
never passed over for one that strands a model on-demand (the worker LRU evicting a placed
model). The packer falls back to the unconstrained choice, and logs the oversubscription warning
above, only when the weights genuinely exceed every worker's budget.

Placement is kept stable mainly by smoothing the demand signal rather than by a large dead band:
the arrival rate and per-request cost are each tracked with an exponential moving average whose
halflife is measured in fleet compute-seconds (`ema_halflife_compute_seconds`, defaulting to one
`rebalance_compute_seconds` interval, so the signal decays about half per repack and ages with
how busy the fleet is rather than with wall-clock). `hysteresis` adds an optional dead band on
top: a single-replica model moves only when the move improves its resulting pressure by more than
`hysteresis` (default `0.0`, i.e. move on any improvement), because batching depends on traffic
staying where the queues are. (`max_worker_share` is accepted but advisory only; load no longer
determines a model's GPU count.)

Repacks are driven by accumulated compute, not wall-clock: the gateway polls the workers every
probe round and recomputes the placement once the fleet has consumed `rebalance_compute_seconds`
GPU-seconds since the last repack. This compute is **cumulative across every GPU**: the gateway
sums each worker's consumed GPU-seconds, so a fleet of N busy GPUs accrues the budget about N
times as fast as one (the accrual rate is the sum of the per-GPU utilizations). The budget is
therefore in GPU-work, not real time, and the wall-clock gap between repacks shrinks as the fleet
gets busier or larger and stretches out when it is idle (`rebalance_compute_seconds` divided by
the sum of utilizations). The same cumulative clock drives `ema_halflife_compute_seconds`, so the
demand signal also ages in GPU-work rather than wall-clock. The first traffic-driven repack can
use a smaller `first_rebalance_compute_seconds` budget so an early rebalance corrects the cold
placement quickly, then later repacks use the larger steady-state budget to limit memory churn
(`0` means the first repack uses `rebalance_compute_seconds` like the rest). An idle fleet does
not repack until traffic resumes. Gateway-driven compaction of worker memory follows placement
changes; see [On-demand Weights](on_demand_weights.md) for `compaction_mode` and
`compaction_interval`.

#### The fill quantum

For a model with more than one replica, the gateway concentrates its requests on one replica at a
time so the workers receive deep same-model groupings to coalesce (the coalescing itself stays at
the worker; see [Scheduling](scheduling.md)). The **fill quantum** `Q` is the model's
worker-reported max batch size scaled by `routing_fill_factor`, and it is denominated in
**items**, not requests. One request carrying a batch of 32 and 32 requests carrying one item
each are the same amount of work, and both spend a quantum of 32.

The gateway does not guess where a request's items are counted. Each worker reports, per model,
the **name** of the wire-facing input carrying the batch axis and that tensor's 1-based axis,
taken from the bundle manifest's reserved `n`/`b` shape marker. The gateway matches that name
against the request's own name-addressed inputs and reads that one dimension, without decoding
the payload. This has to come from the manifest: the axis is per tensor and varies between
bundles (an image model's `whcn` puts it last, a tokenized `nc` puts it first), the first input
need not carry one at all (a cross-encoder's `query` is unbatched while its `keys` are batched),
and it has moved between exports of the same model. A model that declares no batch axis counts
one item per request.

That matters most for a client that batches on its own behalf. If your client already fills
requests to the model's max batch size, a request-denominated quantum would hold one replica for
32 *batches* and your bursts would land on a single GPU; the item denomination makes one such
request close the run. Models the worker does not batch (a meta, or a model compiled unbatched,
both reporting a max batch of 0 or 1) charge one unit per request, since their leading dimension
names an axis rather than a count.

`routing_fill_mode` decides what `Q` counts against:

- **`run`** (default) counts items **routed** to the current replica. A run spends `Q` items on
  one replica, then the next run opens elsewhere, so a model's GPUs serve it in turn: batches
  stay deep and every replica gets an even share (to within one run) at any concurrency. A run
  also ends when the model has nothing in flight, so a low-rate model rotates per request instead
  of committing `Q` in a row, and when a replica falls a whole quantum behind the least-backed-up
  one, so a slow replica loses its turn rather than accumulating a queue.
- **`spread`** equalizes **in-flight items** across the replica set, so all `k` GPUs serve the
  model at once. This is the right mode when your client already batches to the max batch size,
  because then the worker has nothing left to coalesce and concentration only idles GPUs. It is
  also what promotion means for a latency-bound model whose client concurrency sits well below
  its max batch; the cost is coalescing depth, since each GPU now batches roughly `1/k` as many
  requests. Prefer it per model rather than fleet-wide.
- **`inflight`** is the legacy basis: `Q` counts requests **in flight**, so a replica keeps
  receiving the model's traffic until it holds a full quantum at once.

!!! warning "`inflight` can park a model on one GPU indefinitely"
    Because `inflight` compares the quantum against in-flight requests, a replica keeps winning
    until it holds `Q` of them *simultaneously*. A model whose in-flight concurrency stays
    between 2 and `Q` and never drains to zero is therefore served by **one** replica for the
    life of the process, and its other replicas receive no traffic at all. This is not a
    warm-up transient: with `max_batch = 32` and a client holding 4 requests in flight, the
    second GPU receives zero requests, forever. Closed-loop clients (a fixed-size worker pool, a
    benchmark harness) hit this reliably, so promoting such a model to more GPUs under `inflight`
    buys nothing.

    Diagnose it with `gateway_replica_routed_total`: a flat series on one replica of a
    multi-replica model is the signature (`gateway_replica_outstanding` cannot show it, because
    a starved replica reads 0 either way). Use `inflight` only to reproduce the behavior that
    predated `run`.

`routing_fill_factor` is the direct trade between the two concerns: a model's share is even to
within one quantum, so a smaller factor balances more finely at the cost of splitting more
batches, and a larger one commits longer to each replica for deeper batches.

`routing_policy` (lpt_packing only) decides *which* replica each run opens on:

- **`fill_least`** (default) opens each run on the replica whose GPU currently carries the least
  in-flight work, measured across *all* models and denominated by `work_basis` (in-flight
  GPU-seconds by default). This is what you want whenever replicas share GPUs with other models:
  a model's runs open on whichever of its GPUs is least busy rather than always the same one.
- **`fill_rr`** rotates the opening replica across the model's set, so successive runs of the
  same model spread evenly over its GPUs. Deliberately load-blind, which is right only when every
  model on the fleet costs about the same and no GPU is shared with anything else.

Both policies are consulted at every run boundary, and exact ties rotate rather than falling back
to the worker name, so an idle fleet warms every replica instead of pinning to the lowest-named
one. Spreading every request without concentrating it at all is the separate `least_outstanding`
scheduling mode above, not a fill mode. A single-replica model is the degenerate case: all its
requests go to its one GPU (and still count toward that GPU's load for the `fill_least`
decisions of models that share it).

#### Preconditions and startup

`lpt_packing` has two preconditions, verified at gateway startup: every worker must run the
`fifo` or `edf` scheduler discipline, not `fair` (placement decisions move to the gateway, so
workers should not re-order against it; see `scheduler.discipline` in
[Node Configuration](node_config.md)), and
every worker must serve the identical model set. Because a worker compiles and warms up every
model before its control plane answers, the workers are usually not up when the gateway starts,
so the gateway waits for all of them before serving rather than failing, logging which workers
are still pending. Under the node supervisor (the embedded gateway) this wait is enabled
automatically; for a standalone gateway set `REACTANT_GATEWAY_STARTUP_WAIT_SECONDS` (a number of
seconds, or `inf` to wait indefinitely; the default `0` fails fast). Each poll is
watchdog-bounded, and if the gRPC client stack wedges during the long warmup (a known libcurl
failure mode) the gateway exits so the supervisor restarts it with a fresh stack; this self-heals
and you may see one such restart before it serves. Once all workers are up, a wrong discipline
or differing model set is a hard error.

Runtime drift after startup degrades gracefully: a model temporarily missing from some workers is
routed uniformly over its actual replicas with a warning until the fleet converges, and a worker
that drops out is excluded from placement, its traffic failing over to the remaining replicas.

#### Observability

The placement is observable: `gateway_model_replicas` reports each model's replica count,
`gateway_placement_weight` reports its per-worker weight, `gateway_replica_outstanding` reports
the in-flight **items** per replica (summed batch sizes, matching the quantum's denomination;
equal to a request count when every request carries one item), `gateway_replica_routed_total`
counts the **requests** routed to each replica since start (the series that shows whether every
replica is actually being used), `gateway_model_fill_quantum` reports each model's effective
quantum, `gateway_repacks_total` counts repacks by what triggered them, and
`gateway_model_utilization` reports each model's estimated demand in GPU-seconds per second.

`gateway_worker_inflight_work` reports the per-worker load the router actually compares when it
picks a replica, so a routing decision can be explained after the fact. It is exported by both
work-routing mechanisms (`fill_least` under `lpt_packing`, and the `least_outstanding` mode) and
carries a `basis` label, because the unit follows `work_basis`: GPU-seconds under `compute`,
items under `items`, and a per-request cost under `requests`. Read it live at scrape time from
the scheduler's counters, so it never lags a prober tick, and a worker that no longer hosts
anything stops being emitted rather than freezing at its last value. `round_robin` exports
nothing here, since it tracks no load.

The bundled Grafana dashboards use it: **Scheduling & Placement** plots the per-worker load, the
spread across workers (`max - min`, which is what `fill_least` exists to shrink), and the basis
in force as the running gateway reports it, and **Fleet Overview** carries the per-worker load
alongside readiness and device memory. For whether the routing actually produced an even result,
as opposed to what the router intended, the **GPU Work Balance** dashboard is the one to read. It
is built on `rate(worker_compute_seconds_total)` per worker, the device-seconds each GPU really
consumed, and leads with the spread of that rate across devices (0% is even). Its
planned-vs-actual panel is the diagnostic that separates a skewed *plan* (a repack or a replica
count is the lever) from routing not following an even plan. A persistently high spread usually
means a model's replicas cannot absorb the imbalance (too few replicas, or a single-replica
model pinning one GPU) rather than a misbehaving policy, so read it next to the placement table.

## Runtime scheduling control

The gateway answers its own gRPC service, `reactant_control.GatewayControlService`, on the same
port it serves inference. It exposes the live scheduling state and lets an operator retune it
without a restart:

| RPC | Purpose |
| --- | --- |
| `GetSchedulingStatus` | The mode, every runtime knob, repack bookkeeping, and one row per model and per worker (placement, in-flight and routed counts, resolved fill mode and quantum, measured demand and cost, assigned weights against each worker's budget). Answers in every scheduling mode. |
| `SetSchedulingPolicy` | Change any knob in the `scheduling:` block except `mode`. `update_mask` names the fields to apply; one invalid value rejects the whole request and nothing changes. |
| `SetModelPlacement` | Set, change, or clear one model's `replicas`, `fill_mode`, and `fill_factor`. The three are independent, so one call can promote a model and change how it uses its GPUs. Effective at the next repack. |
| `Repack` | Repack now, bypassing the accumulated-compute budget, optionally waiting a bounded time for it to land. |

`tools/gateway_ctl.jl` is the operator front end, run against the gateway's gRPC port:

```text
julia --project=packages/ReactantServerGateway tools/gateway_ctl.jl --gateway HOST:8001 status
... set-replicas big-model 2 --fill-mode spread
... repack --wait 30
... set-policy hysteresis=0.15 routing_policy=fill_least compaction_interval=4
```

`grpcurl -plaintext -proto proto_src/reactant_control_v1.proto ...` works too, with no Julia
installed.

!!! warning "Runtime changes are not persisted, and there is no authentication"
    Every override lives in memory only: a gateway restart reverts it to `gateway.yml` plus the
    environment, and restart is not hypothetical (the gateway deliberately exits on a wedged
    client stack, expecting its supervisor to bring it back). Treat these RPCs as a way to try a
    setting or respond to an incident, and write anything you want to keep into `gateway.yml`.
    `SchedulingPolicy.generation` reads 0 while every knob still matches the config file and is
    bumped by each accepted change, so you can tell a tuned gateway from a fresh one, and each
    accepted change is logged with its before and after values, which is the only durable record
    of it. Like the rest of the control plane these RPCs are unauthenticated (see
    [Deployment](deployment.md)), so the gateway's gRPC port must not be exposed outside the
    trusted network.

## What the gateway does not do

- Streaming RPCs.
- The repository / model-config / statistics / trace / log RPCs in the Triton spec, plus
  `ServerLive`, `ServerReady`, `ModelMetadata`, and `RepositoryIndex` for clients. Of the KServe
  data plane only `ModelInfer`, the two SHM register/unregister RPCs, and `IsSameIPCNamespace`
  are proxied; everything else returns `UNIMPLEMENTED`. On the control plane the gateway answers
  `ControlService/CompactMemory` (fanned out to every worker) and all four
  `GatewayControlService` RPCs itself (see [Runtime scheduling control](#runtime-scheduling-control));
  the other `ControlService` RPCs are worker-only.
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

To tune the gateway, provide a `gateway.yml` and point `REACTANT_GATEWAY_FILE` at it (or leave
it at the conventional `/etc/reactantserver/gateway.yml`, which the supervisor picks up
automatically; or, for the embedded gateway, set the `REACTANT_GATEWAY_*` environment below); it
carries the gateway's own settings (listen addresses, message limits, logging, and the
`scheduling:` block above). Settings can also be overridden by environment with the prefix
`REACTANT_GATEWAY_` and the dotted path uppercased with underscores, for example
`REACTANT_GATEWAY_LOGGING_LEVEL=debug` or `REACTANT_GATEWAY_SCHEDULING_MODE=lpt_packing`.

!!! note "A mounted `gateway.yml` owns the endpoint list"
    Per-model settings (`scheduling.models.<name>`) are the one part of the config with no
    environment equivalent, since the overrides only reach scalar keys, so promoting a single
    model requires a file. When the supervisor finds one it stops synthesizing the worker lists,
    because the file is now the authority: the file must therefore carry `endpoints:` itself,
    plus `metrics_endpoints:` and `worker_names:` if you want the aggregated `/metrics` and the
    `worker0..N` labels the Grafana dashboards join on. The startup wait is *not* suppressed (it
    describes the supervisor's own co-launched workers, not gateway configuration), so an
    lpt_packing node still waits for its workers to finish compiling rather than crash-looping.

For a standalone deployment, the package also ships [`probe_worker_ready`](@ref), the worker
container's healthcheck (a Julia replacement for the former Go `reactant-healthprobe`): it
resolves a worker's port from the node file and calls its KServe `ServerReady` on localhost,
returning whether it reported ready.

## Operational notes

- The gateway is a single point of failure. Each Julia worker stays reachable on its own KServe
  V2 gRPC endpoint during a gateway outage, so a client can fall back to addressing a worker
  directly.
- The routing table is rebuilt every 10s from each worker's `RepositoryIndex` and swapped in
  atomically. If a worker dies, its routes persist until the next successful probe (up to ~10s);
  in the gap, requests to its models fail over to the remaining replicas (on `NotFound` or
  `Unavailable`), and a model with no live replica returns `NotFound`. The worker-side readiness
  probe (`ServerReady`, same 10s loop) drives `/readyz` and the `gateway_worker_ready` metric.
- Under `lpt_packing`, the gateway polls the workers on every 10s probe round to refresh routing
  metadata and accumulate consumed compute, but recomputes the placement only once the fleet has
  consumed `scheduling.rebalance_compute_seconds` GPU-seconds (the first repack uses
  `scheduling.first_rebalance_compute_seconds` when set). Each repack logs a `lpt_packing: repack`
  line with the number of models placed, how many `moved` workers, how many were
  `held_by_hysteresis`, the largest available `max_improvement` against the `hysteresis`
  threshold, and the `compute_seconds`/`wall_seconds` since the last repack, useful for watching
  placement churn and the trigger cadence.
- If a probe to a worker hangs (times out, rather than failing fast), the gateway drops and
  recreates that worker's gRPC connection before the next attempt. This recovers from a
  half-open connection (for example one caught during a worker's brief silent-accept window at
  startup) that would otherwise be reused and stall every later request to that worker, the
  per-worker equivalent of a restart, without dropping HTTP/2 multiplexing for healthy workers.
- Successful `ModelInfer` requests are not logged (to keep the hot path quiet); worker errors and
  a model with no live replica are logged, and per-request latency and gRPC status are exported
  as Prometheus metrics. Logs contain no tensor data.
