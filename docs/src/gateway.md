# Multi-GPU Gateway

The gateway is a gRPC reverse proxy that puts several workers behind one KServe V2 endpoint. It
lives in `ReactantServerGateway` and does not load Reactant. A node with two or more workers runs
one in front of them on the public ports, configured from the node file
([Deployment](deployment.md)); it can also run on its own over workers you start yourself
([Front Workers with a Gateway](tutorials/julia/gateway.md)).

The gateway reads each request's model name and forwards the request bytes unchanged to a worker
that serves the model. On a multi-worker node it also serves the node's merged metrics
([Monitoring](monitoring.md)).

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
  [Scheduling modes](#Scheduling-modes) below. A third mode, `least_outstanding`, spreads by
  live occupancy. Either way, a request fails over to the remaining replicas when a worker
  returns `NotFound` or `Unavailable`, or the call to it fails in the transport; every other
  status reaches the client as the worker sent it, and a request that uses shared memory is
  never retried on another worker (see [Status codes and failover](#Status-codes-and-failover)).
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
  succeeds only if all workers succeed. On a partial failure it rolls back only the registrations
  this call created: each worker is first asked (`SystemSharedMemoryStatus`) whether it already
  holds the name, and one that did (a client re-registering after another worker restarted) keeps
  it. A worker that holds the name for another client's live object answers `ALREADY_EXISTS`, and
  the gateway returns `ALREADY_EXISTS`; other failures return `FAILED_PRECONDITION`. Unregister
  succeeds if any worker does.
- **SHM namespace probe:** `IsSameIPCNamespace` is fanned out to every worker and the gateway
  returns `true` only if all of them can see the client's region (any worker may service a later
  `ModelInfer`). A worker that errors or does not implement the RPC counts as `false`.
- **Model metadata:** `ControlService/DescribeModels` asks every worker and merges the answers,
  so a client learns what the node serves without reading its model repository: each model's
  tensors with the kind of every axis (fixed, batch, or variable), its compiled batch sizes and
  input shapes, and the sub-models a meta model calls. The KServe `ModelMetadata` is forwarded to
  a worker that hosts the model, and `RepositoryIndex` is the union of every worker's index. One
  worker being down does not fail any of them.
- **Observability:** structured logs, Prometheus metrics, `/healthz`, and `/readyz` on a
  separate admin HTTP port.

## Scheduling modes

The gateway routes each model's requests across its replicas according to `scheduling.mode` in
`gateway.yml`. `round_robin` is the predictable default and needs no measurements;
`least_outstanding` spreads by live occupancy; `lpt_packing` concentrates each model's traffic
on few GPUs for deep coalescing batches and is the mode whose placement the gateway manages.
Every setting in the block below except `mode` can also be changed while the gateway runs,
through its own gRPC control plane; see
[Runtime scheduling control](#Runtime-scheduling-control).

```yaml
scheduling:
  mode: lpt_packing             # round_robin (default) | least_outstanding | lpt_packing
  work_basis: compute           # compute (default) | items | requests  (what "least busy" measures,
                                #   for least_outstanding and for fill_least; see below)
  rebalance_compute_seconds: 300 # fleet GPU-seconds consumed that triggers a repack
  first_rebalance_compute_seconds: 60 # smaller budget for the first repack (0 = use rebalance_compute_seconds)
  ema_halflife_compute_seconds: 0 # demand-EMA halflife in fleet compute-seconds (0 = track rebalance_compute_seconds)
  hysteresis: 0.05              # least relative gain a placement change needs, a floor against noisy estimates
  payback_seconds: 600          # a change must save more latency over this window than its uploads cost
  prime_moves: true             # stage a moving model's weights on its new GPU before routing switches
  stage_timeout_seconds: 120    # a change whose weights are not staged by then is dropped from that repack
  default_replicas: 1           # GPUs per model unless overridden below (a number, or "all")
  routing_fill_factor: 1.0      # fill quantum as a multiple of max batch size (lpt_packing only)
  routing_policy: fill_least    # fill_least (default) | fill_rr  (lpt_packing only)
  routing_fill_mode: run        # run (default) | spread | inflight  (what the quantum counts;
                                #   `inflight` can park a model on one GPU, see the warning below)
  forbid_memory_oversubscription: true # never strand a model on-demand when it could fit resident (default on)
  compaction_mode: off          # off (default) | lazy | preload  (see On-demand Weights)
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
- **`requests`** counts requests: a raw count for `least_outstanding`, and the model's measured
  cost per *request* for `fill_least` (which keeps its cross-model cost weighting). Blind to batch
  size either way, so a pre-batching client defeats the balancing.

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
- **A meta** (`kind: meta`) reports no batch axis and counts one row per request, even when its
  own `client_inputs` declare a batch axis, so a work-routing gateway charges each request the
  meta's whole measured cost. That cost is the GPU time of its sub-calls, which are hidden from
  the gateway, so it is counted once. Under `items` a meta charges one unit however expensive it
  is, so fleets that serve metas should use `compute`.

The batch extent is the client's claim, and the gateway never rejects a request over it: the worker
is the authority on what is valid. It believes the claim only as far as the request's own data
goes, since every row takes at least one byte: a request counts as at most as many items as it
carries bytes for the batch input, inline (its `raw_input_contents` entry, or the encoded length of
its typed `contents`) or as the `shared_memory_byte_size` it declares, and as at least one item.
A request that claims 2^62 rows in a few bytes of data therefore counts as a few items, and none
counts as more than 2^40. The in-flight counters are integers (nanoseconds of GPU time, or items or
requests), each request releases exactly what it was charged, and a charge is capped at about 78
hours of GPU time, so no claim can make the arithmetic fail or leave a counter off after release.

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
how busy the fleet is rather than with wall-clock).

#### What a repack optimizes

A repack weighs throughput, latency and memory against each other by reducing all three to one
resource, device time, and one objective, mean latency. A worker's **load** is the device time it
spends per second:

```text
load(w) = sum of u_m over the models routed to w            (compute: arrival rate x cost per request)
        + sum of f_m x U_m over those its budget cannot hold   (upload churn)
```

`u_m` is a model's compute demand, `f_m` its request rate, and `U_m` the seconds one upload of its
weights holds a GPU, which the workers measure and report. A worker keeps the models with the most
upload seconds saved per byte held, which is how its own eviction ranks them; since upload time grows
with size, that is close to keeping the most often requested models whatever their size. Memory
therefore costs nothing until it overflows, and then costs exactly the uploads the overflow causes,
in the same units as compute. A large model that is rarely called is cheap to leave out; one that is
called often is expensive to miss, and the load says by how much.

A queue at utilization `rho` holds `rho / (1 - rho)` requests on average, so by Little's law mean
latency is that summed over the workers, divided by the request rate. The repack minimizes that sum
(`gateway_repack_objective`). It rises steeply as a worker nears saturation, which is why relieving a
saturated GPU is worth a weight upload and reshuffling an idle fleet is not.

Reading a weights file from disk is not part of this. The worker reads in the background, never on
the loop that runs the GPU (see [On-demand Weights](on_demand_weights.md)), so a slow disk costs the
requests that wait for a cold model latency but never costs device time. That latency is decided
where a model's copy in host RAM is evicted, not by placement.

#### Moves and swaps

Only the startup placement packs from scratch, and it is then improved by the same objective with
uploads priced at zero, since nothing is loaded yet. Every later repack starts from the current
placement and changes only what it must or what pays for itself:

1. Place what is new or lost its worker, and adjust replica counts.
2. With `forbid_memory_oversubscription`, move models off any worker over its weight budget (the
   smallest model that clears the overflow first).
3. Repeatedly make the best change off the most loaded worker: move one of its models to another
   worker, or **swap** one for a model of another worker. A swap is the only change that fits when
   every worker is at its weight budget, which is how a node whose weights outgrow its GPUs runs.

A change is made when the latency it saves, held for `payback_seconds`, exceeds the latency its
uploads cost (each upload delays the requests already waiting on that GPU, plus the one that
triggers it), and when it lowers the objective by at least `hysteresis` of the two workers' share of
it, a floor against noise in the rate estimates. The change with the most net saving goes first, and
a move beats a swap that saves the same, since it is one upload instead of two. There is no fixed
number of changes per repack: it makes every change that pays and stops. `payback_seconds: 0` never
moves a model for balance. With a settled fleet a repack changes nothing. A worker that misses the
poll for less than 30 s (three probe rounds) keeps its models: the repack waits rather than moving
them onto the rest of the fleet and back. (`max_worker_share` is accepted but does not affect a
model's GPU count.)

Without `forbid_memory_oversubscription` a worker may be placed more weights than its budget, and the
overflow is priced by the uploads it causes. With it (the default), the budget is a hard limit on
every change.

#### Primed moves

With `prime_moves` (the default), a change that moves a model off a GPU still serving it does not
switch routing straight away. The gateway first asks the new worker to read the model's weights
into host memory (the worker's `StageWeights` RPC, which reads off its GPU loop and holds the weights
for the model's first dispatch), and switches routing only once they are there. The model keeps
serving from its old GPU meanwhile, and its first request on the new one waits for an upload from
RAM, never for a read from disk. A change whose read fails, is deferred (the worker caps the bytes it
holds this way at half its host weight budget) or takes longer than `stage_timeout_seconds` is
dropped from that repack and planned again by the next one. A swap commits only when both of its
models are staged, since half a swap could overfill a GPU. The prober keeps running while the reads
do, and no other repack starts until they end.

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
  requests. Prefer it per model rather than fleet-wide. The exception is a fleet whose workers
  run with `scheduler.coalescing: false` (the `regulated` profile): no worker coalesces, so
  concentration buys nothing, and `spread` is the right fleet-wide choice. The gateway does not
  detect this on its own; set it explicitly.
- **`inflight`**: `Q` counts requests **in flight**, so a replica keeps
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
    a starved replica reads 0 either way). Use `run` instead.

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
automatically; a standalone gateway waits 60 seconds by default, which covers a worker's first
control-plane call after it reports ready (that call compiles the worker's gRPC request path).
Set `REACTANT_GATEWAY_STARTUP_WAIT_SECONDS` to a number of seconds, `inf` to wait indefinitely, or
`0` to check once and fail fast. Each poll is
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
What a repack costs is counted too: `gateway_repack_moved_models_total` and
`gateway_repack_moved_bytes_total` (the weight bytes it placed on workers that did not hold them),
and `gateway_worker_assigned_weight_bytes` against `gateway_worker_weight_budget_bytes` shows each
worker's planned weights next to its budget as of the last repack.

`gateway_worker_probe_seconds` is the round trip of the ServerReady health probe per worker. A
worker whose request handling stalls shows here first, before requests fail; a probe that times out
counts its full wait.

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
| `Repack` | Repack now, bypassing the accumulated-compute budget, optionally waiting a bounded time for it to land (a repack staging its moves lands when it commits them). `full` packs from scratch and installs the result at once, without staging: an escape hatch that can move most models. |

`tools/gateway_ctl.jl` is the operator front end, run against the gateway's gRPC port:

```text
julia --project=packages/ReactantServerGateway tools/gateway_ctl.jl --gateway HOST:8001 status
... set-replicas big-model 2 --fill-mode spread
... repack --wait 30
... repack --full               # pack from scratch; reloads most models at once
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
    [Security](security.md)), so the gateway's gRPC port must not be exposed outside the
    trusted network.

## Status codes and failover

The gateway forwards a worker's gRPC status to the client unchanged, and moves a request on to
the next replica only when the worker it tried could not serve it:

| The worker call ended with | The gateway |
| --- | --- |
| `NOT_FOUND` (the model was unloaded there since the last discovery round) | tries the next replica and refreshes its routes |
| `UNAVAILABLE` (the worker is shutting down or draining, its dispatch loop died, or the model is not resident there or cannot be placed on its device now) | tries the next replica |
| a transport failure (connection refused or reset, a response that broke off, a client handle reset after a hung probe) | tries the next replica, as `UNAVAILABLE` |
| any other status: `DEADLINE_EXCEEDED`, `INTERNAL`, `INVALID_ARGUMENT`, `RESOURCE_EXHAUSTED`, `FAILED_PRECONDITION`, `CANCELLED`, ... | returns it to the client as is |

A deterministic failure (`INTERNAL`) or the request's own mistake (`INVALID_ARGUMENT`) would only
repeat on another replica, a passed deadline cannot be met there either, and `RESOURCE_EXHAUSTED`
and `FAILED_PRECONDITION` are signals the client acts on (back off; re-register shared memory).
When every replica fails over, the client gets the last worker's status. Status messages follow
`grpc.error_detail` (see [Node Configuration](node_config.md)): under `redacted`, the default, a
worker's message passes through as the worker sent it, a failure of the call to a worker names the
worker by its name or position and never by host and port, and the gateway's own `INTERNAL`
failures are reported by request id and logged in full on the gateway; under `full`, messages also
name the worker by host and port and carry the exception text.

A request that reads an input from, or writes an output to, a registered shared-memory region
(a `shared_memory_region` parameter on any input or requested output) is never retried on another
replica: the worker that failed it may still be running it, and a second worker writing the same
output slot, or reading an input slot the client has since reused, would corrupt the client's
data. Whatever ends its attempt (`NOT_FOUND`, `UNAVAILABLE`, an abandoned or reset call, a
transport failure) reaches the client as `UNAVAILABLE`, which `ReactantServerClient` retries. Its
other statuses pass through like any request's. The gateway reads only the tensors' parameter
keys to tell, never their payloads.

When a client cancels its call, the gateway cancels its in-flight worker call (with gRPCClient's
`grpc_cancel`) and the worker withdraws the queued request, so cancelled work is not run. Both
hops depend on gRPCServer reporting a cancelled call (`is_cancelled`); gRPCServer 0.1.2 does not
yet, so until a release that does, a cancelled call runs to completion as before.

## What the gateway does not do

- Streaming RPCs.
- The repository-control / model-config / statistics / trace / log RPCs in the Triton spec,
  plus `ServerLive` and `ServerReady` for clients. Of the KServe data plane the gateway serves
  `ModelInfer`, the two SHM register/unregister RPCs, `IsSameIPCNamespace`, `ModelMetadata`, and
  `RepositoryIndex`; everything else returns `UNIMPLEMENTED`. On the control plane the gateway
  answers `ControlService/CompactMemory` (fanned out to every worker), `DescribeModels` (merged
  from every worker), and all four `GatewayControlService` RPCs itself (see
  [Runtime scheduling control](#Runtime-scheduling-control)); the other `ControlService` RPCs are
  worker-only.
- TLS: parsed but not yet enforced; the listener and the worker back-hop are cleartext h2c.
- CUDA shared memory.
- Dynamic worker membership: the worker endpoint list is fixed at startup (from `gateway.yml`
  or `REACTANT_GATEWAY_WORKERS`). Which models each worker serves is rediscovered continuously,
  but adding or removing workers requires a gateway restart.

## Configuration

On a node, the supervisor passes the embedded gateway its worker endpoints and metrics endpoints
as `REACTANT_GATEWAY_WORKERS` and `REACTANT_GATEWAY_WORKER_METRICS`. Which models each worker
serves is never configured; the gateway discovers it.

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
container's healthcheck: it
resolves a worker's port from the node file and calls its KServe `ServerReady` on localhost,
returning whether it reported ready.

## Operational notes

- The gateway is a single point of failure. Each Julia worker stays reachable on its own KServe
  V2 gRPC endpoint during a gateway outage, so a client can fall back to addressing a worker
  directly.
- The routing table is rebuilt every 10s from each worker's `RepositoryIndex` and swapped in
  atomically. If a worker dies, its routes persist until the next successful probe (up to ~10s);
  in the gap, requests to its models fail over to the remaining replicas (on `NotFound`,
  `Unavailable`, or a transport failure; see [Status codes and failover](#Status-codes-and-failover)),
  and a model with no live replica returns `NotFound`. The worker-side readiness
  probe (`ServerReady`, same 10s loop) drives `/readyz` and the `gateway_worker_ready` metric.
- Under `lpt_packing`, the gateway polls the workers on every 10s probe round to refresh routing
  metadata and accumulate consumed compute, but recomputes the placement only once the fleet has
  consumed `scheduling.rebalance_compute_seconds` GPU-seconds (the first repack uses
  `scheduling.first_rebalance_compute_seconds` when set). Each repack logs a `lpt_packing: repack`
  line with the number of models placed, how many `moved` workers and the weight bytes that
  move puts on their new workers (`moved_weights`), whether it was `incremental` (every repack
  after the startup placement), the `changes` it planned and how many are `staging`, the objective
  before and after (`objective_before`, `objective_after`), the largest relative improvement any
  change offered (`max_improvement`, against the `hysteresis` floor), and the
  `compute_seconds`/`wall_seconds` since the last repack, useful for watching placement churn and
  the trigger cadence. A repack that stages its moves then logs `staged placement changes
  committed` with how many it committed and dropped (also counted by
  `gateway_repack_staged_changes_total{outcome}`). A repack that comes due while a placed worker is
  missing from the poll logs `repack deferred` instead and runs once the worker answers or 30 s have
  passed.
- If a probe to a worker hangs (times out, rather than failing fast), the gateway drops and
  recreates that worker's gRPC connection before the next attempt. This recovers from a
  half-open connection (for example one caught during a worker's brief silent-accept window at
  startup) that would otherwise be reused and stall every later request to that worker, the
  per-worker equivalent of a restart, without dropping HTTP/2 multiplexing for healthy workers.
- Successful `ModelInfer` requests are not logged (to keep the hot path quiet); worker errors and
  a model with no live replica are logged, and per-request latency and gRPC status are exported
  as Prometheus metrics. Logs contain no tensor data.
