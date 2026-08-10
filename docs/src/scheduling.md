# Scheduling & Coalescing

The worker's scheduler decides which model runs next whenever the GPU is free, and how many
queued requests one execution should serve. It is a deficit-weighted, cost-aware, coalescing
dispatch engine: concurrent requests land on per-model queues, a single dispatch loop runs
exactly one GPU execution at a time, same-model requests are merged into one batched execution
at a compiled size, and GPU time is shared by relative model weight and a learned per-batch-size
cost estimate. It holds the model registry, the backend, the device memory pool, the
[`SchedulerConfig`](@ref), per-model dispatch state, and the optional on-demand `WeightCache`.
Submit work with `ReactantServer.infer` and read observability counters with
`ReactantServer.scheduler_metrics`.

Because it serializes the GPU, the scheduler is the component that turns a burst of concurrent
requests into a steady stream of full batches. This page covers that worker-side engine and the
coalescing it performs. Routing a model's requests across several workers, including the gateway
scheduling modes (`round_robin`, `least_outstanding`, `lpt_packing`), is the gateway's job; see
[Gateway](gateway.md). Where model weights live and how on-demand loading works is covered in
[On-demand Weights](on_demand_weights.md).

## How a request flows through the worker

Each gRPC request arrives on its own task. The request runs its `preprocess` hook on that task,
then is queued on its model's queue (FIFO, so the front is always the oldest request). The
dispatch loop wakes when work is available, selects a model, coalesces that model's queued
requests into one execution at a compiled batch size, runs it, and hands each caller back its
raw output slice; the caller's task runs the `postprocess` hook on that slice. The loop runs no
`model.jl` code itself, so user hooks never serialize against the GPU.

```text
       gRPC requests (one task each)
                 |
   preprocess hook runs on the caller's task
                 |
        per-model queues (FIFO, one per model)
                 |
       +---------+---------+
       |   dispatch loop   |   one GPU execution at a time
       | select -> coalesce -> run -> slice
       +---------+---------+
                 |
        per-request output slices
                 |
   postprocess hook runs on the caller's task
                 |
          replies to clients
```

A request becomes visible to the dispatch loop only after its `preprocess` has returned, so the
loop never executes a request whose hook work has not finished. Because the hooks run on the
per-request tasks rather than on the loop, the CPU-side pre/post work of many requests proceeds
in parallel and overlaps the single, serialized GPU execution: while the loop runs one model on
the GPU, other requests are being pre/post-processed on other threads.

A single `Threads.Condition` guards all mutable scheduler state (the per-model queues and the
EMA/cost fields) and signals the dispatch loop when work arrives. The loop holds the lock only
to select and dequeue; the actual execution runs outside the lock, so producers keep enqueuing
during GPU work, and the lock is re-taken briefly afterwards to record the dispatch. Only one
execution runs at a time by design, which keeps memory and concurrency reasoning simple and is
what makes the single-resident-model weight strategy safe.

### The dispatch loop runs on its own thread

The supervisor starts each worker with `--threads=<compute_threads>,1`: a default pool sized to
the worker's share of the host for the per-request pre/post tasks, plus one interactive thread.
The scheduler pins its GPU dispatch loop to the interactive thread when one exists, so the GPU
dispatch is scheduled promptly and is never starved by the hook tasks that saturate the default
pool. With no interactive thread the default pool is used; correctness is unaffected, only the
overlap.

## Decision order

Each time the GPU frees up, the scheduler decides in this order:

1. **Dispatch a committed meta sub-call first, if one is pending.** A meta model's orchestration
   runs off the dispatch loop and re-enters the scheduler for each sub-call (see
   [Meta Models](meta_models.md)); those continuations are committed and jump ahead of the
   discipline scan, at most one per in-flight meta, so a meta already in progress is not stalled
   behind newly arrived work.
2. **Otherwise select the next model**, according to the configured `discipline`: `fair`
   (deficit-weighted and cost-aware, the default), `fifo` (oldest queued request), or `edf`
   (soonest deadline).
3. **Coalesce the winner.** The selected model's queued requests are taken in FIFO order and
   packed into the largest compiled batch size that fits, padding a partial batch up to the
   smallest size when needed and always making forward progress on at least the oldest request.
   The remainder stays queued for the next round.

## Disciplines

The discipline selects which model is dispatched next. Coalescing applies underneath all three:
whichever model wins, its queue is coalesced in FIFO order.

### fair (default)

Each model has a relative `weight` that defines its share of compute:
`share = weight / sum(weights)`. The scheduler tracks a decaying exponential moving average
of how much GPU time each model has recently consumed (decayed to "now" with a half-life of
`ema_halflife_seconds` before every read or write), and scores every model with a non-empty
queue by how far below its share it is, divided by its estimated cost:

```text
priority = clamp(share - recent_compute_ema, -cap, cap) / effective_cost
```

A model that has consumed less than its share recently scores higher. Dividing by cost stops an
expensive model from blocking cheaper ones on a marginal fairness edge. The clamp
(`recency_penalty_cap`, default 0.25) bounds both lockout and domination: a quiet model cannot
run away with the worker, and a saturated one cannot be starved entirely. When every model's
recent-compute EMA is zero (quiet or newly loaded models), the deficit equals the share, so those
models stay schedulable rather than normalizing to zero.

### fifo

Under `fifo`, per-model weights are ignored. The model whose oldest queued request is the oldest
overall is served first, with ties broken by model name for determinism. This is the discipline
to use when the worker sits behind a gateway that is already doing placement; see "Which
discipline to use" below.

### edf

Under `edf` (earliest-deadline-first), the model whose most-urgent queued request has the soonest
deadline is served first. The deadline comes from the request's remaining-budget timeout; a
request with no deadline is treated as least urgent. Ties break by the model's oldest queued
request and then by name, so when all queued requests share a deadline (the common case) `edf` is
exactly `fifo`. It diverges only to promote a model carrying a request with less budget left, so
a meta's sub-call, which inherits the meta's deadline, is served ahead of fresher regular work.
Queues stay arrival-ordered (never reordered), and coalescing still draws FIFO from the front, so
a request deep behind the batch size advances over successive dispatches of its model rather than
jumping the batch.

`edf` also sheds work it cannot finish in time. Before GPU work begins, a request whose deadline
has already passed is dropped under every discipline; under `edf` a laxity check is added: a
request is also dropped if it cannot finish within this dispatch's learned compute cost, so the
GPU is not spent on work that will miss anyway (the classic EDF overload failure mode). Dropped
requests receive a `DeadlineExceeded` error (gRPC `DEADLINE_EXCEEDED` upstream); only feasible
requests run, and when none are feasible the execution is skipped entirely. The cost estimate is
learned from prior dispatches and unseeded pairs default small, so the laxity drop is a no-op
until a model has run once. Committed meta sub-calls skip the laxity drop (their earlier stages
already spent GPU time) but still honor the hard deadline check.

Note that because `edf` derives urgency solely from the deadline, issuing different per-client
deadlines for the same model reorders that model's service and therefore affects fairness across
clients; keep deadlines uniform to retain `fifo`-like fairness.

### Which discipline to use

`fair` is for workers with no upstream placement authority: a single-GPU worker, or a fleet
behind the round-robin gateway, where the worker itself must stop one model from crowding out
the rest. Behind a gateway running `lpt_packing`, the gateway is the fairness and placement
authority (see [Gateway](gateway.md)), and the worker must run `fifo` or `edf` so the two do not
fight; `fair` would double-book GPU time against the gateway's placement decisions. `edf` is for
deadline-sensitive serving, and it degrades gracefully to `fifo` when clients use uniform
deadlines.

## Coalescing

Coalescing merges same-model requests into one GPU execution. The selected model's queue is
scanned in FIFO order, the requests that fit are concatenated along the batch axis into a single
compiled-batch-size dispatch, the execution runs once, and the outputs are sliced back per caller
(zeros padding a partial fill are dropped). The remainder stays queued for the next round. The
batch sizes themselves come from the bundle's manifest batching spec: a bundle may carry several
compiled batch sizes that share one set of weights, and the scheduler picks the executable that
matches each coalesced batch without duplicating parameters.

### When a model can coalesce

A model coalesces multiple requests only when every executable input and every output carries a
batch axis (an all-or-nothing requirement that mirrors the loader's manifest check). Inputs are
then concatenated along that axis and outputs split back per request. Otherwise the model serves
one request per dispatch. Single unbatched modules (the batch-0 executable) are never coalesced.
Inputs without a batch axis are taken from the first request, since they do not vary across the
batch.

Only requests that resolve to the same input-shape variant coalesce together. A model compiled
for several input shapes (aspect-ratio variants) keeps one executable set per variant over a
single shared set of weights, and different shapes cannot be concatenated along the batch axis.
The coalescing window is the leading run of same-variant requests at the front of the queue,
which keeps the taken set a contiguous prefix (so it can be removed in one step) and preserves
FIFO order. A request whose shape has no compiled executable dispatches alone so that the caller
gets the precise "no compiled program for input shape" error.

### Sizing the batch

The scheduler chooses the largest compiled batch size that the queued rows can fill; a partial
fill pads up to the smallest compiled size (the compiled shape may therefore exceed the queued
rows, and the dispatch runs padded). Coalescing uses only what is queued now; there is no
look-ahead.

A per-model `max_batch_size` caps how many rows are coalesced into one dispatch. The cap does not
change compiled shapes: a partial fill still pads up to the smallest compiled size, and a single
request larger than the cap is never split, so it dispatches alone (growing the chosen size to
the smallest compiled batch that fits it when necessary). Forward progress is guaranteed: an
indivisible front request that exceeds the batch size or the cap still dispatches alone rather
than being held back, so the oldest request is always served within a bounded number of rounds. A
lone request that already fills the dispatch passes its inputs straight through with no
concatenation or padding.

### Why coalescing is the throughput lever

Packing many requests into one execution amortizes the fixed per-launch overhead and, for a model
that had to be loaded on demand, the one-time weight transfer, across every item in the batch. On
a representative compute-heavy model, per-image latency drops by roughly three times from batch 1
to batch 32 while images per second more than triples. Coalescing is therefore the reason a
worker can absorb bursts: concurrent requests pile onto a model's queue and the loop converts
them into a small number of full dispatches instead of paying launch cost once per request.

## Cost learning

The scheduler measures the real GPU time of each execution and refines a per-batch-size cost
estimate with an exponential moving average (`cost_ema_alpha`, default 0.2). The estimate
initializes to the first measurement, so it converges immediately rather than drifting up from
zero. It feeds the fair discipline (the cost divisor, so an expensive model does not crowd out
cheaper ones on a marginal fairness edge) and the `edf` laxity check.

Two things keep the estimate honest:

- **Warmup seeds it.** At startup, under the fair discipline, each compiled size is run once on
  zero-filled inputs and the measured time becomes the estimate. Failures are non-fatal: the
  default cost covers any unseeded (model, batch size) pair until the first real dispatch
  measures it. While every candidate is unseeded they all share the default, so the deficit term
  decides. When the on-demand weight cache is enabled with auto-sizing, the startup isolation
  probe doubles as this warmup.
- **Weight load time is excluded.** The measured `compute_time` covers only the model execution,
  not the on-demand weight load that preceded it, so the estimate reflects steady-state
  (resident) execution rather than cold-call latency.

Coalescing is also priced fairly: a `coalescing_discount` (default 0.10) is applied to the
effective cost of a coalesced batch, so a larger dispatch is not penalized for being larger when
it competes against an unbatched one.

## Per-model configuration

Most models need no tuning: with the default `fair` discipline, all-default weights yield uniform
shares. When you want to change that, per-model overrides live under `scheduler.models` in the
node file, which builds the [`ModelSchedConfig`](@ref) entries:

```yaml
global:
  scheduler:
    models:
      resnet50:
        weight: 2.0           # relative compute share under fair (default 1.0)
        max_batch_size: 8     # cap on rows coalesced per dispatch (default uncapped)
```

`weight` sets the model's fair share. `residency` is also settable per model (the initial
residency floor); see [On-demand Weights](on_demand_weights.md) for what the floors mean.
`max_batch_size` caps coalescing rows per dispatch, as described above; it does not change
compiled shapes and never splits a request. The full node-file surface, including the global
`scheduler:` block, is in [Node Configuration](node_config.md).

`max_queue_depth` caps each model's queue independently: a full model rejects new requests
without affecting admission for the others, so one backlogged model cannot starve the rest of the
worker.

### Live policy changes

The control plane can adjust scheduling policy on a running worker. `ReactantServer.set_policy!`
updates
a model's live `weight` (consulted by the fair discipline; available in both residency modes, and
it raises on an unknown model). `ReactantServer.set_residency!` moves a model to a target
residency
floor; it is only meaningful in externally-managed mode and the worker rejects it otherwise. Both
run on the dispatch thread, the sole mutator of residency, and block until applied; control
commands are drained ahead of dispatch selection, so a residency transition takes effect before
the dispatch that may depend on it. `ReactantServer.compact!` frees the resident non-pinned
weights so
the allocator coalesces its free list, then reloads the listed models, defragmenting the device
arena; see [On-demand Weights](on_demand_weights.md).

## Observability

`ReactantServer.scheduler_metrics` snapshots per-model state: dispatch count, requests served,
rows
served, total compute, the current recent-compute EMA, queue depth, queue-wait P50/P99, the
histogram of dispatched batch sizes, the effective max batch, and residency. The counters are
cumulative over the worker's lifetime, so deltas between polls give rates; `dispatch_count`
counts coalesced batch executions, `requests_served` counts individual requests (at least one
per dispatch, more under coalescing), and `rows_served` counts batch-axis rows, the effective
batch size actually executed.

The worker exports these as Prometheus series (`worker_dispatch_total`,
`worker_compute_seconds_total`, `worker_queue_depth`, `worker_queue_wait_seconds`,
`worker_requests_served_total`, `worker_rows_served_total`, `worker_model_max_batch_size`, and
the residency gauges), tagged per model and per GPU, and the node's metrics endpoint aggregates
them. See [Deployment](deployment.md) for the scrape configuration.

`ReactantServer.control_status` is the control-plane view a gateway reads: the worker's
residency mode
and discipline, and for each model its weight, queue depth, cumulative serving counters,
effective max batch, and the wire-facing batch axis (the input name and 1-based axis a gateway
can count items on without decoding a request). A meta is reported as a single unit whose
footprint is the sum of its sub-models, so the gateway packs and routes the group as one. See
[Meta Models](meta_models.md) and [Gateway](gateway.md).

## Configuration reference

The `scheduler:` block in the node file maps to [`SchedulerConfig`](@ref), and entries under
`scheduler.models` map to [`ModelSchedConfig`](@ref). The discipline enum is
[`SchedulingDiscipline`](@ref). Defaults and YAML keys:

```yaml
global:
  scheduler:
    discipline: fair               # fair | fifo | edf (fair is the default)
    ema_halflife_seconds: 30.0     # fair: half-life of the recent-compute EMA
    recency_penalty_cap: 0.25      # fair: clamp on the share-vs-EMA deficit
    coalescing_discount: 0.10      # fair: cost discount applied to coalesced batches
    cost_ema_alpha: 0.2            # fair/edf: smoothing for the learned per-batch-size cost
    max_queue_depth: 1024          # per-model queue cap; a full model rejects new requests
    dispatch_timeout_seconds: 30.0 # per-request execution timeout
    compaction_interval: 0         # worker-local memory compaction every N on-demand loads
    models: {}                     # per-model overrides (weight, residency, max_batch_size)
```

Every field is also settable with an `INFERENCE_SERVER_SCHEDULER_*` environment variable (for
example `INFERENCE_SERVER_SCHEDULER_DISCIPLINE=fifo`). See the [API Reference](api.md) for the
full [`SchedulerConfig`](@ref) documentation.
