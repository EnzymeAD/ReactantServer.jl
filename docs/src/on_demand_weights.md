# On-demand Weights

Because only one model executes at a time, the GPU does not need every model's weights
resident simultaneously. A GPU sized for a handful of models can therefore serve a much
larger catalog, paying a small transfer cost only when a cold model is first called. This
page is the operational guide to the weight memory system: the two memory tiers, the
self-sizing probe that budgets them, residency control, node-shared host weights, BFC
fragmentation and compaction, and the observability that confirms it all. Dispatch and
batching are the job of [Scheduling](scheduling.md); the reverse proxy and its placement
policy live on [Multi-GPU Gateway](gateway.md).

## Two tiers of weight memory

- **Host RAM (a budgeted cache).** `runtime.host_weight_budget` sets how much host RAM may
  hold copies of weights between GPU loads. The default is `0`: no copies, so a model reloaded
  onto the GPU reads its weights file from disk. Give it a budget when models are swapped, so
  reloads copy from RAM instead (see [Host RAM budget](#Host-RAM-budget) and
  [Use Cases](use_cases.md)). A model with `residency: system` is always kept in host RAM.
- **GPU (managed working set).** Pinned models keep their weights on the GPU for the
  server's lifetime. Every other model is loaded onto the GPU on demand when a request
  arrives, kept resident afterward so repeat requests are free, and evicted under a
  configured GPU byte budget by the cost-aware admission described in
  [Memory fragmentation and compaction](#Memory-fragmentation-and-compaction). Eviction frees the device
  memory immediately through an explicit PJRT buffer release
  (`ReactantServer.free_weights!`).

When the weights are already in RAM, an on-demand GPU load is a single host-to-device
transfer (`ReactantServer.transfer_to_device`) rather than a reload from disk, so its cost is the
model's weight size over the host-to-device bandwidth.

## Enabling it

The main setting is `runtime.device_weight_fraction`, the share of the BFC arena
(`device_memory_fraction * device memory`) that holds resident model weights, pinned plus on-demand. It
defaults to **1.0** (use the whole arena, GPU only) and must be greater than 0: models that do not
fit in it are swapped on and off the GPU. You normally do not set it at all. (In 0.1 the three device
settings were `mem_fraction`, `weight_cache_fraction` and `weight_cache_wiggle_fraction`; they are
now `device_memory_fraction`, `device_weight_fraction` and `device_headroom_fraction`, and the old
names are rejected at startup.)

```yaml
global:
  runtime:
    backend: cuda
    device_weight_fraction: 1.0    # default: use the whole arena, self-sized (see below)
    device_headroom_fraction: 0.1  # default: keep 10% of the arena free as headroom
```

These are the `device_weight_fraction` / `device_headroom_fraction` fields of
[`RuntimeConfig`](@ref), also settable via `INFERENCE_SERVER_RUNTIME_DEVICE_WEIGHT_FRACTION`
and `INFERENCE_SERVER_RUNTIME_DEVICE_HEADROOM_FRACTION` environment variables. The cache
is GPU-only; on CPU there is no arena, so weights stay resident regardless.

### Self-sizing: pinned + scratch + headroom

The fraction is a ceiling on weight memory, not a promise it all fits as cache: each
execution also needs transient device memory (activations, conv/GEMM workspace, IO) on top
of resident weights, and pinned models occupy the arena too. The worker resolves the
on-demand budget for you at startup with `ReactantServer.weight_budget`. It probes each model once
in isolation (loads and runs it), reads the allocator's peak device usage to measure the
worst-case scratch, and sets

```text
on_demand_budget = min(fraction*arena, (1-headroom)*arena - max_scratch) - pinned_weights
```

so pinned weights reserve their share, the measured scratch and the headroom are held
back, and the rest is the weight cache. When everything fits, the cache simply holds all
weights resident (no eviction); when it does not, it is memory-safe. The probe is bounded
one-time startup work and doubles as an execution smoke test; the resolved budget is
logged. Pinned weights or scratch large enough to leave no room are logged as a warning,
since no sizing can fix that.

The measurement is exact per model when the backend can reset the allocator's high-water mark,
which the Reactant backend can on a CUDA device (through the allocator's `ClearMemoryStats`):
before each model the peak is reset, so `peak - in_use_before` is that model's own transient, and
nothing that happened earlier, in particular XLA's autotuning scratch during compile (many times a
model's real run scratch), can inflate it. Pinned models are measured too. A model that cannot be
run at startup falls back to the compiler's static scratch accounting
(XLA's compiled memory statistics), padded. The same measurement runs when the directory
watcher loads a model into a running worker, and the budget is re-resolved on the spot, so neither
a fresh autotune nor a hot-loaded model requires a restart to be sized correctly. Without a
resettable peak (any non-CUDA device) the probe falls back to an ordering-based estimate
that is exact for a clean session but inherits whatever moved the peak before it ran.

## Pinning hot models

A model that must never pay the on-demand transfer cost can be pinned to stay GPU-resident
for the server's lifetime. Pinned models are exempt from eviction; the budget bounds only
the unpinned working set.

```yaml
scheduler:
  models:
    resnet50:
      residency: device      # pin_to_gpu: true is a back-compat alias
```

This is the `residency` field of [`ModelSchedConfig`](@ref) (`unpinned`, `system`, or
`device`; an unspecified model is swapped on demand with host copies governed by the host RAM
budget, or pinned in RAM when that budget is `unlimited`; see [`ResidencyState`](@ref)). Pin the latency-sensitive or highest-traffic models, and let the
long tail load on demand.

## Sharing host weights across same-node workers

With several workers on one node, `runtime.shared_host_weights: true` backs each
model's host copy (pinned or cached) with a node-shared POSIX shared-memory region so the
workers share one copy. The default [`PrivateWeightStore`](@ref) gives each worker its own
copy; [`SharedWeightStore`](@ref) places one content-addressed region per model in
`/dev/shm`, coordinated peer-to-peer through an advisory `flock`, so same-node workers
attach one copy and each transfers it to its own device. `host_materialize` routes through
the configured store, which is what makes this a configuration decision rather than a code
change.

Each shared copy is labeled with the version of the weights file the worker actually opened (its
device, inode, size and modification time), so workers never share a copy of one version under
another version's label. Deploy new weights by writing a new file and renaming it over the old
one: a worker that already loaded the model keeps serving the old bytes until it reloads, and its
shared copy stays labeled as the old version. Writing into a loaded weights file in place (for
example `cp` onto it, which truncates it first) is detected and fails that model's load instead of
copying mixed bytes; weights are read with ordinary file reads rather than a memory mapping, so a
truncated file is an error rather than a crashed worker. When the node cannot back
a new shared copy (`/dev/shm` full, or the container's memory limit reached), the worker logs a
warning and gives that model a private copy instead of failing.

The regions and their lock files are created with mode `666` by default so containers
running as unrelated UIDs can share them; that is world-writable, so set
`runtime.shared_host_weights_mode: "660"` on production and multi-user systems (the server
warns at startup when the shared store runs with the `666` default). Both fields belong to
[`RuntimeConfig`](@ref), with `INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS` and
`INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS_MODE` environment overrides.

## Host RAM budget

`runtime.host_weight_budget` caps the host RAM that holds copies of weights between GPU loads:

```yaml
global:
  runtime:
    host_weight_budget: 48g   # default 0; or 512m, 1.5t, unlimited
```

Sizes take an optional unit, case insensitive: `b`, `k`, `m`, `g`, `t`, or the long forms
(`gb`, `gib`); every unit is binary, so `5g` and `5GB` are both 5 GiB. The default `0` keeps no
copies: weights are in host RAM only while a load reads them and transfers them to the device,
which suits a deployment whose models all stay on the GPU. `unlimited` keeps every model's copy
with no cap (the behavior before 0.2).

The host tier is a cache. A GPU load uses the model's cached host copy when there is one, and
otherwise reads the weights file and keeps the copy if the budget has room. When it is full, the
copies that go first are the ones the GPUs load least often. Each copy carries a count of the
model's loads onto a GPU that halves every hour, and a new copy only displaces copies with a lower
count than its own. This is where a slow disk's cost is weighed: a copy is worth a read saved for
each load it serves, so a model the GPUs evict and reload every few minutes keeps its copy, and a
model loaded once does not push it out. Only loads count: requests served by a model already on the
device do not, so a model that stays on a GPU for hours ages out of host RAM (nothing needs its host
copy while it sits on the device). Being on a device gives no protection. With shared host weights
the count is the node's, kept with each copy in `/dev/shm` and raised by whichever worker loads it. A model with `residency: system` keeps its guarantee: it
stays in RAM, is never evicted, and counts against the budget. When pinned models alone exceed the
budget they load anyway and the worker warns that it is overcommitted. A load that misses the cache
reads the whole model and queues every transfer at once, the same as an unbudgeted load; those
temporary arrays are freed by the GC and are not counted.

Reading a model into host RAM does not hold up the worker. When a request arrives for a model that
is not on the GPU, a background task reads its weights (from the cache, or from the weights file)
while the worker keeps serving the models already on the GPU; once the read finishes, the model's
turn in the queue evicts what it must and transfers the weights, exactly as any load does, so a
model being read never takes GPU memory ahead of requests queued before it. Up to two models per
worker are read ahead at a time, oldest request first; a third waits for a free slot rather than
being read on the loop that runs the GPU. The worker never reads a weights file on that loop: a
read there would hold the GPU idle for every model for as long as the disk takes. If a background
read fails, the requests waiting for it get the error, and the next request starts a fresh read.
A model read ahead keeps its slot only while it has requests to transfer the weights for. When the
model's turn ends without the transfer (its requests failed before it, were cancelled, timed out or
passed their deadline) and nothing more is queued for it, or the model is unloaded or reloaded, the
worker lets go of what it read and the slot passes to the next model; a model with requests still
queued keeps it for them.
Workers on the same node read different models at once; two workers wanting the same model share
one read.

A read that stops making progress is abandoned. Weights on network storage can stop answering (a
hard NFS mount whose server went away, or a shared host-weight lock held by a worker that is stuck),
and a read waiting on them sits in the kernel for good, where nothing can interrupt it. Each read
counts the bytes it reads as they arrive, 8 MiB at a time, and one that reads nothing for
`scheduler.weight_read_stall_seconds` (60 s by default; 0 turns this off) is abandoned: the requests
waiting for the model fail with `UNAVAILABLE`, which the gateway retries on another replica, and the
read slot passes to the next model. Storage that is slow but still moving is never cut off (8 MiB per
limit, 140 KiB/s at the default, is enough). A worker waiting for another worker that is reading the
same model into the node's shared copy counts that worker's progress as its own, so the wait is cut
off only when that read stalls too; any other wait on a shared host-weight lock counts as no
progress. The abandoned read still holds a thread and the memory it was reading into. Until it
returns, its model is not read again, and requests for it fail with `UNAVAILABLE` at once; if it
returns, what it read is released. With four abandoned reads outstanding (a quarter of the worker's default threads when
that is fewer, at least one), the worker starts no read at all, and every request for a model that is
not on its GPU fails with `UNAVAILABLE` at once. Models on the GPU keep serving throughout, and the
worker keeps reporting ready. A request still waits no longer than its dispatch timeout
(`scheduler.dispatch_timeout_seconds`), so with the defaults a request that has waited 30 s for a
stalled read fails with `DEADLINE_EXCEEDED` before the read is abandoned at 60 s; set the stall limit
below the dispatch timeout for such requests to fail over instead. Each abandoned read is logged, and
so is its return; `worker_weight_stalled_reads_total` counts them, and `worker_weight_stalled_reads`
shows the ones that have not returned.

The same holds for every other load. Compaction, and the fallback compaction admission runs, reload
only models with a host copy to upload from; a model without one stays evicted until its next request
reads it in the background. A hot reload from the directory watcher reads the new version's weights
before it takes the GPU loop (and attaches a `residency: system` floor after), and a residency change
on an externally managed worker reads before its control step. That read is watched for progress like
the reads above: one that stalls fails the `SetModelResidency` call with `UNAVAILABLE`, and the next
residency change, for any other model, does not wait for it.

A gateway moving a model onto a worker can read it there ahead of time: the worker's
`StageWeights` RPC reads the named models into host memory on background tasks, holds what it read
until each model's first dispatch uploads it (or the hold expires, at most 10 minutes), and returns
once the reads end. Held bytes are capped at half the host weight budget (4 GiB without one); past
the cap a model is reported `deferred`. A model the directory watcher unloads or reloads while the
call runs is reported `failed` and nothing is held for it, so a new version is read for its own
requests as usual. A read that stalls is abandoned as above and its model reported `failed`, so the
call returns within the stall limit even when the read never does; so is a model the worker will not
read because reads have stalled. See [primed moves](gateway.md#Primed-moves).

How the budget is shared depends on `shared_host_weights`:

- **Shared** (`true`): one budget for the whole node. Every worker may use all of it, each copy
  lives once in `/dev/shm` however many workers load from it, and the copies are the ledger: a
  worker sums the regions under a node-wide lock (`/dev/shm/rsw-budget.lock`, held only for that
  bookkeeping, never across a disk read or the reservation of a copy's memory) and evicts copies
  no worker is using. A worker holds a copy from the moment its read finishes until its transfer
  to the GPU ends, so a copy in use is never evicted. Copies survive worker
  restarts, so the cache is warm after a restart, and each worker trims the node to the budget at
  startup.
- **Private** (`false`): each worker keeps its own copies and gets an even share,
  `host_weight_budget ÷ host_weight_budget_workers`, with no coordination. A copy belongs to the
  loaded version of its model: a hot reload reads the new version's weights file and never uploads
  the copy the previous version left.
  `host_weight_budget_workers` defaults to 1; the node supervisor sets it to the node's worker
  count, so set it yourself only when running several workers without the supervisor.

At startup, after measuring every model, the worker loads as many on-demand models onto the GPU as
fit, largest first, so a deployment whose models all fit never reads a weights file after startup.
After that, a model reloaded from disk because it had no host copy logs a
`model weights reloaded from disk` warning, once per model, so slow swaps have a visible cause.

The budget applies to GPU workers; a CPU worker keeps every model's weights resident, so the
setting is ignored there (with a warning when it is not 0).

## Memory fragmentation and compaction

When you size a GPU to squeeze the largest possible working set onto it, fragmentation
becomes a real constraint. The device pool is one BFC (best-fit-with-coalescing) arena,
claimed up front when `runtime.preallocate: true`. As the weight cache loads and evicts
models of different sizes over time, the freed regions return to the arena's free list but
do not always sit next to each other, so the arena can hold plenty of free bytes in total
yet have no single gap large enough for the next model's weights. The load then fails or
forces extra eviction even though the memory is technically there. The tighter you pack the
GPU, the sooner this bites.

Pinned models are not the problem here: they are loaded once at startup, before any
on-demand traffic, so they sit at the base of the arena and never move. Fragmentation
accumulates only in the on-demand working set above them. Compaction targets exactly that
region. It frees every resident on-demand weight buffer at once, so the allocator coalesces
the now-contiguous free space back into one large region above the pinned base. Pinned
models are left in place: they are never freed and never re-read from disk. Host floors
(the system-pinned RAM copies) are also left untouched, so when an on-demand model reloads
it is a fast host-to-device transfer rather than a disk re-materialization
(`ReactantServer.compact!` implements this on the worker).

### Admission: evicting only what a load needs

Before a worker loads a model it checks, from the allocator's own statistics, that the
model's weights and the scratch of the heaviest resident execution will both fit, in bytes
and contiguously, and evicts only what is needed to make that true. It decides before it
uploads because a refused device allocation is slow to discover: XLA's allocator waits about
10 seconds for memory to be released before it reports one.

1. **Feasibility.** A model that cannot fit even with every evictable model gone (weights
   plus scratch plus the memory no eviction frees exceed the device) is rejected at once,
   without evicting anything, and on a serving worker the requests for it fail with
   `UNAVAILABLE`. A serving worker measures the memory no eviction frees: the allocator's
   bytes in use less what the evictable models hold, so the device-pinned models count at
   their allocated size, and so does any other buffer the weight cache does not own. At
   startup the check counts the device-pinned models' weights alone. The worker loads its
   on-demand models one at a time there to measure them (except
   under `model_control_mode: explicit`), and a model that fails this check stops the
   worker from starting: under the node supervisor it restarts into the same failure, and
   none of its models serve until the bundle is removed. A hot load of such a model in
   `dynamic` mode fails too, and the watcher retries it every second poll, recompiling it
   each time; in between, the model stays listed and its requests fail with `UNAVAILABLE`.
   Check that each model fits beside the device-pinned models before you deploy it.
2. **Room for the weights.** Free memory must hold the weights plus the largest scratch any
   resident model needs, the largest free block must hold all of the weights, and the
   on-demand byte budget must allow them. "Hold" follows the allocator's split rule: it
   splits a free block for a tensor only when the block is at least twice the tensor or
   the remainder is at least 128 MiB, so a block of exactly the weights' size can be too
   small. Admission replays that rule over the model's tensors in upload order. A byte shortfall evicts the cheapest *set* of
   models that covers it; a contiguity shortfall evicts the cheapest single model at a time.
3. **Upload**, then **the scratch guarantee**: the largest free block must still hold the
   largest scratch of every resident model, the new one included, so every resident model
   can run. Admission evicts until it does.

Which models to evict is a cost decision. Each model's priority is the device time a reload would
cost per byte, scaled by how often it has been dispatched since it was loaded and aged by how
long ago it was last used (GreedyDual-Size-Frequency). A costly, frequently used model
stays; one nobody has called in a while goes, and among equals the least recently used goes
first, as with LRU. The device time is the model's measured upload time (its bytes over the
worker's measured upload bandwidth before it has one). Reading the weights file is not counted,
because that read runs in the background and never holds the GPU: what a slow read costs is latency
for the requests that wait for a model with no host copy, and that is decided when the host copy is
evicted, not the device copy. Since upload time grows with size, the priority comes down to how
often a model is used, whatever its size: a large model called often stays, and a large model
called rarely gives way to the smaller ones it would crowd out. The worker exports the two halves of
its load time as
`worker_weight_upload_seconds_total` and `worker_weight_read_seconds_total`, with
`worker_weight_read_bytes_total`. `worker_weight_evicts_total` carries a `reason`: `bytes` for a
byte deficit, `contiguity` when enough bytes were free but no block could hold the weights
(fragmentation costing a load), and `scratch` for keeping the execution scratch block free after an
upload; per model, `worker_model_loads_total`, `worker_model_reads_total` and
`worker_model_evicts_total` show which models churn. Reads ahead of the dispatch are counted by
`worker_weight_stages_total`, `worker_weight_stage_seconds_total` and
`worker_weight_stage_failures_total`, and `worker_weight_staging_models{state}` shows the ones in
progress. Choosing a set rather than evicting one model at a time means one large
idle model is evicted in place of two smaller busy ones. Models with requests already queued
are evicted only when nothing else frees enough. The search is allocation-free and bounded:
measured at 64 resident models it takes about 6 to 22 microseconds.

Compaction (below) is the fallback when eviction alone cannot make room: the worker frees
every evictable model, checks from the statistics that the new model fits on its own,
reloads the most valuable of them largest first together with the new model, and retries
once. A former resident is reloaded only if the whole refill, replayed under
the allocator's split rule, still leaves the scratch block free; one that fails to reload
anyway stays evicted rather than failing the request. Device-pinned models never move, so a pinned model that splits the
device can make a large model unplaceable; place pinned models with that in mind. Without
device statistics (the CPU backend) only the byte budget applies.

When the new model does not fit even after compaction, the worker does not ask the allocator
for it, so it never pays the 10 second wait: the model is left off the device, the models
compaction freed (and the ones its admission evicted) are reloaded as far as they fit, and the
request fails with `UNAVAILABLE` and an admission error naming the reason, so a gateway tries
another replica. If the upload succeeds but then leaves no room for the scratch block, the
upload is undone the same way, rather than leaving a model resident that could not run.

A model that failed admission is not tried again at once. Until something changes that could make
it fit (device memory that no eviction frees is released, a device-pinned model is unloaded, a
residency changes, or the model itself is reloaded), its requests fail at once with `UNAVAILABLE`:
nothing is evicted or uploaded, and its weights are not read into host memory for them. Unloading
or loading other on-demand models does not end the refusal, since it cannot change what fits.
`worker_weight_admission_failures_total` counts the admissions that failed and
`worker_weight_admission_refusals_total` the ones refused at once. When memory freed elsewhere ends
a refusal, the request that finds out has no weights read ahead to upload, so it fails with
`UNAVAILABLE` as well, and the model's next request has them read in the background and is
admitted.

Whatever drives compaction frees the on-demand region the same way; what differs is *when*
it fires and *what* gets reloaded eagerly. Underneath, compaction is a worker control RPC,
`CompactMemory` on the `ControlService`, that frees the on-demand region and reloads a list
of models (empty means free only; a listed model with no host copy to upload from is left for its
next request to read in the background, since compaction never reads a weights file on the GPU
loop); both the standalone trigger and the gateway use it.
There are two ways to drive it, matched to the two deployment shapes.

### Standalone worker (no gateway)

A worker runs compaction itself on a cadence counted in **weight-cache loads**, set under
`scheduler:`:

```yaml
scheduler:
  compaction_interval: 200   # compact every 200 on-demand loads (0 disables, the default)
```

A load is what places a variable-size weight block on the device, so loads are what
fragment the arena; a dispatch to an already-resident model does not, which is why the
cadence counts loads rather than requests or time. The worker trigger always refills lazily:
it frees the on-demand region and lets models come back as requests arrive. It is **off by
default**, and a gateway-fronted worker should leave it off, so the gateway is the sole
authority on compaction in a gateway deployment (next section). It is the
`compaction_interval` field of [`SchedulerConfig`](@ref), also exposed as
`INFERENCE_SERVER_SCHEDULER_COMPACTION_INTERVAL`.

### Behind a gateway

In a gateway deployment workers leave `scheduler.compaction_interval` at `0`. By default
the gateway leaves compaction to each worker's admission as well. It can instead tie
compaction to **placement changes**, the fleet-level event that churns worker memory: when
the `lpt_packing` scheduler repacks and moves a model from one worker to another, the new
worker loads it and the old worker's copy goes cold. Two settings under the gateway's
`scheduling:` block control it:

```yaml
scheduling:
  mode: lpt_packing
  compaction_mode: off         # off (default) | lazy | preload
  compaction_interval: 1       # every placement-changing repack (see below)
```

`compaction_mode` selects whether a repack compacts each worker whose placement changed,
and what that worker reloads afterwards:

- **`off`** (the default) sends nothing on a repack. Each worker's admission evicts only
  what a load needs and compacts by itself only when eviction cannot make a model fit.
- **`lazy`** frees the worker's on-demand region and reloads nothing. Models the worker no
  longer serves are dropped immediately; the ones it still serves reload on their next
  request, so each pays its upload only if it is actually used.
- **`preload`** frees the region and then immediately reloads the models the repack just
  assigned to that worker, largest first, so the new placement is warm before traffic
  arrives. The worker's queue waits once while that set uploads. The gateway computes the
  per-worker list from the placement, which is why this is a gateway setting.

Renamed in 0.2: `eager` is now `lazy`, `scheduled` is now `preload`, and `on_demand` is now
`off` (they were the same behavior). The old names fail at startup with the replacement in
the error; there is no alias, so update `gateway.yml` or
`REACTANT_GATEWAY_SCHEDULING_COMPACTION_MODE` before upgrading.

`compaction_interval` is the cadence in repacks (default `1`, so every placement-changing
repack compacts). Because the demand signal is smoothed, most repacks recompute the same
placement and do nothing; the counter advances on every repack but the fan-out only fires
on the first placement-changing repack at or after the interval, so it can land a little
later than exactly N. Both settings are also exposed as
`REACTANT_GATEWAY_SCHEDULING_COMPACTION_MODE` and
`REACTANT_GATEWAY_SCHEDULING_COMPACTION_INTERVAL`. A single client `CompactMemory` call to
the gateway also fans out to every worker on demand, independent of the repack cadence, for
a one-off fleet defragment.

### Cost and when to enable

When every model fits on the GPU nothing churns, so compaction has no working set to defragment.
When it does run it has a cost: freeing the on-demand region drops models
that were warm, so they pay a host-to-device reload (on next request for `lazy`, or up
front for `preload`).

The default, `off`, compacts only when a worker's admission cannot otherwise place a model,
so models that are still warm are not dropped on every placement change. Watch
`worker_weight_fallback_compactions_total` for how often that happens,
`worker_weight_admission_failures_total` for models that could not be placed at all, and
`worker_weight_admission_refusals_total` for the requests refused because such a model still
cannot be placed.
`lazy` compacts every affected worker on every placement-changing repack
(`compaction_interval: 1`), which pays a one-time reload on the first request per moved
model; raise `compaction_interval` to fire it less often, or use `preload` to warm the new
placement up front instead.

## Observability

The `ReactantServer.Scheduler` exposes weight-cache residency and load/evict counters alongside
its dispatch metrics; read them with `ReactantServer.scheduler_metrics`
(`ReactantServer.weight_cache_stats` is the underlying snapshot). Use them to confirm that hot
models stay resident and that the eviction rate is acceptable for your budget. Coalescing
(packing many requests into one execution) amortizes a one-time on-demand transfer across
every item in the batch, so on-demand loading and batching reinforce each other.

Each compaction also logs a `memory compacted` line with the device free space and
on-demand budget before and after, plus the largest free block and fragmentation before and
after, and the weight cache tracks a `compactions` counter. Compare the before/after
largest free block to confirm that compaction actually recovered contiguous space on your
hardware: it relies on the allocator returning freed buffers to the arena and coalescing
them, which the log lets you verify rather than assume. Models named for reload (`preload`,
or a `CompactMemory` reload list) are reloaded largest first, so the refill packs tightly
instead of leaving gaps.

With `JULIA_DEBUG=ReactantServer`, every on-demand load and eviction also logs a
`memory: weight load` or `memory: weight evict` line with raw byte counts: the model's
allocator-rounded weight footprint, its largest single weight tensor, its measured
scratch, its average load time, and the device's bytes in use, largest free block and
fragmentation before and after. These lines are meant to be parsed into a
fragmentation-over-time series.

The worker also exports the BFC allocator's live numbers as Prometheus gauges (aggregated
through the gateway's `/metrics`): `worker_device_memory_in_use_bytes`,
`worker_device_memory_free_bytes`, `worker_device_memory_peak_in_use_bytes` (the session
high-water, your empirical scratch + resident ceiling), and
`worker_device_memory_pool_bytes`. The peak gauge is reset once compilation finishes and again
before each probed model, so after startup it tracks execution only; a rising peak over a long
uptime is the signal to revisit the budget.

Fragmentation is observable directly: `worker_device_memory_largest_free_block_bytes` is the
largest single allocation the allocator can satisfy right now, and
`worker_device_memory_fragmentation` is `1 - largest_free_block / free_bytes` (0 when the
free memory is one block, approaching 1 as it splinters). A model whose largest weight tensor
or execution scratch exceeds the largest free block cannot be placed even when the free byte
count looks ample, which is the condition compaction (above) repairs. Allocation failures
are expensive to discover: XLA's allocator waits about 10 seconds for memory to be released
before it reports one.

Under a host RAM budget, `worker_host_weight_budget_bytes` is the budget the worker enforces (the
node's with shared host weights, its share otherwise), `worker_host_weight_cached_bytes` the host
RAM it counts, and `worker_host_weight_hits_total` and `worker_host_weight_misses_total` how many
loads came from a cached copy or read the weights file.

Per model, `worker_model_scratch_bytes` is the scratch measured at startup and
`worker_model_load_seconds` the moving average of its on-demand load time.
`worker_weight_load_bytes_total` divided by `worker_weight_load_seconds_total` gives the
effective host-to-device bandwidth of weight loads.
