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
otherwise reads the weights file and keeps the copy if the budget has room, evicting the least
recently *loaded* copies first. Only a load from the cache counts as use: requests served by a model already on the
device do not, so a model that stays on a GPU for hours ages out of host RAM (nothing needs its
host copy while it sits on the device), and the models the GPUs load most often stay cached.
Being on a device gives no protection. A model with `residency: system` keeps its guarantee: it
stays in RAM, is never evicted, and counts against the budget. When pinned models alone exceed the
budget they load anyway and the worker warns that it is overcommitted. A load that misses the cache
reads the whole model and queues every transfer at once, the same as an unbudgeted load; those
temporary arrays are freed by the GC and are not counted.

How the budget is shared depends on `shared_host_weights`:

- **Shared** (`true`): one budget for the whole node. Every worker may use all of it, each copy
  lives once in `/dev/shm` however many workers load from it, and the copies are the ledger: a
  worker sums the regions under a node-wide lock (`/dev/shm/rsw-budget.lock`, held only for that
  bookkeeping, never across a disk read) and evicts copies no worker is using. A worker holds a
  copy only while transferring from it, so a copy in use is never evicted. Copies survive worker
  restarts, so the cache is warm after a restart, and each worker trims the node to the budget at
  startup.
- **Private** (`false`): each worker keeps its own copies and gets an even share,
  `host_weight_budget ÷ host_weight_budget_workers`, with no coordination.
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
   plus the device-pinned models plus scratch exceed the device) is rejected at once,
   without evicting anything.
2. **Room for the weights.** Free memory must hold the weights plus the largest scratch any
   resident model needs, the largest free block must hold all of the weights, and the
   on-demand byte budget must allow them. A byte shortfall evicts the cheapest *set* of
   models that covers it; a contiguity shortfall evicts the cheapest single model at a time.
3. **Upload**, then **the scratch guarantee**: the largest free block must still hold the
   largest scratch of every resident model, the new one included, so every resident model
   can run. Admission evicts until it does.

Which models to evict is a cost decision. Each model's priority is its expected reload time
per byte, scaled by how often it has been dispatched since it was loaded and aged by how
long ago it was last used (GreedyDual-Size-Frequency). A costly, frequently used model
stays; one nobody has called in a while goes, and among equals the least recently used goes
first, as with LRU. Choosing a set rather than evicting one model at a time means one large
idle model is evicted in place of two smaller busy ones. Models with requests already queued
are evicted only when nothing else frees enough. The search is allocation-free and bounded:
measured at 64 resident models it takes about 6 to 22 microseconds.

Compaction (below) is the fallback when eviction alone cannot make room: the worker frees
every evictable model, reloads the most valuable of them largest first together with the
new model, and retries once; if that fails too the request fails with an admission error
naming the reason. Device-pinned models never move, so a pinned model that splits the
device can make a large model unplaceable; place pinned models with that in mind. Without
device statistics (the CPU backend) only the byte budget applies.

Whatever drives compaction frees the on-demand region the same way; what differs is *when*
it fires and *what* gets reloaded eagerly. Underneath, compaction is a worker control RPC,
`CompactMemory` on the `ControlService`, that frees the on-demand region and reloads a list
of models (empty means free only); both the standalone trigger and the gateway use it.
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
cadence counts loads rather than requests or time. The worker trigger is always eager: it
frees the on-demand region and lets it refill lazily as requests arrive. It is **off by
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
  compaction_mode: on_demand   # on_demand (default) | eager | off | scheduled
  compaction_interval: 1       # every placement-changing repack (see below)
```

`compaction_mode` selects whether a repack compacts and what each affected worker reloads
eagerly after the free:

- **`on_demand`** (the default) sends nothing on a repack. Each worker's admission evicts
  only what a load needs and compacts by itself only when eviction cannot make a model fit.
- **`eager`** frees the on-demand region on each worker whose placement changed and lets it
  refill with live traffic. Models the worker no longer serves are dropped immediately; the
  ones it still serves reload on their next request. This was the default before admission
  existed, and stays available for one release.
- **`off`** disables gateway-driven compaction.
- **`scheduled`** also reloads the set of models the repack just assigned to that worker,
  so the worker's new placement is warm right away instead of cold-loading on first
  request. The gateway computes that per-worker list from the placement, which is why this
  is a gateway concept.

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
that were warm, so they pay a host-to-device reload (lazily on next request for `eager`,
or up front for `scheduled`).

The default, `on_demand`, compacts only when admission cannot otherwise place a model, so
models that are still warm are not dropped on every placement change. Watch
`worker_weight_fallback_compactions_total` for how often that happens and
`worker_weight_admission_failures_total` for models that could not be placed at all. The
older `eager` mode compacts every affected worker on every placement-changing repack
(`compaction_interval: 1`), which pays a one-time reload on the first request per moved
model; raise `compaction_interval` to fire it less often, or use `scheduled` to warm the
new placement up front instead of lazily.

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
them, which the log lets you verify rather than assume. Models named for an eager reload
are reloaded largest first, so the refill packs tightly instead of leaving gaps.

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
