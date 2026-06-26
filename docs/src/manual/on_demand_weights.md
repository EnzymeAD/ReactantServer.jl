```@meta
CurrentModule = ReactantServer
```

# On-demand Weights

Because only one model executes at a time, the GPU does not need every model's weights resident
simultaneously. This lets a GPU sized for a handful of models serve a much larger catalog,
paying a small transfer cost only when a cold model is first called. The
[Architecture](../design/architecture.md) page covers the rationale; this page is the
operational guide.

## Two tiers of weight memory

- **Host RAM (resident by default).** With the on-demand cache enabled, every model's weights
  default to system-pinned (`residency: system`): materialized once from disk into host RAM at
  startup and kept there. Host memory is plentiful and cheap relative to GPU memory, which
  removes disk from the hot path. A model can opt out with `residency: unpinned`, paying a
  mmap re-materialization on each on-demand load instead.
- **GPU (managed working set).** Pinned models keep their weights on the GPU for the server's
  lifetime. Every other model is loaded onto the GPU on demand when a request arrives, kept
  resident afterward so repeat requests are free, and evicted under a configured GPU byte budget
  using a least-recently-used policy. Eviction frees the device memory immediately through an
  explicit PJRT buffer release.

Because the weights are already in RAM, an on-demand GPU load is a single host-to-device
transfer rather than a reload from disk: tens of milliseconds even for the largest models, the
same order of magnitude as a single inference.

## Enabling it

There is one knob: `runtime.weight_cache_fraction`, the fraction of the BFC arena
(`mem_fraction * device memory`) devoted to all weights, pinned plus on-demand. It defaults to
**1.0** (use the whole arena, GPU only), and `0` disables the cache so every model's weights stay
resident. You normally do not set it at all.

```yaml
global:
  runtime:
    backend: cuda
    weight_cache_fraction: 1.0         # default: use the whole arena, self-sized (see below)
    weight_cache_wiggle_fraction: 0.1  # default: keep 10% of the arena free as headroom
```

These are the `weight_cache_fraction` / `weight_cache_wiggle_fraction` fields of
[`RuntimeConfig`](@ref), also settable via `INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_*` environment
variables. The cache is GPU-only; on CPU there is no arena, so weights stay resident regardless.

### Self-sizing: pinned + scratch + wiggle

The fraction is a ceiling on weight memory, not a promise it all fits as cache: each execution
also needs transient device memory (activations, conv/GEMM workspace, IO) on top of resident
weights, and pinned models occupy the arena too. The worker resolves the on-demand budget for you
at startup. It probes each model once in isolation (loads and runs it), reads the allocator's peak
device usage to measure the worst-case scratch, and sets

```
on_demand_budget = min(fraction*arena, (1-wiggle)*arena - max_scratch) - pinned_weights
```

so pinned weights reserve their share, the measured scratch and the wiggle slack are held back,
and the rest is the on-demand cache. When everything fits, the cache simply holds all weights
resident (no eviction); when it does not, it is memory-safe. The probe is bounded one-time startup
work and doubles as an execution smoke test; the resolved budget is logged. Pinned weights or
scratch large enough to leave no room are logged as a warning, since no sizing can fix that.

## Pinning hot models

A model that must never pay the on-demand transfer cost can be pinned to stay GPU-resident for
the server's lifetime. Pinned models are exempt from eviction;
the budget bounds only the unpinned working set.

```yaml
scheduler:
  models:
    resnet50:
      residency: device      # pin_to_gpu: true is a back-compat alias
```

This is the `residency` field of [`ModelSchedConfig`](@ref) (`unpinned`, `system`, or
`device`; unspecified models default to `system` when the cache is enabled). Pin the
latency-sensitive or highest-traffic models, and let the long tail load on demand.

## Sharing host weights across same-node workers

With several workers on one node, `runtime.shared_host_weights: true` backs each system-pinned
model's host copy with a node-shared POSIX shared-memory region so the workers share one copy.
The regions and their lock files are created with mode `666` by default so containers running
as unrelated UIDs can share them; that is world-writable, so set
`runtime.shared_host_weights_mode: "660"` on production and multi-user systems (the server
warns at startup when the shared store runs with the `666` default).

## Memory fragmentation and compaction

When you size a GPU to squeeze the largest possible working set onto it, fragmentation becomes a
real constraint. The device pool is one BFC (best-fit-with-coalescing) arena, claimed up front
when `runtime.preallocate: true`. As the on-demand cache loads and evicts models of different
sizes over time, the freed regions are returned to the arena's free list but do not always sit
next to each other, so the arena can hold plenty of free bytes in total yet have no single gap
large enough for the next model's weights. The load then fails or forces extra eviction even
though the memory is technically there. The tighter you pack the GPU, the sooner this bites.

Pinned models are not the problem here: they are loaded once at startup, before any on-demand
traffic, so they sit at the base of the arena and never move. Fragmentation accumulates only in
the on-demand working set above them. Compaction targets exactly that region. It frees every
resident on-demand weight buffer at once, so the allocator coalesces the now-contiguous free
space back into one large region above the pinned base. Pinned models are left in place: they are
never freed and never re-read from disk. Host floors (the system-pinned RAM copies) are also left
untouched, so when an on-demand model reloads it is a fast host-to-device transfer rather than a
disk re-materialization.

Whatever drives compaction frees the on-demand region the same way; what differs is *when* it
fires and *what* gets reloaded eagerly. Underneath, compaction is a worker control RPC,
`CompactMemory` on the `ControlService`, that frees the on-demand region and reloads a list of
models (empty means free only); both the standalone trigger and the gateway use it. There are two
ways to drive it, matched to the two deployment shapes.

#### Standalone worker (no gateway)

A worker runs compaction itself on a cadence counted in **weight-cache loads**, set under
`scheduler:`:

```yaml
scheduler:
  compaction_interval: 200   # compact every 200 on-demand loads (0 disables, the default)
```

A load is what places a variable-size weight block on the device, so loads are what fragment the
arena; a dispatch to an already-resident model does not, which is why the cadence counts loads
rather than requests or time. The worker trigger is always eager: it frees the on-demand region
and lets it refill lazily as requests arrive. It is **off by default**, and a gateway-fronted
worker should leave it off, so the gateway is the sole authority on compaction in a gateway
deployment (next section). It is also exposed as `SCHEDULER_COMPACTION_INTERVAL`.

#### Behind a gateway

In a gateway deployment the gateway owns compaction; workers leave `scheduler.compaction_interval`
at `0`. The gateway ties compaction to **placement changes**, the fleet-level event that churns
worker memory: when the `lpt_packing` scheduler repacks and moves a model from one worker to
another, the new worker loads it and the old worker's copy goes cold. Two settings under the
gateway's `scheduling:` block control it:

```yaml
scheduling:
  mode: lpt_packing
  compaction_mode: eager       # eager (default) | off | scheduled
  compaction_interval: 1       # every placement-changing repack (see below)
```

`compaction_mode` selects what each affected worker reloads eagerly after the free:

- **`eager`** (the default) frees the on-demand region on each worker whose placement changed and
  lets it refill with live traffic. Models the worker no longer serves are dropped immediately; the
  ones it still serves reload on their next request.
- **`off`** disables gateway-driven compaction.
- **`scheduled`** also reloads the set of models the repack just assigned to that worker, so the
  worker's new placement is warm right away instead of cold-loading on first request. The gateway
  computes that per-worker list from the placement, which is why this is a gateway concept.

`compaction_interval` is the cadence in repacks (default `1`, so every placement-changing repack
compacts). Because the demand signal is smoothed, most repacks recompute the same placement and do
nothing; the counter advances on every repack but the fan-out only fires on the first
placement-changing repack at or after the interval, so it can land a little later than exactly N.
Both settings are also exposed as
`REACTANT_GATEWAY_SCHEDULING_COMPACTION_MODE` and `REACTANT_GATEWAY_SCHEDULING_COMPACTION_INTERVAL`.
A single client `CompactMemory` call to the gateway also fans out to every worker on demand,
independent of the repack cadence, for a one-off fleet defragment.

#### Cost and when to enable

With the on-demand cache disabled (`weight_cache_fraction: 0`) every model is permanently resident
from startup and nothing churns, so compaction has no working set to defragment and is a no-op. When
it does run it has a cost: freeing the on-demand region drops models that were warm, so they pay a
host-to-device reload (lazily on next request for `eager`, or up front for `scheduled`).

The default is `eager` on every placement-changing repack (`compaction_interval: 1`), which keeps
fragmentation from ever accumulating: whenever placement shifts, the affected workers defragment and
refill from live traffic. The trade is a one-time reload on the first request per moved model, which
the relatively infrequent repack cadence (`rebalance_compute_seconds`, 300 GPU-seconds by default)
keeps modest. To cut reloads further, raise `compaction_interval` so it fires less often, switch to
`scheduled` to warm the new placement up front instead of lazily, or set `compaction_mode: off` to
disable it and watch for allocation failures under memory pressure (the signal that fragmentation
has become the bottleneck).

## Observability

The [`Scheduler`](@ref) exposes weight-cache residency and load/evict counters alongside its
dispatch metrics; read them with [`scheduler_metrics`](@ref). Use them to confirm that hot
models stay resident and that the eviction rate is acceptable for your budget. Coalescing
(packing many requests into one execution) amortizes a one-time on-demand transfer across every
item in the batch, so on-demand loading and batching reinforce each other.

Each compaction also logs a `memory compacted` line with the device free space and on-demand
budget before and after, and the weight cache tracks a `compactions` counter. Compare the
before/after device free figures to confirm that compaction actually recovered contiguous space
on your hardware: it relies on the allocator returning freed buffers to the arena and coalescing
them, which the log lets you verify rather than assume.

The worker also exports the BFC allocator's live numbers as Prometheus gauges (aggregated through
the gateway's `/metrics`): `worker_device_memory_in_use_bytes`, `_free_bytes`,
`_peak_in_use_bytes` (the session high-water, your empirical scratch + resident ceiling), and
`_pool_bytes`. The peak gauge is what the startup auto-sizing measured; a rising peak over a long
uptime is the signal to revisit the budget. Note the GPU BFC allocator does not report a
largest-contiguous-free-block figure, so fragmentation is not directly observable here; in
practice it surfaces as an allocation failing despite ample reported free bytes, which is what
compaction (above) addresses.
