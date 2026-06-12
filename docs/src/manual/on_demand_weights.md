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

Set the GPU byte budget for on-demand (unpinned) weights via `runtime.weight_cache_bytes`. A
value of `0` keeps every model resident (the original behavior); any positive value enables the
on-demand cache and bounds the device memory used for unpinned models.

```yaml
global:
  runtime:
    backend: cuda
    weight_cache_bytes: 8589934592   # 8 GiB budget for on-demand weights
```

This is the `weight_cache_bytes` field of [`RuntimeConfig`](@ref). It can also be set with the
`INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_BYTES` environment variable.

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

## Observability

The [`Scheduler`](@ref) exposes weight-cache residency and load/evict counters alongside its
dispatch metrics; read them with [`scheduler_metrics`](@ref). Use them to confirm that hot
models stay resident and that the eviction rate is acceptable for your budget. Coalescing
(packing many requests into one execution) amortizes a one-time on-demand transfer across every
item in the batch, so on-demand loading and batching reinforce each other.
