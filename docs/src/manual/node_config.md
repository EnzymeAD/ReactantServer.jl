```@meta
CurrentModule = ReactantServer
```

# Node Configuration

A deployment is described by a single node file. It is the only supported config format. It
describes one or more single-GPU workers on one machine and, optionally, the gateway that fronts
them. Each worker reads this same file, resolves its own entry by name, and loads (and can serve)
every bundle in the shared model repository.

A single GPU is just a one-worker node: keep one entry under `workers:` (or omit `workers:`
entirely under the supervisor, below). Growing to more GPUs means adding workers, not changing
the config format.

Under the node supervisor (the container default, see [Docker Deployment](@ref)) the `workers:`
list is optional: omit it and add `gpus: auto` (or an integer count, or an explicit device
list) and the supervisor synthesizes one worker per detected GPU, assigning each its device.
An explicit `workers:` list still wins when present. The keys below describe that explicit form,
which the supervisor also honors.

## Top-level keys

```yaml
# One shared bundle repository. Each immediate subdirectory containing a manifest.yaml is a
# bundle; its directory name is the model name.
model_repo: /var/lib/reactantserver/models

# Worker at index i binds base_port + i unless it sets an explicit `port:`.
base_port: 8080

global:    # defaults merged into every worker (any block may be overridden per worker)
workers:   # one entry per GPU
models:    # optional; pins the named models to device memory on the listed workers
gateway:   # optional; read only by the gateway, never by a worker
```

## Global settings

The `global:` block holds defaults merged into every worker; a worker entry may override any of
these blocks. The sub-blocks map onto the resolved [`ServerConfig`](@ref):

```yaml
global:
  cache_dir: /var/cache/reactantserver
  model_control_mode: dynamic  # dynamic (watch the repo) | static | explicit (control plane)
  model_poll_seconds: 15.0     # repository poll interval in dynamic mode
  runtime:                 # -> RuntimeConfig
    backend: cuda          # cpu or cuda
    mem_fraction: 0.9      # fraction of device memory claimed for the pool (GPU only)
    preallocate: true      # claim the pool up front (GPU only)
    allow_cpu_fallback: false
    weight_cache_fraction: 1.0          # arena fraction for all weights (pinned + on-demand); 0 disables, GPU only
    weight_cache_wiggle_fraction: 0.1   # arena fraction kept free; drives startup peak probe + auto-sizing
  scheduler:               # -> SchedulerConfig
    discipline: fair       # fair | fifo | edf (use fifo or edf behind a gateway running lpt_packing)
    ema_halflife_seconds: 30.0
    max_queue_depth: 1024  # per-model queue cap; a full model rejects new requests
    dispatch_timeout_seconds: 30.0
    compaction_interval: 0 # worker-local: defragment device memory every N on-demand weight loads;
                           # 0 disables (the default). Leave 0 behind a gateway (see On-demand Weights)
    models: {}             # per-model overrides -> ModelSchedConfig (see below)
  endpoints:               # -> EndpointsConfig
    host: 0.0.0.0          # bind all interfaces so the gateway/clients can reach the worker
    max_concurrent_requests: 64  # in-flight RPC cap; 0 = uncapped. Past the cap the worker sheds
                                 # with RESOURCE_EXHAUSTED. Keep it above the gateway's per-worker
                                 # outbound stream limit (worker_client.max_concurrent_streams)
  grpc:                    # -> GrpcConfig
    max_recv_msg_bytes: 536870912   # 512 MiB; max inbound gRPC message (a decode cap, not an allocation)
    max_send_msg_bytes: 536870912   # 512 MiB; max outbound gRPC message
```

The `global.grpc` block is the single node-level place for gRPC message-size limits: every worker
reads it directly, and the supervisor also mirrors it into the embedded gateway (as
`REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` / `_SEND_MSG_BYTES`), so one block sizes the whole node.
Per-component environment overrides still apply (`INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES` for a
worker, `REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` for the gateway, which wins over the mirrored
value).

`model_control_mode` sets how the loaded model set evolves: `dynamic` (the default) watches
the repository and loads, unloads, and reloads bundles online as files change; `static` fixes
the startup set; `explicit` cedes the lifecycle to an external control plane via the worker
control RPCs. `scheduler.discipline` selects the dispatch policy: `fair` shares GPU time
across models by weighted deficit and learned cost, while `fifo` serves the oldest queued
request first. Workers fronted by a gateway in `lpt_packing` mode must run `fifo` or `edf` (not
`fair`), so the gateway stays the placement and fairness authority (see
[Multi-GPU Gateway](multi_gpu_gateway.md)).

`edf` (earliest-deadline-first) serves the model whose most-urgent queued request has the
soonest deadline, where the deadline comes from the request's remaining-budget timeout. A meta
model is not scheduled here (it runs on the request task), but each of its in-flight sub-calls
inherits the meta's deadline, so under `edf` a meta's continuation is ordered ahead of fresher
regular work. It is designed for
deadline-sensitive serving: while every client uses the same deadline it behaves exactly like
`fifo`, and it diverges only to dispatch requests with less budget left ahead of those with more,
so a request close to its deadline is served before a fresher one rather than missing behind it.
`edf` also sheds work it cannot finish within its learned compute cost (laxity), trading some
throughput (batch fragmentation, and no per-model weighting) for meeting more deadlines under
load. Note that because `edf` derives urgency solely from the deadline, issuing **different
per-client deadlines for the same model reorders that model's service and therefore affects
fairness across clients**; keep deadlines uniform to retain `fifo`-like fairness.

Each sub-block corresponds to a typed config struct: [`RuntimeConfig`](@ref),
[`SchedulerConfig`](@ref), [`ModelSchedConfig`](@ref), and [`EndpointsConfig`](@ref). See the
[API Reference](../api/config.md) for every field and its default.

## Workers

```yaml
workers:
  - { name: worker0 }
  - { name: worker1 }
```

`name` is the routing identity (and, under Docker, the compose service name). The listen port
is `base_port + index` unless the worker sets an explicit `port:`.

Under the supervisor (the container default) you do not assign GPUs yourself: it detects the
visible devices, gives each worker one of them, and sets that worker's own
`CUDA_VISIBLE_DEVICES`, so every worker sees a single GPU at ordinal `0`. Influence the assignment
with `gpus:` above (`auto`, a count, or an explicit device list), the `REACTANT_GPUS` environment
variable, or by adding `gpu: N` to a worker entry to pin it to a specific visible device. A
container-level `CUDA_VISIBLE_DEVICES` acts as a coarse filter on which physical GPUs the
supervisor sees (see [Docker Deployment](docker.md)). Running a single worker by hand without the
supervisor (a bare `serve`), the worker uses device ordinal `0`, or `gpu: N` to pick another.

A worker entry may also carry override blocks (for example a `runtime:` block) that merge over
`global`.

## Device pinning (the `models:` map)

```yaml
models:
  resnet50: [worker0, worker1]   # hot on both GPUs
  vsq_coral: [worker0]           # hot on worker0 only
```

Every worker loads (and can serve) every bundle in `model_repo`; the gateway discovers what
each worker serves and schedules requests across them. The optional top-level `models:` map is
a per-model override that pins the named models to device memory on the listed workers for the
lowest latency (it translates into `scheduler.models.<name>.residency: device` on those
workers). Unlisted models stay system-pinned in host RAM and load to the device on demand.
Omit the block entirely to keep every model on-demand. To load only a subset of the repository
on a worker, the resolved config also supports a `models_include` allowlist (empty means load
all).

## Per-model scheduler overrides

Tune individual models under `scheduler.models`, which builds the [`ModelSchedConfig`](@ref)
entries:

```yaml
scheduler:
  models:
    resnet50:
      weight: 2.0                  # relative compute share (default 1.0)
      residency: device            # keep weights GPU-resident for the server's lifetime
      max_batch_size: 8            # cap on rows coalesced per dispatch (default uncapped)
    yolo:
      residency: unpinned          # no host floor; re-materialized from disk on each load
```

`weight` sets the model's fair share. `residency` sets the model's residency floor
(`unpinned`, `system`, or `device`; `pin_to_gpu: true` is a back-compat alias for
`residency: device`). When the on-demand weight cache is enabled, models without an explicit
`residency` default to `system` (weights pinned in host RAM); see
[On-demand Weights](on_demand_weights.md).

`max_batch_size` caps how many rows the scheduler coalesces into one dispatch of the model.
It does not change compiled shapes: the dispatch sizes come from the batch sizes the bundle
was compiled for (a partial fill still pads up to the smallest compiled size), the batch axis
comes from the bundle manifest, and a single request larger than the cap is still served in
one dispatch because requests are never split.

## Gateway configuration

The gateway does not read the node file at all: it is configured by its own standalone
`gateway.yml`, which carries its listen addresses, limits, and a flat `endpoints:` list of
worker `host:port` addresses (see [Multi-GPU Gateway](multi_gpu_gateway.md)). A `gateway.yml`
looks like:

```yaml
listen:
  grpc: "0.0.0.0:8001"
  metrics: "0.0.0.0:8002"
grpc:
  max_recv_msg_bytes: 536870912   # 512 MiB
  max_send_msg_bytes: 536870912
  max_concurrent_requests_per_worker: 64   # inbound cap is this x worker count; 0 = uncapped.
                                           # Sized above the outbound stream limit so a startup
                                           # burst has headroom rather than being shed early
worker_client:
  request_timeout_seconds: 60
  max_concurrent_streams: 32      # outbound in-flight RPCs the gateway multiplexes to one worker
logging:
  level: "info"
  format: "json"
scheduling:
  mode: round_robin               # round_robin | lpt_packing (see Multi-GPU Gateway)
endpoints:                        # worker host:port addresses, across any number of nodes
  - "worker0:8080"
  - "worker1:8081"
```

## Environment-variable overrides

Any worker value can be overridden per process by an environment variable of the form
`INFERENCE_SERVER_<SECTION>_<FIELD>`, for example:

```
INFERENCE_SERVER_ENDPOINTS_PORT=9100
INFERENCE_SERVER_RUNTIME_BACKEND=cpu
INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_FRACTION=0.8
```

List-valued overrides (`INFERENCE_SERVER_MODEL_DIRS`, `INFERENCE_SERVER_MODELS_INCLUDE`) are
colon-separated. Overrides are applied on top of the resolved node config, and the effective
configuration is logged at startup.
