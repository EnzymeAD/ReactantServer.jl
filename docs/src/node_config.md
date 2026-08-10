# Node Configuration

A deployment is described by a single node file. It is the only supported config format for
workers. The file describes one or more single-GPU workers on one machine and, optionally, the
gateway that fronts them. Each worker reads this same file, resolves its own entry by name, and
loads (and can serve) every bundle in the shared model repository.

A single GPU is just a one-worker node: keep one entry under `workers:` (or omit `workers:`
entirely under the supervisor, below). Growing to more GPUs means adding workers, not changing
the config format.

Under the node supervisor (the supervisor default, see [Deployment](deployment.md)) the
`workers:` list is optional: omit it and add `gpus: auto` (or an integer count, or an explicit
device list) and the supervisor synthesizes one worker per detected GPU, assigning each its
device. An explicit `workers:` list still wins when present. The keys below describe that
explicit form, which the supervisor also honors.

## Top-level keys

```yaml
# One shared bundle repository. Each immediate subdirectory containing a manifest.yaml is a
# bundle; its directory name is the model name.
model_repo: /var/lib/reactantserver/models

# Worker at index i binds base_port + i unless it sets an explicit `port:`.
base_port: 8080

# Optional Prometheus metrics port per worker (metrics_base_port + i): worker0 -> 9100,
# worker1 -> 9101, and so on. Each worker serves /metrics, /healthz, /readyz on it.
# Absent disables per-worker metrics.
metrics_base_port: 9100

# Supervisor-only: `auto` (the default), an integer device count, or an explicit device list
# (ordinals or GPU UUIDs). The supervisor synthesizes one worker per selected device.
# gpus: auto

global:    # defaults merged into every worker (any block may be overridden per worker)
workers:   # optional under the supervisor; one entry per GPU
models:    # optional; pins the named models to device memory on the listed workers
gateway:   # optional; read only by the gateway, never by a worker
```

- `model_repo` (required) is the shared bundle repository. Each immediate subdirectory that
  contains a `manifest.yaml` is a bundle, and its directory name is the model name. It resolves
  into every worker's `model_dirs` (a single entry).
- `base_port` is the first worker's gRPC port. Worker at index `i` binds `base_port + i`
  (`worker0` -> `base_port`) unless that worker sets an explicit `port:`.
- `metrics_base_port` (optional) derives each worker's Prometheus metrics port the same way:
  `metrics_base_port + i`. Absent, per-worker metrics are disabled.
- `gpus` (optional, supervisor-only) selects how many workers the supervisor synthesizes when
  no `workers:` list is present: `auto` (the default) enumerates the visible devices, an
  integer is a device count, and a list gives explicit device selectors (ordinals or GPU
  UUIDs). The `REACTANT_GPUS` environment variable overrides this key.
- `global` holds the defaults merged into every worker; any block may be overridden per worker.
- `workers` lists the worker entries (optional under the supervisor).
- `models` optionally pins the named models to device memory on the listed workers.
- `gateway` is read only by the gateway, never by a worker.

## Global settings

The `global:` block holds defaults merged into every worker; a worker entry may override any of
these blocks. The sub-blocks map onto the resolved [`ServerConfig`](@ref):

```yaml
global:
  cache_dir: /var/cache/reactantserver
  model_control_mode: dynamic  # dynamic (watch the repo) | static | explicit
  model_poll_seconds: 15.0     # repository poll interval in dynamic mode
  runtime:                     # -> RuntimeConfig
    backend: cuda              # cpu or cuda
    mem_fraction: 0.9          # fraction of device memory claimed for the pool (GPU only)
    preallocate: true          # claim the pool up front (GPU only)
    allow_cpu_fallback: false
    numerics: auto             # f32 | auto | tf32; see below
    weight_cache_fraction: 1.0 # arena fraction for all weights (pinned + on-demand); 0 disables
    weight_cache_wiggle_fraction: 0.1  # arena fraction kept free; drives startup auto-sizing
    autotune: true             # XLA GPU compile autotuner; false = default kernels, no trials
    # autotune_cache: false    # persistent per-fusion autotune cache; omit to inherit Reactant's
    # autotune_cache_dir: /cache/autotune  # cache directory; omit to inherit Reactant's defaults
    shared_host_weights: false # one shm-backed host copy of each model on this node's workers
    shared_host_weights_mode: "666"  # octal permissions for the shared regions; "660" preferred
  scheduler:                   # -> SchedulerConfig
    discipline: fair           # fair | fifo | edf; fifo or edf behind a gateway in lpt_packing
    ema_halflife_seconds: 30.0
    max_queue_depth: 1024      # per-model queue cap; a full model rejects new requests
    dispatch_timeout_seconds: 30.0
    compaction_interval: 0     # defragment device memory every N on-demand weight loads; 0 disables
    models: {}                 # per-model overrides -> ModelSchedConfig (see below)
  endpoints:                   # -> EndpointsConfig
    host: 0.0.0.0              # bind all interfaces so the gateway and clients can reach the worker
    max_concurrent_requests: 64  # in-flight RPC cap; 0 = uncapped; shed with RESOURCE_EXHAUSTED
  grpc:                        # -> GrpcConfig
    max_recv_msg_bytes: 536870912  # 512 MiB; max inbound gRPC message
    max_send_msg_bytes: 536870912  # 512 MiB; max outbound gRPC message
```

Each sub-block corresponds to a typed config struct ([`RuntimeConfig`](@ref),
[`SchedulerConfig`](@ref), [`ModelSchedConfig`](@ref), [`EndpointsConfig`](@ref),
[`GrpcConfig`](@ref)); the whole resolved `global:` is the [`ServerConfig`](@ref) a worker
serves from. The [API](api.md) page lists every field and its default.

`runtime.numerics` sets the f32 matmul/convolution precision policy. `auto` (the default) is
hardware-adaptive: TF32 is used on GPUs that support it (NVIDIA compute capability 8.0 and up)
and stripped where it would not compile, so one bundle runs everywhere but its numerics follow
the hardware. `f32` pins full f32 on every target: TF32 `DotAlgorithm`s are rewritten to f32,
every algorithm-free f32 `dot_general`/`convolution` gets `precision_config = HIGHEST`, and
`NVIDIA_TF32_OVERRIDE=0` is set as defense in depth. This makes numerics identical across GPU
generations, which is the mode for validated deployments where a hardware refresh must not
change model outputs; the cost is tensor-core throughput for f32 matmuls on TF32-capable GPUs.
`tf32` compiles exactly like `auto` (TF32 is permitted and XLA/cuBLAS pick the kernels; StableHLO
has no way to force TF32 for convolutions) but turns the hardware requirement into a guarantee:
startup fails on hardware that cannot run TF32, so a mixed fleet cannot silently serve divergent
numerics. On CUDA workers a startup probe logs whether TF32 arithmetic is actually in use and,
under `f32`, proves the pin bit-exactly; the per-model outcome (ops pinned, algorithms rewritten
or stripped) is recorded in each "model loaded" log line.

`model_control_mode` sets how the loaded model set evolves: `dynamic` (the default) watches the
repository and loads, unloads, reloads, and renames bundles online as files change (a renamed
directory with unchanged contents renames the model in place, no recompile); `static` fixes the
startup set; `explicit` cedes the lifecycle to an external control plane via the worker control
RPCs. `model_poll_seconds` is the repository poll interval in `dynamic` mode.

`runtime.device_ordinal` is not something you normally set here: the node resolution derives it
from the worker's `gpu:` key (defaulting to 0), and under the supervisor each worker sees a
single device through its own `CUDA_VISIBLE_DEVICES`.

The `global.grpc` block is the single node-level place for gRPC message-size limits: every
worker reads it directly, and the supervisor also mirrors it into the embedded gateway (as
`REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` / `REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES`), so one
block sizes the whole node. The defaults are 512 MiB each, and these are decode/encode caps, not
allocations: raising them costs nothing until a message that large actually arrives.
Per-component environment overrides still apply, with the gateway's own
`REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` winning over the mirrored value:
`INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES` for a worker, `REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES`
for the gateway.

`scheduler.discipline` selects the dispatch policy: `fair` shares GPU time across models by
weighted deficit and learned cost, `fifo` serves the oldest queued request first, and `edf`
(earliest-deadline-first) serves the model whose most-urgent queued request has the soonest
deadline, where the deadline comes from the request's remaining-budget timeout. Workers fronted
by a gateway in `lpt_packing` mode must run `fifo` or `edf` (not `fair`), so the gateway stays
the placement and fairness authority (see [Multi-GPU Gateway](gateway.md)).

Under `edf`, a meta model is not scheduled itself (it runs on the request task), but each of its
in-flight sub-calls inherits the meta's deadline, so its continuation is ordered ahead of fresher
regular work. `edf` also sheds work it cannot finish within its learned compute cost (laxity),
trading some throughput (batch fragmentation, no per-model weighting) for meeting more deadlines
under load. Note that `edf` derives urgency solely from the deadline: issuing different
per-client deadlines for the same model reorders that model's service and therefore affects
fairness across clients, so keep deadlines uniform to retain `fifo`-like fairness. See
[Scheduling](scheduling.md) for how the disciplines and coalescing work internally.

## Workers

```yaml
workers:
  - { name: worker0 }
  - { name: worker1 }
```

`name` is the routing identity (and, under Docker, the compose service name). The listen port is
`base_port + index` unless the worker sets an explicit `port:`. A worker entry may also carry
override blocks (for example a `runtime:` block) that deep-merge over `global:`.

Under the supervisor you do not assign GPUs yourself: it detects the visible devices, gives each
worker one of them, and sets that worker's own `CUDA_VISIBLE_DEVICES`, so every worker sees a
single GPU at ordinal 0. Influence the assignment with the top-level `gpus:` key (`auto`, a
count, or an explicit device list), the `REACTANT_GPUS` environment variable, or by adding
`gpu: N` to a worker entry to pin it to a specific visible device. A host-level
`CUDA_VISIBLE_DEVICES` acts as a coarse filter on which physical GPUs the supervisor sees (see
[Deployment](deployment.md)). Running a single worker by hand without the supervisor (a bare
`serve`), the worker uses device ordinal 0, or `gpu: N` to pick another.

## Device pinning (the `models:` map)

```yaml
models:
  resnet50: [worker0, worker1]   # hot on both GPUs
  vsq_coral: [worker0]           # hot on worker0 only
```

Every worker loads (and can serve) every bundle in `model_repo`; the gateway discovers what
each worker serves and schedules requests across them. The optional top-level `models:` map is a
per-model override that pins the named models to device memory on the listed workers for the
lowest latency. It translates into `scheduler.models.<name>.residency: device` on those
workers, merging into any per-model block already set under `global:` (an explicit `residency`
wins; the map only fills an unspecified one). Unlisted models stay system-pinned in host RAM and
load to the device on demand. Omit the block entirely to keep every model on-demand.

To load only a subset of the repository on a worker, the resolved config also supports a
`models_include` allowlist (empty means load all); set it with the
`INFERENCE_SERVER_MODELS_INCLUDE` override below.

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

`weight` sets the model's fair share (consulted only by the `fair` discipline). `residency` sets
the model's residency floor: `unpinned`, `system`, or `device` (`pin_to_gpu: true` is a
back-compat alias for `residency: device`). When the on-demand weight cache is enabled, models
without an explicit `residency` default to `system` (weights pinned in host RAM); see
[On-demand Weights](on_demand_weights.md).

`max_batch_size` caps how many rows the scheduler coalesces into one dispatch of the model. It
does not change compiled shapes: the dispatch sizes come from the batch sizes the bundle was
compiled for (a partial fill still pads up to the smallest compiled size), the batch axis comes
from the bundle manifest, and a single request larger than the cap is still served in one
dispatch because requests are never split.

## Gateway configuration

The gateway does not read the node file at all: it is configured by its own standalone
`gateway.yml`, which carries its listen addresses, limits, and a flat `endpoints:` list of
worker `host:port` addresses that may span any number of nodes (see
[Multi-GPU Gateway](gateway.md)). A `gateway.yml` looks like:

```yaml
listen:
  grpc: "0.0.0.0:8001"
  metrics: "0.0.0.0:8002"
grpc:
  max_recv_msg_bytes: 536870912   # 512 MiB
  max_send_msg_bytes: 536870912
  max_concurrent_requests_per_worker: 64  # inbound cap is this x worker count; 0 = uncapped.
                                          # Sized above the outbound stream limit so a startup
                                          # burst has headroom rather than being shed early
worker_client:
  request_timeout_seconds: 60
  max_concurrent_streams: 32      # outbound in-flight RPCs the gateway multiplexes to one worker
logging:
  level: "info"
  format: "json"
scheduling:
  mode: round_robin               # round_robin | least_outstanding | lpt_packing
  work_basis: compute             # what "least busy" measures: compute | items | requests
  # lpt_packing only; see Multi-GPU Gateway for the full set and the runtime control plane.
  default_replicas: 1             # GPUs per model (a number, or "all")
  routing_fill_mode: run          # run (default) | spread | inflight
endpoints:                        # worker host:port addresses, across any number of nodes
  - "worker0:8080"
  - "worker1:8081"
```

The gateway is also configured by `REACTANT_GATEWAY_*` environment overrides
(`REACTANT_GATEWAY_WORKERS` replaces the endpoint list, comma separated). The embedded gateway
under the supervisor is launched from defaults plus those synthesized environment variables, so
a single node needs no `gateway.yml` at all. See [Multi-GPU Gateway](gateway.md) for the
scheduling modes, the runtime control plane, and the lpt_packing-specific knobs.

## Environment overrides

Any worker value can be overridden per process by an environment variable of the form
`INFERENCE_SERVER_<SECTION>_<FIELD>`, for example:

```text
INFERENCE_SERVER_ENDPOINTS_PORT=9100
INFERENCE_SERVER_RUNTIME_BACKEND=cpu
INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_FRACTION=0.8
```

Booleans parse as `1`, `true`, `yes`, or `on` (case-insensitive). List-valued overrides
(`INFERENCE_SERVER_MODEL_DIRS`, `INFERENCE_SERVER_MODELS_INCLUDE`) are colon-separated. Overrides
are applied on top of the resolved node config, and the effective configuration, including which
overrides were applied, is logged at startup.

| Variable | Overrides | Values |
|---|---|---|
| `INFERENCE_SERVER_CACHE_DIR` | `cache_dir` | path |
| `INFERENCE_SERVER_MODEL_POLL_SECONDS` | `model_poll_seconds` | seconds |
| `INFERENCE_SERVER_MODEL_CONTROL_MODE` | `model_control_mode` | `static` \| `dynamic` \| `explicit` |
| `INFERENCE_SERVER_MODEL_DIRS` | `model_dirs` | colon-separated paths |
| `INFERENCE_SERVER_MODELS_INCLUDE` | `models_include` | colon-separated model names |
| `INFERENCE_SERVER_RUNTIME_BACKEND` | `runtime.backend` | `cpu` \| `cuda` |
| `INFERENCE_SERVER_RUNTIME_DEVICE_ORDINAL` | `runtime.device_ordinal` | device index |
| `INFERENCE_SERVER_RUNTIME_MEM_FRACTION` | `runtime.mem_fraction` | `0..1` |
| `INFERENCE_SERVER_RUNTIME_PREALLOCATE` | `runtime.preallocate` | bool |
| `INFERENCE_SERVER_RUNTIME_ALLOW_CPU_FALLBACK` | `runtime.allow_cpu_fallback` | bool |
| `INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_FRACTION` | `runtime.weight_cache_fraction` | `0..1` |
| `INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_WIGGLE_FRACTION` | `runtime.weight_cache_wiggle_fraction` | `0..1` |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE` | `runtime.autotune` | bool |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE` | `runtime.autotune_cache` | bool |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE_DIR` | `runtime.autotune_cache_dir` | path |
| `INFERENCE_SERVER_RUNTIME_NUMERICS` | `runtime.numerics` | `f32` \| `auto` \| `tf32` |
| `INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS` | `runtime.shared_host_weights` | bool |
| `INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS_MODE` | `runtime.shared_host_weights_mode` | octal string |
| `INFERENCE_SERVER_SCHEDULER_DISCIPLINE` | `scheduler.discipline` | `fair` \| `fifo` \| `edf` |
| `INFERENCE_SERVER_SCHEDULER_EMA_HALFLIFE_SECONDS` | `scheduler.ema_halflife_seconds` | seconds |
| `INFERENCE_SERVER_SCHEDULER_RECENCY_PENALTY_CAP` | `scheduler.recency_penalty_cap` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_COALESCING_DISCOUNT` | `scheduler.coalescing_discount` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_COST_EMA_ALPHA` | `scheduler.cost_ema_alpha` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_MAX_QUEUE_DEPTH` | `scheduler.max_queue_depth` | integer |
| `INFERENCE_SERVER_SCHEDULER_DISPATCH_TIMEOUT_SECONDS` | `scheduler.dispatch_timeout_seconds` | seconds |
| `INFERENCE_SERVER_SCHEDULER_COMPACTION_INTERVAL` | `scheduler.compaction_interval` | integer |
| `INFERENCE_SERVER_ENDPOINTS_HOST` | `endpoints.host` | address |
| `INFERENCE_SERVER_ENDPOINTS_PORT` | `endpoints.port` | port |
| `INFERENCE_SERVER_ENDPOINTS_METRICS_PORT` | `endpoints.metrics_port` | port |
| `INFERENCE_SERVER_ENDPOINTS_MAX_CONCURRENT_REQUESTS` | `endpoints.max_concurrent_requests` | integer |
| `INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES` | `grpc.max_recv_msg_bytes` | bytes |
| `INFERENCE_SERVER_GRPC_MAX_SEND_MSG_BYTES` | `grpc.max_send_msg_bytes` | bytes |

Removed keys fail loudly rather than being silently ignored: a `batch_policy` block or a
`runtime.residency_mode` key is rejected at startup with a migration error naming the
replacement (`scheduler.models.<name>.max_batch_size` and the top-level `model_control_mode`,
respectively), and the same applies to their `INFERENCE_SERVER_BATCH_POLICY_*` and
`INFERENCE_SERVER_RUNTIME_RESIDENCY_MODE` environment overrides.
