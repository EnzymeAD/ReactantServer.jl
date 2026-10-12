# Node Configuration

A node file describes one machine: its model repository and its workers, one per device. Each
worker reads the same file, resolves its own entry by name, and serves every bundle in the
repository. Under the node supervisor the `workers:` list is optional, and the supervisor creates
one worker per detected device ([Deployment](deployment.md)). The gateway in front of a
multi-worker node is configured separately, by its own `gateway.yml`
([Gateway configuration](#Gateway-configuration)).

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
    executable_cache: true     # cache compiled programs under each bundle's .cache/; see below
    profile: default           # default | regulated; see below
    batch_sizes: all           # all | largest compiled batch size per model (profile default)
    xla_flags: {}              # XLA DebugOptions overrides, e.g. xla_gpu_exclude_nondeterministic_ops: true
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

`runtime.numerics` sets the precision of f32 matrix multiplications and convolutions:

- `auto` (the default) uses TF32 on GPUs that support it (NVIDIA compute capability 8.0 and up)
  and strips it where it would not compile.
- `f32` pins full f32 on every target: TF32 `DotAlgorithm`s are rewritten to f32, every
  algorithm-free f32 `dot_general` and `convolution` gets `precision_config = HIGHEST`, and
  `NVIDIA_TF32_OVERRIDE=0` is set as well.
- `tf32` compiles like `auto`, but startup fails unless TF32 is in effect: on hardware without
  it, with `NVIDIA_TF32_OVERRIDE=0` set, or when the startup probe matmul does not run in TF32.

On CUDA workers a startup probe logs whether TF32 arithmetic is in use, and under `f32` proves the
pin bit-exact. Each "model loaded" log line records the outcome per model: operations pinned, and
algorithms rewritten or stripped.

`runtime.executable_cache` (default `true`) stores every compiled program under the bundle's
`.cache/` directory and loads it on later starts, turning a 5 to 30 second compile per program into
a load of well under 100 ms. Entries are partitioned by the Reactant_jll build, the platform, and
the device kind and compute capability, none of which XLA checks on load, and named by the MLIR
source's content hash and a hash of the compile options (including `xla_flags` and `autotune`) and
the numerics policy, so changing any of those compiles fresh. `.cache/mlir_hashes.json` records the hash of every `model*.mlir`; a module
whose content changes has its programs dropped on the next load, while a weights-only update keeps
them. The directory watcher never reacts to anything under `.cache/`, so cache writes cannot reload
the model that produced them. The bundle directory must be writable by the worker; when it is not,
the cache logs a warning and every program is compiled as before.

`runtime.batch_sizes` selects which of a bundle's compiled batch sizes are loaded. `all` (the
default) loads every `model.b{N}.mlir` the manifest declares, and the scheduler picks the largest
size a dispatch can fill. `largest` loads only the largest declared size, per input-shape variant,
and never reads the others (they need not be present); every dispatch runs that program, padded
with zero rows and sliced back per request. Unbatched bundles are unaffected. Workers with
different settings can share a bundle directory.

`runtime.xla_flags` is a mapping of XLA `DebugOptions` field names (no leading `--`) to values,
applied to every compile. At startup each name is checked against the XLA linked into the worker
and each value against the field's type (booleans, integers, numbers, strings, and enum fields by
enumerator name), so a typo fails startup instead of being ignored. An explicit
`xla_gpu_autotune_level` here wins over `autotune: false`.

`runtime.profile` applies a named set of defaults. `regulated` defaults `batch_sizes` to
`largest` and adds `xla_gpu_exclude_nondeterministic_ops: true` to `xla_flags`; anything set
explicitly wins. It does not change `numerics`.

[Reproducibility & Regulated Use](reproducibility.md) explains how `numerics`, `batch_sizes`,
`executable_cache`, `xla_flags`, and `profile` affect a model's outputs, and which values a
validated deployment should use.

`model_control_mode` sets how the loaded model set evolves:

- `dynamic` (the default) watches the repository and loads, unloads, and reloads bundles online as
  their files change, including the weights, the MLIR, the manifest, and `model.jl`. A bundle must
  look the same on two consecutive polls before it loads, so a half-written bundle never does, and
  a reload swaps one model while every other model keeps serving. A renamed directory with
  unchanged contents renames the live model in place, without recompiling.
- `static` loads the startup set once and never changes it.
- `explicit` cedes the lifecycle to an external control plane: the worker takes no action of its
  own, and a model serves only while the control plane holds it resident through the worker
  control RPCs (`ModelControlStatus`, `SetModelResidency`, `SetModelPolicy`).

`model_poll_seconds` is the repository poll interval in `dynamic` mode.

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

`name` is the worker's identity in logs, metrics, and the `models:` map. The listen port is
`base_port + index` unless the worker sets an explicit `port:`. A worker entry may also carry
override blocks (for example a `runtime:` block) that deep-merge over `global:`.

`gpu: N` pins a worker to a specific visible device. Under the supervisor, devices are otherwise
assigned in order ([Deployment](deployment.md) describes how they are detected); a worker run by
hand with `serve` uses device 0 unless it sets `gpu:`.

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
  - "10.0.0.1:8080"
  - "10.0.0.2:8080"
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
| `INFERENCE_SERVER_RUNTIME_EXECUTABLE_CACHE` | `runtime.executable_cache` | bool |
| `INFERENCE_SERVER_RUNTIME_PROFILE` | `runtime.profile` | `default` \| `regulated` |
| `INFERENCE_SERVER_RUNTIME_BATCH_SIZES` | `runtime.batch_sizes` | `all` \| `largest` |
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
