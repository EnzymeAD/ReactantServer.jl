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

# Supervisor-only: how a worker or the gateway that stops working is found and restarted.
# liveness: { interval_seconds: 10, timeout_seconds: 10, failures: 6, dispatch_stall_seconds: 600 }

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
- `liveness` (optional, supervisor-only) sets how the supervisor checks that each worker, and the
  embedded gateway, still works once it has been ready, and kills and restarts one that does not
  ([Liveness](#Liveness)).
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
    device_memory_fraction: 0.9  # share of GPU memory claimed for the pool (weights, scratch, I/O)
    preallocate: true          # claim the pool up front (GPU only)
    allow_cpu_fallback: false
    numerics: auto             # f32 | auto | tf32; see below
    executable_cache: true     # cache compiled programs under each bundle's .cache/; see below
    profile: default           # default | regulated; see below
    batch_sizes: all           # all | largest compiled batch size per model (profile default)
    xla_flags: {}              # XLA DebugOptions overrides, e.g. xla_gpu_exclude_nondeterministic_ops: true
    device_weight_fraction: 1.0    # GPU arena share for resident weights (pinned + on-demand), (0, 1]
    device_headroom_fraction: 0.1  # GPU arena share kept free; drives startup auto-sizing
    autotune: true             # XLA GPU compile autotuner; false = default kernels, no trials
    # autotune_cache: false    # persistent per-fusion autotune cache; omit to inherit Reactant's
    # autotune_cache_dir: /cache/autotune  # cache directory; omit to inherit Reactant's defaults
    shared_host_weights: false # one shm-backed host copy of each model on this node's workers
    shared_host_weights_mode: "666"  # octal permissions for the shared regions; "660" preferred
    weights_migration: off     # off | copy | replace legacy-order weights files at load; see below
    host_weight_budget: 0      # host RAM for weight copies (0b, 512m, 5g... or unlimited); see Use Cases
    # host_weight_budget_workers: 1  # private copies: workers splitting the budget (set by the supervisor)
  scheduler:                   # -> SchedulerConfig
    discipline: fair           # fair | fifo | edf; fifo or edf behind a gateway in lpt_packing
    ema_halflife_seconds: 30.0
    max_queue_depth: 1024      # per-model queue cap; a full model rejects new requests
    dispatch_timeout_seconds: 30.0
    compaction_interval: 0     # defragment device memory every N on-demand weight loads; 0 disables
    coalescing: true           # merge queued requests into one dispatch (profile default)
    weight_read_stall_seconds: 20.0  # abandon a weights read with no progress this long; 0 disables
    models: {}                 # per-model overrides -> ModelSchedConfig (see below)
  endpoints:                   # -> EndpointsConfig
    host: 0.0.0.0              # bind all interfaces so the gateway and clients can reach the worker
    max_concurrent_requests: 64  # in-flight RPC cap; 0 = uncapped; shed with RESOURCE_EXHAUSTED
  grpc:                        # -> GrpcConfig
    max_recv_msg_bytes: 536870912  # 512 MiB; max inbound gRPC message
    max_send_msg_bytes: 536870912  # 512 MiB; max outbound gRPC message
    error_detail: redacted     # what an INTERNAL error tells clients: redacted | full
    rapid_reset_protection: false  # true: HTTP/2 Rapid-Reset cut-off, gateway too; see below
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

`runtime.weights_migration` converts bundles whose `weights.safetensors` predates ReactantServer
0.2 (legacy axis order, which costs an extra full copy of every tensor on each load) to the
program's axis order. `off` (the default) loads them as they are. `copy` writes a verified
`weights.T.safetensors` beside the original and points the manifest's `weights_file` at it; the
original is untouched, and servers older than 0.2 ignore the new field and keep loading it.
`replace` then swaps the copy in as `weights.safetensors` and marks the manifest
`format_version: "2.1"`, which older servers refuse; use it once every server reading the
directory runs 0.2 or later. The bundle directory must be writable. A migration never fails a model:
on any error it is logged and the bundle loads as it was. In every mode, `off` included, a
`weights.safetensors` deployed after a migration is the file that loads. See
[Migrating older bundles](bundles.md#Migrating-older-bundles).

`runtime.xla_flags` is a mapping of XLA `DebugOptions` field names (no leading `--`) to values,
applied to every compile. At startup each name is checked against the XLA linked into the worker
and each value against the field's type (booleans, integers, numbers, strings, and enum fields by
enumerator name), so a typo fails startup instead of being ignored. An explicit
`xla_gpu_autotune_level` here wins over `autotune: false`.

`runtime.profile` applies a named set of defaults. `regulated` defaults `batch_sizes` to
`largest`, adds `xla_gpu_exclude_nondeterministic_ops: true` to `xla_flags`, and defaults
`scheduler.coalescing` to `false`; anything set explicitly wins. It does not change `numerics`.

`scheduler.coalescing` (default `true`) lets the scheduler merge queued requests to the same model
into one dispatch. With `false`, each dispatch serves exactly one request; see
[Scheduling & Coalescing](scheduling.md).

[Reproducibility & Regulated Use](reproducibility.md) explains how `numerics`, `batch_sizes`,
`coalescing`, `executable_cache`, `xla_flags`, and `profile` affect a model's outputs, and which values a
validated deployment should use.

`model_control_mode` sets how the loaded model set evolves:

- `dynamic` (the default) watches the repository and loads, unloads, and reloads bundles online as
  their files change, including the weights, the MLIR, the manifest, and `model.jl`. The watcher
  compares only each bundle file's name, modification time and size, and acts on a change once it
  looks the same on two consecutive polls. A deploy must therefore change a file's mtime or size to
  be seen: new weights of the same size written with a fixed or preserved mtime (an archive built
  with a fixed `--mtime`, unpacked or copied with `cp -p` or `rsync -a`) are never picked up. The
  two polls let a copy in progress settle, but they do not guarantee a consistent bundle: an update
  of several files that takes longer than two polls can be loaded with some files new and some old.
  Make a multi-file update atomic: stage the new files elsewhere on the same filesystem, then rename
  them into place. A load or reload compiles the bundle on the worker's dispatch loop, so every
  other model on that worker pauses until the compile ends, and a reloaded model answers `NOT_FOUND`
  until its new version is in. The loop shares a thread with request handling, so the worker's
  health probes stall for that time as well. Behind a gateway, a stall longer than the gateway's
  5 second probe deadline marks the worker down, and the gateway sends it no request until a later
  probe finds it ready; `dynamic` is not a production mode, and this is not avoided. An
  executable-cache hit (`runtime.executable_cache`) keeps the pause short; an uncached compile
  takes 5 to 30 seconds per compiled program. A renamed
  directory is an unload of the old name and a load of the new one. A bundle whose files change
  while it loads is not admitted: the next poll loads it again from the files as they are then.
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
block sizes the whole node. The defaults are 512 MiB each. `max_send_msg_bytes` is only a check
on each outgoing message, but `max_recv_msg_bytes` also sizes memory: the gRPC server allocates a
buffer of a request message's full declared length as soon as the message's 5-byte length prefix
arrives, before any of the payload, so every request being received can hold up to
`max_recv_msg_bytes` however few bytes its client has sent. Size it to the largest inline message
the deployment needs (tensors passed through shared memory do not count): the largest request for
a worker, and for the gateway, which applies the same limit to the responses it receives from
workers, the larger of the largest request and the largest response. Lower it wherever untrusted
clients can connect; see [Security](security.md).
Per-component environment overrides still apply, with the gateway's own
`REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` winning over the mirrored value:
`INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES` for a worker, `REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES`
for the gateway.

`global.grpc.error_detail` decides what a client is told when a request fails inside the server
(`INTERNAL`), and like the message sizes it reaches every worker directly and the embedded gateway
through the supervisor's mirror (`REACTANT_GATEWAY_GRPC_ERROR_DETAIL`). Under `redacted`, the
default and the production setting, the client gets only `internal error (request id <id>, worker
<name>); see the worker log` (from the gateway's own failures, `gateway` and the gateway log). The
id is the KServe request `id` when the client sent one, otherwise one the server generates, and
the server logs the full exception and backtrace under that same id and name, so a client report
can be traced to its log line. Under `full`, for development and testing, the message also carries
the exception text (at most 1 KiB). In `redacted` mode the gateway also names workers by their
worker name (or position), never by host and port. Only `INTERNAL` is redacted: every other status
keeps its explanation. In both modes every status message is sanitized: `grpc-status:` and
`grpc-message:` in it become `grpc-status=` and `grpc-message=`, since a client library that finds
them in an error message can misread the call's status. The per-component overrides are
`INFERENCE_SERVER_GRPC_ERROR_DETAIL` (a worker) and `REACTANT_GATEWAY_GRPC_ERROR_DETAIL` (the
gateway).

`global.grpc.rapid_reset_protection` (default `false`) switches on HTTP/2 Rapid-Reset protection
for every worker and the embedded gateway at once; see
[Rapid-Reset protection](#Rapid-Reset-protection) for what that does to clients, the matching
environment variable, and which setting wins.

`scheduler.discipline` selects the dispatch policy: `fair` (the default), `fifo`, or `edf`. Workers
behind a gateway in `lpt_packing` mode must run `fifo` or `edf`. [Scheduling](scheduling.md)
describes each discipline and when to use it.

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
wins; the map only fills an unspecified one). Unlisted models load to the device on demand, from
a host RAM copy when `runtime.host_weight_budget` keeps one ([Use Cases](use_cases.md)). Omit the block entirely to keep every model on-demand.

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
back-compat alias for `residency: device`). Models without an explicit `residency` are
swapped on demand, keeping host RAM copies within `runtime.host_weight_budget`, or pinned in host
RAM when that budget is `unlimited`; see [Use Cases](use_cases.md).

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
  error_detail: redacted          # what an INTERNAL error tells clients: redacted | full
  max_concurrent_requests_per_worker: 64  # inbound cap is this x worker count; 0 = uncapped.
                                          # A call past it is shed at once with RESOURCE_EXHAUSTED
                                          # (gateway_requests_shed_total). Sized above the outbound
                                          # stream limit so a startup burst has headroom rather
                                          # than being shed early
  rapid_reset_protection: false   # true: HTTP.jl's Rapid-Reset cut-off (see below)
worker_client:
  request_timeout_seconds: 60
  max_concurrent_streams: 32      # outbound in-flight RPCs the gateway multiplexes to one worker
                                  # (inference and the rest; the health probes have their own)
  route_retention_seconds: 600    # how long a worker that is unreachable or not ready keeps its
                                  # routes, so its models are UNAVAILABLE rather than NOT_FOUND
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

`grpc.rapid_reset_protection` (default `false`; environment
`REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION`) switches on the gateway's HTTP/2 Rapid-Reset
protection. Under the supervisor the node-wide setting also reaches a gateway run from a mounted
`gateway.yml` that does not set this key; see [Rapid-Reset protection](#Rapid-Reset-protection).

The gateway is also configured by `REACTANT_GATEWAY_*` environment overrides
(`REACTANT_GATEWAY_WORKERS` replaces the endpoint list, comma separated). The embedded gateway
under the supervisor is launched from defaults plus those synthesized environment variables, so
a single node needs no `gateway.yml` at all. See [Multi-GPU Gateway](gateway.md) for the
scheduling modes, the runtime control plane, and the lpt_packing-specific knobs.

## Rapid-Reset protection

HTTP/2 Rapid-Reset (CVE-2023-44487) protection is off by default on every worker and on the
gateway. With it on, a client connection may have 100 streams open at once, and the server ends
the connection (`GOAWAY(ENHANCE_YOUR_CALM)`) once the client has reset more than 200 streams over
the connection's life. A client resets a stream every time it gives up on a call (its deadline
passes, or it cancels), so **a client that gives up on calls is disconnected after about 200
give-ups per connection**, however long the connection has been up. gRPCClient-based clients,
`ReactantServerClient` among them, can then hang on every call to that server until the call's
deadline, for up to about two minutes, if their first calls on the new connection are give-ups
as well (libcurl waits for that connection to answer before using it, and it never does). The
gateway is such a client of each worker: every call it forwards that reaches its deadline resets a
stream on its connection to that worker, so protection on the workers can cut the gateway off the
same way. With it off, no client is disconnected for giving up on calls, and what bounds a client
that resets streams is the server's inbound cap, past which calls are shed with
`RESOURCE_EXHAUSTED`: `endpoints.max_concurrent_requests` on a worker, and
`grpc.max_concurrent_requests_per_worker` times the worker count on the gateway. Nothing bounds it
where that cap is 0. Turn the protection on only where untrusted clients can connect, and only if
the clients that depend on the node can tolerate being cut off as described.

On a node, one setting turns it on or off for the gateway and every worker:

```yaml
global:
  grpc:
    rapid_reset_protection: true
```

or, without editing the node file, `REACTANT_RAPID_RESET_PROTECTION=true` on the node container.
The supervisor reads that variable; it accepts `true`, `false`, `1`, `0`, `yes`, `no`, `on` and
`off`, refuses anything else at startup, and wins over the node file. The supervisor hands the
value to every worker it launches through the node file they read, and to the embedded gateway as
`REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION`, in the `all`, `workers` and `gateway` roles alike. A
value set for one component still wins over it. For each component, the first of these that is
set decides:

- **A worker:** `INFERENCE_SERVER_GRPC_RAPID_RESET_PROTECTION`, then `grpc.rapid_reset_protection`
  in that worker's own `workers:` entry, then `REACTANT_RAPID_RESET_PROTECTION`, then
  `global.grpc.rapid_reset_protection`; otherwise off.
- **The embedded gateway:** `REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION`, then
  `grpc.rapid_reset_protection` in a mounted `gateway.yml`, then `REACTANT_RAPID_RESET_PROTECTION`,
  then `global.grpc.rapid_reset_protection`; otherwise off. Unlike the message sizes and
  `error_detail`, which a mounted `gateway.yml` owns whether or not it sets them, the node-wide
  value reaches a gateway run from a mounted file that does not set this key.

A worker or gateway run without the supervisor has only its own settings: a worker reads its node
file's `global.grpc` and its own entry, with `INFERENCE_SERVER_GRPC_RAPID_RESET_PROTECTION` over
both, and a gateway its `gateway.yml`, with `REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION` over it.

## Liveness

The supervisor restarts a worker or the embedded gateway that exits, and also one that is still
running but no longer works: deadlocked, stopped, stuck in the kernel, or, for a worker, with its
dispatch loop stuck in a GPU call that never returns. It checks each worker over the worker's HTTP
listener (`metrics_base_port + i`, or the public `8002` for a sole worker), and the gateway over its
admin listener (`listen.metrics`, default `8002`, whose `/readyz` answers 200 once at least one
worker is ready). The same steps and settings apply to both:

1. Until the worker's `/readyz` first answers 200, nothing is checked, so a startup that compiles
   models for hours is never interrupted.
2. From then on the supervisor asks `/healthz` every `interval_seconds`. An answer other than 200,
   a refused connection, or no answer within `timeout_seconds` is a failure, and is logged.
3. After `failures` failures in a row the supervisor logs that it is killing the worker, sends it
   SIGKILL, and the usual restart takes over, with its backoff and its restart budget
   (`REACTANT_SUPERVISOR_MAX_RESTARTS`). The restarted worker is checked again once it is ready.

Nothing is checked while the node shuts down. A worker answers `/healthz` (and `ServerLive`) with
"not live", and `/readyz` (and `ServerReady`, `ModelReady`) with "not ready", while its dispatch
loop has had work in hand and finished none of it for `dispatch_stall_seconds`: a dispatch, a
weight upload, a compaction and a control command each count as one piece of work. An idle worker
is never stalled, and the worker says so in its log once per stall. That is how a GPU call that
never returns is caught: the worker's other threads keep answering, but no longer "live".

```yaml
liveness:                       # supervisor-only; every key is optional
  interval_seconds: 10          # REACTANT_LIVENESS_INTERVAL_SECONDS
  timeout_seconds: 10           # REACTANT_LIVENESS_TIMEOUT_SECONDS
  failures: 6                   # REACTANT_LIVENESS_FAILURES; 0 turns the check off
  dispatch_stall_seconds: 600   # REACTANT_LIVENESS_DISPATCH_STALL_SECONDS; 0 turns it off
```

Each environment variable, set on the node container, wins over its key. With the defaults a
worker that stops answering is killed after about two minutes, and one that answers "not live"
after about one; a stalled dispatch loop is reported after ten minutes. These are far longer than
anything a healthy worker does on the thread that answers (a garbage collection or a first-use
compile takes seconds), and the stall limit is far longer than the slowest legitimate piece of work
(minutes for a large model read from slow storage, or a compile in `dynamic` mode). The startup line
`worker liveness checks: ...` shows what is in effect. The supervisor passes
`dispatch_stall_seconds` to each worker in `REACTANT_LIVENESS_DISPATCH_STALL_SECONDS`; a worker run
without the supervisor reads that variable itself, so an orchestrator's own liveness probe on
`/healthz` gets the same answer.

Some limits to know:

- A worker with no HTTP listener (no `metrics_base_port` on a multi-worker node) is not checked; it
  is restarted only when it exits, and the startup line names it.
- In `dynamic` mode a hot load or reload compiles the new bundle on the thread that answers
  `/healthz`, so a reload whose uncached compile stops that thread for longer than about two minutes
  gets the worker killed and restarted, and it then loads the new version at startup. Raise
  `timeout_seconds` or `failures` (or set `failures: 0`) if reloads of large uncached bundles must
  not restart the worker.
- A hung gateway takes the node's public endpoint down with it, so it is killed and restarted the
  same way; its startup line is `gateway liveness check: ...`. Its `/healthz` answers whenever the
  process can answer at all, so the check catches a gateway that is stopped or deadlocked, not one
  that answers while routing badly.

## Environment overrides

Any worker value can be overridden per process by an environment variable of the form
`INFERENCE_SERVER_<SECTION>_<FIELD>`, for example:

```text
INFERENCE_SERVER_ENDPOINTS_PORT=9100
INFERENCE_SERVER_RUNTIME_BACKEND=cpu
INFERENCE_SERVER_RUNTIME_DEVICE_WEIGHT_FRACTION=0.8
```

Booleans parse as `1`, `true`, `yes`, or `on` (case-insensitive). List-valued overrides
(`INFERENCE_SERVER_MODEL_DIRS`, `INFERENCE_SERVER_MODELS_INCLUDE`) are colon-separated. Overrides
are applied on top of the resolved node file, and the effective configuration, including which
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
| `INFERENCE_SERVER_RUNTIME_DEVICE_MEMORY_FRACTION` | `runtime.device_memory_fraction` | `(0, 1]` |
| `INFERENCE_SERVER_RUNTIME_PREALLOCATE` | `runtime.preallocate` | bool |
| `INFERENCE_SERVER_RUNTIME_ALLOW_CPU_FALLBACK` | `runtime.allow_cpu_fallback` | bool |
| `INFERENCE_SERVER_RUNTIME_DEVICE_WEIGHT_FRACTION` | `runtime.device_weight_fraction` | `(0, 1]` |
| `INFERENCE_SERVER_RUNTIME_DEVICE_HEADROOM_FRACTION` | `runtime.device_headroom_fraction` | `[0, 1)` |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE` | `runtime.autotune` | bool |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE` | `runtime.autotune_cache` | bool |
| `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE_DIR` | `runtime.autotune_cache_dir` | path |
| `INFERENCE_SERVER_RUNTIME_NUMERICS` | `runtime.numerics` | `f32` \| `auto` \| `tf32` |
| `INFERENCE_SERVER_RUNTIME_EXECUTABLE_CACHE` | `runtime.executable_cache` | bool |
| `INFERENCE_SERVER_RUNTIME_PROFILE` | `runtime.profile` | `default` \| `regulated` |
| `INFERENCE_SERVER_RUNTIME_BATCH_SIZES` | `runtime.batch_sizes` | `all` \| `largest` |
| `INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS` | `runtime.shared_host_weights` | bool |
| `INFERENCE_SERVER_RUNTIME_SHARED_HOST_WEIGHTS_MODE` | `runtime.shared_host_weights_mode` | octal string |
| `INFERENCE_SERVER_RUNTIME_WEIGHTS_MIGRATION` | `runtime.weights_migration` | `off` \| `copy` \| `replace` |
| `INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET` | `runtime.host_weight_budget` | byte size, e.g. `0b`, `5g` |
| `INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET_WORKERS` | `runtime.host_weight_budget_workers` | worker count |
| `INFERENCE_SERVER_SCHEDULER_DISCIPLINE` | `scheduler.discipline` | `fair` \| `fifo` \| `edf` |
| `INFERENCE_SERVER_SCHEDULER_EMA_HALFLIFE_SECONDS` | `scheduler.ema_halflife_seconds` | seconds |
| `INFERENCE_SERVER_SCHEDULER_RECENCY_PENALTY_CAP` | `scheduler.recency_penalty_cap` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_COALESCING_DISCOUNT` | `scheduler.coalescing_discount` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_COST_EMA_ALPHA` | `scheduler.cost_ema_alpha` | `0..1` |
| `INFERENCE_SERVER_SCHEDULER_MAX_QUEUE_DEPTH` | `scheduler.max_queue_depth` | integer |
| `INFERENCE_SERVER_SCHEDULER_DISPATCH_TIMEOUT_SECONDS` | `scheduler.dispatch_timeout_seconds` | seconds |
| `INFERENCE_SERVER_SCHEDULER_COMPACTION_INTERVAL` | `scheduler.compaction_interval` | integer |
| `INFERENCE_SERVER_SCHEDULER_COALESCING` | `scheduler.coalescing` | bool |
| `INFERENCE_SERVER_SCHEDULER_WEIGHT_READ_STALL_SECONDS` | `scheduler.weight_read_stall_seconds` | seconds |
| `INFERENCE_SERVER_ENDPOINTS_HOST` | `endpoints.host` | address |
| `INFERENCE_SERVER_ENDPOINTS_PORT` | `endpoints.port` | port |
| `INFERENCE_SERVER_ENDPOINTS_METRICS_PORT` | `endpoints.metrics_port` | port |
| `INFERENCE_SERVER_ENDPOINTS_MAX_CONCURRENT_REQUESTS` | `endpoints.max_concurrent_requests` | integer |
| `INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES` | `grpc.max_recv_msg_bytes` | bytes |
| `INFERENCE_SERVER_GRPC_MAX_SEND_MSG_BYTES` | `grpc.max_send_msg_bytes` | bytes |
| `INFERENCE_SERVER_GRPC_ERROR_DETAIL` | `grpc.error_detail` | `redacted`, `full` |
| `INFERENCE_SERVER_GRPC_RAPID_RESET_PROTECTION` | `grpc.rapid_reset_protection` | bool |
