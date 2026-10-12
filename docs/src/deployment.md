# Deployment

A node runs as a single supervisor process (`ReactantServerNode`): it detects every visible GPU,
spawns one single-GPU worker subprocess per device, multiplexes their logs onto one stream with
`[name]` line prefixes, and restarts children that die. With two or more workers it also runs an
embedded gateway on the public ports; with a single worker it binds that worker to the public
ports directly. The external interface is the same either way: KServe V2 gRPC on `:8001`, health
and metrics on `:8002` (`/readyz`, `/healthz`, `/metrics`), matching Triton's ports.

The supported deployment is the **container image**: pull the published image, or build your own
with Bazel, and run it under podman or Docker. The image carries Julia, every package, their
precompiled caches, and the CUDA userspace Reactant needs, so a host needs only an NVIDIA driver
and a container runtime with the NVIDIA Container Toolkit. The supervisor also runs from any
Julia environment that holds ReactantServerNode, ReactantServer, and ReactantServerGateway.

This page is the reference for how a node behaves and how the image is configured. To run one,
follow a tutorial: [Serve on One GPU](tutorials/container_single_gpu.md) and
[Scale to Many GPUs](tutorials/container_multi_gpu.md) for the container, or
[Run a Node](tutorials/native/node.md) for the supervisor from a Julia environment. Scraping the
node's metrics is covered under [Monitoring](monitoring.md), and what to know before exposing an
endpoint under [Security](security.md).

## The node supervisor

The supervisor's job on startup is to turn "this host plus this node file" into a concrete set of
child processes. It does so in three steps.

**1. Detect the devices.** The first of these that yields a non-empty answer wins
(`ReactantServerNode.detect_gpus`):

1. `REACTANT_GPUS` environment variable: a count (`2`) or an explicit list (`0,2` or GPU UUIDs);
   `0` means a CPU node.
2. the node file's `gpus:` key (`auto`, a count, or a list).
3. a `CUDA_VISIBLE_DEVICES` already set on the host.
4. `nvidia-smi` enumeration.
5. `/dev/nvidiaN` device nodes.

For a CUDA node that finds no devices, startup fails with guidance (run with `--gpus all`, set
`REACTANT_GPUS`, or set `backend: cpu`).

**2. Materialize the workers** (`ReactantServerCore.materialize_node!`). With no `workers:` list,
one worker is synthesized per detected device (`worker0..workerN-1`); an explicit `workers:` list
wins and assigns each worker a device positionally, or by its `gpu:` key. Either way each worker
is pinned to exactly one device through its own `CUDA_VISIBLE_DEVICES`, so inside the worker the
device is always ordinal 0 and the single-GPU worker code runs unchanged. The materialized node
file is written to `/run/reactantserver/node.yaml` for inspection.

Each worker's compute-thread pool is sized to its share of the host, `min(CPU_THREADS ÷ workers,
16)` threads, plus one interactive thread for the GPU dispatch loop. This avoids the
oversubscription that `--threads=auto` would cause: with several workers on one node, each `auto`
worker (and its GC and host library pools) would size itself to the whole machine and the workers
would fight for every core under load. The cap keeps a very large box from handing any one worker
an unhelpfully huge pool. Set `REACTANT_WORKER_THREADS` to override the computed value; it is used
verbatim, bypassing the split and the cap.

**3. Decide on the gateway** by worker count, in the default `all` role:

- **One worker, no gateway.** A lone worker already serves the full KServe V2 API, so the
  supervisor binds it directly to the public ports (8001/8002) and starts no gateway. No extra
  process, no extra hop.
- **Two or more workers, workers plus the embedded gateway.** Each worker binds `base_port + i`
  (and `metrics_base_port + i`), and the gateway binds the public 8001/8002. The supervisor
  synthesizes the gateway's worker list (and worker metrics list) from the node file, so the
  gateway needs no config of its own.

The external interface is therefore identical whether you run 1 GPU or 8, which is why your client
and the public ports never change as you scale. `REACTANT_ROLE` selects what the supervisor runs;
the default `all` (workers plus the embedded gateway on one host) is the documented deployment.
The `workers` and `gateway` roles exist in the code to split a deployment across machines, but
multi-node is not a shipped example.

One [`ReactantServerNode.supervise`](@ref ReactantServerNode.supervise) call does the whole
fan-out, for one GPU or many: the single parent process spawns one [`ReactantServer.serve`](@ref)
worker subprocess per device and the gateway as another subprocess, multiplexes their logs onto
its stdout with `[worker0]` / `[gateway]` prefixes, and restarts any child that dies. (Each worker
and the gateway can also run by hand, as separate `serve` / `serve_gateway` processes; see
[Front Workers with a Gateway](tutorials/native/gateway.md).) `ReactantServerNode.main()` is the
container entrypoint: it reads `REACTANT_NODE_FILE` for the node config and runs the supervisor to
completion. On a host without GPUs, `REACTANT_GPUS=0` with `backend: cpu` makes a CPU node, and
`REACTANT_CPU_WORKERS` sets its worker count, so the multi-worker shape can be exercised anywhere.

### The node file is unchanged when you scale

`gpus: auto` already means "one worker per visible GPU", so a single-GPU node file scales as-is: give the host more GPUs and the supervisor runs more
workers. You only edit the node file to do something non-default:

```yaml
model_repo: /var/lib/reactantserver/models
base_port: 8080           # worker i listens on base_port + i (8080, 8081, ...)
metrics_base_port: 9100   # and metrics on metrics_base_port + i (9100, 9101, ...)
gpus: auto                # or an integer count, or an explicit device list

global:
  runtime:
    backend: cuda
  endpoints:
    host: 0.0.0.0
```

A model listed on more than one worker is replicated, and the gateway load-balances requests for
it across those workers. See [Node Configuration](node_config.md) for the `models:` map and
[On-demand Weights](on_demand_weights.md) for fitting more models than GPU memory holds. The node
is described by one YAML node file; the commented templates under `config/` (`node.default.yaml`,
`node.yaml`) are reference configs, and gateway scheduling (`round_robin` or `lpt_packing`) is
covered on the [Multi-GPU Gateway](gateway.md) page.

## The container image

The image is published as `ghcr.io/enzymead/reactantserver`, tagged with each ReactantServer
release (`:0.1.0`, the moving `:0.1`, and `:latest` for the newest). Its entrypoint is the node
supervisor, so one container runs the whole node: one worker per GPU it is given, plus the
embedded gateway when there are two or more. [Serve on One GPU](tutorials/container_single_gpu.md)
walks through running it; the repository's `docker-compose.yml` is the same deployment as a
compose file (`REACTANTSERVER_MODELS=/path/to/bundles docker compose up -d`).

Each published image carries signed [SLSA build provenance](https://slsa.dev/spec/v1.0/provenance)
naming the commit and workflow run that built it. Check it before deploying (with the GitHub CLI,
logged in):

```text
gh attestation verify oci://ghcr.io/enzymead/reactantserver:latest --repo EnzymeAD/ReactantServer.jl
```

Pin a deployment to a release tag such as `:0.1.0`, or by digest
(`ghcr.io/enzymead/reactantserver@sha256:...`) for an image that can never change, rather than
`:latest`, which moves with every release; `deploy/README.md` describes how the image is published
and attested.

Every model compiles to a device executable on every worker before the gRPC plane accepts
traffic, so the first start is slow (minutes to hours for a large model set). Compiled programs
are cached inside each bundle (`<bundle>/.cache/`), so later starts load them in milliseconds.
Watch readiness with `curl -sf http://127.0.0.1:8002/readyz` or the container's health status,
not the container state. On stop, the supervisor drains its workers on SIGTERM; give it room
before the runtime escalates to SIGKILL (`--stop-timeout`, or `stop_grace_period` in compose).

### Container settings

Most of these flags exist because a container's defaults are sized for small services, not for
a process that maps model weights into shared memory and runs several thread pools per GPU.

**GPU access.** The host needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
Podman reads GPUs through CDI (`--device nvidia.com/gpu=all`, or `nvidia.com/gpu=0` for one
card; generate the spec with `nvidia-ctk cdi generate`), Docker through `--gpus`. The supervisor
runs one worker per GPU it can see, so the GPUs you grant the container are the topology. The
image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`; keep `utility`, which injects
`nvidia-smi` and NVML for the entrypoint's GPU-reclaim wait and the out-of-pool memory metric. If CDI is unavailable on an older podman, bind
the device nodes and driver libraries instead; `deploy/README.md` lists them.

**Shared memory (`/dev/shm`).** Two features place data in POSIX shared memory: clients that send
tensors through the [shared-memory transport](client.md) (the client's staging pool is 256 MiB by
default), and `runtime.shared_host_weights`, which keeps one host copy of every model's weights
for all of a node's workers (see [On-demand Weights](on_demand_weights.md)). A container's own
`/dev/shm` defaults to 64 MiB, far too small for either. There are two ways to provide it:

- `--ipc=host` (compose: `ipc: host`) shares the host's IPC namespace and its `/dev/shm`, which
  is usually sized at half of RAM. This is also what lets a client running on the host, outside
  the container, use the shared-memory transport.
- A private namespace with an explicit size, `--shm-size 32g` (compose: `shm_size: 32g`), or a
  podman pod created with `--share ipc --shm-size` so that client containers in the same pod
  share it. Use this when the host namespace is not yours to share.

With `shared_host_weights` on, size `/dev/shm` above the combined weights of every model the node
serves, plus room for the client regions. A tmpfs is allocated lazily, so a generous limit costs
nothing until it is used. `--shm-size` does not apply together with `--ipc=host`.

**Process and thread limits.** Podman limits a container to 2048 processes by default
(`--pids-limit`), and the limit counts threads. Each worker runs a Julia compute pool, XLA's
asynchronous runners, and the host libraries' own pools, so a multi-GPU node can cross 2048
while it loads models; the worker then aborts with `pthread_create() failed` (errno 11, EAGAIN).
Pass `--pids-limit=-1` (compose: `pids_limit: -1`) to defer to the host's limits, or a value
large enough for your worker count. Docker sets no limit by default unless its daemon is
configured with one.

Each worker's compute pool is sized to its share of the CPUs Julia detects, `min(CPUs ÷ workers,
16)` plus one interactive thread (see [The node supervisor](@ref)); the startup
log prints the result as `worker compute threads: ...`. If you limit the container's CPUs (for
example with `--cpus`), check that line and set `REACTANT_WORKER_THREADS` to the number of
compute threads each worker should run.

**Model repository.** Mount it at `/var/lib/reactantserver/models`, writable: the compiled
executable cache lives inside each bundle, so a read-only mount recompiles every program on every
start. In `dynamic` mode the server watches the mount and hot-loads changes.

**Persistent caches.** Mount a volume at `/var/cache/reactant-compile` to keep XLA's autotune
results across container recreation; without it every start re-times every GEMM and convolution.
Never mount anything over `/opt/julia-depot/compiled`, which holds the image's precompiled
packages.

**Node file.** The image runs `/etc/reactantserver/node.yaml`, a copy of
`config/node.default.yaml` (`gpus: auto`, one worker per visible GPU). Mount your own file over
that path to change it; `config/node.yaml` is the commented template, and
[Node Configuration](node_config.md) covers every key and its `INFERENCE_SERVER_*` environment
override, which can be passed with `-e`.

**Health.** An OCI image cannot carry a healthcheck, so pass one at run time:
`/usr/local/bin/healthcheck.node.sh` reports healthy once the node's `/readyz` answers. Set the
start period longer than a cold start, which compiles every model before `/readyz` succeeds.

### Building the image

The image is built with Bazel from committed locks: `deploy/Manifest.toml` pins every Julia
package and artifact, and the CUDA base image is pinned by digest, so the same commit always
yields the same image. To build it and load it into podman:

```text
bazel run //deploy:image_load    # loads localhost/reactantserver:bazel
REACTANTSERVER_IMAGE=localhost/reactantserver:bazel docker compose up -d
```

See `deploy/README.md` for the build itself, the precompiled caches, pushing to a registry, and
consuming the image from another Bazel module, and `deploy/runtime/README.md` for the entrypoint
and healthcheck scripts.

## Outside the image

The supervisor runs the same way from a Julia environment, as in [Run a Node](tutorials/native/node.md).
Start it with `julia --handle-signals=no` so its own handler runs and it shuts its workers down on
SIGTERM. Unlike the image, such an environment resolves its dependencies itself, so it does not get
the image's locked versions or its precompiled caches. On a GPU host, select the Reactant CUDA build
before the environment precompiles, with `REACTANT_GPU=cuda` and `REACTANT_GPU_VERSION` (`12.9` or
`13.1`), and make sure the host provides the CUDA libraries Reactant loads by name
(`libnvJitLink.so.13`, `libnvrtc.so.13`, `libcupti.so`). `INFERENCE_SERVER_MODEL_DIRS` overrides the
node file's model repository (colon-separated). Use the container for anything long-running.

## Where the children load from

The supervisor never loads Reactant itself; each worker is a `julia` subprocess that loads
`ReactantServer`, and the gateway one that loads `ReactantServerGateway`. Before spawning
anything, the supervisor decides where those come from, first match wins:

1. **A workspace named by `REACTANT_WORKSPACE_ROOT`.** Each child runs from its own member project
   (`<root>/packages/ReactantServer`, `<root>/packages/ReactantServerGateway`). The image sets
   the variable to `/opt/reactantserver`.
2. **The supervisor's own environment**, when it lists `ReactantServer` (and
   `ReactantServerGateway`, if a gateway will run) as dependencies. This is the Pkg workflow: add
   the three packages to one environment and every child runs in it, with `JULIA_LOAD_PATH`
   limited to that environment and the standard library so a different copy in your default
   environment cannot be picked up instead. ReactantServerNode declares the other two as weak
   dependencies with compat bounds, so Pkg keeps the three on versions that work together.
3. **A checkout around the active project**: `--project=packages/ReactantServerNode` or
   `--project=.` in a clone of this repository.

What is needed depends on the role: `all` needs both packages, `workers` only `ReactantServer`,
and `gateway` only `ReactantServerGateway`. When nothing provides them, the supervisor stops with
a `ConfigError` naming the packages to add, rather than starting children that cannot load and
restarting them indefinitely.

## Beyond one host

The project deliberately does not ship a multi-node control plane: anyone who needs multi-node
coordination is usually at a scale where a generic one would not fit. Instead, each node exposes
the interface a control plane integrates against. The endpoint contract is the KServe V2 gRPC data
plane (`ModelInfer`, `RepositoryIndex`, `ServerReady`) plus the worker control RPCs
(`ModelControlStatus`, `SetModelResidency`, `SetModelPolicy` for residency and policy,
`CompactMemory` to defragment device memory), and an admin HTTP port serving `/healthz`,
`/readyz`, and Prometheus `/metrics`. A node's embedded gateway additionally answers
`GatewayControlService` for its own scheduling state. A control plane discovers which models a
node serves via `RepositoryIndex` and routes `ModelInfer` to a node that reports the model ready.
The `workers` and `gateway` values of `REACTANT_ROLE` split a node's roles across machines.

The simplest multi-node setup reuses the standalone gateway pointed at every node's worker
endpoints (see [Front Workers with a Gateway](tutorials/native/gateway.md)); a larger deployment
replaces it with its own control plane speaking the same gRPC interface. Who the project is built
for, and how each deployment shape serves that mission, is on the
[Philosophy](design/philosophy.md) page.
