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
and a container runtime with the NVIDIA Container Toolkit. See
[Running the container](@ref) below. Running the supervisor straight from a
source checkout also works and is handy for development; see
[Running from source](@ref).

## Audience and mission

The project targets small and mid-size labs and engineering organizations that need big-tech-level
efficiency without big-tech scale: startups and small companies serving production ML where each
GPU is a meaningful fraction of infrastructure spend, scientific and research groups operating
within bounded compute budgets, and on-premise or cloud deployments where GPU-hours are the
dominant cost line. The audience is engineers and operators who understand what is happening
inside their systems; the project gives them tools rather than hiding the system's behavior behind
convenient defaults. Larger organizations can use the project too, as a component of a larger
system; the difference is what ships in the box, not what is possible.

The mission is to make serving compiled models elegant: a hackable, Julia-first inference stack
that maximizes the economic efficiency of GPU-based inference. GPU memory is roughly two thirds of
GPU cost, so serving infrastructure that wastes memory wastes money; the concrete goal is serving
the largest number of models per GPU at a given quality of service, and serving models larger than
a GPU's memory. Local serving of large language models is a goal of the project, built on splitting
a model into compiled stages and streaming their weights from storage; see
[Philosophy](design/philosophy.md). The
whole stack is plain Julia, legible end to end, adoptable off the shelf, and open to being bent
toward a workload nobody anticipated.

Where other tools fit better:

- **Hyperscale platform requirements.** Multi-tenant isolation, complex traffic shaping, and deep
  integration with bespoke internal platforms stay out of the core, where every smaller deployment
  would pay for them. Large deployments bring that machinery through the control-plane seam
  described under [Multi-node](#multi-node-bring-your-own-control-plane).
- **Hosted, high-concurrency LLM services.** Engines such as vLLM, TGI, and TensorRT-LLM are
  purpose-built for many concurrent users across many GPUs; the project's LLM focus is local
  serving.
- **A managed service.** The project is infrastructure for builders who run their own stack.
  Teams that would rather not operate one are better served by a managed inference service.
- **Serving models without converting them.** Every model is lowered by Reactant to a device
  executable (today via StableHLO/XLA). Teams that need to serve PyTorch, TensorFlow, and ONNX
  models side by side unconverted are better served by Triton or similar.

## Deployment shapes

The deployment decision comes down to how many GPUs you have and, with more than one, whether
your constraint is memory or compute:

| Situation | Shape | Optimizes for |
|---|---|---|
| One GPU, many models | Single GPU | Fitting many models on limited hardware |
| Several GPUs, models do not all fit everywhere | Multi-GPU distributed | Serving more models than any one GPU holds |
| Several GPUs, models fit everywhere, need throughput | Multi-GPU replicated | Spreading compute load across replicas |
| Many machines | Multi-node | Whatever your control plane decides |

Each shape is selected by a configuration value, not a different architecture, and each is
strictly additive: a single-GPU deployment never pays for the gateway in dependencies or runtime
cost, and a gateway deployment never requires a control plane.

### Single GPU

A single worker is the entire deployment. The worker speaks the full KServe V2 gRPC surface
itself: one Julia process, one YAML file, nothing else to operate. This is the recommended
starting point for small labs and the conceptual foundation for the distributed multi-GPU case,
which scales the same idea across cards. The [Scheduling](scheduling.md) page covers the fair
scheduler and batch coalescing; [On-demand Weights](on_demand_weights.md) covers the weight
cache that serves a catalog larger than device memory.

### Multi-GPU, distributed without replication

You have several cards, but your model library is large enough that it cannot fit on every GPU.
Your constraint is memory. The gateway uses LPT (longest-processing-time) packing to place each
model on one GPU, balancing memory footprint against compute load; spreading models across cards
by their load keeps any one card from being monopolized. The workers switch to the simpler FIFO
discipline and the placement intelligence moves upstream to the gateway. There is still no
external infrastructure to stand up: the gateway is one more Julia process and one more YAML file,
and there is no placement file to maintain.

### Multi-GPU, replicated

You have enough memory that your models fit on more than one card. Your constraint is compute:
you want to spread a model's request load across replicas for throughput. The gateway gives a
replicated model a replica count, places it on that many distinct GPUs, and routes its requests
to fill one replica's batch before moving to the next, so batch coalescing is preserved across
replicas. Size replica counts for the model's expected concurrency, and make sure the memory is
there: replicating a model puts its full weight footprint on every GPU it lands on, and an
over-subscribed GPU thrashes, loading and evicting weights on nearly every request. See
[Multi-GPU Gateway](gateway.md) for the scheduling modes and knobs.

### Multi-node (bring your own control plane)

Beyond a single host, the project deliberately does not ship a multi-node control plane: anyone
who needs multi-node coordination is usually at a scale where a generic control plane would not
fit. Instead, each node exposes the interface a control plane integrates against, so you can build
or adapt your own coordination layer on top of ReactantServer nodes. The endpoint contract is the
KServe V2 gRPC data plane (`ModelInfer`, `RepositoryIndex`, `ServerReady`) plus the worker
control RPCs (`ModelControlStatus`, `SetModelResidency`, `SetModelPolicy` for residency and
policy, `CompactMemory` to defragment device memory), and an admin HTTP port serving `/healthz`,
`/readyz`, and Prometheus `/metrics`. A node's embedded gateway additionally answers
`GatewayControlService` for its own scheduling state. Your control plane discovers which models a
node serves via `RepositoryIndex` and routes `ModelInfer` to a node that reports the model ready.
The project supplies the seam; the organization supplies the tooling that encodes needs only it
can know.

The simplest multi-node setup reuses the standalone gateway pointed at every node's worker
endpoints; a larger deployment replaces it with your own control plane speaking the same gRPC
interface. See [Multi-GPU Gateway](gateway.md) for the standalone gateway's configuration.

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

One [`ReactantServerNode.supervise`](@ref ReactantServerNode.supervise) call on the multi-GPU host
does the whole fan-out, just as it does for one GPU:

```julia
using ReactantServerNode
ReactantServerNode.supervise("node.yaml")   # one worker per GPU + the embedded gateway
```

This single parent process spawns one [`ReactantServer.serve`](@ref) worker subprocess per GPU and
the gateway as another subprocess, multiplexes their logs onto its stdout with `[worker0]` /
`[gateway]` prefixes, and restarts any child that dies. (You can still run each worker and the
gateway by hand, as separate `serve` / `serve_gateway` processes, but the supervisor is the
intended path and the one the launcher uses.) `ReactantServerNode.main()` is the container
entrypoint: it reads `REACTANT_NODE_FILE` for the node config and runs the supervisor to
completion.

### Watch it without a GPU

To see the decision logic and the prefixed logs on a machine with no GPU, run the supervisor as a
CPU node with two synthetic workers, with `backend: cpu` in the node file:

```text
REACTANT_GPUS=0 REACTANT_CPU_WORKERS=2 \
  julia --project=packages/ReactantServerNode -e 'using ReactantServerNode; ReactantServerNode.supervise("node.yaml")'
```

You will see `worker0`, `worker1`, and `gateway` start, their logs interleaved with `[name]`
prefixes, and the gateway serving on 8001/8002, exactly the multi-worker shape it takes on a
multi-GPU host.

### The node file is unchanged when you scale

`gpus: auto` already means "one worker per visible GPU", so the single-GPU node file from the
[Tutorial](tutorial.md) scales as-is: give the host more GPUs and the supervisor runs more
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

## Running the container

The image is published as `ghcr.io/enzymead/reactantserver:latest`. Its entrypoint is the node
supervisor, so one container runs the whole node: one worker per GPU it is given, plus the
embedded gateway when there are two or more. Mount a model repository and publish the two ports:

```text
podman run -d --name reactantserver \
  --device nvidia.com/gpu=all \
  --ipc=host --pids-limit=-1 \
  -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models \
  -v reactant-compile-cache:/var/cache/reactant-compile \
  --health-cmd /usr/local/bin/healthcheck.node.sh --health-start-period 3600s \
  --stop-timeout 30 \
  ghcr.io/enzymead/reactantserver:latest
```

With Docker, replace `--device nvidia.com/gpu=all` with `--gpus all`; the other flags are the
same. The repository's `docker-compose.yml` is the same deployment as a compose file:

```text
REACTANTSERVER_MODELS=/path/to/bundles docker compose up -d
```

Each published image carries signed [SLSA build provenance](https://slsa.dev/spec/v1.0/provenance)
naming the commit and workflow run that built it. Check it before deploying (with the GitHub CLI,
logged in):

```text
gh attestation verify oci://ghcr.io/enzymead/reactantserver:latest --repo EnzymeAD/ReactantServer.jl
```

Pin a deployment by digest (`ghcr.io/enzymead/reactantserver@sha256:...`) rather than `:latest`,
which moves with every publish; `deploy/README.md` describes how the image is published and
attested.

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

## Running from source

For development, or a quick look on a machine where you already have Julia, run the supervisor
from a checkout. This is the same entry point the image runs:

```text
# once, to resolve and precompile the workspace (selects the CUDA build via REACTANT_GPU_*):
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 \
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CUDA_VISIBLE_DEVICES=0 \
INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles \
REACTANT_NODE_FILE=config/node.default.yaml \
  julia --handle-signals=no --project=packages/ReactantServerNode \
    -e 'using ReactantServerNode; ReactantServerNode.main()'
```

`--handle-signals=no` lets the supervisor's own handler run so it shuts its workers down on
SIGTERM. `REACTANT_GPU_VERSION` selects the Reactant CUDA build (`12.9` or `13.1`) and must be
set before `instantiate`. `INFERENCE_SERVER_MODEL_DIRS` overrides the node file's model
repository (colon-separated). Unlike the image, a checkout resolves its dependencies fresh, so it
does not get the image's locked versions or its precompiled caches, and the host needs the CUDA
libraries Reactant loads by name (`libnvJitLink.so.13`, `libnvrtc.so.13`, `libcupti.so`). Use
the container for anything long-running.

## Metrics

One scrape on `:8002` covers everything. With multiple workers, the embedded gateway serves its
own `gateway_*` series and fans out to each worker's metrics endpoint, merging them into a single
exposition; with a single worker, `:8002` is that worker's own `/metrics`. Each worker tags its
series with `worker` and `gpu` labels itself (the `gpu` value is the physical device behind its
`CUDA_VISIBLE_DEVICES`), so per-GPU and per-worker breakdowns need no Prometheus relabeling, for
example `sum by (gpu) (rate(worker_dispatch_total[1m]))`.

A ready-to-run Prometheus + Grafana stack lives under `config/monitoring/` with a seven-dashboard
suite. Its Prometheus scrapes `host.docker.internal:8002`, the metrics port the node container
publishes on the host; if the node runs on a different host, edit the target in `prometheus.yml`
to that host's `address:8002`. Grafana is at
`http://<host>:3000` (anonymous viewing on; `admin` / `admin` to edit) and Prometheus at
`http://<host>:9090`. See `config/monitoring/README.md` for the compose commands and dashboards.

## Security

ReactantServer is designed to run on a trusted network behind your own perimeter. Be aware of the
following before exposing any endpoint:

- **Cleartext h2c.** All gRPC traffic (worker and gateway) is cleartext h2c. TLS settings are
  parsed by the gateway config but not yet enforced; a configured cert triggers a startup warning.
- **No authentication or authorization** on the KServe data plane, the worker control-plane RPCs
  (residency and policy), the gateway's own scheduling control plane (`GatewayControlService`,
  which can repack the fleet and change a model's replica count and routing at runtime), or the
  Prometheus metrics listener (which binds `0.0.0.0:8002` by default). The control plane is
  mutating, so treat access to the gRPC ports as administrative access.
- **Trusted bundles.** A bundle's optional `model.jl` executes arbitrary Julia in the server
  process, so bundles are trusted input. Only serve bundles you built or audited; see
  [Bundles](bundles.md).
- **Shared memory is a local trust boundary.** Client-registered regions and the optional
  node-shared host-weight store live in `/dev/shm`; the shared weight regions default to mode
  `666` (world-writable) for friction-free sharing. Set `runtime.shared_host_weights_mode: "660"`
  on production or multi-user systems.
