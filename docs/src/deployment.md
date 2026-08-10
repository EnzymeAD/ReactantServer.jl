# Deployment

A node runs as a single supervisor process (`ReactantServerNode`): it detects every visible GPU,
spawns one single-GPU worker subprocess per device, multiplexes their logs onto one stream with
`[name]` line prefixes, and restarts children that die. With two or more workers it also runs an
embedded gateway on the public ports; with a single worker it binds that worker to the public
ports directly. The external interface is the same either way: KServe V2 gRPC on `:8001`, health
and metrics on `:8002` (`/readyz`, `/healthz`, `/metrics`), matching Triton's ports.

The supported deployment is **native** (no containers): the supervisor uses the host's NVIDIA
driver directly. A container image is also provided as an alternative and runs the same node with
the same interface; see [Docker (container) deployment](#docker-container-deployment) below.

## Audience and mission

The project targets small and mid-size labs and engineering organizations that need big-tech-level
efficiency without big-tech scale: startups and small companies serving production ML where each
GPU is a meaningful fraction of infrastructure spend, scientific and research groups operating
within bounded compute budgets, and on-premise or cloud deployments where GPU-hours are the
dominant cost line. The audience is engineers and operators who understand what is happening
inside their systems; the project gives them tools rather than hiding the system's behavior behind
convenient defaults. Larger organizations can use the project too, as a component of a larger
system; the difference is what ships in the box, not what is possible.

The mission is to make serving compiled (non-LLM) models elegant: a hackable, Julia-first
inference stack that maximizes the economic efficiency of GPU-based inference. GPU memory is
roughly two thirds of GPU cost, so serving infrastructure that wastes memory wastes money; the
concrete goal is serving the largest number of models per GPU at a given quality of service. The
whole stack is plain Julia, legible end to end, adoptable off the shelf, and open to being bent
toward a workload nobody anticipated.

The project's explicit non-goals, from its design:

- **Hyperscale platform requirements.** Multi-tenant isolation, complex traffic shaping, and deep
  integration with bespoke internal platforms are not built into the core, where every smaller
  deployment would pay for them. Large deployments are supported through the control-plane seam
  described under [Multi-node](#multi-node-bring-your-own-control-plane).
- **LLM serving at scale.** vLLM, TGI, TensorRT-LLM, and similar projects are purpose-built for
  that domain and do it well; this project does not compete there.
- **A packaged, managed solution.** This is infrastructure for builders, not a hosted service.
  Users who do not want to think about the underlying architecture should choose a managed
  inference service.
- **Multi-framework serving.** A model must be lowered by Reactant to a device executable first
  (today via StableHLO/XLA). Teams that need to serve PyTorch, TensorFlow, and ONNX models side by
  side without converting them are better served by Triton or similar.

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

## Running natively

Instantiate the workspace once, then run the node supervisor. `ReactantServerNode.main()` reads
`REACTANT_NODE_FILE` for the node config and honors the standard environment overrides:

```text
# once, to resolve and precompile the workspace (selects the CUDA build via REACTANT_GPU_*):
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 \
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# run the supervisor across four GPUs, serving a bundle directory:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles \
REACTANT_NODE_FILE=config/node.default.yaml \
  julia --handle-signals=no --project=packages/ReactantServerNode \
    -e 'using ReactantServerNode; ReactantServerNode.main()'
```

`--handle-signals=no` lets the supervisor's own handler run so it shuts its worker children down
on SIGTERM. `REACTANT_GPU_VERSION` selects the Reactant CUDA build (`12.9` or `13.1`) and must be
set before `instantiate`. `INFERENCE_SERVER_MODEL_DIRS` overrides the node file's model repository
(colon-separated); the runtime tunables under [Node Configuration](node_config.md) take
`INFERENCE_SERVER_*` overrides the same way. For an always-on service, run this command under a
process manager such as systemd, with `Restart=on-failure` and a SIGTERM-based graceful stop
(`KillMode=mixed` pairs with `--handle-signals=no`).

Every model compiles to a device executable on every worker before the gRPC plane accepts traffic,
so first startup is slow (minutes to hours for a large model set). Watch readiness with
`curl -sf http://127.0.0.1:8002/readyz`, not the process state.

## Running under systemd

For an always-on node, run the supervisor from a system service. Put the tunables in an
`EnvironmentFile` and let the unit run the same command as above. Adjust the user, the checkout
path, and the GPU list for your host.

`/etc/reactantserver/reactantserver.env`:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles
REACTANT_NODE_FILE=config/node.default.yaml
REACTANT_GPU=cuda
REACTANT_GPU_VERSION=13.1
```

`/etc/systemd/system/reactantserver.service`:

```text
[Unit]
Description=ReactantServer node supervisor
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=YOUR_DEPLOY_USER
WorkingDirectory=/path/to/ReactantServer.jl
EnvironmentFile=/etc/reactantserver/reactantserver.env
# Absolute path to julia: systemd does not source your shell rc, so a juliaup install under the
# user's home is not on PATH. `julia --version` in a login shell shows the binary to use here.
ExecStart=/home/YOUR_DEPLOY_USER/.juliaup/bin/julia --handle-signals=no --project=packages/ReactantServerNode -e 'using ReactantServerNode; ReactantServerNode.main()'
Restart=on-failure
RestartSec=10
# First boot compiles every model on every worker (minutes to hours) AFTER the unit is already
# active; systemd cannot gate that (the supervisor sends no sd_notify). Check readiness with
# `curl -sf http://127.0.0.1:8002/readyz`, not `systemctl is-active`.
TimeoutStartSec=infinity
# Graceful stop: SIGTERM to the supervisor only, which drains its workers; pairs with
# --handle-signals=no. Anything still alive after TimeoutStopSec is SIGKILLed.
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
```

Enable and watch it:

```text
sudo systemctl daemon-reload
sudo systemctl enable --now reactantserver.service
journalctl -u reactantserver -f
until curl -sf http://127.0.0.1:8002/readyz; do sleep 15; done; echo READY
```

`sudo systemctl stop reactantserver` sends SIGTERM to the supervisor, which drains its workers
within `TimeoutStopSec` before exiting. Run the workspace `Pkg.instantiate()` once (as in the
previous section) before enabling the unit, so the first start is not also resolving dependencies.

## Docker (container) deployment

A container image is an alternative to running the supervisor directly. It runs the same node
(supervisor + workers + embedded gateway) with the same `:8001`/`:8002` interface:

```text
git submodule update --init lib/gRPCServer.jl
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 julia --project=. -e 'using Pkg; Pkg.instantiate()'  # Manifest.toml is gitignored
make image        # or: docker build -f docker/Dockerfile -t reactantserver .
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

`make image` builds through the configured engine (`ENGINE`, default `podman`). The equivalent
without compose is:

```text
docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
```

The image is a multi-stage build on `julia:1.12.6-trixie` that copies only `libnvJitLink.so.13`
(the one CUDA userspace library Reactant needs but does not bundle) from an official CUDA 13.1
image; it deliberately does not bring the rest of the CUDA userspace, so the base cannot shadow
Reactant's bundled cuDNN/NCCL. Why that one library matters: Reactant statically links cuBLASLt
into `libReactantExtra.so`, but CUDA-13 cuBLASLt dlopens `libnvJitLink.so.13` to JIT its GEMM
kernels, and the library is neither in the Reactant artifact nor injected by the NVIDIA container
runtime (which adds only `libcuda.so.1`). A bare Julia image without it aborts at the first GPU
compile with cuBLASLt's "Invalid handle was passed to cublasLtCreate"; the copied library fixes
that.

The container needs the host NVIDIA Container Toolkit for GPU access, and the compose file mounts
the model repository read-only plus a persistent volume for the Reactant compile cache (autotune
results), so tuned kernels survive container recreation. The container shares the host IPC
namespace (`ipc: host`) so POSIX shared-memory regions created by a client are visible to the
workers. The autotune knobs are settable as container env
(`INFERENCE_SERVER_RUNTIME_AUTOTUNE`, `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE`,
`INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE_DIR`); the baked default node file sits at
`/etc/reactantserver/node.yaml` and can be overridden by mounting your own over that path. See
`docker/README.md` for details.

## Metrics

One scrape on `:8002` covers everything. With multiple workers, the embedded gateway serves its
own `gateway_*` series and fans out to each worker's metrics endpoint, merging them into a single
exposition; with a single worker, `:8002` is that worker's own `/metrics`. Each worker tags its
series with `worker` and `gpu` labels itself (the `gpu` value is the physical device behind its
`CUDA_VISIBLE_DEVICES`), so per-GPU and per-worker breakdowns need no Prometheus relabeling, for
example `sum by (gpu) (rate(worker_dispatch_total[1m]))`.

A ready-to-run Prometheus + Grafana stack lives under `config/monitoring/` with a seven-dashboard
suite. Because the node runs natively (not in a container), that stack's Prometheus scrapes the
host at `host.docker.internal:8002` rather than over a Docker network; if the node listens on a
different host, edit the target in `prometheus.yml` to that host's `address:8002`. Grafana is at
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
