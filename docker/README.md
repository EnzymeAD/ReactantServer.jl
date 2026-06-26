# Docker deployment

One container runs the whole node. The image's default entrypoint is the supervisor
(`ReactantServerNode`): it detects every GPU granted to the container, spawns one single-GPU
worker subprocess per device, multiplexes all logs onto the container's stdout with `[worker0]` /
`[gateway]` line prefixes, and restarts children that die. With two or more workers it also runs
an embedded gateway on the public ports; with a single worker it skips the gateway and binds that
worker to the public ports directly (a lone worker serves the full KServe V2 API itself). The
external interface (8001/8002) is the same either way.

```
docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
```

Clients connect to the KServe V2 gRPC endpoint on `localhost:8001`; health and metrics are on
`localhost:8002` (`/readyz`, `/healthz`, `/metrics`), matching Triton's ports. Adding a GPU to
the host changes nothing in the configuration: the supervisor sees one more device and spawns
one more worker.

## Files

- `Dockerfile` — the `reactantserver` image (`julia:1.12.5-trixie`). Copies the whole
  `packages/` tree and builds the workspace root (so the shared `Manifest.toml` pins the
  HTTP/Reactant forks). Default entrypoint is the supervisor; `entrypoint.worker.sh` remains in
  the image as the single-worker escape hatch.
- `entrypoint.node.sh` — launches `ReactantServerNode.main()` (the supervisor).
- `entrypoint.worker.sh` — launches a single `ReactantServer.serve` worker named by
  `REACTANT_WORKER_NAME` (the single-worker escape hatch).
- `healthcheck.node.sh` — role-aware container healthcheck: curls the gateway's `/readyz` in the
  `all`/`gateway` roles (for a single worker, the worker's own `/readyz`), falls back to the
  Julia worker probe in the `workers` role.
- `healthcheck.worker.jl` — lightweight Julia readiness probe (imports only gRPCClient and
  YAML). With `REACTANT_WORKER_NAME` set it probes that worker; unset, it probes every worker in
  the node file and passes when at least one is ready.
- `node.default.yaml` — the zero-config node file baked into the image at
  `/etc/reactantserver/node.yaml` (`gpus: auto`, no workers list).
- `node.yaml` — the fully commented node-file template to mount over the baked default.
- `gateway.yml` — optional config for the embedded gateway (scheduling knobs); the supervisor
  synthesizes the worker endpoints, so it needs no file by default.
- `../docker-compose.yml` — the single-service stack (equivalent to the `docker run` above).

## Prerequisites

1. Populate the vendored submodules:
   ```
   git submodule update --init --recursive
   ```
   This fetches `lib/Reactant.jl`, `lib/gRPCServer.jl`, `lib/gRPCClient.jl`, and `lib/HTTP.jl`.
2. Install the NVIDIA Container Toolkit on the host (for GPU access).
3. Have a model bundle repository on the host. Each immediate subdirectory with a
   `manifest.yaml` is a bundle; its directory name is the model name.

## Build and run

```
make image        # or: docker compose build
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

Every model compiles (Reactant -> device executable) before the gRPC plane accepts traffic, so
first startup is slow and the image healthcheck's `start_period` is generous (300s); raise it
for large model sets. Logs from all workers and the gateway appear on the container's stdout,
one line each, prefixed `[worker0]`, `[worker1]`, ..., `[gateway]`, `[supervisor]`.

## Configuring

Zero configuration is the default: the baked node file sets `gpus: auto` and the supervisor
synthesizes `worker0..workerN-1`, one per detected device. To customize, mount a node file over
`/etc/reactantserver/node.yaml` (see `node.yaml` for the commented template):

- `gpus:` — `auto` (default), an integer count, or an explicit device list (ordinals or GPU
  UUIDs). The `REACTANT_GPUS` environment variable overrides this key.
- `workers:` — optional. When present it wins over auto-detection: device i goes to worker i
  (or a worker's `gpu:` key picks a specific visible device). Per-worker config override blocks
  merge over `global:` as before.
- `models:` — optional per-model pinning to device memory on named workers.
- `global:` — defaults merged into every worker (runtime, scheduler, cache_dir, endpoints).

Supervisor environment variables: `REACTANT_GPUS` (count or device list; `0` for a CPU node),
`REACTANT_CPU_WORKERS` (workers on a CPU node, default 1), `REACTANT_WORKER_THREADS` (compute
threads per worker; default is the host's share, `min(CPU_THREADS ÷ workers, 16)`, so co-located
workers do not oversubscribe the CPU), `REACTANT_ROLE` (below),
`REACTANT_SUPERVISOR_MAX_RESTARTS` (consecutive crash budget per child before the node exits 1;
default unlimited, with the healthcheck reporting unready instead), `REACTANT_NODE_FILE`, and
`REACTANT_GATEWAY_FILE`.

The supervisor writes the materialized node file (with the synthesized workers list) to
`/run/reactantserver/node.yaml` for inspection; children and the healthcheck read that file.

## Roles

`REACTANT_ROLE` selects what the supervisor runs. The default is `all` (workers plus the embedded
gateway, on one host), which is the only documented deployment here. The `workers` and `gateway`
roles exist in the code to split a deployment across machines, but multi-node is not a shipped
example.

## Metrics

One scrape on `8002` covers everything. With multiple workers, the embedded gateway's `/metrics`
serves its own `gateway_*` series and fans out to every worker's metrics endpoint, merging the
results into a single exposition; with a single worker (no gateway), `8002` is that worker's own
`/metrics` directly. Either way, each worker tags all of its series (`worker_*` plus
`process_*`/`julia_gc_*`) with `worker` and `gpu` labels itself, where `gpu` is the physical
device behind the worker's `CUDA_VISIBLE_DEVICES`, so nothing needs to be configured in
Prometheus:

```yaml
scrape_configs:
  - job_name: reactantserver
    static_configs:
      - targets: ['node:8002']
```

Per-GPU and per-worker views fall out of the labels, e.g.
`sum by (gpu) (rate(worker_dispatch_total[1m]))` or
`worker_queue_wait_seconds{worker="worker1",quantile="0.99"}`. Per-endpoint scrape health is
reported as `gateway_worker_metrics_up{endpoint=...}`.

Worker metrics include per-model dispatch count, GPU compute seconds, queue depth and wait
quantiles, weight-cache load/evict churn, and device memory. With two or more workers each also
serves its own `/metrics`, `/healthz`, `/readyz` on `metrics_base_port + i` (`worker0` → `9100`,
…); publish that range (`-p 9100-9107:9100-9107`) only if you also want to scrape workers
directly. The supervisor wires the embedded gateway's aggregation automatically.

## Gateway scheduling

The gateway's `scheduling:` block (gateway.yml, or `REACTANT_GATEWAY_SCHEDULING_*` env for the
embedded gateway) selects how requests spread across workers. `round_robin` (default) rotates
each model's requests uniformly over its replicas. `lpt_packing` concentrates each model's
traffic on as few GPUs as the load allows, computed from the measured arrival rate and the
workers' reported compute cost, so the workers' batch coalescing sees deep same-model queues;
placement is rebalanced as the fleet consumes compute, with the demand signal smoothed so it stays
stable, and is observable via `gateway_placement_weight{model,worker}`. It requires `scheduler.discipline: fifo` in the node
file and all models loaded on all workers (the load-all default); the gateway refuses to start
otherwise. See the `scheduling:` block in `docker/gateway.yml` for the knobs (or set
`REACTANT_GATEWAY_SCHEDULING_*` for the embedded gateway).

The FIFO requirement is by design, not a limitation: under lpt_packing the gateway is the single
fairness authority (concentration plus the per-worker share cap), so a worker-level fair
scheduler would fight the placement by throttling exactly the models the gateway concentrated
for batching. Keep the worker `fair` discipline for deployments without an upstream placement
authority: a single-GPU worker, or a multi-GPU fleet served round-robin.

## Single-GPU soak test

`docker-compose.gpu2.yml` brings up the supervised container on one GPU to exercise inference
with dummy data and watch for memory leaks, races, and instability: the supervisor runs one
worker on GPU 2 (`CUDA_VISIBLE_DEVICES=2`) and no gateway, the worker serving 8001/8002 directly,
exactly as a production single-GPU deployment does, alongside a `loadgen` service that drives
sustained concurrent requests. It uses on-demand weight caching so every bundle need not be
GPU-resident at once.

Files:

- `Dockerfile.loadgen` / `entrypoint.loadgen.sh` / `loadgen/loadgen.jl` — a light, Reactant-free
  load generator built from `ReactantServerClient`. It reads each manifest (`manifest_io_spec`)
  to synthesize correctly shaped zero inputs, then fires concurrent inferences at the gateway.
- `monitor_gpu2.sh` — host-side CSV logger (nvidia-smi for GPU 2 plus docker stats for the
  container) for leak detection; GPU memory and container RSS should plateau, not climb.

Prerequisites: built images (`make image loadgen`), checked-out submodules, and the
NVIDIA container runtime configured for Docker
(`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

Run:

```
REACTANTSERVER_MODELS=/path/to/bundles docker compose -f docker-compose.gpu2.yml up -d
docker/monitor_gpu2.sh &          # optional: log GPU/RSS to soak_monitor.csv
docker compose -f docker-compose.gpu2.yml down
```

The worker compiles and warms up every model's executables before it serves, so first startup is
slow (potentially hours for all 85 bundles). The worker healthcheck `start_period` is set high
to cover this. For a quick check, set `LOADGEN_MODELS` to a couple of bundle names in the
compose file and mount only those bundles.

Load parameters are the `loadgen` service's `LOADGEN_*` environment variables:
`LOADGEN_TRANSPORT` (`tcp`, `shm`, or `mixed`; `shm` exercises shared-memory
register/unregister), `LOADGEN_CONCURRENCY`, `LOADGEN_DURATION_SECONDS`,
`LOADGEN_REPORT_SECONDS`, and `LOADGEN_MODELS`. The server under test sizes its on-demand weight
cache from `runtime.weight_cache_fraction` (default 1.0, auto-sized at startup). The loadgen prints rolling throughput, latency, and error
counts, plus the fleet weight-cache `loads=`/`evicts=` totals (with per-window deltas, scraped
from the aggregated `/metrics`), and exits nonzero if any request errored. A steadily rising
`evicts` means the model set does not fit resident and the workers are reloading weights from host
to device — weight thrash, distinct from CPU oversubscription.

## Two-GPU lpt_packing soak test

`docker-compose.gpu23.yml` is the multi-GPU counterpart, exercising the gateway's placement and
coalescing-aware routing rather than a lone worker: the supervisor runs two workers (GPUs 2 and 3,
`CUDA_VISIBLE_DEVICES=2,3`) behind the embedded gateway, with the gateway put in `lpt_packing`
mode via `REACTANT_GATEWAY_SCHEDULING_*` environment. It mounts `docker/node.gpu23.yaml`, which
sets `scheduler.discipline: fifo` (required by `lpt_packing`); every worker loads every bundle, so
the gateway places each model on one of the two GPUs and routes its requests to fill batches. Same
prerequisites and `LOADGEN_*` knobs as the single-GPU stack:

```
REACTANTSERVER_MODELS=/path/to/bundles docker compose -f docker-compose.gpu23.yml up
```

During warmup the embedded gateway waits for both workers to come up (it logs which are pending)
before serving; that is expected.
