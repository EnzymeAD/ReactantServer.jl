# Docker deployment

A `docker compose` setup for a multi-GPU ReactantServer deployment: one single-GPU worker per GPU
behind one `reactant-gateway`. All services read a single cluster file, `docker/cluster.yaml`.

## Files

- `Dockerfile.worker` — ReactantServer.jl worker (`julia:1.12.5-trixie`). Copies the whole
  `packages/` tree and builds the workspace root (so the shared `Manifest.toml` pins the
  HTTP/Reactant forks); the entrypoint runs the `ReactantServer` member project.
- `Dockerfile.gateway` — pure-Julia `reactant-gateway` (`julia:1.12.5-trixie`). Built from the
  `ReactantServerGateway` member alone (just `ReactantServerCore` + the gRPC/HTTP forks), so the
  image pulls **no Reactant**.
- `entrypoint.worker.sh` — launches `ReactantServer.serve` (project
  `packages/ReactantServer`) for the worker named by `REACTANT_WORKER_NAME`.
- `entrypoint.gateway.sh` — launches `ReactantServerGateway.serve_gateway` (project
  `packages/ReactantServerGateway`) against the cluster file.
- `healthcheck.worker.jl` — lightweight Julia worker readiness probe (imports only gRPCClient and
  YAML), used as the worker container's healthcheck; run against the workspace root project.
- `cluster.yaml` — the shared cluster config mounted into every service.
- `../docker-compose.yml` — the two-worker + gateway stack.

## Prerequisites

1. Populate the vendored submodules:
   ```
   git submodule update --init --recursive
   ```
   This fetches `lib/Reactant.jl`, `lib/gRPCServer.jl`, `lib/gRPCClient.jl`, and `lib/HTTP.jl`.
2. Install the NVIDIA Container Toolkit on the host (for GPU access).
3. Have a model bundle repository on the host. Each immediate subdirectory with a
   `manifest.yaml` is a bundle; its directory name is the model name used in `cluster.yaml`.

## Build and run

```
docker compose build
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

Clients connect to the gateway's KServe V2 gRPC endpoint on `localhost:8001`; health and
metrics are on `localhost:8002` (`/readyz`, `/healthz`, `/metrics`), matching Triton's metrics port.

## Metrics

The gateway exposes `gateway_*` metrics on its admin port (`8002`). Each worker also exposes its
own `worker_*` metrics (per-model dispatch count, GPU compute seconds, queue depth and wait
quantiles, weight-cache load/evict churn, device memory, plus `process_*`/`julia_gc_*`) on the port
derived from `metrics_base_port` in the node file (`worker0` → `9100`, `worker1` → `9101`, …),
serving `/metrics`, `/healthz`, `/readyz`.

Scrape each worker as a separate Prometheus target and label it by GPU (the worker can't know its
physical GPU under per-container `CUDA_VISIBLE_DEVICES`, so the mapping lives here):

```yaml
scrape_configs:
  - job_name: reactantserver-gateway
    static_configs:
      - targets: ['gateway:8002']
  - job_name: reactantserver-worker
    static_configs:
      - { targets: ['worker0:9100'], labels: { worker: worker0, gpu: '2' } }
      # add one line per worker/GPU as you scale out
```

Then per-GPU views fall out of the `gpu` label, e.g.
`sum by (gpu) (rate(worker_dispatch_total[1m]))` or
`worker_queue_wait_seconds{gpu="2",quantile="0.99"}`. The in-band `worker_info{worker=...}` metric
carries the same handle when relabeling isn't configured.

## Gateway scheduling

The gateway's `scheduling:` block (gateway.yml) selects how requests spread across workers.
`round_robin` (default) rotates each model's requests uniformly over its replicas. `lpt_packing`
concentrates each model's traffic on as few GPUs as the load allows, computed from the measured
arrival rate and the workers' reported compute cost, so the workers' batch coalescing sees deep
same-model queues; placement is rebalanced periodically with hysteresis and is observable via
`gateway_placement_weight{model,worker}`. It requires `scheduler.discipline: fifo` in the
node file and all models loaded on all workers (the load-all default); the gateway refuses to
start otherwise. See `docker/gateway.gpu2.yml` for the knobs.

The FIFO requirement is by design, not a limitation: under lpt_packing the gateway is the single
fairness authority (concentration plus the per-worker share cap), so a worker-level fair
scheduler would fight the placement by throttling exactly the models the gateway concentrated for
batching. Keep the worker `fair` discipline for deployments without an upstream placement
authority: a single-GPU worker, or a multi-GPU fleet served round-robin.

## Configuring

Edit `cluster.yaml`:

- `workers:` — one entry per GPU (`name`, `gpu`). The name is the routing identity and the
  compose service name; the listen port is `base_port + index`.
- `models:` — assign each model to one or more workers. A model on multiple workers is
  replicated and load-balanced round-robin by the gateway.
- `global:` — defaults merged into every worker; a worker entry may override any block.
- `gateway:` — gateway listen addresses, worker-client request timeout, message limits, logging.

To add a GPU: add a `workers:` entry and a matching `models:` assignment, then add a matching
service in `docker-compose.yml` (copy a worker service and set `REACTANT_WORKER_NAME`).

GPU selection: every worker container reserves all GPUs (the shared `x-worker` deploy
template), and each worker sets `CUDA_VISIBLE_DEVICES` to a single host GPU. CUDA renumbers
that GPU to index 0, so each worker uses device ordinal 0, the default. When you add a worker,
give its service the next host GPU id in `CUDA_VISIBLE_DEVICES` (e.g. `"2"`). The cluster
file's optional `gpu:` field is only for bare-metal deployments that address a specific ordinal
among several visible GPUs; it is omitted here.

## Single-GPU soak test

`docker-compose.gpu2.yml` brings up the whole stack on one GPU to exercise inference with dummy
data and watch for memory leaks, races, and instability: one worker on GPU 2
(`CUDA_VISIBLE_DEVICES=2`), the gateway, and a `loadgen` service that drives sustained concurrent
requests. It uses on-demand weight caching so every bundle need not be GPU-resident at once.

Files:

- `gen_cluster_gpu2.sh` — scans the model repo and writes `cluster.gpu2.yaml` with one worker and
  a `models:` map routing every bundle to it (the gateway needs the map; the worker loads the same
  set). Re-run when the bundle set changes.
- `Dockerfile.loadgen` / `entrypoint.loadgen.sh` / `loadgen/loadgen.jl` — a light, Reactant-free
  load generator built from `ReactantServerClient`. It reads each manifest (`manifest_io_spec`) to
  synthesize correctly shaped zero inputs, then fires concurrent inferences at the gateway.
- `monitor_gpu2.sh` — host-side CSV logger (nvidia-smi for GPU 2 plus docker stats for the worker)
  for leak detection; GPU memory and worker RSS should plateau, not climb.

Prerequisites: built images (`make worker gateway loadgen`), checked-out submodules, and the
NVIDIA container runtime configured for Docker
(`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

Run:

```
make gpu2-up                      # gen_cluster_gpu2.sh then docker compose up
docker/monitor_gpu2.sh &          # optional: log GPU/RSS to soak_monitor.csv
make gpu2-down                    # tear down
```

The worker compiles and warms up every model's executables before it serves, so first startup is
slow (potentially hours for all 85 bundles). The worker healthcheck `start_period` is set high to
cover this. For a quick check, set `LOADGEN_MODELS` to a couple of bundle names in the compose file
and mount only those bundles.

Load parameters are the `loadgen` service's `LOADGEN_*` environment variables: `LOADGEN_TRANSPORT`
(`tcp`, `shm`, or `mixed`; `shm` exercises shared-memory register/unregister), `LOADGEN_CONCURRENCY`,
`LOADGEN_DURATION_SECONDS`, `LOADGEN_REPORT_SECONDS`, and `LOADGEN_MODELS`. The generator's weight
cache budget is `WEIGHT_CACHE_BYTES` (default 24 GiB). The loadgen prints rolling throughput,
latency, error counts, and scraped gateway metrics, and exits nonzero if any request errored.
