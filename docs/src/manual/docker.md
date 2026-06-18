# Docker Deployment

The repository ships a unified `reactantserver` image whose default entry point is the node
supervisor ([`ReactantServerNode.supervise`](@ref ReactantServerNode.supervise)). It detects
every GPU granted to the container, spawns one single-GPU worker subprocess per device, and
multiplexes the children's logs onto the container's stdout with `[worker0]` / `[gateway]` line
prefixes, restarting children that die. With two or more workers it also runs an embedded
gateway on the public ports; with a single worker it skips the gateway entirely and binds that
worker to the public ports directly, since a lone worker already serves the full KServe V2 API.
Either way the external interface is the same. A multi-GPU deployment is therefore a single
container with no per-GPU configuration:

```
docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
  -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
```

Clients connect to the KServe V2 gRPC endpoint on `localhost:8001`; health and metrics are on
`localhost:8002` (`/readyz`, `/healthz`, `/metrics`). These match NVIDIA Triton's gRPC (8001)
and metrics (8002) ports; the server is gRPC only, so Triton's HTTP port 8000 is unused. Adding
a GPU to the host changes nothing in the configuration: the supervisor sees one more device and
spawns one more worker.

## Files

- `docker/Dockerfile` — the `reactantserver` image (`julia:1.12.5-trixie`). It copies the whole
  `packages/` tree and builds the workspace root (the shared `Manifest.toml` pins the
  Reactant/HTTP forks). The default entry point is the supervisor; `entrypoint.worker.sh` stays
  in the image as the single-worker escape hatch.
- `docker/entrypoint.node.sh` — the default entry point; launches the supervisor under
  `julia --handle-signals=no` (so its own SIGTERM/SIGINT handler runs) via `tini`.
- `docker/entrypoint.worker.sh` — launches a single `ReactantServer.serve` worker named by
  `REACTANT_WORKER_NAME`; the single-worker escape hatch, not the default.
- `docker/healthcheck.node.sh` — role-aware container healthcheck: curls the gateway's `/readyz`
  in the `all`/`gateway` roles (for a single worker, the worker's own `/readyz`), falls back to
  the Julia worker probe for `workers`.
- `docker/healthcheck.worker.jl` — lightweight Julia readiness probe (imports only gRPCClient
  and YAML, never Reactant). With `REACTANT_WORKER_NAME` set it probes that worker; unset, it
  probes every worker in the node file and passes when at least one is ready.
- `docker/node.default.yaml` — the zero-config node file baked into the image at
  `/etc/reactantserver/node.yaml` (`gpus: auto`, no `workers:` list).
- `docker/node.yaml` — the fully commented node-file template to mount over the baked default
  (see [Node Configuration](node_config.md)).
- `docker/gateway.yml` — optional config for the embedded gateway (scheduling knobs); the
  supervisor synthesizes the worker endpoints, so the gateway needs no file by default.
- `docker-compose.yml` — the single-service stack (equivalent to the `docker run` above).

## Prerequisites

1. Populate the vendored submodules:
   ```
   git submodule update --init --recursive
   ```
   This fetches `lib/Reactant.jl`, `lib/gRPCServer.jl`, `lib/gRPCClient.jl`, and `lib/HTTP.jl`.
2. Install the NVIDIA Container Toolkit on the host (for GPU access).
3. Have a model bundle repository on the host. Each immediate subdirectory with a
   `manifest.yaml` is a bundle; its directory name is the model name (see
   [Bundles & model.jl](bundles.md)).

## Build and run

```
make image        # or: docker compose build
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

Every model compiles (Reactant → device executable) before the gRPC plane accepts traffic, so
first startup is slow and the image healthcheck's `start_period` is generous (300s); raise it
for large model sets. Logs from all workers and the gateway appear on the container's stdout,
one line each, prefixed `[worker0]`, `[worker1]`, …, `[gateway]`, `[supervisor]`.

## Configuring

Zero configuration is the default: the baked node file sets `gpus: auto` and the supervisor
synthesizes `worker0..workerN-1`, one per detected device. To customize, mount a node file over
`/etc/reactantserver/node.yaml` (see [Node Configuration](node_config.md) for the full
surface):

- `gpus:` — `auto` (default), an integer count, or an explicit device list (ordinals or GPU
  UUIDs). The `REACTANT_GPUS` environment variable overrides this key.
- `workers:` — optional. When present it wins over auto-detection: device *i* goes to worker
  *i* (or a worker's `gpu:` key picks a specific visible device).
- `models:` — optional per-model pinning to device memory on named workers.
- `global:` — defaults merged into every worker (runtime, scheduler, cache_dir, endpoints).

Supervisor environment variables: `REACTANT_GPUS` (count or device list; `0` for a CPU node),
`REACTANT_CPU_WORKERS` (workers on a CPU node, default 1), `REACTANT_WORKER_THREADS` (compute
threads per worker; default is the host's share, `min(CPU_THREADS ÷ workers, 16)`, see
[Scaling to Multiple GPUs](scaling.md)), `REACTANT_ROLE` (below),
`REACTANT_SUPERVISOR_MAX_RESTARTS` (consecutive crash budget per child before the node exits 1;
default unlimited, with the healthcheck reporting unready instead), `REACTANT_NODE_FILE`, and
`REACTANT_GATEWAY_FILE`. The supervisor writes the materialized node file (with the synthesized
`workers:` list) to `/run/reactantserver/node.yaml` for inspection; children and the healthcheck
read that file.

## Roles

`REACTANT_ROLE` selects what the supervisor runs; the default `all` (workers plus the embedded
gateway, on one host) is what these guides cover. The `workers` and `gateway` roles exist to
split a deployment across machines (GPU nodes behind a separate gateway host), but multi-node is
not a shipped example here. See the [Multi-GPU Gateway](multi_gpu_gateway.md) page for the
gateway's behavior.

## Health status

The container's healthcheck is `healthcheck.node.sh`, which dispatches on `REACTANT_ROLE`. In the
`all` / `gateway` roles it curls `/readyz` on the metrics port (the embedded gateway's, or, for a
single-worker node, the worker's own); both report ready once the process is up and at least one
worker has reported `ServerReady`. In the `workers` role it runs the Julia probe against every
worker in the materialized node file. "At least one worker ready" is
the right liveness signal for a multi-GPU container: one failed GPU should not get a container
serving the others killed. Because model compilation runs before the gRPC plane accepts traffic,
`start_period` is generous (300s by default); raise it for large model sets.

Per-worker readiness remains visible as the `gateway_worker_ready{worker="..."}` Prometheus
metric, and the supervisor logs each child's crash and restart. Compose health reflects process
and serving readiness, not raw GPU hardware state, which the NVIDIA tooling exposes on the host.

## Metrics

One scrape on `8002` covers everything. With multiple workers, the embedded gateway's `/metrics`
serves its own `gateway_*` series and fans out to every worker's metrics endpoint, merging the
results into one exposition; with a single worker (no gateway), `8002` is that worker's own
`/metrics` directly. Either way, each worker tags all of its series (`worker_*` plus
`process_*`/`julia_gc_*`) with `worker` and `gpu` labels itself, where `gpu` is the physical
device behind the worker's `CUDA_VISIBLE_DEVICES`, so no scrape-config relabeling is needed:

```yaml
scrape_configs:
  - job_name: reactantserver
    static_configs:
      - targets: ['node:8002']
```

Per-GPU and per-worker views fall out of the labels, e.g.
`sum by (gpu) (rate(worker_dispatch_total[1m]))`. Per-endpoint scrape health is reported as
`gateway_worker_metrics_up{endpoint=...}`.

## Single-GPU soak test

`docker-compose.gpu2.yml` runs the same supervised container on one GPU
(`CUDA_VISIBLE_DEVICES=2`, so one worker and no gateway, serving 8001/8002 directly) plus a
`loadgen` service that drives sustained concurrent requests, to exercise the full serving path
and watch for memory leaks. It pins a persistent XLA compile-cache volume so warm restarts skip
recompilation, and `docker/monitor_gpu2.sh` logs GPU memory and container RSS to a CSV for leak
detection. Because it runs the production single-GPU container, the soak tests exactly what users
deploy.

## Two-GPU lpt_packing soak test

`docker-compose.gpu23.yml` is the multi-GPU counterpart: the same supervised container across two
GPUs (`CUDA_VISIBLE_DEVICES=2,3`, so two workers behind the embedded gateway) with the gateway in
`lpt_packing` mode. It mounts `docker/node.gpu23.yaml`, which sets `scheduler.discipline: fifo`
(required by `lpt_packing`), and enables packing via `REACTANT_GATEWAY_SCHEDULING_*` environment.
Use it to exercise the gateway's placement and coalescing-aware routing on the full serving path,
not just a lone worker.

The `loadgen` report line includes the fleet weight-cache `loads=`/`evicts=` totals with
per-window deltas (scraped from the aggregated `/metrics`). A steadily rising `evicts` means the
model set does not fit resident and the workers are thrashing weights (host-to-device reloads,
which cost CPU) — a different problem from CPU oversubscription, addressed by more
`weight_cache_bytes`, pinning hot models, or more GPUs.

## Security

ReactantServer is designed to run on a trusted network behind your own perimeter. Be aware of the
following before exposing any endpoint:

- All gRPC traffic (worker and gateway) is cleartext h2c. TLS settings are parsed by the gateway
  config but not yet enforced; a configured cert triggers a startup warning.
- There is no authentication or authorization on the KServe data plane, the worker control-plane
  RPCs (residency and policy), or the Prometheus metrics listener (which binds `0.0.0.0:8002` by
  default).
- Model bundles are trusted input: a bundle's optional `model.jl` executes arbitrary Julia in the
  server process. Only serve bundles you built or audited.
- POSIX shared memory is a local trust boundary. Client-registered regions and the optional
  node-shared host-weight store live in `/dev/shm`; the shared weight regions default to mode `666`
  (world-writable) for cross-container friction-free sharing. Set
  `runtime.shared_host_weights_mode: "660"` on production or multi-user systems.
