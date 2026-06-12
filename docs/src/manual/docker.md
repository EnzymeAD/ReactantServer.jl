# Docker Deployment

The repository ships a `docker compose` setup for a multi-GPU deployment: one single-GPU worker
per GPU behind one `reactant-gateway`. All services read a single cluster file, `docker/cluster.yaml`.
The relevant files live under `docker/` and in `docker-compose.yml`.

## Files

- `docker/Dockerfile.worker` — the ReactantServer.jl worker (`julia:1.12.5-trixie`). It copies
  the whole `packages/` tree and builds the workspace root (the shared `Manifest.toml` pins the
  Reactant/HTTP forks); the entrypoint then activates the `ReactantServer` member project.
- `docker/Dockerfile.gateway` — the pure-Julia `reactant-gateway` (`julia:1.12.5-trixie`). Built
  from the `ReactantServerGateway` member alone (just `ReactantServerCore` + the gRPC/HTTP
  forks), so it pulls **no Reactant**.
- `docker/entrypoint.worker.sh` — launches `ReactantServer.serve` (project
  `packages/ReactantServer`) for the worker named by `REACTANT_WORKER_NAME`.
- `docker/entrypoint.gateway.sh` — launches `ReactantServerGateway.serve_gateway` (project
  `packages/ReactantServerGateway`) against the cluster file.
- `docker/healthcheck.worker.jl` — lightweight Julia worker readiness probe (imports only
  gRPCClient and YAML), run against the workspace root project as the worker container's
  healthcheck.
- `docker/cluster.yaml` — the shared cluster config mounted into every service.
- `docker-compose.yml` — the two-worker + gateway stack.

## Prerequisites

1. Populate the vendored submodules:
   ```
   git submodule update --init --recursive
   ```
   This fetches `lib/Reactant.jl`, `lib/gRPCServer.jl`, `lib/gRPCClient.jl`, and `lib/HTTP.jl`.
2. Install the NVIDIA Container Toolkit on the host (for GPU access).
3. Have a model bundle repository on the host. Each immediate subdirectory with a
   `manifest.yaml` is a bundle; its directory name is the model name used in `cluster.yaml` (see
   [Bundles & model.jl](bundles.md)).

## Build and run

```
docker compose build
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

Clients connect to the gateway's KServe V2 gRPC endpoint on `localhost:8001`; health and
metrics are on `localhost:8002` (`/readyz`, `/healthz`, `/metrics`). These match NVIDIA Triton's
gRPC (8001) and metrics (8002) ports; the server is gRPC only, so Triton's HTTP port 8000 is unused.

## Health status

Every service has a Compose healthcheck, so `docker compose ps` shows healthy or unhealthy per
container and the gateway waits for the workers before starting.

- Each worker is checked with `healthcheck.worker.jl`, a lightweight Julia probe baked into the
  worker image that calls the worker's KServe `ServerReady` RPC. A worker is healthy once its
  gRPC plane is up and every assigned model is compiled and resident on its GPU, which is the
  meaningful per-GPU serving signal. The probe imports only gRPCClient and YAML (never
  ReactantServer, so it does not pay the Reactant load), resolves the worker's own port from the
  mounted cluster file plus `REACTANT_WORKER_NAME`, and connects over loopback, so it needs no
  extra configuration. Because model compilation runs before the gRPC plane accepts traffic, the
  worker healthcheck uses a generous `start_period` (300s by default); raise it for large model
  sets.
- The gateway is checked with `curl -fsS http://127.0.0.1:8002/readyz` (curl is installed in the
  gateway image). `/readyz` reports serving capability: the gateway process is up and at least
  one backing worker has reported ready.
- The gateway's `depends_on` uses `condition: service_healthy`, so it starts only after both
  workers pass their healthchecks. Its own readiness then goes green promptly.

Compose health reflects process and serving readiness, not raw GPU hardware state; the latter
is exposed by the NVIDIA tooling on the host. Per-worker readiness is also available as the
`gateway_worker_ready{worker="..."}` Prometheus metric on the gateway's metrics port.

## Configuring

Edit `docker/cluster.yaml` (see [Cluster Configuration](cluster_config.md) for the full surface):

- `workers:` — one entry per GPU (`name`, `gpu`). The name is the routing identity and the
  compose service name; the listen port is `base_port + index`.
- `models:` — assign each model to one or more workers. A model on multiple workers is
  replicated and load-balanced round-robin by the gateway.
- `global:` — defaults merged into every worker; a worker entry may override any block.
- `gateway:` — gateway listen addresses, message limits, logging, TLS.

To add a GPU, add a `workers:` entry and a matching `models:` assignment, then add a matching
service in `docker-compose.yml` (copy a worker service and set `REACTANT_WORKER_NAME`).

## GPU selection

Every worker container reserves all GPUs (the shared `x-worker` deploy template), and each
worker sets `CUDA_VISIBLE_DEVICES` to a single host GPU. CUDA renumbers that GPU to index 0, so
each worker uses device ordinal 0, the default. When you add a worker, give its service the next
host GPU id in `CUDA_VISIBLE_DEVICES` (for example `"2"`). The cluster file's optional `gpu:`
field is only for bare-metal deployments that address a specific ordinal among several visible
GPUs; it is omitted here.
