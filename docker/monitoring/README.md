# ReactantServer monitoring stack (Grafana + Prometheus)

A self-contained Grafana + Prometheus stack that scrapes a running ReactantServer node and ships a
six-dashboard suite. It runs as its own compose project on the **same external Docker network** as
the server, so it reaches the gateway's metrics by container name and has an independent lifecycle
(restart the server without touching Grafana, and vice versa).

## Run it

The shared network is created by the server deploy (`deploy.netai02.sh`); bring the server up first.

```bash
sudo ./private/deploy/deploy.monitoring.sh up        # start (joins the netai02 network)
sudo ./private/deploy/deploy.monitoring.sh logs -f
sudo ./private/deploy/deploy.monitoring.sh ps
sudo ./private/deploy/deploy.monitoring.sh down       # leaves the server stack + network in place
```

- **Grafana**: `http://<host>:3000` — anonymous viewing is on; log in `admin` / `admin` to edit
  (override with `GRAFANA_ADMIN_PASSWORD`). Lands on Fleet Overview.
- **Prometheus**: `http://<host>:9090` — check `Status -> Targets` shows `reactantserver:8002` UP.

Environment overrides: `REACTANTSERVER_NETWORK` (network to join, default `netai02`),
`PROJECT` (compose project, default `netai02-mon`), `GRAFANA_ADMIN_PASSWORD`.

## How it scrapes

One target is enough. The embedded gateway's `:8002/metrics` serves its own `gateway_*` series and
fans out to every worker's metrics endpoint, merging them into a single exposition; each worker
self-tags its series with `worker` and `gpu` labels, so a single scrape of `reactantserver:8002`
covers the whole fleet (`prometheus.yml`). No model names are configured anywhere, dashboards
discover models dynamically via Grafana template variables (`$worker`, `$model`).

## Dashboards

1. **Fleet Overview** — RED top-line: request rate, error rate, in-flight/shed, latency
   percentiles, worker-readiness state timeline, per-worker device-memory saturation. "Is it OK right now."
2. **Latency & Throughput** — request-latency **heatmap** (the full distribution, not just p99),
   percentile lines, throughput by gRPC status, **coalescing factor** (rows per dispatch = effective
   batch size) by worker, queue-wait by worker, top-N models by request rate.
3. **Device Memory Anatomy** — per-worker stacked **Live occupancy** (resident weights + transient + free
   = pool limit) and **Budget plan** (pinned + on-demand budget + scratch reserve + wiggle), peak
   vs limit, weight-cache load/evict churn, out-of-pool driver memory, on-demand budget utilization, resident model count.
4. **Scheduling & Placement** — lpt_packing: models placed per worker, in-flight load balance,
   gateway-to-worker call p99, top models by utilization, the model->worker placement table, and
   worker metrics-scrape health.
5. **Per-Model Drilldown** (`$model`) — one model's rate/errors, handler-latency heatmap, queue
   depth & wait, its coalescing, residency-by-worker timeline, and placement.
6. **Coalescing & Batching** — every model's **coalescing factor** (requests merged per dispatch =
   effective batch size) vs its compiled **max batch**, with a **fill** column (factor / max),
   rank-ordered lowest-fill first. Surfaces models receiving traffic but not batching well, i.e.
   where raising the batch window or arrival concurrency could help. Covers meta sub-models too.

## A label note (important when extending)

Everything is keyed by a single `worker` label (`worker0..N`); each worker owns exactly one device,
so there is no separate `gpu` label in the dashboards. `worker_*` series self-tag `worker` (and still
carry a `gpu` index, unused here). The gateway's own `gateway_worker_*` / `gateway_placement_weight`
/ `gateway_replica_outstanding` / `gateway_worker_metrics_up` series **also** carry `worker="worker0..N"`
now: the supervisor threads each worker's name to the embedded gateway (`REACTANT_GATEWAY_WORKER_NAMES`),
which maps its endpoint urls to those names. So gateway-side and worker-side panels share one label
space and join directly (no host:port, no index mapping). A **standalone** gateway given bare
`endpoints:` and no names falls back to labelling by url.

Terminology: the hardware is called "device" (GPU/TPU-agnostic) in titles; the identity is the
`worker`. So e.g. "device s/s" is device-seconds per second for a `worker`.

## Files

```
docker-compose.monitoring.yml      prometheus + grafana, joins external network ${REACTANTSERVER_NETWORK}
prometheus.yml                     single scrape job -> reactantserver:8002 (15s, 15d retention)
grafana/provisioning/datasources/  Prometheus datasource (uid: prometheus)
grafana/provisioning/dashboards/   file provider -> /var/lib/grafana/dashboards
grafana/dashboards/*.json          the five dashboards (datasource referenced by uid)
```
