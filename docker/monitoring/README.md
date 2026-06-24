# ReactantServer monitoring stack (Grafana + Prometheus)

A self-contained Grafana + Prometheus stack that scrapes a running ReactantServer node and ships a
five-dashboard suite. It runs as its own compose project on the **same external Docker network** as
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
discover models dynamically via Grafana template variables (`$worker`, `$gpu`, `$model`).

## Dashboards

1. **Fleet Overview** — RED top-line: request rate, error rate, in-flight/shed, latency
   percentiles, worker-readiness state timeline, per-GPU memory saturation. "Is it OK right now."
2. **Latency & Throughput** — request-latency **heatmap** (the full distribution, not just p99),
   percentile lines, throughput by gRPC status, **coalescing factor** (rows per dispatch = effective
   batch size) by GPU, queue-wait by worker, top-N models by request rate.
3. **GPU Memory Anatomy** — per-GPU stacked **Live occupancy** (resident weights + transient + free
   = pool limit) and **Budget plan** (pinned + on-demand budget + scratch reserve + wiggle), peak
   vs limit, weight-cache load/evict churn, on-demand budget utilization, resident model count.
4. **Scheduling & Placement** — lpt_packing: models placed per worker, in-flight load balance,
   gateway-to-worker call p99, top models by utilization, the model->worker placement table, and
   worker metrics-scrape health.
5. **Per-Model Drilldown** (`$model`) — one model's rate/errors, handler-latency heatmap, queue
   depth & wait, its coalescing, residency-by-worker timeline, and placement.

## A label-space note (important when extending)

`worker_*` series carry `worker="worker0".."worker3"` plus `gpu="0".."3"`. The gateway's own
`gateway_worker_*` / `gateway_placement_weight` / `gateway_replica_outstanding` series carry
`worker="<host:port>"` (and `gateway_worker_metrics_up` uses `endpoint=`). **These two `worker`
label spaces do not join.** Per-worker/GPU panels are built on the `worker_*` series (`$worker` /
`$gpu`); gateway worker-call latency and placement live in their own panels keyed by the host:port
label. Cross-correlating the two needs an explicit index mapping (worker_i <-> base_port + i).

## Files

```
docker-compose.monitoring.yml      prometheus + grafana, joins external network ${REACTANTSERVER_NETWORK}
prometheus.yml                     single scrape job -> reactantserver:8002 (15s, 15d retention)
grafana/provisioning/datasources/  Prometheus datasource (uid: prometheus)
grafana/provisioning/dashboards/   file provider -> /var/lib/grafana/dashboards
grafana/dashboards/*.json          the five dashboards (datasource referenced by uid)
```
