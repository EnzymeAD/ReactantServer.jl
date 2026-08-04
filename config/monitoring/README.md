# ReactantServer monitoring stack (Grafana + Prometheus)

A self-contained Grafana + Prometheus stack that scrapes a running ReactantServer node and ships a
seven-dashboard suite. It runs as its own compose project with an independent lifecycle (restart the
server without touching Grafana, and vice versa). The node runs natively on the host, so Prometheus
scrapes the host's metrics port rather than a container over a shared network.

## Run it

Bring the node up first (see the Deployment guide), then:

```bash
COMPOSE=config/monitoring/docker-compose.monitoring.yml
docker compose -f "$COMPOSE" up -d
docker compose -f "$COMPOSE" logs -f
docker compose -f "$COMPOSE" ps
docker compose -f "$COMPOSE" down
```

Prometheus reaches the natively-running node at `host.docker.internal:8002`, mapped to the host
gateway by `extra_hosts` in the compose file (Docker 20.10+). If the node listens on a **different
host**, edit the target in `prometheus.yml` to that host's `address:8002` instead.

- **Grafana**: `http://<host>:3000` — anonymous viewing is on; log in `admin` / `admin` to edit
  (override with `GRAFANA_ADMIN_PASSWORD`). Lands on Fleet Overview.
- **Prometheus**: `http://<host>:9090` — check `Status -> Targets` shows the `reactantserver` job UP.

Environment overrides: `GRAFANA_ADMIN_PASSWORD`.

## How it scrapes

One target is enough. The embedded gateway's `:8002/metrics` serves its own `gateway_*` series and
fans out to every worker's metrics endpoint, merging them into a single exposition; each worker
self-tags its series with `worker` and `gpu` labels, so a single scrape of the node's `:8002`
covers the whole fleet (`prometheus.yml`). No model names are configured anywhere, dashboards
discover models dynamically via Grafana template variables (`$worker`, `$model`).

## Dashboards

1. **Fleet Overview** — RED top-line: request rate, error rate, in-flight/shed, latency
   percentiles, worker-readiness state timeline, per-worker device-memory saturation, and the
   per-worker routing load. "Is it OK right now."
2. **Latency & Throughput** — request-latency **heatmap** (the full distribution, not just p99),
   percentile lines, throughput by gRPC status, **coalescing factor** (rows per dispatch = effective
   batch size) by worker, queue-wait by worker, top-N models by request rate.
3. **Device Memory Anatomy** — per-worker stacked **Live occupancy** (resident weights + transient + free
   = pool limit) and **Budget plan** (pinned + on-demand budget + scratch reserve + wiggle), peak
   vs limit, weight-cache load/evict churn, out-of-pool driver memory, on-demand budget utilization, resident model count.
4. **Scheduling & Placement** — lpt_packing: models placed per worker, in-flight load balance,
   gateway-to-worker call p99, top models by utilization, the model->worker placement table, and
   worker metrics-scrape health. `gateway_replica_routed_total` (cumulative requests routed per
   replica) is the series to watch when a replicated model looks like it is only using one GPU;
   `gateway_model_fill_quantum` and `gateway_repacks_total{trigger}` explain why. Note the units
   differ on purpose: `gateway_replica_outstanding` counts in-flight ITEMS (summed batch sizes) so it
   is comparable to the quantum, while `gateway_replica_routed_total` counts routing decisions.
   `gateway_worker_inflight_work` is the third unit and the one that decides routing: the per-worker
   load `fill_least` (or the `least_outstanding` mode) compares, denominated by `scheduling.work_basis`
   and labelled with it, so the dashboard states which basis is in force rather than trusting the
   config file, which a runtime `set-policy` can have superseded. The spread panel (`max - min`) is
   what `fill_least` exists to shrink; read a persistently high spread next to the placement table,
   since it usually means the replicas cannot absorb the imbalance rather than that the policy is
   failing.
5. **Per-Model Drilldown** (`$model`) — one model's rate/errors, handler-latency heatmap, queue
   depth & wait, its coalescing, residency-by-worker timeline, and placement.
6. **Coalescing & Batching** — every model's **coalescing factor** (requests merged per dispatch =
   effective batch size) vs its compiled **max batch**, with a **fill** column (factor / max),
   rank-ordered lowest-fill first. Surfaces models receiving traffic but not batching well, i.e.
   where raising the batch window or arrival concurrency could help. Covers meta sub-models too.
7. **GPU Work Balance** — is the fleet's work actually spread across the devices? Built entirely on
   `rate(worker_compute_seconds_total)` per worker, which is device-seconds per wall-second and so
   reads directly as a utilization between 0 and 1. The headline is **work spread**, the coefficient
   of variation of that rate across devices: 0% means every device is doing the same amount of work,
   green under 10%, red over 25%. It is scale free (comparable across load levels) and stays finite
   when a device idles, which busiest/idlest does not. Alongside it: each device's **share of fleet
   work** as a bar gauge (even is 1/N), per-device utilization over time against the fleet mean, the
   spread as a time series (a spike during a burst is normal, a flat elevated line is a standing
   imbalance that costs headroom as utilization climbs), **planned vs actual share** (the packer's
   intent from `gateway_placement_weight * gateway_model_utilization` against what the devices really
   did, which separates "the plan is skewed" from "routing is not following the plan"), and two
   tables: per-device totals with deviation from even, and the top model-on-device pairs so an
   imbalance can be attributed to a model.

   Deliberately carries no `$worker` variable: filtering out a device would distort every share on
   the page. Note the utilization measures time inside `run_model` only, excluding queue wait and
   host-side pre/postprocess, so it answers "what did the device do", not "what did the request cost".
   Calibration from a real 4x T4 node: a routing policy leaving one device with 1.9x another's work
   read 26% spread, and switching to even rotation read 2%.

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
grafana/dashboards/*.json          the seven dashboards (datasource referenced by uid)
```
