# Monitoring

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
