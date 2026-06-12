# Worker Prometheus metrics: a small HTTP exposition endpoint mirroring the gateway's admin server
# (ReactantServerGateway/src/metrics.jl). Two sources feed one CollectorRegistry:
#
#   1. a pull-based collector that snapshots scheduler / weight-cache / device state on each scrape,
#      emitting exactly the currently-loaded models (no parallel mutator state, no stale labels);
#   2. hot-path Family counters incremented by the ModelInfer handler.
#
# Reuses the existing snapshot functions (scheduler_metrics, weight_cache_metrics,
# resident_weight_bytes, device_memory_stats). Prometheus.jl collectors are atomically thread-safe.

import HTTP
import Prometheus

# Mirror the gateway's latency buckets: ExponentialBuckets(0.001, 2, 14).
const _WM_BUCKETS = Float64[0.001 * 2.0^k for k in 0:13]

# --- Pull collector ---------------------------------------------------------------------------

# Reads live worker state at scrape time. Holds only references; no metric state of its own.
struct WorkerSnapshotCollector <: Prometheus.Collector
    sched::Scheduler
    backend::AbstractBackend
    pool::MemoryPool
    cfg::ServerConfig
    worker_name::String
end

Prometheus.metric_names(::WorkerSnapshotCollector) = (
    "worker_dispatch_total", "worker_compute_seconds_total", "worker_queue_depth",
    "worker_queue_wait_seconds", "worker_model_resident", "worker_model_pinned",
    "worker_model_weight_bytes", "worker_weight_cache_resident_bytes",
    "worker_weight_cache_max_bytes", "worker_weight_loads_total", "worker_weight_evicts_total",
    "worker_weight_load_seconds_total", "worker_device_memory_in_use_bytes",
    "worker_device_memory_limit_bytes", "worker_device_memory_free_bytes",
    "worker_models_loaded", "worker_models_resident", "worker_resident_weight_bytes",
    "worker_info",
)

_scalar(name, type, help, v) =
    Prometheus.Metric(type, name, help, Prometheus.Sample(nothing, nothing, nothing, Float64(v)))

function Prometheus.collect!(metrics::Vector, c::WorkerSnapshotCollector)
    sm = scheduler_metrics(c.sched)   # Dict: model name => per-model NamedTuple
    model_ln = Prometheus.LabelNames(("model",))
    persamples(f) = Prometheus.Sample[
        Prometheus.Sample(nothing, model_ln, Prometheus.LabelValues((name,)), Float64(f(m)))
        for (name, m) in sm]

    push!(metrics,
        Prometheus.Metric("counter", "worker_dispatch_total", "Total dispatches per model.",
            persamples(m -> m.dispatch_count)),
        Prometheus.Metric("counter", "worker_compute_seconds_total",
            "Total GPU compute time per model (seconds).", persamples(m -> m.total_compute)),
        Prometheus.Metric("gauge", "worker_queue_depth", "Pending requests queued per model.",
            persamples(m -> m.queue_depth)),
        Prometheus.Metric("gauge", "worker_model_resident",
            "1 if the model's weights are device-resident, else 0.",
            persamples(m -> m.resident ? 1 : 0)),
        Prometheus.Metric("gauge", "worker_model_pinned",
            "1 if the model is device-pinned, else 0.", persamples(m -> m.pinned ? 1 : 0)),
        Prometheus.Metric("gauge", "worker_model_weight_bytes",
            "Per-model weight footprint (bytes).", persamples(m -> m.weight_nbytes)),
    )

    # Queue-wait quantiles, labelled by model and quantile.
    qln = Prometheus.LabelNames(("model", "quantile"))
    wait = Prometheus.Sample[]
    for (name, m) in sm
        push!(wait, Prometheus.Sample(nothing, qln, Prometheus.LabelValues((name, "0.5")), Float64(m.wait_p50)))
        push!(wait, Prometheus.Sample(nothing, qln, Prometheus.LabelValues((name, "0.99")), Float64(m.wait_p99)))
    end
    push!(metrics, Prometheus.Metric("gauge", "worker_queue_wait_seconds",
        "Queue-wait latency per model (seconds).", wait))

    # Server-level counts and the server's own device-resident-weight accounting.
    rw = resident_weight_bytes(c.sched.registry)
    push!(metrics,
        _scalar("worker_models_loaded", "gauge", "Number of loaded models.", length(c.sched.registry.by_name)),
        _scalar("worker_models_resident", "gauge", "Number of device-resident models.", rw.count),
        _scalar("worker_resident_weight_bytes", "gauge",
            "Total device-resident weight footprint (bytes).", rw.bytes),
    )

    # On-demand weight cache (only when enabled).
    wc = weight_cache_metrics(c.sched)
    if wc !== nothing
        push!(metrics,
            _scalar("worker_weight_cache_resident_bytes", "gauge",
                "On-demand weight-cache resident bytes.", wc.resident_bytes),
            _scalar("worker_weight_cache_max_bytes", "gauge",
                "On-demand weight-cache byte budget.", wc.max_bytes),
            _scalar("worker_weight_loads_total", "counter", "On-demand weight loads.", wc.loads),
            _scalar("worker_weight_evicts_total", "counter", "On-demand weight evictions.", wc.evicts),
            _scalar("worker_weight_load_seconds_total", "counter",
                "Cumulative time spent loading weights (seconds).", wc.load_seconds),
        )
    end

    # Device memory, when the backend can report it (absent on CPU / MockBackend).
    dm = device_memory_stats(c.backend, c.pool)
    if dm !== nothing
        push!(metrics,
            _scalar("worker_device_memory_in_use_bytes", "gauge", "Device bytes in use.", dm.in_use),
            _scalar("worker_device_memory_limit_bytes", "gauge",
                "Device memory pool limit (bytes).", dm.limit),
            _scalar("worker_device_memory_free_bytes", "gauge",
                "Device bytes free to allocate.", dm.free),
        )
    end

    # Identity + config, for grouping. The physical-GPU label is applied in the Prometheus scrape
    # config (the worker cannot know its physical GPU under per-container CUDA_VISIBLE_DEVICES).
    info_ln = Prometheus.LabelNames(("worker", "device_ordinal", "control_mode", "discipline", "residency_mode"))
    info_lv = Prometheus.LabelValues((
        c.worker_name,
        string(c.cfg.runtime.device_ordinal),
        lowercase(string(c.cfg.model_control_mode)),
        lowercase(string(c.cfg.scheduler.discipline)),
        lowercase(string(c.cfg.runtime.residency_mode)),
    ))
    push!(metrics, Prometheus.Metric("gauge", "worker_info",
        "Worker identity and configuration (value is always 1).",
        Prometheus.Sample[Prometheus.Sample(nothing, info_ln, info_lv, 1.0)]))
    return metrics
end

# --- Registry + hot-path counters -------------------------------------------------------------

struct WorkerMetrics
    registry::Prometheus.CollectorRegistry
    requests_total::Prometheus.Family{Prometheus.Counter}
    request_latency::Prometheus.Family{Prometheus.Histogram}
end

function WorkerMetrics(sched::Scheduler, backend::AbstractBackend, pool::MemoryPool,
                       cfg::ServerConfig; worker_name::AbstractString="")
    reg = Prometheus.CollectorRegistry()
    requests_total = Prometheus.Family{Prometheus.Counter}(
        "worker_requests_total", "Worker ModelInfer requests by model and gRPC status.",
        (:model, :status); registry = reg)
    request_latency = Prometheus.Family{Prometheus.Histogram}(
        "worker_request_latency_seconds", "Worker ModelInfer handler latency (seconds).",
        (:model,); buckets = _WM_BUCKETS, registry = reg)
    Prometheus.register(reg, WorkerSnapshotCollector(sched, backend, pool, cfg, String(worker_name)))
    # Free process/runtime metrics (julia_gc_*, process_resident_memory_bytes, etc.). Guarded so a
    # platform without /proc cannot break worker startup.
    try
        Prometheus.GCCollector(; registry = reg)
        Prometheus.ProcessCollector(; registry = reg)
    catch err
        @warn "worker metrics: GC/Process collectors unavailable" exception = err
    end
    return WorkerMetrics(reg, requests_total, request_latency)
end

inc_request!(m::WorkerMetrics, model, status) =
    Prometheus.inc(Prometheus.labels(m.requests_total, (String(model), String(status))))

observe_request!(m::WorkerMetrics, model, secs) =
    Prometheus.observe(Prometheus.labels(m.request_latency, (String(model),)), Float64(secs))

# --- HTTP exposition --------------------------------------------------------------------------

"""
    start_worker_metrics(metrics, host, port; ready_fn) -> HTTP server

Serve `/metrics` (Prometheus text exposition), `/healthz`, and `/readyz` (`ready_fn()`) on an
HTTP/1.1 listener. Mirrors the gateway's admin server. Close the returned server to stop it.
"""
function start_worker_metrics(m::WorkerMetrics, host::AbstractString, port::Integer; ready_fn)
    handler = function (req)
        target = req.target
        if target == "/metrics" || startswith(target, "/metrics?")
            io = IOBuffer()
            Prometheus.expose(io, m.registry)
            return HTTP.Response(200, ["Content-Type" => Prometheus.CONTENT_TYPE_LATEST];
                                 body = String(take!(io)))
        elseif target == "/healthz"
            return HTTP.Response(200; body = "ok")
        elseif target == "/readyz"
            return ready_fn() ? HTTP.Response(200; body = "ok") : HTTP.Response(503; body = "not ready")
        else
            return HTTP.Response(404; body = "not found")
        end
    end
    server = HTTP.serve!(handler, host, port)
    @info "worker metrics: listening" host = host port = port
    return server
end
