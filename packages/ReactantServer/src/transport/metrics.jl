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
    "worker_model_weight_bytes", "worker_requests_served_total", "worker_rows_served_total",
    "worker_model_max_batch_size",
    "worker_weight_cache_resident_bytes",
    "worker_weight_cache_max_bytes", "worker_weight_cache_pinned_bytes",
    "worker_weight_cache_max_scratch_bytes", "worker_weight_pool_bytes",
    "worker_weight_loads_total", "worker_weight_evicts_total",
    "worker_weight_load_seconds_total", "worker_device_memory_in_use_bytes",
    "worker_device_memory_limit_bytes", "worker_device_memory_free_bytes",
    "worker_device_memory_peak_in_use_bytes", "worker_device_memory_pool_bytes",
    "worker_device_memory_process_used_bytes", "worker_device_memory_out_of_pool_bytes",
    "worker_models_loaded", "worker_models_resident", "worker_resident_weight_bytes",
    "worker_info",
)

_scalar(name, type, help, v) =
    Prometheus.Metric(type, name, help, Prometheus.Sample(nothing, nothing, nothing, Float64(v)))

# Reads the gRPC server's live admission counters at scrape time: in-flight RPCs, the cumulative
# count shed at the concurrency cap, and the configured cap (0 = uncapped). Holds only references.
struct AdmissionCollector <: Prometheus.Collector
    inflight::Threads.Atomic{Int}
    shed::Threads.Atomic{Int}
    max_concurrent::Int
end

Prometheus.metric_names(::AdmissionCollector) =
    ("worker_inflight_requests", "worker_requests_shed_total", "worker_max_concurrent_requests")

function Prometheus.collect!(metrics::Vector, c::AdmissionCollector)
    push!(metrics,
        _scalar("worker_inflight_requests", "gauge",
            "RPCs currently being handled (counted only when the cap is enabled).", c.inflight[]),
        _scalar("worker_requests_shed_total", "counter",
            "RPCs rejected with RESOURCE_EXHAUSTED at the concurrency cap.", c.shed[]),
        _scalar("worker_max_concurrent_requests", "gauge",
            "Configured in-flight RPC cap (0 = uncapped).", c.max_concurrent),
    )
    return metrics
end

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
        Prometheus.Metric("counter", "worker_requests_served_total",
            "Separate requests coalesced into dispatches per model (served/dispatch = server-side "
            * "request merging; stays 1 when each request is dispatched alone).", persamples(m -> m.requests_served)),
        Prometheus.Metric("counter", "worker_rows_served_total",
            "Batch-axis rows processed per model (rows/dispatch = effective batch size; counts "
            * "client-prebatched rows too, so it reveals batching that request-merging misses).",
            persamples(m -> m.rows_served)),
        Prometheus.Metric("gauge", "worker_model_max_batch_size",
            "Largest batch size the worker coalesces this model to (<=1 means non-coalescable).",
            persamples(m -> m.max_batch_size)),
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
                "On-demand weight-cache byte budget (the auto-sized on-demand allotment).", wc.max_bytes),
            _scalar("worker_weight_cache_pinned_bytes", "gauge",
                "Device-pinned weight footprint reserved off the weight pool.", wc.pinned_bytes),
            _scalar("worker_weight_cache_max_scratch_bytes", "gauge",
                "Measured worst-case execution scratch reserved as headroom.", wc.max_scratch),
            _scalar("worker_weight_pool_bytes", "gauge",
                "Arena bytes allotted to all weights (pinned + on-demand).", wc.weight_pool),
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
            _scalar("worker_device_memory_peak_in_use_bytes", "gauge",
                "Peak device bytes in use since startup (allocator high-water mark).", dm.peak_in_use),
            _scalar("worker_device_memory_pool_bytes", "gauge",
                "Bytes the allocator has claimed from the device for its pool.", dm.pool_bytes),
        )
        # Out-of-pool driver memory: what the driver says this process holds (nvidia-smi) minus the
        # BFC arena. The arena is preallocated, so the remainder is the CUDA context + loaded modules
        # + command buffers / CUDA graphs, i.e. the memory that lives OUTSIDE the pool and competes
        # for the headroom between the arena and the card. This is the quantity behind intermittent
        # command-buffer startup OOMs, and the allocator stats above cannot see it.
        proc = process_device_used_bytes()
        if proc !== nothing
            push!(metrics,
                _scalar("worker_device_memory_process_used_bytes", "gauge",
                    "Device bytes this worker process holds per the driver (nvidia-smi): BFC arena + out-of-pool.", proc),
                _scalar("worker_device_memory_out_of_pool_bytes", "gauge",
                    "Driver memory outside the BFC arena (CUDA context + modules + command buffers/graphs); process used minus arena.",
                    max(0, proc - dm.limit)),
            )
        end
    end

    # Identity + config, for grouping. Every exported series additionally carries worker/gpu
    # labels injected at exposition time (see start_worker_metrics), so an aggregated scrape
    # through the gateway stays per-worker attributable without scrape-config relabeling.
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
                       cfg::ServerConfig; worker_name::AbstractString="",
                       inflight::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
                       shed::Threads.Atomic{Int}=Threads.Atomic{Int}(0))
    reg = Prometheus.CollectorRegistry()
    requests_total = Prometheus.Family{Prometheus.Counter}(
        "worker_requests_total", "Worker ModelInfer requests by model and gRPC status.",
        (:model, :status); registry = reg)
    request_latency = Prometheus.Family{Prometheus.Histogram}(
        "worker_request_latency_seconds", "Worker ModelInfer handler latency (seconds).",
        (:model,); buckets = _WM_BUCKETS, registry = reg)
    Prometheus.register(reg, WorkerSnapshotCollector(sched, backend, pool, cfg, String(worker_name)))
    Prometheus.register(reg, AdmissionCollector(inflight, shed, cfg.endpoints.max_concurrent_requests))
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

_escape_label_value(v::AbstractString) =
    replace(String(v), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")

"""
    inject_metric_labels(text, labels) -> String

Rewrite a Prometheus text exposition so every sample line carries the given constant labels
(e.g. `worker="worker0"`, `gpu="1"`). Comment lines (`# HELP` / `# TYPE`) pass through; a label
key already present on a line is left alone (e.g. `worker_info`'s own `worker`). This is how a
worker tags its entire export (including `process_*`/`julia_gc_*`) with its identity, so series
stay attributable when the gateway aggregates many workers into one scrape.
"""
function inject_metric_labels(text::AbstractString, labels::Vector{Pair{String,String}})
    isempty(labels) && return String(text)
    out = IOBuffer()
    for line in eachline(IOBuffer(text); keep=true)
        stripped = rstrip(line)
        if isempty(stripped) || startswith(stripped, '#')
            write(out, line)
            continue
        end
        brace = findfirst('{', stripped)
        sep = findfirst(' ', stripped)
        if brace !== nothing && (sep === nothing || brace < sep)
            existing = stripped[brace:end]
            add = join(("$k=\"$(_escape_label_value(v))\"" for (k, v) in labels
                        if !occursin("$k=\"", existing)), ",")
            write(out, isempty(add) ? line :
                  stripped[1:brace] * add * "," * stripped[(brace + 1):end] * "\n")
        elseif sep !== nothing
            add = join(("$k=\"$(_escape_label_value(v))\"" for (k, v) in labels), ",")
            write(out, stripped[1:(sep - 1)] * "{" * add * "}" * stripped[sep:end] * "\n")
        else
            write(out, line)
        end
    end
    return String(take!(out))
end

# The worker's physical-GPU identity for the metrics `gpu` label. Under the supervisor (and the
# per-GPU-container layout) CUDA_VISIBLE_DEVICES holds exactly the physical selector; on bare
# metal with several visible devices, pick the token the worker actually addresses.
function _gpu_identity(cfg::ServerConfig, env::AbstractDict=ENV)
    cfg.runtime.backend == CUDA_BACKEND || return ""
    cvd = get(env, "CUDA_VISIBLE_DEVICES", "")
    toks = [strip(t) for t in split(cvd, ',') if !isempty(strip(t))]
    length(toks) == 1 && return String(toks[1])
    n = cfg.runtime.device_ordinal
    length(toks) > 1 && n + 1 <= length(toks) && return String(toks[n + 1])
    return isempty(toks) ? string(n) : ""
end

"""
    start_worker_metrics(metrics, host, port; ready_fn, worker_name="", gpu="") -> HTTP server

Serve `/metrics` (Prometheus text exposition), `/healthz`, and `/readyz` (`ready_fn()`) on an
HTTP/1.1 listener. Mirrors the gateway's admin server. Non-empty `worker_name` / `gpu` are
injected as constant `worker` / `gpu` labels on every exported series. Close the returned
server to stop it.
"""
function start_worker_metrics(m::WorkerMetrics, host::AbstractString, port::Integer; ready_fn,
                              worker_name::AbstractString="", gpu::AbstractString="")
    labels = Pair{String,String}[]
    isempty(worker_name) || push!(labels, "worker" => String(worker_name))
    isempty(gpu) || push!(labels, "gpu" => String(gpu))
    handler = function (req)
        target = req.target
        if target == "/metrics" || startswith(target, "/metrics?")
            io = IOBuffer()
            Prometheus.expose(io, m.registry)
            return HTTP.Response(200, ["Content-Type" => Prometheus.CONTENT_TYPE_LATEST];
                                 body = inject_metric_labels(String(take!(io)), labels))
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
