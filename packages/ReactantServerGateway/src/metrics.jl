# Gateway metrics, backed by Prometheus.jl. The collectors below preserve the exact metric
# names, labels, help text, and histogram buckets the gateway has always exported; only the
# implementation changed from a hand-rolled text exposition to the vendored Prometheus.jl
# (which now allows the project's HTTP 2.x fork via its patched compat). The public mutator API
# (inc_requests!/observe_*/set_*) is unchanged, so the call sites in server.jl/health.jl are
# untouched. Prometheus collectors are atomically thread-safe, so no external lock is needed.

# ExponentialBuckets(0.001, 2, 14); Prometheus.jl appends the +Inf bucket.
const _LE_BUCKETS = Float64[0.001 * 2.0^k for k in 0:13]

struct GatewayMetrics
    registry::Prometheus.CollectorRegistry
    requests_total::Prometheus.Family{Prometheus.Counter}
    request_latency::Prometheus.Family{Prometheus.Histogram}
    worker_latency::Prometheus.Family{Prometheus.Histogram}
    routing_table_size::Prometheus.Gauge
    worker_ready::Prometheus.Family{Prometheus.Gauge}
    placement_weight::Prometheus.Family{Prometheus.Gauge}
    model_utilization::Prometheus.Family{Prometheus.Gauge}
end

function GatewayMetrics()
    reg = Prometheus.CollectorRegistry()
    requests_total = Prometheus.Family{Prometheus.Counter}(
        "gateway_requests_total",
        "Count of gateway RPCs by method, model, and gRPC status code.",
        (:rpc, :model, :status); registry = reg)
    request_latency = Prometheus.Family{Prometheus.Histogram}(
        "gateway_request_latency_seconds", "Gateway-internal latency.",
        (:rpc, :model); buckets = _LE_BUCKETS, registry = reg)
    worker_latency = Prometheus.Family{Prometheus.Histogram}(
        "gateway_worker_latency_seconds", "Latency of the worker gRPC call.",
        (:rpc, :worker); buckets = _LE_BUCKETS, registry = reg)
    routing_table_size = Prometheus.Gauge(
        "gateway_routing_table_size", "Number of known models in the routing table.";
        registry = reg)
    worker_ready = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_ready",
        "1 if the worker reported ServerReady on the most recent health probe, else 0.",
        (:worker,); registry = reg)
    placement_weight = Prometheus.Family{Prometheus.Gauge}(
        "gateway_placement_weight",
        "LPT-packing sampling weight of a model on a worker (0 when unplaced).",
        (:model, :worker); registry = reg)
    model_utilization = Prometheus.Family{Prometheus.Gauge}(
        "gateway_model_utilization",
        "Estimated per-model expected utilization (arrival rate x compute cost, GPU-seconds/second).",
        (:model,); registry = reg)
    return GatewayMetrics(reg, requests_total, request_latency, worker_latency,
        routing_table_size, worker_ready, placement_weight, model_utilization)
end

inc_requests!(m::GatewayMetrics, rpc, model, status) =
    Prometheus.inc(Prometheus.labels(m.requests_total, (String(rpc), String(model), String(status))))

observe_request!(m::GatewayMetrics, rpc, model, secs) =
    Prometheus.observe(Prometheus.labels(m.request_latency, (String(rpc), String(model))), secs)

observe_worker!(m::GatewayMetrics, rpc, worker, secs) =
    Prometheus.observe(Prometheus.labels(m.worker_latency, (String(rpc), String(worker))), secs)

set_routing_size!(m::GatewayMetrics, n) = Prometheus.set(m.routing_table_size, Float64(n))

set_worker_ready!(m::GatewayMetrics, worker, ready::Bool) =
    Prometheus.set(Prometheus.labels(m.worker_ready, (String(worker),)), ready ? 1.0 : 0.0)

set_placement_weight!(m::GatewayMetrics, model, worker, w) =
    Prometheus.set(Prometheus.labels(m.placement_weight, (String(model), String(worker))), Float64(w))

set_model_utilization!(m::GatewayMetrics, model, u) =
    Prometheus.set(Prometheus.labels(m.model_utilization, (String(model),)), Float64(u))

"""
    expose(io, metrics)

Write all collectors to `io` in the Prometheus text exposition format.
"""
expose(io::IO, m::GatewayMetrics) = Prometheus.expose(io, m.registry)

# --- Admin HTTP server ------------------------------------------------------------------------

# Exposes /metrics, /healthz, and /readyz on a separate HTTP/1.1 listener. /readyz reports 200
# once at least one worker has reported ServerReady.
mutable struct AdminServer
    ready::Threads.Atomic{Bool}
    metrics::GatewayMetrics
    server::Any
end

set_ready!(a::AdminServer, v::Bool) = (a.ready[] = v; nothing)

function start_admin(metrics::GatewayMetrics, addr::AbstractString)
    host, port = _split_hostport(addr)
    ready = Threads.Atomic{Bool}(false)
    handler = function (req)
        target = req.target
        if target == "/metrics" || startswith(target, "/metrics?")
            io = IOBuffer()
            expose(io, metrics)
            return HTTP.Response(200, ["Content-Type" => Prometheus.CONTENT_TYPE_LATEST];
                                 body = String(take!(io)))
        elseif target == "/healthz"
            return HTTP.Response(200; body = "ok")
        elseif target == "/readyz"
            return ready[] ? HTTP.Response(200; body = "ok") : HTTP.Response(503; body = "not ready")
        else
            return HTTP.Response(404; body = "not found")
        end
    end
    server = HTTP.serve!(handler, host, port)
    @info "admin: listening" addr = addr
    return AdminServer(ready, metrics, server)
end
