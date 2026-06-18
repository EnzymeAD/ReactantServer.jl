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
    model_replicas::Prometheus.Family{Prometheus.Gauge}
    replica_outstanding::Prometheus.Family{Prometheus.Gauge}
    worker_metrics_up::Prometheus.Family{Prometheus.Gauge}
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
    model_replicas = Prometheus.Family{Prometheus.Gauge}(
        "gateway_model_replicas",
        "LPT-packing replica count for a model (number of distinct GPUs hosting it; 0 when unplaced).",
        (:model,); registry = reg)
    replica_outstanding = Prometheus.Family{Prometheus.Gauge}(
        "gateway_replica_outstanding",
        "In-flight requests routed to a model's replica on a worker, sampled at the last repack.",
        (:model, :worker); registry = reg)
    worker_metrics_up = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_metrics_up",
        "1 if the worker's metrics endpoint answered the most recent aggregated scrape, else 0.",
        (:endpoint,); registry = reg)
    return GatewayMetrics(reg, requests_total, request_latency, worker_latency,
        routing_table_size, worker_ready, placement_weight, model_utilization,
        model_replicas, replica_outstanding, worker_metrics_up)
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

set_model_replicas!(m::GatewayMetrics, model, k) =
    Prometheus.set(Prometheus.labels(m.model_replicas, (String(model),)), Float64(k))

set_replica_outstanding!(m::GatewayMetrics, model, worker, n) =
    Prometheus.set(Prometheus.labels(m.replica_outstanding, (String(model), String(worker))), Float64(n))

"""
    expose(io, metrics)

Write all collectors to `io` in the Prometheus text exposition format.
"""
expose(io::IO, m::GatewayMetrics) = Prometheus.expose(io, m.registry)

# --- Worker metrics aggregation -----------------------------------------------------------------

"""
    merge_expositions(texts) -> String

Merge several Prometheus text expositions into one valid exposition: each metric family keeps a
single `# HELP` / `# TYPE` header (the first seen) and the sample lines from every source are
grouped under it. The sources are the gateway's own export plus each worker's; workers tag all
their series with `worker`/`gpu` labels, so samples never collide.
"""
function merge_expositions(texts::Vector{String})
    order = String[]                                  # family emission order
    help = Dict{String,String}()                      # family -> "# HELP ..." line
    type = Dict{String,String}()                      # family -> "# TYPE ..." line
    samples = Dict{String,Vector{String}}()           # family -> sample lines
    current = ""
    family_of(line) = begin
        # Sample lines belong to the family of the preceding header when their name extends it
        # (histogram _bucket/_sum/_count); otherwise the bare metric name is its own family.
        stop = something(findfirst(c -> c == '{' || c == ' ', line), lastindex(line) + 1)
        name = line[1:(stop - 1)]
        (!isempty(current) && (name == current || startswith(name, current * "_"))) ? current : name
    end
    ensure!(fam) = fam in order || push!(order, fam)
    for text in texts
        current = ""
        for line in eachline(IOBuffer(text))
            isempty(strip(line)) && continue
            if startswith(line, "# HELP ")
                current = split(line; limit=4)[3]
                ensure!(current)
                get!(help, current, line)
            elseif startswith(line, "# TYPE ")
                current = split(line; limit=4)[3]
                ensure!(current)
                get!(type, current, line)
            elseif startswith(line, '#')
                continue
            else
                fam = family_of(line)
                ensure!(fam)
                push!(get!(samples, fam, String[]), line)
            end
        end
    end
    io = IOBuffer()
    for fam in order
        haskey(help, fam) && println(io, help[fam])
        haskey(type, fam) && println(io, type[fam])
        for s in get(samples, fam, String[])
            println(io, s)
        end
    end
    return String(take!(io))
end

# Fetch every worker's /metrics concurrently (best effort, bounded by readtimeout) and record
# per-endpoint reachability in gateway_worker_metrics_up.
function _fetch_worker_metrics(m::GatewayMetrics, endpoints::Vector{String}; readtimeout::Int=5)
    bodies = Vector{Union{Nothing,String}}(nothing, length(endpoints))
    @sync for (i, ep) in enumerate(endpoints)
        @async begin
            ok = false
            try
                resp = HTTP.get("http://$ep/metrics"; retry = false, status_exception = true,
                                connect_timeout = readtimeout, request_timeout = readtimeout)
                bodies[i] = String(resp.body)
                ok = true
            catch e
                @debug "worker metrics scrape failed" endpoint = ep exception = e
            end
            Prometheus.set(Prometheus.labels(m.worker_metrics_up, (ep,)), ok ? 1.0 : 0.0)
        end
    end
    return String[b for b in bodies if b !== nothing]
end

# --- Admin HTTP server ------------------------------------------------------------------------

# Exposes /metrics, /healthz, and /readyz on a separate HTTP/1.1 listener. /readyz reports 200
# once at least one worker has reported ServerReady. With `worker_metrics` endpoints configured,
# /metrics aggregates every worker's export behind the gateway's own, so one scrape covers the
# whole node (workers self-tag their series with worker/gpu labels).
mutable struct AdminServer
    ready::Threads.Atomic{Bool}
    metrics::GatewayMetrics
    server::Any
end

set_ready!(a::AdminServer, v::Bool) = (a.ready[] = v; nothing)

function start_admin(metrics::GatewayMetrics, addr::AbstractString;
                     worker_metrics::Vector{String}=String[])
    host, port = _split_hostport(addr)
    ready = Threads.Atomic{Bool}(false)
    handler = function (req)
        target = req.target
        if target == "/metrics" || startswith(target, "/metrics?")
            body = if isempty(worker_metrics)
                io = IOBuffer()
                expose(io, metrics)
                String(take!(io))
            else
                workers = _fetch_worker_metrics(metrics, worker_metrics)
                io = IOBuffer()
                expose(io, metrics)   # after the fetch, so worker_metrics_up is current
                merge_expositions(pushfirst!(workers, String(take!(io))))
            end
            return HTTP.Response(200, ["Content-Type" => Prometheus.CONTENT_TYPE_LATEST];
                                 body = body)
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
