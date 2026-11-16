# Gateway metrics, backed by Prometheus.jl. The collectors below preserve the exact metric
# names, labels, help text, and histogram buckets the gateway has always exported; only the
# implementation changed from a hand-rolled text exposition to the vendored Prometheus.jl
# (which now allows the project's HTTP 2.x fork via its patched compat). The public mutator API
# (inc_requests!/observe_*/set_*) is unchanged, so the call sites in server.jl/health.jl are
# untouched. Prometheus collectors are atomically thread-safe, so no external lock is needed.

# ExponentialBuckets(0.001, 2, 18): 1 ms doubling to 131.072 s; Prometheus.jl appends the +Inf
# bucket. The range reaches past the longest deadlines clients use (60 s), so a request that takes
# 10 s or 50 s lands in a bucket that says so instead of in +Inf, where a quantile can only report
# the last finite bound. The worker's _WM_BUCKETS must match, so the two layers compare bucket for
# bucket.
const _LE_BUCKETS = Float64[0.001 * 2.0^k for k in 0:17]

# The media type of the Prometheus text exposition format (version 0.0.4), from the Prometheus
# exposition spec. Prometheus.jl used to export it as a constant and removed it in 1.6.0; it is
# a protocol value, so state it here rather than reaching into the package.
const PROMETHEUS_EXPOSITION_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

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
    model_fill_quantum::Prometheus.Family{Prometheus.Gauge}
    repacks_total::Prometheus.Family{Prometheus.Counter}
    worker_metrics_up::Prometheus.Family{Prometheus.Gauge}
    probe_latency::Prometheus.Family{Prometheus.Histogram}
    repack_moved_models::Prometheus.Counter
    repack_moved_bytes::Prometheus.Counter
    assigned_weight_bytes::Prometheus.Family{Prometheus.Gauge}
    weight_budget_bytes::Prometheus.Family{Prometheus.Gauge}
    repack_objective::Prometheus.Gauge
    staged_changes::Prometheus.Family{Prometheus.Counter}
    # Maps a worker endpoint url (gRPC or metrics) to its friendly name (worker0..N), so every
    # worker-labelled gateway series matches the workers' own self-tagged `worker` label instead of
    # showing a host:port. Empty for a standalone gateway given no names: labels fall back to the url.
    worker_names::Dict{String, String}
end

function GatewayMetrics(worker_names::Dict{String, String} = Dict{String, String}())
    reg = Prometheus.CollectorRegistry()
    requests_total = Prometheus.Family{Prometheus.Counter}(
        "gateway_requests_total",
        "Count of gateway RPCs by method, model, and gRPC status code.",
        (:rpc, :model, :status); registry = reg
    )
    request_latency = Prometheus.Family{Prometheus.Histogram}(
        "gateway_request_latency_seconds", "Gateway-internal latency.",
        (:rpc, :model); buckets = _LE_BUCKETS, registry = reg
    )
    worker_latency = Prometheus.Family{Prometheus.Histogram}(
        "gateway_worker_latency_seconds", "Latency of the worker gRPC call.",
        (:rpc, :worker); buckets = _LE_BUCKETS, registry = reg
    )
    routing_table_size = Prometheus.Gauge(
        "gateway_routing_table_size", "Number of known models in the routing table.";
        registry = reg
    )
    worker_ready = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_ready",
        "1 if the worker reported ServerReady on the most recent health probe, else 0.",
        (:worker,); registry = reg
    )
    placement_weight = Prometheus.Family{Prometheus.Gauge}(
        "gateway_placement_weight",
        "LPT-packing sampling weight of a model on a worker (0 when unplaced).",
        (:model, :worker); registry = reg
    )
    model_utilization = Prometheus.Family{Prometheus.Gauge}(
        "gateway_model_utilization",
        "Estimated per-model expected utilization (arrival rate x compute cost, GPU-seconds/second).",
        (:model,); registry = reg
    )
    model_replicas = Prometheus.Family{Prometheus.Gauge}(
        "gateway_model_replicas",
        "LPT-packing replica count for a model (number of distinct GPUs hosting it; 0 when unplaced).",
        (:model,); registry = reg
    )
    replica_outstanding = Prometheus.Family{Prometheus.Gauge}(
        "gateway_replica_outstanding",
        "In-flight ITEMS on a model's replica: the summed batch size of the requests in flight, " *
            "not a request count. Equals the request count for a client whose requests all carry one item.",
        (:model, :worker); registry = reg
    )
    model_fill_quantum = Prometheus.Family{Prometheus.Gauge}(
        "gateway_model_fill_quantum",
        "Effective per-replica fill quantum for a model (requests; what routing_fill_mode counts).",
        (:model,); registry = reg
    )
    repacks_total = Prometheus.Family{Prometheus.Counter}(
        "gateway_repacks_total",
        "LPT-packing repacks by what triggered them (startup, compute budget, operator, or a change in the healthy workers: membership).",
        (:trigger,); registry = reg
    )
    worker_metrics_up = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_metrics_up",
        "1 if the worker's metrics endpoint answered the most recent aggregated scrape, else 0.",
        (:worker,); registry = reg
    )
    probe_latency = Prometheus.Family{Prometheus.Histogram}(
        "gateway_worker_probe_seconds",
        "Round trip of the ServerReady health probe (a probe that times out counts its full wait). " *
            "A worker whose request handling is stalled shows here first.",
        (:worker,); buckets = _LE_BUCKETS, registry = reg
    )
    repack_moved_models = Prometheus.Counter(
        "gateway_repack_moved_models_total",
        "Models whose set of workers a repack changed (a model's first placement is not a move).";
        registry = reg
    )
    repack_moved_bytes = Prometheus.Counter(
        "gateway_repack_moved_bytes_total",
        "Weight bytes repacks placed on workers that did not hold the model before: what they cost in loads.";
        registry = reg
    )
    assigned_weight_bytes = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_assigned_weight_bytes",
        "Weight bytes of the models the last repack placed on the worker.",
        (:worker,); registry = reg
    )
    weight_budget_bytes = Prometheus.Family{Prometheus.Gauge}(
        "gateway_worker_weight_budget_bytes",
        "The worker's on-demand weight budget as the last repack saw it (0 = unconstrained).",
        (:worker,); registry = reg
    )
    repack_objective = Prometheus.Gauge(
        "gateway_repack_objective",
        "The placement objective after the last repack: expected requests in the system summed over " *
            "the workers, from each one's load (compute plus the upload churn its weight budget causes). " *
            "Lower is better; mean latency is this over the request rate.";
        registry = reg
    )
    staged_changes = Prometheus.Family{Prometheus.Counter}(
        "gateway_repack_staged_changes_total",
        "Placement changes that waited for their weights to be staged on the new worker: committed once " *
            "staged, or dropped when staging failed, was deferred or timed out.",
        (:outcome,); registry = reg
    )
    return GatewayMetrics(
        reg, requests_total, request_latency, worker_latency,
        routing_table_size, worker_ready, placement_weight, model_utilization,
        model_replicas, replica_outstanding, model_fill_quantum, repacks_total,
        worker_metrics_up, probe_latency, repack_moved_models, repack_moved_bytes,
        assigned_weight_bytes, weight_budget_bytes, repack_objective, staged_changes, worker_names
    )
end

# Friendly name (worker0..N) for a worker endpoint url; identity if no name was supplied for it.
_wname(m::GatewayMetrics, w) = get(m.worker_names, String(w), String(w))

inc_requests!(m::GatewayMetrics, rpc, model, status) =
    Prometheus.inc(Prometheus.labels(m.requests_total, (String(rpc), String(model), String(status))))

observe_request!(m::GatewayMetrics, rpc, model, secs) =
    Prometheus.observe(Prometheus.labels(m.request_latency, (String(rpc), String(model))), secs)

observe_worker!(m::GatewayMetrics, rpc, worker, secs) =
    Prometheus.observe(Prometheus.labels(m.worker_latency, (String(rpc), _wname(m, worker))), secs)

set_routing_size!(m::GatewayMetrics, n) = Prometheus.set(m.routing_table_size, Float64(n))

set_worker_ready!(m::GatewayMetrics, worker, ready::Bool) =
    Prometheus.set(Prometheus.labels(m.worker_ready, (_wname(m, worker),)), ready ? 1.0 : 0.0)

set_placement_weight!(m::GatewayMetrics, model, worker, w) =
    Prometheus.set(Prometheus.labels(m.placement_weight, (String(model), _wname(m, worker))), Float64(w))

set_model_utilization!(m::GatewayMetrics, model, u) =
    Prometheus.set(Prometheus.labels(m.model_utilization, (String(model),)), Float64(u))

set_model_replicas!(m::GatewayMetrics, model, k) =
    Prometheus.set(Prometheus.labels(m.model_replicas, (String(model),)), Float64(k))

set_replica_outstanding!(m::GatewayMetrics, model, worker, n) =
    Prometheus.set(Prometheus.labels(m.replica_outstanding, (String(model), _wname(m, worker))), Float64(n))

set_model_fill_quantum!(m::GatewayMetrics, model, q) =
    Prometheus.set(Prometheus.labels(m.model_fill_quantum, (String(model),)), Float64(q))

inc_repacks!(m::GatewayMetrics, trigger) =
    Prometheus.inc(Prometheus.labels(m.repacks_total, (String(trigger),)))

observe_probe!(m::GatewayMetrics, worker, secs) =
    Prometheus.observe(Prometheus.labels(m.probe_latency, (_wname(m, worker),)), Float64(secs))

function inc_repack_moves!(m::GatewayMetrics, models, bytes)
    Prometheus.inc(m.repack_moved_models, Float64(models))
    Prometheus.inc(m.repack_moved_bytes, Float64(bytes))
    return nothing
end

set_repack_objective!(m::GatewayMetrics, n) = Prometheus.set(m.repack_objective, Float64(n))

function inc_staged_changes!(m::GatewayMetrics, committed, dropped)
    Prometheus.inc(Prometheus.labels(m.staged_changes, ("committed",)), Float64(committed))
    Prometheus.inc(Prometheus.labels(m.staged_changes, ("dropped",)), Float64(dropped))
    return nothing
end

function set_worker_weights!(m::GatewayMetrics, worker, assigned, budget)
    Prometheus.set(Prometheus.labels(m.assigned_weight_bytes, (_wname(m, worker),)), Float64(assigned))
    Prometheus.set(Prometheus.labels(m.weight_budget_bytes, (_wname(m, worker),)), Float64(budget))
    return nothing
end

# Time the Julia compiler has spent in this process (JIT of methods on first use), from
# `Base.cumulative_compile_time_ns`. Compilation runs on the thread that first calls a method, so a
# burst of it on the thread that handles requests stalls every call in flight; these counters make
# that visible next to julia_gc_*. Timing must be switched on first (`enable_compile_timing!`), and
# compilation before that is not counted.
struct CompileTimeCollector <: Prometheus.Collector end

Prometheus.metric_names(::CompileTimeCollector) = ("julia_compile_seconds_total", "julia_recompile_seconds_total")

function Prometheus.collect!(metrics::Vector, ::CompileTimeCollector)
    compile_ns, recompile_ns = Base.cumulative_compile_time_ns()
    push!(
        metrics,
        Prometheus.Metric(
            "counter", "julia_compile_seconds_total",
            "Time the Julia compiler has spent compiling code, since compile timing was enabled at startup.",
            Prometheus.Sample(nothing, nothing, nothing, compile_ns / 1.0e9)
        ),
        Prometheus.Metric(
            "counter", "julia_recompile_seconds_total",
            "The part of julia_compile_seconds_total spent recompiling methods invalidated by later definitions.",
            Prometheus.Sample(nothing, nothing, nothing, recompile_ns / 1.0e9)
        ),
    )
    return metrics
end

# Switch on the runtime's cumulative compile timing for the life of the process. `@time` toggles the
# same switch on and off around its expression; this leaves it on, so the counters above keep
# accumulating. The cost is a clock read around each compilation.
enable_compile_timing!() = (Base.cumulative_compile_timing(true); nothing)

# The gateway process's own runtime metrics (julia_gc_*, julia_compile_*, process_*), unlabelled: worker series
# carry a `worker` label, so `{worker=""}` selects the gateway's. Called once, for the serving
# gateway's registry. Guarded so a platform without /proc cannot break startup.
function register_process_metrics!(m::GatewayMetrics)
    try
        Prometheus.GCCollector(; registry = m.registry)
        Prometheus.ProcessCollector(; registry = m.registry)
        Prometheus.register(m.registry, CompileTimeCollector())
    catch err
        @warn "gateway metrics: GC/Process collectors unavailable" exception = err
    end
    enable_compile_timing!()
    return nothing
end

# Reads the gRPC server's live admission counters at scrape time: in-flight RPCs, the cumulative
# count shed at the inbound concurrency cap, and the configured cap (0 = uncapped). Holds only
# references, so the values are always current.
struct AdmissionCollector <: Prometheus.Collector
    inflight::Threads.Atomic{Int}
    shed::Threads.Atomic{Int}
    max_concurrent::Int
end

Prometheus.metric_names(::AdmissionCollector) =
    ("gateway_inflight_requests", "gateway_requests_shed_total", "gateway_max_concurrent_requests")

_adm_scalar(name, type, help, v) =
    Prometheus.Metric(type, name, help, Prometheus.Sample(nothing, nothing, nothing, Float64(v)))

function Prometheus.collect!(metrics::Vector, c::AdmissionCollector)
    push!(
        metrics,
        _adm_scalar(
            "gateway_inflight_requests", "gauge",
            "RPCs admitted and still being handled.", c.inflight[]
        ),
        _adm_scalar(
            "gateway_requests_shed_total", "counter",
            "RPCs rejected with RESOURCE_EXHAUSTED at the inbound concurrency cap, before any handler ran (so not in gateway_requests_total).", c.shed[]
        ),
        _adm_scalar(
            "gateway_max_concurrent_requests", "gauge",
            "Configured inbound RPC cap across all workers (0 = uncapped).", c.max_concurrent
        ),
    )
    return metrics
end

# Register the admission collector against the gateway's metrics registry, exposing the gRPC
# server's live in-flight / shed counters on /metrics.
register_admission!(
    m::GatewayMetrics, inflight::Threads.Atomic{Int}, shed::Threads.Atomic{Int},
    max_concurrent::Integer
) =
    Prometheus.register(m.registry, AdmissionCollector(inflight, shed, Int(max_concurrent)))

"""
    expose(io, metrics)

Write all collectors to `io` in the Prometheus text exposition format.
"""
expose(io::IO, m::GatewayMetrics) = Prometheus.expose(io, m.registry)

# --- Worker metrics aggregation -----------------------------------------------------------------

# The merge reads each exposition as bytes, in place. A scrape of a four-worker node merges about
# 1 MiB in 11,000 lines; making a String of every line, name and prefix used to allocate ten times
# that per scrape, and the half-built result, still live when a young collection ran, was promoted
# and set off full collections at idle.

# A line of input `src` (an index into the merged inputs), as its bytes `first:last` without the
# line's '\n' or the '\r' before it. A `src` of 0 is no line.
struct _ExpoLine
    src::Int
    first::Int
    last::Int
end
const _NO_LINE = _ExpoLine(0, 1, 0)

# A metric family's name, as bytes of one input. Hashed and compared by content, so finding a family
# by name needs no String.
struct _FamilyName
    bytes::Vector{UInt8}
    first::Int
    last::Int
end
_name_bytes(n::_FamilyName) = view(n.bytes, n.first:n.last)
Base.:(==)(a::_FamilyName, b::_FamilyName) = _name_bytes(a) == _name_bytes(b)
Base.hash(n::_FamilyName, h::UInt) = hash(_name_bytes(n), h)

# Whether an ASCII byte is whitespace to `isspace`.
_isblank(b::UInt8) = b == UInt8(' ') || UInt8('\t') <= b <= UInt8('\r')

_same_bytes(bytes::Vector{UInt8}, a::UnitRange{Int}, b::UnitRange{Int}) =
    length(a) == length(b) && view(bytes, a) == view(bytes, b)

function _startswith(bytes::Vector{UInt8}, r::UnitRange{Int}, prefix::String)
    n = ncodeunits(prefix)
    return length(r) >= n && view(bytes, first(r):(first(r) + n - 1)) == codeunits(prefix)
end

# Whether a sample's name is the family name `fam` or extends it with '_' (a histogram's _bucket,
# _sum and _count), both as ranges of `bytes`.
function _in_family(bytes::Vector{UInt8}, name::UnitRange{Int}, fam::UnitRange{Int})
    _same_bytes(bytes, name, fam) && return true
    (length(name) > length(fam) && bytes[first(name) + length(fam)] == UInt8('_')) || return false
    return _same_bytes(bytes, first(name):(first(name) + length(fam) - 1), fam)
end

# What a line is, and the metric name it carries (a range of the same bytes) when it has one.
const _SKIP, _HELP, _TYPE, _SAMPLE = 0, 1, 2, 3

# Blank and comment lines are skipped; a `# HELP` or `# TYPE` line names its family in its third
# whitespace-separated field; a sample's name runs up to its first '{' or ' '.
function _classify(bytes::Vector{UInt8}, line::UnitRange{Int})
    # Past ASCII, what is whitespace and where a character starts are Unicode's to say, so such a
    # line (in practice a HELP text or a label value) is classified as a String, which also fails
    # on a malformed one exactly as the merge always has.
    if !all(<(0x80), view(bytes, line))
        return _classify_string(String(view(bytes, line)), first(line))
    end
    all(_isblank, view(bytes, line)) && return _SKIP, 1:0
    kind = _startswith(bytes, line, "# HELP ") ? _HELP :
        _startswith(bytes, line, "# TYPE ") ? _TYPE : _SAMPLE
    if kind != _SAMPLE
        a = first(line) + ncodeunits("# HELP ")
        while a <= last(line) && _isblank(bytes[a])
            a += 1
        end
        # A header with no name: the String path fails on it as the merge always has.
        a > last(line) && return _classify_string(String(view(bytes, line)), first(line))
        z = a
        while z < last(line) && !_isblank(bytes[z + 1])
            z += 1
        end
        return kind, a:z
    end
    bytes[first(line)] == UInt8('#') && return _SKIP, 1:0
    z = first(line) - 1
    while z < last(line) && bytes[z + 1] != UInt8('{') && bytes[z + 1] != UInt8(' ')
        z += 1
    end
    return _SAMPLE, first(line):z
end

# `_classify` of a line held as a String, by the String functions the merge was first written with.
# `at` is where the line starts in its input, so the name comes back as a range of the input.
function _classify_string(line::String, at::Int)
    isempty(strip(line)) && return _SKIP, 1:0
    if startswith(line, "# HELP ") || startswith(line, "# TYPE ")
        name = split(line; limit = 4)[3]
        a = at + first(only(parentindices(name))) - 1
        return (startswith(line, "# HELP ") ? _HELP : _TYPE), a:(a + ncodeunits(name) - 1)
    end
    startswith(line, '#') && return _SKIP, 1:0
    stop = something(findfirst(c -> c == '{' || c == ' ', line), lastindex(line) + 1)
    return _SAMPLE, at:(at + ncodeunits(line[1:(stop - 1)]) - 1)
end

"""
    merge_expositions(texts) -> String

Merge several Prometheus text expositions into one valid exposition: each metric family keeps a
single `# HELP` / `# TYPE` header (the first seen) and the sample lines from every source are
grouped under it. The sources are the gateway's own export plus each worker's; workers tag all
their series with `worker`/`gpu` labels, so samples never collide.

`texts` holds each exposition as bytes (`Vector{UInt8}`, as an HTTP response body comes), which are
read in place; `String`s are copied to bytes first.
"""
merge_expositions(texts::AbstractVector{String}) =
    merge_expositions(Vector{UInt8}[Vector{UInt8}(t) for t in texts])

function merge_expositions(texts::AbstractVector{Vector{UInt8}})
    # Families are numbered by first appearance, the order they are written in. Each keeps its first
    # HELP and TYPE line, and its sample lines as a list linked through `next` from `head` to
    # `tail`, so they come out in input order with no vector per family.
    ids = Dict{_FamilyName, Int}()
    help = _ExpoLine[]
    type = _ExpoLine[]
    head = Int[]
    tail = Int[]
    lines = _ExpoLine[]
    next = Int[]
    family!(bytes, name) = get!(ids, _FamilyName(bytes, first(name), last(name))) do
        push!(help, _NO_LINE)
        push!(type, _NO_LINE)
        push!(head, 0)
        push!(tail, 0)
        length(help)
    end
    for (src, bytes) in enumerate(texts)
        cur = 0             # the family of this input's latest HELP or TYPE line; 0 before any
        curname = 1:0       # its name, in `bytes`
        i = 1
        n = length(bytes)
        while i <= n
            # A line ends at '\n' or at the end of the input; a '\r' before the '\n' is dropped.
            j = something(findnext(==(UInt8('\n')), bytes, i), n + 1)
            z = j - 1
            j <= n && z >= i && bytes[z] == UInt8('\r') && (z -= 1)
            line = i:z
            i = j + 1
            kind, name = _classify(bytes, line)
            if kind == _SAMPLE
                # A sample belongs to the family of the header before it when its name is that
                # family's or extends it; otherwise its bare name is its own family.
                fam = cur != 0 && _in_family(bytes, name, curname) ? cur : family!(bytes, name)
                # A sample that directly follows the family's previous one in the same input extends
                # its record, so a block of samples is written with one copy.
                t = tail[fam]
                if t != 0 && lines[t].src == src && lines[t].last + 2 == first(line)
                    lines[t] = _ExpoLine(src, lines[t].first, last(line))
                else
                    push!(lines, _ExpoLine(src, first(line), last(line)))
                    push!(next, 0)
                    t == 0 ? (head[fam] = length(lines)) : (next[t] = length(lines))
                    tail[fam] = length(lines)
                end
            elseif kind != _SKIP
                fam = cur != 0 && _same_bytes(bytes, name, curname) ? cur : family!(bytes, name)
                cur, curname = fam, name
                headers = kind == _HELP ? help : type
                headers[fam].src == 0 && (headers[fam] = _ExpoLine(src, first(line), last(line)))
            end
        end
    end
    # Each line is written with its '\n'. The buffer is sized to the output exactly, so it is filled
    # without growing and the String takes it over without a copy.
    width(l::_ExpoLine) = l.src == 0 ? 0 : l.last - l.first + 2
    nbytes = sum(width, help; init = 0) + sum(width, type; init = 0) + sum(width, lines; init = 0)
    io = IOBuffer(; sizehint = nbytes)
    function put(l::_ExpoLine)
        l.src == 0 && return
        write(io, view(texts[l.src], l.first:l.last))
        write(io, UInt8('\n'))
        return
    end
    for fam in eachindex(help)
        put(help[fam])
        put(type[fam])
        r = head[fam]
        while r != 0
            put(lines[r])
            r = next[r]
        end
    end
    return String(take!(io))
end

# Fetch every worker's /metrics concurrently (best effort, bounded by readtimeout) and record
# per-endpoint reachability in gateway_worker_metrics_up.
function _fetch_worker_metrics(m::GatewayMetrics, endpoints::Vector{String}; readtimeout::Int = 5)
    bodies = Vector{Union{Nothing, Vector{UInt8}}}(nothing, length(endpoints))
    @sync for (i, ep) in enumerate(endpoints)
        @async begin
            ok = false
            try
                resp = HTTP.get(
                    "http://$ep/metrics"; retry = false, status_exception = true,
                    connect_timeout = readtimeout, request_timeout = readtimeout
                )
                # HTTP.jl reads each body into a Vector{UInt8} of its own that only `resp` holds, so
                # the merge reads it in place. String(resp.body) copied it: only a vector whose
                # memory was made for a String can hand it over.
                bodies[i] = resp.body
                ok = true
            catch e
                @debug "worker metrics scrape failed" endpoint = ep exception = e
            end
            Prometheus.set(Prometheus.labels(m.worker_metrics_up, (_wname(m, ep),)), ok ? 1.0 : 0.0)
        end
    end
    return Vector{UInt8}[b for b in bodies if b !== nothing]
end

# --- Admin HTTP server ------------------------------------------------------------------------

# Exposes /metrics, /healthz, and /readyz on a separate HTTP/1.1 listener. /readyz reports 200
# once at least one worker has reported ServerReady. With `worker_metrics` endpoints configured,
# /metrics aggregates every worker's export behind the gateway's own, so one scrape covers the
# whole node (workers self-tag their series with worker/gpu labels).
mutable struct AdminServer{S}
    ready::Threads.Atomic{Bool}
    metrics::GatewayMetrics
    server::S          # the HTTP listener handle; type inferred from HTTP.serve!
end

set_ready!(a::AdminServer, v::Bool) = (a.ready[] = v; nothing)

function start_admin(
        metrics::GatewayMetrics, addr::AbstractString;
        worker_metrics::Vector{String} = String[]
    )
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
                merge_expositions(pushfirst!(workers, take!(io)))
            end
            return HTTP.Response(
                200, ["Content-Type" => PROMETHEUS_EXPOSITION_CONTENT_TYPE];
                body = body
            )
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
