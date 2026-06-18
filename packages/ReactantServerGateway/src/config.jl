# Gateway configuration, resolved from its own `gateway.yml`. The gateway is decoupled from the
# node files: it is given a flat list of worker `endpoints` (host:port) that may span any number
# of nodes, and it autodiscovers which models each serves via RepositoryIndex (see routing.jl and
# health.jl). `gateway.yml` carries only the gateway's own settings plus the endpoint list.
# Environment overrides use the `REACTANT_GATEWAY_*` prefix; `REACTANT_GATEWAY_WORKERS` (comma
# separated) overrides the endpoint list.

# Replica count sentinel: "all" in the config means "place on every ready worker". It is stored as
# typemax(Int) so the placement clamp (clamp(replicas, 1, nworkers)) resolves it to the current
# worker count, which tracks the fleet as workers come and go.
const REPLICAS_ALL = typemax(Int)

# Per-model scheduling override (an entry under `scheduling.models`). Currently only the replica
# count: how many distinct GPUs host the model under lpt_packing (default `default_replicas`).
struct GatewayModelConfig
    replicas::Int
end

struct GatewayConfig
    listen_grpc::String
    listen_metrics::String
    workers::Vector{String}                 # worker URLs (host:port) to discover and forward to
    worker_metrics::Vector{String}          # worker metrics URLs (host:port) aggregated into /metrics
    request_timeout_seconds::Int
    max_recv_msg_bytes::Int
    max_send_msg_bytes::Int
    log_level::String
    log_format::String
    # Scheduling (the `scheduling:` block). `round_robin` spreads each model's requests uniformly
    # across its replicas (the original behavior); `lpt_packing` concentrates each model's traffic on
    # few workers, placing each model on a fixed, operator-configured number of distinct GPUs and
    # routing a model's requests to fill one replica's batch before the next, to maximize the
    # workers' batch coalescing. LPT packing requires worker FIFO discipline and all models on all
    # workers (checked at startup). The remaining knobs apply to lpt_packing only.
    scheduling_mode::String                 # "round_robin" | "lpt_packing"
    # Repack cadence is driven by accumulated fleet compute, not wall-clock: a repack fires once the
    # fleet has consumed `rebalance_compute_seconds` GPU-seconds since the last one, subject to a
    # `min_rebalance_seconds` wall-clock floor (0 = none).
    rebalance_compute_seconds::Float64
    min_rebalance_seconds::Float64
    max_worker_share::Float64               # advisory only: load no longer drives a model's GPU count
    hysteresis::Float64                     # min relative max-load improvement required to move a placement
    rate_halflife_seconds::Float64          # EWMA halflife for per-model arrival-rate and cost smoothing
    # Replica placement: a model lives on exactly `replicas` distinct GPUs (per-model override under
    # `scheduling.models`, else `default_replicas`). Set at startup; never grows automatically.
    # `default_replicas: all` (stored as REPLICAS_ALL) places every model on every ready worker.
    default_replicas::Int
    # Request routing within a model's replica set. `routing_fill_factor` is the per-replica fill
    # target as a multiple of the model's max batch size (1.0 fills exactly one batch before moving
    # on; >1 over-provisions to keep the next batch queued). `routing_policy` is "fill" (concentrate
    # to fill a batch, then the least-filled replica) or "least_outstanding" (the GPU with the least
    # total in-flight work).
    routing_fill_factor::Float64
    routing_policy::String
    models::Dict{String,GatewayModelConfig}
end

const GW_ENV_PREFIX = "REACTANT_GATEWAY_"
const GW_ENV_PATHS = Tuple{String,Vector{String},DataType}[
    ("LISTEN_GRPC", ["listen", "grpc"], String),
    ("LISTEN_METRICS", ["listen", "metrics"], String),
    ("SCHEDULING_MODE", ["scheduling", "mode"], String),
    ("SCHEDULING_REBALANCE_COMPUTE_SECONDS", ["scheduling", "rebalance_compute_seconds"], Float64),
    ("SCHEDULING_MIN_REBALANCE_SECONDS", ["scheduling", "min_rebalance_seconds"], Float64),
    ("SCHEDULING_MAX_WORKER_SHARE", ["scheduling", "max_worker_share"], Float64),
    ("SCHEDULING_HYSTERESIS", ["scheduling", "hysteresis"], Float64),
    ("SCHEDULING_RATE_HALFLIFE_SECONDS", ["scheduling", "rate_halflife_seconds"], Float64),
    ("SCHEDULING_DEFAULT_REPLICAS", ["scheduling", "default_replicas"], String),
    ("SCHEDULING_ROUTING_FILL_FACTOR", ["scheduling", "routing_fill_factor"], Float64),
    ("SCHEDULING_ROUTING_POLICY", ["scheduling", "routing_policy"], String),
    ("WORKER_CLIENT_REQUEST_TIMEOUT_SECONDS", ["worker_client", "request_timeout_seconds"], Int),
    ("GRPC_MAX_RECV_MSG_BYTES", ["grpc", "max_recv_msg_bytes"], Int),
    ("GRPC_MAX_SEND_MSG_BYTES", ["grpc", "max_send_msg_bytes"], Int),
    ("LOGGING_LEVEL", ["logging", "level"], String),
    ("LOGGING_FORMAT", ["logging", "format"], String),
    ("TLS_CERT_FILE", ["tls", "cert_file"], String),
    ("TLS_KEY_FILE", ["tls", "key_file"], String),
    ("TLS_CLIENT_CA_FILE", ["tls", "client_ca_file"], String),
]

function _apply_gateway_env!(raw::AbstractDict)
    applied = String[]
    for (suffix, path, T) in GW_ENV_PATHS
        var = GW_ENV_PREFIX * suffix
        haskey(ENV, var) || continue
        _set_nested!(raw, path, _parse_env_var(T, var, ENV[var]))
        push!(applied, var)
    end
    return applied
end

# Parse an endpoint list (`endpoints:` / `metrics_endpoints:`) into a deduplicated,
# order-preserving vector of host:port strings.
function _gateway_endpoints(raw::AbstractDict, key::String="endpoints")
    eps = get(raw, key, nothing)
    eps === nothing && return String[]
    eps isa AbstractVector || throw(ConfigError("gateway config '$key' must be a list of host:port strings"))
    out = String[]
    for e in eps
        e isa AbstractString || throw(ConfigError("gateway config '$key' entries must be host:port strings"))
        url = String(strip(e))
        isempty(url) && continue
        _split_hostport(url)              # validate shape
        url in out || push!(out, url)
    end
    return out
end

# A comma-separated env override for an endpoint list, validated; nothing when unset.
function _env_endpoints(var::String)
    haskey(ENV, var) || return nothing
    out = String[strip(String(x)) for x in split(ENV[var], ',') if !isempty(strip(x))]
    for u in out
        _split_hostport(u)
    end
    return out
end

"""
    load_gateway(gateway_path) -> GatewayConfig

Load and validate the gateway's `gateway.yml`: its listen addresses, worker-client and gRPC
limits, logging, the `endpoints:` worker list, and the optional `metrics_endpoints:` worker
metrics list (aggregated into the admin `/metrics`). Applies `REACTANT_GATEWAY_*` environment
overrides; `REACTANT_GATEWAY_WORKERS` / `REACTANT_GATEWAY_WORKER_METRICS` (comma separated)
replace the respective lists. TLS settings are parsed but not enforced (cleartext h2c only for
now); a configured cert triggers a warning.

`gateway_path` may be `nothing`: the config starts from defaults and the environment alone,
which is how the node supervisor launches an embedded gateway (it synthesizes
`REACTANT_GATEWAY_WORKERS` from the node file, so no gateway.yml is needed).
"""
function load_gateway(gateway_path::AbstractString)
    isfile(gateway_path) || throw(ConfigError("gateway config file not found: $gateway_path"))
    parsed = YAML.load_file(gateway_path; dicttype=Dict{String,Any})
    parsed isa AbstractDict || throw(ConfigError("gateway config root must be a mapping"))
    return _build_gateway_config(Dict{String,Any}(parsed))
end

load_gateway(::Nothing) = _build_gateway_config(Dict{String,Any}())

function _build_gateway_config(raw::Dict{String,Any})
    applied = _apply_gateway_env!(raw)

    listen = _subdict(raw, "listen")
    grpc = _subdict(raw, "grpc")
    wc = _subdict(raw, "worker_client")
    logging = _subdict(raw, "logging")
    tls = _subdict(raw, "tls")
    sched = _subdict(raw, "scheduling")

    workers = _gateway_endpoints(raw)
    wenv = GW_ENV_PREFIX * "WORKERS"
    env_workers = _env_endpoints(wenv)
    if env_workers !== nothing
        workers = env_workers
        push!(applied, wenv)
    end

    # Optional worker metrics endpoints, aggregated into the admin /metrics so one scrape covers
    # the gateway plus every worker (workers self-tag their series with worker/gpu labels).
    worker_metrics = _gateway_endpoints(raw, "metrics_endpoints")
    menv = GW_ENV_PREFIX * "WORKER_METRICS"
    env_metrics = _env_endpoints(menv)
    if env_metrics !== nothing
        worker_metrics = env_metrics
        push!(applied, menv)
    end

    scheduling_mode = lowercase(strip(_opt(sched, "mode", String, "round_robin")))
    scheduling_mode in ("round_robin", "lpt_packing") ||
        throw(ConfigError("scheduling.mode must be 'round_robin' or 'lpt_packing', got '$scheduling_mode'"))
    routing_policy = lowercase(strip(_opt(sched, "routing_policy", String, "fill")))
    routing_policy in ("fill", "least_outstanding") ||
        throw(ConfigError("scheduling.routing_policy must be 'fill' or 'least_outstanding', got '$routing_policy'"))

    cfg = GatewayConfig(
        _opt(listen, "grpc", String, "0.0.0.0:8001"),
        _opt(listen, "metrics", String, "0.0.0.0:8002"),
        workers,
        worker_metrics,
        _opt(wc, "request_timeout_seconds", Int, 60),
        _opt(grpc, "max_recv_msg_bytes", Int, 256 * 1024 * 1024),
        _opt(grpc, "max_send_msg_bytes", Int, 256 * 1024 * 1024),
        _opt(logging, "level", String, "info"),
        _opt(logging, "format", String, "json"),
        scheduling_mode,
        _opt(sched, "rebalance_compute_seconds", Float64, 30.0),
        _opt(sched, "min_rebalance_seconds", Float64, 0.0),
        _opt(sched, "max_worker_share", Float64, 0.8),
        _opt(sched, "hysteresis", Float64, 0.1),
        _opt(sched, "rate_halflife_seconds", Float64, 30.0),
        _parse_replicas(get(sched, "default_replicas", 1), "scheduling.default_replicas"),
        _opt(sched, "routing_fill_factor", Float64, 1.0),
        routing_policy,
        _parse_gateway_sched_models(sched),
    )
    cfg.rebalance_compute_seconds > 0 || throw(ConfigError("scheduling.rebalance_compute_seconds must be positive"))
    cfg.min_rebalance_seconds >= 0 || throw(ConfigError("scheduling.min_rebalance_seconds must be non-negative"))
    0 < cfg.max_worker_share <= 1 || throw(ConfigError("scheduling.max_worker_share must be in (0, 1]"))
    0 <= cfg.hysteresis < 1 || throw(ConfigError("scheduling.hysteresis must be in [0, 1)"))
    cfg.rate_halflife_seconds > 0 || throw(ConfigError("scheduling.rate_halflife_seconds must be positive"))
    cfg.routing_fill_factor > 0 || throw(ConfigError("scheduling.routing_fill_factor must be positive"))

    if !isempty(_opt(tls, "cert_file", String, "")) || !isempty(_opt(tls, "key_file", String, ""))
        @warn "gateway TLS is configured but not yet enforced; serving cleartext h2c"
    end

    isempty(cfg.workers) && throw(ConfigError("gateway has no endpoints; set 'endpoints:' in gateway.yml or REACTANT_GATEWAY_WORKERS"))
    isempty(applied) || @info "gateway configuration overridden by environment" overrides = applied
    return cfg
end

# Parse a replica count: a positive integer, or the string "all" (every ready worker, stored as
# REPLICAS_ALL). Accepts an Int (from YAML) or a String (from YAML or a `REACTANT_GATEWAY_*` env).
function _parse_replicas(v, key::AbstractString)
    if v isa AbstractString
        s = lowercase(strip(v))
        s == "all" && return REPLICAS_ALL
        n = tryparse(Int, s)
        (n === nothing || n < 1) &&
            throw(ConfigError("$key must be a positive integer or 'all', got '$v'"))
        return n
    elseif v isa Integer
        Int(v) >= 1 || throw(ConfigError("$key must be a positive integer or 'all', got $v"))
        return Int(v)
    else
        throw(ConfigError("$key must be a positive integer or 'all'"))
    end
end

# Per-model scheduling overrides under `scheduling.models`. Each entry may set `replicas` (the
# number of distinct GPUs that host the model under lpt_packing; a positive integer or "all").
# Unlisted models use `default_replicas`.
function _parse_gateway_sched_models(sched::AbstractDict)
    raw = get(sched, "models", nothing)
    raw === nothing && return Dict{String,GatewayModelConfig}()
    raw isa AbstractDict || throw(ConfigError("scheduling.models must be a mapping"))
    out = Dict{String,GatewayModelConfig}()
    for (name, v) in raw
        key = "scheduling.models.$name"
        v isa AbstractDict || throw(ConfigError("$key must be a mapping"))
        out[String(name)] = GatewayModelConfig(_parse_replicas(get(v, "replicas", 1), "$key.replicas"))
    end
    return out
end

# Split a worker URL "host:port" into (host::String, port::Int).
function _split_hostport(url::AbstractString)
    idx = findlast(==(':'), url)
    idx === nothing && throw(ConfigError("worker target '$url' is not host:port"))
    host = url[1:(idx - 1)]
    port = tryparse(Int, url[(idx + 1):end])
    port === nothing && throw(ConfigError("worker target '$url' has a non-integer port"))
    return String(host), port
end
