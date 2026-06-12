# Gateway configuration, resolved from its own `gateway.yml`. The gateway is decoupled from the
# node files: it is given a flat list of worker `endpoints` (host:port) that may span any number
# of nodes, and it autodiscovers which models each serves via RepositoryIndex (see routing.jl and
# health.jl). `gateway.yml` carries only the gateway's own settings plus the endpoint list.
# Environment overrides use the `REACTANT_GATEWAY_*` prefix; `REACTANT_GATEWAY_WORKERS` (comma
# separated) overrides the endpoint list.

struct GatewayConfig
    listen_grpc::String
    listen_metrics::String
    workers::Vector{String}                 # worker URLs (host:port) to discover and forward to
    request_timeout_seconds::Int
    max_recv_msg_bytes::Int
    max_send_msg_bytes::Int
    log_level::String
    log_format::String
    # Scheduling (the `scheduling:` block). `round_robin` spreads each model's requests uniformly
    # across its replicas (the original behavior); `lpt_packing` concentrates each model's traffic on
    # few workers from measured load and worker-reported compute cost, to maximize the workers'
    # batch coalescing. LPT packing requires worker FIFO discipline and all models on all workers
    # (checked at startup). The remaining knobs apply to lpt_packing only.
    scheduling_mode::String                 # "round_robin" | "lpt_packing"
    rebalance_seconds::Float64              # lpt_packing assignment recompute interval
    max_worker_share::Float64               # cap of one worker's expected capacity a single model may claim
    hysteresis::Float64                     # min relative max-load improvement required to move a placement
    rate_halflife_seconds::Float64          # EWMA halflife for per-model arrival-rate and cost smoothing
end

const GW_ENV_PREFIX = "REACTANT_GATEWAY_"
const GW_ENV_PATHS = Tuple{String,Vector{String},DataType}[
    ("LISTEN_GRPC", ["listen", "grpc"], String),
    ("LISTEN_METRICS", ["listen", "metrics"], String),
    ("SCHEDULING_MODE", ["scheduling", "mode"], String),
    ("SCHEDULING_REBALANCE_SECONDS", ["scheduling", "rebalance_seconds"], Float64),
    ("SCHEDULING_MAX_WORKER_SHARE", ["scheduling", "max_worker_share"], Float64),
    ("SCHEDULING_HYSTERESIS", ["scheduling", "hysteresis"], Float64),
    ("SCHEDULING_RATE_HALFLIFE_SECONDS", ["scheduling", "rate_halflife_seconds"], Float64),
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

# Parse the `endpoints:` list into a deduplicated, order-preserving vector of host:port strings.
function _gateway_endpoints(raw::AbstractDict)
    eps = get(raw, "endpoints", nothing)
    eps === nothing && return String[]
    eps isa AbstractVector || throw(ConfigError("gateway config 'endpoints' must be a list of host:port strings"))
    out = String[]
    for e in eps
        e isa AbstractString || throw(ConfigError("gateway config 'endpoints' entries must be host:port strings"))
        url = String(strip(e))
        isempty(url) && continue
        _split_hostport(url)              # validate shape
        url in out || push!(out, url)
    end
    return out
end

"""
    load_gateway(gateway_path) -> GatewayConfig

Load and validate the gateway's `gateway.yml`: its listen addresses, worker-client and gRPC
limits, logging, and the `endpoints:` worker list. Applies `REACTANT_GATEWAY_*` environment
overrides; `REACTANT_GATEWAY_WORKERS` (comma separated) replaces the endpoint list. TLS settings
are parsed but not enforced (cleartext h2c only for now); a configured cert triggers a warning.
"""
function load_gateway(gateway_path::AbstractString)
    isfile(gateway_path) || throw(ConfigError("gateway config file not found: $gateway_path"))
    parsed = YAML.load_file(gateway_path; dicttype=Dict{String,Any})
    parsed isa AbstractDict || throw(ConfigError("gateway config root must be a mapping"))
    raw = Dict{String,Any}(parsed)

    applied = _apply_gateway_env!(raw)

    listen = _subdict(raw, "listen")
    grpc = _subdict(raw, "grpc")
    wc = _subdict(raw, "worker_client")
    logging = _subdict(raw, "logging")
    tls = _subdict(raw, "tls")
    sched = _subdict(raw, "scheduling")

    workers = _gateway_endpoints(raw)
    wenv = GW_ENV_PREFIX * "WORKERS"
    if haskey(ENV, wenv)
        workers = String[strip(String(x)) for x in split(ENV[wenv], ',') if !isempty(strip(x))]
        for u in workers
            _split_hostport(u)
        end
        push!(applied, wenv)
    end

    scheduling_mode = lowercase(strip(_opt(sched, "mode", String, "round_robin")))
    scheduling_mode in ("round_robin", "lpt_packing") ||
        throw(ConfigError("scheduling.mode must be 'round_robin' or 'lpt_packing', got '$scheduling_mode'"))

    cfg = GatewayConfig(
        _opt(listen, "grpc", String, "0.0.0.0:8001"),
        _opt(listen, "metrics", String, "0.0.0.0:8002"),
        workers,
        _opt(wc, "request_timeout_seconds", Int, 60),
        _opt(grpc, "max_recv_msg_bytes", Int, 256 * 1024 * 1024),
        _opt(grpc, "max_send_msg_bytes", Int, 256 * 1024 * 1024),
        _opt(logging, "level", String, "info"),
        _opt(logging, "format", String, "json"),
        scheduling_mode,
        _opt(sched, "rebalance_seconds", Float64, 15.0),
        _opt(sched, "max_worker_share", Float64, 0.8),
        _opt(sched, "hysteresis", Float64, 0.1),
        _opt(sched, "rate_halflife_seconds", Float64, 30.0),
    )
    cfg.rebalance_seconds > 0 || throw(ConfigError("scheduling.rebalance_seconds must be positive"))
    0 < cfg.max_worker_share <= 1 || throw(ConfigError("scheduling.max_worker_share must be in (0, 1]"))
    0 <= cfg.hysteresis < 1 || throw(ConfigError("scheduling.hysteresis must be in [0, 1)"))
    cfg.rate_halflife_seconds > 0 || throw(ConfigError("scheduling.rate_halflife_seconds must be positive"))

    if !isempty(_opt(tls, "cert_file", String, "")) || !isempty(_opt(tls, "key_file", String, ""))
        @warn "gateway TLS is configured but not yet enforced; serving cleartext h2c"
    end

    isempty(cfg.workers) && throw(ConfigError("gateway has no endpoints; set 'endpoints:' in gateway.yml or REACTANT_GATEWAY_WORKERS"))
    isempty(applied) || @info "gateway configuration overridden by environment" overrides = applied
    return cfg
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
