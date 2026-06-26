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
    worker_names::Vector{String}            # friendly names (worker0..N), index-aligned to workers/worker_metrics; empty -> labels show the raw url
    request_timeout_seconds::Int
    max_recv_msg_bytes::Int
    max_send_msg_bytes::Int
    log_level::String
    log_format::String
    # Scheduling (the `scheduling:` block); the mode selects the gateway scheduler (see scheduler.jl).
    # `round_robin` spreads each model's requests uniformly across its replicas (the original
    # behavior); `least_outstanding` sends each request to the replica with the least in-flight work
    # (spreads without concentrating); `lpt_packing` concentrates each model's traffic on few workers,
    # placing each model on a fixed, operator-configured number of distinct GPUs and routing a model's
    # requests to fill one replica's batch before the next, to maximize the workers' batch coalescing.
    # LPT packing requires worker FIFO discipline and all models on all workers (checked at startup).
    # The remaining knobs apply to lpt_packing only.
    scheduling_mode::String                 # "round_robin" | "least_outstanding" | "lpt_packing"
    # Repack cadence is driven by accumulated fleet compute, not wall-clock: a repack fires once the
    # fleet has consumed `rebalance_compute_seconds` GPU-seconds since the last one. The first
    # tick-driven repack can use a separate, typically smaller, budget so an early rebalance corrects
    # the cold placement quickly; later repacks use the larger steady-state budget to avoid memory
    # churn and fragmentation. `first_rebalance_compute_seconds` of 0 means "use
    # `rebalance_compute_seconds`" (no separate first budget). The startup cold placement does not
    # consume the first budget; only the first repack driven by real traffic does.
    rebalance_compute_seconds::Float64
    first_rebalance_compute_seconds::Float64
    max_worker_share::Float64               # advisory only: load no longer drives a model's GPU count
    hysteresis::Float64                     # min relative max-load improvement required to move a placement
    # EWMA halflife for the per-model arrival-rate and cost smoothing, measured in fleet
    # compute-seconds (the same clock that triggers repacks), not wall-clock. The EWMA folds only at
    # a repack and decays against the compute consumed since the last one, so the demand signal ages
    # in proportion to how much inference the fleet is doing (faster when busy, coasting when idle)
    # and stays aligned to the repack cadence regardless of GPU count. 0 means "track the rebalance
    # interval" (use `rebalance_compute_seconds`), which gives about 50% decay per rebalance.
    ema_halflife_compute_seconds::Float64
    # Replica placement: a model lives on exactly `replicas` distinct GPUs (per-model override under
    # `scheduling.models`, else `default_replicas`). Set at startup; never grows automatically.
    # `default_replicas: all` (stored as REPLICAS_ALL) places every model on every ready worker.
    default_replicas::Int
    # Request routing within a model's replica set (lpt_packing only). `routing_fill_factor` is the
    # per-replica fill target as a multiple of the model's max batch size (1.0 fills exactly one batch
    # before moving on; >1 over-provisions to keep the next batch queued). `routing_policy` selects
    # where a new batch starts (both variants still concentrate to fill one replica's batch before the
    # next; they differ only in which replica a fresh batch opens on):
    #   "fill_rr"     round-robins the starting replica across the model's set (default).
    #   "fill_least"  starts on the replica with the least in-flight compute load (in-flight
    #                 requests weighted by the model's measured per-request cost), so a model's
    #                 batches open on whichever GPU is least busy across all models.
    # Spreading every request without concentration is the separate `least_outstanding` scheduling
    # mode (see scheduling_mode above), not a routing policy.
    routing_fill_factor::Float64
    routing_policy::String
    models::Dict{String,GatewayModelConfig}
    # Concurrency limits. `max_concurrent_streams_per_worker` is the outbound cap: the in-flight
    # gRPC streams the gateway will multiplex over one worker's shared libcurl handle (the rest block
    # until a slot frees). `max_concurrent_requests_per_worker` sizes the inbound cap: the gateway
    # sheds inbound RPCs past `max_concurrent_requests_per_worker * n_workers` with RESOURCE_EXHAUSTED.
    # The inbound multiple is set above the outbound limit so a startup burst has wiggle room rather
    # than being rejected before the workers are even saturated.
    max_concurrent_streams_per_worker::Int
    max_concurrent_requests_per_worker::Int
    # Memory compaction (lpt_packing only). After a placement-changing repack, the gateway can fan a
    # `CompactMemory` RPC out to the workers whose assignment changed, defragmenting their on-demand
    # weight region. `compaction_mode` selects what each worker reloads eagerly afterward: `:off`
    # disables it, `:eager` reloads nothing (the region refills lazily as traffic arrives), and
    # `:scheduled` reloads the set of models the repack just assigned to that worker (warming the new
    # placement). `compaction_interval` is the cadence in repacks: the first placement-changing repack
    # at or after this many repacks fires it (so it can land later than exactly N).
    compaction_mode::Symbol
    compaction_interval::Int
    # No-eviction guarantee (lpt_packing only). When true, the packer treats each worker's weight
    # budget as a hard constraint: a model is placed only on workers where its weights still fit,
    # so a feasible non-oversubscribed placement is never passed over in favor of one that strands a
    # model on-demand (the worker LRU evicting a placed model). It falls back to the unconstrained
    # choice only when no worker can fit the model (the total weights genuinely outgrew the fleet),
    # so it never throws and never does worse than the soft max-norm. Being greedy (first-fit
    # decreasing) it is not guaranteed to find a feasible packing that exists, but it never strands a
    # model when the greedy pass can avoid it. Default on.
    forbid_memory_oversubscription::Bool
end

const GW_ENV_PREFIX = "REACTANT_GATEWAY_"
const GW_ENV_PATHS = Tuple{String,Vector{String},DataType}[
    ("LISTEN_GRPC", ["listen", "grpc"], String),
    ("LISTEN_METRICS", ["listen", "metrics"], String),
    ("SCHEDULING_MODE", ["scheduling", "mode"], String),
    ("SCHEDULING_REBALANCE_COMPUTE_SECONDS", ["scheduling", "rebalance_compute_seconds"], Float64),
    ("SCHEDULING_FIRST_REBALANCE_COMPUTE_SECONDS", ["scheduling", "first_rebalance_compute_seconds"], Float64),
    ("SCHEDULING_MAX_WORKER_SHARE", ["scheduling", "max_worker_share"], Float64),
    ("SCHEDULING_HYSTERESIS", ["scheduling", "hysteresis"], Float64),
    ("SCHEDULING_EMA_HALFLIFE_COMPUTE_SECONDS", ["scheduling", "ema_halflife_compute_seconds"], Float64),
    ("SCHEDULING_FORBID_MEMORY_OVERSUBSCRIPTION", ["scheduling", "forbid_memory_oversubscription"], Bool),
    ("SCHEDULING_DEFAULT_REPLICAS", ["scheduling", "default_replicas"], String),
    ("SCHEDULING_ROUTING_FILL_FACTOR", ["scheduling", "routing_fill_factor"], Float64),
    ("SCHEDULING_ROUTING_POLICY", ["scheduling", "routing_policy"], String),
    ("SCHEDULING_COMPACTION_MODE", ["scheduling", "compaction_mode"], String),
    ("SCHEDULING_COMPACTION_INTERVAL", ["scheduling", "compaction_interval"], Int),
    ("WORKER_CLIENT_REQUEST_TIMEOUT_SECONDS", ["worker_client", "request_timeout_seconds"], Int),
    ("WORKER_CLIENT_MAX_CONCURRENT_STREAMS", ["worker_client", "max_concurrent_streams"], Int),
    ("GRPC_MAX_RECV_MSG_BYTES", ["grpc", "max_recv_msg_bytes"], Int),
    ("GRPC_MAX_SEND_MSG_BYTES", ["grpc", "max_send_msg_bytes"], Int),
    ("GRPC_MAX_CONCURRENT_REQUESTS_PER_WORKER", ["grpc", "max_concurrent_requests_per_worker"], Int),
    ("LOGGING_LEVEL", ["logging", "level"], String),
    ("LOGGING_FORMAT", ["logging", "format"], String),
    ("TLS_CERT_FILE", ["tls", "cert_file"], String),
    ("TLS_KEY_FILE", ["tls", "key_file"], String),
    ("TLS_CLIENT_CA_FILE", ["tls", "client_ca_file"], String),
]

function _apply_gateway_env!(raw::AbstractDict)
    applied = Tuple{String,String}[]
    for (suffix, path, T) in GW_ENV_PATHS
        var = GW_ENV_PREFIX * suffix
        haskey(ENV, var) || continue
        _set_nested!(raw, path, _parse_env_var(T, var, ENV[var]))
        push!(applied, (var, ENV[var]))
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
        push!(applied, (wenv, ENV[wenv]))
    end

    # Optional worker metrics endpoints, aggregated into the admin /metrics so one scrape covers
    # the gateway plus every worker (workers self-tag their series with worker/gpu labels).
    worker_metrics = _gateway_endpoints(raw, "metrics_endpoints")
    menv = GW_ENV_PREFIX * "WORKER_METRICS"
    env_metrics = _env_endpoints(menv)
    if env_metrics !== nothing
        worker_metrics = env_metrics
        push!(applied, (menv, ENV[menv]))
    end

    # Optional friendly worker names (worker0..N), index-aligned to the endpoint lists. The supervisor
    # synthesizes these from each worker's REACTANT_WORKER_NAME so gateway series carry the same
    # `worker` label the workers self-tag; absent (standalone gateway), worker labels stay as the url.
    worker_names = String[String(x) for x in get(raw, "worker_names", String[])]
    nenv = GW_ENV_PREFIX * "WORKER_NAMES"
    if haskey(ENV, nenv)
        worker_names = String[strip(String(x)) for x in split(ENV[nenv], ','; keepempty = false)]
        push!(applied, (nenv, ENV[nenv]))
    end

    scheduling_mode = lowercase(strip(_opt(sched, "mode", String, "round_robin")))
    scheduling_mode in ("round_robin", "least_outstanding", "lpt_packing") ||
        throw(ConfigError("scheduling.mode must be 'round_robin', 'least_outstanding', or 'lpt_packing', got '$scheduling_mode'"))
    routing_policy = lowercase(strip(_opt(sched, "routing_policy", String, "fill_rr")))
    routing_policy == "least_outstanding" &&
        throw(ConfigError("scheduling.routing_policy no longer accepts 'least_outstanding'; it is now a top-level scheduling mode. Set scheduling.mode: least_outstanding instead."))
    routing_policy in ("fill_rr", "fill_least") ||
        throw(ConfigError("scheduling.routing_policy must be 'fill_rr' or 'fill_least', got '$routing_policy'"))
    compaction_mode = _parse_gateway_compaction_mode(_opt(sched, "compaction_mode", String, "eager"))

    cfg = GatewayConfig(
        _opt(listen, "grpc", String, "0.0.0.0:8001"),
        _opt(listen, "metrics", String, "0.0.0.0:8002"),
        workers,
        worker_metrics,
        worker_names,
        _opt(wc, "request_timeout_seconds", Int, 60),
        _opt(grpc, "max_recv_msg_bytes", Int, 512 * 1024 * 1024),
        _opt(grpc, "max_send_msg_bytes", Int, 512 * 1024 * 1024),
        _opt(logging, "level", String, "info"),
        _opt(logging, "format", String, "json"),
        scheduling_mode,
        _opt(sched, "rebalance_compute_seconds", Float64, 300.0),
        _opt(sched, "first_rebalance_compute_seconds", Float64, 60.0),
        _opt(sched, "max_worker_share", Float64, 0.8),
        _opt(sched, "hysteresis", Float64, 0.0),
        _opt(sched, "ema_halflife_compute_seconds", Float64, 0.0),
        _parse_replicas(get(sched, "default_replicas", 1), "scheduling.default_replicas"),
        _opt(sched, "routing_fill_factor", Float64, 1.0),
        routing_policy,
        _parse_gateway_sched_models(sched),
        _opt(wc, "max_concurrent_streams", Int, 32),
        _opt(grpc, "max_concurrent_requests_per_worker", Int, 64),
        compaction_mode,
        _opt(sched, "compaction_interval", Int, 1),
        _opt(sched, "forbid_memory_oversubscription", Bool, true),
    )
    cfg.max_concurrent_streams_per_worker > 0 ||
        throw(ConfigError("worker_client.max_concurrent_streams must be positive"))
    cfg.max_concurrent_requests_per_worker >= 0 ||
        throw(ConfigError("grpc.max_concurrent_requests_per_worker must be non-negative (0 = uncapped)"))
    cfg.max_concurrent_requests_per_worker == 0 ||
        cfg.max_concurrent_requests_per_worker > cfg.max_concurrent_streams_per_worker ||
        @warn "gateway inbound cap per worker is not above the outbound stream limit; a burst may shed before workers saturate" inbound_per_worker = cfg.max_concurrent_requests_per_worker outbound_streams = cfg.max_concurrent_streams_per_worker
    cfg.rebalance_compute_seconds > 0 || throw(ConfigError("scheduling.rebalance_compute_seconds must be positive"))
    cfg.first_rebalance_compute_seconds >= 0 || throw(ConfigError("scheduling.first_rebalance_compute_seconds must be non-negative (0 = use rebalance_compute_seconds)"))
    0 < cfg.max_worker_share <= 1 || throw(ConfigError("scheduling.max_worker_share must be in (0, 1]"))
    0 <= cfg.hysteresis < 1 || throw(ConfigError("scheduling.hysteresis must be in [0, 1)"))
    cfg.ema_halflife_compute_seconds >= 0 || throw(ConfigError("scheduling.ema_halflife_compute_seconds must be non-negative (0 = use rebalance_compute_seconds)"))
    cfg.routing_fill_factor > 0 || throw(ConfigError("scheduling.routing_fill_factor must be positive"))
    cfg.compaction_interval >= 0 || throw(ConfigError("scheduling.compaction_interval must be non-negative (0 = disabled)"))

    if !isempty(_opt(tls, "cert_file", String, "")) || !isempty(_opt(tls, "key_file", String, ""))
        @warn "gateway TLS is configured but not yet enforced; serving cleartext h2c"
    end

    isempty(cfg.workers) && throw(ConfigError("gateway has no endpoints; set 'endpoints:' in gateway.yml or REACTANT_GATEWAY_WORKERS"))
    isempty(applied) || @info "gateway configuration overridden by environment" overrides = ["$k=$v" for (k, v) in applied]
    return cfg
end

function _parse_gateway_compaction_mode(s)
    ls = lowercase(strip(s))
    ls == "off" && return :off
    ls == "eager" && return :eager
    ls == "scheduled" && return :scheduled
    throw(ConfigError("scheduling.compaction_mode must be 'off', 'eager', or 'scheduled', got '$s'"))
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
