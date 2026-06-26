# Typed server configuration loaded from YAML, with environment-variable overrides.
#
# The config is parsed into immutable structs so it is frozen for the process lifetime
# and errors are caught at startup. Overrides are driven by a declared path table
# rather than by splitting variable names, because field names contain underscores.

struct ConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ConfigError) = print(io, "ConfigError: ", e.msg)

@enum BackendKind CPU_BACKEND CUDA_BACKEND

"""
    ResidencyState

The residency floor an operator (self-managed) or control plane (externally-managed) sets for a
model's weights. `UNPINNED` keeps no guaranteed residency (loaded from the mmap on demand);
`PINNED_SYSTEM` guarantees the weights stay resident in host RAM (and must be transferred to the
device before execution); `PINNED_DEVICE` guarantees them resident on the GPU for the server's
lifetime (exempt from eviction).
"""
@enum ResidencyState UNPINNED PINNED_SYSTEM PINNED_DEVICE

"""
    ResidencyMode

Who owns device residency on a worker, fixed at startup. `SELF_MANAGED` lets the worker
autonomously transfer and evict weights above each model's floor within the device budget;
`EXTERNALLY_MANAGED` makes a control plane authoritative (no autonomous eviction, non-resident
models are not served until pinned). This is no longer configured directly: it is derived from
[`ModelControlMode`](@ref) (`explicit` ⇒ `EXTERNALLY_MANAGED`, otherwise `SELF_MANAGED`).
"""
@enum ResidencyMode SELF_MANAGED EXTERNALLY_MANAGED

"""
    ModelControlMode

How a worker manages the set of loaded models over its lifetime (mirrors NVIDIA Triton's
model-control-mode). `STATIC` loads and compiles every bundle once at startup and never changes
the set. `DYNAMIC` (the default) additionally runs a filesystem watcher that polls the model
repository every `model_poll_seconds` and hot-swaps bundles as they are added, changed, or
removed on disk. `EXPLICIT` cedes authority to an upstream control plane (the externally-managed
residency behavior): no autonomous watcher, and non-resident models are not served until the
control plane pins them.
"""
@enum ModelControlMode STATIC DYNAMIC EXPLICIT

"""
    SchedulingDiscipline

The inter-model dispatch ordering. `FAIR` is the deficit-weighted, cost-aware policy with
per-model weights and the coalescing discount; `FIFO` serves in global arrival order; `EDF`
(earliest-deadline-first) serves the model whose most-urgent queued request has the soonest
deadline. Coalescing runs underneath all three.

Guidance: `FAIR` is for deployments where models share this worker with no upstream placement
authority, a single-GPU worker or a multi-GPU fleet behind the round-robin gateway, where the
worker itself must stop one model from crowding out the rest. Under the gateway's `lpt_packing`
scheduling the gateway is the fairness authority (placement concentration plus the per-worker
share cap), and workers must run `FIFO` or `EDF` so the two do not fight; `lpt_packing` supersedes
the role `FAIR` played on manually-assigned multi-GPU fleets.

`EDF` is for deadline-sensitive deployments where requests carry a remaining-budget timeout (see
the request-level timeout parameter). It degrades to `FIFO` whenever queued requests share the
same deadline, so its only divergence from `FIFO` is to promote requests with less budget left,
in practice the in-flight meta-model sub-calls that have already spent part of their budget on an
earlier stage. It also sheds work that cannot finish within its learned compute cost (laxity), so
it trades some throughput (batch fragmentation, and no per-model fairness) for hitting more
deadlines under load. NOTE: `EDF` derives urgency purely from the deadline, so issuing different
per-client deadlines for the same model reorders that model's service and therefore affects
fairness across clients; uniform deadlines keep it behaving like `FIFO`.
"""
@enum SchedulingDiscipline FAIR FIFO EDF

"""
    RuntimeConfig

Runtime and device settings (the `runtime:` config block). `backend` selects CPU or CUDA
execution; `device_ordinal` picks the GPU among several visible ones; `mem_fraction` is the
fraction of device memory claimed for the pool; `preallocate` claims that pool up front;
`allow_cpu_fallback` permits falling back to CPU when the device is unavailable;
`weight_cache_fraction` is the single knob sizing on-demand (unpinned) weight residency, the
fraction of the BFC arena (`mem_fraction * device memory`) devoted to all weights, pinned plus
on-demand (`1.0`, the default, uses the whole arena minus measured scratch and wiggle; `0`
disables the cache so every model's weights stay resident; resolved to a byte budget at startup,
GPU only); `weight_cache_wiggle_fraction` is the fraction of the arena kept free as
anti-fragmentation slack and feeds the startup auto-sizing (the worker probes each model's peak
device usage and sizes the cache so pinned weights + on-demand + scratch + this slack fit);
`residency_mode` selects self-managed or externally-managed residency; `shared_host_weights` opts
the node into the shared-memory host-weight store so same-node workers share one host copy of each
system-pinned model; and `shared_host_weights_mode` sets the permission bits (an octal string,
default `"666"`) for those shared regions and their lock files. The `"666"` default keeps cross-UID
container setups working but is world-writable; `"660"` is recommended for production and
multi-user systems.
"""
struct RuntimeConfig
    backend::BackendKind
    device_ordinal::Int
    mem_fraction::Float64
    preallocate::Bool
    allow_cpu_fallback::Bool
    residency_mode::ResidencyMode
    shared_host_weights::Bool
    shared_host_weights_mode::UInt16   # permission bits for the shared host-weight regions
    weight_cache_fraction::Float64        # arena fraction for all weights (pinned + on-demand); 0 = cache off
    weight_cache_wiggle_fraction::Float64 # arena fraction kept free as headroom (drives startup auto-sizing)
end

# Five-argument form: device/backend only; residency self-managed, private host weights, and the
# on-demand cache off (fraction 0). Used by tests and programmatic construction; the YAML path
# (`build_config`) supplies the 1.0 fraction default.
RuntimeConfig(backend::BackendKind, device_ordinal::Integer, mem_fraction::Real,
    preallocate::Bool, allow_cpu_fallback::Bool) =
    RuntimeConfig(backend, Int(device_ordinal), Float64(mem_fraction), preallocate, allow_cpu_fallback,
        SELF_MANAGED, false, 0o666, 0.0, 0.0)

# Seven-argument form: adds residency mode and the shared-host-weights flag (cache still off unless
# a fraction is given). Used where a test needs to exercise externally-managed residency.
RuntimeConfig(backend::BackendKind, device_ordinal::Integer, mem_fraction::Real,
    preallocate::Bool, allow_cpu_fallback::Bool, residency_mode::ResidencyMode,
    shared_host_weights::Bool) =
    RuntimeConfig(backend, Int(device_ordinal), Float64(mem_fraction), preallocate, allow_cpu_fallback,
        residency_mode, shared_host_weights, 0o666, 0.0, 0.0)

"""
    ModelSchedConfig

Per-model scheduler overrides (an entry under `scheduler.models`). `weight` is the model's
relative compute share (default `1.0`, so all-default weights yield uniform shares; consulted
only by the fair discipline). `residency` is the model's initial residency floor (see
[`ResidencyState`](@ref)); `nothing` means unspecified, which the server resolves at startup
to `PINNED_SYSTEM` when the on-demand weight cache is enabled (so every model's weights are
materialized into host RAM and an on-demand GPU load is a pure host-to-device transfer) and
`UNPINNED` otherwise. `max_batch_size` caps how many rows the scheduler coalesces into one
dispatch of this model; `nothing` means uncapped. The cap does not change compiled shapes:
a partial fill still pads up to the smallest compiled batch size, and a single request larger
than the cap is still served (requests are never split).
"""
struct ModelSchedConfig
    weight::Float64
    residency::Union{ResidencyState,Nothing}   # nothing = unspecified, resolved at startup
    max_batch_size::Union{Int,Nothing}         # nothing = uncapped
end

# Convenience constructor; residency and the batch cap default to unspecified.
ModelSchedConfig(weight::Real=1.0; residency::Union{ResidencyState,Nothing}=nothing,
    max_batch_size::Union{Integer,Nothing}=nothing) =
    ModelSchedConfig(Float64(weight), residency,
        max_batch_size === nothing ? nothing : Int(max_batch_size))

"""
    SchedulerConfig

Global scheduler settings (the `scheduler:` config block). `ema_halflife_seconds` is the
half-life of the recent-compute moving average that drives fairness; `recency_penalty_cap`
bounds the recency adjustment; `coalescing_discount` is the cost discount applied to coalesced
batches; `cost_ema_alpha` is the smoothing factor for the learned per-batch-size cost;
`max_queue_depth` caps each model's queue independently (a full model rejects new requests
without affecting admission for the others); `dispatch_timeout_seconds` is the per-request
execution timeout; `discipline` selects the inter-model ordering (see
[`SchedulingDiscipline`](@ref)); `compaction_interval` runs worker-local memory compaction every N
on-demand weight-cache loads (0 disables), the standalone (no-gateway) trigger, off by default so a
gateway-fronted worker never self-compacts; and `models` holds the per-model
[`ModelSchedConfig`](@ref) overrides.
"""
struct SchedulerConfig
    ema_halflife_seconds::Float64
    recency_penalty_cap::Float64
    coalescing_discount::Float64
    cost_ema_alpha::Float64
    max_queue_depth::Int
    dispatch_timeout_seconds::Float64
    discipline::SchedulingDiscipline
    compaction_interval::Int
    models::Dict{String,ModelSchedConfig}
end

# Convenience constructor preserving the original three-argument form. The cost-aware knobs,
# discipline, compaction interval, and per-model overrides take their defaults unless passed as keywords.
SchedulerConfig(ema_halflife_seconds::Real, max_queue_depth::Integer, dispatch_timeout_seconds::Real;
    recency_penalty_cap::Real=0.25, coalescing_discount::Real=0.10, cost_ema_alpha::Real=0.2,
    discipline::SchedulingDiscipline=FAIR, compaction_interval::Integer=0,
    models::Dict{String,ModelSchedConfig}=Dict{String,ModelSchedConfig}()) =
    SchedulerConfig(Float64(ema_halflife_seconds), Float64(recency_penalty_cap),
        Float64(coalescing_discount), Float64(cost_ema_alpha), Int(max_queue_depth),
        Float64(dispatch_timeout_seconds), discipline, Int(compaction_interval), models)

"""
    EndpointsConfig

The listen addresses (the `endpoints:` config block): `host`, the gRPC `port`, the optional
`metrics_port` for the Prometheus exposition endpoint (`0` = disabled), and
`max_concurrent_requests`, the cap on simultaneously-handled RPCs (`0` = uncapped). For a worker
fronted by the gateway, bind `host` to all interfaces (`0.0.0.0`) so the gateway and Prometheus can
reach it; the gRPC port is usually derived from the node file's `base_port` and the metrics port
from `metrics_base_port`.

`max_concurrent_requests` is a worker-level overload backstop: past the cap, new requests are shed
immediately with `RESOURCE_EXHAUSTED` rather than queued. Keep it strictly above the gateway's
per-worker outbound stream limit so it never sheds traffic the gateway has already admitted (and so
it never rejects a meta-model's loopback sub-call); in single-worker mode (no gateway, clients hit
the worker directly) it is the only inbound admission control.
"""
struct EndpointsConfig
    host::String
    port::Int
    metrics_port::Int          # Prometheus metrics HTTP port; 0 = disabled
    max_concurrent_requests::Int   # cap on in-flight RPCs; 0 = uncapped
end

# Back-compat: the gRPC-only form (metrics disabled, uncapped) and the host/port/metrics form
# (uncapped). The cap defaults to 0 for programmatic construction; the YAML path (`build_config`)
# supplies its own default.
EndpointsConfig(host, port) = EndpointsConfig(host, port, 0, 0)
EndpointsConfig(host, port, metrics_port) = EndpointsConfig(host, port, metrics_port, 0)

# Default gRPC message-size cap, shared by the worker, gateway, and client. The 4 MiB gRPC default
# is far below what inference tensors routinely need, so the whole stack sizes generously; 512 MiB
# is the common default. These are decode/encode caps, not allocations: raising them costs nothing
# until a message that large actually arrives.
const DEFAULT_GRPC_MSG_BYTES = 512 * 1024 * 1024

"""
    GrpcConfig

gRPC transport limits for a worker's server. `max_recv_msg_bytes` / `max_send_msg_bytes` bound a
single gRPC message in each direction (decode/encode caps, not allocations). Configured under the
`grpc:` block of a worker (or node `global:`) config, with `INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES`
/ `INFERENCE_SERVER_GRPC_MAX_SEND_MSG_BYTES` environment overrides. Default `DEFAULT_GRPC_MSG_BYTES`
(512 MiB).
"""
struct GrpcConfig
    max_recv_msg_bytes::Int
    max_send_msg_bytes::Int
end

GrpcConfig() = GrpcConfig(DEFAULT_GRPC_MSG_BYTES, DEFAULT_GRPC_MSG_BYTES)

"""
    ServerConfig

The fully resolved configuration for a single worker process, frozen for the process lifetime.
It is produced from a node file (see `node.jl`) with environment-variable overrides
applied, then checked by `validate_config`. Fields: `model_dirs` (bundle search paths),
`cache_dir`, the [`RuntimeConfig`](@ref), [`SchedulerConfig`](@ref), and
[`EndpointsConfig`](@ref) sub-configs, `models_include` (an allowlist of model names to
load; empty means load all), `model_poll_seconds` (the `dynamic`-mode interval at which the
worker re-scans its `model_dirs` for added, changed, or removed bundles and hot-swaps them),
`model_control_mode` (see [`ModelControlMode`](@ref): `static`, `dynamic`, or `explicit`), and the
[`GrpcConfig`](@ref) `grpc` sub-config (gRPC message-size limits).
`ReactantServer.serve` also accepts a `ServerConfig` directly.
"""
struct ServerConfig
    model_dirs::Vector{String}
    cache_dir::String
    runtime::RuntimeConfig
    scheduler::SchedulerConfig
    endpoints::EndpointsConfig
    models_include::Vector{String}   # allowlist of model names to load; empty means load all
    model_poll_seconds::Float64      # dynamic-mode model-repository poll interval
    model_control_mode::ModelControlMode   # static | dynamic | explicit
    grpc::GrpcConfig                 # gRPC server message-size limits
end

# Preserve the positional forms. Programmatic construction defaults to STATIC (no surprise
# background watcher for embedders/unit tests); the YAML path (`build_config`) defaults to DYNAMIC.
ServerConfig(model_dirs, cache_dir, runtime::RuntimeConfig, scheduler::SchedulerConfig,
    endpoints::EndpointsConfig) =
    ServerConfig(model_dirs, cache_dir, runtime, scheduler, endpoints, String[], 0.0, STATIC)

ServerConfig(model_dirs, cache_dir, runtime::RuntimeConfig, scheduler::SchedulerConfig,
    endpoints::EndpointsConfig, models_include) =
    ServerConfig(model_dirs, cache_dir, runtime, scheduler, endpoints, models_include, 0.0, STATIC)

# Pre-grpc positional form (eight args): default the gRPC limits so existing callers keep working.
ServerConfig(model_dirs, cache_dir, runtime::RuntimeConfig, scheduler::SchedulerConfig,
    endpoints::EndpointsConfig, models_include, model_poll_seconds, model_control_mode::ModelControlMode) =
    ServerConfig(model_dirs, cache_dir, runtime, scheduler, endpoints, models_include,
        model_poll_seconds, model_control_mode, GrpcConfig())

const ENV_PREFIX = "INFERENCE_SERVER_"

# (env suffix, path into the raw dict, target type). Single source of truth for overrides.
const ENV_PATHS = Tuple{String,Vector{String},DataType}[
    ("CACHE_DIR", ["cache_dir"], String),
    ("MODEL_POLL_SECONDS", ["model_poll_seconds"], Float64),
    ("MODEL_CONTROL_MODE", ["model_control_mode"], String),
    ("RUNTIME_BACKEND", ["runtime", "backend"], String),
    ("RUNTIME_DEVICE_ORDINAL", ["runtime", "device_ordinal"], Int),
    ("RUNTIME_MEM_FRACTION", ["runtime", "mem_fraction"], Float64),
    ("RUNTIME_PREALLOCATE", ["runtime", "preallocate"], Bool),
    ("RUNTIME_ALLOW_CPU_FALLBACK", ["runtime", "allow_cpu_fallback"], Bool),
    ("RUNTIME_WEIGHT_CACHE_FRACTION", ["runtime", "weight_cache_fraction"], Float64),
    ("RUNTIME_WEIGHT_CACHE_WIGGLE_FRACTION", ["runtime", "weight_cache_wiggle_fraction"], Float64),
    ("RUNTIME_SHARED_HOST_WEIGHTS", ["runtime", "shared_host_weights"], Bool),
    ("RUNTIME_SHARED_HOST_WEIGHTS_MODE", ["runtime", "shared_host_weights_mode"], String),
    ("SCHEDULER_DISCIPLINE", ["scheduler", "discipline"], String),
    ("SCHEDULER_EMA_HALFLIFE_SECONDS", ["scheduler", "ema_halflife_seconds"], Float64),
    ("SCHEDULER_RECENCY_PENALTY_CAP", ["scheduler", "recency_penalty_cap"], Float64),
    ("SCHEDULER_COALESCING_DISCOUNT", ["scheduler", "coalescing_discount"], Float64),
    ("SCHEDULER_COST_EMA_ALPHA", ["scheduler", "cost_ema_alpha"], Float64),
    ("SCHEDULER_MAX_QUEUE_DEPTH", ["scheduler", "max_queue_depth"], Int),
    ("SCHEDULER_DISPATCH_TIMEOUT_SECONDS", ["scheduler", "dispatch_timeout_seconds"], Float64),
    ("SCHEDULER_COMPACTION_INTERVAL", ["scheduler", "compaction_interval"], Int),
    ("ENDPOINTS_HOST", ["endpoints", "host"], String),
    ("ENDPOINTS_PORT", ["endpoints", "port"], Int),
    ("ENDPOINTS_METRICS_PORT", ["endpoints", "metrics_port"], Int),
    ("ENDPOINTS_MAX_CONCURRENT_REQUESTS", ["endpoints", "max_concurrent_requests"], Int),
    ("GRPC_MAX_RECV_MSG_BYTES", ["grpc", "max_recv_msg_bytes"], Int),
    ("GRPC_MAX_SEND_MSG_BYTES", ["grpc", "max_send_msg_bytes"], Int),
]

_parse_env(::Type{Int}, s) = parse(Int, s)
_parse_env(::Type{Float64}, s) = parse(Float64, s)
_parse_env(::Type{Bool}, s) = lowercase(strip(s)) in ("1", "true", "yes", "on")
_parse_env(::Type{String}, s) = String(s)

# Parse an environment override, converting a parse failure into a ConfigError that names
# the offending variable instead of a bare ArgumentError from `parse`.
function _parse_env_var(::Type{T}, var::AbstractString, s::AbstractString) where {T}
    try
        return _parse_env(T, s)
    catch
        throw(ConfigError("environment override $var has invalid $(T) value $(repr(s))"))
    end
end

function _set_nested!(d::AbstractDict, path::Vector{String}, val)
    cur = d
    for k in path[1:(end - 1)]
        nxt = get(cur, k, nothing)
        if !(nxt isa AbstractDict)
            nxt = Dict{String,Any}()
            cur[k] = nxt
        end
        cur = nxt
    end
    cur[path[end]] = val
    return nothing
end

# The batch_policy block was removed in favor of a per-model cap; unknown keys and unused env
# vars are otherwise silently ignored, so fail loudly with a migration message instead.
const _BATCH_POLICY_REMOVED_MSG =
    "the 'batch_policy' config has been removed; set a per-model cap via " *
    "scheduler.models.<name>.max_batch_size (the batch axis comes from the model manifest; " *
    "allow_padding had no effect)"

# runtime.residency_mode was folded into the top-level model_control_mode (explicit ⇒ the former
# externally_managed). Fail loudly rather than silently ignore a stale setting.
const _RESIDENCY_MODE_REMOVED_MSG =
    "runtime.residency_mode has been removed; use the top-level model_control_mode instead " *
    "('explicit' replaces 'externally_managed'; 'static'/'dynamic' are self-managed)"

function apply_env_overrides!(raw::AbstractDict)
    for var in keys(ENV)
        startswith(var, ENV_PREFIX * "BATCH_POLICY_") &&
            throw(ConfigError("environment override $var: " * _BATCH_POLICY_REMOVED_MSG))
        var == ENV_PREFIX * "RUNTIME_RESIDENCY_MODE" &&
            throw(ConfigError("environment override $var: " * _RESIDENCY_MODE_REMOVED_MSG))
    end
    applied = Tuple{String,String}[]
    for (suffix, path, T) in ENV_PATHS
        var = ENV_PREFIX * suffix
        haskey(ENV, var) || continue
        _set_nested!(raw, path, _parse_env_var(T, var, ENV[var]))
        push!(applied, (var, ENV[var]))
    end
    mdvar = ENV_PREFIX * "MODEL_DIRS"
    if haskey(ENV, mdvar)
        raw["model_dirs"] = String[String(x) for x in split(ENV[mdvar], ':'; keepempty=false)]
        push!(applied, (mdvar, ENV[mdvar]))
    end
    mivar = ENV_PREFIX * "MODELS_INCLUDE"
    if haskey(ENV, mivar)
        raw["models_include"] = String[String(x) for x in split(ENV[mivar], ':'; keepempty=false)]
        push!(applied, (mivar, ENV[mivar]))
    end
    return applied
end

_coerce(::Type{Int}, v, key) = v isa Integer ? Int(v) : throw(ConfigError("config '$key' must be an integer"))
_coerce(::Type{Float64}, v, key) = v isa Real ? Float64(v) : throw(ConfigError("config '$key' must be a number"))
_coerce(::Type{Bool}, v, key) = v isa Bool ? v : throw(ConfigError("config '$key' must be a boolean"))
_coerce(::Type{String}, v, key) = v isa AbstractString ? String(v) : throw(ConfigError("config '$key' must be a string"))

_opt(d, key, ::Type{T}, default) where {T} = haskey(d, key) ? _coerce(T, d[key], key) : default

function _coerce_strvec(v, key)
    v isa AbstractVector || throw(ConfigError("config '$key' must be a list"))
    return String[x isa AbstractString ? String(x) :
                  throw(ConfigError("config '$key' entries must be strings")) for x in v]
end

function _subdict(raw, key)
    v = get(raw, key, nothing)
    v === nothing && return Dict{String,Any}()
    v isa AbstractDict || throw(ConfigError("config '$key' must be a mapping"))
    return v
end

function _parse_backend(s)
    ls = lowercase(s)
    ls == "cpu" && return CPU_BACKEND
    ls in ("cuda", "gpu") && return CUDA_BACKEND
    throw(ConfigError("runtime.backend must be 'cpu' or 'cuda', got '$s'"))
end

function _parse_control_mode(s)
    ls = lowercase(strip(s))
    ls == "static" && return STATIC
    ls == "dynamic" && return DYNAMIC
    ls == "explicit" && return EXPLICIT
    throw(ConfigError("model_control_mode must be 'static', 'dynamic', or 'explicit', got '$s'"))
end

# Permission bits for the shared host-weight regions, from an octal string like "666".
function _parse_shm_mode(s)
    m = tryparse(UInt16, s; base=8)
    (m !== nothing && m <= 0o777) ||
        throw(ConfigError("runtime.shared_host_weights_mode must be an octal permission string like \"666\" or \"660\", got '$s'"))
    return m
end

function _parse_discipline(s)
    ls = lowercase(strip(s))
    ls == "fair" && return FAIR
    ls == "fifo" && return FIFO
    ls == "edf" && return EDF
    throw(ConfigError("scheduler.discipline must be 'fair', 'fifo', or 'edf', got '$s'"))
end

function _parse_residency(s)
    ls = lowercase(strip(s))
    ls == "unpinned" && return UNPINNED
    ls in ("system", "pinned_system") && return PINNED_SYSTEM
    ls in ("device", "pinned_device") && return PINNED_DEVICE
    throw(ConfigError("residency must be 'unpinned', 'system', or 'device', got '$s'"))
end

# Per-model scheduler overrides under scheduler.models. Each entry may set `weight` (relative
# compute share, default 1.0), `residency` (initial residency floor), and `max_batch_size`
# (coalescing cap, default uncapped). `pin_to_gpu: true` is accepted as a back-compat alias for
# `residency: device`. Unlisted models fall back to the defaults at scheduler-build time.
function _parse_sched_models(sc)
    raw = get(sc, "models", nothing)
    raw === nothing && return Dict{String,ModelSchedConfig}()
    raw isa AbstractDict || throw(ConfigError("config 'scheduler.models' must be a mapping"))
    out = Dict{String,ModelSchedConfig}()
    for (name, v) in raw
        key = "scheduler.models.$name"
        v isa AbstractDict || throw(ConfigError("config '$key' must be a mapping"))
        weight = _opt(v, "weight", Float64, 1.0)
        residency = if haskey(v, "residency")
            _parse_residency(_coerce(String, v["residency"], "$key.residency"))
        elseif _opt(v, "pin_to_gpu", Bool, false)
            PINNED_DEVICE
        else
            nothing            # unspecified: the server resolves the default at startup
        end
        max_batch_size = haskey(v, "max_batch_size") ?
                         _coerce(Int, v["max_batch_size"], "$key.max_batch_size") : nothing
        out[String(name)] = ModelSchedConfig(weight, residency, max_batch_size)
    end
    return out
end

function build_config(raw::AbstractDict)
    haskey(raw, "model_dirs") || throw(ConfigError("missing required config key 'model_dirs'"))
    model_dirs = _coerce_strvec(raw["model_dirs"], "model_dirs")
    cache_dir = _opt(raw, "cache_dir", String, "")

    # Model lifecycle control mode (the single user-facing switch). Defaults to dynamic. Residency
    # ownership is derived from it: explicit ⇒ externally-managed, otherwise self-managed.
    model_control_mode = haskey(raw, "model_control_mode") ?
        _parse_control_mode(_coerce(String, raw["model_control_mode"], "model_control_mode")) :
        DYNAMIC
    residency_mode = model_control_mode == EXPLICIT ? EXTERNALLY_MANAGED : SELF_MANAGED

    rt = _subdict(raw, "runtime")
    haskey(rt, "residency_mode") && throw(ConfigError(_RESIDENCY_MODE_REMOVED_MSG))
    runtime = RuntimeConfig(
        _parse_backend(_opt(rt, "backend", String, "cpu")),
        _opt(rt, "device_ordinal", Int, 0),
        _opt(rt, "mem_fraction", Float64, 0.9),
        _opt(rt, "preallocate", Bool, true),
        _opt(rt, "allow_cpu_fallback", Bool, true),
        residency_mode,
        _opt(rt, "shared_host_weights", Bool, false),
        _parse_shm_mode(_opt(rt, "shared_host_weights_mode", String, "666")),
        _opt(rt, "weight_cache_fraction", Float64, 1.0),
        _opt(rt, "weight_cache_wiggle_fraction", Float64, 0.1),
    )

    sc = _subdict(raw, "scheduler")
    scheduler = SchedulerConfig(
        _opt(sc, "ema_halflife_seconds", Float64, 30.0),
        _opt(sc, "recency_penalty_cap", Float64, 0.25),
        _opt(sc, "coalescing_discount", Float64, 0.10),
        _opt(sc, "cost_ema_alpha", Float64, 0.2),
        _opt(sc, "max_queue_depth", Int, 1024),
        _opt(sc, "dispatch_timeout_seconds", Float64, 30.0),
        haskey(sc, "discipline") ?
            _parse_discipline(_coerce(String, sc["discipline"], "scheduler.discipline")) : FAIR,
        _opt(sc, "compaction_interval", Int, 0),
        _parse_sched_models(sc),
    )

    haskey(raw, "batch_policy") && throw(ConfigError(_BATCH_POLICY_REMOVED_MSG))

    ep = _subdict(raw, "endpoints")
    endpoints = EndpointsConfig(
        _opt(ep, "host", String, "127.0.0.1"),
        _opt(ep, "port", Int, 8080),
        _opt(ep, "metrics_port", Int, 0),
        _opt(ep, "max_concurrent_requests", Int, 64),
    )

    models_include = haskey(raw, "models_include") ?
                     _coerce_strvec(raw["models_include"], "models_include") : String[]

    # Default to a sensible interval so `dynamic` (the default mode) watches out of the box.
    model_poll_seconds = _opt(raw, "model_poll_seconds", Float64, 15.0)

    gc = _subdict(raw, "grpc")
    grpc = GrpcConfig(
        _opt(gc, "max_recv_msg_bytes", Int, DEFAULT_GRPC_MSG_BYTES),
        _opt(gc, "max_send_msg_bytes", Int, DEFAULT_GRPC_MSG_BYTES),
    )

    return ServerConfig(model_dirs, cache_dir, runtime, scheduler, endpoints, models_include,
        model_poll_seconds, model_control_mode, grpc)
end

function validate_config(cfg::ServerConfig)
    isempty(cfg.model_dirs) && throw(ConfigError("at least one model_dir is required"))
    for d in cfg.model_dirs
        isdir(d) || throw(ConfigError("model_dir does not exist: $d"))
    end
    1 <= cfg.endpoints.port <= 65535 || throw(ConfigError("endpoints.port out of range: $(cfg.endpoints.port)"))
    0 <= cfg.endpoints.metrics_port <= 65535 ||
        throw(ConfigError("endpoints.metrics_port out of range: $(cfg.endpoints.metrics_port)"))
    cfg.endpoints.metrics_port == 0 || cfg.endpoints.metrics_port != cfg.endpoints.port ||
        throw(ConfigError("endpoints.metrics_port must differ from endpoints.port ($(cfg.endpoints.port))"))
    cfg.endpoints.max_concurrent_requests >= 0 ||
        throw(ConfigError("endpoints.max_concurrent_requests must be non-negative (0 = uncapped)"))
    cfg.grpc.max_recv_msg_bytes > 0 || throw(ConfigError("grpc.max_recv_msg_bytes must be positive"))
    cfg.grpc.max_send_msg_bytes > 0 || throw(ConfigError("grpc.max_send_msg_bytes must be positive"))
    cfg.scheduler.ema_halflife_seconds > 0 || throw(ConfigError("scheduler.ema_halflife_seconds must be positive"))
    0 < cfg.scheduler.recency_penalty_cap <= 1 || throw(ConfigError("scheduler.recency_penalty_cap must be in (0, 1]"))
    0 <= cfg.scheduler.coalescing_discount < 1 || throw(ConfigError("scheduler.coalescing_discount must be in [0, 1)"))
    0 < cfg.scheduler.cost_ema_alpha <= 1 || throw(ConfigError("scheduler.cost_ema_alpha must be in (0, 1]"))
    cfg.scheduler.max_queue_depth > 0 || throw(ConfigError("scheduler.max_queue_depth must be positive"))
    cfg.scheduler.compaction_interval >= 0 ||
        throw(ConfigError("scheduler.compaction_interval must be non-negative (0 = disabled)"))
    for (name, mc) in cfg.scheduler.models
        mc.weight > 0 || throw(ConfigError("scheduler.models.$name.weight must be positive"))
        mc.max_batch_size === nothing || mc.max_batch_size >= 1 ||
            throw(ConfigError("scheduler.models.$name.max_batch_size must be >= 1"))
    end
    0 < cfg.runtime.mem_fraction <= 1 || throw(ConfigError("runtime.mem_fraction must be in (0, 1]"))
    0 <= cfg.runtime.weight_cache_fraction <= 1 ||
        throw(ConfigError("runtime.weight_cache_fraction must be in [0, 1] (0 disables the on-demand cache)"))
    0 <= cfg.runtime.weight_cache_wiggle_fraction < 1 ||
        throw(ConfigError("runtime.weight_cache_wiggle_fraction must be in [0, 1)"))
    cfg.model_poll_seconds >= 0 || throw(ConfigError("model_poll_seconds must be non-negative"))
    cfg.model_control_mode != DYNAMIC || cfg.model_poll_seconds > 0 ||
        throw(ConfigError("model_control_mode 'dynamic' requires model_poll_seconds > 0"))
    return cfg
end

# The server is configured by a node file (see node.jl); a worker's raw config dict is
# produced by `worker_raw_config` and then turned into a `ServerConfig` by `build_config`.
# `apply_env_overrides!` is applied on top by `node_server_config`.

function log_effective_config(cfg::ServerConfig, applied)
    @info "Effective configuration" model_dirs=cfg.model_dirs models_include=cfg.models_include model_control_mode=cfg.model_control_mode model_poll_seconds=cfg.model_poll_seconds cache_dir=cfg.cache_dir backend=cfg.runtime.backend device_ordinal=cfg.runtime.device_ordinal mem_fraction=cfg.runtime.mem_fraction preallocate=cfg.runtime.preallocate allow_cpu_fallback=cfg.runtime.allow_cpu_fallback weight_cache_fraction=cfg.runtime.weight_cache_fraction weight_cache_wiggle_fraction=cfg.runtime.weight_cache_wiggle_fraction residency_mode=cfg.runtime.residency_mode shared_host_weights=cfg.runtime.shared_host_weights shared_host_weights_mode=string(cfg.runtime.shared_host_weights_mode; base=8) host=cfg.endpoints.host port=cfg.endpoints.port metrics_port=cfg.endpoints.metrics_port max_concurrent_requests=cfg.endpoints.max_concurrent_requests discipline=cfg.scheduler.discipline ema_halflife_seconds=cfg.scheduler.ema_halflife_seconds recency_penalty_cap=cfg.scheduler.recency_penalty_cap coalescing_discount=cfg.scheduler.coalescing_discount cost_ema_alpha=cfg.scheduler.cost_ema_alpha max_queue_depth=cfg.scheduler.max_queue_depth compaction_interval=cfg.scheduler.compaction_interval scheduler_models=collect(keys(cfg.scheduler.models))
    isempty(applied) || @info "Configuration overridden by environment" overrides=["$k=$v" for (k, v) in applied]
    return nothing
end
