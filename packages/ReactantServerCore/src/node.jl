# Node configuration: a single file describing one machine (a single shared IPC space).
#
# A node is one physical machine with a single shared POSIX IPC space; `node.yaml` is the source
# of truth for the workers running on it. Every worker on the node reads the same file and
# resolves its own per-worker `ServerConfig` by name: global settings are merged with the
# worker's overrides, the model repository is shared, the listen port is derived from a base
# port, the device ordinal comes from the worker's GPU, and the node-level `shared_host_weights`
# flag is propagated. Every worker loads (and can serve) every bundle in the repo; the optional
# top-level `models:` map is a per-model override that pins the named models to device memory on
# the listed workers (translated into `scheduler.models.<name>.residency: device`).
#
# The resolved per-worker raw dict has exactly the shape `build_config` expects, so it flows
# through the same `apply_env_overrides!` -> `build_config` -> `validate_config` pipeline used
# for any server config. The node format is the only supported worker config file format. The
# gateway is configured separately (see ReactantServerGateway's gateway.yml).

# Worker-entry keys with structural meaning; any other key is treated as a config override
# block (e.g. `runtime`, `scheduler`) that is deep-merged over `global`.
const _NODE_WORKER_RESERVED = ("name", "gpu", "host", "port")

function _deep_merge!(base::AbstractDict, over::AbstractDict)
    for (k, v) in over
        bk = get(base, k, nothing)
        if v isa AbstractDict && bk isa AbstractDict
            _deep_merge!(bk, v)
        else
            base[k] = v isa AbstractDict ? deepcopy(v) : v
        end
    end
    return base
end

function _node_workers(node::AbstractDict)
    ws = get(node, "workers", nothing)
    ws isa AbstractVector || throw(ConfigError("node config 'workers' must be a list"))
    isempty(ws) && throw(ConfigError("node config 'workers' must list at least one worker"))
    for w in ws
        w isa AbstractDict || throw(ConfigError("each node 'workers' entry must be a mapping"))
    end
    return ws
end

function _worker_name(w::AbstractDict)
    n = get(w, "name", nothing)
    n isa AbstractString || throw(ConfigError("each node worker must have a string 'name'"))
    return String(n)
end

"""
    worker_names(node) -> Vector{String}

The worker names in declaration order.
"""
worker_names(node::AbstractDict) = String[_worker_name(w) for w in _node_workers(node)]

function _worker_port(node::AbstractDict, w::AbstractDict, index::Int)
    if haskey(w, "port")
        p = w["port"]
        p isa Integer || throw(ConfigError("node worker '$(_worker_name(w))' port must be an integer"))
        return Int(p)
    end
    base = get(node, "base_port", nothing)
    base isa Integer || throw(ConfigError("node config 'base_port' must be an integer"))
    return Int(base) + index
end

# Top-level `models:` map (model name -> list of worker names). Returns nothing when absent.
function _model_assignments(node::AbstractDict)
    models = get(node, "models", nothing)
    models === nothing && return nothing
    models isa AbstractDict || throw(ConfigError("node config 'models' must be a mapping of model name to worker list"))
    out = Dict{String,Vector{String}}()
    for (m, targets) in models
        targets isa AbstractVector ||
            throw(ConfigError("node config 'models.$m' must be a list of worker names"))
        names = String[]
        for t in targets
            t isa AbstractString || throw(ConfigError("node config 'models.$m' entries must be worker names (strings)"))
            push!(names, String(t))
        end
        out[String(m)] = names
    end
    return out
end

"""
    validate_node(node)

Structural validation of a parsed node config. Raises `ConfigError` on a malformed file:
missing `model_repo`, duplicate worker names, colliding ports, or a `models:` entry that
targets an undefined worker. The `models:` map is optional for any node: when omitted, every
worker loads every bundle in the repo; when present, it is a per-model override that pins the
named models to device memory on the listed workers (see `worker_raw_config`).
"""
function validate_node(node::AbstractDict)
    repo = get(node, "model_repo", nothing)
    (repo isa AbstractString && !isempty(repo)) ||
        throw(ConfigError("node config 'model_repo' must be a non-empty string"))

    workers = _node_workers(node)
    names = String[]
    ports = Dict{Int,String}()
    for (i, w) in enumerate(workers)
        name = _worker_name(w)
        name in names && throw(ConfigError("duplicate node worker name '$name'"))
        push!(names, name)
        if haskey(w, "gpu") && !(w["gpu"] isa Integer)
            throw(ConfigError("node worker '$name' gpu must be an integer"))
        end
        port = _worker_port(node, w, i - 1)
        if haskey(ports, port)
            throw(ConfigError("node workers '$(ports[port])' and '$name' both bind port $port"))
        end
        ports[port] = name
        # The optional metrics endpoint derives from a node-level metrics_base_port (per-worker
        # offset, like base_port). Check its port against the same map so a metrics port cannot
        # collide with another worker's gRPC or metrics port.
        if haskey(node, "metrics_base_port")
            mbp = node["metrics_base_port"]
            mbp isa Integer || throw(ConfigError("node config 'metrics_base_port' must be an integer"))
            mport = Int(mbp) + (i - 1)
            if haskey(ports, mport)
                throw(ConfigError("node worker '$name' metrics port $mport collides with '$(ports[mport])'"))
            end
            ports[mport] = "$name (metrics)"
        end
    end

    nameset = Set(names)
    assignments = _model_assignments(node)
    # The `models:` map is optional. When omitted, every worker loads (and can serve) every bundle
    # in the repository. When present, it is a per-model override: each (model -> worker) entry
    # pins that model to device memory on that worker; everything else still loads everywhere with
    # default (system-pinned) residency. So the only validation here is that targets name a
    # defined worker.
    if assignments !== nothing
        for (m, targets) in assignments
            isempty(targets) && throw(ConfigError("node config 'models.$m' assigns the model to no workers"))
            for t in targets
                t in nameset ||
                    throw(ConfigError("node config 'models.$m' targets undefined worker '$t' (have: $(join(names, ", ")))"))
            end
        end
    end
    return node
end

"""
    load_node_raw(path) -> Dict{String,Any}

Read a node config file without structural validation. Used by the supervisor, which may need
to synthesize the `workers:` list (see [`materialize_node!`](@ref)) before validating.
"""
function load_node_raw(path::AbstractString)
    isfile(path) || throw(ConfigError("node config file not found: $path"))
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    raw isa AbstractDict || throw(ConfigError("node config root must be a mapping"))
    return Dict{String,Any}(raw)
end

"""
    load_node(path) -> Dict{String,Any}

Read and structurally validate a node config file.
"""
function load_node(path::AbstractString)
    node = load_node_raw(path)
    validate_node(node)
    return node
end

"""
    node_gpus(node) -> :auto | Int | Vector{String}

Parse the optional top-level `gpus:` key: `auto` (the default when absent) asks the supervisor
to enumerate visible devices; an integer is a device count (expanded to ordinals `0..n-1`); a
list gives explicit device selectors (ordinals or GPU UUIDs, passed to `CUDA_VISIBLE_DEVICES`
verbatim).
"""
function node_gpus(node::AbstractDict)
    haskey(node, "gpus") || return :auto
    g = node["gpus"]
    g isa AbstractString && lowercase(strip(g)) == "auto" && return :auto
    if g isa Integer
        Int(g) >= 0 || throw(ConfigError("node config 'gpus' count must be >= 0"))
        return Int(g)
    end
    if g isa AbstractVector
        out = String[]
        for d in g
            d isa Integer ? push!(out, string(Int(d))) :
            d isa AbstractString ? push!(out, String(strip(d))) :
            throw(ConfigError("node config 'gpus' entries must be device ordinals or UUID strings"))
        end
        isempty(out) && throw(ConfigError("node config 'gpus' list must not be empty"))
        return out
    end
    throw(ConfigError("node config 'gpus' must be 'auto', an integer count, or a list of devices"))
end

"""
    materialize_node!(node, devices; cpu_workers=1) -> Vector{Union{String,Nothing}}

Prepare a raw node config for supervised single-container deployment: assign one visible device
per worker and return the per-worker device selectors (in worker declaration order; `nothing`
means the worker gets no `CUDA_VISIBLE_DEVICES` of its own, the CPU case).

With no `workers:` list, one is synthesized: `worker0..workerN-1`, one per device (or
`cpu_workers` workers when `devices` is empty). An explicit `workers:` list wins: each worker is
assigned `devices[i]` positionally, or `devices[gpu+1]` when the entry carries a `gpu:` key. The
`gpu:` key is consumed here (deleted after assignment) so each child process, seeing a single
device through `CUDA_VISIBLE_DEVICES`, resolves device ordinal 0. Assigning one device to two
workers, or having more workers than devices, is a `ConfigError`. Call `validate_node` on the
result before use.
"""
function materialize_node!(node::AbstractDict, devices::AbstractVector{<:AbstractString};
                           cpu_workers::Int=1)
    devs = String[String(d) for d in devices]
    if !haskey(node, "workers")
        n = isempty(devs) ? cpu_workers : length(devs)
        n >= 1 || throw(ConfigError("cannot materialize a node with no devices and cpu_workers < 1"))
        node["workers"] = Any[Dict{String,Any}("name" => "worker$(i)") for i in 0:(n - 1)]
        return isempty(devs) ? Union{String,Nothing}[nothing for _ in 1:n] :
               Union{String,Nothing}[d for d in devs]
    end

    workers = _node_workers(node)
    selectors = Union{String,Nothing}[]
    assigned = Dict{String,String}()   # device selector -> worker name
    for (i, w) in enumerate(workers)
        name = _worker_name(w)
        if isempty(devs)
            push!(selectors, nothing)
            continue
        end
        idx = if haskey(w, "gpu")
            g = w["gpu"]
            g isa Integer || throw(ConfigError("node worker '$name' gpu must be an integer"))
            Int(g) + 1
        else
            i
        end
        idx <= length(devs) ||
            throw(ConfigError("node worker '$name' needs device $(idx - 1) but only $(length(devs)) device(s) are visible"))
        dev = devs[idx]
        haskey(assigned, dev) &&
            throw(ConfigError("node workers '$(assigned[dev])' and '$name' are both assigned device '$dev'; set distinct 'gpu' keys"))
        assigned[dev] = name
        # The child sees exactly this device via CUDA_VISIBLE_DEVICES, renumbered to ordinal 0;
        # a surviving `gpu:` key would make worker_raw_config address a nonexistent ordinal.
        delete!(w, "gpu")
        push!(selectors, dev)
    end
    return selectors
end

"""
    worker_raw_config(node, name) -> Dict{String,Any}

Resolve a single worker's raw config dict from a (validated) node config: deep-merge `global`
with the named worker's override blocks, then set `model_dirs` to the shared repo, the endpoint
port from `base_port`, the runtime device ordinal from the worker's GPU, and the node-level
`shared_host_weights` flag. Every worker loads every bundle in the repo; the top-level `models:`
map is an optional per-model override that pins the named models to device memory on the listed
workers (translated here into `scheduler.models.<name>.residency: device`). The result has the
shape `build_config` consumes.
"""
function worker_raw_config(node::AbstractDict, name::AbstractString)
    workers = _node_workers(node)
    index = findfirst(w -> _worker_name(w) == String(name), workers)
    index === nothing &&
        throw(ConfigError("worker '$name' not defined in node (have: $(join(worker_names(node), ", ")))"))
    w = workers[index]

    g = get(node, "global", Dict{String,Any}())
    g isa AbstractDict || throw(ConfigError("node config 'global' must be a mapping"))
    raw = deepcopy(Dict{String,Any}(g))

    # Worker override blocks (any non-structural key) deep-merge over global.
    overrides = Dict{String,Any}(k => v for (k, v) in w if !(k in _NODE_WORKER_RESERVED))
    _deep_merge!(raw, overrides)

    raw["model_dirs"] = String[String(node["model_repo"])]

    ep = get!(raw, "endpoints", Dict{String,Any}())
    ep isa AbstractDict || throw(ConfigError("node 'global.endpoints' must be a mapping"))
    ep["port"] = _worker_port(node, w, index - 1)
    # Optional Prometheus metrics port, derived per-worker from the node-level metrics_base_port
    # (parallel to base_port). Absent ⇒ metrics disabled (build_config defaults metrics_port to 0).
    if haskey(node, "metrics_base_port")
        mbp = node["metrics_base_port"]
        mbp isa Integer || throw(ConfigError("node config 'metrics_base_port' must be an integer"))
        ep["metrics_port"] = Int(mbp) + (index - 1)
    end

    rt = get!(raw, "runtime", Dict{String,Any}())
    rt isa AbstractDict || throw(ConfigError("node 'global.runtime' must be a mapping"))
    # Each worker is expected to see a single GPU at index 0 (the default working assumption);
    # physical-GPU selection is a deployment concern (e.g. CUDA_VISIBLE_DEVICES). `gpu` is
    # optional and defaults to ordinal 0; set it only for a bare-metal worker that must address
    # a specific ordinal among several visible GPUs.
    rt["device_ordinal"] = haskey(w, "gpu") ? Int(w["gpu"]) : 0
    # `shared_host_weights` is a node-level property (one shared IPC space); propagate it into
    # every worker's runtime block unless the worker already set it explicitly.
    if haskey(node, "shared_host_weights") && !haskey(rt, "shared_host_weights")
        rt["shared_host_weights"] = node["shared_host_weights"]
    end

    # The top-level `models:` map is an optional per-model override: a model assigned to this
    # worker is pinned to device memory here. Translate each such assignment into
    # `scheduler.models.<model>.residency: device`, merging into any existing per-model block so a
    # `weight`/`max_batch_size` set under `global.scheduler.models` is preserved. An explicit
    # `residency` already set for the model wins (the map only fills an unspecified one). The map
    # no longer restricts which models load; with no `models_include` set, every worker loads all.
    assignments = _model_assignments(node)
    if assignments !== nothing
        sc = get!(raw, "scheduler", Dict{String,Any}())
        sc isa AbstractDict || throw(ConfigError("node 'global.scheduler' must be a mapping"))
        sm = get!(sc, "models", Dict{String,Any}())
        sm isa AbstractDict || throw(ConfigError("node 'global.scheduler.models' must be a mapping"))
        for (m, targets) in assignments
            String(name) in targets || continue
            mc = get!(sm, m, Dict{String,Any}())
            mc isa AbstractDict || throw(ConfigError("node 'scheduler.models.$m' must be a mapping"))
            haskey(mc, "residency") || (mc["residency"] = "device")
        end
    end

    return raw
end

"""
    node_server_config(node, worker) -> (ServerConfig, applied_overrides, worker_name)

Resolve the `ServerConfig` for one worker of a node. `worker` may be `nothing` when the node has
exactly one worker (it defaults to that sole entry); otherwise it must name a defined worker.
Environment overrides (`INFERENCE_SERVER_*`) are applied on top, as for any server config. Does
not validate; call `validate_config` on the result.
"""
function node_server_config(node::AbstractDict, worker::Union{AbstractString,Nothing})
    names = worker_names(node)
    wname = if worker !== nothing
        String(worker) in names ||
            throw(ConfigError("worker '$worker' not defined in node (have: $(join(names, ", ")))"))
        String(worker)
    elseif length(names) == 1
        names[1]
    else
        throw(ConfigError("node has $(length(names)) workers; specify which to serve via worker=..."))
    end
    raw = worker_raw_config(node, wname)
    applied = apply_env_overrides!(raw)
    return build_config(raw), applied, wname
end

_trigger_ci = true
