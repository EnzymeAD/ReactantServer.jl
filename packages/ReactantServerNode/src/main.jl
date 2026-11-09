# Top-level assembly: resolve the role, detect devices, materialize the node file, build the
# child specs, and run the supervisor. `main()` is the container entrypoint.

function _resolve_role(
        role::Union{Symbol, AbstractString, Nothing}, env::AbstractDict,
        node::Union{AbstractDict, Nothing}
    )
    r = if role !== nothing
        Symbol(lowercase(String(role)))
    elseif haskey(env, "REACTANT_ROLE")
        Symbol(lowercase(strip(env["REACTANT_ROLE"])))
    elseif node !== nothing && haskey(node, "role")
        node["role"] isa AbstractString || throw(ConfigError("node config 'role' must be a string"))
        Symbol(lowercase(strip(node["role"])))
    else
        :all
    end
    r in (:all, :workers, :gateway) ||
        throw(ConfigError("role must be 'all', 'workers', or 'gateway', got '$r'"))
    return r
end

function _node_backend(node::AbstractDict)
    g = get(node, "global", Dict{String, Any}())
    g isa AbstractDict || return "cpu"
    rt = get(g, "runtime", Dict{String, Any}())
    rt isa AbstractDict || return "cpu"
    return lowercase(String(get(rt, "backend", "cpu")))   # build_config's default backend is cpu
end

# The node's global.grpc dict, or an empty dict when absent/malformed. Workers pick this block up
# directly through their own config (it deep-merges via `global:`); the supervisor reads it here only
# to mirror its settings (message-size limits, error detail, Rapid-Reset protection) into the
# embedded gateway's REACTANT_GATEWAY_GRPC_* env, so one node-level `global.grpc` block configures
# the workers and the gateway alike.
function _node_grpc(node::AbstractDict)
    g = get(node, "global", nothing)
    g isa AbstractDict || return Dict{String, Any}()
    gc = get(g, "grpc", nothing)
    return gc isa AbstractDict ? gc : Dict{String, Any}()
end

# A node-level grpc message-size value as an Int, or nothing when absent (so the gateway falls back
# to its own default). Errors loudly on a non-integer rather than silently ignoring a typo.
function _node_grpc_bytes(node::AbstractDict, key::AbstractString)
    gc = _node_grpc(node)
    haskey(gc, key) || return nothing
    v = gc[key]
    v isa Integer || throw(ArgumentError("node config 'global.grpc.$key' must be an integer, got $(repr(v))"))
    return Int(v)
end

# The node-level `global.grpc.error_detail` (`redacted` or `full`), or nothing when absent. Workers
# read it through their own config; the supervisor mirrors it into the embedded gateway, so one
# setting decides how much an INTERNAL error tells clients on the whole node.
function _node_grpc_error_detail(node::AbstractDict)
    gc = _node_grpc(node)
    haskey(gc, "error_detail") || return nothing
    v = gc["error_detail"]
    (v isa AbstractString && lowercase(strip(v)) in ("redacted", "full")) ||
        throw(ArgumentError("node config 'global.grpc.error_detail' must be \"redacted\" or \"full\", got $(repr(v))"))
    return lowercase(strip(String(v)))
end

# The node-wide HTTP/2 Rapid-Reset switch (GW-18, GW-19): `REACTANT_RAPID_RESET_PROTECTION` from the
# supervisor's environment, else `global.grpc.rapid_reset_protection` from the node file, else
# nothing (unset, so every child keeps its own default, off). The environment wins over the file, as
# `REACTANT_GPUS` and `REACTANT_ROLE` do over their keys, and an empty value counts as unset. Unlike
# the children's own boolean overrides, which read any unrecognized value as false, a value that is
# not a boolean is refused: a typo must not silently leave a security setting off.
const NODE_RAPID_RESET_ENV = "REACTANT_RAPID_RESET_PROTECTION"

function _node_rapid_reset_protection(node::Union{AbstractDict, Nothing}, env::AbstractDict)
    v = lowercase(strip(get(env, NODE_RAPID_RESET_ENV, "")))
    if !isempty(v)
        v in ("1", "true", "yes", "on") && return true
        v in ("0", "false", "no", "off") && return false
        throw(ConfigError("$NODE_RAPID_RESET_ENV must be true or false, got $(repr(env[NODE_RAPID_RESET_ENV]))"))
    end
    node === nothing && return nothing
    gc = _node_grpc(node)
    haskey(gc, "rapid_reset_protection") || return nothing
    b = gc["rapid_reset_protection"]
    b isa Bool ||
        throw(ConfigError("node config 'global.grpc.rapid_reset_protection' must be true or false, got $(repr(b))"))
    return b
end

# Record the node-wide value as `global.grpc.rapid_reset_protection` in the node file the workers
# read. Each worker merges its own entry over `global:`, so the value reaches every worker the
# supervisor launches while a worker's own `grpc.rapid_reset_protection` still wins. (Passing it as
# INFERENCE_SERVER_GRPC_RAPID_RESET_PROTECTION instead would override that per-worker value.)
function _set_node_rapid_reset!(node::AbstractDict, value::Bool)
    g = get(node, "global", nothing)
    g === nothing && (g = node["global"] = Dict{String, Any}())
    g isa AbstractDict || throw(ConfigError("node config 'global' must be a mapping"))
    gc = get(g, "grpc", nothing)
    gc === nothing && (gc = g["grpc"] = Dict{String, Any}())
    gc isa AbstractDict || throw(ConfigError("node config 'global.grpc' must be a mapping"))
    gc["rapid_reset_protection"] = value
    return node
end

# The node's worker liveness settings: for each one its `REACTANT_LIVENESS_*` variable, else the
# node file's top-level `liveness:` block, else the default (spec.jl). `failures: 0` turns the
# supervisor's check off; `dispatch_stall_seconds: 0` turns off the worker's own stall report.
const _LIVENESS_KEYS = (
    (:interval_seconds, "interval_seconds", "REACTANT_LIVENESS_INTERVAL_SECONDS", LIVENESS_INTERVAL_SECONDS),
    (:timeout_seconds, "timeout_seconds", "REACTANT_LIVENESS_TIMEOUT_SECONDS", LIVENESS_TIMEOUT_SECONDS),
    (:failures, "failures", "REACTANT_LIVENESS_FAILURES", LIVENESS_FAILURES),
    (:dispatch_stall_seconds, "dispatch_stall_seconds", DISPATCH_STALL_ENV, DISPATCH_STALL_SECONDS),
)

function _liveness_settings(node::Union{AbstractDict, Nothing}, env::AbstractDict)
    block = node === nothing ? nothing : get(node, "liveness", nothing)
    block === nothing && (block = Dict{String, Any}())
    block isa AbstractDict || throw(ConfigError("node config 'liveness' must be a mapping"))
    known = [k[2] for k in _LIVENESS_KEYS]
    for k in keys(block)
        String(k) in known ||
            throw(ConfigError("node config 'liveness' has no setting '$k' (it takes $(join(known, ", ")))"))
    end
    values = map(_LIVENESS_KEYS) do (_, key, var, default)
        raw = strip(get(env, var, ""))
        if !isempty(raw)
            x = tryparse(Float64, raw)
            x === nothing && throw(ConfigError("$var must be a number, got $(repr(env[var]))"))
            return x
        end
        haskey(block, key) || return Float64(default)
        x = block[key]
        (x isa Real && !(x isa Bool)) ||
            throw(ConfigError("node config 'liveness.$key' must be a number, got $(repr(x))"))
        return Float64(x)
    end
    s = NamedTuple{first.(_LIVENESS_KEYS)}(values)
    for key in (:interval_seconds, :timeout_seconds)
        (s[key] > 0 && isfinite(s[key])) ||
            throw(ConfigError("liveness.$key must be a positive number of seconds, got $(s[key])"))
    end
    (s.failures >= 0 && isinteger(s.failures)) ||
        throw(ConfigError("liveness.failures must be a whole number, 0 to turn the check off, got $(s.failures)"))
    s.dispatch_stall_seconds >= 0 ||
        throw(ConfigError("liveness.dispatch_stall_seconds must be 0 (off) or more, got $(s.dispatch_stall_seconds)"))
    return merge(s, (; failures = Int(s.failures)))
end

# The address the supervisor reaches a listener bound to `host` on: loopback for a wildcard bind,
# the address itself otherwise, or `nothing` for a name that does not resolve.
function _probe_ip(host::AbstractString)
    h = strip(host)
    h in ("", "0.0.0.0", "*", "localhost") && return Sockets.IPv4("127.0.0.1")
    h == "::" && return Sockets.IPv6("::1")
    try
        return parse(Sockets.IPAddr, h)
    catch
    end
    try
        return Sockets.getaddrinfo(h)
    catch
        return nothing
    end
end

# Where a worker's liveness is checked: its HTTP listener (`endpoints.metrics_port`) as the worker
# itself resolves it, from its node entry, then the inherited `INFERENCE_SERVER_ENDPOINTS_*`
# variables, then `metrics_port` (a sole worker's public port, which the supervisor sets). Returns
# `(ip, port)`, or `nothing` when the worker has no such listener.
function _liveness_target(
        node::AbstractDict, name::AbstractString, env::AbstractDict;
        metrics_port::Union{Integer, Nothing} = nothing
    )
    ep = get(worker_raw_config(node, name), "endpoints", Dict{String, Any}())
    ep isa AbstractDict || return nothing
    port = if metrics_port !== nothing
        Int(metrics_port)
    elseif haskey(env, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT")
        something(tryparse(Int, strip(env["INFERENCE_SERVER_ENDPOINTS_METRICS_PORT"])), 0)
    else
        p = get(ep, "metrics_port", 0)
        p isa Integer ? Int(p) : 0
    end
    0 < port <= 65535 || return nothing
    host = get(env, "INFERENCE_SERVER_ENDPOINTS_HOST", get(ep, "host", "127.0.0.1"))
    ip = host isa AbstractString ? _probe_ip(host) : nothing
    return ip === nothing ? nothing : (ip, port)
end

# Where the embedded gateway's liveness is checked: its admin listener (`/readyz` and `/healthz`)
# as the gateway resolves it: REACTANT_GATEWAY_LISTEN_METRICS, else `listen.metrics` in a mounted
# gateway.yml, else 0.0.0.0:8002. Returns `(ip, port)`, or `nothing` when that is not an address.
function _gateway_liveness_target(gateway_path::Union{AbstractString, Nothing}, env::AbstractDict)
    addr = get(env, "REACTANT_GATEWAY_LISTEN_METRICS", nothing)
    if addr === nothing && gateway_path !== nothing && isfile(gateway_path)
        raw = try
            YAML.load_file(gateway_path; dicttype = Dict{String, Any})
        catch
            nothing
        end
        listen = raw isa AbstractDict ? get(raw, "listen", nothing) : nothing
        listen isa AbstractDict && (addr = get(listen, "metrics", nothing))
    end
    addr === nothing && (addr = "0.0.0.0:8002")
    addr isa AbstractString || return nothing
    i = findlast(==(':'), addr)
    i === nothing && return nothing
    port = tryparse(Int, addr[(i + 1):end])
    (port !== nothing && 0 < port <= 65535) || return nothing
    ip = _probe_ip(strip(addr[1:(i - 1)], ['[', ']']))
    return ip === nothing ? nothing : (ip, port)
end

# The embedded gateway's check: the workers' `liveness:` settings, on its admin listener.
function _gateway_check(s, gateway_path::Union{AbstractString, Nothing}, env::AbstractDict)
    s.failures == 0 && return nothing
    t = _gateway_liveness_target(gateway_path, env)
    t === nothing && return nothing
    return LivenessCheck(
        t...; interval_seconds = s.interval_seconds, timeout_seconds = s.timeout_seconds, failures = s.failures
    )
end

function _gateway_liveness_note(chk::Union{LivenessCheck, Nothing}, s)
    chk === nothing || return "gateway liveness check: /healthz on $(_hostport(chk)) once its /readyz has answered, same settings as the workers"
    s.failures == 0 && return "gateway liveness check: off (liveness.failures is 0)"
    return "gateway liveness check: off, its admin listen address is not one the supervisor can reach"
end

# The GPU-reclaim wait for each worker's restarts (see `GpuReclaim`), from the same variables as the
# image's startup wait, and the startup line describing it (`nothing` on a CPU node). No wait for a
# worker without a GPU, on a CPU node, without nvidia-smi, or with the timeout set to 0.
function _gpu_reclaims(node::AbstractDict, selectors, env::AbstractDict)
    none = Union{GpuReclaim, Nothing}[nothing for _ in selectors]
    (_node_backend(node) == "cpu" || all(isnothing, selectors)) && return none, nothing
    timeout = _env_seconds(env, GPU_RECLAIM_TIMEOUT_ENV, GPU_RECLAIM_TIMEOUT_SECONDS)
    max_used = _env_seconds(env, GPU_RECLAIM_MAX_USED_ENV, GPU_RECLAIM_MAX_USED_PERCENT)
    max_used <= 100 || throw(ConfigError("$GPU_RECLAIM_MAX_USED_ENV must be a percentage (0 to 100), got $max_used"))
    timeout == 0 && return none, "worker restarts do not wait for GPU memory ($GPU_RECLAIM_TIMEOUT_ENV is 0)"
    smi = _which("nvidia-smi", env)
    smi === nothing && return none, "worker restarts do not wait for GPU memory: nvidia-smi not found"
    smi_timeout = nvidia_smi_timeout(env)
    reclaims = Union{GpuReclaim, Nothing}[
        sel === nothing ? nothing :
            GpuReclaim(sel, smi; timeout_seconds = timeout, max_used_percent = max_used, smi_timeout_seconds = smi_timeout)
            for sel in selectors
    ]
    return reclaims, "worker restarts wait up to $(_secs(timeout))s for their GPU's used memory to drop to $(_secs(max_used))% ($GPU_RECLAIM_TIMEOUT_ENV, $GPU_RECLAIM_MAX_USED_ENV)"
end

# A non-negative number of seconds (or percent) from `env[var]`, else `default`; anything else is a
# ConfigError naming the variable.
function _env_seconds(env::AbstractDict, var::AbstractString, default::Real)
    v = strip(get(env, var, ""))
    isempty(v) && return Float64(default)
    x = tryparse(Float64, v)
    (x !== nothing && x >= 0 && isfinite(x)) ||
        throw(ConfigError("$var must be a non-negative number, got $(repr(env[var]))"))
    return x
end

# The startup line saying how the workers' liveness is checked; `unchecked` names the workers with
# no HTTP listener to check.
function _liveness_note(s, unchecked::Vector{String}, nworkers::Integer)
    stall = s.dispatch_stall_seconds == 0 ? "off" : "$(_secs(s.dispatch_stall_seconds))s"
    s.failures == 0 && return "worker liveness checks: off (liveness.failures is 0); worker dispatch stall limit: $stall"
    length(unchecked) == nworkers &&
        return "worker liveness checks: off, no worker has an HTTP listener to check (set metrics_base_port); worker dispatch stall limit: $stall"
    note = "worker liveness checks: /healthz every $(_secs(s.interval_seconds))s once ready, $(_secs(s.timeout_seconds))s to answer, SIGKILL after $(s.failures) failures in a row; worker dispatch stall limit: $stall"
    isempty(unchecked) || (note *= "; not checked (no HTTP listener): $(join(unchecked, ", "))")
    return note
end

# Write the materialized (workers synthesized, devices assigned) node file where children and
# the healthcheck can read it. /run/reactantserver in the container; a temp dir elsewhere.
function _write_materialized(node::AbstractDict, runtime_dir::Union{AbstractString, Nothing})
    if runtime_dir !== nothing
        mkpath(runtime_dir)
        path = joinpath(runtime_dir, "node.yaml")
        YAML.write_file(path, node)
        return path
    end
    try
        mkpath("/run/reactantserver")
        path = "/run/reactantserver/node.yaml"
        YAML.write_file(path, node)
        return path
    catch
        path = joinpath(mktempdir(), "node.yaml")
        YAML.write_file(path, node)
        return path
    end
end

# The conventional gateway.yml mount point in the image. A Ref only so the tests can point it
# elsewhere: on a host that has this file (a deployment's), build_supervisor would read it.
const _DEFAULT_GATEWAY_FILE = Ref("/etc/reactantserver/gateway.yml")

# An explicitly configured gateway file must exist; the conventional mount point is picked up
# opportunistically. `nothing` means env-only gateway config (endpoints synthesized by us).
function _resolve_gateway_path(gateway_path::Union{AbstractString, Nothing}, env::AbstractDict)
    gateway_path !== nothing && return String(gateway_path)
    if haskey(env, "REACTANT_GATEWAY_FILE")
        p = String(env["REACTANT_GATEWAY_FILE"])
        isfile(p) || throw(ConfigError("REACTANT_GATEWAY_FILE points at a missing file: $p"))
        return p
    end
    p = _DEFAULT_GATEWAY_FILE[]
    return isfile(p) ? p : nothing
end

# Every port the node's workers bind (gRPC plus optional metrics), for the gateway collision check.
function _worker_ports(node::AbstractDict)
    ws = _node_workers(node)
    ports = Int[]
    for (i, w) in enumerate(ws)
        push!(ports, _worker_port(node, w, i - 1))
        mbp = get(node, "metrics_base_port", nothing)
        mbp isa Integer && push!(ports, Int(mbp) + (i - 1))
    end
    return ports
end

"""
    build_supervisor(node_path; role=nothing, gateway_path=nothing, sink=stdout, env=ENV,
                     workspace_root=nothing, runtime_dir=nothing, max_restarts=nothing,
                     active_project=Base.active_project())
        -> Supervisor

Assemble the node's children without running them. Role precedence: keyword, `REACTANT_ROLE`
env, `role:` in the node file, then `all`. Tests and the in-process e2e use this directly
(`run_supervisor!` + `request_shutdown!`); `supervise` is the blocking wrapper.

Where the children load ReactantServer and ReactantServerGateway from is resolved by
`ReactantServerNode.child_environment` from `workspace_root`, `REACTANT_WORKSPACE_ROOT`,
and `active_project`: the `all` role needs both packages, `workers` only ReactantServer, and
`gateway` only ReactantServerGateway.
"""
function build_supervisor(
        node_path::AbstractString;
        role::Union{Symbol, AbstractString, Nothing} = nothing,
        gateway_path::Union{AbstractString, Nothing} = nothing,
        sink::IO = stdout, env::AbstractDict = ENV,
        workspace_root::Union{AbstractString, Nothing} = nothing,
        runtime_dir::Union{AbstractString, Nothing} = nothing,
        max_restarts::Union{Integer, Nothing} = nothing,
        active_project::Union{AbstractString, Nothing} = Base.active_project()
    )
    node = isfile(node_path) ? load_node_raw(node_path) : nothing
    r = _resolve_role(role, env, node)
    r === :gateway || node !== nothing ||
        throw(ConfigError("node config file not found: $node_path"))
    # Resolved before anything is spawned, so a missing package is one clear error rather than a
    # child that dies on startup and is restarted indefinitely.
    needs = r === :all ? ["ReactantServer", "ReactantServerGateway"] :
        r === :workers ? ["ReactantServer"] : ["ReactantServerGateway"]
    children = child_environment(needs; env = env, workspace_root = workspace_root, active_project = active_project)

    mr = max_restarts !== nothing ? Int(max_restarts) :
        parse(Int, get(env, "REACTANT_SUPERVISOR_MAX_RESTARTS", "0"))

    specs = ChildSpec[]
    notes = String[]
    gw_path = _resolve_gateway_path(gateway_path, env)

    # One node-level setting turns HTTP/2 Rapid-Reset protection on or off for every child launched
    # here: the workers through the node file they read, the gateway through its environment.
    rapid_reset = _node_rapid_reset_protection(node, env)
    liveness = _liveness_settings(node, env)

    if r === :gateway
        push!(
            specs, gateway_spec(
                children; gateway_path = gw_path, rapid_reset_protection = rapid_reset,
                liveness = _gateway_check(liveness, gw_path, env)
            )
        )
        push!(notes, _gateway_liveness_note(specs[end].liveness, liveness))
    else
        devices = detect_gpus(env; node = node)
        if isempty(devices) && _node_backend(node) != "cpu"
            throw(ConfigError("no GPUs detected for a CUDA node; run the container with --gpus all, set REACTANT_GPUS, or set global.runtime.backend: cpu"))
        end
        cpu_workers = parse(Int, get(env, "REACTANT_CPU_WORKERS", "1"))
        selectors = materialize_node!(node, devices; cpu_workers = cpu_workers)
        validate_node(node)
        rapid_reset === nothing || _set_node_rapid_reset!(node, rapid_reset)
        length(devices) > length(selectors) &&
            push!(notes, "node file defines $(length(selectors)) worker(s); $(length(devices) - length(selectors)) visible device(s) left unused")
        node_file = _write_materialized(node, runtime_dir)
        push!(notes, "materialized node file: $node_file")
        ws = _node_workers(node)

        # A single worker needs no gateway: it serves the full KServe V2 API on its own. In the
        # all-in-one role it becomes the node's public endpoint directly, binding the gateway's
        # ports (8001/8002) so the external interface is identical to the multi-worker case but
        # without the extra process or hop. The gateway is spawned only for two or more workers.
        # Being the public endpoint, a sole worker also gets the gateway's drain budget on shutdown.
        gpub, mpub = public_ports(env)
        sole_public = r === :all && length(ws) == 1
        # Size each worker's compute-thread pool to its share of the host (cores ÷ workers, capped),
        # not `auto`, so N workers on one node do not each grab every core. REACTANT_WORKER_THREADS
        # overrides the computed value verbatim (bypassing the share split and the cap).
        worker_threads = let v = strip(get(env, "REACTANT_WORKER_THREADS", ""))
            isempty(v) ? _worker_thread_count(Sys.CPU_THREADS, length(ws)) :
                max(1, parse(Int, v))
        end
        push!(notes, "worker compute threads: $worker_threads (host $(Sys.CPU_THREADS) / $(length(ws)) worker(s))")
        # Each worker's liveness is checked on its HTTP listener once it has been ready (see
        # `LivenessCheck`); a worker without one is only restarted when it exits.
        checks = Union{LivenessCheck, Nothing}[
            liveness.failures == 0 ? nothing :
                let t = _liveness_target(node, _worker_name(w), env; metrics_port = sole_public ? mpub : nothing)
                    t === nothing ? nothing : LivenessCheck(
                        t...; interval_seconds = liveness.interval_seconds,
                        timeout_seconds = liveness.timeout_seconds, failures = liveness.failures
                    )
            end
                for w in ws
        ]
        push!(notes, _liveness_note(liveness, String[_worker_name(w) for (w, c) in zip(ws, checks) if c === nothing], length(ws)))
        # A restarted GPU worker first waits for its GPU to release the memory its last run held.
        reclaims, note = _gpu_reclaims(node, selectors, env)
        note === nothing || push!(notes, note)
        # Meta sub-calls run in-process on each worker now (no loopback, no shared-memory mesh), so a
        # worker needs neither a gateway endpoint nor a fan-out region injected. Each worker keeps its
        # own local meta-scratch pool (sized by REACTANT_FANOUT_BYTES/_SLOTS, read in serve()).
        for (i, w) in enumerate(ws)
            push!(
                specs, sole_public ?
                    worker_spec(
                        _worker_name(w), node_file, selectors[i], children;
                        compute_threads = worker_threads, grpc_port = gpub, metrics_port = mpub,
                        node_workers = length(ws), drain_seconds = PUBLIC_DRAIN_SECONDS,
                        liveness = checks[i], dispatch_stall_seconds = liveness.dispatch_stall_seconds,
                        reclaim = reclaims[i]
                    ) :
                    worker_spec(
                        _worker_name(w), node_file, selectors[i], children;
                        compute_threads = worker_threads, node_workers = length(ws),
                        liveness = checks[i], dispatch_stall_seconds = liveness.dispatch_stall_seconds,
                        reclaim = reclaims[i]
                    )
            )
        end
        if sole_public
            push!(notes, "single worker: serving directly on $gpub (gRPC) / $mpub (metrics); no gateway")
        elseif r === :all
            overlap = intersect(Set(_worker_ports(node)), _gateway_listen_ports(gw_path, env))
            isempty(overlap) ||
                push!(notes, "WARNING: worker port(s) $(sort!(collect(overlap))) collide with the gateway listen ports; adjust base_port / metrics_base_port")
            push!(
                specs, gateway_spec(
                    children; gateway_path = gw_path,
                    endpoints = gw_path === nothing ? worker_endpoints(node) : nothing,
                    metrics_endpoints = gw_path === nothing ? worker_metrics_endpoints(node) : nothing,
                    worker_names = gw_path === nothing ? String[_worker_name(w) for w in ws] : nothing,
                    grpc_max_recv = gw_path === nothing ? _node_grpc_bytes(node, "max_recv_msg_bytes") : nothing,
                    grpc_max_send = gw_path === nothing ? _node_grpc_bytes(node, "max_send_msg_bytes") : nothing,
                    grpc_error_detail = gw_path === nothing ? _node_grpc_error_detail(node) : nothing,
                    rapid_reset_protection = rapid_reset,
                    liveness = _gateway_check(liveness, gw_path, env)
                )
            )
            push!(notes, _gateway_liveness_note(specs[end].liveness, liveness))
        end
    end

    sup = Supervisor(specs; sink = sink, max_restarts = mr)
    foreach(n -> _slog(sup, n), notes)
    _slog(sup, "role=$r children=$(join([s.name for s in specs], ", "))")
    return sup
end

"""
    supervise(node_path; role=nothing, gateway_path=nothing, sink=stdout, env=ENV,
              install_signal_handlers=true, kwargs...) -> Int

Run the node: spawn one worker subprocess per visible GPU (and the embedded gateway in the
default all-in-one role), multiplex their output onto `sink` with `[name]` line prefixes,
restart children that die, kill and restart a worker or the embedded gateway that stops
answering its liveness check once it has been ready (the node file's `liveness:` block; see
Liveness under Node Configuration), and block until SIGTERM/SIGINT (or a crash-loop budget
breach). Returns the process exit code.
"""
function supervise(
        node_path::AbstractString; sink::IO = stdout, env::AbstractDict = ENV,
        install_signal_handlers::Bool = true, kwargs...
    )
    sup = build_supervisor(node_path; sink = sink, env = env, kwargs...)
    return run_supervisor!(sup; install_signal_handlers = install_signal_handlers)
end

"""
    main()

Container entrypoint: `supervise` on `REACTANT_NODE_FILE` (default
`/etc/reactantserver/node.yaml`), exiting with the supervisor's code.
"""
main() = exit(supervise(get(ENV, "REACTANT_NODE_FILE", "/etc/reactantserver/node.yaml")))
