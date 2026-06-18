# Top-level assembly: resolve the role, detect devices, materialize the node file, build the
# child specs, and run the supervisor. `main()` is the container entrypoint.

function _resolve_role(role::Union{Symbol,AbstractString,Nothing}, env::AbstractDict,
                       node::Union{AbstractDict,Nothing})
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
    g = get(node, "global", Dict{String,Any}())
    g isa AbstractDict || return "cpu"
    rt = get(g, "runtime", Dict{String,Any}())
    rt isa AbstractDict || return "cpu"
    return lowercase(String(get(rt, "backend", "cpu")))   # build_config's default backend is cpu
end

# Write the materialized (workers synthesized, devices assigned) node file where children and
# the healthcheck can read it. /run/reactantserver in the container; a temp dir elsewhere.
function _write_materialized(node::AbstractDict, runtime_dir::Union{AbstractString,Nothing})
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

# An explicitly configured gateway file must exist; the conventional mount point is picked up
# opportunistically. `nothing` means env-only gateway config (endpoints synthesized by us).
function _resolve_gateway_path(gateway_path::Union{AbstractString,Nothing}, env::AbstractDict)
    gateway_path !== nothing && return String(gateway_path)
    if haskey(env, "REACTANT_GATEWAY_FILE")
        p = String(env["REACTANT_GATEWAY_FILE"])
        isfile(p) || throw(ConfigError("REACTANT_GATEWAY_FILE points at a missing file: $p"))
        return p
    end
    p = "/etc/reactantserver/gateway.yml"
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
                     workspace_root=nothing, runtime_dir=nothing, max_restarts=nothing)
        -> Supervisor

Assemble the node's children without running them. Role precedence: keyword, `REACTANT_ROLE`
env, `role:` in the node file, then `all`. Tests and the in-process e2e use this directly
(`run_supervisor!` + `request_shutdown!`); `supervise` is the blocking wrapper.
"""
function build_supervisor(node_path::AbstractString;
                          role::Union{Symbol,AbstractString,Nothing}=nothing,
                          gateway_path::Union{AbstractString,Nothing}=nothing,
                          sink::IO=stdout, env::AbstractDict=ENV,
                          workspace_root::Union{AbstractString,Nothing}=nothing,
                          runtime_dir::Union{AbstractString,Nothing}=nothing,
                          max_restarts::Union{Integer,Nothing}=nothing)
    root = workspace_root !== nothing ? String(workspace_root) : default_workspace_root(env)
    node = isfile(node_path) ? load_node_raw(node_path) : nothing
    r = _resolve_role(role, env, node)
    r === :gateway || node !== nothing ||
        throw(ConfigError("node config file not found: $node_path"))

    mr = max_restarts !== nothing ? Int(max_restarts) :
         parse(Int, get(env, "REACTANT_SUPERVISOR_MAX_RESTARTS", "0"))

    specs = ChildSpec[]
    notes = String[]
    gw_path = _resolve_gateway_path(gateway_path, env)

    if r === :gateway
        push!(specs, gateway_spec(root; gateway_path=gw_path))
    else
        devices = detect_gpus(env; node=node)
        if isempty(devices) && _node_backend(node) != "cpu"
            throw(ConfigError("no GPUs detected for a CUDA node; run the container with --gpus all, set REACTANT_GPUS, or set global.runtime.backend: cpu"))
        end
        cpu_workers = parse(Int, get(env, "REACTANT_CPU_WORKERS", "1"))
        selectors = materialize_node!(node, devices; cpu_workers=cpu_workers)
        validate_node(node)
        length(devices) > length(selectors) &&
            push!(notes, "node file defines $(length(selectors)) worker(s); $(length(devices) - length(selectors)) visible device(s) left unused")
        node_file = _write_materialized(node, runtime_dir)
        push!(notes, "materialized node file: $node_file")
        ws = _node_workers(node)

        # A single worker needs no gateway: it serves the full KServe V2 API on its own. In the
        # all-in-one role it becomes the node's public endpoint directly, binding the gateway's
        # ports (8001/8002) so the external interface is identical to the multi-worker case but
        # without the extra process or hop. The gateway is spawned only for two or more workers.
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
        for (i, w) in enumerate(ws)
            push!(specs, sole_public ?
                worker_spec(_worker_name(w), node_file, selectors[i], root;
                            compute_threads=worker_threads, grpc_port=gpub, metrics_port=mpub) :
                worker_spec(_worker_name(w), node_file, selectors[i], root;
                            compute_threads=worker_threads))
        end
        if sole_public
            push!(notes, "single worker: serving directly on $gpub (gRPC) / $mpub (metrics); no gateway")
        elseif r === :all
            overlap = intersect(Set(_worker_ports(node)), _gateway_listen_ports(gw_path, env))
            isempty(overlap) ||
                push!(notes, "WARNING: worker port(s) $(sort!(collect(overlap))) collide with the gateway listen ports; adjust base_port / metrics_base_port")
            push!(specs, gateway_spec(root; gateway_path=gw_path,
                                      endpoints=gw_path === nothing ? worker_endpoints(node) : nothing,
                                      metrics_endpoints=gw_path === nothing ? worker_metrics_endpoints(node) : nothing))
        end
    end

    sup = Supervisor(specs; sink=sink, max_restarts=mr)
    foreach(n -> _slog(sup, n), notes)
    _slog(sup, "role=$r children=$(join([s.name for s in specs], ", "))")
    return sup
end

"""
    supervise(node_path; role=nothing, gateway_path=nothing, sink=stdout, env=ENV,
              install_signal_handlers=true, kwargs...) -> Int

Run the node: spawn one worker subprocess per visible GPU (and the embedded gateway in the
default all-in-one role), multiplex their output onto `sink` with `[name]` line prefixes,
restart children that die, and block until SIGTERM/SIGINT (or a crash-loop budget breach).
Returns the process exit code.
"""
function supervise(node_path::AbstractString; sink::IO=stdout, env::AbstractDict=ENV,
                   install_signal_handlers::Bool=true, kwargs...)
    sup = build_supervisor(node_path; sink=sink, env=env, kwargs...)
    return run_supervisor!(sup; install_signal_handlers=install_signal_handlers)
end

"""
    main()

Container entrypoint: `supervise` on `REACTANT_NODE_FILE` (default
`/etc/reactantserver/node.yaml`), exiting with the supervisor's code.
"""
main() = exit(supervise(get(ENV, "REACTANT_NODE_FILE", "/etc/reactantserver/node.yaml")))
