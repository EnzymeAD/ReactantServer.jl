# Child process specifications: the exact Cmd (program, args, env) for each worker and for the
# embedded gateway. Workers run the unchanged `ReactantServer.serve(node_file; worker=name)`
# entry, exactly like the standalone per-GPU container did; the only supervisor-added environment
# is the worker identity and its single-device CUDA_VISIBLE_DEVICES. Bind hosts stay the node
# file's concern (`endpoints.host`); in a container, which ports are reachable is decided by
# what the operator publishes, and worker metrics ports must remain scrapable.

# Worker liveness defaults (the node file's `liveness:` block, see `_liveness_settings` in main.jl).
# A worker that has been ready is asked `/healthz` every 10 s and given 10 s to answer, and is killed
# after 6 failures in a row: about two minutes without an answer, or one minute of "not live"
# answers. That is far beyond a garbage collection or a first-use compile on the thread that
# answers (seconds), and the check never runs before the worker's first ready answer, so a startup
# that compiles for hours is never touched.
const LIVENESS_INTERVAL_SECONDS = 10.0
const LIVENESS_TIMEOUT_SECONDS = 10.0
const LIVENESS_FAILURES = 6
# The worker's own stall limit (`ReactantServer.DISPATCH_STALL_SECONDS`), passed to every worker
# in REACTANT_LIVENESS_DISPATCH_STALL_SECONDS. The worker reads that variable itself; its name is
# repeated here because this package never loads ReactantServer.
const DISPATCH_STALL_SECONDS = 600.0
const DISPATCH_STALL_ENV = "REACTANT_LIVENESS_DISPATCH_STALL_SECONDS"

"""
    LivenessCheck(host, port; interval_seconds=10, timeout_seconds=10, failures=6)

How the supervisor checks that a running worker still works, through the worker's HTTP listener
(`endpoints.metrics_port`) at `host:port`. Until the worker's `/readyz` first answers 200 nothing is
checked, so a worker that compiles its models for hours at startup is left alone. From then on
`/healthz` is asked every `interval_seconds`; an answer other than 200, a refused connection, or no
answer within `timeout_seconds` is a failure, and after `failures` in a row the worker is sent
SIGKILL, so the ordinary restart, with its backoff and restart budget, takes over. The worker's
`/healthz` says not live while its dispatch loop is stalled with work in hand, so a GPU call that
never returns is caught too, not only a process that stops answering.
"""
struct LivenessCheck
    ip::Sockets.IPAddr
    port::Int
    interval::Float64
    timeout::Float64
    failures::Int
end
function LivenessCheck(
        host::Union{AbstractString, Sockets.IPAddr}, port::Integer;
        interval_seconds::Real = LIVENESS_INTERVAL_SECONDS, timeout_seconds::Real = LIVENESS_TIMEOUT_SECONDS,
        failures::Integer = LIVENESS_FAILURES
    )
    ip = host isa Sockets.IPAddr ? host : parse(Sockets.IPAddr, host)
    (0 < port <= 65535) || throw(ArgumentError("liveness port must be in 1..65535, got $port"))
    (interval_seconds > 0 && isfinite(interval_seconds)) ||
        throw(ArgumentError("liveness interval must be a positive number of seconds, got $interval_seconds"))
    (timeout_seconds > 0 && isfinite(timeout_seconds)) ||
        throw(ArgumentError("liveness timeout must be a positive number of seconds, got $timeout_seconds"))
    failures > 0 || throw(ArgumentError("liveness failures must be at least 1, got $failures"))
    return LivenessCheck(ip, Int(port), Float64(interval_seconds), Float64(timeout_seconds), Int(failures))
end

struct ChildSpec
    name::String
    cmd::Cmd
    grace_seconds::Float64   # SIGTERM-to-SIGKILL window at shutdown (after the drain, for a draining child)
    drain_seconds::Float64   # graceful drain budget, requested by closing the child's stdin; 0 = SIGTERM at once
    stage::Int               # shutdown order: every child of a lower stage stops before the next stage is asked
    liveness::Union{LivenessCheck, Nothing}   # how a running child is checked; nothing = only its exit is watched
end
ChildSpec(name::AbstractString, cmd::Cmd, grace_seconds::Real) =
    ChildSpec(String(name), cmd, Float64(grace_seconds), 0.0, 1, nothing)
ChildSpec(name::AbstractString, cmd::Cmd, grace_seconds::Real, drain_seconds::Real, stage::Integer) =
    ChildSpec(String(name), cmd, Float64(grace_seconds), Float64(drain_seconds), Int(stage), nothing)

# How long the node's public endpoint (the embedded gateway, or a sole worker bound to the public
# ports itself) may take on shutdown to finish the requests it has already accepted.
const PUBLIC_DRAIN_SECONDS = 60.0
# How long a worker behind the gateway may take to finish its queued and in-flight requests. The
# gateway has drained before the workers are asked, so this is mostly for direct clients.
const WORKER_DRAIN_SECONDS = 15.0
# Shutdown stages: the gateway stops (draining) while the workers still serve its in-flight requests,
# and only then are the workers asked to drain.
const _GATEWAY_STAGE = 1
const _WORKER_STAGE = 2

"""
    ChildEnvironment(worker_project, gateway_project, load_path)

Where the supervisor's children load their packages from. Two shapes:

- **workspace**: each child runs from its own member project of a ReactantServer workspace
  (`<root>/packages/ReactantServer`, `<root>/packages/ReactantServerGateway`) and inherits the
  load path. This is the container image and a source checkout.
- **environment**: every child runs in the one Julia environment the supervisor runs in, which
  declares ReactantServer and ReactantServerGateway as dependencies next to ReactantServerNode.
  This is the Pkg workflow. `load_path` pins the children to that environment plus the standard
  library, so a copy in the user's default environment cannot stand in for the one checked here.
"""
struct ChildEnvironment
    worker_project::String
    gateway_project::String
    load_path::Union{String, Nothing}   # JULIA_LOAD_PATH for the children; nothing inherits it
end

workspace_environment(root::AbstractString) = ChildEnvironment(
    joinpath(root, "packages", "ReactantServer"),
    joinpath(root, "packages", "ReactantServerGateway"), nothing
)

# The packages a child may need, by name and UUID. They are also this package's weakdeps, whose
# compat bounds keep an environment holding all three on versions that agree.
const _CHILD_PACKAGE_UUIDS = Dict(
    "ReactantServer" => "faa8a0e4-1926-4c6b-a0e2-e70699715ecf",
    "ReactantServerGateway" => "b92f1031-7cb7-4fde-89de-b6dd50303018",
)

# Whether the project file declares every package in `packages` as a dependency (by UUID).
function _declares_all(project_file::AbstractString, packages)
    isfile(project_file) || return false
    deps = get(TOML.parsefile(project_file), "deps", Dict{String, Any}())
    return all(p -> get(deps, p, nothing) == _CHILD_PACKAGE_UUIDS[p], packages)
end

_has_members(root::AbstractString, packages) =
    all(p -> isfile(joinpath(root, "packages", p, "Project.toml")), packages)

"""
    child_environment(packages; env=ENV, workspace_root=nothing,
                      active_project=Base.active_project()) -> ChildEnvironment

Where the children that need `packages` (a subset of `"ReactantServer"`,
`"ReactantServerGateway"`) run from. The first match wins:

1. `workspace_root`, else `REACTANT_WORKSPACE_ROOT`: the workspace shape at that root. The
   container image sets the variable.
2. The active environment, when it declares every package in `packages`: the environment shape.
   This is the Pkg workflow, with ReactantServerNode, ReactantServer, and ReactantServerGateway
   added to one environment.
3. A checkout around the active project: the workspace shape at the project's own directory
   (`--project=.`) or three levels up (`--project=packages/ReactantServerNode`).

Anything else is a `ConfigError` naming the fixes, rather than a child that dies on startup and
is restarted indefinitely.
"""
function child_environment(
        packages; env::AbstractDict = ENV,
        workspace_root::Union{AbstractString, Nothing} = nothing,
        active_project::Union{AbstractString, Nothing} = Base.active_project()
    )
    root = workspace_root !== nothing ? workspace_root : get(env, "REACTANT_WORKSPACE_ROOT", nothing)
    if root !== nothing
        _has_members(root, packages) && return workspace_environment(root)
        throw(
            ConfigError(
                "workspace root $(repr(root)) has no $(join(packages, " or ")) project under " *
                    "packages/; point REACTANT_WORKSPACE_ROOT at the root of a ReactantServer.jl checkout"
            )
        )
    end
    if active_project !== nothing
        if _declares_all(active_project, packages)
            dir = dirname(active_project)
            return ChildEnvironment(dir, dir, string("@", Sys.iswindows() ? ';' : ':', "@stdlib"))
        end
        for root in (dirname(active_project), dirname(dirname(dirname(active_project))))
            _has_members(root, packages) && return workspace_environment(root)
        end
    end
    throw(
        ConfigError(
            "the supervisor cannot find $(join(packages, " and ")) to run its children: add " *
                "$(join(packages, " and ")) to the environment ReactantServerNode runs in, run it " *
                "from the container image or a ReactantServer.jl checkout, or set " *
                "REACTANT_WORKSPACE_ROOT to the root of a checkout"
        )
    )
end

# PR_SET_PDEATHSIG: the kernel SIGTERMs the child if the supervisor dies, however it dies
# (including SIGKILL, which no userspace forwarding can survive). Linux-only; a no-op guard
# elsewhere.
const _PDEATHSIG_BOOT = """
Sys.islinux() && ccall(:prctl, Cint, (Cint, Culong, Culong, Culong, Culong), 1, 15, 0, 0, 0)
"""

const _WORKER_BOOT = _PDEATHSIG_BOOT * """
    using ReactantServer
    w = get(ENV, "REACTANT_WORKER_NAME", "")
    ReactantServer.serve(ARGS[1]; worker = isempty(w) ? nothing : w)
    """

# The default per-worker compute-thread cap. `--threads=auto` would size each worker's pool to the
# whole host; with N workers per node that is an N-way oversubscription (each worker, plus its GC
# and host library pools, grabbing every core), which pegs the CPU under load. The supervisor sizes
# each worker to its share of the host (cores ÷ workers), capped here so a very large box does not
# hand any one worker an unhelpfully huge pool.
const _MAX_WORKER_THREADS = 16

# Per-worker compute-thread count: the host's share among the workers, at least 1, capped.
_worker_thread_count(cpu_threads::Integer, nworkers::Integer; cap::Integer = _MAX_WORKER_THREADS) =
    clamp(cpu_threads ÷ max(1, nworkers), 1, cap)

function worker_spec(
        name::AbstractString, node_file::AbstractString,
        device::Union{AbstractString, Nothing}, children::Union{AbstractString, ChildEnvironment};
        compute_threads::Integer = _worker_thread_count(Sys.CPU_THREADS, 1),
        grpc_port::Union{Integer, Nothing} = nothing,
        metrics_port::Union{Integer, Nothing} = nothing,
        node_workers::Integer = 1,
        grace_seconds::Real = 15.0,
        drain_seconds::Real = WORKER_DRAIN_SECONDS,
        liveness::Union{LivenessCheck, Nothing} = nothing,
        dispatch_stall_seconds::Union{Real, Nothing} = nothing
    )
    ce = children isa ChildEnvironment ? children : workspace_environment(children)
    proj = ce.worker_project
    # `--threads=<compute_threads>,1`: a default pool sized to this worker's share of the host for
    # the per-request preprocess/postprocess tasks, plus one interactive thread the scheduler pins
    # its GPU dispatch loop to, so CPU hook work overlaps the serialized GPU execution. The share
    # (cores ÷ workers, capped) avoids the N-way oversubscription that `auto` causes when several
    # workers run on one node. (Base.julia_cmd() carries no thread setting, so this is the sole
    # source.)
    cmd = `$(Base.julia_cmd()) --threads=$(compute_threads),1 --project=$proj -e $_WORKER_BOOT $node_file`
    pairs = Pair{String, String}["REACTANT_WORKER_NAME" => String(name)]
    ce.load_path === nothing || push!(pairs, "JULIA_LOAD_PATH" => ce.load_path)
    # Always set device visibility explicitly: the assigned selector, or empty for a CPU worker,
    # so a container-level CUDA_VISIBLE_DEVICES is never inherited by accident.
    push!(pairs, "CUDA_VISIBLE_DEVICES" => (device === nothing ? "" : String(device)))
    # A sole worker that is the node's public endpoint (no gateway) overrides its node-file ports
    # to the public ones, so the external interface matches the multi-worker gateway's.
    grpc_port === nothing || push!(pairs, "INFERENCE_SERVER_ENDPOINTS_PORT" => string(Int(grpc_port)))
    metrics_port === nothing || push!(pairs, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT" => string(Int(metrics_port)))
    # Without shared host weights a host weight budget is split evenly among the node's workers;
    # the supervisor knows how many there are. An explicit value in the environment wins.
    haskey(ENV, "INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET_WORKERS") ||
        push!(pairs, "INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET_WORKERS" => string(Int(node_workers)))
    # Meta sub-calls run in-process now, so no loopback gRPC endpoint or shared-memory fan-out mesh is
    # injected; each worker builds its own local meta-scratch pool in serve().
    _push_drain!(pairs, drain_seconds)
    # The worker's dispatch-stall limit, which decides when its /healthz says not live.
    dispatch_stall_seconds === nothing || push!(pairs, DISPATCH_STALL_ENV => string(Float64(dispatch_stall_seconds)))
    return ChildSpec(
        String(name), addenv(cmd, pairs...), Float64(grace_seconds), Float64(drain_seconds),
        _WORKER_STAGE, liveness
    )
end

# A child that drains on shutdown gets a pipe on stdin (see supervisor.jl), and the variable that
# tells it end of file there is a drain request with this budget (ReactantServerCore.STDIN_DRAIN_ENV).
function _push_drain!(pairs, drain_seconds::Real)
    drain_seconds > 0 &&
        push!(pairs, ReactantServerCore.STDIN_DRAIN_ENV => string(Float64(drain_seconds)))
    return pairs
end

# The node's public gRPC and metrics ports (where clients and Prometheus connect): the gateway's
# listen ports, default 8001/8002, overridable via REACTANT_GATEWAY_LISTEN_* so a single worker
# bound directly to them stays consistent with the gateway it replaces.
function public_ports(env::AbstractDict = ENV)
    _port(addr, default) = begin
        i = findlast(==(':'), String(addr))
        i === nothing ? default : something(tryparse(Int, String(addr)[(i + 1):end]), default)
    end
    return (
        _port(get(env, "REACTANT_GATEWAY_LISTEN_GRPC", "0.0.0.0:8001"), 8001),
        _port(get(env, "REACTANT_GATEWAY_LISTEN_METRICS", "0.0.0.0:8002"), 8002),
    )
end

const _GATEWAY_BOOT = _PDEATHSIG_BOOT * """
    using ReactantServerGateway
    ReactantServerGateway.serve_gateway(isempty(ARGS) ? nothing : ARGS[1])
    """

"""
    gateway_spec(children; gateway_path=nothing, endpoints=nothing,
                 metrics_endpoints=nothing) -> ChildSpec

The embedded gateway child. With no `gateway_path`, the gateway runs from defaults plus
`REACTANT_GATEWAY_*` environment: `endpoints` (the node's loopback worker addresses) is
synthesized into `REACTANT_GATEWAY_WORKERS`, and `metrics_endpoints` (the workers' metrics
addresses) into `REACTANT_GATEWAY_WORKER_METRICS`, so the gateway's /metrics aggregates every
worker's export. A mounted gateway.yml wins: it is passed through and no endpoint env is set.
The node-wide `rapid_reset_protection` is the exception: it is synthesized into
`REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION` with or without a mounted file, unless that variable
is already set or the file sets `grpc.rapid_reset_protection` itself.
"""
function gateway_spec(
        children::Union{AbstractString, ChildEnvironment};
        gateway_path::Union{AbstractString, Nothing} = nothing,
        endpoints::Union{Vector{String}, Nothing} = nothing,
        metrics_endpoints::Union{Vector{String}, Nothing} = nothing,
        worker_names::Union{Vector{String}, Nothing} = nothing,
        grpc_max_recv::Union{Integer, Nothing} = nothing,
        grpc_max_send::Union{Integer, Nothing} = nothing,
        grpc_error_detail::Union{AbstractString, Nothing} = nothing,
        rapid_reset_protection::Union{Bool, Nothing} = nothing,
        grace_seconds::Real = 10.0,
        drain_seconds::Real = PUBLIC_DRAIN_SECONDS
    )
    ce = children isa ChildEnvironment ? children : workspace_environment(children)
    proj = ce.gateway_project
    cmd = gateway_path === nothing ?
        `$(Base.julia_cmd()) --project=$proj -e $_GATEWAY_BOOT` :
        `$(Base.julia_cmd()) --project=$proj -e $_GATEWAY_BOOT $gateway_path`
    pairs = Pair{String, String}[]
    ce.load_path === nothing || push!(pairs, "JULIA_LOAD_PATH" => ce.load_path)
    # The supervisor co-launches the workers, which compile every model before answering. Under
    # lpt_packing the gateway must wait for all of them before its startup checks pass, so make the
    # embedded gateway wait indefinitely by default (the worker subprocesses are this supervisor's
    # responsibility) rather than fail fast. An explicit env value wins.
    #
    # This is set even with a mounted gateway.yml, unlike the endpoint synthesis below. The wait is a
    # fact about the supervisor's own children, not a piece of gateway configuration, and a file that
    # legitimately owns the endpoint list has no way to express it. Suppressing it here used to leave
    # a mounted-file deployment crash-looping for the whole (potentially hours-long) compile window.
    haskey(ENV, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS") ||
        push!(pairs, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS" => "inf")
    if gateway_path === nothing
        endpoints === nothing || push!(pairs, "REACTANT_GATEWAY_WORKERS" => join(endpoints, ","))
        metrics_endpoints === nothing || isempty(metrics_endpoints) ||
            push!(pairs, "REACTANT_GATEWAY_WORKER_METRICS" => join(metrics_endpoints, ","))
        # Friendly names (worker0..N), index-aligned to the endpoint lists, so the gateway labels its
        # worker series with the same `worker` names the workers self-tag (host:port disappears).
        worker_names === nothing || isempty(worker_names) ||
            push!(pairs, "REACTANT_GATEWAY_WORKER_NAMES" => join(worker_names, ","))
        # Mirror the node-level gRPC message-size limits (global.grpc) into the embedded gateway, so a
        # single node knob configures the workers and the gateway alike. An explicit env value wins.
        # Unlike the startup wait, a mounted gateway.yml can express these itself, so it owns them.
        grpc_max_recv === nothing || haskey(ENV, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") ||
            push!(pairs, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => string(grpc_max_recv))
        grpc_max_send === nothing || haskey(ENV, "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES") ||
            push!(pairs, "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES" => string(grpc_max_send))
        # Likewise how much an INTERNAL error tells clients (global.grpc.error_detail).
        grpc_error_detail === nothing || haskey(ENV, "REACTANT_GATEWAY_GRPC_ERROR_DETAIL") ||
            push!(pairs, "REACTANT_GATEWAY_GRPC_ERROR_DETAIL" => String(grpc_error_detail))
    end
    # The node-wide Rapid-Reset switch (global.grpc.rapid_reset_protection, or
    # REACTANT_RAPID_RESET_PROTECTION on the supervisor). Unlike the settings above it also reaches a
    # gateway run from a mounted gateway.yml, since it is meant to cover the whole node at once, but
    # only when that file does not set grpc.rapid_reset_protection itself: the gateway's environment
    # wins over its file, so the variable would silently override what the operator wrote there. An
    # explicit env value wins too.
    rapid_reset_protection === nothing ||
        haskey(ENV, "REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION") ||
        _gateway_file_sets(gateway_path, "grpc", "rapid_reset_protection") ||
        push!(pairs, "REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION" => string(rapid_reset_protection))
    _push_drain!(pairs, drain_seconds)
    isempty(pairs) || (cmd = addenv(cmd, pairs...))
    return ChildSpec("gateway", cmd, Float64(grace_seconds), Float64(drain_seconds), _GATEWAY_STAGE, nothing)
end

"""
    worker_endpoints(node; host="127.0.0.1") -> Vector{String}

The node's worker gRPC addresses (`host:port` per worker, ports via the node's base_port math),
as fed to the embedded gateway.
"""
function worker_endpoints(node::AbstractDict; host::AbstractString = "127.0.0.1")
    ws = _node_workers(node)
    return String["$host:$(_worker_port(node, w, i - 1))" for (i, w) in enumerate(ws)]
end

"""
    worker_metrics_endpoints(node; host="127.0.0.1") -> Vector{String}

The node's worker metrics addresses (`metrics_base_port + i` per worker); empty when the node
has no `metrics_base_port`.
"""
function worker_metrics_endpoints(node::AbstractDict; host::AbstractString = "127.0.0.1")
    mbp = get(node, "metrics_base_port", nothing)
    mbp isa Integer || return String[]
    return String["$host:$(Int(mbp) + i - 1)" for i in 1:length(_node_workers(node))]
end

# Whether the gateway file at `path` sets the key at the nested `keys`. A file the supervisor cannot
# read or parse counts as setting it, so nothing is synthesized over it; the gateway reports the
# file's problem itself.
function _gateway_file_sets(path::Union{AbstractString, Nothing}, keys::AbstractString...)
    path === nothing && return false
    raw = try
        YAML.load_file(path; dicttype = Dict{String, Any})
    catch
        return true
    end
    for k in keys
        (raw isa AbstractDict && haskey(raw, k)) || return false
        raw = raw[k]
    end
    return true
end

# The gateway's listen ports (for the worker-port collision warning): defaults 8001/8002,
# overridden by a mounted gateway.yml's listen block, overridden by REACTANT_GATEWAY_LISTEN_*.
function _gateway_listen_ports(gateway_path::Union{AbstractString, Nothing}, env::AbstractDict)
    listen = Dict{String, Any}()
    if gateway_path !== nothing && isfile(gateway_path)
        raw = try
            YAML.load_file(gateway_path; dicttype = Dict{String, Any})
        catch
            nothing
        end
        raw isa AbstractDict && get(raw, "listen", nothing) isa AbstractDict &&
            (listen = raw["listen"])
    end
    g = get(env, "REACTANT_GATEWAY_LISTEN_GRPC", get(listen, "grpc", "0.0.0.0:8001"))
    m = get(env, "REACTANT_GATEWAY_LISTEN_METRICS", get(listen, "metrics", "0.0.0.0:8002"))
    ports = Set{Int}()
    for addr in (g, m)
        i = findlast(==(':'), String(addr))
        i === nothing && continue
        p = tryparse(Int, String(addr)[(i + 1):end])
        p === nothing || push!(ports, p)
    end
    return ports
end
