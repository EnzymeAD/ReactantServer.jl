# Child process specifications: the exact Cmd (program, args, env) for each worker and for the
# embedded gateway. Workers run the unchanged `ReactantServer.serve(node_file; worker=name)`
# entry, exactly like the standalone per-GPU container did; the only supervisor-added environment
# is the worker identity and its single-device CUDA_VISIBLE_DEVICES. Bind hosts stay the node
# file's concern (`endpoints.host`); in a container, which ports are reachable is decided by
# what the operator publishes, and worker metrics ports must remain scrapable.

struct ChildSpec
    name::String
    cmd::Cmd
    grace_seconds::Float64   # SIGTERM-to-SIGKILL window at shutdown
end

"""
    default_workspace_root(env=ENV) -> String

The monorepo root containing `packages/`. `REACTANT_WORKSPACE_ROOT` wins; otherwise it is
derived from the active project, which for the supervisor is
`<root>/packages/ReactantServerNode/Project.toml`.
"""
function default_workspace_root(env::AbstractDict=ENV)
    haskey(env, "REACTANT_WORKSPACE_ROOT") && return String(env["REACTANT_WORKSPACE_ROOT"])
    proj = Base.active_project()
    proj === nothing && throw(ConfigError("no active project; set REACTANT_WORKSPACE_ROOT"))
    return dirname(dirname(dirname(proj)))
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
_worker_thread_count(cpu_threads::Integer, nworkers::Integer; cap::Integer=_MAX_WORKER_THREADS) =
    clamp(cpu_threads ÷ max(1, nworkers), 1, cap)

function worker_spec(name::AbstractString, node_file::AbstractString,
                     device::Union{AbstractString,Nothing}, workspace_root::AbstractString;
                     compute_threads::Integer=_worker_thread_count(Sys.CPU_THREADS, 1),
                     grpc_port::Union{Integer,Nothing}=nothing,
                     metrics_port::Union{Integer,Nothing}=nothing,
                     grace_seconds::Real=15.0)
    proj = joinpath(workspace_root, "packages", "ReactantServer")
    # `--threads=<compute_threads>,1`: a default pool sized to this worker's share of the host for
    # the per-request preprocess/postprocess tasks, plus one interactive thread the scheduler pins
    # its GPU dispatch loop to, so CPU hook work overlaps the serialized GPU execution. The share
    # (cores ÷ workers, capped) avoids the N-way oversubscription that `auto` causes when several
    # workers run on one node. (Base.julia_cmd() carries no thread setting, so this is the sole
    # source.)
    cmd = `$(Base.julia_cmd()) --threads=$(compute_threads),1 --project=$proj -e $_WORKER_BOOT $node_file`
    pairs = Pair{String,String}["REACTANT_WORKER_NAME" => String(name)]
    # Always set device visibility explicitly: the assigned selector, or empty for a CPU worker,
    # so a container-level CUDA_VISIBLE_DEVICES is never inherited by accident.
    push!(pairs, "CUDA_VISIBLE_DEVICES" => (device === nothing ? "" : String(device)))
    # A sole worker that is the node's public endpoint (no gateway) overrides its node-file ports
    # to the public ones, so the external interface matches the multi-worker gateway's.
    grpc_port === nothing || push!(pairs, "INFERENCE_SERVER_ENDPOINTS_PORT" => string(Int(grpc_port)))
    metrics_port === nothing || push!(pairs, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT" => string(Int(metrics_port)))
    # Meta sub-calls run in-process now, so no loopback gRPC endpoint or shared-memory fan-out mesh is
    # injected; each worker builds its own local meta-scratch pool in serve().
    return ChildSpec(String(name), addenv(cmd, pairs...), Float64(grace_seconds))
end

# The node's public gRPC and metrics ports (where clients and Prometheus connect): the gateway's
# listen ports, default 8001/8002, overridable via REACTANT_GATEWAY_LISTEN_* so a single worker
# bound directly to them stays consistent with the gateway it replaces.
function public_ports(env::AbstractDict=ENV)
    _port(addr, default) = begin
        i = findlast(==(':'), String(addr))
        i === nothing ? default : something(tryparse(Int, String(addr)[(i + 1):end]), default)
    end
    return (_port(get(env, "REACTANT_GATEWAY_LISTEN_GRPC", "0.0.0.0:8001"), 8001),
            _port(get(env, "REACTANT_GATEWAY_LISTEN_METRICS", "0.0.0.0:8002"), 8002))
end

const _GATEWAY_BOOT = _PDEATHSIG_BOOT * """
using ReactantServerGateway
ReactantServerGateway.serve_gateway(isempty(ARGS) ? nothing : ARGS[1])
"""

"""
    gateway_spec(workspace_root; gateway_path=nothing, endpoints=nothing,
                 metrics_endpoints=nothing) -> ChildSpec

The embedded gateway child. With no `gateway_path`, the gateway runs from defaults plus
`REACTANT_GATEWAY_*` environment: `endpoints` (the node's loopback worker addresses) is
synthesized into `REACTANT_GATEWAY_WORKERS`, and `metrics_endpoints` (the workers' metrics
addresses) into `REACTANT_GATEWAY_WORKER_METRICS`, so the gateway's /metrics aggregates every
worker's export. A mounted gateway.yml wins: it is passed through and no endpoint env is set.
"""
function gateway_spec(workspace_root::AbstractString;
                      gateway_path::Union{AbstractString,Nothing}=nothing,
                      endpoints::Union{Vector{String},Nothing}=nothing,
                      metrics_endpoints::Union{Vector{String},Nothing}=nothing,
                      grace_seconds::Real=10.0)
    proj = joinpath(workspace_root, "packages", "ReactantServerGateway")
    cmd = gateway_path === nothing ?
          `$(Base.julia_cmd()) --project=$proj -e $_GATEWAY_BOOT` :
          `$(Base.julia_cmd()) --project=$proj -e $_GATEWAY_BOOT $gateway_path`
    if gateway_path === nothing
        pairs = Pair{String,String}[]
        endpoints === nothing || push!(pairs, "REACTANT_GATEWAY_WORKERS" => join(endpoints, ","))
        metrics_endpoints === nothing || isempty(metrics_endpoints) ||
            push!(pairs, "REACTANT_GATEWAY_WORKER_METRICS" => join(metrics_endpoints, ","))
        # The supervisor co-launches the workers, which compile every model before answering. Under
        # lpt_packing the gateway must wait for all of them before its startup checks pass, so make
        # the embedded gateway wait indefinitely by default (the worker subprocesses are this
        # supervisor's responsibility) rather than fail fast. An explicit env value wins.
        haskey(ENV, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS") ||
            push!(pairs, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS" => "inf")
        isempty(pairs) || (cmd = addenv(cmd, pairs...))
    end
    return ChildSpec("gateway", cmd, Float64(grace_seconds))
end

"""
    worker_endpoints(node; host="127.0.0.1") -> Vector{String}

The node's worker gRPC addresses (`host:port` per worker, ports via the node's base_port math),
as fed to the embedded gateway.
"""
function worker_endpoints(node::AbstractDict; host::AbstractString="127.0.0.1")
    ws = _node_workers(node)
    return String["$host:$(_worker_port(node, w, i - 1))" for (i, w) in enumerate(ws)]
end

"""
    worker_metrics_endpoints(node; host="127.0.0.1") -> Vector{String}

The node's worker metrics addresses (`metrics_base_port + i` per worker); empty when the node
has no `metrics_base_port`.
"""
function worker_metrics_endpoints(node::AbstractDict; host::AbstractString="127.0.0.1")
    mbp = get(node, "metrics_base_port", nothing)
    mbp isa Integer || return String[]
    return String["$host:$(Int(mbp) + i - 1)" for i in 1:length(_node_workers(node))]
end

# The gateway's listen ports (for the worker-port collision warning): defaults 8001/8002,
# overridden by a mounted gateway.yml's listen block, overridden by REACTANT_GATEWAY_LISTEN_*.
function _gateway_listen_ports(gateway_path::Union{AbstractString,Nothing}, env::AbstractDict)
    listen = Dict{String,Any}()
    if gateway_path !== nothing && isfile(gateway_path)
        raw = try
            YAML.load_file(gateway_path; dicttype=Dict{String,Any})
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
