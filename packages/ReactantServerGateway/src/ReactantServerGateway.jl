# Pure-Julia gateway: a KServe V2 gRPC reverse proxy in front of one or many ReactantServer nodes.
# It terminates client gRPC on one endpoint, reads model_name out of each ModelInferRequest
# (partial decode only), and forwards the raw protobuf bytes to a worker that serves the model,
# with round-robin load balancing and failover across replicas. SHM register/unregister fan out
# to every worker. The gateway is configured by its own gateway.yml (listen addresses and a flat
# list of worker endpoints across any number of nodes); it autodiscovers which models each
# endpoint serves via RepositoryIndex and refreshes its routing table on the health-probe tick,
# so it never reads a node file. Prometheus metrics plus health endpoints are served on a separate
# admin port. This replaces the former Go module under gateway/.
#
# The gateway references nothing from the runtime/Reactant side; it speaks only the KServe
# GRPCInferenceService (forwarding plus RepositoryIndex discovery).
module ReactantServerGateway

import ProtoBuf as PB
import gRPCServer
import gRPCClient
import HTTP
import Prometheus
import YAML

using ReactantServerCore
using ReactantServerCore.inference   # KServe message types in scope for the gRPC stubs
using ReactantServerCore.control     # control-plane message types (lpt_packing cost polling)

# Internal config/parsing helpers reused from ReactantServerCore (not exported). The node-file
# helpers are used only by `probe_worker_ready`, the worker container's healthcheck.
import ReactantServerCore:
    _node_workers,
    _worker_name,
    _worker_port,
    _subdict,
    _opt,
    _set_nested!,
    _parse_env,
    _parse_env_var

# gRPC service stubs: client stubs for forwarding to workers, server stubs for terminating the
# client connection. Core ships the files but does not compile them; included here so the bare
# message-type references resolve and the gRPC imports run against this package's deps.
include(ReactantServerCore.inference_client_stubs_path())
include(ReactantServerCore.inference_server_stubs_path())
include(ReactantServerCore.control_client_stubs_path())

include("headers.jl")
include("config.jl")
include("routing.jl")
include("metrics.jl")
include("client.jl")
include("refresh.jl")
include("lpt_packing.jl")
include("server.jl")
include("health.jl")

"""
    RunningGateway

Handle to a gateway started with `serve_gateway(...; blocking=false)`. Pass it to [`stop!`](@ref)
to shut the gRPC server, the readiness prober, and the admin HTTP server down.
"""
struct RunningGateway{S}
    cfg::GatewayConfig
    pool::ClientPool
    routes::DiscoveredRoutes
    gate::RegisterGate
    metrics::GatewayMetrics
    admin::AdminServer
    prober::HealthProber
    server::S          # the gRPC server handle; type inferred from gRPCServer.serve!
end

# HTTP/2 receive flow-control windows the gateway advertises to its clients. The protocol default
# is only 64 KiB per stream and per connection, which throttles a large inline-tensor *upload*
# (the client's inference request) to ~window/RTT. Match the gRPC client (libcurl), which
# advertises a large receive window by default, so the receive path is not the bottleneck; the
# connection window also bounds total in-flight DATA across streams. Hardcoded for now; intended
# to become config/env-tunable later.
const _H2_INITIAL_WINDOW_BYTES = 32 * 1024 * 1024     # per-stream receive window
const _H2_CONNECTION_WINDOW_BYTES = 32 * 1024 * 1024  # connection-level receive window

# How long lpt_packing startup waits for all workers to come up over the control plane before
# serving (REACTANT_GATEWAY_STARTUP_WAIT_SECONDS). Default 0 fails fast; "inf"/"forever"/"-1" waits
# indefinitely. The node supervisor sets this to wait for the workers it co-launches (workers
# compile before they answer), so the embedded gateway gates on them instead of crash-looping.
function _startup_wait_seconds()
    s = lowercase(strip(get(ENV, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS", "0")))
    s in ("inf", "forever", "-1") && return Inf
    return something(tryparse(Float64, s), 0.0)
end

"""
    serve_gateway(gateway_path=nothing; blocking=true) -> nothing | RunningGateway

Load `gateway.yml` (listen addresses and the worker endpoint list), build the worker client pool,
start the admin HTTP server and the readiness/discovery prober (which probes each endpoint's
ServerReady and RepositoryIndex and swaps in the discovered routing table), and serve the KServe
gRPC proxy. When `blocking` is false the server runs in the background and a [`RunningGateway`](@ref)
is returned (stop it with [`stop!`](@ref)).

`gateway_path` may be omitted (or `nothing`) to configure the gateway from defaults and
`REACTANT_GATEWAY_*` environment variables alone; the endpoint list then comes from
`REACTANT_GATEWAY_WORKERS`. The node supervisor uses this to run an embedded gateway without a
gateway.yml.
"""
function serve_gateway(gateway_path::Union{AbstractString,Nothing} = nothing; blocking::Bool = true)
    cfg = load_gateway(gateway_path)
    pool = ClientPool(cfg)
    routes = DiscoveredRoutes()
    gate = RegisterGate()
    metrics = GatewayMetrics()
    refresher = RouteRefresher(pool, routes, metrics)
    # LPT-packing scheduling: hard startup preconditions (all workers reachable, FIFO discipline,
    # identical model sets), then an initial rebalance so the first requests already route by
    # packing rather than waiting a prober tick.
    packing = nothing
    if cfg.scheduling_mode == "lpt_packing"
        verify_lpt_packing_preconditions!(pool; wait_seconds = _startup_wait_seconds())
        packing = LptPackingState(cfg)
        @info "gateway scheduling: lpt_packing" rebalance_compute_seconds = cfg.rebalance_compute_seconds min_rebalance_seconds = cfg.min_rebalance_seconds default_replicas = (cfg.default_replicas == REPLICAS_ALL ? "all" : cfg.default_replicas) routing_policy = cfg.routing_policy
    end
    state = GatewayState(pool, routes, gate, metrics, refresher, packing)

    admin = start_admin(metrics, cfg.listen_metrics; worker_metrics = cfg.worker_metrics)
    # Discover routes once synchronously so the table is populated before we accept traffic;
    # reachable endpoints are routable immediately, and the prober refreshes the rest.
    try
        table = discover_routes(pool)
        swap_table!(routes, table)
        set_routing_size!(metrics, nmodels(table))
    catch e
        @warn "initial route discovery failed; the health prober will retry" exception = e
    end
    packing === nothing || rebalance!(packing, pool, copy(pool.order), metrics)
    prober = start_prober!(HealthProber(pool, metrics, admin, routes; packing = packing))
    router = build_gateway_router(state, cfg)

    host, port = _split_hostport(cfg.listen_grpc)
    @info "Starting reactant-gateway" grpc = cfg.listen_grpc metrics = cfg.listen_metrics endpoints = cfg.workers

    if blocking
        gRPCServer.serve(router, host, port; context = state,
            h2_initial_window_size = _H2_INITIAL_WINDOW_BYTES,
            h2_connection_window_size = _H2_CONNECTION_WINDOW_BYTES)
        return nothing
    end
    server = gRPCServer.serve!(router, host, port; context = state,
        h2_initial_window_size = _H2_INITIAL_WINDOW_BYTES,
        h2_connection_window_size = _H2_CONNECTION_WINDOW_BYTES)
    return RunningGateway(cfg, pool, routes, gate, metrics, admin, prober, server)
end

"""
    stop!(g::RunningGateway)

Shut down a gateway started with `serve_gateway(...; blocking=false)`: close the gRPC server,
halt the readiness prober, and close the admin HTTP server.
"""
function stop!(g::RunningGateway)
    close(g.server)
    stop_prober!(g.prober)
    close(g.admin.server)
    close_pool!(g.pool)
    return nothing
end

"""
    probe_worker_ready(node_path, worker=nothing) -> Bool

Resolve a worker's port from the node file and call its KServe `ServerReady` on localhost,
returning whether it reported ready. Used as the worker container's healthcheck (a Julia
replacement for the former Go `reactant-healthprobe`). `worker` may be omitted when the node has a
single worker.
"""
function probe_worker_ready(node_path::AbstractString, worker::Union{AbstractString,Nothing} = nothing)
    node = load_node(node_path)
    names = worker_names(node)
    wname = if worker !== nothing
        String(worker)
    elseif length(names) == 1
        names[1]
    else
        throw(ConfigError("node has $(length(names)) workers; specify which to probe"))
    end
    workers = _node_workers(node)
    idx = findfirst(w -> _worker_name(w) == wname, workers)
    idx === nothing && throw(ConfigError("worker '$wname' not defined in node"))
    port = _worker_port(node, workers[idx], idx - 1)
    client = GRPCInferenceService_ServerReady_Client("127.0.0.1", port; deadline = 5)
    try
        resp = gRPCClient.grpc_sync_request(client, ServerReadyRequest())
        return resp.ready
    catch
        return false
    end
end

export serve_gateway, probe_worker_ready

end # module ReactantServerGateway
