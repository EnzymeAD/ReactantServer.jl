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

# Own generated gRPC codegen (messages + client stubs + registration); protojl output from
# proto_src; never hand-edit the pb content. These modules shadow Core's message-only modules,
# so the gateway's handlers and outbound clients use the gateway-local message types.
include("proto/inference/inference.jl")
include("proto/control/control.jl")
using .inference
using .control
export inference, control

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
    _parse_env_var,
    _parse_error_detail

# gRPC codegen: the generated modules carry client constructors and the per-service
# registration functions, but `using` does not re-export them, so re-export the three
# registration entry points the gateway uses (tests call them qualified and unqualified).
import .inference: register_GRPCInferenceService!
import .control: register_ControlService!, register_GatewayControlService!
export register_GRPCInferenceService!, register_ControlService!, register_GatewayControlService!

include("headers.jl")
include("config.jl")
include("knobs.jl")      # runtime-mutable scheduling knobs; after config.jl (built from GatewayConfig)
include("routing.jl")
include("metrics.jl")
include("client.jl")
include("refresh.jl")
include("routing_meta.jl")   # shared per-model routing metadata; before scheduler.jl (which reads it)
include("scheduler.jl")
include("lpt_packing.jl")
include("server.jl")
include("control_service.jl")   # the gateway's own GatewayControlService handlers
include("health.jl")
include("responsiveness.jl")   # the event-loop lag monitor
include("drain.jl")   # the graceful-drain gate on the gRPC server

"""
    RunningGateway

Handle to a gateway started with `serve_gateway(...; blocking=false)`. Pass it to
[`stop!`](@ref ReactantServer.stop!)
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
    monitor::LoopMonitor
    server::S          # the gRPC server handle (GRPCServer)
    drain::ReactantServerCore.DrainGate   # in-flight ModelInfer count; refuses new ones once draining
end

"""
    bound_ports(g::RunningGateway) -> (; grpc, metrics)

The ports a running gateway is listening on. Configure a listen address with port 0
(`127.0.0.1:0`) to have the OS pick a free one, then read the actual ports back here.
"""
bound_ports(g::RunningGateway) =
    (grpc = something(gRPCServer.bound_port(g.server)), metrics = HTTP.port(g.admin.server))

# HTTP/2 receive flow-control windows the gateway advertises to its clients. The protocol default
# is only 64 KiB per stream and per connection, which throttles a large inline-tensor *upload*
# (the client's inference request) to ~window/RTT. Match the gRPC client (libcurl), which
# advertises a large receive window by default, so the receive path is not the bottleneck; the
# connection window also bounds total in-flight DATA across streams. Hardcoded for now; intended
# to become config/env-tunable later.
const _H2_INITIAL_WINDOW_BYTES = 32 * 1024 * 1024     # per-stream receive window
const _H2_CONNECTION_WINDOW_BYTES = 32 * 1024 * 1024  # connection-level receive window

# The streams a client connection may have open at once (SETTINGS_MAX_CONCURRENT_STREAMS) by default:
# 2^31 - 1, more than a client can ever open on one connection (its stream ids are odd and below
# 2^31). It is this large because HTTP.jl derives its Rapid-Reset (CVE-2023-44487) budget from it:
# once the peer has reset more than `max_concurrent_streams + 100` streams over the connection's whole
# life, the server sends GOAWAY(ENHANCE_YOUR_CALM). Every call a client gives up on (its deadline
# passed, or it cancelled) resets one stream, so at gRPCServer's default of 100 an ordinary client is
# cut off after its 200th give-up. libcurl, which gRPCClient and so ReactantServerClient drive with
# CURLOPT_PIPEWAIT, then parks every later call behind the replacement connection until the call's
# deadline when that connection's first calls were give-ups too: it waits for a connection to show it
# multiplexes before reusing it, and one that never answered a call never does, until it has been
# idle for 118 s and libcurl drops it (GW-18). With this limit the budget is out of reach. What bounds
# a client that resets streams is the gateway's inbound admission cap (`inbound_cap`, unless it is
# configured to 0): a reset call's handler counts against it until it returns, and a call past it is
# shed at once with RESOURCE_EXHAUSTED. `grpc.rapid_reset_protection` restores gRPCServer's default
# limit, and with it the budget, for an operator who prefers HTTP.jl's protection to clients that
# keep their connection (see `_listener_streams`).
const _H2_MAX_CONCURRENT_STREAMS = Int(typemax(Int32))

# How long lpt_packing startup waits for all workers to come up over the control plane before
# serving (REACTANT_GATEWAY_STARTUP_WAIT_SECONDS). "inf"/"forever"/"-1" waits indefinitely and 0
# checks once. The default is a short grace rather than a single check: a worker that already
# answers /readyz still compiles its gRPC request path on its first call (about 4s on a fast
# machine, past the 5s probe deadline on a slow or cold one), so one check can reject a healthy
# worker. The node supervisor sets "inf" to wait for the workers it co-launches (workers compile
# before they answer), so the embedded gateway gates on them instead of crash-looping.
const _DEFAULT_STARTUP_WAIT_SECONDS = 60.0

function _startup_wait_seconds()
    s = lowercase(strip(get(ENV, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS", "")))
    isempty(s) && return _DEFAULT_STARTUP_WAIT_SECONDS
    s in ("inf", "forever", "-1") && return Inf
    return something(tryparse(Float64, s), _DEFAULT_STARTUP_WAIT_SECONDS)
end

"""
    serve_gateway(gateway_path=nothing; blocking=true) -> nothing | RunningGateway

Load `gateway.yml` (listen addresses and the worker endpoint list), build the worker client pool,
start the admin HTTP server and the readiness/discovery prober (which probes each endpoint's
ServerReady and RepositoryIndex and swaps in the discovered routing table), and serve the KServe
gRPC proxy. When `blocking` is false the server runs in the background and a
[`RunningGateway`](@ref) is returned (stop it with [`stop!`](@ref ReactantServer.stop!)). A
blocking gateway serves until a drain is requested (`request_drain!`, or under the node supervisor
end of file on stdin), then drains, shuts down and returns.

`gateway_path` may be omitted (or `nothing`) to configure the gateway from defaults and
`REACTANT_GATEWAY_*` environment variables alone; the endpoint list then comes from
`REACTANT_GATEWAY_WORKERS`. The node supervisor uses this to run an embedded gateway without a
gateway.yml.
"""
function serve_gateway(gateway_path::Union{AbstractString, Nothing} = nothing; blocking::Bool = true)
    # Under the node supervisor, end of file on stdin is the request to drain (see
    # ReactantServerCore's drain.jl). Watch from the start: a request during startup exits at once.
    blocking && ReactantServerCore.watch_stdin_drain()
    cfg = load_gateway(gateway_path)
    pool = ClientPool(cfg)
    routes = DiscoveredRoutes(; retention = cfg.route_retention_seconds)
    gate = RegisterGate()
    # Map every worker endpoint url (gRPC and metrics, index-aligned to worker_names) to its friendly
    # name so gateway series read worker0..N instead of host:port and join the workers' own labels.
    name_by_url = Dict{String, String}()
    for (i, u) in enumerate(cfg.workers)
        i <= length(cfg.worker_names) && (name_by_url[u] = cfg.worker_names[i])
    end
    for (i, u) in enumerate(cfg.worker_metrics)
        i <= length(cfg.worker_names) && (name_by_url[u] = cfg.worker_names[i])
    end
    metrics = GatewayMetrics(name_by_url)
    register_process_metrics!(metrics)
    # Measure the event loop from the start, so a stall during startup (compilation, the initial
    # discovery) is recorded too.
    monitor = start_loop_monitor!(LoopMonitor(metrics))
    refresher = RouteRefresher(pool, routes, metrics)
    # Build the gateway scheduler for the configured mode and run its startup hook before anything
    # else starts: lpt_packing verifies hard preconditions (all workers reachable, FIFO discipline,
    # identical model sets, raising on violation) and does an initial rebalance so the first requests
    # already route by packing; round_robin / least_outstanding are no-ops here. Run before the admin
    # server starts so a precondition failure leaves nothing running.
    # Per-model routing metadata (batch axis, measured cost per item), refreshed by the prober from
    # one control-status poll per round and read by whichever scheduler routes by work.
    meta = RoutingMeta(cfg)
    scheduler = make_scheduler(cfg, meta)
    scheduler_start!(scheduler, pool, metrics)
    # The work basis is logged where it applies, so an operator can confirm from the startup line what
    # "least busy" means on this fleet. Under lpt_packing it is also runtime-mutable and reported by
    # GetSchedulingStatus; under least_outstanding this line is the only place it is stated.
    cfg.scheduling_mode == "round_robin" ?
        (@info "gateway scheduling" mode = cfg.scheduling_mode) :
        (@info "gateway scheduling" mode = cfg.scheduling_mode work_basis = cfg.work_basis)
    state = GatewayState(
        pool, routes, gate, metrics, refresher, scheduler, cfg.error_detail, worker_labels(cfg)
    )

    admin = start_admin(metrics, cfg.listen_metrics; worker_metrics = cfg.worker_metrics)
    # Discover routes once synchronously so the table is populated before we accept traffic;
    # reachable endpoints are routable immediately, and the prober refreshes the rest.
    try
        table = discover_routes!(routes, pool)
        set_routing_size!(metrics, nmodels(table))
    catch e
        @warn "initial route discovery failed; the health prober will retry" exception = e
    end
    prober = start_prober!(HealthProber(pool, metrics, admin, routes; scheduler = scheduler, meta = meta, gate = gate))
    host, port = _split_hostport(cfg.listen_grpc)
    # Inbound admission cap: a per-worker multiple of the configured budget, scaled by the fleet
    # size so the gateway sheds (RESOURCE_EXHAUSTED) only past what the workers can plausibly absorb.
    # 0 (per-worker = 0) disables the cap. `build_gateway_server` gives it to the gRPC server, whose
    # own admission atomics the admission metrics read live.
    max_concurrent = inbound_cap(cfg)
    server = build_gateway_server(state, cfg, host, port)
    register_admission!(metrics, server.inflight, server.shed_total, max_concurrent)
    # Refuses new inference with UNAVAILABLE once the gateway drains.
    drain = ReactantServerCore.DrainGate()
    gRPCServer.add_interceptor!(server, _DrainInterceptor(drain))
    @info "Starting reactant-gateway" grpc = cfg.listen_grpc metrics = cfg.listen_metrics endpoints = cfg.workers max_concurrent_requests = max_concurrent outbound_streams_per_worker = cfg.max_concurrent_streams_per_worker rapid_reset_protection = cfg.rapid_reset_protection

    blocking || return _start_gateway(cfg, pool, routes, gate, metrics, admin, prober, monitor, server, drain)
    # Blocking: serve until a drain is requested (the supervisor closing stdin, or `request_drain!`),
    # then drain and tear everything down, so the process exits 0. Interrupting the wait (Ctrl-C at a
    # REPL) stops at once, without draining.
    ReactantServerCore.begin_serving!()
    try
        g = _start_gateway(cfg, pool, routes, gate, metrics, admin, prober, monitor, server, drain)
        secs = 0.0
        try
            secs = ReactantServerCore.wait_for_drain_request()
        catch err
            err isa InterruptException || rethrow()
        finally
            stop!(g; drain_seconds = secs)
        end
    finally
        ReactantServerCore.end_serving!()
    end
    return nothing
end

function _start_gateway(cfg, pool, routes, gate, metrics, admin, prober, monitor, server, drain)
    gRPCServer.start!(server)
    return RunningGateway(cfg, pool, routes, gate, metrics, admin, prober, monitor, server, drain)
end

"""
    stop!(g::RunningGateway; drain_seconds=0)

Shut down a gateway started with `serve_gateway(...; blocking=false)`: halt the readiness prober,
report not ready on `/readyz`, refuse new ModelInfer calls with UNAVAILABLE, close the gRPC
server, and close the admin HTTP server. With `drain_seconds > 0` the calls already forwarded get
up to `drain_seconds` to finish before the gRPC server closes.
"""
function stop!(g::RunningGateway; drain_seconds::Real = 0)
    # Refuse new inference (UNAVAILABLE) and report not ready from here on; with a drain budget, first
    # let the forwarded calls finish. The prober stops first so it cannot report the gateway ready
    # again, and /readyz is turned down on every check while the drain waits (a probe round already
    # under way may still set it).
    stop_prober!(g.prober)
    set_ready!(g.admin, false)
    ReactantServerCore.drain_inflight!(
        g.drain, drain_seconds; what = "gateway", poll = () -> set_ready!(g.admin, false)
    )
    close(g.server)
    stop_loop_monitor!(g.monitor)
    _close_http_bounded(g.admin.server)
    close_pool!(g.pool)
    return nothing
end

# `close(::HTTP.Server)` waits until every connection is idle, without bound; a scrape in progress
# (or a stalled client) would hold shutdown. Close gracefully for a bounded time, then force it.
function _close_http_bounded(server; seconds::Real = 5.0)
    closing = Threads.@spawn try
        close(server)
    catch
    end
    if timedwait(() -> istaskdone(closing), Float64(seconds); pollint = 0.05) !== :ok
        @warn "stop: the admin listener did not close within $(seconds)s; forcing it"
        try
            HTTP.forceclose(server)
        catch
        end
    end
    return nothing
end

"""
    probe_worker_ready(node_path, worker=nothing) -> Bool

Resolve a worker's port from the node file and call its KServe `ServerReady` on localhost,
returning whether it reported ready. Used as the worker container's healthcheck (a Julia
replacement for the former Go `reactant-healthprobe`). `worker` may be omitted when the node has a
single worker.
"""
function probe_worker_ready(node_path::AbstractString, worker::Union{AbstractString, Nothing} = nothing)
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
    try
        resp = GRPCInferenceService.ServerReady("127.0.0.1", port, ServerReadyRequest(); deadline = 5)
        return resp.ready
    catch
        return false
    end
end

export serve_gateway, probe_worker_ready

# Build-time workload: the real gateway against an in-process mock worker; see the file.
include("precompile.jl")

end # module ReactantServerGateway
