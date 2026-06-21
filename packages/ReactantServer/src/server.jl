# Top-level assembly: config, runtime client, compiled bundles, scheduler, gRPC server.
#
# Sequencing: config first (fail fast), then the runtime client (so compilation has a
# device), then bundles compiled and weights pinned, then the dispatch task, then the gRPC
# server last so traffic is accepted only once models and the scheduler are live.

"""
    RunningServer

Handle to a server started with `serve(...; blocking=false)`. It holds the resolved
[`ServerConfig`](@ref), the model registry, the running [`Scheduler`](@ref), the device
memory pool, the shared-memory registry, the underlying gRPC server, and the listen `port`.
Pass it to [`stop!`](@ref) to shut the server down.
"""
struct RunningServer
    config::ServerConfig
    registry::ModelRegistry
    scheduler::Scheduler
    pool::MemoryPool
    shm::SharedMemoryRegistry
    server::Any
    port::Int
    watcher::Union{BundleWatcher,Nothing}
    metrics_server::Any        # HTTP metrics listener, or nothing when metrics_port == 0
end

"""
    stop!(s::RunningServer)

Shut down a server started with `serve(...; blocking=false)`. Stops the model-directory watcher
(if running), closes the metrics endpoint (if running) and the gRPC server, halts the scheduler's
dispatch loop, and tears down any registered shared-memory regions. Returns `nothing`.
"""
function stop!(s::RunningServer)
    s.watcher === nothing || stop_watching!(s.watcher)
    s.metrics_server === nothing || close(s.metrics_server)
    close(s.server)
    shutdown!(s.scheduler)
    shm_teardown!(s.shm)
    return nothing
end

# Resolve a model's initial residency floor identically at startup and on a dynamic load: an
# explicit per-model `residency` wins; otherwise PINNED_SYSTEM when the on-demand cache is enabled
# (the host floor is materialized so an on-demand GPU load is a pure host-to-device transfer) and
# UNPINNED otherwise.
function _resolve_residency(cfg::ServerConfig, name::AbstractString, on_demand::Bool)
    declared = get(cfg.scheduler.models, name, ModelSchedConfig(1.0)).residency
    declared === nothing && return on_demand ? PINNED_SYSTEM : UNPINNED
    return declared
end

# The per-request execution timeout is parsed and validated but not enforced by the dispatch
# loop (a mid-execution watchdog is out of scope). Warn operators who tuned it so the silence
# is not mistaken for an effect.
function _warn_unenforced_config(cfg::ServerConfig)
    cfg.scheduler.dispatch_timeout_seconds != 30.0 &&
        @warn "scheduler.dispatch_timeout_seconds is set but not enforced (no mid-execution watchdog)" value = cfg.scheduler.dispatch_timeout_seconds
    return nothing
end

function _bring_up(cfg::ServerConfig, backend::AbstractBackend)
    _warn_unenforced_config(cfg)
    pool = resolve_client(backend, cfg.runtime)
    include = isempty(cfg.models_include) ? nothing : cfg.models_include
    registry = load_bundles(cfg.model_dirs; include=include)
    isempty(registry.by_name) && @warn "no model bundles found" model_dirs = cfg.model_dirs models_include = cfg.models_include
    on_demand = cfg.runtime.weight_cache_bytes > 0
    if cfg.runtime.residency_mode == EXTERNALLY_MANAGED && !on_demand
        @warn "externally-managed residency has no effect without the on-demand weight cache; set runtime.weight_cache_bytes > 0 to enable residency control"
    end
    # One host-weight store for the worker: node-shared SHM when opted in, otherwise private.
    store = if cfg.runtime.shared_host_weights
        cfg.runtime.shared_host_weights_mode == 0o666 &&
            @warn "shared host-weight regions are created world-writable (mode 666); set runtime.shared_host_weights_mode: \"660\" for production / multi-user systems"
        SharedWeightStore(mode=cfg.runtime.shared_host_weights_mode)
    else
        PrivateWeightStore()
    end
    # Unspecified residency resolves to system-pinned under the on-demand cache (so every
    # model's weights are materialized into host RAM at startup and an on-demand GPU load is a
    # pure host-to-device transfer) and to unpinned otherwise. An explicit
    # `residency: unpinned` opts a model out of the host floor. The same rule is applied to
    # models loaded later by the directory watcher (see `_resolve_residency`).
    for entry in values(registry.by_name)
        state = _resolve_residency(cfg, entry.name, on_demand)
        @info "Compiling model" name = entry.name residency = state on_demand = on_demand
        entry.executable = build_loaded_model(backend, pool, entry; state=state, on_demand=on_demand, store=store)
    end
    sched = Scheduler(registry, backend, pool, cfg.scheduler)
    if on_demand
        sched.weight_cache = WeightCache(backend, pool, registry, cfg.runtime.weight_cache_bytes;
                                         mode=cfg.runtime.residency_mode, store=store)
    end
    start!(sched)
    # Model lifecycle is governed by model_control_mode. Only `dynamic` runs the filesystem
    # watcher (Triton-style POLL): it polls the model dirs on a background task and hot-swaps
    # bundles via the scheduler's control queue, reusing the residency/on-demand/store decisions
    # made above. `static` keeps the startup set fixed; `explicit` cedes authority to the control
    # plane (externally-managed residency). Neither starts a watcher.
    watcher = nothing
    if cfg.model_control_mode == DYNAMIC
        include_names = isempty(cfg.models_include) ? nothing : cfg.models_include
        watcher = BundleWatcher(sched, backend, pool, cfg;
                                interval=cfg.model_poll_seconds, on_demand=on_demand,
                                store=store, include=include_names)
        start_watching!(watcher)
    end
    @info "models ready" count = length(registry.by_name) control_mode = cfg.model_control_mode memory = memory_report(backend, pool; registry=registry, weight_cache=sched.weight_cache)
    return pool, registry, sched, watcher
end

"""
    serve(node_path; worker=nothing, backend=ReactantBackend(), blocking=true) -> nothing | RunningServer

Load the node config at `node_path`, resolve this process's worker, bring up the runtime and its
assigned models, and start the gRPC control plane. `worker` selects which worker entry to serve;
it may be omitted when the node has exactly one worker. When `blocking` is false the server runs
in the background and a `RunningServer` is returned (stop it with `stop!`).
"""
function serve(node_path::AbstractString; worker::Union{AbstractString,Nothing}=nothing,
               backend::AbstractBackend=ReactantBackend(), blocking::Bool=true)
    node = load_node(node_path)
    cfg, applied, wname = node_server_config(node, worker)
    validate_config(cfg)
    @info "Resolved node worker" worker = wname node = node_path
    log_effective_config(cfg, applied)
    return serve(cfg; backend=backend, blocking=blocking, worker_name=wname)
end

"""
    serve_worker(node_path, worker; backend=ReactantBackend(), blocking=true)

Convenience alias for [`serve`](@ref) that names the worker positionally.
"""
serve_worker(node_path::AbstractString, worker::AbstractString; kwargs...) =
    serve(node_path; worker=worker, kwargs...)

# Build the worker's local meta-scratch pool (plain Memory, use_shm=false) from env, or nothing to
# disable. Defaults: 1 GiB across 16 slots. A scratch buffer larger than one slot spans several.
function _meta_scratch_pool()
    bytes = something(tryparse(Int, strip(get(ENV, "REACTANT_FANOUT_BYTES", ""))), 1 << 30)
    slots = something(tryparse(Int, strip(get(ENV, "REACTANT_FANOUT_SLOTS", ""))), 16)
    (bytes > 0 && slots > 0) || return nothing
    return BufferPool(bytes; n_slots=slots, use_shm=false)
end

# How many GPU-using meta orchestrations may run at once on this worker (REACTANT_META_CONCURRENCY,
# default 2). The committed set tracks this, so one knob sets both how many metas run and how many cut
# the line; behavior stays consistent at any value because all in-flight metas cut the line symmetrically
# (no race for a single slot). At 1 two heavy metas cannot overlap: while one holds the gate the other
# waits, burns its budget, and sheds (measured), so 2 is the floor that lets two heavy pipelines run
# without serializing. Higher favors meta completion via more concurrency, but more metas share the
# serial GPU and the scratch pool (size it accordingly). Compute-only metas bypass the gate entirely.
function _meta_concurrency()
    n = something(tryparse(Int, strip(get(ENV, "REACTANT_META_CONCURRENCY", ""))), 2)
    return max(1, n)
end

function serve(cfg::ServerConfig; backend::AbstractBackend=ReactantBackend(), blocking::Bool=true,
               worker_name::AbstractString="")
    shm = SharedMemoryRegistry()   # client-facing SystemSharedMemory feature (decode/encode shm path)
    pool, registry, sched, watcher = _bring_up(cfg, backend)
    # Local reuse pool for meta-model intermediates: plain Memory, never shared. With one meta at a time
    # (the gate below) the pool sees no cross-meta contention; it keeps large per-request intermediates
    # (e.g. an ROI feature tensor) off the allocation path and holds down GC pressure. Sized by env.
    sched.scratch_pool = _meta_scratch_pool()
    # Admit one meta orchestration at a time (configurable). A meta holds a permit across its whole run,
    # including the CPU glue between stages, but not the GPU itself: its sub-calls dispatch on the loop,
    # so other models run while the meta computes between stages.
    sched.meta_gate = MetaGate(_meta_concurrency())
    # Admission counters shared with the gRPC server: the cap (endpoints.max_concurrent_requests,
    # 0 = uncapped) sheds past `inflight`, counting each rejection in `shed`. The metrics collector
    # reads both live, so they are exported even while `serve` blocks the calling task.
    inflight = Threads.Atomic{Int}(0)
    shed = Threads.Atomic{Int}(0)
    metrics = WorkerMetrics(sched, backend, pool, cfg; worker_name=worker_name,
        inflight=inflight, shed=shed)
    router = build_grpc_router(sched, registry, pool.platform, shm)
    ctx = InferContext(sched, registry, shm, pool.platform, metrics)
    # Optional Prometheus metrics endpoint (opt-in via endpoints.metrics_port > 0). Request counting
    # is always on (the InferContext carries `metrics`); only the HTTP listener is gated.
    metrics_server = nothing
    if cfg.endpoints.metrics_port > 0
        ready_fn = () -> (es = values(registry.by_name);
            (!isempty(es) || !isempty(registry.meta)) && all(e -> e.executable !== nothing, es))
        metrics_server = start_worker_metrics(metrics, cfg.endpoints.host, cfg.endpoints.metrics_port;
            ready_fn=ready_fn, worker_name=worker_name, gpu=_gpu_identity(cfg))
    end
    @info "Starting gRPC control plane" host = cfg.endpoints.host port = cfg.endpoints.port metrics_port = cfg.endpoints.metrics_port max_concurrent_requests = cfg.endpoints.max_concurrent_requests models = model_names(registry)
    if blocking
        _G.serve(router, cfg.endpoints.host, cfg.endpoints.port; context=ctx,
            max_concurrent_requests=cfg.endpoints.max_concurrent_requests,
            inflight=inflight, shed_total=shed,
            h2_initial_window_size=_H2_INITIAL_WINDOW_BYTES,
            h2_connection_window_size=_H2_CONNECTION_WINDOW_BYTES)
        return nothing
    end
    server = _G.serve!(router, cfg.endpoints.host, cfg.endpoints.port; context=ctx,
        max_concurrent_requests=cfg.endpoints.max_concurrent_requests,
        inflight=inflight, shed_total=shed,
        h2_initial_window_size=_H2_INITIAL_WINDOW_BYTES,
        h2_connection_window_size=_H2_CONNECTION_WINDOW_BYTES)
    return RunningServer(cfg, registry, sched, pool, shm, server, cfg.endpoints.port, watcher, metrics_server)
end
