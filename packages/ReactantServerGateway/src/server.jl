# The gateway's gRPC handlers. ModelInfer routes a request to a worker that hosts the model
# (round-robin across replicas, failover on retryable errors). The SHM RPCs fan out to every
# worker because POSIX shared-memory regions live on the host and each worker attaches
# independently. All three are raw Vector{UInt8} in and out; only the small routing headers are
# decoded (see headers.jl). Mirrors the Go gateway's internal/grpcserver.

# Per-request state threaded through gRPCServer's context payload. Parametric on the pool and
# scheduler types so the request hot path (`st.pool` -> `get_clients` -> `wc.infer`, and the
# `select_replicas` dispatch) stays type-stable.
struct GatewayState{P<:ClientPool,S<:GatewayScheduler}
    pool::P
    routes::DiscoveredRoutes
    gate::RegisterGate
    metrics::GatewayMetrics
    refresher::RouteRefresher
    scheduler::S
end

# Map a gateway status string to a server-side gRPC exception with the matching code.
function _server_exc(status::AbstractString, msg::AbstractString)
    status == STATUS_NOT_FOUND && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND, msg)
    status == STATUS_RESOURCE_EXHAUSTED && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_FAILED_PRECONDITION, msg)
    status == STATUS_INTERNAL && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL, msg)
    return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, msg)
end

# --- ModelInfer -------------------------------------------------------------------------------

# `deadline_ns` is the request's absolute deadline in the gateway's clock (0 = none). We recompute the
# remaining budget HERE — the latest point before the worker call, after the register-gate wait and
# any failover retries — and forward it as the worker's grpc-timeout, so the budget shrinks for time
# burned at the gateway instead of resetting to a fresh full budget at the worker. If it is already
# gone, shed here (DEADLINE_EXCEEDED) rather than forward doomed work into the worker's queue.
function _post_infer(st::GatewayState, url, model, id, body, deadline_ns::Int64)
    wc = get_clients(st.pool, url)
    if wc === nothing
        @error "infer: routing table referenced unknown worker" worker = url model
        return nothing, STATUS_INTERNAL,
            gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL, "routing table referenced unknown worker")
    end
    gate_wait(st.gate, url)
    deadline_s = nothing
    if deadline_ns != 0
        rem = deadline_ns - Int64(time_ns())
        if rem <= 0
            return nothing, STATUS_DEADLINE,
                gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_DEADLINE_EXCEEDED,
                    "deadline exceeded at gateway for model \"$model\"")
        end
        # Forward as WHOLE SECONDS: grpc_timeout_header_val only encodes integer seconds without
        # falling back to a >8-digit nanosecond value the worker rejects; ceil so a sub-second budget
        # never becomes 0 (which would disable curl's timeout entirely).
        deadline_s = max(1, cld(rem, 1_000_000_000))
    end
    t0 = time()
    try
        resp = invoke_infer(wc, body; deadline=deadline_s)
        observe_worker!(st.metrics, "ModelInfer", url, time() - t0)
        return resp, STATUS_OK, nothing
    catch e
        observe_worker!(st.metrics, "ModelInfer", url, time() - t0)
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        status = client_status(e)
        @warn "infer: worker error" model worker = url request_id = id status = status msg = e.message
        return nothing, status, _server_exc(status, "worker $url: $(e.message)")
    end
end

# Try the replicas in order (urls[1] is the round-robin choice; the rest are failover targets),
# moving on only for a retryable status (worker NotFound or Unavailable). A worker NOT_FOUND means
# the model was unloaded there since discovery, so kick an async route refresh to drop the stale
# route (the request itself still fails over / returns the error; it is not retried after refresh).
function _try_replicas(st::GatewayState, urls, model, id, body, deadline_ns::Int64)
    last_status = STATUS_UNAVAILABLE
    last_exc = gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, "no replica available")
    for url in urls
        resp, status, exc = _post_infer(st, url, model, id, body, deadline_ns)
        exc === nothing && return resp, status, nothing
        status == STATUS_NOT_FOUND && request_refresh!(st.refresher)
        last_status, last_exc = status, exc
        (status == STATUS_NOT_FOUND || status == STATUS_UNAVAILABLE) && continue
        return nothing, status, exc
    end
    return nothing, last_status, last_exc
end

function _dispatch_infer(st::GatewayState, model, id, body, deadline_ns::Int64)
    # Ask the scheduler which replicas to try (urls[1] is its choice; the rest are failover order)
    # and for an opaque reservation to release once the request completes. `nothing` means the
    # scheduler has no route for the model: it may have just been loaded on a worker, so refresh the
    # routing table on demand (single-flight, rate-limited) and re-select before giving up.
    ctx = ScheduleContext(model, id, st.pool, st.routes, st.metrics, st.refresher)
    sel = select_replicas(st.scheduler, ctx)
    if sel === nothing
        refresh_now!(st.refresher)
        sel = select_replicas(st.scheduler, ctx)
    end
    if sel === nothing
        @info "infer: model not found" model request_id = id
        return nothing, STATUS_NOT_FOUND,
            gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND, "model \"$model\" not found on any worker")
    end
    urls, reservation = sel
    # Release the reservation exactly once, on every path (success, retryable error, hard error, or
    # timeout), so a scheduler's outstanding counters never leak.
    try
        return _try_replicas(st, urls, model, id, body, deadline_ns)
    finally
        release!(st.scheduler, reservation)
    end
end

function _gw_infer(body::Vector{UInt8}, st::GatewayState, deadline_ns::Integer=0)
    t0 = time()
    local model, id
    try
        model, id = peek_model_name_and_id(body)
    catch e
        inc_requests!(st.metrics, "ModelInfer", "", STATUS_INVALID)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT,
            "malformed ModelInferRequest: $(sprint(showerror, e))"))
    end
    if isempty(model)
        inc_requests!(st.metrics, "ModelInfer", "", STATUS_INVALID)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT,
            "ModelInferRequest.model_name is empty"))
    end
    record_arrival!(st.scheduler, model)
    resp, status, exc = _dispatch_infer(st, model, id, body, Int64(deadline_ns))
    observe_request!(st.metrics, "ModelInfer", model, time() - t0)
    inc_requests!(st.metrics, "ModelInfer", model, status)
    exc === nothing || throw(exc)
    return resp
end

# --- System shared memory ---------------------------------------------------------------------

# Serialize a SystemSharedMemoryUnregisterRequest with only `name` set, for rollback.
function _encode_unregister(region::AbstractString)
    io = IOBuffer()
    PB.encode(PB.ProtoEncoder(io), SystemSharedMemoryUnregisterRequest(; name = String(region)))
    return take!(io)
end

# Invoke `op` (a function of WorkerClients) on every worker concurrently, holding the per-worker
# register gate. Returns (first_response, succeeded_urls, failed_urls).
function _broadcast_shm(st::GatewayState, op, region, rpc_label)
    workers = all_clients(st.pool)
    succeeded = String[]
    failed = String[]
    first_body = Ref{Union{Nothing,Vector{UInt8}}}(nothing)
    lk = ReentrantLock()
    @sync for wc in workers
        @async begin
            release = gate_begin!(st.gate, wc.url)
            t0 = time()
            try
                resp = op(wc)
                observe_worker!(st.metrics, rpc_label, wc.url, time() - t0)
                lock(lk) do
                    push!(succeeded, wc.url)
                    first_body[] === nothing && (first_body[] = resp)
                end
            catch e
                observe_worker!(st.metrics, rpc_label, wc.url, time() - t0)
                lock(lk) do
                    push!(failed, wc.url)
                end
                @warn "shm: worker failed" region worker = wc.url exception = e
            finally
                release()
            end
        end
    end
    return first_body[], succeeded, failed
end

function _fanout_unregister(st::GatewayState, urls, body)
    @sync for url in urls
        wc = get_clients(st.pool, url)
        wc === nothing && continue
        @async try
            invoke_shm_unregister(wc, body)
        catch e
            @warn "shm.rollback: worker error" worker = url exception = e
        end
    end
    return nothing
end

function _gw_shm_register(body::Vector{UInt8}, st::GatewayState)
    t0 = time()
    region = ""
    try
        region = peek_shm_name(body)
    catch e
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_INVALID)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT,
            "malformed SystemSharedMemoryRegisterRequest: $(sprint(showerror, e))"))
    end
    if isempty(region)
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_INVALID)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT,
            "SystemSharedMemoryRegisterRequest.name is empty"))
    end

    resp, succeeded, failed = _broadcast_shm(st, wc -> invoke_shm_register(wc, body), region, "SystemSharedMemory")
    observe_request!(st.metrics, "SystemSharedMemoryRegister", region, time() - t0)

    if isempty(failed)
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", region, STATUS_OK)
        @info "shm.register: ok" region workers = succeeded
        return resp === nothing ? UInt8[] : resp
    end

    if !isempty(succeeded)
        @warn "shm.register: partial failure, rolling back" region ok_workers = succeeded failed_workers = failed
        _fanout_unregister(st, succeeded, _encode_unregister(region))
    end
    inc_requests!(st.metrics, "SystemSharedMemoryRegister", region, STATUS_FAILED_PRE)
    throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_FAILED_PRECONDITION,
        "SHM register failed on workers: $(failed)"))
end

function _gw_shm_unregister(body::Vector{UInt8}, st::GatewayState)
    t0 = time()
    region = ""
    try
        region = peek_shm_name(body)
    catch e
        inc_requests!(st.metrics, "SystemSharedMemoryUnregister", "", STATUS_INVALID)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INVALID_ARGUMENT,
            "malformed SystemSharedMemoryUnregisterRequest: $(sprint(showerror, e))"))
    end

    resp, succeeded, failed = _broadcast_shm(st, wc -> invoke_shm_unregister(wc, body), region, "SystemSharedMemory")
    observe_request!(st.metrics, "SystemSharedMemoryUnregister", region, time() - t0)

    if isempty(succeeded) && !isempty(failed)
        @warn "shm.unregister: every worker failed" region failed_workers = failed
        inc_requests!(st.metrics, "SystemSharedMemoryUnregister", region, STATUS_UNAVAILABLE)
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE,
            "SHM unregister failed on all workers: $(failed)"))
    end
    if !isempty(failed)
        @warn "shm.unregister: partial failures" region ok_workers = succeeded failed_workers = failed
    else
        @info "shm.unregister: ok" region workers = succeeded
    end
    inc_requests!(st.metrics, "SystemSharedMemoryUnregister", region, STATUS_OK)
    return resp === nothing ? UInt8[] : resp
end

# --- SHM namespace probe ----------------------------------------------------------------------

# Answer the client's IsSameIPCNamespace probe. Any worker may service a later ModelInfer, so
# system shared memory is only usable if EVERY worker can see the client's region: fan the probe
# out and AND the results. A worker that errors or does not implement the RPC (UNIMPLEMENTED)
# counts as "not same", and a pool with no workers is "not same". The probed name carries a
# random per-client token, so it is never used as a metric label (cardinality).
function _gw_is_same_ipc_namespace(req::IsSameIPCNamespaceRequest, st::GatewayState)
    t0 = time()
    workers = all_clients(st.pool)
    same = if isempty(workers)
        false
    else
        results = Vector{Bool}(undef, length(workers))
        @sync for (i, wc) in enumerate(workers)
            @async begin
                results[i] = try
                    invoke_is_same_ipc_namespace(wc, req).same
                catch e
                    @warn "is_same_ipc_namespace: worker probe failed" worker = wc.url exception = e
                    false
                end
            end
        end
        all(results)
    end
    observe_request!(st.metrics, "IsSameIPCNamespace", "", time() - t0)
    inc_requests!(st.metrics, "IsSameIPCNamespace", "", STATUS_OK)
    return IsSameIPCNamespaceResponse(; same = same)
end

# --- Memory compaction ------------------------------------------------------------------------

# Fan CompactMemory out to a set of workers concurrently, each with its own reload list, keyed by
# worker URL (no register gate: compaction is not a SHM region op). Returns (total_reloaded,
# succeeded_urls, failed_urls). One slow or failing worker is isolated to its own @async branch and
# never aborts the others. Shared by the operator RPC (`_gw_compact`, all workers, one list) and the
# placement-driven path (lpt_packing's `_maybe_compact_fleet!`, changed workers, per-worker lists).
function _compact_workers(pool::ClientPool, metrics::Union{GatewayMetrics,Nothing}, perworker::AbstractDict)
    succeeded = String[]
    failed = String[]
    total = Ref(0)
    lk = ReentrantLock()
    @sync for (url, reload) in perworker
        wc = get_clients(pool, url)
        wc === nothing && continue
        @async begin
            t0 = time()
            try
                resp = invoke_compact(wc, CompactMemoryRequest(; reload_models = collect(String, reload)))
                metrics === nothing || observe_worker!(metrics, "CompactMemory", url, time() - t0)
                lock(lk) do
                    push!(succeeded, url)
                    total[] += Int(resp.reloaded_models)
                end
            catch e
                metrics === nothing || observe_worker!(metrics, "CompactMemory", url, time() - t0)
                lock(lk) do
                    push!(failed, url)
                end
                @warn "compact: worker failed" worker = url exception = e
            end
        end
    end
    return total[], succeeded, failed
end

function _gw_compact(req::CompactMemoryRequest, st::GatewayState)
    t0 = time()
    reload = collect(String, req.reload_models)
    perworker = Dict(wc.url => reload for wc in all_clients(st.pool))
    total, succeeded, failed = _compact_workers(st.pool, st.metrics, perworker)
    observe_request!(st.metrics, "CompactMemory", "", time() - t0)

    if isempty(succeeded) && !isempty(failed)
        inc_requests!(st.metrics, "CompactMemory", "", STATUS_UNAVAILABLE)
        @warn "compact: every worker failed" failed_workers = failed
        throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE,
            "compaction failed on all workers: $(failed)"))
    end
    isempty(failed) ? (@info "compact: ok" workers = succeeded reloaded = total) :
        (@warn "compact: partial failures" ok_workers = succeeded failed_workers = failed reloaded = total)
    inc_requests!(st.metrics, "CompactMemory", "", STATUS_OK)
    return CompactMemoryResponse(; reloaded_models = Int64(total))
end

# --- Router -----------------------------------------------------------------------------------

"""
    build_gateway_router(state, cfg) -> gRPCRouter

Register the forwarded RPCs. ModelInfer and the two SHM register/unregister RPCs are raw
`Vector{UInt8}` methods (bytes pass through unchanged); IsSameIPCNamespace is typed because the
gateway aggregates each worker's answer rather than forwarding one. Every other
GRPCInferenceService RPC is left unimplemented (clients get UNIMPLEMENTED).
"""
function build_gateway_router(state::GatewayState, cfg::GatewayConfig)
    router = gRPCServer.gRPCRouter(; max_receive_message_length = cfg.max_recv_msg_bytes,
        max_send_message_length = cfg.max_send_msg_bytes)
    raw(rpc) = rpc(; TRequest = Vector{UInt8}, TResponse = Vector{UInt8})
    gRPCServer.handle!(router, raw(GRPCInferenceService_ModelInfer_Method),
        (req, ctx) -> _gw_infer(req, ctx.payload, ctx.deadline_ns))
    gRPCServer.handle!(router, raw(GRPCInferenceService_SystemSharedMemoryRegister_Method),
        (req, ctx) -> _gw_shm_register(req, ctx.payload))
    gRPCServer.handle!(router, raw(GRPCInferenceService_SystemSharedMemoryUnregister_Method),
        (req, ctx) -> _gw_shm_unregister(req, ctx.payload))
    # Typed (decoded) so the handler can read each worker's `.same` and aggregate; not forwarded raw.
    gRPCServer.handle!(router, GRPCInferenceService_IsSameIPCNamespace_Method(),
        (req, ctx) -> _gw_is_same_ipc_namespace(req, ctx.payload))
    # Control plane: a single CompactMemory RPC to the gateway fans out to every worker.
    register_ControlService!(router;
        CompactMemory = (req, ctx) -> _gw_compact(req, ctx.payload))
    return router
end
