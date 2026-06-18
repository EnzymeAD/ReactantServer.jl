# The gateway's gRPC handlers. ModelInfer routes a request to a worker that hosts the model
# (round-robin across replicas, failover on retryable errors). The SHM RPCs fan out to every
# worker because POSIX shared-memory regions live on the host and each worker attaches
# independently. All three are raw Vector{UInt8} in and out; only the small routing headers are
# decoded (see headers.jl). Mirrors the Go gateway's internal/grpcserver.

# Per-request state threaded through gRPCServer's context payload. Parametric on the pool type so
# the request hot path (`st.pool` -> `get_clients` -> `wc.infer`) stays type-stable.
struct GatewayState{P<:ClientPool}
    pool::P
    routes::DiscoveredRoutes
    gate::RegisterGate
    metrics::GatewayMetrics
    refresher::RouteRefresher
    packing::Union{LptPackingState,Nothing}   # nothing in round_robin mode
end

# Map a gateway status string to a server-side gRPC exception with the matching code.
function _server_exc(status::AbstractString, msg::AbstractString)
    status == STATUS_NOT_FOUND && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND, msg)
    status == STATUS_RESOURCE_EXHAUSTED && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_FAILED_PRECONDITION, msg)
    status == STATUS_INTERNAL && return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL, msg)
    return gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, msg)
end

# --- ModelInfer -------------------------------------------------------------------------------

function _post_infer(st::GatewayState, url, model, id, body)
    wc = get_clients(st.pool, url)
    if wc === nothing
        @error "infer: routing table referenced unknown worker" worker = url model
        return nothing, STATUS_INTERNAL,
            gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL, "routing table referenced unknown worker")
    end
    gate_wait(st.gate, url)
    t0 = time()
    try
        resp = invoke_infer(wc, body)
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
function _try_replicas(st::GatewayState, urls, model, id, body)
    last_status = STATUS_UNAVAILABLE
    last_exc = gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, "no replica available")
    for url in urls
        resp, status, exc = _post_infer(st, url, model, id, body)
        exc === nothing && return resp, status, nothing
        status == STATUS_NOT_FOUND && request_refresh!(st.refresher)
        last_status, last_exc = status, exc
        (status == STATUS_NOT_FOUND || status == STATUS_UNAVAILABLE) && continue
        return nothing, status, exc
    end
    return nothing, last_status, last_exc
end

# Try a model's placement replicas in order (urls[1] is the routing choice; the rest are failover),
# then, if every placement replica returns a retryable error, the remaining discovered replicas as a
# last resort so a concentrated model survives its worker dying between repacks. The reservation made
# by route_replica is released exactly once here, on every path (success, retryable error, hard
# error, or timeout), so the outstanding counter never leaks.
function _try_packing(st::GatewayState, urls, counters, model, id, body)
    try
        last_status = STATUS_UNAVAILABLE
        last_exc = gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, "no replica available")
        for url in urls
            resp, status, exc = _post_infer(st, url, model, id, body)
            exc === nothing && return resp, status, nothing
            status == STATUS_NOT_FOUND && request_refresh!(st.refresher)
            last_status, last_exc = status, exc
            (status == STATUS_NOT_FOUND || status == STATUS_UNAVAILABLE) || return nothing, status, exc
        end
        # Last-resort failover to replicas outside the placement (untracked by the fill counters).
        rr = pick(st.routes, model)
        if rr !== nothing
            extra = String[u for u in rr if !(u in urls)]
            isempty(extra) || return _try_replicas(st, extra, model, id, body)
        end
        return nothing, last_status, last_exc
    finally
        _release_route!(counters)
    end
end

function _dispatch_infer(st::GatewayState, model, id, body)
    # LPT-packing mode: route to the replica that fills its batch first (route_replica reserves it),
    # the remaining placement replicas following as failover. A model without a placement yet (cold,
    # or new since the last repack) falls through to the round-robin path below.
    if st.packing !== nothing
        routed = route_replica(st.packing, model)
        if routed !== nothing
            urls, counters = routed
            return _try_packing(st, urls, counters, model, id, body)
        end
    end
    urls = pick(st.routes, model)
    if urls === nothing
        # The model is unknown to the routing table. It may have just been loaded on a worker, so
        # refresh on demand (single-flight, rate-limited) and re-pick before giving up.
        refresh_now!(st.refresher)
        urls = pick(st.routes, model)
    end
    if urls === nothing
        @info "infer: model not found" model request_id = id
        return nothing, STATUS_NOT_FOUND,
            gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND, "model \"$model\" not found on any worker")
    end
    return _try_replicas(st, urls, model, id, body)
end

function _gw_infer(body::Vector{UInt8}, st::GatewayState)
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
    st.packing === nothing || record_arrival!(st.packing, model)
    resp, status, exc = _dispatch_infer(st, model, id, body)
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

# --- Router -----------------------------------------------------------------------------------

"""
    build_gateway_router(state, cfg) -> gRPCRouter

Register the three forwarded RPCs as raw `Vector{UInt8}` methods. Every other
GRPCInferenceService RPC is left unimplemented (clients get UNIMPLEMENTED), matching the Go
gateway's service descriptor.
"""
function build_gateway_router(state::GatewayState, cfg::GatewayConfig)
    router = gRPCServer.gRPCRouter(; max_receive_message_length = cfg.max_recv_msg_bytes,
        max_send_message_length = cfg.max_send_msg_bytes)
    raw(rpc) = rpc(; TRequest = Vector{UInt8}, TResponse = Vector{UInt8})
    gRPCServer.handle!(router, raw(GRPCInferenceService_ModelInfer_Method),
        (req, ctx) -> _gw_infer(req, ctx.payload))
    gRPCServer.handle!(router, raw(GRPCInferenceService_SystemSharedMemoryRegister_Method),
        (req, ctx) -> _gw_shm_register(req, ctx.payload))
    gRPCServer.handle!(router, raw(GRPCInferenceService_SystemSharedMemoryUnregister_Method),
        (req, ctx) -> _gw_shm_unregister(req, ctx.payload))
    return router
end
