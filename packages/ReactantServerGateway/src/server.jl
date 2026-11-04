# The gateway's gRPC handlers. ModelInfer routes a request to a worker that hosts the model
# (round-robin across replicas, failover on retryable errors). The SHM RPCs fan out to every
# worker because POSIX shared-memory regions live on the host and each worker attaches
# independently. All three are raw Vector{UInt8} in and out; only the small routing headers are
# decoded (see headers.jl). Mirrors the Go gateway's internal/grpcserver. The metadata RPCs
# (ModelMetadata, RepositoryIndex, DescribeModels) are answered from the workers' registries.

# Per-request state threaded through gRPCServer's context payload. Parametric on the pool and
# scheduler types so the request hot path (`st.pool` -> `get_clients` -> `wc.chan`, and the
# `select_replicas` dispatch) stays type-stable.
struct GatewayState{P <: ClientPool, S <: GatewayScheduler}
    pool::P
    routes::DiscoveredRoutes
    gate::RegisterGate
    metrics::GatewayMetrics
    refresher::RouteRefresher
    scheduler::S
end

# Map a gateway status string to a server-side gRPC exception with the matching code.
function _server_exc(status::AbstractString, msg::AbstractString)
    status == STATUS_NOT_FOUND && return gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, msg)
    # A worker shedding at its concurrency cap is transient overload, so surface it verbatim as
    # RESOURCE_EXHAUSTED (not FAILED_PRECONDITION): it is the standard signal for a client to back
    # off and retry, which ReactantServerClient does within the request's deadline budget.
    status == STATUS_RESOURCE_EXHAUSTED && return gRPCServer.GRPCError(gRPCServer.StatusCode.RESOURCE_EXHAUSTED, msg)
    # A stale shared-memory registration surfaces as FAILED_PRECONDITION; preserve the code so the
    # client re-registers and retries instead of treating it as a transient UNAVAILABLE. _try_replicas
    # returns it immediately (no failover): the client's pool registration is host-global, so every
    # worker rejects it identically until the client re-registers through the gateway's fan-out.
    status == STATUS_FAILED_PRE && return gRPCServer.GRPCError(gRPCServer.StatusCode.FAILED_PRECONDITION, msg)
    status == STATUS_INTERNAL && return gRPCServer.GRPCError(gRPCServer.StatusCode.INTERNAL, msg)
    return gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, msg)
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
            gRPCServer.GRPCError(gRPCServer.StatusCode.INTERNAL, "routing table referenced unknown worker")
    end
    gate_wait(st.gate, url)
    deadline_s = nothing
    if deadline_ns != 0
        rem = deadline_ns - Int64(time_ns())
        if rem <= 0
            return nothing, STATUS_DEADLINE,
                gRPCServer.GRPCError(
                    gRPCServer.StatusCode.DEADLINE_EXCEEDED,
                    "deadline exceeded at gateway for model \"$model\""
                )
        end
        # Forward as WHOLE SECONDS: grpc_timeout_header_val only encodes integer seconds without
        # falling back to a >8-digit nanosecond value the worker rejects; ceil so a sub-second budget
        # never becomes 0 (which would disable curl's timeout entirely).
        deadline_s = max(1, cld(rem, 1_000_000_000))
    end
    t0 = time()
    try
        resp = invoke_infer(wc, body; deadline = deadline_s)
        observe_worker!(st.metrics, "ModelInfer", url, time() - t0)
        # A worker's reply always carries model_name, so an empty body is a call that ended without
        # a response: gRPCClient before 1.1.2 returned one when `reset_clients!` aborted the call.
        # Fail over rather than hand the client an empty response as a success.
        if isempty(resp)
            @warn "infer: worker returned an empty response" model worker = url request_id = id
            return nothing, STATUS_UNAVAILABLE,
                _server_exc(STATUS_UNAVAILABLE, "worker $url: empty response (connection reset?)")
        end
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
    last_exc = gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, "no replica available")
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
    # How much work this request is, per the scheduler (a no-op for the modes that route by request
    # count). Resolved once here rather than per selection attempt.
    ctx = ScheduleContext(
        model, id, request_units(st.scheduler, model, body),
        st.pool, st.routes, st.metrics, st.refresher
    )
    sel = select_replicas(st.scheduler, ctx)
    if sel === nothing
        refresh_now!(st.refresher)
        sel = select_replicas(st.scheduler, ctx)
    end
    if sel === nothing
        @info "infer: model not found" model request_id = id
        return nothing, STATUS_NOT_FOUND,
            gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "model \"$model\" not found on any worker")
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

function _gw_infer(body::Vector{UInt8}, st::GatewayState, deadline_ns::Integer = 0)
    t0 = time()
    local model, id
    try
        model, id = peek_model_name_and_id(body)
    catch e
        inc_requests!(st.metrics, "ModelInfer", "", STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "malformed ModelInferRequest: $(sprint(showerror, e))"
            )
        )
    end
    if isempty(model)
        inc_requests!(st.metrics, "ModelInfer", "", STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "ModelInferRequest.model_name is empty"
            )
        )
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
    first_body = Ref{Union{Nothing, Vector{UInt8}}}(nothing)
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
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "malformed SystemSharedMemoryRegisterRequest: $(sprint(showerror, e))"
            )
        )
    end
    if isempty(region)
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "SystemSharedMemoryRegisterRequest.name is empty"
            )
        )
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
    throw(
        gRPCServer.GRPCError(
            gRPCServer.StatusCode.FAILED_PRECONDITION,
            "SHM register failed on workers: $(failed)"
        )
    )
end

function _gw_shm_unregister(body::Vector{UInt8}, st::GatewayState)
    t0 = time()
    region = ""
    try
        region = peek_shm_name(body)
    catch e
        inc_requests!(st.metrics, "SystemSharedMemoryUnregister", "", STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "malformed SystemSharedMemoryUnregisterRequest: $(sprint(showerror, e))"
            )
        )
    end

    resp, succeeded, failed = _broadcast_shm(st, wc -> invoke_shm_unregister(wc, body), region, "SystemSharedMemory")
    observe_request!(st.metrics, "SystemSharedMemoryUnregister", region, time() - t0)

    if isempty(succeeded) && !isempty(failed)
        @warn "shm.unregister: every worker failed" region failed_workers = failed
        inc_requests!(st.metrics, "SystemSharedMemoryUnregister", region, STATUS_UNAVAILABLE)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.UNAVAILABLE,
                "SHM unregister failed on all workers: $(failed)"
            )
        )
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
function _compact_workers(pool::ClientPool, metrics::Union{GatewayMetrics, Nothing}, perworker::AbstractDict)
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
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.UNAVAILABLE,
                "compaction failed on all workers: $(failed)"
            )
        )
    end
    isempty(failed) ? (@info "compact: ok" workers = succeeded reloaded = total) :
        (@warn "compact: partial failures" ok_workers = succeeded failed_workers = failed reloaded = total)
    inc_requests!(st.metrics, "CompactMemory", "", STATUS_OK)
    return CompactMemoryResponse(; reloaded_models = Int64(total))
end

# --- Model metadata ---------------------------------------------------------------------------
#
# Workers normally load the same bundles, but any one of them may be down or restarting, so the
# metadata RPCs never depend on a single worker. DescribeModels and RepositoryIndex ask every worker
# at once and merge the answers; ModelMetadata tries the workers that host the model, then the rest.
# None of them goes through the scheduler or moves a round-robin cursor.

# Run `f(wc)` against every worker concurrently. Returns the workers in pool order and, for each,
# its result or the exception it threw.
function _ask_workers(f, st::GatewayState)
    workers = all_clients(st.pool)
    answers = Vector{Any}(undef, length(workers))
    @sync for (i, wc) in enumerate(workers)
        @async answers[i] = try
            f(wc)
        catch e
            e
        end
    end
    return workers, answers
end

_call_status(e) = e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : -1

# The error for a fan-out every worker failed: UNIMPLEMENTED when they all lack the RPC (workers older
# than the gateway, which a client should hear about as such), UNAVAILABLE otherwise.
function _all_workers_failed(rpc::AbstractString, workers, answers)
    if !isempty(answers) && all(a -> _call_status(a) == gRPCClient.GRPC_UNIMPLEMENTED, answers)
        return STATUS_UNAVAILABLE, gRPCServer.GRPCError(
                gRPCServer.StatusCode.UNIMPLEMENTED, "no worker implements $rpc; the workers are older than the gateway"
            )
    end
    return STATUS_UNAVAILABLE, gRPCServer.GRPCError(
            gRPCServer.StatusCode.UNAVAILABLE,
            isempty(workers) ? "$rpc: the gateway has no workers" :
            "$rpc failed on every worker: $(join((wc.url for wc in workers), ", "))"
        )
end

# Record a metadata RPC's latency and outcome. The model label stays empty: these take arbitrary
# names from clients, which must not become label values.
function _record_metadata!(st::GatewayState, rpc, status, t0)
    observe_request!(st.metrics, rpc, "", time() - t0)
    inc_requests!(st.metrics, rpc, "", status)
    return nothing
end

# Merge every worker's description of `names` (empty: every model it routes) into `merged`, the first
# worker in pool order winning a name several describe. Returns the workers that answered.
function _merge_descriptions!(merged::Dict{String, ModelDescription}, st::GatewayState, names)
    workers, answers = _ask_workers(wc -> invoke_describe_models(wc, DescribeModelsRequest(; names = names)), st)
    answered = String[]
    for (wc, a) in zip(workers, answers)
        a isa DescribeModelsResponse || continue
        push!(answered, wc.url)
        for m in a.models
            get!(merged, m.name, m)
        end
    end
    return workers, answers, answered
end

function _gw_describe_models(req::DescribeModelsRequest, st::GatewayState)
    t0 = time()
    merged = Dict{String, ModelDescription}()
    workers, answers, answered = _merge_descriptions!(merged, st, String[])
    if isempty(answered)
        status, err = _all_workers_failed("DescribeModels", workers, answers)
        _record_metadata!(st, "DescribeModels", status, t0)
        throw(err)
    end
    names = isempty(req.names) ? sort!(collect(keys(merged))) : sort!(unique(collect(String, req.names)))
    # A named model missing from the routed set may be a meta model's internal sub-model, which
    # workers describe only when asked for it by name. Ask once more for exactly those, one at a time:
    # a worker answers NOT_FOUND for the whole call if it lacks any one of them.
    for n in filter(n -> !haskey(merged, n), names)
        _merge_descriptions!(merged, st, [n])
    end
    unknown = filter(n -> !haskey(merged, n), names)
    if !isempty(unknown)
        _record_metadata!(st, "DescribeModels", STATUS_NOT_FOUND, t0)
        throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "unknown model: $(join(unknown, ", "))"))
    end
    _record_metadata!(st, "DescribeModels", STATUS_OK, t0)
    return DescribeModelsResponse(; models = [merged[n] for n in names])
end

# KServe ModelMetadata, forwarded to a worker: the replicas the routing table lists first (a model
# loaded since the last discovery round is not listed yet), then every other worker. NOT_FOUND only
# when every worker that answered said so.
function _gw_model_metadata(req::ModelMetadataRequest, st::GatewayState)
    t0 = time()
    routed = something(replicas(st.routes, req.name), String[])
    order = vcat(routed, filter(u -> !(u in routed), st.pool.order))
    errors = Any[]
    for url in order
        wc = get_clients(st.pool, url)
        wc === nothing && continue
        try
            resp = invoke_model_metadata(wc, req)
            _record_metadata!(st, "ModelMetadata", STATUS_OK, t0)
            return resp
        catch e
            push!(errors, e)
        end
    end
    if !isempty(errors) && all(e -> _call_status(e) == gRPCClient.GRPC_NOT_FOUND, errors)
        _record_metadata!(st, "ModelMetadata", STATUS_NOT_FOUND, t0)
        throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "unknown model: $(req.name)"))
    end
    status, err = _all_workers_failed("ModelMetadata", all_clients(st.pool), errors)
    _record_metadata!(st, "ModelMetadata", status, t0)
    throw(err)
end

# KServe RepositoryIndex: the union of every worker's index, a model READY when any worker has it
# READY (otherwise the first worker's entry, with its reason).
function _gw_repository_index(req::RepositoryIndexRequest, st::GatewayState)
    t0 = time()
    workers, answers = _ask_workers(wc -> invoke_repository_index(wc, req), st)
    best = Dict{String, var"RepositoryIndexResponse.ModelIndex"}()
    nanswered = 0
    for a in answers
        a isa RepositoryIndexResponse || continue
        nanswered += 1
        for m in a.models
            cur = get(best, m.name, nothing)
            (cur === nothing || (cur.state != "READY" && m.state == "READY")) && (best[m.name] = m)
        end
    end
    if nanswered == 0
        status, err = _all_workers_failed("RepositoryIndex", workers, answers)
        _record_metadata!(st, "RepositoryIndex", status, t0)
        throw(err)
    end
    _record_metadata!(st, "RepositoryIndex", STATUS_OK, t0)
    return RepositoryIndexResponse(; models = [best[n] for n in sort!(collect(keys(best)))])
end

# --- Server ---------------------------------------------------------------------------------

# Old gRPCServer handed handlers an absolute deadline on Julia's machine-relative time_ns()
# clock (time_ns() + grpc-timeout). The merged package parses grpc-timeout into a wall-clock
# DateTime; the shared ReactantServerCore.deadline_to_time_ns converts ctx.deadline back onto
# the local clock the gateway compares against (deadline_ns - time_ns() in _post_infer).

"""
    build_gateway_server(state, cfg, host, port) -> GRPCServer

Register the forwarded RPCs on a fresh [`gRPCServer.GRPCServer`](@ref) bound to `host:port`.
ModelInfer and the two SHM register/unregister RPCs are raw `Vector{UInt8}` methods (bytes pass
through unchanged); IsSameIPCNamespace is typed because the gateway aggregates each worker's
answer rather than forwarding one. ModelMetadata, RepositoryIndex, and the control plane's
DescribeModels and CompactMemory are typed too (see "Model metadata" above). Every other
GRPCInferenceService RPC is left unimplemented (clients get UNIMPLEMENTED).
"""
function build_gateway_server(
        state::GatewayState, cfg::GatewayConfig, host::AbstractString, port::Integer;
        h2_initial_window_size::Integer = _H2_INITIAL_WINDOW_BYTES,
        h2_connection_window_size::Integer = _H2_CONNECTION_WINDOW_BYTES
    )
    server = gRPCServer.GRPCServerHTTPJl(
        host, Int(port);
        context = state,
        max_receive_message_length = cfg.max_recv_msg_bytes,
        max_send_message_length = cfg.max_send_msg_bytes,
        h2_initial_window_size = h2_initial_window_size,
        h2_connection_window_size = h2_connection_window_size
    )
    # Raw-byte passthrough RPCs: the generated register functions take the raw_* kwargs.
    register_GRPCInferenceService_ModelInfer!(
        server,
        (ctx, body) -> _gw_infer(body, ctx.payload, deadline_to_time_ns(ctx.deadline, gRPCServer.remaining_time(ctx)));
        raw_request = true, raw_response = true
    )
    register_GRPCInferenceService_SystemSharedMemoryRegister!(
        server,
        (ctx, body) -> _gw_shm_register(body, ctx.payload);
        raw_request = true, raw_response = true
    )
    register_GRPCInferenceService_SystemSharedMemoryUnregister!(
        server,
        (ctx, body) -> _gw_shm_unregister(body, ctx.payload);
        raw_request = true, raw_response = true
    )
    register_GRPCInferenceService_IsSameIPCNamespace!(
        server,
        (ctx, req) -> _gw_is_same_ipc_namespace(req, ctx.payload)
    )
    register_GRPCInferenceService_ModelMetadata!(server, (ctx, req) -> _gw_model_metadata(req, ctx.payload))
    register_GRPCInferenceService_RepositoryIndex!(server, (ctx, req) -> _gw_repository_index(req, ctx.payload))
    # Control plane: CompactMemory fans out to every worker, and DescribeModels merges every worker's.
    register_ControlService!(
        server;
        CompactMemory = (ctx, req) -> _gw_compact(req, ctx.payload),
        DescribeModels = (ctx, req) -> _gw_describe_models(req, ctx.payload)
    )
    # The gateway's own scheduling control plane (see control_service.jl). A distinct service, so the
    # worker-facing ControlService above keeps its meaning.
    register_gateway_control_service!(server, state)
    return server
end
