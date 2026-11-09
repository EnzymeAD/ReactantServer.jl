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
    error_detail::ErrorDetail          # how much an INTERNAL error tells the client (`grpc.error_detail`)
    labels::Dict{String, String}       # worker url -> the name client-visible messages use for it
end
GatewayState(pool, routes, gate, metrics, refresher, scheduler) = GatewayState(
    pool, routes, gate, metrics, refresher, scheduler, ERROR_DETAIL_REDACTED, Dict{String, String}()
)

"""
    worker_labels(cfg) -> Dict{String, String}

The name each worker goes by in client-visible messages, by url: its configured name
(`worker_names`, which the supervisor sets to worker0..N), else `worker <i>` by its 0-based position
in the endpoint list. A redacted gateway names workers this way and never by host:port (GW-15).
"""
function worker_labels(cfg::GatewayConfig)
    return Dict(
        url => (i <= length(cfg.worker_names) && !isempty(cfg.worker_names[i]) ? cfg.worker_names[i] : "worker $(i - 1)")
            for (i, url) in enumerate(cfg.workers)
    )
end

# How a client-visible message names the worker at `url`: by host:port in `full` mode, by its label
# in `redacted` mode.
_shown(st::GatewayState, url::AbstractString) =
    st.error_detail == ERROR_DETAIL_FULL ? String(url) : get(st.labels, url, "a worker")

# The gateway's own INTERNAL failure `e` (an exception, or a message) as the GRPCError the client
# receives: logged with its backtrace under a request id (the client's KServe `id`, else a generated
# one), and reported by that id only (`redacted`) or with its text (`full`).
function _gw_internal(st::GatewayState, what::AbstractString, id::AbstractString, e, bt = nothing)
    rid = error_request_id(id)
    text = e isa AbstractString ? String(e) : sprint(showerror, e)
    if bt === nothing
        @error "$what failed" request_id = rid message = text
    else
        @error "$what failed" request_id = rid exception = (e, bt)
    end
    ncodeunits(text) <= 1024 || (text = text[1:prevind(text, 1025)] * " ...")
    msg = st.error_detail == ERROR_DETAIL_FULL ? "$text (request id $rid, gateway)" :
        "internal error (request id $rid, gateway); see the gateway log"
    return gRPCServer.GRPCError(gRPCServer.StatusCode.INTERNAL, sanitize_status_message(msg))
end

# Run a gateway handler so every status it ends with leaves sanitized (`sanitize_status_message`):
# a GRPCError keeps its code, and any other exception is the gateway's own INTERNAL failure
# (`_gw_internal`). Every registered handler runs through this.
function _gw_status(f, st::GatewayState, what::AbstractString, id::AbstractString = "")
    try
        return f()
    catch e
        e isa gRPCServer.GRPCError &&
            throw(gRPCServer.GRPCError(e.code, sanitize_status_message(e.message), e.details))
        throw(_gw_internal(st, what, id, e, catch_backtrace()))
    end
end

# Gateway status string -> the gRPC code the client receives: the worker's own code, verbatim. A
# worker shedding at its concurrency cap stays RESOURCE_EXHAUSTED (the standard signal to back off and
# retry, which ReactantServerClient does within the request's deadline budget); a stale shared-memory
# registration stays FAILED_PRECONDITION, so the client re-registers rather than retrying against
# the same stale region (every worker would reject it alike, since the registration is host-global);
# a deadline stays DEADLINE_EXCEEDED, so the client stops.
const _CODE_BY_STATUS = Dict{String, gRPCServer.StatusCode.T}(
    STATUS_OK => gRPCServer.StatusCode.OK,
    STATUS_CANCELLED => gRPCServer.StatusCode.CANCELLED,
    STATUS_UNKNOWN => gRPCServer.StatusCode.UNKNOWN,
    STATUS_INVALID => gRPCServer.StatusCode.INVALID_ARGUMENT,
    STATUS_DEADLINE => gRPCServer.StatusCode.DEADLINE_EXCEEDED,
    STATUS_NOT_FOUND => gRPCServer.StatusCode.NOT_FOUND,
    STATUS_ALREADY_EXISTS => gRPCServer.StatusCode.ALREADY_EXISTS,
    STATUS_PERMISSION_DENIED => gRPCServer.StatusCode.PERMISSION_DENIED,
    STATUS_RESOURCE_EXHAUSTED => gRPCServer.StatusCode.RESOURCE_EXHAUSTED,
    STATUS_FAILED_PRE => gRPCServer.StatusCode.FAILED_PRECONDITION,
    STATUS_ABORTED => gRPCServer.StatusCode.ABORTED,
    STATUS_OUT_OF_RANGE => gRPCServer.StatusCode.OUT_OF_RANGE,
    STATUS_UNIMPLEMENTED => gRPCServer.StatusCode.UNIMPLEMENTED,
    STATUS_INTERNAL => gRPCServer.StatusCode.INTERNAL,
    STATUS_UNAVAILABLE => gRPCServer.StatusCode.UNAVAILABLE,
    STATUS_DATA_LOSS => gRPCServer.StatusCode.DATA_LOSS,
    STATUS_UNAUTHENTICATED => gRPCServer.StatusCode.UNAUTHENTICATED,
)

# Map a gateway status string to a server-side gRPC exception with the matching code.
_server_exc(status::AbstractString, msg::AbstractString) =
    gRPCServer.GRPCError(get(_CODE_BY_STATUS, status, gRPCServer.StatusCode.UNKNOWN), msg)

# The worker failures a request moves on to the next replica for: the model is not there (NOT_FOUND:
# unloaded since discovery) or the worker cannot serve right now (UNAVAILABLE: shutting down, its
# dispatch loop died, the model not resident there or not placeable on its device now, or the call
# failed in the transport). Every other status is the request's own outcome and would only repeat on
# another replica.
_fails_over(status::AbstractString) = status == STATUS_NOT_FOUND || status == STATUS_UNAVAILABLE

# The answer for a request whose model is served only by workers that are not ready: the gateway
# sends nothing to a worker the prober has marked down, so the client hears UNAVAILABLE and retries.
_no_ready_worker(model) = gRPCServer.GRPCError(
    gRPCServer.StatusCode.UNAVAILABLE,
    "model \"$model\" is unavailable: the workers that serve it are not ready"
)

_is_cancelled(cancelled) = cancelled !== nothing && (
    try
        cancelled() === true
    catch
        false
    end
)

# --- ModelInfer -------------------------------------------------------------------------------

# `deadline_ns` is the request's absolute deadline in the gateway's clock (0 = none). We recompute the
# remaining budget HERE — the latest point before the worker call, after the register-gate wait and
# any failover retries — and forward it as the worker's grpc-timeout, so the budget shrinks for time
# burned at the gateway instead of resetting to a fresh full budget at the worker. If it is already
# gone, shed here (DEADLINE_EXCEEDED) rather than forward doomed work into the worker's queue.
function _post_infer(st::GatewayState, url, model, id, body, deadline_ns::Int64, cancelled = nothing)
    wc = get_clients(st.pool, url)
    if wc === nothing
        return nothing, STATUS_INTERNAL,
            _gw_internal(st, "infer for model \"$model\"", id, "routing table referenced unknown worker $url")
    end
    gate_wait(st.gate, url)
    if _is_cancelled(cancelled)
        return nothing, STATUS_CANCELLED,
            gRPCServer.GRPCError(gRPCServer.StatusCode.CANCELLED, "request for model \"$model\" was cancelled by the client")
    end
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
    # In `full` mode a failure names the worker by host:port, prefixed to whatever the call ended
    # with. In `redacted` mode a worker's own status message passes through verbatim (the worker has
    # already redacted it), and a failure of the call itself is described by `invoke_infer` under the
    # worker's label, never with the transport's text (which names hosts and ports).
    full = st.error_detail == ERROR_DETAIL_FULL
    try
        resp = invoke_infer(wc, body; deadline = deadline_s, cancelled = cancelled, label = full ? nothing : _shown(st, url))
        observe_worker!(st.metrics, "ModelInfer", url, time() - t0)
        # A worker's reply always carries model_name, so an empty body is a call that ended without
        # a response: gRPCClient before 1.1.2 returned one when a handle reset aborted the call.
        # Fail over rather than hand the client an empty response as a success.
        if isempty(resp)
            @warn "infer: worker returned an empty response" model worker = url request_id = id
            return nothing, STATUS_UNAVAILABLE,
                _server_exc(STATUS_UNAVAILABLE, "$(full ? "worker $url" : _shown(st, url)): empty response (connection reset?)")
        end
        return resp, STATUS_OK, nothing
    catch e
        observe_worker!(st.metrics, "ModelInfer", url, time() - t0)
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        status = client_status(e)
        @warn "infer: worker error" model worker = url request_id = id status = status msg = e.message
        return nothing, status, _server_exc(status, full ? "worker $url: $(e.message)" : e.message)
    end
end

# Try the replicas in order (urls[1] is the round-robin choice; the rest are failover targets),
# moving on only for a retryable status (`_fails_over`: worker NotFound or Unavailable, a transport
# failure included); every other status reaches the client verbatim. A worker NOT_FOUND means the
# model was unloaded there since discovery, so kick an async route refresh to drop the stale route
# (the request itself still fails over / returns the error; it is not retried after refresh).
#
# A request that references shared memory is never moved to another replica: the worker that
# failed it may still be running it (an abandoned or reset call is not a stopped one), and a second
# worker writing the same output slot, or reading an input slot the client has since reused, would
# corrupt the client's data. Its failure reaches the client as UNAVAILABLE, the client's cue to retry.
#
# A replica marked down after the scheduler chose it (the prober's round landed in between) is
# skipped: no request is sent to a worker marked down, as a first choice or as a failover.
function _try_replicas(st::GatewayState, urls, model, id, body, deadline_ns::Int64, cancelled = nothing)
    last_status = STATUS_UNAVAILABLE
    last_exc = _no_ready_worker(model)
    shm = nothing           # read from the body only once a worker has failed the request
    for url in urls
        is_healthy(st.routes, url) || continue
        resp, status, exc = _post_infer(st, url, model, id, body, deadline_ns, cancelled)
        exc === nothing && return resp, status, nothing
        status == STATUS_NOT_FOUND && request_refresh!(st.refresher)
        _fails_over(status) || return nothing, status, exc
        shm === nothing && (shm = peek_uses_shared_memory(body))
        if shm
            msg = "$(exc.message) (the request uses shared memory, so it is not retried on another worker)"
            return nothing, STATUS_UNAVAILABLE, _server_exc(STATUS_UNAVAILABLE, msg)
        end
        last_status, last_exc = status, exc
    end
    return nothing, last_status, last_exc
end

function _dispatch_infer(st::GatewayState, model, id, body, deadline_ns::Int64, cancelled = nothing)
    # Ask the scheduler which replicas to try (urls[1] is its choice; the rest are failover order)
    # and for an opaque reservation to release once the request completes. Every URL it returns is a
    # worker that is not marked down. `nothing` means the scheduler has no route for the model: it
    # may have just been loaded on a worker, so refresh the routing table on demand (single-flight,
    # rate-limited) and re-select before giving up. An empty list means the model is known but no
    # worker that serves it is ready.
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
    # A known model whose workers are all down: answered without calling any of them. UNAVAILABLE,
    # because the model exists and the client should retry once a worker is back. A known model
    # skips the refresh above: a discovery scan never changes which workers are down.
    if isempty(urls)
        release!(st.scheduler, reservation)
        @info "infer: no ready worker serves the model" model request_id = id
        return nothing, STATUS_UNAVAILABLE, _no_ready_worker(model)
    end
    # Release the reservation exactly once, on every path (success, retryable error, hard error, or
    # timeout), so a scheduler's outstanding counters never leak.
    try
        return _try_replicas(st, urls, model, id, body, deadline_ns, cancelled)
    finally
        release!(st.scheduler, reservation)
    end
end

# The `model` label of the ModelInfer series for every name the routing table does not hold when the
# request is counted: names nobody serves, names a client made up, and an empty or unreadable one. A
# client chooses the names it sends, so labelling with them would let any client add series (each a
# counter and an 18-bucket histogram) without bound; with this, the series are bounded by the models
# the gateway has actually routed, plus this one. The workers use the same label for the same
# purpose. The name is reserved: workers refuse to load a bundle named `_unknown`, so no routed model
# shares this series.
const UNKNOWN_MODEL_LABEL = "_unknown"

# The `model` label for a ModelInfer for `model`: its name if the routing table holds it right now,
# else `UNKNOWN_MODEL_LABEL`. A model that leaves the table later keeps its series, which stays
# bounded by the models the gateway has served.
_model_label(st::GatewayState, model::AbstractString) =
    has_model(st.routes, model) ? String(model) : UNKNOWN_MODEL_LABEL

# `cancelled` is the inbound call's cancellation probe (see `_CallCancelled`): a cancelled call stops
# trying replicas and cancels its in-flight worker call.
function _gw_infer(body::Vector{UInt8}, st::GatewayState, deadline_ns::Integer = 0, cancelled = nothing)
    t0 = time()
    local model, id
    try
        model, id = peek_model_name_and_id(body)
    catch e
        inc_requests!(st.metrics, "ModelInfer", UNKNOWN_MODEL_LABEL, STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "malformed ModelInferRequest: $(sprint(showerror, e))"
            )
        )
    end
    if isempty(model)
        inc_requests!(st.metrics, "ModelInfer", UNKNOWN_MODEL_LABEL, STATUS_INVALID)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.INVALID_ARGUMENT,
                "ModelInferRequest.model_name is empty"
            )
        )
    end
    # Only a routed model's arrivals are counted: a scheduler keeps a counter per name it is given.
    has_model(st.routes, model) && record_arrival!(st.scheduler, model)
    resp, status, exc = _dispatch_infer(st, model, id, body, Int64(deadline_ns), cancelled)
    label = _model_label(st, model)
    observe_request!(st.metrics, "ModelInfer", label, time() - t0)
    inc_requests!(st.metrics, "ModelInfer", label, status)
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
# register gate. Returns (first_response, succeeded_urls, failed_urls, errors), where `errors` maps
# each failed worker's URL to what it threw.
function _broadcast_shm(st::GatewayState, op, region, rpc_label)
    workers = all_clients(st.pool)
    succeeded = String[]
    failed = String[]
    errors = Dict{String, Any}()
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
                    errors[wc.url] = e
                end
                @warn "shm: worker failed" region worker = wc.url exception = e
            finally
                release()
            end
        end
    end
    return first_body[], succeeded, failed, errors
end

# Whether `wc` has `region` registered right now: true, false, or `nothing` when the status call
# failed. NOT_FOUND counts as absent (Triton answers an unknown name that way; this repo's worker
# returns an empty map).
function _shm_registered_on(wc::WorkerClients, region::AbstractString)
    try
        return haskey(invoke_shm_status(wc, region).regions, region)
    catch e
        (e isa gRPCClient.gRPCServiceCallException && e.grpc_status == gRPCClient.GRPC_NOT_FOUND) && return false
        @warn "shm.register: status pre-check failed; the registration will not be rolled back on this worker" region worker = wc.url exception = e
        return nothing
    end
end

_already_exists(e) = e isa gRPCClient.gRPCServiceCallException && e.grpc_status == gRPCClient.GRPC_ALREADY_EXISTS

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

# The shared-memory RPCs are counted with an empty `model` label: they name a region, not a model, and
# a region's name is chosen by each client process (usually at random), so it must not label a series.
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

    # A worker that already holds the registration (the same client re-registering after another
    # worker restarted) succeeds idempotently, so a rollback must not remove it: that would break a
    # registration that worked before this call. Each worker is asked first, under the same gate as
    # the register, whether it holds the name; only the registrations this call created are rolled
    # back. An unanswered pre-check counts as "held" (never roll back what may have existed). A
    # registration created concurrently by another call for the same object can still be rolled
    # back; that client then sees FAILED_PRECONDITION on its next request and re-registers.
    created = Set{String}()
    created_lock = ReentrantLock()
    register_op = function (wc)
        held = _shm_registered_on(wc, region)
        resp = invoke_shm_register(wc, body)
        held === false && @lock created_lock push!(created, wc.url)
        return resp
    end
    resp, succeeded, failed, errors = _broadcast_shm(st, register_op, region, "SystemSharedMemory")
    observe_request!(st.metrics, "SystemSharedMemoryRegister", "", time() - t0)

    if isempty(failed)
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_OK)
        @info "shm.register: ok" region workers = succeeded
        return resp === nothing ? UInt8[] : resp
    end

    rollback = filter(in(created), succeeded)
    if !isempty(succeeded)
        @warn "shm.register: partial failure, rolling back what this call created" region ok_workers = succeeded failed_workers = failed rolled_back = rollback
        isempty(rollback) || _fanout_unregister(st, rollback, _encode_unregister(region))
    end
    # The name belongs to another client's live object on some worker: say so, as Triton does, so
    # the client does not mistake it for a stale registration.
    if any(_already_exists, values(errors))
        inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_ALREADY_EXISTS)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.ALREADY_EXISTS,
                "SHM region \"$region\" is already registered for a different object on workers: " *
                    "$(sort!([_shown(st, u) for (u, e) in errors if _already_exists(e)]))"
            )
        )
    end
    inc_requests!(st.metrics, "SystemSharedMemoryRegister", "", STATUS_FAILED_PRE)
    throw(
        gRPCServer.GRPCError(
            gRPCServer.StatusCode.FAILED_PRECONDITION,
            "SHM register failed on workers: $([_shown(st, u) for u in failed])"
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

    resp, succeeded, failed, _ = _broadcast_shm(st, wc -> invoke_shm_unregister(wc, body), region, "SystemSharedMemory")
    observe_request!(st.metrics, "SystemSharedMemoryUnregister", "", time() - t0)

    if isempty(succeeded) && !isempty(failed)
        @warn "shm.unregister: every worker failed" region failed_workers = failed
        inc_requests!(st.metrics, "SystemSharedMemoryUnregister", "", STATUS_UNAVAILABLE)
        throw(
            gRPCServer.GRPCError(
                gRPCServer.StatusCode.UNAVAILABLE,
                "SHM unregister failed on all workers: $([_shown(st, u) for u in failed])"
            )
        )
    end
    if !isempty(failed)
        @warn "shm.unregister: partial failures" region ok_workers = succeeded failed_workers = failed
    else
        @info "shm.unregister: ok" region workers = succeeded
    end
    inc_requests!(st.metrics, "SystemSharedMemoryUnregister", "", STATUS_OK)
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
                "compaction failed on all workers: $([_shown(st, u) for u in failed])"
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
function _all_workers_failed(st::GatewayState, rpc::AbstractString, workers, answers)
    if !isempty(answers) && all(a -> _call_status(a) == gRPCClient.GRPC_UNIMPLEMENTED, answers)
        return STATUS_UNAVAILABLE, gRPCServer.GRPCError(
                gRPCServer.StatusCode.UNIMPLEMENTED, "no worker implements $rpc; the workers are older than the gateway"
            )
    end
    return STATUS_UNAVAILABLE, gRPCServer.GRPCError(
            gRPCServer.StatusCode.UNAVAILABLE,
            isempty(workers) ? "$rpc: the gateway has no workers" :
            "$rpc failed on every worker: $(join((_shown(st, wc.url) for wc in workers), ", "))"
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
        status, err = _all_workers_failed(st, "DescribeModels", workers, answers)
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
    status, err = _all_workers_failed(st, "ModelMetadata", all_clients(st.pool), errors)
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
        status, err = _all_workers_failed(st, "RepositoryIndex", workers, answers)
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

# A registered handler `f(req, state)` run through `_gw_status`, as gRPCServer calls it.
_gw_handler(f, what::AbstractString) = (ctx, req) -> _gw_status(() -> f(req, ctx.payload), ctx.payload, what)

# Whether the client has cancelled one inbound call, the probe `_gw_infer` passes down to the worker
# call. gRPCServer 0.1.2 never marks a call cancelled, so with that release this never fires; it
# takes effect once gRPCServer sets `ServerContext.cancelled`.
struct _CallCancelled
    ctx::gRPCServer.ServerContext
end
(c::_CallCancelled)() = gRPCServer.is_cancelled(c.ctx)

_model_infer_rpc(ctx, body) = _gw_status(ctx.payload, "ModelInfer") do
    _gw_infer(body, ctx.payload, deadline_to_time_ns(ctx.deadline, gRPCServer.remaining_time(ctx)), _CallCancelled(ctx))
end

# What the gateway passes gRPCServer as `max_concurrent_requests` for an inbound cap of `cap`
# (0 = uncapped). gRPCServer's own uncapped forms (`nothing`, 0) also stop it counting the calls in
# flight, which `gateway_inflight_requests` reads, so an uncapped gateway passes a cap that cannot be
# reached instead: no call is shed and the count stays live.
_admission_limit(cap::Integer) = cap > 0 ? Int(cap) : typemax(Int)

# The listener's per-connection stream limit, as keyword arguments for gRPCServer. By default it is
# `_H2_MAX_CONCURRENT_STREAMS`, so a client is never cut off for giving up on calls (GW-18). With
# `grpc.rapid_reset_protection` none is passed: gRPCServer's default of 100 streams per connection
# applies, and with it HTTP.jl's Rapid-Reset budget of 200 resets per connection.
_listener_streams(cfg::GatewayConfig) =
    cfg.rapid_reset_protection ? (;) : (; max_concurrent_streams = _H2_MAX_CONCURRENT_STREAMS)

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
        # The inbound cap: past it gRPCServer sheds a call at once with RESOURCE_EXHAUSTED, before
        # any handler runs, and counts it in the `shed_total` that `gateway_requests_shed_total` reads.
        max_concurrent_requests = _admission_limit(inbound_cap(cfg)),
        _listener_streams(cfg)...,
        h2_initial_window_size = h2_initial_window_size,
        h2_connection_window_size = h2_connection_window_size
    )
    # Raw-byte passthrough RPCs: the generated register functions take the raw_* kwargs.
    register_GRPCInferenceService_ModelInfer!(server, _model_infer_rpc; raw_request = true, raw_response = true)
    register_GRPCInferenceService_SystemSharedMemoryRegister!(
        server,
        _gw_handler(_gw_shm_register, "SystemSharedMemoryRegister");
        raw_request = true, raw_response = true
    )
    register_GRPCInferenceService_SystemSharedMemoryUnregister!(
        server,
        _gw_handler(_gw_shm_unregister, "SystemSharedMemoryUnregister");
        raw_request = true, raw_response = true
    )
    register_GRPCInferenceService_IsSameIPCNamespace!(server, _gw_handler(_gw_is_same_ipc_namespace, "IsSameIPCNamespace"))
    register_GRPCInferenceService_ModelMetadata!(server, _gw_handler(_gw_model_metadata, "ModelMetadata"))
    register_GRPCInferenceService_RepositoryIndex!(server, _gw_handler(_gw_repository_index, "RepositoryIndex"))
    # Control plane: CompactMemory fans out to every worker, and DescribeModels merges every worker's.
    register_ControlService!(
        server;
        CompactMemory = _gw_handler(_gw_compact, "CompactMemory"),
        DescribeModels = _gw_handler(_gw_describe_models, "DescribeModels")
    )
    # The gateway's own scheduling control plane (see control_service.jl). A distinct service, so the
    # worker-facing ControlService above keeps its meaning.
    register_gateway_control_service!(server, state)
    return server
end
