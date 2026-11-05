# Per-worker gRPC connections used to forward requests to the Julia inference workers. ModelInfer
# and the SHM RPCs send and receive raw Vector{UInt8} so request and response bytes pass through
# without re-marshaling; the probes are typed (the messages are tiny). Each worker has one
# gRPCChannel over its own libcurl multi handle, built up front and reused. The RegisterGate
# serializes ModelInfer against a worker while a SHM register/unregister to that worker is in flight
# (POSIX shm registration is host-global).

# Status strings surfaced to clients and recorded in metrics, mirroring the Go gateway. There is
# one for every gRPC code, so a worker's code reaches the client verbatim (see `client_status`).
const STATUS_OK = "OK"
const STATUS_NOT_FOUND = "NotFound"
const STATUS_UNAVAILABLE = "Unavailable"
const STATUS_RESOURCE_EXHAUSTED = "ResourceExhausted"
const STATUS_INVALID = "InvalidArgument"
const STATUS_FAILED_PRE = "FailedPrecondition"
const STATUS_INTERNAL = "Internal"
const STATUS_DEADLINE = "DeadlineExceeded"
const STATUS_CANCELLED = "Cancelled"
const STATUS_UNKNOWN = "Unknown"
const STATUS_ALREADY_EXISTS = "AlreadyExists"
const STATUS_PERMISSION_DENIED = "PermissionDenied"
const STATUS_ABORTED = "Aborted"
const STATUS_OUT_OF_RANGE = "OutOfRange"
const STATUS_UNIMPLEMENTED = "Unimplemented"
const STATUS_DATA_LOSS = "DataLoss"
const STATUS_UNAUTHENTICATED = "Unauthenticated"

# gRPC status code -> gateway status string.
const _STATUS_BY_CODE = Dict{Int, String}(
    gRPCClient.GRPC_OK => STATUS_OK,
    gRPCClient.GRPC_CANCELLED => STATUS_CANCELLED,
    gRPCClient.GRPC_UNKNOWN => STATUS_UNKNOWN,
    gRPCClient.GRPC_INVALID_ARGUMENT => STATUS_INVALID,
    gRPCClient.GRPC_DEADLINE_EXCEEDED => STATUS_DEADLINE,
    gRPCClient.GRPC_NOT_FOUND => STATUS_NOT_FOUND,
    gRPCClient.GRPC_ALREADY_EXISTS => STATUS_ALREADY_EXISTS,
    gRPCClient.GRPC_PERMISSION_DENIED => STATUS_PERMISSION_DENIED,
    gRPCClient.GRPC_RESOURCE_EXHAUSTED => STATUS_RESOURCE_EXHAUSTED,
    gRPCClient.GRPC_FAILED_PRECONDITION => STATUS_FAILED_PRE,
    gRPCClient.GRPC_ABORTED => STATUS_ABORTED,
    gRPCClient.GRPC_OUT_OF_RANGE => STATUS_OUT_OF_RANGE,
    gRPCClient.GRPC_UNIMPLEMENTED => STATUS_UNIMPLEMENTED,
    gRPCClient.GRPC_INTERNAL => STATUS_INTERNAL,
    gRPCClient.GRPC_UNAVAILABLE => STATUS_UNAVAILABLE,
    gRPCClient.GRPC_DATA_LOSS => STATUS_DATA_LOSS,
    gRPCClient.GRPC_UNAUTHENTICATED => STATUS_UNAUTHENTICATED,
)

# A worker's gRPC connection: one per-worker libcurl multi handle (`grpc`) and the channel over it
# that every RPC to the worker uses. Each worker gets its own gRPCCURL: HTTP/2 multiplexing only
# reuses connections within one host:port, so workers share nothing, and a separate handle gives
# each its own GRPC_MAX_STREAMS (16) in-flight budget and request semaphore, isolating a slow or
# wedged worker to its own slots. The channel carries the data-plane defaults (request timeout,
# message-size limits); the probes and compaction override the deadline per call.
struct WorkerClients
    url::String
    grpc::gRPCClient.gRPCCURL
    chan::gRPCClient.gRPCChannel
    probe_deadline::Float64      # ServerReady, RepositoryIndex, ModelControlStatus, IsSameIPCNamespace
    compact_deadline::Float64    # ControlService/CompactMemory (memory-compaction fan-out)
end

struct ClientPool
    order::Vector{String}
    clients::Dict{String, WorkerClients}
end

# Build one worker's connection over a freshly-created multi handle.
function _worker_clients(cfg::GatewayConfig, url::AbstractString)
    host, port = _split_hostport(url)
    # sticky=true pins each handle's libcurl driving tasks (socket watchers + timer) to the thread
    # that owns them via @async rather than Threads.@spawn. The gateway terminates and forwards on
    # one event-loop thread, so sticky scheduling keeps the per-request driving on that thread and
    # avoids cross-thread handoff overhead on the hot path.
    # one running multi handle per worker; its stream semaphore caps outbound in-flight requests to
    # this worker (the rest block until a slot frees).
    grpc = gRPCClient.gRPCCURL(; sticky = true, max_streams = cfg.max_concurrent_streams_per_worker)
    chan = gRPCClient.gRPCChannel(
        host, port; grpc = grpc,
        deadline = cfg.request_timeout_seconds,
        max_send_message_length = cfg.max_send_msg_bytes,
        max_recieve_message_length = cfg.max_recv_msg_bytes
    )
    # Compaction frees and reloads weights (device-pinned models re-read from the mmap), so it can
    # run far longer than a status probe; give it a generous deadline rather than the 5s probe one.
    return WorkerClients(url, grpc, chan, 5.0, max(cfg.request_timeout_seconds, 300.0))
end

function ClientPool(cfg::GatewayConfig)
    order = String[]
    for url in cfg.workers
        url in order || push!(order, url)
    end
    clients = Dict(url => _worker_clients(cfg, url) for url in order)
    return ClientPool(order, clients)
end

get_clients(p::ClientPool, url::AbstractString) = get(p.clients, url, nothing)
all_clients(p::ClientPool) = [p.clients[u] for u in p.order]

# Reset a worker's multi handle in place: close it (dropping every pooled connection and aborting
# any in-flight request on it) and re-open it. The worker's channel holds this handle by
# identity, so it transparently uses the fresh handle afterward. This recovers from a poisoned
# connection: when a worker is caught mid-stall (TCP accepted but HTTP/2 not yet negotiated),
# libcurl keeps the half-open connection pooled and every later request reuses (PIPEWAITs on) it
# and hangs forever; only dropping the connection recovers it (the per-worker equivalent of a
# process restart). Call this when a probe to the worker *hangs* (times out or hits its deadline),
# not on a fast refuse.
function reset_clients!(wc::WorkerClients)
    try
        gRPCClient.grpc_shutdown(wc.grpc)
        gRPCClient.grpc_init(wc.grpc)
    catch e
        @warn "gateway: error resetting worker client handle" worker = wc.url exception = e
    end
    return nothing
end

# Shut down every worker's multi handle (closes connections, in-flight requests, the event-loop
# timer and socket watchers). Called on gateway shutdown; finalizers would otherwise reclaim them.
function close_pool!(p::ClientPool)
    for wc in all_clients(p)
        try
            gRPCClient.grpc_shutdown(wc.grpc)
        catch e
            @warn "gateway: error shutting down worker client handle" worker = wc.url exception = e
        end
    end
    return nothing
end

# `deadline` (seconds, or nothing) overrides the channel's default per-call timeout so the
# gateway can forward the request's REMAINING budget as grpc-timeout, decrementing it for the time
# already spent reaching the gateway rather than granting the worker a fresh full budget.
#
# A call on a handle that `reset_clients!` has shut down (a worker probe hit its deadline) fails
# with UNAVAILABLE, so it fails over like any other unavailable worker rather than reaching the
# client as the worker's stale shared-memory signal (FAILED_PRECONDITION). So does a call the
# transport failed (a refused or reset connection, a response that broke off): gRPCClient reports
# those as INTERNAL, the same code a worker sends for its own failures, so they are told apart
# here, where the call itself is at hand, and only the worker's INTERNAL reaches the client as such.
#
# `cancelled` (a zero-argument function, or nothing) is polled while the call is in flight: once
# it returns true (the gateway's own inbound call was cancelled), the worker call is cancelled with
# `grpc_cancel` and fails with CANCELLED.
#
# `label`, when given, is how a client-visible message names this worker: a failure of the call
# itself (not a status the worker sent) is then described as "<label>: <what happened>" instead of
# with gRPCClient's text, which can name the worker's host and port (a redacted gateway, GW-15).
function invoke_infer(
        wc::WorkerClients, body::Vector{UInt8};
        deadline::Union{Real, Nothing} = nothing, cancelled = nothing,
        label::Union{AbstractString, Nothing} = nothing
    )
    rpc = try
        deadline === nothing ? GRPCInferenceService.ModelInfer(wc.chan, body, gRPCClient.gRPCAsync()) :
            GRPCInferenceService.ModelInfer(wc.chan, body, gRPCClient.gRPCAsync(); deadline = deadline)
    catch e
        (label !== nothing && e isa gRPCClient.gRPCServiceCallException) && throw(_relabel(e, label))
        rethrow()
    end
    req = rpc.req
    watch = cancelled === nothing ? nothing : _cancel_when(req, cancelled)
    try
        return gRPCClient.grpc_async_await(req, Vector{UInt8})
    catch e
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        if _transport_failure(req, e)
            e = gRPCClient.gRPCServiceCallException(gRPCClient.GRPC_UNAVAILABLE, "transport error: $(e.message)")
        end
        (label === nothing || _worker_sent(req)) || (e = _relabel(e, label))
        throw(e)
    finally
        watch === nothing || close(watch)
    end
end

# Whether a finished call's status is one the worker sent (as opposed to one recorded locally: a
# deadline, a cancellation, a reset handle, a transport failure).
_worker_sent(req::gRPCClient.gRPCRequest) = req.ex === nothing && req.grpc_status != gRPCClient.GRPC_OK

# A local failure of the call to the worker `label`, described without gRPCClient's own text.
function _relabel(e::gRPCClient.gRPCServiceCallException, label::AbstractString)
    c = e.grpc_status
    what = c == gRPCClient.GRPC_DEADLINE_EXCEEDED ? "the call reached its deadline" :
        c == gRPCClient.GRPC_CANCELLED ? "the call was cancelled" :
        c == gRPCClient.GRPC_UNAVAILABLE ? "the call failed in the transport (refused, reset, or abandoned)" :
        c == gRPCClient.GRPC_RESOURCE_EXHAUSTED ? "the message exceeded the gateway's size limit" :
        "the call failed ($(get(gRPCClient.GRPC_CODE_TABLE, c, "status $c")))"
    return gRPCClient.gRPCServiceCallException(c, "$label: $what")
end

# How often an in-flight worker call checks whether the gateway's inbound call was cancelled.
const _CANCEL_POLL_SECONDS = 0.1

# A repeating timer that cancels `req` once `cancelled()` returns true. Close it when the call ends.
function _cancel_when(req::gRPCClient.gRPCRequest, cancelled)
    return Timer(_CANCEL_POLL_SECONDS; interval = _CANCEL_POLL_SECONDS) do t
        fired = try
            cancelled() === true
        catch
            false
        end
        fired || return
        close(t)
        gRPCClient.grpc_cancel(req)
    end
end

# Whether a failed call ended in the transport rather than with a status the worker sent. A worker's
# status arrives as the call's `grpc-status` with no local exception recorded; a call that broke off
# records none and has no status, and libcurl's error comes back as INTERNAL. A call failed locally for
# any other reason (its deadline, a cancellation, a shut-down handle) keeps the code it was given.
_transport_failure(req::gRPCClient.gRPCRequest, e::gRPCClient.gRPCServiceCallException) =
    e.grpc_status == gRPCClient.GRPC_INTERNAL && req.ex === nothing && !_worker_sent(req)
invoke_shm_register(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryRegister(wc.chan, body, Vector{UInt8})
invoke_shm_unregister(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryUnregister(wc.chan, body, Vector{UInt8})
# Ask the worker which regions named `name` it has registered; returns its
# SystemSharedMemoryStatusResponse. Used by the register fan-out to learn what existed before it.
invoke_shm_status(wc::WorkerClients, name::AbstractString) =
    GRPCInferenceService.SystemSharedMemoryStatus(
    wc.chan, SystemSharedMemoryStatusRequest(; name = String(name)); deadline = wc.probe_deadline
)

# Forward a CompactMemory request to the worker; returns its CompactMemoryResponse.
invoke_compact(wc::WorkerClients, req::CompactMemoryRequest) =
    ControlService.CompactMemory(wc.chan, req; deadline = wc.compact_deadline)

# Ask the worker to read models' weights into host memory ahead of a placement change and hold them
# for `hold` seconds (StageWeights); returns its StageWeightsResponse. The reads take as long as the
# disk does, so the caller passes a deadline sized for them rather than the probe deadline.
invoke_stage_weights(wc::WorkerClients, names::Vector{String}, hold::Real, deadline::Real) =
    ControlService.StageWeights(
    wc.chan, StageWeightsRequest(; names = names, hold_seconds = Float64(hold)); deadline = Float64(deadline)
)

# Forward an IsSameIPCNamespace probe to the worker; returns its IsSameIPCNamespaceResponse.
invoke_is_same_ipc_namespace(wc::WorkerClients, req::IsSameIPCNamespaceRequest) =
    GRPCInferenceService.IsSameIPCNamespace(wc.chan, req; deadline = wc.probe_deadline)

# The metadata RPCs a client sends the gateway, forwarded with the probe deadline: they read the
# worker's registry and never wait on the GPU.
invoke_describe_models(wc::WorkerClients, req::DescribeModelsRequest) =
    ControlService.DescribeModels(wc.chan, req; deadline = wc.probe_deadline)
invoke_model_metadata(wc::WorkerClients, req::ModelMetadataRequest) =
    GRPCInferenceService.ModelMetadata(wc.chan, req; deadline = wc.probe_deadline)
invoke_repository_index(wc::WorkerClients, req::RepositoryIndexRequest) =
    GRPCInferenceService.RepositoryIndex(wc.chan, req; deadline = wc.probe_deadline)

# A call that failed at its gRPC deadline: the connection stalled rather than refused. The probe
# helpers below rethrow it so `_bounded` can reset the worker's connection (see there).
_deadline_exceeded(e) =
    e isa gRPCClient.gRPCServiceCallException && e.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED

# Returns the worker's ServerReadyResponse.ready, or false on any other transport error.
function probe_ready(wc::WorkerClients)
    try
        resp = GRPCInferenceService.ServerReady(wc.chan, ServerReadyRequest(); deadline = wc.probe_deadline)
        return resp.ready
    catch e
        _deadline_exceeded(e) && rethrow()
        return false
    end
end

# Poll the worker's ModelControlStatus (discipline, residency mode, per-model serving counters).
# Returns the response, or `nothing` on any other transport error.
function fetch_control_status(wc::WorkerClients)
    try
        return ControlService.ModelControlStatus(wc.chan, ModelControlStatusRequest(); deadline = wc.probe_deadline)
    catch e
        _deadline_exceeded(e) && rethrow()
        return nothing
    end
end

# Query the worker's ready models for discovery. Returns the names this worker reports READY, or
# `nothing` on any other transport error (so the caller can leave the worker's routes untouched).
function discover_models(wc::WorkerClients)
    try
        resp = GRPCInferenceService.RepositoryIndex(
            wc.chan, RepositoryIndexRequest(; ready = true); deadline = wc.probe_deadline
        )
        return String[m.name for m in resp.models if m.state == "READY"]
    catch e
        _deadline_exceeded(e) && rethrow()
        return nothing
    end
end

# Map a client-side gRPC status code to the gateway's status string for the same code. Every code
# keeps its identity: a worker's FAILED_PRECONDITION (a stale shared-memory registration) must reach
# the client so it re-registers, its DEADLINE_EXCEEDED so the client stops rather than retries, its
# INVALID_ARGUMENT because every replica would reject the same request, and its INTERNAL because a
# deterministic failure would only fail again elsewhere. A transport failure arrives here already as
# UNAVAILABLE (see `invoke_infer`). Which statuses fail over is `_fails_over`'s call.
client_status(e::gRPCClient.gRPCServiceCallException) = get(_STATUS_BY_CODE, Int(e.grpc_status), STATUS_UNKNOWN)

# --- RegisterGate -----------------------------------------------------------------------------

struct RegisterGate
    lock::ReentrantLock
    inflight::Dict{String, Base.Event}
end
RegisterGate() = RegisterGate(ReentrantLock(), Dict{String, Base.Event}())

# Mark a SHM operation in flight against `url`; returns a function that releases the gate.
function gate_begin!(g::RegisterGate, url::AbstractString)
    ev = Base.Event()
    lock(g.lock) do
        g.inflight[String(url)] = ev
    end
    return function ()
        lock(g.lock) do
            delete!(g.inflight, String(url))
        end
        return notify(ev)
    end
end

# Block until any in-flight SHM operation against `url` completes.
function gate_wait(g::RegisterGate, url::AbstractString)
    ev = lock(g.lock) do
        get(g.inflight, String(url), nothing)
    end
    ev === nothing || wait(ev)
    return nothing
end
