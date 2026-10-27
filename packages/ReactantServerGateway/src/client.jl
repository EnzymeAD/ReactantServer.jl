# Per-worker gRPC connections used to forward requests to the Julia inference workers. ModelInfer
# and the SHM RPCs send and receive raw Vector{UInt8} so request and response bytes pass through
# without re-marshaling; the probes are typed (the messages are tiny). Each worker has one
# gRPCChannel over its own libcurl multi handle, built up front and reused. The RegisterGate
# serializes ModelInfer against a worker while a SHM register/unregister to that worker is in flight
# (POSIX shm registration is host-global).

# Status strings surfaced to clients and recorded in metrics, mirroring the Go gateway.
const STATUS_OK = "OK"
const STATUS_NOT_FOUND = "NotFound"
const STATUS_UNAVAILABLE = "Unavailable"
const STATUS_RESOURCE_EXHAUSTED = "ResourceExhausted"
const STATUS_INVALID = "InvalidArgument"
const STATUS_FAILED_PRE = "FailedPrecondition"
const STATUS_INTERNAL = "Internal"
const STATUS_DEADLINE = "DeadlineExceeded"

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
# client as the worker's stale shared-memory signal (FAILED_PRECONDITION).
function invoke_infer(wc::WorkerClients, body::Vector{UInt8}; deadline::Union{Real, Nothing} = nothing)
    return deadline === nothing ? GRPCInferenceService.ModelInfer(wc.chan, body, Vector{UInt8}) :
        GRPCInferenceService.ModelInfer(wc.chan, body, Vector{UInt8}; deadline = deadline)
end
invoke_shm_register(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryRegister(wc.chan, body, Vector{UInt8})
invoke_shm_unregister(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryUnregister(wc.chan, body, Vector{UInt8})

# Forward a CompactMemory request to the worker; returns its CompactMemoryResponse.
invoke_compact(wc::WorkerClients, req::CompactMemoryRequest) =
    ControlService.CompactMemory(wc.chan, req; deadline = wc.compact_deadline)

# Forward an IsSameIPCNamespace probe to the worker; returns its IsSameIPCNamespaceResponse.
invoke_is_same_ipc_namespace(wc::WorkerClients, req::IsSameIPCNamespaceRequest) =
    GRPCInferenceService.IsSameIPCNamespace(wc.chan, req; deadline = wc.probe_deadline)

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

# Map a client-side gRPC status code to one of the gateway's status strings.
function client_status(e::gRPCClient.gRPCServiceCallException)
    st = e.grpc_status
    st == gRPCClient.GRPC_NOT_FOUND && return STATUS_NOT_FOUND
    st == gRPCClient.GRPC_RESOURCE_EXHAUSTED && return STATUS_RESOURCE_EXHAUSTED
    # A stale shared-memory registration (region unknown after a worker restart) comes back as
    # FAILED_PRECONDITION; forward it verbatim so the client can re-register and retry, rather than
    # folding it into UNAVAILABLE (which the client would blindly retry against the same stale region).
    st == gRPCClient.GRPC_FAILED_PRECONDITION && return STATUS_FAILED_PRE
    # A deadline observed at the worker (admission shed, server-enforced grpc-timeout, or a
    # local curl timeout) is a distinct outcome from a worker being down; give it its own status
    # so metrics and logs do not fold it into STATUS_UNAVAILABLE.
    st == gRPCClient.GRPC_DEADLINE_EXCEEDED && return STATUS_DEADLINE
    return STATUS_UNAVAILABLE
end

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
