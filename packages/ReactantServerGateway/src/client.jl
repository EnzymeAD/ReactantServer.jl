# Per-worker gRPC clients used to forward requests to the Julia inference workers. ModelInfer and
# the SHM RPCs use raw Vector{UInt8} clients so request and response bytes pass through without
# re-marshaling; ServerReady uses a typed client (the message is tiny). Clients are lightweight
# value structs over a shared libcurl multi handle, so one is built per worker and method up front
# and reused. The RegisterGate serializes ModelInfer against a worker while a SHM
# register/unregister to that worker is in flight (POSIX shm registration is host-global).

# Status strings surfaced to clients and recorded in metrics, mirroring the Go gateway.
const STATUS_OK = "OK"
const STATUS_NOT_FOUND = "NotFound"
const STATUS_UNAVAILABLE = "Unavailable"
const STATUS_RESOURCE_EXHAUSTED = "ResourceExhausted"
const STATUS_INVALID = "InvalidArgument"
const STATUS_FAILED_PRE = "FailedPrecondition"
const STATUS_INTERNAL = "Internal"
const STATUS_DEADLINE = "DeadlineExceeded"

# A worker's gRPC clients, all sharing one per-worker libcurl multi handle (`grpc`). Each worker
# gets its own gRPCCURL: HTTP/2 multiplexing only reuses connections within one host:port, so
# workers share nothing, and a separate handle gives each its own GRPC_MAX_STREAMS (16) in-flight
# budget and request semaphore, isolating a slow or wedged worker to its own slots. The struct is
# parametric so every field keeps its concrete client type (the request path forwards through
# `infer` with no dynamic dispatch); the type parameters are inferred at construction.
struct WorkerClients{G,I,SR,SU,RD,RI,CS,CM}
    url::String
    grpc::G
    infer::I
    shm_register::SR
    shm_unregister::SU
    ready::RD
    repo_index::RI
    control_status::CS    # ControlService/ModelControlStatus (lpt_packing cost polling + preconditions)
    compact::CM           # ControlService/CompactMemory (memory-compaction fan-out)
end

struct ClientPool{W<:WorkerClients}
    order::Vector{String}
    clients::Dict{String,W}
end

# Build one worker's clients over a freshly-created multi handle.
function _worker_clients(cfg::GatewayConfig, url::AbstractString)
    host, port = _split_hostport(url)
    # sticky=true pins each handle's libcurl driving tasks (socket watchers + timer) to the thread
    # that owns them via @async rather than Threads.@spawn. The gateway terminates and forwards on
    # one event-loop thread, so sticky scheduling keeps the per-request driving on that thread and
    # avoids cross-thread handoff overhead on the hot path.
    # one running multi handle per worker; its stream semaphore caps outbound in-flight requests to
    # this worker (the rest block until a slot frees).
    grpc = gRPCClient.gRPCCURL(; sticky = true, max_streams = cfg.max_concurrent_streams_per_worker)
    infer = GRPCInferenceService_ModelInfer_Client(host, port; grpc = grpc,
        TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
        deadline = cfg.request_timeout_seconds,
        max_send_message_length = cfg.max_send_msg_bytes,
        max_recieve_message_length = cfg.max_recv_msg_bytes)
    shm_reg = GRPCInferenceService_SystemSharedMemoryRegister_Client(host, port; grpc = grpc,
        TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
        deadline = cfg.request_timeout_seconds)
    shm_unreg = GRPCInferenceService_SystemSharedMemoryUnregister_Client(host, port; grpc = grpc,
        TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
        deadline = cfg.request_timeout_seconds)
    ready = GRPCInferenceService_ServerReady_Client(host, port; grpc = grpc, deadline = 5)
    repo_index = GRPCInferenceService_RepositoryIndex_Client(host, port; grpc = grpc, deadline = 5)
    control_status = ControlService_ModelControlStatus_Client(host, port; grpc = grpc, deadline = 5)
    # Compaction frees and reloads weights (device-pinned models re-read from the mmap), so it can
    # run far longer than a status probe; give it a generous deadline rather than the 5s probe one.
    compact = ControlService_CompactMemory_Client(host, port; grpc = grpc,
        deadline = max(cfg.request_timeout_seconds, 300))
    return WorkerClients(url, grpc, infer, shm_reg, shm_unreg, ready, repo_index, control_status, compact)
end

function ClientPool(cfg::GatewayConfig)
    order = String[]
    for url in cfg.workers
        url in order || push!(order, url)
    end
    wcs = map(url -> _worker_clients(cfg, url), order)   # Vector{W}, W concrete and identical
    clients = Dict(wc.url => wc for wc in wcs)           # Dict{String,W}
    return ClientPool(order, clients)
end

get_clients(p::ClientPool, url::AbstractString) = get(p.clients, url, nothing)
all_clients(p::ClientPool) = [p.clients[u] for u in p.order]

# Reset a worker's multi handle in place: close it (dropping every pooled connection and aborting
# any in-flight request on it) and re-open it. The worker's client stubs hold this handle by
# identity, so they transparently use the fresh handle afterward. This recovers from a poisoned
# connection: when a worker is caught mid-stall (TCP accepted but HTTP/2 not yet negotiated),
# libcurl keeps the half-open connection pooled and every later request reuses (PIPEWAITs on) it
# and hangs forever; only dropping the connection recovers it (the per-worker equivalent of a
# process restart). Call this when a probe to the worker *hangs* (times out), not on a fast refuse.
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

# `deadline` (seconds, or nothing) overrides the worker client's default per-call timeout so the
# gateway can forward the request's REMAINING budget as grpc-timeout — decrementing it for the time
# already spent reaching the gateway, rather than granting the worker a fresh full budget.
function invoke_infer(wc::WorkerClients, body::Vector{UInt8}; deadline::Union{Real,Nothing}=nothing)
    deadline === nothing && return gRPCClient.grpc_sync_request(wc.infer, body)
    req = gRPCClient.grpc_async_request(wc.infer, body; deadline=deadline)
    return gRPCClient.grpc_async_await(wc.infer, req)
end
invoke_shm_register(wc::WorkerClients, body::Vector{UInt8}) = gRPCClient.grpc_sync_request(wc.shm_register, body)
invoke_shm_unregister(wc::WorkerClients, body::Vector{UInt8}) = gRPCClient.grpc_sync_request(wc.shm_unregister, body)

# Forward a CompactMemory request to the worker; returns its CompactMemoryResponse.
invoke_compact(wc::WorkerClients, req::CompactMemoryRequest) = gRPCClient.grpc_sync_request(wc.compact, req)

# Returns the worker's ServerReadyResponse.ready, or false on any transport error.
function probe_ready(wc::WorkerClients)
    try
        resp = gRPCClient.grpc_sync_request(wc.ready, ServerReadyRequest())
        return resp.ready
    catch
        return false
    end
end

# Poll the worker's ModelControlStatus (discipline, residency mode, per-model serving counters).
# Returns the response, or `nothing` on any transport error.
function fetch_control_status(wc::WorkerClients)
    try
        return gRPCClient.grpc_sync_request(wc.control_status, ModelControlStatusRequest())
    catch
        return nothing
    end
end

# Query the worker's ready models for discovery. Returns the names this worker reports READY, or
# `nothing` on any transport error (so the caller can leave the worker's routes untouched).
function discover_models(wc::WorkerClients)
    try
        resp = gRPCClient.grpc_sync_request(wc.repo_index, RepositoryIndexRequest(; ready = true))
        return String[m.name for m in resp.models if m.state == "READY"]
    catch
        return nothing
    end
end

# Map a client-side gRPC status code to one of the gateway's status strings.
function client_status(e::gRPCClient.gRPCServiceCallException)
    st = e.grpc_status
    st == gRPCClient.GRPC_NOT_FOUND && return STATUS_NOT_FOUND
    st == gRPCClient.GRPC_RESOURCE_EXHAUSTED && return STATUS_RESOURCE_EXHAUSTED
    return STATUS_UNAVAILABLE
end

# --- RegisterGate -----------------------------------------------------------------------------

struct RegisterGate
    lock::ReentrantLock
    inflight::Dict{String,Base.Event}
end
RegisterGate() = RegisterGate(ReentrantLock(), Dict{String,Base.Event}())

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
        notify(ev)
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
