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

struct WorkerClients
    url::String
    infer::Any
    shm_register::Any
    shm_unregister::Any
    ready::Any
    repo_index::Any
    control_status::Any    # ControlService/ModelControlStatus (lpt_packing cost polling + preconditions)
end

struct ClientPool
    order::Vector{String}
    clients::Dict{String,WorkerClients}
end

function ClientPool(cfg::GatewayConfig)
    clients = Dict{String,WorkerClients}()
    order = String[]
    for url in cfg.workers
        haskey(clients, url) && continue
        host, port = _split_hostport(url)
        infer = GRPCInferenceService_ModelInfer_Client(host, port;
            TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
            deadline = cfg.request_timeout_seconds,
            max_send_message_length = cfg.max_send_msg_bytes,
            max_recieve_message_length = cfg.max_recv_msg_bytes)
        shm_reg = GRPCInferenceService_SystemSharedMemoryRegister_Client(host, port;
            TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
            deadline = cfg.request_timeout_seconds)
        shm_unreg = GRPCInferenceService_SystemSharedMemoryUnregister_Client(host, port;
            TRequest = Vector{UInt8}, TResponse = Vector{UInt8},
            deadline = cfg.request_timeout_seconds)
        ready = GRPCInferenceService_ServerReady_Client(host, port; deadline = 5)
        repo_index = GRPCInferenceService_RepositoryIndex_Client(host, port; deadline = 5)
        control_status = ControlService_ModelControlStatus_Client(host, port; deadline = 5)
        clients[url] = WorkerClients(url, infer, shm_reg, shm_unreg, ready, repo_index, control_status)
        push!(order, url)
    end
    return ClientPool(order, clients)
end

get_clients(p::ClientPool, url::AbstractString) = get(p.clients, url, nothing)
all_clients(p::ClientPool) = WorkerClients[p.clients[u] for u in p.order]

invoke_infer(wc::WorkerClients, body::Vector{UInt8}) = gRPCClient.grpc_sync_request(wc.infer, body)
invoke_shm_register(wc::WorkerClients, body::Vector{UInt8}) = gRPCClient.grpc_sync_request(wc.shm_register, body)
invoke_shm_unregister(wc::WorkerClients, body::Vector{UInt8}) = gRPCClient.grpc_sync_request(wc.shm_unregister, body)

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
