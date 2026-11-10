# Per-worker gRPC connections used to forward requests to the Julia inference workers. ModelInfer
# and the SHM RPCs send and receive raw Vector{UInt8} so request and response bytes pass through
# without re-marshaling; the probes are typed (the messages are tiny). Each worker has two
# gRPCChannels, each over its own libcurl multi handle, built up front and reused (see
# `WorkerClients`). The RegisterGate serializes ModelInfer against a worker while a SHM
# register/unregister to that worker is in flight (POSIX shm registration is host-global).

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

# A worker's gRPC connections: two libcurl multi handles of its own, each with the channel over it.
# A handle is a connection pool and a stream budget (`max_streams` slots, past which calls wait), and
# HTTP/2 multiplexing reuses connections only within one handle and one host:port, so workers share
# nothing and a slow or wedged worker is isolated to its own slots.
#
#   - `grpc`/`chan`, `max_concurrent_streams_per_worker` slots: ModelInfer, and every other call the
#     gateway makes for a client or for placement (shared memory, the metadata RPCs,
#     IsSameIPCNamespace, CompactMemory, StageWeights). Its channel carries the data-plane defaults
#     (request timeout, message-size limits); the others override the deadline per call.
#   - `probe_grpc`/`probe_chan`: the prober's own polling (ServerReady, RepositoryIndex discovery and
#     ModelControlStatus) and nothing else. A call's gRPCClient deadline counts the time it waits for
#     a slot, so on a shared handle a probe waited behind every inference call queued at a busy
#     worker, timed out although the worker was healthy, and the reset that followed aborted all of
#     them; three such rounds made the gateway exit (GW-1). On its own handle a probe waits for no
#     inference, so a probe that times out means the worker (or the gateway's client stack) stopped
#     answering.
#
# CompactMemory and StageWeights stay on the first handle. They run for minutes, and the probe handle
# is reset whenever a probe hangs, which is when a worker is busy with long work like theirs: on the
# probe handle they would be aborted. Each holds one of the first handle's slots while it runs, one per
# worker per repack.
struct WorkerClients
    url::String
    grpc::gRPCClient.gRPCCURL
    chan::gRPCClient.gRPCChannel
    probe_grpc::gRPCClient.gRPCCURL
    probe_chan::gRPCClient.gRPCChannel
    probe_deadline::Float64      # ServerReady, RepositoryIndex, ModelControlStatus, IsSameIPCNamespace
    compact_deadline::Float64    # ControlService/CompactMemory (memory-compaction fan-out)
    # ModelInfer calls in a row that ended at their deadline without an answer, and the resets of
    # `grpc` they caused; see `reset_if_parked!`.
    unanswered::Threads.Atomic{Int}
    resets::Threads.Atomic{Int}
end

# One handle for every call, for a test that drives `invoke_infer` over a handle of its own.
WorkerClients(url, grpc, chan, probe_deadline, compact_deadline) = WorkerClients(
    url, grpc, chan, grpc, chan, probe_deadline, compact_deadline, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
)

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
    options = (;
        deadline = cfg.request_timeout_seconds,
        max_send_message_length = cfg.max_send_msg_bytes,
        max_recieve_message_length = cfg.max_recv_msg_bytes,
    )
    chan = gRPCClient.gRPCChannel(host, port; grpc = grpc, options...)
    # The prober has a few calls at most in flight to one worker (a probe, a discovery, a poll, and an
    # on-demand discovery), so gRPCClient's default stream budget is ample.
    probe_grpc = gRPCClient.gRPCCURL(; sticky = true)
    probe_chan = gRPCClient.gRPCChannel(host, port; grpc = probe_grpc, options...)
    # Compaction frees and reloads weights (device-pinned models re-read from the mmap), so it can
    # run far longer than a status probe; give it a generous deadline rather than the 5s probe one.
    return WorkerClients(
        url, grpc, chan, probe_grpc, probe_chan, 5.0, max(cfg.request_timeout_seconds, 300.0),
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
    )
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

# Reset a multi handle in place: close it (dropping every pooled connection and aborting any
# in-flight request on it) and re-open it. A channel holds its handle by identity, so it
# transparently uses the fresh handle afterward: the per-worker equivalent of a process restart.
function _reset_handle!(grpc::gRPCClient.gRPCCURL, url::AbstractString)
    try
        gRPCClient.grpc_shutdown(grpc)
        gRPCClient.grpc_init(grpc)
    catch e
        @warn "gateway: error resetting worker client handle" worker = url exception = e
    end
    return nothing
end

# Reset a worker's probe handle. Called when a probe or control-status poll to the worker *hangs*
# (times out or hits its deadline), not on a fast refuse: the probe connection may be one that never
# answered, which every later probe would wait behind (see `reset_if_parked!`). Only the prober's own
# calls use this handle, so no inference is touched.
reset_probe_connection!(wc::WorkerClients) = _reset_handle!(wc.probe_grpc, wc.url)

# ModelInfer calls in a row to one worker that must end at their deadline with no answer from it
# before `reset_if_parked!` resets the worker's inference connection.
const PARKED_AFTER_UNANSWERED = 5

"""
    reset_if_parked!(wc) -> Bool

Reset the worker's inference handle (`wc.grpc`) when its connection is parked: the last
`PARKED_AFTER_UNANSWERED` ModelInfer calls to the worker all ended at their deadline with no answer
from it. The prober calls this only for a worker that has just answered its ServerReady probe, ready,
on its own handle, so the worker is up and the fault is in the inference connection.

It exists for one gRPCClient 1.2.1 behavior (GW-18): gRPCClient sets CURLOPT_PIPEWAIT on every call,
and with it libcurl does not use a pooled HTTP/2 connection until a call on it has had a response,
and makes a new call wait for that connection instead of opening another, even when no call is
using it. When every call on a new connection is given up before the worker answers (a restarted
worker whose first answers are slow, or a worker frozen while the connection opened), the connection
stays pooled unanswered, and every later call to the worker waits behind it and fails at its own
deadline, without reaching the worker, until libcurl drops the connection after 118 s idle
(CURLOPT_MAXAGE_CONN). Closing the handle drops it at once, and the calls parked on it fail over
(they were never sent). Delete this, and the counting in `invoke_infer`, once gRPCClient no longer
parks calls behind a connection that never answered.

Healthy traffic never resets the handle: any answer from the worker, a success or a status it sent,
clears the count, and a call that expired waiting for one of the handle's slots (the worker busy with
the calls ahead of it) does not count. A worker that is slow but answering keeps its calls in flight.
What can still look parked is a worker that answers its probes but none of the calls it receives
before their deadlines; resetting then fails over the calls in flight to it, which a worker in that
state would not have answered in time either.
"""
function reset_if_parked!(wc::WorkerClients)
    n = wc.unanswered[]
    n >= PARKED_AFTER_UNANSWERED || return false
    @warn "gateway: the last $n inference calls to a worker that answers its health probes all reached their deadline unanswered; resetting its inference connection (gRPCClient parks calls behind a connection that never answered)" worker = wc.url
    _reset_handle!(wc.grpc, wc.url)
    wc.unanswered[] = 0
    Threads.atomic_add!(wc.resets, 1)
    return true
end

"""
    reset_returning_connections!(wc)

Reset the worker's data handle (`wc.grpc`: inference, shared memory, the metadata RPCs, StageWeights
and CompactMemory) when the prober finds the worker back from an outage, before it is marked healthy.

The worker may be a new process at the same address (killed and restarted), and the connections the
handle pooled for the old one are then dead. libcurl takes an idle HTTP/2 connection for alive when
its server went away without a GOAWAY, which an exit or a kill never sends, and the first call on it
fails to send ("Send failure: Broken pipe"). Under lpt_packing that call was the StageWeights of the
membership repack that gave the returning worker its replicas back: it failed at once, the changes
were dropped, and on an idle fleet nothing planned them again, so the worker was healthy but placed
nothing (CPU e2e step 11). The probe handle needs no reset: the probe that found the worker down
failed on its dead connection, and libcurl dropped it then.

A worker that was only hung keeps its process; it costs one connection setup. A call still in flight
on the handle, sent before the worker was marked down, is aborted and fails over like a call to an
unavailable worker. Not counted in `wc.resets`, which counts the resets for parked connections
(`reset_if_parked!`).
"""
function reset_returning_connections!(wc::WorkerClients)
    @debug "health: a worker back from an outage gets fresh data connections" worker = wc.url
    _reset_handle!(wc.grpc, wc.url)
    wc.unanswered[] = 0
    return nothing
end

# Record how a ModelInfer call ended, for `reset_if_parked!`: any answer from the worker clears the
# count of calls in a row that ended at their deadline unanswered, and such a call adds one. A call
# that expired waiting for a slot on the handle (gRPCClient's "Deadline exceeded while queued ...")
# never left the gateway, so it says nothing about the connection.
function _note_infer_outcome!(wc::WorkerClients, req::gRPCClient.gRPCRequest, e)
    if e === nothing || _worker_sent(req)
        wc.unanswered[] = 0
    elseif e.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED &&
            !startswith(e.message, "Deadline exceeded while queued")
        Threads.atomic_add!(wc.unanswered, 1)
    end
    return nothing
end

# Shut down every worker's multi handles (closes connections, in-flight requests, the event-loop
# timer and socket watchers). Called on gateway shutdown; finalizers would otherwise reclaim them.
function close_pool!(p::ClientPool)
    for wc in all_clients(p), grpc in unique((wc.grpc, wc.probe_grpc))
        try
            gRPCClient.grpc_shutdown(grpc)
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
# A call on a handle that `reset_if_parked!` has shut down fails with UNAVAILABLE, so it fails over
# like any other unavailable worker rather than reaching the client as the worker's stale
# shared-memory signal (FAILED_PRECONDITION). So does a call the
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
#
# `refused`, when given, is set to whether the call failed because the connection to the worker was
# refused (see `_connection_refused`): the request never left the gateway.
function invoke_infer(
        wc::WorkerClients, body::Vector{UInt8};
        deadline::Union{Real, Nothing} = nothing, cancelled = nothing,
        label::Union{AbstractString, Nothing} = nothing,
        refused::Union{Base.RefValue{Bool}, Nothing} = nothing
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
        resp = gRPCClient.grpc_async_await(req, Vector{UInt8})
        _note_infer_outcome!(wc, req, nothing)
        return resp
    catch e
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        _note_infer_outcome!(wc, req, e)
        if _transport_failure(req, e)
            refused === nothing || (refused[] = _connection_refused(req))
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

"""
    _connection_refused(req) -> Bool

Whether a failed call ended because the connection to the worker could not be opened, so that no
byte of the request reached it. gRPCClient 1.2.1 keeps libcurl's result code on the request
(`req.code`) and reports a transport failure as INTERNAL with libcurl's error text; this reads the
code: `CURLE_COULDNT_CONNECT` (7), which libcurl returns only when connect() itself failed. A
connect that runs out of time is `CURLE_OPERATION_TIMEDOUT` (28) or, past the call's deadline,
DEADLINE_EXCEEDED; a name that does not resolve is `CURLE_COULDNT_RESOLVE_HOST` (6); a connection
reset or broken off after it was made is a send or receive error (55, 56) or an HTTP/2 error (16,
92). None of those match.

What this cannot tell apart is why connect() failed. libcurl 8.15 (the production lock) writes
"Failed to connect to <host> port <port> after <n> ms: Could not connect to server" for a refusal,
with no errno in the text, and gRPCClient frees the easy handle before the call returns, so
`CURLINFO_OS_ERRNO` cannot be read. The match therefore also covers the rarer connect() failures
with no route to the host (EHOSTUNREACH, ENETUNREACH), and a refused connect to a proxy, should the
environment configure one; in every case the request was not sent. The message text is not used: it
varies across libcurl versions (older ones end with "Connection refused") and carries nothing this
code does not.
"""
_connection_refused(req::gRPCClient.gRPCRequest) =
    req.ex === nothing && req.code == gRPCClient.CURLE_COULDNT_CONNECT
invoke_shm_register(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryRegister(wc.chan, body, Vector{UInt8})
invoke_shm_unregister(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryUnregister(wc.chan, body, Vector{UInt8})
# Ask the worker which regions named `name` it has registered; returns its
# SystemSharedMemoryStatusResponse. Used by the register fan-out to learn what existed before it.
# Unregister `name` on a worker for the prober's catch-up of a returning worker (`catch_up_shm!`).
# It goes over the probe handle: this round's probe has just reached the worker on it, whereas the
# inference handle may still hold a connection to the process that went away, whose first call
# would fail. It is a short call, like the probes.
invoke_shm_catch_up(wc::WorkerClients, body::Vector{UInt8}) =
    GRPCInferenceService.SystemSharedMemoryUnregister(wc.probe_chan, body, Vector{UInt8}; deadline = wc.probe_deadline)
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
        resp = GRPCInferenceService.ServerReady(wc.probe_chan, ServerReadyRequest(); deadline = wc.probe_deadline)
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
        return ControlService.ModelControlStatus(wc.probe_chan, ModelControlStatusRequest(); deadline = wc.probe_deadline)
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
            wc.probe_chan, RepositoryIndexRequest(; ready = true); deadline = wc.probe_deadline
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

# The gateway's per-worker shared-memory bookkeeping: which register or unregister is in flight
# against each worker (requests to it wait for it, `gate_wait`), and which region names each worker
# missed while it was marked down (`missed`, see `note_missed!`).
struct RegisterGate
    lock::ReentrantLock
    inflight::Dict{String, Base.Event}
    missed::Dict{String, Set{String}}
end
RegisterGate() = RegisterGate(ReentrantLock(), Dict{String, Base.Event}(), Dict{String, Set{String}}())

# How many distinct names a worker's missed record holds before it collapses to "every region" (""),
# which keeps it bounded however many regions clients register during one outage.
const MISSED_NAMES_MAX = 1024

"""
    note_missed!(gate, urls, name)

Record that the workers at `urls`, marked down, were skipped by a register or unregister of the
region `name` ("" for every region). A down worker that is still alive keeps whatever it had
registered under that name, which may be an object the client has since unlinked and replaced, and a
request under the name would then read and write the old object without an error. So before the
prober lets such a worker take requests again, it unregisters every name the worker missed
(`catch_up_shm!`); the worker then knows none of them, and its first request for one fails with
FAILED_PRECONDITION, on which a client registers again (ReactantServerClient does so by itself).
"""
function note_missed!(g::RegisterGate, urls, name::AbstractString)
    isempty(urls) && return nothing
    lock(g.lock) do
        for url in urls
            s = get!(Set{String}, g.missed, String(url))
            "" in s && continue                     # already "every region"
            if isempty(name) || length(s) >= MISSED_NAMES_MAX
                empty!(s)
                push!(s, "")
            else
                push!(s, String(name))
            end
        end
    end
    return nothing
end

# The names the worker at `url` missed, as recorded so far (a copy).
missed_names(g::RegisterGate, url::AbstractString) =
    lock(() -> collect(String, get(g.missed, url, Set{String}())), g.lock)

# Forget the names the worker at `url` has been told (those `missed_names` returned, all now
# unregistered there). A name recorded again meanwhile is forgotten too: the worker was still down,
# so it holds nothing under it.
function clear_missed!(g::RegisterGate, url::AbstractString, names)
    lock(g.lock) do
        s = get(g.missed, url, nothing)
        s === nothing && return
        setdiff!(s, names)
        isempty(s) && delete!(g.missed, url)
    end
    return nothing
end

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
