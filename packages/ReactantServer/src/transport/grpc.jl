# The worker's gRPC server: the KServe V2 inference API and the control service. This is the only
# file that imports gRPCServer.
#
# gRPCServer hands each handler the already-decoded request message and serializes the
# message the handler returns, so the codec, scheduler, and runtime are untouched: the codec
# translates between wire messages and the boundary types, never touching framing. The system
# shared-memory extension RPCs register, unregister, and report client-provided regions, and
# RepositoryIndex lists the loaded models with per-model readiness (reflecting residency); the
# gateway calls it to autodiscover which workers currently serve each model.

import gRPCServer
const _G = gRPCServer

const _SERVER_NAME = "ReactantServer"
const _SERVER_VERSION = "0.1.0"
const _SERVER_EXTENSIONS = String["system_shared_memory"]

# gRPCServer defaults to a 4 MiB per-message cap; inference tensors routinely exceed that and
# the HTTP transport this replaces had no body limit. Size generously so large inline tensors
# are not rejected with RESOURCE_EXHAUSTED. Shared-memory-backed tensors stay small on the wire.
const _MAX_MESSAGE_BYTES = 512 * 1024 * 1024

# HTTP/2 receive flow-control windows the server advertises. The protocol default is only 64 KiB
# per stream and per connection, which throttles a large inline-tensor *upload* (the inference
# request) to ~window/RTT and gates it on WINDOW_UPDATE round-trips. The gRPC client (libcurl)
# already advertises a large receive window by default (nghttp2's "huge window"), so we match it
# on the server's receive path; the connection window also caps total in-flight DATA across
# streams, bounding buffering. Hardcoded for now; intended to become config/env-tunable later.
const _H2_INITIAL_WINDOW_BYTES = 32 * 1024 * 1024     # per-stream receive window
const _H2_CONNECTION_WINDOW_BYTES = 32 * 1024 * 1024  # connection-level receive window

# Per-request state threaded through gRPCServer's context payload. `metrics` is `nothing` only for
# control-plane-only contexts (e.g. tests that never serve inference); `serve` always supplies it.
# `error_detail` is how much an INTERNAL error tells the client (`grpc.error_detail`), and `name` is
# the worker's name as INTERNAL messages and their log lines give it.
struct InferContext
    sched::Scheduler
    registry::ModelRegistry
    shm::SharedMemoryRegistry
    platform::String
    metrics::Union{WorkerMetrics, Nothing}
    error_detail::ErrorDetail
    name::String
end
# Meta sub-calls are not routed from here: `infer` runs a meta's orchestration on this request task and
# its sub-calls re-enter the scheduler in-process (see scheduler.jl `_run_meta_request`), so the context
# needs no caller. The 4-arg form is used by control-plane-only contexts (tests) that never serve inference.
InferContext(sched, registry, shm, platform) = InferContext(sched, registry, shm, platform, nothing)
InferContext(sched, registry, shm, platform, metrics) =
    InferContext(sched, registry, shm, platform, metrics, ERROR_DETAIL_REDACTED, _worker_name())

# This worker's name in INTERNAL messages and logs: the configured worker name, else the
# supervisor's REACTANT_WORKER_NAME, else `fallback`.
function _worker_name(configured::AbstractString = ""; fallback::AbstractString = "worker")
    isempty(configured) || return String(configured)
    env = strip(get(ENV, "REACTANT_WORKER_NAME", ""))
    return isempty(env) ? String(fallback) : String(env)
end

# gRPC status code -> Prometheus label string, for worker_requests_total.
# Re-keyed from the old gRPCServer GRPC_* Int constants to the merged StatusCode enum
# (identical integer values).
const _GRPC_STATUS_NAME = Dict{Int, String}(
    Int(gRPCServer.StatusCode.OK) => "OK", Int(gRPCServer.StatusCode.CANCELLED) => "CANCELLED",
    Int(gRPCServer.StatusCode.UNKNOWN) => "UNKNOWN", Int(gRPCServer.StatusCode.INVALID_ARGUMENT) => "INVALID_ARGUMENT",
    Int(gRPCServer.StatusCode.DEADLINE_EXCEEDED) => "DEADLINE_EXCEEDED", Int(gRPCServer.StatusCode.NOT_FOUND) => "NOT_FOUND",
    Int(gRPCServer.StatusCode.ALREADY_EXISTS) => "ALREADY_EXISTS", Int(gRPCServer.StatusCode.PERMISSION_DENIED) => "PERMISSION_DENIED",
    Int(gRPCServer.StatusCode.RESOURCE_EXHAUSTED) => "RESOURCE_EXHAUSTED", Int(gRPCServer.StatusCode.FAILED_PRECONDITION) => "FAILED_PRECONDITION",
    Int(gRPCServer.StatusCode.ABORTED) => "ABORTED", Int(gRPCServer.StatusCode.OUT_OF_RANGE) => "OUT_OF_RANGE",
    Int(gRPCServer.StatusCode.UNIMPLEMENTED) => "UNIMPLEMENTED", Int(gRPCServer.StatusCode.INTERNAL) => "INTERNAL",
    Int(gRPCServer.StatusCode.UNAVAILABLE) => "UNAVAILABLE", Int(gRPCServer.StatusCode.DATA_LOSS) => "DATA_LOSS",
)
_status_label(e) = get(_GRPC_STATUS_NAME, Int(_grpc_status(e)), "UNKNOWN")

# The worker's status contract, the one place a failure gets its gRPC code. Clients and the gateway
# act on the code (back off, fail over to another replica, give up, re-register shared memory), so
# each kind of failure has exactly one, chosen by its type:
#
# - RESOURCE_EXHAUSTED: the model's queue is full (backpressure; retry after a backoff)
# - UNAVAILABLE: the worker is shutting down or draining, its dispatch loop died, or the model is
#   not resident under externally-managed residency (another replica may serve it; never
#   FAILED_PRECONDITION, which a client reads as a stale shared-memory registration)
# - NOT_FOUND: the model is unknown here, or was unloaded or replaced while the request waited
# - DEADLINE_EXCEEDED: the deadline passed before dispatch (EDF laxity drops included), or the
#   dispatch timeout did
# - CANCELLED: the client cancelled the call, or gave up the shared-memory slot the request writes
#   (`RequestCancelled`, CORE-3: dropped at dispatch, or its output write was refused)
# - INVALID_ARGUMENT: the request's own data is wrong (a hook's `InvalidInputError`, bad text)
# - FAILED_PRECONDITION: a shared-memory region the registry does not know (re-register and retry)
# - ALREADY_EXISTS: a shared-memory region name registered for another client's live object
# - INTERNAL: anything else, a failure on the server's side
#
# Decode, input validation and encode classify the codec's client errors themselves (see
# `_as_invalid`, `_validate_inputs`, `_as_encode_error`) and arrive here as GRPCErrors.
function _grpc_status(e)
    S = gRPCServer.StatusCode
    e isa gRPCServer.GRPCError && return e.code
    e isa QueueFullError && return S.RESOURCE_EXHAUSTED
    (e isa WorkerUnavailableError || e isa NotResidentError) && return S.UNAVAILABLE
    e isa ModelNotFoundError && return S.NOT_FOUND
    (e isa DeadlineExceeded || e isa DispatchTimeoutError) && return S.DEADLINE_EXCEEDED
    (e isa RequestCancelledError || e isa RequestCancelled) && return S.CANCELLED
    _client_data_error(e) && return S.INVALID_ARGUMENT
    e isa UnregisteredRegionError && return S.FAILED_PRECONDITION
    e isa ShmRegionExistsError && return S.ALREADY_EXISTS
    return S.INTERNAL
end

# Errors a bundle hook raises for client data it cannot accept.
_client_data_error(e) = e isa InvalidInputError || e isa BertText.TextInputError

# gRPC carries the message in a header, so keep it to a line's worth.
const _STATUS_MESSAGE_MAX = 1024
function _status_message(e)
    msg = sprint(showerror, e)
    return ncodeunits(msg) <= _STATUS_MESSAGE_MAX ? msg :
        msg[1:prevind(msg, _STATUS_MESSAGE_MAX + 1)] * " ..."
end

# `e` as the GRPCError the client receives: the contract's code, and a message that is always
# sanitized (`sanitize_status_message`). A server-side failure (INTERNAL) is logged here with its
# backtrace `bt` under a request id (the client's KServe `id`, else a generated one) and this worker's
# name; the client is told only that id and name (`ERROR_DETAIL_REDACTED`), or the exception text as
# well (`ERROR_DETAIL_FULL`). Every other code keeps its explanation in either mode.
function _as_grpc_error(
        e, bt, what::AbstractString; detail::ErrorDetail = ERROR_DETAIL_REDACTED,
        worker::AbstractString = _worker_name(), id::AbstractString = ""
    )
    e isa gRPCServer.GRPCError &&
        return gRPCServer.GRPCError(e.code, sanitize_status_message(e.message), e.details)
    code = _grpc_status(e)
    code == gRPCServer.StatusCode.INTERNAL ||
        return gRPCServer.GRPCError(code, sanitize_status_message(_status_message(e)))
    rid = error_request_id(id)
    @error "$what failed" request_id = rid worker = worker exception = (e, bt)
    msg = detail == ERROR_DETAIL_FULL ? "$(_status_message(e)) (request id $rid, worker $worker)" :
        "internal error (request id $rid, worker $worker); see the worker log"
    return gRPCServer.GRPCError(code, sanitize_status_message(msg))
end

# Run a handler body with every exception it throws mapped through the status contract, rather than
# left to gRPCServer, which reads a server-side KeyError as NOT_FOUND, an ArgumentError as
# INVALID_ARGUMENT and a BoundsError as OUT_OF_RANGE (and sends no message through the sanitizer).
function _with_status(f, what::AbstractString, ctx::InferContext)
    try
        return f()
    catch e
        throw(_as_grpc_error(e, catch_backtrace(), what; detail = ctx.error_detail, worker = ctx.name))
    end
end

# The cancellation probe for one call: whether its client has cancelled it, polled while the
# request waits (see `infer`). gRPCServer 0.1.2 never marks a call cancelled, so with that release
# this never fires; it takes effect once gRPCServer sets `ServerContext.cancelled`.
struct _CallCancelled
    ctx::gRPCServer.ServerContext
end
(c::_CallCancelled)() = gRPCServer.is_cancelled(c.ctx)

# A request's effective absolute deadline (local time_ns()): the TIGHTEST of the in-body KV timeout
# (carried through the gateway's raw-byte forwarding) and the grpc-timeout (which the gateway
# recomputes per hop, so it reflects time already burned reaching this worker). Both are absolute
# local times after decode; 0 means "not set" on that channel. Taking the min means a gateway that
# decremented grpc-timeout for transit wins over a stale, never-decremented KV budget.
function _effective_deadline(decoded_dl::Integer, grpc_dl::Integer)
    a, b = Int64(decoded_dl), Int64(grpc_dl)
    a == 0 && return b
    b == 0 && return a
    return min(a, b)
end

_not_found(msg) = throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, msg))
_invalid(msg) = throw(gRPCServer.GRPCError(gRPCServer.StatusCode.INVALID_ARGUMENT, msg))

# Run `f`, converting any thrown error into INVALID_ARGUMENT unless it is already a
# GRPCError (which carries its own status, e.g. NOT_FOUND). A reference to a
# shared-memory region the registry does not know (e.g. the client registered with a previous
# incarnation of this process, before a restart wiped the in-memory registry) is mapped to
# FAILED_PRECONDITION instead: it is recoverable by re-registering, so it is a distinct signal
# from a genuinely malformed request. The client re-registers and retries on this status. Running
# out of memory is the server's failure, not the request's, and is rethrown (INTERNAL). For a
# request decode and the shared-memory RPCs, whose every other failure comes from client data.
function _as_invalid(f)
    try
        return f()
    catch e
        e isa gRPCServer.GRPCError && rethrow()
        e isa OutOfMemoryError && rethrow()
        e isa UnregisteredRegionError &&
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.FAILED_PRECONDITION, sprint(showerror, e)))
        # A name registered for another client's live object (CORE-2), as Triton reports it.
        e isa ShmRegionExistsError &&
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.ALREADY_EXISTS, sprint(showerror, e)))
        _invalid(sprint(showerror, e))
    end
end

# Run the response encode. The codec rejects outputs that cannot be returned as the client asked
# (a requested output the model does not produce, an output larger than its shared-memory slot, an
# access outside the client's region) with an ErrorException or ArgumentError: INVALID_ARGUMENT. A
# region the registry no longer knows is a stale registration: FAILED_PRECONDITION. Anything else
# (out of memory copying an output, a bug) is the server's own failure and is rethrown, to become
# INTERNAL.
function _as_encode_error(f)
    try
        return f()
    catch e
        e isa gRPCServer.GRPCError && rethrow()
        e isa UnregisteredRegionError &&
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.FAILED_PRECONDITION, sprint(showerror, e)))
        # A shared-memory output write skipped because its client is gone (CORE-3): the slot was
        # given up (CANCELLED), or, for a client without slot control, the deadline had passed.
        e isa RequestCancelled && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.CANCELLED, sprint(showerror, e)))
        e isa DeadlineExceeded &&
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.DEADLINE_EXCEEDED, sprint(showerror, e)))
        (e isa ErrorException || e isa ArgumentError) && _invalid(sprint(showerror, e))
        rethrow()
    end
end

# A model is ready to serve when it is compiled and, in externally-managed mode, currently
# resident on the device (the worker does not autonomously load there). In self-managed mode an
# assigned model is always ready because the scheduler loads it on demand. The gateway discovers
# routes from this readiness, so a control-plane pin/unpin flips routing on the next probe.
function _model_ready(ctx::InferContext, name::AbstractString)
    sched = ctx.sched
    dispatch_alive(sched) || return false
    externally_managed = sched.weight_cache !== nothing && sched.weight_cache.mode == EXTERNALLY_MANAGED
    # A meta runs its sub-models in-process, so it is ready only where they can run. In
    # externally-managed mode that means every sub's weights must be resident on this worker (the
    # group is co-placed); in self-managed mode the meta loads them on demand, so it is always ready.
    meta = get_meta(ctx.registry, name)
    if meta !== nothing
        externally_managed || return true
        return all(meta.calls; init = true) do sub
            e = get_model(ctx.registry, sub)
            e !== nothing && e.executable !== nothing && e.executable.weights !== nothing
        end
    end
    entry = get_model(ctx.registry, name)
    (entry === nothing || entry.executable === nothing) && return false
    externally_managed && return entry.executable.weights !== nothing
    return true
end

_handle_model_ready(ctx::InferContext, req) =
    inference.ModelReadyResponse(; ready = _model_ready(ctx, req.name))

# The server is ready only once at least one model is registered and every registered model
# has been compiled. Compilation finishes before the gRPC plane accepts traffic, so this
# guards mainly against a misconfigured model_dirs that left the registry empty. A worker whose
# dispatch loop has died is never ready (nor live, see `_handle_server_live`).
function _handle_server_ready(ctx::InferContext)
    entries = values(ctx.registry.by_name)
    has_any = !isempty(entries) || !isempty(ctx.registry.meta)
    ready = has_any && all(e -> e.executable !== nothing, entries) && dispatch_alive(ctx.sched)
    return inference.ServerReadyResponse(; ready = ready)
end

_handle_server_live(ctx::InferContext) = inference.ServerLiveResponse(; live = dispatch_alive(ctx.sched))

_handle_server_metadata(::InferContext) =
    inference.ServerMetadataResponse(;
    name = _SERVER_NAME, version = _SERVER_VERSION,
    extensions = copy(_SERVER_EXTENSIONS)
)

function _handle_model_metadata(ctx::InferContext, req)
    entry = get_model(ctx.registry, req.name)
    if entry === nothing
        meta = get_meta(ctx.registry, req.name)
        meta === nothing && _not_found("unknown model: $(req.name)")
        return encode_model_metadata(inference, req.name, meta.manifest, ctx.platform)
    end
    return encode_model_metadata(inference, req.name, entry.manifest, ctx.platform)
end

# Validate decoded inputs against the model's client-facing spec so a malformed request is
# rejected as INVALID_ARGUMENT here, before it is queued. Failing later, inside the dispatch,
# would surface as INTERNAL and also fail the innocent requests coalesced into the same batch.
# Batch and variable axes accept any extent; fixed axes must match exactly. Shapes are Julia
# (column-major) order on both sides. Inputs whose batch extents disagree are accepted too, since
# preprocess may change an input's row count; the scheduler checks the preprocessed inputs and
# dispatches such a request alone (`_batch_extents_agree`), so it fails without touching others.
function _validate_inputs(entry, request::InferRequest)
    specs = client_input_spec(entry.manifest)
    byname = Dict(sp.name => sp for sp in specs)
    seen = Set{String}()
    for t in request.inputs
        sp = get(byname, t.name, nothing)
        sp === nothing && _invalid("input '$(t.name)' is not declared by model '$(request.model_name)'")
        push!(seen, t.name)
        t.dtype == sp.dtype ||
            _invalid("input '$(t.name)' has dtype $(dtype_token(t.dtype)), expected $(dtype_token(sp.dtype))")
        length(t.shape) == length(sp.shape) ||
            _invalid("input '$(t.name)' has $(length(t.shape)) dims, expected $(length(sp.shape))")
        for (i, d) in enumerate(sp.shape)
            d.kind == FIXED || continue
            t.shape[i] == d.size ||
                _invalid("input '$(t.name)' dim $i has extent $(t.shape[i]), expected $(d.size)")
        end
    end
    for sp in specs
        sp.name in seen || _invalid("required input '$(sp.name)' is missing")
    end
    return nothing
end

# Run a request through the scheduler: the handlers' one way into `infer`, with its keywords passed
# through (the handlers' `cancelled`, and the meta handler's `scratch_out`). A failure keeps its type;
# `_handle_infer` maps it to the contract's code (`_grpc_status`), which replaced the message matching
# this helper once did.
_infer_or_not_found(ctx::InferContext, request; kw...) = infer(ctx.sched, request; kw...)

# Time and count every ModelInfer for the worker's Prometheus export (worker_requests_total by
# model+status, worker_request_latency_seconds), then delegate to the handler body. `finish` is
# applied to the response inside the timed region (the raw path encodes there), and `t0` lets a
# caller start the clock before its own work (the raw path starts it before decoding). Every
# failure leaves as a GRPCError carrying the status contract's code (`_grpc_status`), and the
# status label counted is that same code. `cancelled` is the call's cancellation probe (see `infer`).
function _handle_infer(
        ctx::InferContext, req, grpc_deadline_ns::Integer = 0;
        finish = identity, t0::Float64 = time(), cancelled = nothing
    )
    name = req.model_name
    try
        resp = finish(_handle_infer_impl(ctx, req, grpc_deadline_ns; cancelled = cancelled))
        if ctx.metrics !== nothing
            observe_request!(ctx.metrics, name, time() - t0)
            inc_request!(ctx.metrics, name, "OK")
        end
        return resp
    catch e
        err = _as_grpc_error(
            e, catch_backtrace(), "ModelInfer for model '$name'";
            detail = ctx.error_detail, worker = ctx.name, id = req.id
        )
        if ctx.metrics !== nothing
            observe_request!(ctx.metrics, name, time() - t0)
            inc_request!(ctx.metrics, name, _status_label(err))
        end
        throw(err)
    finally
        # Fired on success and failure alike: a failing request is still a client using this server.
        _beat(:request, String(name))
    end
end

# The ModelInfer RPC as registered: raw request bytes in, encoded response bytes out, so that the
# protobuf decode and encode run with the rest of the request on a default-pool task (see
# `_offloaded`). Decoding a large request or encoding a large response is a copy of the tensor data,
# and the first decode of the process compiles; neither belongs on the interactive thread.
function _handle_infer_raw(
        ctx::InferContext, body::Vector{UInt8}, grpc_deadline_ns::Integer = 0; cancelled = nothing
    )
    t0 = time()
    req = try
        ProtoBuf.decode(ProtoBuf.ProtoDecoder(IOBuffer(body)), inference.ModelInferRequest)
    catch err
        err isa OutOfMemoryError && rethrow()
        _invalid("malformed ModelInferRequest: " * sprint(showerror, err))
    end
    return _handle_infer(ctx, req, grpc_deadline_ns; finish = _encode_message, t0 = t0, cancelled = cancelled)
end

function _encode_message(msg)
    io = IOBuffer()
    ProtoBuf.encode(ProtoBuf.ProtoEncoder(io), msg)
    return take!(io)
end

function _handle_infer_impl(ctx::InferContext, req, grpc_deadline_ns::Integer = 0; cancelled = nothing)
    name = req.model_name
    isempty(name) && _invalid("ModelInferRequest.model_name is empty")
    meta = get_meta(ctx.registry, name)
    meta === nothing || return _handle_meta_infer(ctx, meta, req, grpc_deadline_ns; cancelled = cancelled)
    entry = get_model(ctx.registry, name)
    entry === nothing && _not_found("unknown model: $name")
    decoded = _as_invalid(() -> decode_infer_request(inference, req, ctx.shm))
    deadline_ns = _effective_deadline(decoded.request.deadline_ns, grpc_deadline_ns)
    request = InferRequest(
        name, decoded.request.requested_outputs, decoded.request.inputs, deadline_ns,
        decoded.request.slot_control
    )
    _validate_inputs(entry, request)
    outputs = _infer_or_not_found(ctx, request; cancelled = cancelled)
    # Encoding rejects a requested output the model does not produce: a client mistake, so
    # surface it as INVALID_ARGUMENT (matching the codec's documented contract).
    return _as_encode_error(() -> encode_infer_response(inference, name, decoded, outputs, ctx.shm; deadline_ns = deadline_ns))
end

# A meta's orchestration runs on this request task (`infer` -> `_run_meta_request`), under the worker's
# one-meta-at-a-time gate; its in-process sub-calls dispatch on the loop. Input validation reuses the
# regular path (it reads only `manifest`, which MetaEntry also carries). A deadline bail and an
# unknown or unloaded sub-model keep their types, and `_handle_infer` maps them as on the regular path.
# The outputs may alias the pool slots the meta's `call.scratch` acquired, so `infer` hands those slots
# here (`scratch_out`) rather than releasing them when the orchestration returns, and they go back to
# the pool only after the response is encoded, on every path. `encode_infer_response` copies each
# output's bytes (inline, or into the client's shared-memory region), so its message holds no alias.
function _handle_meta_infer(ctx::InferContext, meta, req, grpc_deadline_ns::Integer = 0; cancelled = nothing)
    decoded = _as_invalid(() -> decode_infer_request(inference, req, ctx.shm))
    deadline_ns = _effective_deadline(decoded.request.deadline_ns, grpc_deadline_ns)
    request = InferRequest(
        meta.name, decoded.request.requested_outputs, decoded.request.inputs, deadline_ns,
        decoded.request.slot_control
    )
    _validate_inputs(meta, request)
    scratch = PoolSlot[]
    try
        outputs = _infer_or_not_found(ctx, request; cancelled = cancelled, scratch_out = scratch)
        return _as_encode_error(
            () -> encode_infer_response(inference, meta.name, decoded, outputs, ctx.shm; deadline_ns = deadline_ns)
        )
    finally
        release_scratch!(scratch)
    end
end

# List models with per-model readiness reflecting residency. `req.ready` filters to ready models
# (the gateway's discovery call), matching Triton's RepositoryIndex semantics. Internal sub-models
# (called only in-process by a meta) are hidden: the gateway must not discover, route, or place them.
function _handle_repository_index(ctx::InferContext, req)
    entries = [name => _model_ready(ctx, name) for name in routable_model_names(ctx.registry)]
    req.ready && filter!(last, entries)
    return encode_repository_index(inference, entries)
end

_handle_shm_status(shm::SharedMemoryRegistry, name) =
    _as_invalid(() -> encode_shm_status(inference, shm, name))

function _handle_shm_register(shm::SharedMemoryRegistry, req)
    return _as_invalid() do
        shm_register!(shm, req.name, req.key, req.offset, req.byte_size)
        encode_shm_register_response(inference)
    end
end

function _handle_shm_unregister(shm::SharedMemoryRegistry, name)
    return _as_invalid() do
        shm_unregister!(shm, name)
        encode_shm_unregister_response(inference)
    end
end

# Non-Triton extension: answer whether we can see the client's shared-memory object, i.e.
# whether system shared-memory transport can work with this client. No registry interaction.
_handle_is_same_ipc_namespace(name) =
    _as_invalid(() -> encode_is_same_ipc_namespace_response(inference, same_ipc_namespace(name)))

# Old gRPCServer handed handlers an absolute deadline on Julia's machine-relative time_ns()
# clock (time_ns() + grpc-timeout). The merged package parses grpc-timeout into a wall-clock
# DateTime; the shared ReactantServerCore.deadline_to_time_ns converts ctx.deadline back onto
# the local clock the scheduler compares against (time_ns() >= dl in scheduler.jl; the codec's
# _decode_deadline_ns uses time_ns() + budget).

"""
    build_grpc_server(sched, registry, platform, shm, host, port, ctx;
                      max_recv_msg_bytes=_MAX_MESSAGE_BYTES, max_send_msg_bytes=_MAX_MESSAGE_BYTES,
                      max_concurrent_requests=0,
                      h2_initial_window_size=_H2_INITIAL_WINDOW_BYTES,
                      h2_connection_window_size=_H2_CONNECTION_WINDOW_BYTES) -> GRPCServer

Register the KServe V2 GRPCInferenceService and ControlService handlers on a fresh
[`gRPCServer.GRPCServer`](@ref). The server threads `ctx` (an [`InferContext`](@ref)) into
every request's `ServerContext.payload`; `serve` supplies the metrics-bearing context after
constructing the server. The message-size caps default to `_MAX_MESSAGE_BYTES`; `serve` passes
the configured `grpc.max_recv_msg_bytes` / `max_send_msg_bytes` (passed through as the merged
package's per-direction caps).
"""
# HTTP.jl starts every request handler on the :interactive thread pool, which has one thread here,
# shared with the scheduler's dispatch loop. A handler that does real work there (decoding, a lock
# wait, file or shared-memory I/O, or code compiling on first use) stalls every other request and the
# dispatch loop until it finishes. `_offloaded` wraps a handler so its body runs on a default-pool task
# while the interactive task only waits; a GRPCError propagates unchanged, and any other error leaves
# with the status contract's code (`_with_status`).
_offloaded(handler) = (ctx, req) -> off_interactive(() -> _with_status(() -> handler(ctx, req), ctx.method, ctx.payload))

# The ModelInfer RPC: raw bytes in and out (see `_handle_infer_raw`), offloaded like the rest. The
# deadline is read here, on entry, as gRPCServer computes it, and the call's cancellation probe is
# handed down to the wait for its result. A failure before the request is decoded (a malformed body)
# leaves through the status contract too.
function _model_infer_rpc(ctx, body)
    deadline_ns = deadline_to_time_ns(ctx.deadline, gRPCServer.remaining_time(ctx))
    cancelled = _CallCancelled(ctx)
    return off_interactive() do
        _with_status("ModelInfer", ctx.payload) do
            _handle_infer_raw(ctx.payload, body, deadline_ns; cancelled = cancelled)
        end
    end
end

function build_grpc_server(
        sched::Scheduler, registry::ModelRegistry, platform::AbstractString,
        shm::SharedMemoryRegistry, host::AbstractString, port::Integer, ctx::InferContext;
        max_recv_msg_bytes::Integer = _MAX_MESSAGE_BYTES,
        max_send_msg_bytes::Integer = _MAX_MESSAGE_BYTES,
        max_concurrent_requests::Integer = 0,
        h2_initial_window_size::Integer = _H2_INITIAL_WINDOW_BYTES,
        h2_connection_window_size::Integer = _H2_CONNECTION_WINDOW_BYTES
    )
    server = gRPCServer.GRPCServerHTTPJl(
        host, Int(port);
        context = ctx,
        max_receive_message_length = max_recv_msg_bytes,
        max_send_message_length = max_send_msg_bytes,
        max_concurrent_requests = (max_concurrent_requests > 0 ? max_concurrent_requests : nothing),
        h2_initial_window_size = h2_initial_window_size,
        h2_connection_window_size = h2_connection_window_size
    )
    # ServerLive and ServerReady answer from the interactive thread itself: they are trivial, and
    # they are how a prober tells whether that thread is responsive. Everything else is offloaded.
    register_GRPCInferenceService!(
        server;
        ServerLive = (ctx, req) -> _handle_server_live(ctx.payload),
        ServerReady = (ctx, req) -> _handle_server_ready(ctx.payload),
        ModelReady = _offloaded((ctx, req) -> _handle_model_ready(ctx.payload, req)),
        ServerMetadata = _offloaded((ctx, req) -> _handle_server_metadata(ctx.payload)),
        ModelMetadata = _offloaded((ctx, req) -> _handle_model_metadata(ctx.payload, req)),
        RepositoryIndex = _offloaded((ctx, req) -> _handle_repository_index(ctx.payload, req)),
        SystemSharedMemoryStatus = _offloaded((ctx, req) -> _handle_shm_status(ctx.payload.shm, req.name)),
        SystemSharedMemoryRegister = _offloaded((ctx, req) -> _handle_shm_register(ctx.payload.shm, req)),
        SystemSharedMemoryUnregister = _offloaded((ctx, req) -> _handle_shm_unregister(ctx.payload.shm, req.name)),
        IsSameIPCNamespace = _offloaded((ctx, req) -> _handle_is_same_ipc_namespace(req.name))
    )
    # ModelInfer takes and returns raw bytes so its protobuf decode and encode are offloaded too.
    inference.register_GRPCInferenceService_ModelInfer!(
        server, _model_infer_rpc; raw_request = true, raw_response = true
    )
    register_control_service!(server)
    return server
end
