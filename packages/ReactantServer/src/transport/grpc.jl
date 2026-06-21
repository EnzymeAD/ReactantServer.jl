# The KServe V2 gRPC control plane. This is the only file that imports gRPCServer.
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

# gRPCRouter defaults to a 4 MiB per-message cap; inference tensors routinely exceed that and
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
struct InferContext
    sched::Scheduler
    registry::ModelRegistry
    shm::SharedMemoryRegistry
    platform::String
    metrics::Union{WorkerMetrics,Nothing}
    caller::ModelCaller        # routes meta-model sub-calls (local scheduler or loopback gateway)
end
# `caller` defaults to a LocalCaller over this worker's scheduler; `serve` overrides it with a
# GatewayCaller when a loopback gateway is configured. The 4-arg form is used by control-plane-only
# contexts (tests) that never serve inference.
InferContext(sched, registry, shm, platform, metrics) =
    InferContext(sched, registry, shm, platform, metrics, LocalCaller(sched))
InferContext(sched, registry, shm, platform) = InferContext(sched, registry, shm, platform, nothing)

# gRPC status code -> Prometheus label string, for worker_requests_total.
const _GRPC_STATUS_NAME = Dict{Int,String}(
    _G.GRPC_OK => "OK", _G.GRPC_CANCELLED => "CANCELLED", _G.GRPC_UNKNOWN => "UNKNOWN",
    _G.GRPC_INVALID_ARGUMENT => "INVALID_ARGUMENT", _G.GRPC_DEADLINE_EXCEEDED => "DEADLINE_EXCEEDED",
    _G.GRPC_NOT_FOUND => "NOT_FOUND", _G.GRPC_ALREADY_EXISTS => "ALREADY_EXISTS",
    _G.GRPC_PERMISSION_DENIED => "PERMISSION_DENIED", _G.GRPC_RESOURCE_EXHAUSTED => "RESOURCE_EXHAUSTED",
    _G.GRPC_FAILED_PRECONDITION => "FAILED_PRECONDITION", _G.GRPC_ABORTED => "ABORTED",
    _G.GRPC_OUT_OF_RANGE => "OUT_OF_RANGE", _G.GRPC_UNIMPLEMENTED => "UNIMPLEMENTED",
    _G.GRPC_INTERNAL => "INTERNAL", _G.GRPC_UNAVAILABLE => "UNAVAILABLE", _G.GRPC_DATA_LOSS => "DATA_LOSS",
)
_status_label(e) = e isa _G.gRPCServiceCallException ? get(_GRPC_STATUS_NAME, e.grpc_status, "UNKNOWN") :
                   e isa DeadlineExceeded ? "DEADLINE_EXCEEDED" : "INTERNAL"

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

_not_found(msg) = throw(_G.gRPCServiceCallException(_G.GRPC_NOT_FOUND, msg))
_invalid(msg) = throw(_G.gRPCServiceCallException(_G.GRPC_INVALID_ARGUMENT, msg))

# Run `f`, converting any thrown error into INVALID_ARGUMENT unless it is already a
# gRPCServiceCallException (which carries its own status, e.g. NOT_FOUND).
function _as_invalid(f)
    try
        return f()
    catch e
        e isa _G.gRPCServiceCallException && rethrow()
        _invalid(sprint(showerror, e))
    end
end

# A model is ready to serve when it is compiled and, in externally-managed mode, currently
# resident on the device (the worker does not autonomously load there). In self-managed mode an
# assigned model is always ready because the scheduler loads it on demand. The gateway discovers
# routes from this readiness, so a control-plane pin/unpin flips routing on the next probe.
function _model_ready(ctx::InferContext, name::AbstractString)
    # A meta model has no executable or device residency; it is ready as soon as it is loaded.
    is_meta_name(ctx.registry, name) && return true
    entry = get_model(ctx.registry, name)
    (entry === nothing || entry.executable === nothing) && return false
    sched = ctx.sched
    if sched.weight_cache !== nothing && sched.weight_cache.mode == EXTERNALLY_MANAGED
        return entry.executable.weights !== nothing
    end
    return true
end

_handle_model_ready(ctx::InferContext, req) =
    inference.ModelReadyResponse(; ready = _model_ready(ctx, req.name))

# The server is ready only once at least one model is registered and every registered model
# has been compiled. Compilation finishes before the gRPC plane accepts traffic, so this
# guards mainly against a misconfigured model_dirs that left the registry empty.
function _handle_server_ready(ctx::InferContext)
    entries = values(ctx.registry.by_name)
    has_any = !isempty(entries) || !isempty(ctx.registry.meta)
    ready = has_any && all(e -> e.executable !== nothing, entries)
    return inference.ServerReadyResponse(; ready = ready)
end

_handle_server_metadata(::InferContext) =
    inference.ServerMetadataResponse(; name=_SERVER_NAME, version=_SERVER_VERSION,
                                     extensions=copy(_SERVER_EXTENSIONS))

function _handle_model_metadata(ctx::InferContext, req)
    entry = get_model(ctx.registry, req.name)
    if entry === nothing
        meta = get_meta(ctx.registry, req.name)
        meta === nothing && _not_found("unknown model: $(req.name)")
        return encode_model_metadata(req.name, meta.manifest, ctx.platform)
    end
    return encode_model_metadata(req.name, entry.manifest, ctx.platform)
end

# Validate decoded inputs against the model's client-facing spec so a malformed request is
# rejected as INVALID_ARGUMENT here, before it is queued. Failing later, inside the dispatch,
# would surface as INTERNAL and also fail the innocent requests coalesced into the same batch.
# Batch and variable axes accept any extent; fixed axes must match exactly. Shapes are Julia
# (column-major) order on both sides.
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

# Map the scheduler's model-availability rejections to NOT_FOUND. The dynamic watcher can unload or
# reload a model in the narrow window between the readiness lookup above and dispatch, in which case
# `submit!` reports "unknown model" or a queued request is rejected with "was unloaded". Surfacing
# those as NOT_FOUND (rather than the default INTERNAL) keeps them consistent with the pre-dispatch
# lookup, so the gateway treats them as retryable and refreshes its routes. Other runtime errors
# keep their default mapping. Mirrors the message match in control_grpc.jl's `_as_control`.
function _infer_or_not_found(ctx::InferContext, request)
    try
        return infer(ctx.sched, request)
    catch e
        e isa _G.gRPCServiceCallException && rethrow()
        # A request the scheduler dropped at admission for an expired deadline -> DEADLINE_EXCEEDED.
        e isa DeadlineExceeded && throw(_G.gRPCServiceCallException(_G.GRPC_DEADLINE_EXCEEDED, sprint(showerror, e)))
        msg = sprint(showerror, e)
        (occursin("unknown model", msg) || occursin("was unloaded", msg)) && _not_found(msg)
        rethrow()
    end
end

# Time and count every ModelInfer for the worker's Prometheus export (worker_requests_total by
# model+status, worker_request_latency_seconds), then delegate to the handler body.
function _handle_infer(ctx::InferContext, req, grpc_deadline_ns::Integer=0)
    ctx.metrics === nothing && return _handle_infer_impl(ctx, req, grpc_deadline_ns)
    t0 = time()
    name = req.model_name
    try
        resp = _handle_infer_impl(ctx, req, grpc_deadline_ns)
        observe_request!(ctx.metrics, name, time() - t0)
        inc_request!(ctx.metrics, name, "OK")
        return resp
    catch e
        observe_request!(ctx.metrics, name, time() - t0)
        inc_request!(ctx.metrics, name, _status_label(e))
        rethrow()
    end
end

function _handle_infer_impl(ctx::InferContext, req, grpc_deadline_ns::Integer=0)
    name = req.model_name
    isempty(name) && _invalid("ModelInferRequest.model_name is empty")
    meta = get_meta(ctx.registry, name)
    meta === nothing || return _handle_meta_infer(ctx, meta, req, grpc_deadline_ns)
    entry = get_model(ctx.registry, name)
    entry === nothing && _not_found("unknown model: $name")
    decoded = _as_invalid(() -> decode_infer_request(req, ctx.shm))
    deadline_ns = _effective_deadline(decoded.request.deadline_ns, grpc_deadline_ns)
    request = InferRequest(name, decoded.request.requested_outputs, decoded.request.inputs, deadline_ns)
    _validate_inputs(entry, request)
    outputs = _infer_or_not_found(ctx, request)   # availability rejections -> NOT_FOUND; else INTERNAL
    # Encoding rejects a requested output the model does not produce: a client mistake, so
    # surface it as INVALID_ARGUMENT (matching the codec's documented contract).
    return _as_invalid(() -> encode_infer_response(name, decoded, outputs, ctx.shm))
end

# Map a meta model's sub-call availability failures to NOT_FOUND so they are retryable upstream,
# mirroring `_infer_or_not_found`. This covers the LocalCaller path ("unknown model"/"was
# unloaded" thrown by the scheduler); the GatewayCaller path surfaces a gRPCClient exception that
# the worker's gRPC server already renders with its own status.
function _meta_or_not_found(ctx::InferContext, meta, inputs::Vector{NamedTensor}, deadline_ns::Integer)
    try
        return run_meta(meta, ctx.caller, inputs; deadline_ns=deadline_ns)
    catch e
        e isa _G.gRPCServiceCallException && rethrow()
        # The orchestration bailed (or a sub-call was dropped) because the deadline passed.
        e isa DeadlineExceeded && throw(_G.gRPCServiceCallException(_G.GRPC_DEADLINE_EXCEEDED, sprint(showerror, e)))
        msg = sprint(showerror, e)
        (occursin("unknown model", msg) || occursin("was unloaded", msg)) && _not_found(msg)
        rethrow()
    end
end

# A meta model runs its Julia orchestration on this request task (off the GPU dispatch loop). Input
# validation reuses the regular path (it reads only `manifest`, which MetaEntry also carries). The
# orchestration's own sub-calls go through `ctx.caller`.
function _handle_meta_infer(ctx::InferContext, meta, req, grpc_deadline_ns::Integer=0)
    decoded = _as_invalid(() -> decode_infer_request(req, ctx.shm))
    deadline_ns = _effective_deadline(decoded.request.deadline_ns, grpc_deadline_ns)
    request = InferRequest(meta.name, decoded.request.requested_outputs, decoded.request.inputs, deadline_ns)
    _validate_inputs(meta, request)
    outputs = _meta_or_not_found(ctx, meta, request.inputs, deadline_ns)
    return _as_invalid(() -> encode_infer_response(meta.name, decoded, outputs, ctx.shm))
end

# List models with per-model readiness reflecting residency. `req.ready` filters to ready models
# (the gateway's discovery call), matching Triton's RepositoryIndex semantics.
function _handle_repository_index(ctx::InferContext, req)
    entries = [name => _model_ready(ctx, name) for name in model_names(ctx.registry)]
    req.ready && filter!(last, entries)
    return encode_repository_index(entries)
end

_handle_shm_status(shm::SharedMemoryRegistry, name) =
    _as_invalid(() -> encode_shm_status(shm, name))

function _handle_shm_register(shm::SharedMemoryRegistry, req)
    return _as_invalid() do
        shm_register!(shm, req.name, req.key, req.offset, req.byte_size)
        encode_shm_register_response()
    end
end

function _handle_shm_unregister(shm::SharedMemoryRegistry, name)
    return _as_invalid() do
        shm_unregister!(shm, name)
        encode_shm_unregister_response()
    end
end

"""
    build_grpc_router(sched, registry, platform, shm) -> gRPCRouter

Register the KServe V2 GRPCInferenceService handlers. The returned router is served by
gRPCServer with an `InferContext` payload (see `serve`).
"""
function build_grpc_router(sched::Scheduler, registry::ModelRegistry, platform::AbstractString,
                           shm::SharedMemoryRegistry)
    router = _G.gRPCRouter(; max_receive_message_length=_MAX_MESSAGE_BYTES,
                           max_send_message_length=_MAX_MESSAGE_BYTES)
    register_GRPCInferenceService!(router;
        ServerLive    = (req, ctx) -> inference.ServerLiveResponse(; live=true),
        ServerReady   = (req, ctx) -> _handle_server_ready(ctx.payload),
        ModelReady    = (req, ctx) -> _handle_model_ready(ctx.payload, req),
        ServerMetadata = (req, ctx) -> _handle_server_metadata(ctx.payload),
        ModelMetadata = (req, ctx) -> _handle_model_metadata(ctx.payload, req),
        ModelInfer    = (req, ctx) -> _handle_infer(ctx.payload, req, ctx.deadline_ns),
        RepositoryIndex = (req, ctx) -> _handle_repository_index(ctx.payload, req),
        SystemSharedMemoryStatus     = (req, ctx) -> _handle_shm_status(ctx.payload.shm, req.name),
        SystemSharedMemoryRegister   = (req, ctx) -> _handle_shm_register(ctx.payload.shm, req),
        SystemSharedMemoryUnregister = (req, ctx) -> _handle_shm_unregister(ctx.payload.shm, req.name),
    )
    # The worker control plane (residency + live policy) rides on the same router and payload.
    register_control_service!(router)
    return router
end
