# Meta-model execution: a meta model is a user-authored Julia workflow (a bundle's model.jl that
# calls register_meta_model) that chains several models with data-dependent logic in between. Its
# orchestration runs on the gRPC request task (off the GPU dispatch loop, exactly like the
# pre/post hooks), and issues ordinary inference calls through an injected ModelCaller. The caller
# abstracts the destination: in single-worker mode it calls the local scheduler in-process; in
# multi-worker mode it calls back into the gateway over gRPC, so a backbone on another worker is
# reached transparently. The scheduler, dispatch loop, and weight cache never see meta models.

# Environment variable naming the loopback gRPC endpoint (the gateway in multi-worker mode). When
# unset, the worker has no gateway and meta sub-calls go in-process through the local scheduler.
const LOOPBACK_ENV = "REACTANT_LOOPBACK_GRPC"

abstract type ModelCaller end

# In-process caller: routes a sub-call straight through the local scheduler's `infer`. No gRPC, no
# serialization. Used when no loopback gateway is configured (single-worker deployments).
struct LocalCaller <: ModelCaller
    sched::Scheduler
end

call_model(c::LocalCaller, name::AbstractString, inputs::Vector{NamedTensor};
           requested_outputs::Vector{String}=String[]) =
    infer(c.sched, InferRequest(String(name), requested_outputs, inputs))

# Loopback caller: routes a sub-call to the gateway over gRPC. The gateway's existing routing places
# the backbone on whichever worker hosts it, so device placement stays abstracted from the meta
# author. One client over one libcurl multi handle, built once at startup and reused concurrently.
struct GatewayCaller{C} <: ModelCaller
    url::String
    client::C
end

# Split "host:port" (optionally with a grpc:// / grpcs:// scheme) into (host, port).
function _split_loopback(url::AbstractString)
    s = String(url)
    for scheme in ("grpc://", "grpcs://", "http://", "https://")
        startswith(s, scheme) && (s = s[(length(scheme) + 1):end])
    end
    i = findlast(==(':'), s)
    i === nothing && throw(ArgumentError("loopback endpoint '$url' is not host:port"))
    host = s[1:(i - 1)]
    port = tryparse(Int, s[(i + 1):end])
    port === nothing && throw(ArgumentError("loopback endpoint '$url' has a non-numeric port"))
    return host, port
end

function GatewayCaller(url::AbstractString; deadline::Real=300,
                       max_msg_bytes::Integer=_MAX_MESSAGE_BYTES)
    host, port = _split_loopback(url)
    # sticky=false: meta orchestrations run on the worker's default (compute) thread pool, so the
    # multi handle's driving tasks must be schedulable on whichever thread issues the call.
    grpc = gRPCClient.gRPCCURL(; sticky=false)
    client = GRPCInferenceService_ModelInfer_Client(host, port; grpc=grpc, deadline=deadline,
        TRequest=ModelInferRequest, TResponse=ModelInferResponse,
        max_send_message_length=max_msg_bytes, max_recieve_message_length=max_msg_bytes)
    return GatewayCaller{typeof(client)}(String(url), client)
end

function call_model(c::GatewayCaller, name::AbstractString, inputs::Vector{NamedTensor};
                    requested_outputs::Vector{String}=String[])
    req = encode_infer_request(name, inputs; requested_outputs=requested_outputs)
    resp = gRPCClient.grpc_sync_request(c.client, req)
    return decode_infer_response(resp)
end

"""
    build_caller(sched) -> ModelCaller

Construct the worker's process-wide meta-model caller from the environment: a [`GatewayCaller`](@ref)
when `REACTANT_LOOPBACK_GRPC` names a gateway (multi-worker mode), otherwise a [`LocalCaller`](@ref)
over the local scheduler (single-worker mode).
"""
function build_caller(sched::Scheduler)
    url = strip(get(ENV, LOOPBACK_ENV, ""))
    if isempty(url)
        return LocalCaller(sched)
    end
    @info "Meta models will route sub-calls through the loopback gateway" endpoint = url
    return GatewayCaller(String(url))
end

"""
    run_meta(entry, caller, inputs) -> Vector{NamedTensor}

Run a meta model's orchestration. The injected `call(name, inputs)` closure dispatches a sub-call
through `caller`; it rejects any callee the meta model did not declare in its manifest `meta.calls`,
so the declared dependency set is authoritative at runtime.
"""
function run_meta(entry::MetaEntry, caller::ModelCaller, inputs::Vector{NamedTensor})
    declared = Set(entry.calls)
    call = function (name::AbstractString, ins::Vector{NamedTensor})
        String(name) in declared ||
            error("meta model '$(entry.name)' called undeclared model '$name'; add it to meta.calls")
        return call_model(caller, name, ins)
    end
    # The orchestration is defined in a sandboxed model.jl (a newer world age), so cross it with
    # invokelatest, exactly as infer() does for pre/post hooks.
    out = Base.invokelatest(entry.run, inputs, call)
    out isa Vector{NamedTensor} ||
        error("meta model '$(entry.name)' returned $(typeof(out)); expected Vector{NamedTensor}")
    return out
end
