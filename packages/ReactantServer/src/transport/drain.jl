# The worker's side of a graceful drain (see ReactantServerCore's drain.jl for the stop protocol):
# a gRPC interceptor that counts in-flight ModelInfer calls and, once the server is draining,
# refuses new ones with UNAVAILABLE and reports the worker not ready, so a gateway stops routing
# here and fails the refused calls over to another worker.

const _DRAINING_MESSAGE = "worker is draining for shutdown; retry on another worker"

struct _DrainInterceptor <: gRPCServer.Interceptor
    gate::ReactantServerCore.DrainGate
end

# Every RPC passes through here, so `request` and `next` are left unspecialized: one compiled
# method serves them all, and the per-call cost is one dynamic call.
function (d::_DrainInterceptor)(ctx, @nospecialize(request), info::gRPCServer.MethodInfo, @nospecialize(next))
    method = info.method_name
    if method == "ModelInfer"
        ReactantServerCore.enter!(d.gate) ||
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, _DRAINING_MESSAGE))
        try
            return next(ctx, request)
        finally
            ReactantServerCore.leave!(d.gate)
        end
    end
    if ReactantServerCore.is_draining(d.gate) && info.service_name == "inference.GRPCInferenceService"
        method == "ServerReady" && return inference.ServerReadyResponse(; ready = false)
        method == "ModelReady" && return inference.ModelReadyResponse(; ready = false)
        # The gateway routes by the models RepositoryIndex reports ready: report none.
        method == "RepositoryIndex" && return encode_repository_index(
            inference,
            request.ready ? Pair{String, Bool}[] :
                [n => false for n in routable_model_names(ctx.payload.registry)]
        )
    end
    return next(ctx, request)
end
