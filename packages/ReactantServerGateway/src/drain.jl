# The gateway's side of a graceful drain (see ReactantServerCore's drain.jl for the stop protocol):
# a gRPC interceptor that counts in-flight ModelInfer calls and, once the gateway is draining,
# refuses new ones with UNAVAILABLE. Readiness (`/readyz`) is turned down by `stop!` itself.

const _DRAINING_MESSAGE = "gateway is draining for shutdown; retry shortly"

struct _DrainInterceptor <: gRPCServer.Interceptor
    gate::ReactantServerCore.DrainGate
end

# Every RPC passes through here, so `request` and `next` are left unspecialized: one compiled
# method serves them all, and the per-call cost is one dynamic call.
function (d::_DrainInterceptor)(ctx, @nospecialize(request), info::gRPCServer.MethodInfo, @nospecialize(next))
    info.method_name == "ModelInfer" || return next(ctx, request)
    ReactantServerCore.enter!(d.gate) ||
        throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, _DRAINING_MESSAGE))
    try
        return next(ctx, request)
    finally
        ReactantServerCore.leave!(d.gate)
    end
end
