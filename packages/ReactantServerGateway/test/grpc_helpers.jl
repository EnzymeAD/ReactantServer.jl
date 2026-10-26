# Shared helpers for the gRPC end-to-end tests: ephemeral-port startup and a unary-call wrapper
# that builds a gRPCClient.jl service client for a GRPCInferenceService RPC and sends one
# request. The runtime proto stubs are server-only, so the client is constructed generically
# from the message types and the method path rather than from generated client stubs.

import gRPCClient
import gRPCServer
using Sockets

const _GRPC_SERVICE = "/inference.GRPCInferenceService"

# A port nothing is listening on, for a deliberately unreachable endpoint. Do not listen on it:
# another process can take it between this returning and the bind. Servers listen on port 0 instead.
grpc_free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

# Start a gRPC server built with port 0 and return the port the OS gave it.
grpc_start_ephemeral!(server) = (gRPCServer.start!(server); something(gRPCServer.bound_port(server)))

# Send `request` to `rpc` on the local server at `port`, returning the decoded response. `service`
# defaults to the inference data plane; pass it to reach another service on the same router (the
# gateway also terminates the two control services).
function grpc_call(
        ::Type{Req}, ::Type{Resp}, rpc::AbstractString, port::Integer, request;
        service::AbstractString = _GRPC_SERVICE, deadline = 10
    ) where {Req, Resp}
    client = gRPCClient.gRPCServiceClient{Req, false, Resp, false}(
        "127.0.0.1", port, "$service/$rpc";
        deadline = deadline
    )
    return gRPCClient.grpc_sync_request(client, request)
end
