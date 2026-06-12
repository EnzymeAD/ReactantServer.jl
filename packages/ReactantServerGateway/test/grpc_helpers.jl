# Shared helpers for the gRPC end-to-end tests: a free-port picker and a unary-call wrapper
# that builds a gRPCClient.jl service client for a GRPCInferenceService RPC and sends one
# request. The runtime proto stubs are server-only, so the client is constructed generically
# from the message types and the method path rather than from generated client stubs.

import gRPCClient
using Sockets

const _GRPC_SERVICE = "/inference.GRPCInferenceService"

grpc_free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

# Send `request` to `rpc` on the local server at `port`, returning the decoded response.
function grpc_call(::Type{Req}, ::Type{Resp}, rpc::AbstractString, port::Integer, request) where {Req,Resp}
    client = gRPCClient.gRPCServiceClient{Req,false,Resp,false}("127.0.0.1", port, "$_GRPC_SERVICE/$rpc")
    return gRPCClient.grpc_sync_request(client, request)
end
