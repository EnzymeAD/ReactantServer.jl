# Shared helpers for the gRPC end-to-end tests: a free-port picker and a unary-call wrapper that
# sends one request through a generated GRPCInferenceService RPC function. The worker package ships
# only server stubs, so the client stubs come from test/proto/grpc_inference_client.jl, included into
# a module that has the worker's message types in scope.

import gRPCClient
using Sockets

module WorkerInferenceClient
    using ReactantServer.inference
    include(joinpath(@__DIR__, "proto", "grpc_inference_client.jl"))
end
const WorkerRPC = WorkerInferenceClient.GRPCInferenceService

grpc_free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

# Send `request` through the generated RPC function `rpc` (e.g. `WorkerRPC.ModelInfer`) to the local
# server at `port`, returning the decoded response.
grpc_call(rpc, port::Integer, request) = rpc("127.0.0.1", port, request)
