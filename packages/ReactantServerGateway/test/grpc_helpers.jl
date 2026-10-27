# Shared helpers for the gRPC end-to-end tests: ephemeral-port startup and a unary-call wrapper
# that sends one request through a generated RPC function (e.g.
# `inference.GRPCInferenceService.ModelInfer`) to a local server.

import gRPCClient
import gRPCServer
using Sockets

# A port nothing is listening on, for a deliberately unreachable endpoint. Do not listen on it:
# another process can take it between this returning and the bind. Servers listen on port 0 instead.
grpc_free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

# Start a gRPC server built with port 0 and return the port the OS gave it.
grpc_start_ephemeral!(server) = (gRPCServer.start!(server); something(gRPCServer.bound_port(server)))

# Send `request` through the generated RPC function `rpc` to the local server at `port`, returning
# the decoded response. The RPC's service module fixes the path, so the inference data plane and
# the gateway's two control services all go through here.
grpc_call(rpc, port::Integer, request; deadline = 10) = rpc("127.0.0.1", port, request; deadline = deadline)
