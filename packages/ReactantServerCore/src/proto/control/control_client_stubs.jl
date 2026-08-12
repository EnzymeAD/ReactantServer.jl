# gRPC client service stubs for the ReactantServer ControlService. Authored by hand like the
# server stubs (ProtoBuf.jl emits only the message types); kept beside the generated pb so the two
# travel together. Included by consumer packages (the gateway) into a module that has done
# `using ReactantServerCore.control`. Mirrors the structure of inference's grpc_client_stubs.jl.
import gRPCClient

ControlService_ModelControlStatus_Client(
    host, port;
    TRequest = ModelControlStatusRequest,
    TResponse = ModelControlStatusResponse,
    secure = false,
    grpc = gRPCClient.grpc_global_handle(),
    deadline = 10,
    keepalive = 60,
    max_send_message_length = 4 * 1024 * 1024,
    max_recieve_message_length = 4 * 1024 * 1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.ControlService/ModelControlStatus";
    secure = secure,
    grpc = grpc,
    deadline = deadline,
    keepalive = keepalive,
    max_send_message_length = max_send_message_length,
    max_recieve_message_length = max_recieve_message_length,
)
export ControlService_ModelControlStatus_Client

ControlService_CompactMemory_Client(
    host, port;
    TRequest = CompactMemoryRequest,
    TResponse = CompactMemoryResponse,
    secure = false,
    grpc = gRPCClient.grpc_global_handle(),
    deadline = 120,
    keepalive = 60,
    max_send_message_length = 4 * 1024 * 1024,
    max_recieve_message_length = 4 * 1024 * 1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.ControlService/CompactMemory";
    secure = secure,
    grpc = grpc,
    deadline = deadline,
    keepalive = keepalive,
    max_send_message_length = max_send_message_length,
    max_recieve_message_length = max_recieve_message_length,
)
export ControlService_CompactMemory_Client

# --- GatewayControlService (the gateway's own scheduling control plane) -------------------------
# These four take their deadline through `options...` rather than hardcoding one. Pass a deadline
# above the gateway's bounded-wait cap when calling `Repack` with `wait_seconds` set, or the client
# gives up before the gateway answers.
GatewayControlService_GetSchedulingStatus_Client(
    host, port;
    TRequest = GetSchedulingStatusRequest,
    TResponse = GetSchedulingStatusResponse,
    grpc = gRPCClient.grpc_global_handle(),
    options...
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.GatewayControlService/GetSchedulingStatus";
    grpc = grpc,
    options...
)
export GatewayControlService_GetSchedulingStatus_Client

GatewayControlService_SetSchedulingPolicy_Client(
    host, port;
    TRequest = SetSchedulingPolicyRequest,
    TResponse = SetSchedulingPolicyResponse,
    grpc = gRPCClient.grpc_global_handle(),
    options...
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.GatewayControlService/SetSchedulingPolicy";
    grpc = grpc,
    options...
)
export GatewayControlService_SetSchedulingPolicy_Client

GatewayControlService_SetModelPlacement_Client(
    host, port;
    TRequest = SetModelPlacementRequest,
    TResponse = SetModelPlacementResponse,
    grpc = gRPCClient.grpc_global_handle(),
    options...
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.GatewayControlService/SetModelPlacement";
    grpc = grpc,
    options...
)
export GatewayControlService_SetModelPlacement_Client

GatewayControlService_Repack_Client(
    host, port;
    TRequest = RepackRequest,
    TResponse = RepackResponse,
    grpc = gRPCClient.grpc_global_handle(),
    options...
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
    host, port, "/reactant_control.GatewayControlService/Repack";
    grpc = grpc,
    options...
)
export GatewayControlService_Repack_Client
