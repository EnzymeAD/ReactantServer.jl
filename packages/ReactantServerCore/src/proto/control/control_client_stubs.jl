# gRPC client service stubs for the ReactantServer ControlService. Authored by hand like the
# server stubs (ProtoBuf.jl emits only the message types); kept beside the generated pb so the two
# travel together. Included by consumer packages (the gateway) into a module that has done
# `using ReactantServerCore.control`. Mirrors the structure of inference's grpc_client_stubs.jl.
import gRPCClient

ControlService_ModelControlStatus_Client(
	host, port;
	TRequest=ModelControlStatusRequest,
	TResponse=ModelControlStatusResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/reactant_control.ControlService/ModelControlStatus";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export ControlService_ModelControlStatus_Client

ControlService_CompactMemory_Client(
	host, port;
	TRequest=CompactMemoryRequest,
	TResponse=CompactMemoryResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=120,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/reactant_control.ControlService/CompactMemory";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export ControlService_CompactMemory_Client
