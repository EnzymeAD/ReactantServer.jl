# gRPC client service stubs split from the generated protobuf. Included by consumer
# packages (client, gateway) into a module that has done `using ReactantServerCore.inference`.
import gRPCClient

# gRPCClient.jl BEGIN
GRPCInferenceService_ServerLive_Client(
	host, port;
	TRequest=ServerLiveRequest,
	TResponse=ServerLiveResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ServerLive";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ServerLive_Client

GRPCInferenceService_ServerReady_Client(
	host, port;
	TRequest=ServerReadyRequest,
	TResponse=ServerReadyResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ServerReady";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ServerReady_Client

GRPCInferenceService_ModelReady_Client(
	host, port;
	TRequest=ModelReadyRequest,
	TResponse=ModelReadyResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ModelReady";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ModelReady_Client

GRPCInferenceService_ServerMetadata_Client(
	host, port;
	TRequest=ServerMetadataRequest,
	TResponse=ServerMetadataResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ServerMetadata";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ServerMetadata_Client

GRPCInferenceService_ModelMetadata_Client(
	host, port;
	TRequest=ModelMetadataRequest,
	TResponse=ModelMetadataResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ModelMetadata";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ModelMetadata_Client

GRPCInferenceService_ModelInfer_Client(
	host, port;
	TRequest=ModelInferRequest,
	TResponse=ModelInferResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/ModelInfer";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_ModelInfer_Client

GRPCInferenceService_RepositoryIndex_Client(
	host, port;
	TRequest=RepositoryIndexRequest,
	TResponse=RepositoryIndexResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/RepositoryIndex";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_RepositoryIndex_Client

GRPCInferenceService_SystemSharedMemoryStatus_Client(
	host, port;
	TRequest=SystemSharedMemoryStatusRequest,
	TResponse=SystemSharedMemoryStatusResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/SystemSharedMemoryStatus";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_SystemSharedMemoryStatus_Client

GRPCInferenceService_SystemSharedMemoryRegister_Client(
	host, port;
	TRequest=SystemSharedMemoryRegisterRequest,
	TResponse=SystemSharedMemoryRegisterResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/SystemSharedMemoryRegister";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_SystemSharedMemoryRegister_Client

GRPCInferenceService_SystemSharedMemoryUnregister_Client(
	host, port;
	TRequest=SystemSharedMemoryUnregisterRequest,
	TResponse=SystemSharedMemoryUnregisterResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/SystemSharedMemoryUnregister";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_SystemSharedMemoryUnregister_Client

GRPCInferenceService_IsSameIPCNamespace_Client(
	host, port;
	TRequest=IsSameIPCNamespaceRequest,
	TResponse=IsSameIPCNamespaceResponse,
	secure=false,
	grpc=gRPCClient.grpc_global_handle(),
	deadline=10,
	keepalive=60,
	max_send_message_length = 4*1024*1024,
	max_recieve_message_length = 4*1024*1024,
) = gRPCClient.gRPCServiceClient{TRequest, false, TResponse, false}(
	host, port, "/inference.GRPCInferenceService/IsSameIPCNamespace";
	secure=secure,
	grpc=grpc,
	deadline=deadline,
	keepalive=keepalive,
	max_send_message_length=max_send_message_length,
	max_recieve_message_length=max_recieve_message_length,
)
export GRPCInferenceService_IsSameIPCNamespace_Client
# gRPCClient.jl END
