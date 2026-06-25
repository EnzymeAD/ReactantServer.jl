# gRPC server service stubs split from the generated protobuf. Included by consumer
# packages (worker, gateway) into a module that has done `using ReactantServerCore.inference`.
import gRPCServer

# gRPCServer.jl BEGIN
GRPCInferenceService_ServerLive_Method(; TRequest=ServerLiveRequest, TResponse=ServerLiveResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ServerLive")
export GRPCInferenceService_ServerLive_Method

GRPCInferenceService_ServerReady_Method(; TRequest=ServerReadyRequest, TResponse=ServerReadyResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ServerReady")
export GRPCInferenceService_ServerReady_Method

GRPCInferenceService_ModelReady_Method(; TRequest=ModelReadyRequest, TResponse=ModelReadyResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ModelReady")
export GRPCInferenceService_ModelReady_Method

GRPCInferenceService_ServerMetadata_Method(; TRequest=ServerMetadataRequest, TResponse=ServerMetadataResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ServerMetadata")
export GRPCInferenceService_ServerMetadata_Method

GRPCInferenceService_ModelMetadata_Method(; TRequest=ModelMetadataRequest, TResponse=ModelMetadataResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ModelMetadata")
export GRPCInferenceService_ModelMetadata_Method

GRPCInferenceService_ModelInfer_Method(; TRequest=ModelInferRequest, TResponse=ModelInferResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/ModelInfer")
export GRPCInferenceService_ModelInfer_Method

GRPCInferenceService_RepositoryIndex_Method(; TRequest=RepositoryIndexRequest, TResponse=RepositoryIndexResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/RepositoryIndex")
export GRPCInferenceService_RepositoryIndex_Method

GRPCInferenceService_SystemSharedMemoryStatus_Method(; TRequest=SystemSharedMemoryStatusRequest, TResponse=SystemSharedMemoryStatusResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/SystemSharedMemoryStatus")
export GRPCInferenceService_SystemSharedMemoryStatus_Method

GRPCInferenceService_SystemSharedMemoryRegister_Method(; TRequest=SystemSharedMemoryRegisterRequest, TResponse=SystemSharedMemoryRegisterResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/SystemSharedMemoryRegister")
export GRPCInferenceService_SystemSharedMemoryRegister_Method

GRPCInferenceService_SystemSharedMemoryUnregister_Method(; TRequest=SystemSharedMemoryUnregisterRequest, TResponse=SystemSharedMemoryUnregisterResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/SystemSharedMemoryUnregister")
export GRPCInferenceService_SystemSharedMemoryUnregister_Method

GRPCInferenceService_IsSameIPCNamespace_Method(; TRequest=IsSameIPCNamespaceRequest, TResponse=IsSameIPCNamespaceResponse) = gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/inference.GRPCInferenceService/IsSameIPCNamespace")
export GRPCInferenceService_IsSameIPCNamespace_Method

function register_GRPCInferenceService!(router; ServerLive=nothing, ServerReady=nothing, ModelReady=nothing, ServerMetadata=nothing, ModelMetadata=nothing, ModelInfer=nothing, RepositoryIndex=nothing, SystemSharedMemoryStatus=nothing, SystemSharedMemoryRegister=nothing, SystemSharedMemoryUnregister=nothing, IsSameIPCNamespace=nothing)
	ServerLive === nothing || gRPCServer.handle!(router, GRPCInferenceService_ServerLive_Method(), ServerLive)
	ServerReady === nothing || gRPCServer.handle!(router, GRPCInferenceService_ServerReady_Method(), ServerReady)
	ModelReady === nothing || gRPCServer.handle!(router, GRPCInferenceService_ModelReady_Method(), ModelReady)
	ServerMetadata === nothing || gRPCServer.handle!(router, GRPCInferenceService_ServerMetadata_Method(), ServerMetadata)
	ModelMetadata === nothing || gRPCServer.handle!(router, GRPCInferenceService_ModelMetadata_Method(), ModelMetadata)
	ModelInfer === nothing || gRPCServer.handle!(router, GRPCInferenceService_ModelInfer_Method(), ModelInfer)
	RepositoryIndex === nothing || gRPCServer.handle!(router, GRPCInferenceService_RepositoryIndex_Method(), RepositoryIndex)
	SystemSharedMemoryStatus === nothing || gRPCServer.handle!(router, GRPCInferenceService_SystemSharedMemoryStatus_Method(), SystemSharedMemoryStatus)
	SystemSharedMemoryRegister === nothing || gRPCServer.handle!(router, GRPCInferenceService_SystemSharedMemoryRegister_Method(), SystemSharedMemoryRegister)
	SystemSharedMemoryUnregister === nothing || gRPCServer.handle!(router, GRPCInferenceService_SystemSharedMemoryUnregister_Method(), SystemSharedMemoryUnregister)
	IsSameIPCNamespace === nothing || gRPCServer.handle!(router, GRPCInferenceService_IsSameIPCNamespace_Method(), IsSameIPCNamespace)
	return router
end
export register_GRPCInferenceService!

# gRPCServer.jl END
