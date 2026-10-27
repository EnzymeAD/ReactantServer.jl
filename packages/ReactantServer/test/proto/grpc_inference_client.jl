# Generated gRPCClient.jl client stubs for GRPCInferenceService, for the worker's tests only.
# protojl output from proto_src/grpc_predict_v2.proto (ProtoBuf.jl 1.3.0, gRPCClient.jl 1.2.0,
# `gRPCClient.grpc_register_service_codegen(legacy = false)`): the client section of the generated
# file, between its `# gRPCClient.jl BEGIN` / `END` markers; never hand-edit it. The worker package
# ships only server stubs, so test/grpc_helpers.jl includes this into a module that has the worker's
# message types in scope (the stubs resolve them through `parentmodule`).

# gRPCClient.jl BEGIN

baremodule GRPCInferenceService
    import gRPCClient
    import Base

    # Check compatibility between the loaded version of `gRPCClient` and
    # the version used to generate this module (1.2.0).
    gRPCClient.check_codegen_compat(Base.VersionNumber("1.2.0"))

    const ModelReadyRequest::DataType = Base.parentmodule(GRPCInferenceService).ModelReadyRequest
    const ModelMetadataRequest::DataType = Base.parentmodule(GRPCInferenceService).ModelMetadataRequest
    const ModelInferRequest::DataType = Base.parentmodule(GRPCInferenceService).ModelInferRequest
    const SystemSharedMemoryUnregisterRequest::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryUnregisterRequest
    const IsSameIPCNamespaceRequest::DataType = Base.parentmodule(GRPCInferenceService).IsSameIPCNamespaceRequest
    const SystemSharedMemoryRegisterResponse::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryRegisterResponse
    const SystemSharedMemoryUnregisterResponse::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryUnregisterResponse
    const IsSameIPCNamespaceResponse::DataType = Base.parentmodule(GRPCInferenceService).IsSameIPCNamespaceResponse
    const RepositoryIndexRequest::DataType = Base.parentmodule(GRPCInferenceService).RepositoryIndexRequest
    const SystemSharedMemoryStatusRequest::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryStatusRequest
    const ServerMetadataResponse::DataType = Base.parentmodule(GRPCInferenceService).ServerMetadataResponse
    const ServerLiveRequest::DataType = Base.parentmodule(GRPCInferenceService).ServerLiveRequest
    const ModelReadyResponse::DataType = Base.parentmodule(GRPCInferenceService).ModelReadyResponse
    const ServerReadyRequest::DataType = Base.parentmodule(GRPCInferenceService).ServerReadyRequest
    const RepositoryIndexResponse::DataType = Base.parentmodule(GRPCInferenceService).RepositoryIndexResponse
    const ServerMetadataRequest::DataType = Base.parentmodule(GRPCInferenceService).ServerMetadataRequest
    const ModelInferResponse::DataType = Base.parentmodule(GRPCInferenceService).ModelInferResponse
    const ServerLiveResponse::DataType = Base.parentmodule(GRPCInferenceService).ServerLiveResponse
    const ServerReadyResponse::DataType = Base.parentmodule(GRPCInferenceService).ServerReadyResponse
    const SystemSharedMemoryRegisterRequest::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryRegisterRequest
    const ModelMetadataResponse::DataType = Base.parentmodule(GRPCInferenceService).ModelMetadataResponse
    const SystemSharedMemoryStatusResponse::DataType = Base.parentmodule(GRPCInferenceService).SystemSharedMemoryStatusResponse

    # GRPCInferenceService.ServerLive
    Base.@inline function ServerLive(chan::gRPCClient.gRPCChannel, req::ServerLiveRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerLive), req, args...; kws...)
    end
    Base.@inline function ServerLive(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerLive), req, args...; kws...)
    end
    Base.@inline function ServerLive(host::AbstractString, port::Integer, args...; kws...)
        return ServerLive(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ServerLive)}) = "/inference.GRPCInferenceService/ServerLive"
    gRPCClient.isstreaming_request(::Type{typeof(ServerLive)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ServerLive)}) = false
    gRPCClient.request_type(::Type{typeof(ServerLive)}) = ServerLiveRequest
    gRPCClient.response_type(::Type{typeof(ServerLive)}) = ServerLiveResponse
    gRPCClient.request_type_displayname(::Type{typeof(ServerLive)}) = "ServerLiveRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ServerLive)}) = "ServerLiveResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ServerLive)) ServerLive
    export ServerLive


    # GRPCInferenceService.ServerReady
    Base.@inline function ServerReady(chan::gRPCClient.gRPCChannel, req::ServerReadyRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerReady), req, args...; kws...)
    end
    Base.@inline function ServerReady(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerReady), req, args...; kws...)
    end
    Base.@inline function ServerReady(host::AbstractString, port::Integer, args...; kws...)
        return ServerReady(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ServerReady)}) = "/inference.GRPCInferenceService/ServerReady"
    gRPCClient.isstreaming_request(::Type{typeof(ServerReady)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ServerReady)}) = false
    gRPCClient.request_type(::Type{typeof(ServerReady)}) = ServerReadyRequest
    gRPCClient.response_type(::Type{typeof(ServerReady)}) = ServerReadyResponse
    gRPCClient.request_type_displayname(::Type{typeof(ServerReady)}) = "ServerReadyRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ServerReady)}) = "ServerReadyResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ServerReady)) ServerReady
    export ServerReady


    # GRPCInferenceService.ModelReady
    Base.@inline function ModelReady(chan::gRPCClient.gRPCChannel, req::ModelReadyRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelReady), req, args...; kws...)
    end
    Base.@inline function ModelReady(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelReady), req, args...; kws...)
    end
    Base.@inline function ModelReady(host::AbstractString, port::Integer, args...; kws...)
        return ModelReady(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ModelReady)}) = "/inference.GRPCInferenceService/ModelReady"
    gRPCClient.isstreaming_request(::Type{typeof(ModelReady)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ModelReady)}) = false
    gRPCClient.request_type(::Type{typeof(ModelReady)}) = ModelReadyRequest
    gRPCClient.response_type(::Type{typeof(ModelReady)}) = ModelReadyResponse
    gRPCClient.request_type_displayname(::Type{typeof(ModelReady)}) = "ModelReadyRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ModelReady)}) = "ModelReadyResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ModelReady)) ModelReady
    export ModelReady


    # GRPCInferenceService.ServerMetadata
    Base.@inline function ServerMetadata(chan::gRPCClient.gRPCChannel, req::ServerMetadataRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerMetadata), req, args...; kws...)
    end
    Base.@inline function ServerMetadata(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ServerMetadata), req, args...; kws...)
    end
    Base.@inline function ServerMetadata(host::AbstractString, port::Integer, args...; kws...)
        return ServerMetadata(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ServerMetadata)}) = "/inference.GRPCInferenceService/ServerMetadata"
    gRPCClient.isstreaming_request(::Type{typeof(ServerMetadata)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ServerMetadata)}) = false
    gRPCClient.request_type(::Type{typeof(ServerMetadata)}) = ServerMetadataRequest
    gRPCClient.response_type(::Type{typeof(ServerMetadata)}) = ServerMetadataResponse
    gRPCClient.request_type_displayname(::Type{typeof(ServerMetadata)}) = "ServerMetadataRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ServerMetadata)}) = "ServerMetadataResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ServerMetadata)) ServerMetadata
    export ServerMetadata


    # GRPCInferenceService.ModelMetadata
    Base.@inline function ModelMetadata(chan::gRPCClient.gRPCChannel, req::ModelMetadataRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelMetadata), req, args...; kws...)
    end
    Base.@inline function ModelMetadata(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelMetadata), req, args...; kws...)
    end
    Base.@inline function ModelMetadata(host::AbstractString, port::Integer, args...; kws...)
        return ModelMetadata(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ModelMetadata)}) = "/inference.GRPCInferenceService/ModelMetadata"
    gRPCClient.isstreaming_request(::Type{typeof(ModelMetadata)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ModelMetadata)}) = false
    gRPCClient.request_type(::Type{typeof(ModelMetadata)}) = ModelMetadataRequest
    gRPCClient.response_type(::Type{typeof(ModelMetadata)}) = ModelMetadataResponse
    gRPCClient.request_type_displayname(::Type{typeof(ModelMetadata)}) = "ModelMetadataRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ModelMetadata)}) = "ModelMetadataResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ModelMetadata)) ModelMetadata
    export ModelMetadata


    # GRPCInferenceService.ModelInfer
    Base.@inline function ModelInfer(chan::gRPCClient.gRPCChannel, req::ModelInferRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelInfer), req, args...; kws...)
    end
    Base.@inline function ModelInfer(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(ModelInfer), req, args...; kws...)
    end
    Base.@inline function ModelInfer(host::AbstractString, port::Integer, args...; kws...)
        return ModelInfer(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(ModelInfer)}) = "/inference.GRPCInferenceService/ModelInfer"
    gRPCClient.isstreaming_request(::Type{typeof(ModelInfer)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(ModelInfer)}) = false
    gRPCClient.request_type(::Type{typeof(ModelInfer)}) = ModelInferRequest
    gRPCClient.response_type(::Type{typeof(ModelInfer)}) = ModelInferResponse
    gRPCClient.request_type_displayname(::Type{typeof(ModelInfer)}) = "ModelInferRequest"
    gRPCClient.response_type_displayname(::Type{typeof(ModelInfer)}) = "ModelInferResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(ModelInfer)) ModelInfer
    export ModelInfer


    # GRPCInferenceService.RepositoryIndex
    Base.@inline function RepositoryIndex(chan::gRPCClient.gRPCChannel, req::RepositoryIndexRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(RepositoryIndex), req, args...; kws...)
    end
    Base.@inline function RepositoryIndex(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(RepositoryIndex), req, args...; kws...)
    end
    Base.@inline function RepositoryIndex(host::AbstractString, port::Integer, args...; kws...)
        return RepositoryIndex(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(RepositoryIndex)}) = "/inference.GRPCInferenceService/RepositoryIndex"
    gRPCClient.isstreaming_request(::Type{typeof(RepositoryIndex)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(RepositoryIndex)}) = false
    gRPCClient.request_type(::Type{typeof(RepositoryIndex)}) = RepositoryIndexRequest
    gRPCClient.response_type(::Type{typeof(RepositoryIndex)}) = RepositoryIndexResponse
    gRPCClient.request_type_displayname(::Type{typeof(RepositoryIndex)}) = "RepositoryIndexRequest"
    gRPCClient.response_type_displayname(::Type{typeof(RepositoryIndex)}) = "RepositoryIndexResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(RepositoryIndex)) RepositoryIndex
    export RepositoryIndex


    # GRPCInferenceService.SystemSharedMemoryStatus
    Base.@inline function SystemSharedMemoryStatus(chan::gRPCClient.gRPCChannel, req::SystemSharedMemoryStatusRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryStatus), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryStatus(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryStatus), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryStatus(host::AbstractString, port::Integer, args...; kws...)
        return SystemSharedMemoryStatus(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(SystemSharedMemoryStatus)}) = "/inference.GRPCInferenceService/SystemSharedMemoryStatus"
    gRPCClient.isstreaming_request(::Type{typeof(SystemSharedMemoryStatus)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(SystemSharedMemoryStatus)}) = false
    gRPCClient.request_type(::Type{typeof(SystemSharedMemoryStatus)}) = SystemSharedMemoryStatusRequest
    gRPCClient.response_type(::Type{typeof(SystemSharedMemoryStatus)}) = SystemSharedMemoryStatusResponse
    gRPCClient.request_type_displayname(::Type{typeof(SystemSharedMemoryStatus)}) = "SystemSharedMemoryStatusRequest"
    gRPCClient.response_type_displayname(::Type{typeof(SystemSharedMemoryStatus)}) = "SystemSharedMemoryStatusResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(SystemSharedMemoryStatus)) SystemSharedMemoryStatus
    export SystemSharedMemoryStatus


    # GRPCInferenceService.SystemSharedMemoryRegister
    Base.@inline function SystemSharedMemoryRegister(chan::gRPCClient.gRPCChannel, req::SystemSharedMemoryRegisterRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryRegister), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryRegister(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryRegister), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryRegister(host::AbstractString, port::Integer, args...; kws...)
        return SystemSharedMemoryRegister(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(SystemSharedMemoryRegister)}) = "/inference.GRPCInferenceService/SystemSharedMemoryRegister"
    gRPCClient.isstreaming_request(::Type{typeof(SystemSharedMemoryRegister)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(SystemSharedMemoryRegister)}) = false
    gRPCClient.request_type(::Type{typeof(SystemSharedMemoryRegister)}) = SystemSharedMemoryRegisterRequest
    gRPCClient.response_type(::Type{typeof(SystemSharedMemoryRegister)}) = SystemSharedMemoryRegisterResponse
    gRPCClient.request_type_displayname(::Type{typeof(SystemSharedMemoryRegister)}) = "SystemSharedMemoryRegisterRequest"
    gRPCClient.response_type_displayname(::Type{typeof(SystemSharedMemoryRegister)}) = "SystemSharedMemoryRegisterResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(SystemSharedMemoryRegister)) SystemSharedMemoryRegister
    export SystemSharedMemoryRegister


    # GRPCInferenceService.SystemSharedMemoryUnregister
    Base.@inline function SystemSharedMemoryUnregister(chan::gRPCClient.gRPCChannel, req::SystemSharedMemoryUnregisterRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryUnregister), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryUnregister(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(SystemSharedMemoryUnregister), req, args...; kws...)
    end
    Base.@inline function SystemSharedMemoryUnregister(host::AbstractString, port::Integer, args...; kws...)
        return SystemSharedMemoryUnregister(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(SystemSharedMemoryUnregister)}) = "/inference.GRPCInferenceService/SystemSharedMemoryUnregister"
    gRPCClient.isstreaming_request(::Type{typeof(SystemSharedMemoryUnregister)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(SystemSharedMemoryUnregister)}) = false
    gRPCClient.request_type(::Type{typeof(SystemSharedMemoryUnregister)}) = SystemSharedMemoryUnregisterRequest
    gRPCClient.response_type(::Type{typeof(SystemSharedMemoryUnregister)}) = SystemSharedMemoryUnregisterResponse
    gRPCClient.request_type_displayname(::Type{typeof(SystemSharedMemoryUnregister)}) = "SystemSharedMemoryUnregisterRequest"
    gRPCClient.response_type_displayname(::Type{typeof(SystemSharedMemoryUnregister)}) = "SystemSharedMemoryUnregisterResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(SystemSharedMemoryUnregister)) SystemSharedMemoryUnregister
    export SystemSharedMemoryUnregister


    # GRPCInferenceService.IsSameIPCNamespace
    Base.@inline function IsSameIPCNamespace(chan::gRPCClient.gRPCChannel, req::IsSameIPCNamespaceRequest, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(IsSameIPCNamespace), req, args...; kws...)
    end
    Base.@inline function IsSameIPCNamespace(chan::gRPCClient.gRPCChannel, req::Base.Vector{UInt8}, args...; kws...)
        return gRPCClient.grpc_call_unary(chan, typeof(IsSameIPCNamespace), req, args...; kws...)
    end
    Base.@inline function IsSameIPCNamespace(host::AbstractString, port::Integer, args...; kws...)
        return IsSameIPCNamespace(gRPCClient.gRPCChannel(host, port), args...; kws...)
    end
    gRPCClient.rpc_path(::Type{typeof(IsSameIPCNamespace)}) = "/inference.GRPCInferenceService/IsSameIPCNamespace"
    gRPCClient.isstreaming_request(::Type{typeof(IsSameIPCNamespace)}) = false
    gRPCClient.isstreaming_response(::Type{typeof(IsSameIPCNamespace)}) = false
    gRPCClient.request_type(::Type{typeof(IsSameIPCNamespace)}) = IsSameIPCNamespaceRequest
    gRPCClient.response_type(::Type{typeof(IsSameIPCNamespace)}) = IsSameIPCNamespaceResponse
    gRPCClient.request_type_displayname(::Type{typeof(IsSameIPCNamespace)}) = "IsSameIPCNamespaceRequest"
    gRPCClient.response_type_displayname(::Type{typeof(IsSameIPCNamespace)}) = "IsSameIPCNamespaceResponse"
    Base.@doc gRPCClient.grpc_generate_rpc_docstring(typeof(IsSameIPCNamespace)) IsSameIPCNamespace
    export IsSameIPCNamespace


end # module GRPCInferenceService
export GRPCInferenceService

# gRPCClient.jl END
