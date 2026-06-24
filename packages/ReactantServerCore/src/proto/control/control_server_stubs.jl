# gRPC server service stubs for the ReactantServer ControlService. Authored by hand (ProtoBuf.jl
# emits only the message types); kept beside the generated pb so the two travel together. Included
# by the worker into a module that has done `using ReactantServerCore.control`, so the bare
# message-type references resolve. Mirrors the structure of inference's grpc_server_stubs.jl.
import gRPCServer

ControlService_ModelControlStatus_Method(; TRequest=ModelControlStatusRequest, TResponse=ModelControlStatusResponse) =
    gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/reactant_control.ControlService/ModelControlStatus")
export ControlService_ModelControlStatus_Method

ControlService_SetModelResidency_Method(; TRequest=SetModelResidencyRequest, TResponse=SetModelResidencyResponse) =
    gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/reactant_control.ControlService/SetModelResidency")
export ControlService_SetModelResidency_Method

ControlService_SetModelPolicy_Method(; TRequest=SetModelPolicyRequest, TResponse=SetModelPolicyResponse) =
    gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/reactant_control.ControlService/SetModelPolicy")
export ControlService_SetModelPolicy_Method

ControlService_CompactMemory_Method(; TRequest=CompactMemoryRequest, TResponse=CompactMemoryResponse) =
    gRPCServer.gRPCMethod{TRequest, false, TResponse, false}("/reactant_control.ControlService/CompactMemory")
export ControlService_CompactMemory_Method

function register_ControlService!(router; ModelControlStatus=nothing, SetModelResidency=nothing, SetModelPolicy=nothing, CompactMemory=nothing)
    ModelControlStatus === nothing || gRPCServer.handle!(router, ControlService_ModelControlStatus_Method(), ModelControlStatus)
    SetModelResidency === nothing || gRPCServer.handle!(router, ControlService_SetModelResidency_Method(), SetModelResidency)
    SetModelPolicy === nothing || gRPCServer.handle!(router, ControlService_SetModelPolicy_Method(), SetModelPolicy)
    CompactMemory === nothing || gRPCServer.handle!(router, ControlService_CompactMemory_Method(), CompactMemory)
    return router
end
export register_ControlService!
