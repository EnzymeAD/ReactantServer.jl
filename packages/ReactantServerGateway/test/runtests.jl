using Test
using ReactantServerGateway
import ReactantServerCore

include("grpc_helpers.jl")

@testset "ReactantServerGateway" begin
    include("test_gateway.jl")
    include("test_scheduler.jl")
    include("test_lpt_packing.jl")
    include("test_gateway_control.jl")   # after test_lpt_packing.jl: reuses its mock-worker harness
end
