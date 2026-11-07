using Test
using ReactantServerGateway
import ReactantServerCore

include("grpc_helpers.jl")

@testset "ReactantServerGateway" begin
    include("test_gateway.jl")
    include("test_gateway_shm.jl")
    include("test_scheduler.jl")
    include("test_lpt_packing.jl")
    include("test_claimed_items.jl")    # after test_lpt_packing.jl: reuses `_pk_state` (and test_scheduler.jl's helpers)
    include("test_gateway_control.jl")   # after test_lpt_packing.jl: reuses its mock-worker harness
    include("test_drain.jl")
    include("test_admission.jl")        # after test_drain.jl: reuses its slow worker
    include("test_client_giveups.jl")   # after test_drain.jl: reuses its slow worker
end
