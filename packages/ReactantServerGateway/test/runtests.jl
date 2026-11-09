using Test
using ReactantServerGateway
import ReactantServerCore

include("grpc_helpers.jl")

@testset "ReactantServerGateway" begin
    include("test_gateway.jl")
    include("test_route_retention.jl")   # after test_gateway.jl: reuses its mock workers
    include("test_gateway_shm.jl")
    include("test_scheduler.jl")
    include("test_lpt_packing.jl")
    include("test_claimed_items.jl")    # after test_lpt_packing.jl: reuses `_pk_state` (and test_scheduler.jl's helpers)
    include("test_metric_labels.jl")    # after test_lpt_packing.jl: reuses `_pk_state`
    include("test_gateway_control.jl")   # after test_lpt_packing.jl: reuses its mock-worker harness
    include("test_drain.jl")
    include("test_admission.jl")        # after test_drain.jl: reuses its slow worker
    include("test_client_giveups.jl")   # after test_drain.jl: reuses its slow worker
    include("test_worker_connections.jl")   # after test_drain.jl and test_gateway.jl: reuses their workers
    include("test_healthy_routing.jl")      # last: reuses the mock workers and helpers of the files above
end
