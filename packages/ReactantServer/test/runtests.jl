using Test
using ReactantServer

include("stablehlo_fixtures.jl")
include("grpc_helpers.jl")

@testset "ReactantServer" begin
    include("test_scheduler.jl")
    include("test_observe.jl")
    include("test_worker_metrics.jl")
    include("test_mock_runtime.jl")
    include("test_weight_cache.jl")
    include("test_reactant_runtime.jl")
    include("test_server_e2e.jl")
    include("test_watcher.jl")
    include("test_shared_memory.jl")
end
