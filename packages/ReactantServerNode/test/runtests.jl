using Test
using ReactantServerNode
import ReactantServerCore
import YAML

const RSN = ReactantServerNode

# Poll `pred` until it holds or `timeout` elapses; supervisor tests are timing-based by nature.
function wait_for(pred; timeout::Real = 30.0, interval::Real = 0.05)
    deadline = time() + timeout
    while time() < deadline
        pred() && return true
        sleep(interval)
    end
    return pred()
end

@testset "ReactantServerNode" begin
    include("test_gpus.jl")
    include("test_spec.jl")
    include("test_supervisor.jl")
end
