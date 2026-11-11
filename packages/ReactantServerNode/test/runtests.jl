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
    include("test_wall_clock.jl")
    include("test_gpus.jl")
    # build_supervisor falls back to the image's gateway.yml mount point whenever that file exists,
    # and a host running a deployment may have one. Point the fallback at a path that does not exist
    # so the spec tests see the env-only gateway on any host, then restore the production default.
    default_gateway_file = RSN._DEFAULT_GATEWAY_FILE[]
    RSN._DEFAULT_GATEWAY_FILE[] = joinpath(mktempdir(), "gateway.yml")
    try
        include("test_spec.jl")
    finally
        RSN._DEFAULT_GATEWAY_FILE[] = default_gateway_file
    end
    @test RSN._DEFAULT_GATEWAY_FILE[] == "/etc/reactantserver/gateway.yml"
    include("test_supervisor.jl")
end
