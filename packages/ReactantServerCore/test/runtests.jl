using Test
using ReactantServerCore

# The moved unit tests were written against the original flat `ReactantServer` module and
# reference Core symbols as `ReactantServer.X`. Core now owns those symbols, so this alias lets
# the files run unchanged.
const ReactantServer = ReactantServerCore

@testset "ReactantServerCore" begin
    include("test_dtypes.jl")
    include("test_manifest.jl")
    include("test_config.jl")
    include("test_codec.jl")
    include("test_buffer_pool.jl")
    include("test_weight_store.jl")
end
