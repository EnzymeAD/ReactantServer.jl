using Test

# The package's LocalPreferences.toml pins Reactant's persistent compile cache to
# /var/cache/reactant-compile (a Docker volume mount point), which is not writable on a dev
# host and makes every CPU compile error with EACCES. Preferences are read at Reactant load
# time from the load path with earlier entries taking precedence, so before loading
# ReactantServer prepend an override that disables the persistent cache for this test process.
# Test compiles are tiny; skipping the cache also keeps them deterministic.
let prefdir = mktempdir()
    write(joinpath(prefdir, "Project.toml"), """
    [extras]
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    """)
    write(joinpath(prefdir, "LocalPreferences.toml"), """
    [Reactant]
    persistent_cache_enabled = false
    """)
    pushfirst!(LOAD_PATH, prefdir)
end

using ReactantServer

include("stablehlo_fixtures.jl")
include("grpc_helpers.jl")

@testset "ReactantServer" begin
    include("test_scheduler.jl")
    include("test_detection.jl")
    include("test_multishape.jl")
    include("test_observe.jl")
    include("test_worker_metrics.jl")
    include("test_mock_runtime.jl")
    include("test_meta.jl")
    include("test_weight_cache.jl")
    include("test_reactant_runtime.jl")
    include("test_tf32.jl")
    include("test_server_e2e.jl")
    include("test_watcher.jl")
    include("test_shared_memory.jl")
end
