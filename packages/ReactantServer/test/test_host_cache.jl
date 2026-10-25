# The host weight cache (runtime.host_weight_budget) against the MockBackend. The device budget is
# one model, so every dispatch to a different model is a GPU load; a counting weights double shows
# which loads read the file (a miss) and which came from a cached host copy (a hit). Uses the
# helpers of test_weight_cache.jl (_W, _WBYTES, _batched_manifest, _dispatch1!).

mutable struct CountingWeights
    tensors::Dict{String, Array}
    reads::Int
end
CountingWeights() = CountingWeights(Dict{String, Array}("w" => copy(_W)), 0)
Base.haskey(f::CountingWeights, k) = haskey(f.tensors, k)
Base.getindex(f::CountingWeights, k) = (f.reads += 1; f.tensors[k])
# Shape queries are metadata, not a read of the weights.
ReactantServer.weight_spec(f::CountingWeights, k) = (eltype(f.tensors[k]), size(f.tensors[k]))

function _hc_entry(name)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    lm = ReactantServer.LoadedModel(sig, Dict{Int, Any}(1 => exec), nothing, false, _WBYTES, nothing)
    return ReactantServer.ModelEntry(
        name, _batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", CountingWeights(), lm, nothing,
        identity, identity
    )
end

function _hc_scheduler(names...; device_models::Int = 1, host_budget = nothing, store = ReactantServer.PrivateWeightStore())
    entries = [_hc_entry(n) for n in names]
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
    end
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    for e in entries
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched.weight_cache = ReactantServer.WeightCache(
        backend, pool, reg, device_models * _WBYTES; store = store, host_budget = host_budget
    )
    return sched
end

_reads(sched, name) = sched.registry.by_name[name].weights.reads
_served(sched, name) = _dispatch1!(sched, name)[1].data == reshape(Float32[10, 100], 2, 1)

@testset "host cache (private): misses read the file, hits do not, least recently loaded is evicted" begin
    sched = _hc_scheduler("A", "B", "C"; host_budget = 2 * sizeof(_W))
    wc = sched.weight_cache
    @test ReactantServer.host_budgeted(wc)
    @test _served(sched, "A") && _reads(sched, "A") == 1
    @test _served(sched, "B") && _reads(sched, "B") == 1           # A leaves the device
    @test _served(sched, "A") && _reads(sched, "A") == 1           # reloaded from the host copy
    @test (wc.host_hits, wc.host_misses) == (1, 2)
    @test _served(sched, "C") && _reads(sched, "C") == 1           # evicts B, the least recently loaded
    @test sort(collect(keys(wc.host_cached))) == ["A", "C"]
    @test _served(sched, "B") && _reads(sched, "B") == 2           # B was evicted, so a miss
    @test ReactantServer.host_cached_bytes(wc) == 2 * sizeof(_W)
    st = ReactantServer.weight_cache_stats(wc)
    @test st.host_budget == 2 * sizeof(_W) && st.host_hits == 1 && st.host_misses == 4
end

@testset "host cache: a model kept on the device ages out ahead of one that is reloaded" begin
    # Room on the device for A and B, so A stays resident while C and B take turns.
    sched = _hc_scheduler("A", "B", "C"; device_models = 2, host_budget = 2 * sizeof(_W))
    wc = sched.weight_cache
    _served(sched, "A")
    _served(sched, "B")
    for _ in 1:5
        _served(sched, "A")                      # served from the device: not a load
    end
    @test wc.host_lru == ["A", "B"]              # requests alone do not refresh A
    ReactantServer.compact!(wc, sched.registry)  # frees the device copies; both reload from host
    _served(sched, "B")
    _served(sched, "C")                          # needs room: evicts A, loaded least recently
    @test !haskey(wc.host_cached, "A") && haskey(wc.host_cached, "B") && haskey(wc.host_cached, "C")
end

@testset "host cache: a zero budget keeps no copies" begin
    sched = _hc_scheduler("A", "B"; host_budget = 0)
    for _ in 1:2
        _served(sched, "A")
        _served(sched, "B")
    end
    @test _reads(sched, "A") == 2 && _reads(sched, "B") == 2
    @test isempty(sched.weight_cache.host_cached) && ReactantServer.host_cached_bytes(sched.weight_cache) == 0
end

@testset "host cache: pinned floors count against the budget and move to the cache when unpinned" begin
    sched = _hc_scheduler("A", "B", "C"; host_budget = 2 * sizeof(_W))
    wc = sched.weight_cache
    A, B, C = (sched.registry.by_name[n] for n in ("A", "B", "C"))
    ReactantServer.set_residency_state!(wc, A, ReactantServer.PINNED_SYSTEM)
    @test ReactantServer.host_cached_bytes(wc) == sizeof(_W)
    _served(sched, "B")
    _served(sched, "C")                          # the floor leaves room for one copy: B goes
    @test collect(keys(wc.host_cached)) == ["C"]
    # Pinning C reuses its cached copy (no read) and leaves no room for copies.
    n = _reads(sched, "C")
    ReactantServer.set_residency_state!(wc, C, ReactantServer.PINNED_SYSTEM)
    @test _reads(sched, "C") == n && isempty(wc.host_cached)
    @test ReactantServer.host_cached_bytes(wc) == 2 * sizeof(_W)
    # A third floor overcommits: it loads, with a warning.
    @test_logs (:warn, r"overcommitted") match_mode = :any ReactantServer.set_residency_state!(wc, B, ReactantServer.PINNED_SYSTEM)
    @test B.executable.host_weights !== nothing
    ReactantServer.set_residency_state!(wc, B, ReactantServer.UNPINNED)
    ReactantServer.set_residency_state!(wc, A, ReactantServer.UNPINNED)
    @test haskey(wc.host_cached, "A")           # the dropped floor stays cached
    n = _reads(sched, "A")
    _served(sched, "A")
    @test _reads(sched, "A") == n               # and serves the next load
end

@testset "host cache: removing or renaming a model updates its cached copy" begin
    sched = _hc_scheduler("A", "B"; device_models = 2, host_budget = 4 * sizeof(_W))
    wc = sched.weight_cache
    _served(sched, "A")
    _served(sched, "B")
    ReactantServer.rename!(wc, "A", "A2")
    @test haskey(wc.host_cached, "A2") && wc.host_lru == ["A2", "B"]
    ReactantServer.release_all!(wc, sched.registry.by_name["B"])
    @test !haskey(wc.host_cached, "B") && wc.host_cached_bytes == sizeof(_W)
end

@testset "host cache (shared): one node budget in /dev/shm, least recently loaded evicted" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        prefix = "thc" * string(getpid())
        R = ReactantServer.ReactantServerCore._layout([(Float32, (2,))])[2]
        store = ReactantServer.SharedWeightStore(; budget = 2R, prefix = prefix)
        region(k) = "/dev/shm/" * prefix * "-" * k * "-" * string(ReactantServer.weights_digest(k, [(Float32, (2,))]); base = 16)
        try
            sched = _hc_scheduler("A", "B", "C"; store = store)
            wc = sched.weight_cache
            @test ReactantServer.host_budgeted(wc) && ReactantServer.host_budget(wc) == 2R
            @test _served(sched, "A") && _served(sched, "B")
            @test isfile(region("A")) && isfile(region("B"))
            @test _served(sched, "A") && _reads(sched, "A") == 1   # a hit on the region
            sleep(0.02)
            @test _served(sched, "C")                              # evicts B
            @test !isfile(region("B")) && isfile(region("A")) && isfile(region("C"))
            @test (wc.host_hits, wc.host_misses) == (1, 3)
            @test ReactantServer.host_cached_bytes(wc) == 2R
            # A second worker on the node (a second store, same prefix) shares the cache.
            sched2 = _hc_scheduler("A"; store = ReactantServer.SharedWeightStore(; budget = 2R, prefix = prefix))
            @test _served(sched2, "A") && _reads(sched2, "A") == 0
            @test sched2.weight_cache.host_hits == 1
        finally
            for f in readdir("/dev/shm"; join = true)
                startswith(basename(f), prefix * "-") && rm(f; force = true)
            end
        end
    end
end

@testset "a reload from disk after startup warns once per model; startup loads do not" begin
    sched = _hc_scheduler("A", "B"; host_budget = 0)
    wc = sched.weight_cache
    @test_logs _served(sched, "A")                              # startup (not serving): silent
    wc.serving = true
    @test_logs _served(sched, "B")                              # B's first load: expected, silent
    @test_logs (:warn, r"reloaded from disk") _served(sched, "A")
    @test_logs (:warn, r"reloaded from disk") _served(sched, "B")
    @test_logs _served(sched, "A")                              # once per model
    # A reload served from a host copy does not warn.
    sched = _hc_scheduler("A", "B"; host_budget = 2 * sizeof(_W))
    sched.weight_cache.serving = true
    _served(sched, "A")
    _served(sched, "B")
    @test_logs _served(sched, "A")
end

@testset "startup fills the device with the on-demand models that fit, largest first" begin
    sched = _hc_scheduler("A", "B", "C"; device_models = 2, host_budget = 0)
    wc = sched.weight_cache
    @test ReactantServer._fill_device!(sched) == 2
    @test wc.resident_bytes == 2 * _WBYTES
    @test count(n -> sched.registry.by_name[n].executable.weights !== nothing, ("A", "B", "C")) == 2
    @test ReactantServer._fill_device!(sched) == 0              # nothing more fits
    sched = _hc_scheduler("A"; host_budget = 0)
    sched.weight_cache.mode = ReactantServer.EXTERNALLY_MANAGED
    @test ReactantServer._fill_device!(sched) == 0              # the control plane decides there
end

@testset "the host weight budget startup line states each worker's share" begin
    summary = ReactantServer._host_budget_summary
    @test occursin("each worker may use 2.000 GiB", summary(8 << 30, false, 4))
    @test occursin("split across 4 workers", summary(8 << 30, false, 4))
    @test occursin("shared by every worker", summary(8 << 30, true, 4))
    @test occursin("for this worker", summary(8 << 30, false, 1))
    @test occursin("no host RAM copies", summary(0, false, 4))
    @test occursin("unlimited", summary(nothing, false, 4))
end

@testset "a host weight budget resolves unspecified residency to cached, not pinned" begin
    cfg(budget) = ReactantServer.build_config(
        Dict{String, Any}(
            "model_dirs" => ["/x"],
            "runtime" => budget === nothing ? Dict{String, Any}() : Dict{String, Any}("host_weight_budget" => budget),
            "scheduler" => Dict{String, Any}("models" => Dict{String, Any}("P" => Dict{String, Any}("residency" => "system"))),
        )
    )
    @test ReactantServer._resolve_residency(cfg(nothing), "M", true) == ReactantServer.UNPINNED   # default 0
    @test ReactantServer._resolve_residency(cfg("unlimited"), "M", true) == ReactantServer.PINNED_SYSTEM
    @test ReactantServer._resolve_residency(cfg("4g"), "M", true) == ReactantServer.UNPINNED
    @test ReactantServer._resolve_residency(cfg("4g"), "P", true) == ReactantServer.PINNED_SYSTEM
    @test ReactantServer._resolve_residency(cfg("0b"), "M", false) == ReactantServer.UNPINNED
end
