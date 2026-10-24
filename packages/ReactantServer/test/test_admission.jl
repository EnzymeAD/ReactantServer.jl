# Fragmentation-aware admission (runtime/admission.jl) and its victim search
# (runtime/victim_search.jl), against a fake allocator that fragments the way XLA's BFC does:
# address-ordered best fit, 256-byte rounding, coalescing frees, and a refused allocation when no
# single free block is large enough. Executions claim their scratch transiently, so running a model
# fails exactly when its scratch block does not fit. Reactant-free and deterministic.

using Random

const _KiB = 1024

# ── The fake allocator ──────────────────────────────────────────────────────────────────────────

mutable struct FragBackend <: ReactantServer.AbstractBackend
    limit::Int
    free::Vector{Tuple{Int, Int}}              # (offset, size), address-ordered, coalesced
    live::IdDict{Any, Tuple{Int, Int}}         # device buffer -> (offset, size)
    scratch::IdDict{Any, Int}                  # executable -> transient bytes per run
    fail_next_upload::Bool
    refusals::Int
end
FragBackend(limit) = FragBackend(limit, [(0, limit)], IdDict{Any, Tuple{Int, Int}}(), IdDict{Any, Int}(), false, 0)

function _frag_alloc!(b::FragBackend, n::Int)
    r = cld(n, 256) * 256
    best = 0
    for (k, (_, sz)) in enumerate(b.free)
        sz >= r && (best == 0 || sz < b.free[best][2]) && (best = k)
    end
    if best == 0
        b.refusals += 1
        error("RESOURCE_EXHAUSTED: fake allocator cannot place $r bytes")
    end
    off, sz = b.free[best]
    sz == r ? deleteat!(b.free, best) : (b.free[best] = (off + r, sz - r))
    return off, r
end

function _frag_release!(b::FragBackend, off::Int, r::Int)
    k = searchsortedfirst(b.free, (off, r))
    insert!(b.free, k, (off, r))
    if k < length(b.free) && b.free[k][1] + b.free[k][2] == b.free[k + 1][1]
        b.free[k] = (b.free[k][1], b.free[k][2] + b.free[k + 1][2]); deleteat!(b.free, k + 1)
    end
    if k > 1 && b.free[k - 1][1] + b.free[k - 1][2] == b.free[k][1]
        b.free[k - 1] = (b.free[k - 1][1], b.free[k - 1][2] + b.free[k][2]); deleteat!(b.free, k)
    end
    return nothing
end

ReactantServer.make_client(::FragBackend, platform::String; kwargs...) = ReactantServer.MockClient()
ReactantServer.select_device(::FragBackend, ::ReactantServer.MockClient, ordinal::Int) = ReactantServer.MockDevice(ordinal)
ReactantServer.device_ordinal(::FragBackend, d::ReactantServer.MockDevice) = d.ordinal
function ReactantServer.to_device(b::FragBackend, ::ReactantServer.MockClient, a::Array, ::ReactantServer.MockDevice)
    if b.fail_next_upload
        b.fail_next_upload = false
        error("RESOURCE_EXHAUSTED: injected upload failure")
    end
    buf = ReactantServer.MockBuffer(copy(a))
    b.live[buf] = _frag_alloc!(b, sizeof(a))
    return buf
end
ReactantServer.buffer_eltype(::FragBackend, m::ReactantServer.MockBuffer) = eltype(m.data)
ReactantServer.buffer_size(::FragBackend, m::ReactantServer.MockBuffer) = reverse(size(m.data))
ReactantServer.to_host!(::FragBackend, m::ReactantServer.MockBuffer, dest::Array) = (copyto!(dest, m.data); dest)
function ReactantServer.free_buffer!(b::FragBackend, m::ReactantServer.MockBuffer)
    m.freed && return nothing
    m.freed = true
    off, r = pop!(b.live, m)
    _frag_release!(b, off, r)
    return nothing
end
function ReactantServer.execute_single_device(
        b::FragBackend, exec::ReactantServer.MockExecutable, ::ReactantServer.MockDevice,
        buffers::AbstractVector, donated::AbstractVector{Bool}, num_outputs::Int
    )
    s = get(b.scratch, exec, 0)
    if s > 0                                   # the run's scratch: claimed, then released
        off, r = _frag_alloc!(b, s)
        _frag_release!(b, off, r)
    end
    outs = exec.fn([m.data for m in buffers])
    return Any[(buf = ReactantServer.MockBuffer(o); b.live[buf] = _frag_alloc!(b, sizeof(o)); buf) for o in outs]
end
function ReactantServer.device_memory_stats(b::FragBackend, pool)
    free = sum(last, b.free; init = 0)
    largest = maximum(last, b.free; init = 0)
    return (
        in_use = b.limit - free, limit = b.limit, free = free, peak_in_use = 0,
        pool_bytes = b.limit, peak_pool_bytes = b.limit, largest_free = largest,
    )
end

# ── Models and a scheduler on it ─────────────────────────────────────────────────────────────────

# A model with weight tensors of `kib` KiB each and a measured scratch of `scratch` bytes.
function _frag_entry(b::FragBackend, name; kib::Vector{Int}, scratch::Int = 0, pinned::Bool = false)
    names = ["w$i" for i in eachindex(kib)]
    tensors = Dict{String, Array}(n => zeros(Float32, k * _KiB ÷ 4) for (n, k) in zip(names, kib))
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], names, 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [copy(args[1])], 1)
    b.scratch[exec] = scratch
    lm = ReactantServer.LoadedModel(sig, Dict{Int, Any}(1 => exec), nothing, pinned, sum(kib) * _KiB, nothing)
    return ReactantServer.ModelEntry(
        name, _batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", FakeWeights(tensors), lm, nothing,
        identity, identity
    )
end

function _frag_scheduler(b::FragBackend, entries...; budget::Int = b.limit, min_residency::Real = 0.0)
    pool = ReactantServer.MemoryPool(b, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched = ReactantServer.Scheduler(reg, b, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    cache = ReactantServer.WeightCache(b, pool, reg, budget; min_residency_seconds = min_residency)
    sched.weight_cache = cache
    ReactantServer.preload_pinned!(cache, reg)
    # The startup probe's per-model scratch: the run's transient plus its input and output buffers,
    # which the real probe's peak measurement includes.
    for e in entries
        ReactantServer.model_footprint(cache, e).scratch_bytes = b.scratch[e.executable.execs[Int[]][1]] + 1024
    end
    return sched
end

_frag_load!(sched, name) = ReactantServer.acquire!(sched.weight_cache, sched.registry.by_name[name])
_frag_run(sched, name) = ReactantServer.run_model(
    sched.backend, sched.pool, sched.registry.by_name[name].executable,
    [ReactantServer.NamedTensor("x", reshape(Float32[1, 2], 2, 1))]
)
_resident(sched, name) = sched.registry.by_name[name].executable.weights !== nothing

# Pin each model's reload cost so priorities depend only on use, not on timing noise.
function _set_reload!(sched, name, seconds)
    fp = sched.weight_cache.footprints[name]
    fp.load_seconds = seconds
    fp.priority = sched.weight_cache.inflation + ReactantServer._priority_gain(sched.weight_cache, fp)
    return nothing
end

# ── Victim search ───────────────────────────────────────────────────────────────────────────────

function _fill!(s, bytes, cost; protected = falses(length(bytes)))
    ReactantServer.reset_candidates!(s, length(bytes))
    for i in eachindex(bytes)
        ReactantServer.set_candidate!(s, i, "m$i", bytes[i], cost[i], protected[i])
    end
    return s
end
_chosen(s) = [i for i in 1:s.n if s.best[i]]

@testset "victim search: optimal on small instances (brute force)" begin
    rng = MersenneTwister(7)
    s = ReactantServer.VictimSearch()
    for _ in 1:300
        n = rand(rng, 1:12)
        bytes = rand(rng, 1:1000, n)
        cost = rand(rng, n) .* rand(rng, [0.0, 1.0, 10.0], n)
        deficit = rand(rng, 1:sum(bytes))
        @test ReactantServer.select_cover!(_fill!(s, bytes, cost), deficit)
        got = _chosen(s)
        @test sum(bytes[got]; init = 0) >= deficit
        opt = minimum(
            sum(cost[i] for i in 1:n if (mask >> (i - 1)) & 1 == 1; init = 0.0)
                for mask in 0:((1 << n) - 1) if sum(bytes[i] for i in 1:n if (mask >> (i - 1)) & 1 == 1; init = 0) >= deficit
        )
        @test sum(cost[got]; init = 0.0) ≈ opt atol = 1.0e-9
    end
end

@testset "victim search: set choice beats sequential eviction; ties and protection" begin
    s = ReactantServer.VictimSearch()
    # Two medium models are cheap per byte but together cost more than one large cheap model.
    @test ReactantServer.select_cover!(_fill!(s, [60, 60, 100], [3.0, 3.0, 5.0]), 100)
    @test _chosen(s) == [3]
    # All costs zero (stale models): the least recently used (lowest index) goes first.
    @test ReactantServer.select_cover!(_fill!(s, [10, 10, 10], [0.0, 0.0, 0.0]), 10)
    @test _chosen(s) == [1]
    @test ReactantServer.select_cheapest!(_fill!(s, [10, 10], [0.0, 0.0])) == 1
    # A protected model is taken only when the others cannot cover the deficit.
    _fill!(s, [50, 50, 50], [0.0, 1.0, 1.0]; protected = [true, false, false])
    @test ReactantServer.select_cover!(s, 50) && _chosen(s) == [2]
    @test !ReactantServer.select_cover!(s, 120)
    @test ReactantServer.select_cover!(s, 120; allow_protected = true) && sum(s.bytes[_chosen(s)]) >= 120
    @test ReactantServer.select_cheapest!(_fill!(s, [5], [0.0]; protected = [true])) == 0
    # Not coverable at all.
    @test !ReactantServer.select_cover!(_fill!(s, [10, 10], [1.0, 1.0]), 25)
end

@testset "victim search: allocation-free and inside the latency budget" begin
    rng = MersenneTwister(11)
    s = ReactantServer.VictimSearch()
    n = 64
    bytes = rand(rng, (10^6):(10^9), n)
    for cost in (rand(rng, n), bytes .* (1 .+ 1.0e-3 .* rand(rng, n)) ./ 1.0e9)   # random, and equal density
        _fill!(s, bytes, cost)
        deficit = sum(bytes) ÷ 3
        ReactantServer.select_cover!(s, deficit)                 # compile
        ReactantServer.select_cheapest!(s)
        @test (@allocated ReactantServer.select_cover!(s, deficit)) == 0
        @test (@allocated ReactantServer.select_cheapest!(s)) == 0
        @test (@allocated ReactantServer.reset_candidates!(s, n)) == 0
        _fill!(s, bytes, cost)
        times = [(@elapsed ReactantServer.select_cover!(s, rand(rng, 1:sum(bytes)))) for _ in 1:500]
        sort!(times)
        @test times[250] < 100.0e-6                              # median well under 100 us
        @test times[495] < 1.0e-3                                # p99 inside the 1 ms budget
    end
end

# ── Admission ───────────────────────────────────────────────────────────────────────────────────

@testset "admission: evicts to make contiguous room for the largest tensor, no compaction" begin
    b = FragBackend(16 * _KiB)
    sched = _frag_scheduler(
        b, _frag_entry(b, "A"; kib = [3]), _frag_entry(b, "B"; kib = [3]), _frag_entry(b, "C"; kib = [3]),
        _frag_entry(b, "D"; kib = [3]), _frag_entry(b, "E"; kib = [6])
    )
    for m in ("A", "B", "C", "D")
        _frag_load!(sched, m)                                    # A..D at 0, 3, 6, 9 KiB; 4 KiB free at the top
        _set_reload!(sched, m, 1.0)                              # equal costs: recency decides
    end
    # E's one 6 KiB tensor fits by bytes after one eviction but needs a contiguous 6 KiB block.
    _frag_load!(sched, "E")
    @test _resident(sched, "E")
    @test sched.weight_cache.compactions == 0
    @test b.refusals == 0                                        # decided from statistics, never by trying
    @test !_resident(sched, "A") && !_resident(sched, "B")       # the two least recently used, adjacent
    @test _resident(sched, "C") && _resident(sched, "D")
    for m in ("C", "D", "E")
        @test _frag_run(sched, m)[1].data == reshape(Float32[1, 2], 2, 1)
    end
end

@testset "admission: keeps a frequently used model over a merely recent one" begin
    b = FragBackend(64 * _KiB)
    sched = _frag_scheduler(
        b, _frag_entry(b, "hot"; kib = [8]), _frag_entry(b, "B"; kib = [8]), _frag_entry(b, "C"; kib = [8]);
        budget = 16 * _KiB                                       # room for two models
    )
    _frag_load!(sched, "hot"); _set_reload!(sched, "hot", 1.0)
    for _ in 1:20
        _frag_load!(sched, "hot")                                # 20 more dispatches while resident
    end
    _frag_load!(sched, "B"); _set_reload!(sched, "B", 1.0)
    # LRU would now evict "hot" (touched before B); the cost-aware policy evicts B.
    _frag_load!(sched, "C")
    @test _resident(sched, "hot") && _resident(sched, "C") && !_resident(sched, "B")
end

@testset "admission: a model with queued requests is evicted only as a last resort" begin
    b = FragBackend(64 * _KiB)
    sched = _frag_scheduler(
        b, _frag_entry(b, "A"; kib = [8]), _frag_entry(b, "B"; kib = [8]), _frag_entry(b, "C"; kib = [8]);
        budget = 16 * _KiB
    )
    _frag_load!(sched, "A"); _frag_load!(sched, "B")
    wreq = ReactantServer.InferRequest("A", ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))])
    push!(sched.registry.by_name["A"].sched.queue, ReactantServer.QueuedRequest(wreq, wreq.inputs, 0.0, Channel{Any}(1)))
    _frag_load!(sched, "C")                                      # A is LRU but has a queued request
    @test _resident(sched, "A") && !_resident(sched, "B")
end

@testset "admission: minimum residency protects a just-loaded model" begin
    b = FragBackend(64 * _KiB)
    sched = _frag_scheduler(
        b, _frag_entry(b, "A"; kib = [8]), _frag_entry(b, "B"; kib = [8]), _frag_entry(b, "C"; kib = [8]);
        budget = 16 * _KiB, min_residency = 3600.0
    )
    _frag_load!(sched, "A"); _frag_load!(sched, "B")
    sched.weight_cache.footprints["A"].loaded_at -= 7200.0       # A is old enough to evict
    for _ in 1:50
        _frag_load!(sched, "B")                                  # B is costlier to lose, but young
    end
    sched.weight_cache.footprints["A"].hits = 1000; _set_reload!(sched, "A", 1.0)
    _frag_load!(sched, "C")
    @test !_resident(sched, "A")                                 # evicted despite its priority: B is protected
    @test _resident(sched, "B")
end

@testset "admission: infeasible model is rejected without evicting anything" begin
    b = FragBackend(16 * _KiB)
    sched = _frag_scheduler(b, _frag_entry(b, "P"; kib = [10], pinned = true), _frag_entry(b, "A"; kib = [2]), _frag_entry(b, "big"; kib = [8]))
    _frag_load!(sched, "A")
    @test_throws ReactantServer.AdmissionError _frag_load!(sched, "big")
    @test _resident(sched, "A")
    @test sched.weight_cache.evicts == 0
    @test sched.weight_cache.admission_failures == 1
end

@testset "admission: a failed upload frees its partial transfer and falls back to compaction" begin
    b = FragBackend(32 * _KiB)
    sched = _frag_scheduler(b, _frag_entry(b, "A"; kib = [4, 4]), _frag_entry(b, "B"; kib = [4, 4]))
    _frag_load!(sched, "A")
    b.fail_next_upload = true
    _frag_load!(sched, "B")
    @test _resident(sched, "B") && _resident(sched, "A")         # A, worth keeping, reloaded with B
    @test sched.weight_cache.fallback_compactions == 1
    @test isempty(setdiff(Set(keys(b.live)), Set(vcat(sched.registry.by_name["A"].executable.weights, sched.registry.by_name["B"].executable.weights))))
end

@testset "admission: compaction cannot beat a pinned model splitting the device" begin
    b = FragBackend(16 * _KiB)
    # P (pinned) sits at 4..8 KiB, so no free block can exceed 8 KiB whatever is evicted.
    sched = _frag_scheduler(
        b, _frag_entry(b, "pad"; kib = [4]), _frag_entry(b, "P"; kib = [4]), _frag_entry(b, "big"; kib = [9])
    )
    _frag_load!(sched, "pad")
    sched.registry.by_name["P"].executable.state = ReactantServer.PINNED_DEVICE
    ReactantServer.preload_pinned!(sched.weight_cache, sched.registry)
    @test_throws ReactantServer.AdmissionError _frag_load!(sched, "big")
    @test sched.weight_cache.fallback_compactions == 1
    @test sched.weight_cache.admission_failures == 1
    @test !_resident(sched, "big")
end

@testset "admission: random churn never strands a resident model without scratch" begin
    rng = MersenneTwister(3)
    b = FragBackend(64 * _KiB)
    entries = [
        _frag_entry(b, "m$i"; kib = rand(rng, 1:6, rand(rng, 1:4)), scratch = rand(rng, 0:(6 * _KiB)))
            for i in 1:10
    ]
    sched = _frag_scheduler(b, entries...)
    cache = sched.weight_cache
    for step in 1:400
        name = "m$(rand(rng, 1:10))"
        _frag_load!(sched, name)
        # Every resident model can still run: its scratch block fits right now.
        for e in entries
            _resident(sched, e.name) && @test !isempty(_frag_run(sched, e.name))
        end
    end
    @test b.refusals == 0                                        # admission never discovered a misfit by trying
    @test cache.admission_failures == 0
    @test cache.loads > 10 && cache.evicts > 0
end
