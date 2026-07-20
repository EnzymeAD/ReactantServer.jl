# On-demand weight residency against the Reactant-free MockBackend.
#
# The mock backend already models load/free: to_device copies an array into a MockBuffer and
# free_buffer! flips MockBuffer.freed. A tiny FakeWeights stands in for the safetensors handle
# (load_pinned_weights and weights_nbytes only need haskey/getindex over real arrays). Tests
# drive select_dispatch! + execute_and_record! directly (no spawned loop) so load/evict
# decisions are deterministic.

# Minimal safetensors double: name -> host array.
struct FakeWeights
    tensors::Dict{String,Array}
end
Base.haskey(f::FakeWeights, k) = haskey(f.tensors, k)
Base.getindex(f::FakeWeights, k) = f.tensors[k]

const _W = Float32[10, 100]                      # the model's single weight tensor (y = x .* w)
const _WBYTES = length(_W) * sizeof(Float32)     # 8 bytes

# A safetensors double that errors on any access, to prove the mmap is never touched once
# weights are pinned in host RAM.
struct PoisonWeights end
Base.haskey(::PoisonWeights, k) = error("safetensors mmap accessed; weights should be RAM-pinned")
Base.getindex(::PoisonWeights, k) = error("safetensors mmap accessed; weights should be RAM-pinned")

# A batched scale model whose weights may start resident (pinned) or evicted (on-demand). When
# host_pinned, the materialized host array is kept resident and the mmap handle is poisoned so a
# load that reaches for the mmap would fail the test.
function _od_entry(name; pinned::Bool, nbytes::Int, host_pinned::Bool=false)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    weights = pinned ? Any[ReactantServer.MockBuffer(copy(_W))] : nothing
    host_weights = host_pinned ? Any[copy(_W)] : nothing
    lm = ReactantServer.LoadedModel(sig, Dict{Int,Any}(1 => exec), weights, pinned, nbytes, host_weights)
    st = host_pinned ? PoisonWeights() : FakeWeights(Dict{String,Array}("w" => copy(_W)))
    return ReactantServer.ModelEntry(name, _batched_manifest(name), Dict{Int,Vector{UInt8}}(),
        "", st, lm, nothing, identity, identity)
end

function _od_scheduler(entries::ReactantServer.ModelEntry...; budget::Int,
                      mode::ReactantServer.ResidencyMode=ReactantServer.SELF_MANAGED,
                      store::ReactantServer.WeightStore=ReactantServer.PrivateWeightStore())
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
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, budget; mode=mode, store=store)
    ReactantServer.preload_pinned!(sched.weight_cache, reg)
    return sched
end

# Dispatch a single one-row request to `name` and return the per-caller result.
function _dispatch1!(sched, name; val::Float32=1.0f0)
    wreq = ReactantServer.InferRequest(name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[val, val], 2, 1))])
    qr = ReactantServer.QueuedRequest(wreq, wreq.inputs, 0.0, Channel{Any}(1))
    push!(sched.registry.by_name[name].sched.queue, qr)
    d = ReactantServer.select_dispatch!(sched, 0.0)
    ReactantServer.execute_and_record!(sched, d)
    return take!(qr.reply)
end

@testset "on-demand weights load on first dispatch and survive reuse" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES)
    A = sched.registry.by_name["A"].executable
    @test A.weights === nothing                      # evicted at startup

    res = _dispatch1!(sched, "A"; val=1.0f0)
    @test res[1].data == reshape(Float32[10, 100], 2, 1)
    @test A.weights !== nothing                      # loaded on demand
    @test sched.weight_cache.loads == 1
    @test sched.weight_cache.resident_bytes == _WBYTES

    # A second dispatch to the same model reuses the resident weights (no reload). This is what
    # keeps back-to-back requests coalescible.
    _dispatch1!(sched, "A"; val=2.0f0)
    @test sched.weight_cache.loads == 1
    @test sched.weight_cache.evicts == 0
end

@testset "LRU eviction under budget pressure" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES),
                          _od_entry("B"; pinned=false, nbytes=_WBYTES); budget=_WBYTES)  # holds one model
    _dispatch1!(sched, "A")
    a_bufs = sched.registry.by_name["A"].executable.weights
    @test a_bufs !== nothing

    _dispatch1!(sched, "B")                           # must evict A to make room for B
    @test sched.registry.by_name["A"].executable.weights === nothing
    @test all(b -> b.freed, a_bufs)                   # A's device buffers were released
    @test sched.registry.by_name["B"].executable.weights !== nothing
    @test sched.weight_cache.evicts == 1
    @test sched.weight_cache.loads == 2
    @test sched.weight_cache.resident_bytes == _WBYTES   # stays within budget
end

@testset "pinned weights are preloaded and never evicted" begin
    sched = _od_scheduler(_od_entry("P"; pinned=true, nbytes=_WBYTES),
                          _od_entry("Q"; pinned=false, nbytes=_WBYTES); budget=_WBYTES)
    p_bufs = sched.registry.by_name["P"].executable.weights
    @test p_bufs !== nothing                          # pinned: resident from the start

    _dispatch1!(sched, "Q")                           # loads Q on demand (pinned P is off-budget)
    res = _dispatch1!(sched, "P")                     # pinned: acquire! is a no-op
    @test res[1].data == reshape(Float32[10, 100], 2, 1)
    @test sched.registry.by_name["P"].executable.weights === p_bufs   # untouched
    @test !any(b -> b.freed, p_bufs)
    @test sched.weight_cache.loads == 1               # only Q ever loaded on demand
    @test sched.weight_cache.evicts == 0
end

@testset "a model larger than the whole budget still loads" begin
    sched = _od_scheduler(_od_entry("Big"; pinned=false, nbytes=_WBYTES * 100); budget=_WBYTES)
    res = (@test_logs (:warn,) match_mode=:any _dispatch1!(sched, "Big"))
    @test res[1].data == reshape(Float32[10, 100], 2, 1)
    @test sched.registry.by_name["Big"].executable.weights !== nothing
    @test sched.weight_cache.loads == 1
end

@testset "on-demand load transfers from RAM-pinned host weights without touching the mmap" begin
    sched = _od_scheduler(_od_entry("H"; pinned=false, nbytes=_WBYTES, host_pinned=true); budget=_WBYTES)
    H = sched.registry.by_name["H"].executable
    @test H.weights === nothing          # not on the GPU yet
    @test H.host_weights !== nothing     # but resident in host RAM

    # The mmap handle is poisoned; a correct load must transfer from host_weights, not re-collect.
    res = _dispatch1!(sched, "H")
    @test res[1].data == reshape(Float32[10, 100], 2, 1)
    @test H.weights !== nothing
    @test sched.weight_cache.loads == 1
end

@testset "externally-managed: acquire! rejects a non-resident model, residency moves it" begin
    sched = _od_scheduler(_od_entry("E"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 10,
                          mode=ReactantServer.EXTERNALLY_MANAGED)
    entry = sched.registry.by_name["E"]
    @test entry.executable.weights === nothing
    @test entry.executable.state == ReactantServer.UNPINNED

    # No autonomous loading in externally-managed mode: a request for an evicted model is rejected.
    @test_throws ReactantServer.NotResidentError ReactantServer.acquire!(sched.weight_cache, entry)

    # The control plane pins it to the device; it is then resident, exempt, and serveable.
    @test ReactantServer.set_residency_state!(sched.weight_cache, entry, ReactantServer.PINNED_DEVICE) ==
          ReactantServer.PINNED_DEVICE
    @test entry.executable.weights !== nothing
    @test ReactantServer.is_device_pinned(entry.executable)
    @test !("E" in sched.weight_cache.lru)            # device-pinned is off-budget
    @test ReactantServer.acquire!(sched.weight_cache, entry) === nothing   # no-op now

    dev = entry.executable.weights
    # Demote to system: host floor materialized, device copy released (no evictor to reclaim it).
    @test ReactantServer.set_residency_state!(sched.weight_cache, entry, ReactantServer.PINNED_SYSTEM) ==
          ReactantServer.PINNED_SYSTEM
    @test entry.executable.host_weights !== nothing
    @test entry.executable.weights === nothing
    @test all(b -> b.freed, dev)

    # Unpin entirely: host floor dropped too.
    ReactantServer.set_residency_state!(sched.weight_cache, entry, ReactantServer.UNPINNED)
    @test entry.executable.host_weights === nothing
    @test entry.executable.state == ReactantServer.UNPINNED
end

@testset "control_status reports mode, discipline, and per-model residency" begin
    sched = _od_scheduler(_od_entry("P"; pinned=true, nbytes=_WBYTES),
                          _od_entry("U"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4,
                          mode=ReactantServer.EXTERNALLY_MANAGED)
    st = ReactantServer.control_status(sched)
    @test st.residency_mode == ReactantServer.EXTERNALLY_MANAGED
    @test st.discipline == ReactantServer.FAIR
    @test st.models["P"].state == ReactantServer.PINNED_DEVICE
    @test st.models["P"].device_resident == true
    @test st.models["U"].state == ReactantServer.UNPINNED
    @test st.models["U"].device_resident == false

    # set_policy! updates the live fair-discipline weight.
    ReactantServer.set_policy!(sched, "U"; weight=3.5)
    @test sched.registry.by_name["U"].sched.weight == 3.5
    @test_throws Exception ReactantServer.set_policy!(sched, "ghost"; weight=1.0)
end

@testset "ControlService handlers map residency/policy and drive the dispatch loop" begin
    sched = _od_scheduler(_od_entry("P"; pinned=true, nbytes=_WBYTES),
                          _od_entry("U"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4,
                          mode=ReactantServer.EXTERNALLY_MANAGED)
    ctx = ReactantServer.InferContext(sched, sched.registry, ReactantServer.SharedMemoryRegistry(), "mock")
    _Ctl = ReactantServer.control

    # Status snapshot reflects mode and per-model residency (no dispatch loop needed).
    status = ReactantServer._handle_model_control_status(ctx)
    @test status.residency_mode == "externally_managed"
    @test status.discipline == "fair"
    bymodel = Dict(m.name => m for m in status.models)
    @test bymodel["P"].residency == _Ctl.Residency.PINNED_DEVICE
    @test bymodel["P"].device_resident == true
    @test bymodel["U"].residency == _Ctl.Residency.UNPINNED

    # SetModelPolicy applies without the loop (it locks scheduler state directly).
    ReactantServer._handle_set_model_policy(ctx, _Ctl.SetModelPolicyRequest(; name="U", has_weight=true, weight=2.0))
    @test sched.registry.by_name["U"].sched.weight == 2.0

    # SetModelResidency routes through the control-command queue, so run the dispatch loop.
    ReactantServer.start!(sched)
    try
        resp = ReactantServer._handle_set_model_residency(ctx,
            _Ctl.SetModelResidencyRequest(; name="U", target=_Ctl.Residency.PINNED_DEVICE))
        @test resp.residency == _Ctl.Residency.PINNED_DEVICE
        @test ReactantServer.is_device_pinned(sched.registry.by_name["U"].executable)

        # An unspecified target is rejected as INVALID_ARGUMENT.
        @test_throws Exception ReactantServer._handle_set_model_residency(ctx,
            _Ctl.SetModelResidencyRequest(; name="U", target=_Ctl.Residency.RESIDENCY_UNSPECIFIED))
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "compaction frees on-demand weights and leaves pinned in place (eager)" begin
    sched = _od_scheduler(_od_entry("P"; pinned=true, nbytes=_WBYTES),
                          _od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4)
    _dispatch1!(sched, "A")                              # load A on demand
    A = sched.registry.by_name["A"].executable
    P = sched.registry.by_name["P"].executable
    a_bufs = A.weights; p_bufs = P.weights
    @test a_bufs !== nothing && p_bufs !== nothing

    n = ReactantServer.compact!(sched.weight_cache, sched.registry)   # eager: free non-pinned only
    @test n == 0                                         # nothing eagerly reloaded
    @test all(b -> b.freed, a_bufs)                      # on-demand A's device buffers freed
    @test A.weights === nothing                          # ...and left freed (reloads lazily)
    @test P.weights === p_bufs                           # pinned P untouched (not freed, not reloaded)
    @test !any(b -> b.freed, p_bufs)
    @test !("A" in sched.weight_cache.lru)
    @test sched.weight_cache.resident_bytes == 0         # on-demand region emptied; pinned is off-budget
    @test sched.weight_cache.compactions == 1

    res = _dispatch1!(sched, "A")                        # A reloads on its next dispatch
    @test res[1].data == reshape(Float32[10, 100], 2, 1)
    @test A.weights !== nothing
end

@testset "compaction reloads the requested non-pinned models eagerly" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES),
                          _od_entry("B"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4)
    _dispatch1!(sched, "A"); _dispatch1!(sched, "B")
    A = sched.registry.by_name["A"].executable
    B = sched.registry.by_name["B"].executable
    @test A.weights !== nothing && B.weights !== nothing

    n = ReactantServer.compact!(sched.weight_cache, sched.registry; reload=["A"])
    @test n == 1
    @test A.weights !== nothing                          # listed: reloaded eagerly
    @test B.weights === nothing                          # unlisted: left for lazy reload
    @test "A" in sched.weight_cache.lru
    @test sched.weight_cache.resident_bytes == _WBYTES
end

@testset "compaction is a no-op without the on-demand weight cache" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES)
    A = sched.registry.by_name["A"].executable
    A.weights = Any[ReactantServer.MockBuffer(copy(_W))]   # always-resident (no on-demand cache to churn)
    sched.weight_cache = nothing
    bufs = A.weights
    @test ReactantServer._compact_entry!(sched) == 0
    @test A.weights === bufs                              # untouched: nothing churns without the cache
    @test !any(b -> b.freed, bufs)
end

@testset "worker auto-compaction fires once on-demand loads advance by the interval" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES),
                          _od_entry("B"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 8)
    sched.cfg = ReactantServer.SchedulerConfig(30.0, 1024, 30.0; compaction_interval=2)

    _dispatch1!(sched, "A")                              # loads = 1
    ReactantServer._maybe_compact!(sched)
    @test sched.weight_cache.compactions == 0            # one load, below the interval
    @test sched.registry.by_name["A"].executable.weights !== nothing

    _dispatch1!(sched, "B")                              # loads = 2
    ReactantServer._maybe_compact!(sched)
    @test sched.weight_cache.compactions == 1            # loads advanced by the interval -> compacted (eager)
    @test sched.registry.by_name["A"].executable.weights === nothing
    @test sched.registry.by_name["B"].executable.weights === nothing
    @test sched.weight_cache.resident_bytes == 0
end

@testset "CompactMemory handler runs compaction through the dispatch loop" begin
    sched = _od_scheduler(_od_entry("P"; pinned=true, nbytes=_WBYTES),
                          _od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4)
    ctx = ReactantServer.InferContext(sched, sched.registry, ReactantServer.SharedMemoryRegistry(), "mock")
    _Ctl = ReactantServer.control
    ReactantServer.start!(sched)
    try
        ReactantServer.infer(sched, ReactantServer.InferRequest("A", ["y"],
            [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))]))
        @test sched.registry.by_name["A"].executable.weights !== nothing

        p_bufs = sched.registry.by_name["P"].executable.weights
        resp = ReactantServer._handle_compact_memory(ctx, _Ctl.CompactMemoryRequest())
        @test resp.reloaded_models == 0                  # eager: nothing eagerly reloaded
        @test sched.registry.by_name["A"].executable.weights === nothing   # on-demand A freed (lazy reload)
        @test sched.registry.by_name["P"].executable.weights === p_bufs    # pinned P untouched
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "weight_budget reserves pinned + scratch + wiggle from the arena" begin
    wb(; kw...) = ReactantServer.weight_budget(; kw...)
    # No pinned, no scratch: on-demand gets fraction*arena, capped by (1-wiggle)*arena.
    b = wb(arena=1000, fraction=1.0, wiggle=0.1, max_scratch=0, pinned_bytes=0)
    @test b.scratch_ceiling == 900 && b.weight_pool == 900 && b.on_demand_budget == 900
    # Fraction caps below the scratch ceiling.
    b = wb(arena=1000, fraction=0.5, wiggle=0.1, max_scratch=0, pinned_bytes=0)
    @test b.weight_pool == 500 && b.on_demand_budget == 500
    # Scratch eats into the pool.
    b = wb(arena=1000, fraction=1.0, wiggle=0.1, max_scratch=300, pinned_bytes=0)
    @test b.weight_pool == 600 && b.on_demand_budget == 600
    # Pinned reserves its portion off the top.
    b = wb(arena=1000, fraction=1.0, wiggle=0.1, max_scratch=300, pinned_bytes=200)
    @test b.weight_pool == 600 && b.on_demand_budget == 400 && !b.pinned_over_commit
    # Pinned over-commit: pinned exceed the pool -> on-demand 0, flagged.
    b = wb(arena=1000, fraction=1.0, wiggle=0.1, max_scratch=300, pinned_bytes=700)
    @test b.on_demand_budget == 0 && b.pinned_over_commit
    # Scratch alone exceeds the ceiling -> empty pool.
    b = wb(arena=1000, fraction=1.0, wiggle=0.1, max_scratch=950, pinned_bytes=0)
    @test b.weight_pool == 0 && b.on_demand_budget == 0
    # Arena 0 (CPU/mock): everything zero.
    b = wb(arena=0, fraction=1.0, wiggle=0.1, max_scratch=0, pinned_bytes=0)
    @test b.weight_pool == 0 && b.on_demand_budget == 0
end

@testset "shared weight store backs a system-pinned model and unlinks on unpin" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        store = ReactantServer.SharedWeightStore()
        sched = _od_scheduler(_od_entry("S"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 4,
                              mode=ReactantServer.EXTERNALLY_MANAGED, store=store)
        entry = sched.registry.by_name["S"]
        dg = ReactantServer.weights_digest("S", [(Float32, (2,))])
        region = "/dev/shm/rsw-S-" * string(dg; base=16)

        ReactantServer.set_residency_state!(sched.weight_cache, entry, ReactantServer.PINNED_SYSTEM)
        @test entry.executable.host_weights !== nothing
        @test entry.executable.host_weights[1] == _W       # populated from the (fake) safetensors
        @test isfile(region)                                # node-shared region created

        ReactantServer.set_residency_state!(sched.weight_cache, entry, ReactantServer.UNPINNED)
        @test entry.executable.host_weights === nothing
        @test !isfile(region)                               # sole holder unlinked it
    end
end

@testset "admit! and evict! add and remove a model at runtime" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 8)
    ReactantServer.start!(sched)
    req(name) = ReactantServer.InferRequest(name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))])
    try
        # A freshly compiled entry is admitted into the running system and immediately serveable.
        ReactantServer.admit!(sched, _od_entry("B"; pinned=false, nbytes=_WBYTES))
        @test haskey(sched.registry.by_name, "B")
        @test ReactantServer.infer(sched, req("B"))[1].data == reshape(Float32[10, 100], 2, 1)

        # Evicting removes it from the registry and frees its residency.
        bexec = sched.registry.by_name["B"].executable
        bufs = bexec.weights
        @test bufs !== nothing
        @test ReactantServer.evict!(sched, "B") !== nothing
        @test !haskey(sched.registry.by_name, "B")
        @test bexec.weights === nothing
        @test all(b -> b.freed, bufs)                       # device buffers released
        # The compiled executables are freed eagerly too (their command buffers live outside the
        # BFC arena and would otherwise linger until GC happens to run the finalizer).
        @test all(e -> e.freed, collect(values(bexec.execs[Int[]])))

        # A request to the unloaded model now errors; eviction is idempotent; re-admitting a live
        # name errors.
        @test_throws Exception ReactantServer.infer(sched, req("B"))
        @test ReactantServer.evict!(sched, "B") === nothing
        @test_throws Exception ReactantServer.admit!(sched, _od_entry("A"; pinned=false, nbytes=_WBYTES))
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "rename_model! renames in place: no reload, no recompile, residency rekeyed" begin
    sched = _od_scheduler(_od_entry("A"; pinned=false, nbytes=_WBYTES); budget=_WBYTES * 8)
    ReactantServer.start!(sched)
    req(name) = ReactantServer.InferRequest(name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))])
    try
        # Serve once so A's weights are LRU-resident under the old name.
        @test ReactantServer.infer(sched, req("A"))[1].data == reshape(Float32[10, 100], 2, 1)
        entry = sched.registry.by_name["A"]
        exec = entry.executable.execs[Int[]][1]
        @test "A" in sched.weight_cache.lru

        ReactantServer.rename_model!(sched, "A", "B")
        @test !haskey(sched.registry.by_name, "A")
        e2 = sched.registry.by_name["B"]
        @test e2 === entry                              # the same live entry: nothing was reloaded
        @test e2.executable.execs[Int[]][1] === exec    # the same executable: nothing was recompiled
        @test !exec.freed                               # and it was NOT freed (rename is not evict)
        @test e2.name == "B" && e2.sched.name == "B"
        @test !("A" in sched.weight_cache.lru) && ("B" in sched.weight_cache.lru)
        @test ReactantServer.infer(sched, req("B"))[1].data == reshape(Float32[10, 100], 2, 1)
        @test_throws Exception ReactantServer.infer(sched, req("A"))

        # The residency floor resolved for the new name is applied when it differs (the config is
        # keyed by model name, e.g. a `-production` name may be pinned where `-staging` was not).
        ReactantServer.rename_model!(sched, "B", "C"; residency=ReactantServer.PINNED_DEVICE)
        @test sched.registry.by_name["C"].executable.state == ReactantServer.PINNED_DEVICE
        @test !("C" in sched.weight_cache.lru)          # device-pinned models leave the LRU budget

        # An unknown old name and a colliding new name are rejected.
        @test_throws Exception ReactantServer.rename_model!(sched, "missing", "X")
        ReactantServer.admit!(sched, _od_entry("D"; pinned=false, nbytes=_WBYTES))
        @test_throws Exception ReactantServer.rename_model!(sched, "D", "C")
    finally
        ReactantServer.shutdown!(sched)
    end
end
