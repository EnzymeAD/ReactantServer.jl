# The per-bundle serialized-executable cache (runtime/executable_cache.jl) and the resettable
# memory probe (scheduler.jl), exercised without a GPU: the cache's layout, hash record,
# invalidation and atomic writes are pure filesystem logic, the probe is driven against a mock
# backend with a simulated allocator, and the real round trip (compile, serialize, store, load, run)
# runs on the CPU PJRT client through Reactant's own bindings when the installed Reactant has them.

using ReactantServer: ExecutableCacheSlot, module_filename, target_slug, entry_path,
    sync_mlir_hashes!, lookup_entry, store_entry, drop_entry, executable_cache_slots,
    exec_cache_dir, sha256hex, EXEC_CACHE_EXT, EXEC_CACHE_HASHES_FILE,
    record_exec_cache!, exec_cache_snapshot, reset_exec_cache_stats!,
    bundle_signature, _is_bundle_file, load_bundle_entry,
    supports_executable_cache, clear_memory_stats!, compiled_memory_stats,
    _probe_max_scratch!, _probe_entry_scratch!, _reprobe_after_load!, weight_budget

@testset "executable cache: naming" begin
    m = _batched_manifest("nm")
    @test module_filename(m, ReactantServer.VariantKey(), 0) == "model.mlir"
    @test module_filename(m, ReactantServer.VariantKey(), 8) == "model.b8.mlir"
    @test target_slug("NVIDIA RTX A6000") == "nvidia-rtx-a6000"
    @test target_slug("cuda 13010") == "cuda-13010"
    slot = ExecutableCacheSlot("/tmp/b/.cache", "model.b1.mlir", "a"^64)
    p = entry_path(slot, "jll-0.0.407+0_cuda-13010_nvidia-rtx-a6000_sm86", "f"^64)
    @test p == joinpath("/tmp/b/.cache", "exec", "jll-0.0.407+0_cuda-13010_nvidia-rtx-a6000_sm86", "model.b1.mlir." * "a"^16 * "." * "f"^16 * EXEC_CACHE_EXT)
    @test sha256hex(UInt8[]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
end

@testset "executable cache: hash record and MLIR-content invalidation" begin
    mktempdir() do dir
        cache = exec_cache_dir(dir)
        target = "jll-x_cuda_dev_sm86"
        # Two sources with cached programs, plus a second target partition for one of them.
        srcs = Dict("model.b1.mlir" => "1"^64, "model.b8.mlir" => "2"^64)
        @test isempty(sync_mlir_hashes!(cache, srcs))
        @test isfile(joinpath(cache, EXEC_CACHE_HASHES_FILE))
        e1 = entry_path(ExecutableCacheSlot(cache, "model.b1.mlir", srcs["model.b1.mlir"]), target, "k"^64)
        e8 = entry_path(ExecutableCacheSlot(cache, "model.b8.mlir", srcs["model.b8.mlir"]), target, "k"^64)
        e8b = entry_path(ExecutableCacheSlot(cache, "model.b8.mlir", srcs["model.b8.mlir"]), "other-target", "k"^64)
        for p in (e1, e8, e8b)
            @test store_entry(p, UInt8[1, 2, 3])
            @test lookup_entry(p) == UInt8[1, 2, 3]
        end
        # No temp files are left behind by the atomic write.
        @test all(!occursin(".tmp.", f) for (_, _, files) in walkdir(cache) for f in files)

        # Unchanged sources: nothing invalidated, entries intact.
        @test isempty(sync_mlir_hashes!(cache, srcs))
        @test isfile(e1) && isfile(e8) && isfile(e8b)

        # The b8 module changed content: only its programs go, under every target; b1 survives.
        srcs2 = Dict("model.b1.mlir" => "1"^64, "model.b8.mlir" => "3"^64)
        @test sync_mlir_hashes!(cache, srcs2) == ["model.b8.mlir"]
        @test isfile(e1)
        @test !isfile(e8) && !isfile(e8b)

        # A source that disappears from the bundle is invalidated too.
        @test sync_mlir_hashes!(cache, Dict("model.b1.mlir" => "1"^64)) == ["model.b8.mlir"]
        @test isfile(e1)

        # A corrupt record is treated as "nothing known": no sweep, record rewritten.
        write(joinpath(cache, EXEC_CACHE_HASHES_FILE), "not json")
        @test isempty(sync_mlir_hashes!(cache, Dict("model.b1.mlir" => "1"^64)))
        @test isfile(e1)

        # Lookups fail open; drop removes.
        @test lookup_entry(joinpath(cache, "nope" * EXEC_CACHE_EXT)) === nothing
        drop_entry(e1)
        @test !isfile(e1)
    end
end

@testset "executable cache: slots from a bundle, and the watcher ignores .cache" begin
    mktempdir() do root
        dir = _write_scale_bundle(root, "cached")
        entry = load_bundle_entry(dir)
        slots = executable_cache_slots(entry)
        @test slots !== nothing
        @test collect(keys(slots)) == [(ReactantServer.VariantKey(), 0)]
        slot = slots[(ReactantServer.VariantKey(), 0)]
        @test slot.dir == joinpath(dir, ".cache")
        @test slot.source == "model.mlir"
        @test slot.source_sha == sha256hex(read(joinpath(dir, "model.mlir")))
        @test isfile(joinpath(dir, ".cache", EXEC_CACHE_HASHES_FILE))

        # Everything the server writes under .cache/ is invisible to the watcher's fingerprint, so a
        # cache write never reloads the model that produced it in dynamic mode.
        sig = bundle_signature(dir)
        @test !_is_bundle_file(".cache")
        store_entry(entry_path(slot, "t", "k"^64), rand(UInt8, 64))
        touch(joinpath(dir, ".cache", "scratch.txt"))
        @test bundle_signature(dir) == sig
        # Weights-only updates are not part of the hash record either.
        @test !haskey(ReactantServer._read_hashes(slot.dir), "weights.safetensors")
        # Multi-shape variant modules are bundle files.
        @test _is_bundle_file("model.v0.b4.mlir") && _is_bundle_file("model.v1.mlir")
        @test !_is_bundle_file("model.b1.mlir.tmp")

        # An entry with no bundle directory (hand-built) yields no slots.
        @test executable_cache_slots(_od_entry("nodir"; pinned = false, nbytes = 8)) === nothing
    end
end

@testset "executable cache: counters and backend defaults" begin
    reset_exec_cache_stats!()
    record_exec_cache!(:hit, 0.5)
    record_exec_cache!(:miss, 2.0)
    record_exec_cache!(:store, 0.0)
    record_exec_cache!(:failure, 0.0)
    s = exec_cache_snapshot()
    @test (s.hits, s.misses, s.stores, s.failures) == (1, 1, 1, 1)
    @test s.load_seconds == 0.5 && s.compile_seconds == 2.0
    reset_exec_cache_stats!()
    @test exec_cache_snapshot().hits == 0

    mock = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(mock, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    @test !supports_executable_cache(mock)
    @test clear_memory_stats!(mock, pool) === false
    @test compiled_memory_stats(mock, ReactantServer.MockExecutable(x -> x, 1)) === nothing
    # The Reactant backend reports the cache exactly when its Reactant exposes the serialization API.
    @test supports_executable_cache(ReactantServer.ReactantBackend()) ==
        (isdefined(Reactant.XLA, :serialize_executable) && isdefined(Reactant.XLA, :load_serialized_executable))
end

# ── The real thing: compile, serialize, store, load, run, through Reactant's bindings only ──────

@testset "executable cache: Reactant backend round trip (CPU)" begin
    backend = ReactantServer.ReactantBackend()
    if !supports_executable_cache(backend)
        @warn "the installed Reactant has no executable serialization (Reactant.XLA.serialize_executable); the round-trip test is skipped"
        @test_skip supports_executable_cache(backend)
    else
        cfg = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        pool = ReactantServer.resolve_client(backend, cfg)
        mktempdir() do root
            dir = _write_scale_bundle(root, "cached"; w = Float32[3, 3, 3, 3])
            x = ReactantServer.NamedTensor("x", Float32[1, 2, 3, 4])
            expected = Float32[3, 6, 9, 12]
            cached_programs() = String[
                joinpath(d, f) for (d, _, files) in walkdir(joinpath(dir, ".cache")) for f in files if endswith(f, EXEC_CACHE_EXT)
            ]
            load(; cache) = (
                e = load_bundle_entry(dir);
                e.executable = ReactantServer.build_loaded_model(backend, pool, e; executable_cache = cache);
                e
            )
            run(e) = ReactantServer.run_model(backend, pool, e.executable, [x])[1].data

            reset_exec_cache_stats!()
            # Cold start: no entry, so compile and store.
            e1 = load(; cache = true)
            s = exec_cache_snapshot()
            @test (s.hits, s.misses, s.stores, s.failures) == (0, 1, 1, 0)
            files = cached_programs()
            @test length(files) == 1
            @test startswith(basename(dirname(first(files))), "jll-")          # the target partition
            @test startswith(basename(first(files)), "model.mlir.")            # named after its source
            @test run(e1) == expected

            # Warm start: the stored program is loaded, not compiled, and computes the same result.
            e2 = load(; cache = true)
            s = exec_cache_snapshot()
            @test (s.hits, s.misses, s.stores, s.failures) == (1, 1, 1, 0)
            @test run(e2) == expected

            # The compiler's static accounting is available on the loaded program too; the CPU
            # allocator keeps no statistics, so the reset reports false and callers fall back.
            exec = first(values(e2.executable.execs[ReactantServer.VariantKey()]))
            cms = compiled_memory_stats(backend, exec)
            @test cms !== nothing && cms.temp >= 0 && cms.outputs >= 0
            @test clear_memory_stats!(backend, pool) === false

            # A corrupt entry is dropped, then the program is compiled and stored again.
            write(first(files), rand(UInt8, 32))
            e3 = load(; cache = true)
            s = exec_cache_snapshot()
            @test (s.hits, s.misses, s.stores, s.failures) == (1, 2, 2, 1)
            @test length(cached_programs()) == 1
            @test run(e3) == expected

            # A changed weight keeps the program (weights are not part of the key) and hits.
            _write_scale_bundle(root, "cached"; w = Float32[5, 5, 5, 5])
            e4 = load(; cache = true)
            @test exec_cache_snapshot().hits == 2
            @test run(e4) == Float32[5, 10, 15, 20]

            # With the cache off nothing is read or written.
            before = exec_cache_snapshot()
            e5 = load(; cache = false)
            @test exec_cache_snapshot() == before
            @test run(e5) == Float32[5, 10, 15, 20]
        end
    end
end

# ── The resettable memory probe ──────────────────────────────────────────────────────────────────
#
# A MockBackend twin with a simulated BFC allocator: uploads and outputs count toward in_use, an
# execution transiently claims `scratch` bytes, and the peak is either resettable or monotone.
mutable struct StatsBackend <: ReactantServer.AbstractBackend
    in_use::Int
    peak::Int
    scratch::Int
    resettable::Bool
    static_temp::Int        # what compiled_memory_stats reports as temp (+ outputs = 0)
end
StatsBackend(; scratch, resettable, static_temp = 0) = StatsBackend(0, 0, scratch, resettable, static_temp)

_sb_bump!(b::StatsBackend, n) = (b.in_use += n; b.peak = max(b.peak, b.in_use); nothing)

ReactantServer.make_client(::StatsBackend, platform::String; kwargs...) = ReactantServer.MockClient()
ReactantServer.select_device(::StatsBackend, ::ReactantServer.MockClient, ordinal::Int) = ReactantServer.MockDevice(ordinal)
ReactantServer.device_ordinal(::StatsBackend, d::ReactantServer.MockDevice) = d.ordinal
function ReactantServer.to_device(b::StatsBackend, ::ReactantServer.MockClient, a::Array, ::ReactantServer.MockDevice)
    _sb_bump!(b, sizeof(a))
    return ReactantServer.MockBuffer(copy(a))
end
ReactantServer.buffer_eltype(::StatsBackend, m::ReactantServer.MockBuffer) = eltype(m.data)
ReactantServer.buffer_size(::StatsBackend, m::ReactantServer.MockBuffer) = reverse(size(m.data))
ReactantServer.to_host!(::StatsBackend, m::ReactantServer.MockBuffer, dest::Array) = (copyto!(dest, m.data); dest)
function ReactantServer.free_buffer!(b::StatsBackend, m::ReactantServer.MockBuffer)
    m.freed || (b.in_use -= sizeof(m.data))
    m.freed = true
    return nothing
end
ReactantServer.free_executable!(::StatsBackend, e::ReactantServer.MockExecutable) = (e.freed = true; nothing)
function ReactantServer.execute_single_device(
        b::StatsBackend, exec::ReactantServer.MockExecutable, ::ReactantServer.MockDevice,
        buffers::AbstractVector, donated::AbstractVector{Bool}, num_outputs::Int
    )
    outs = exec.fn([m.data for m in buffers])
    _sb_bump!(b, b.scratch)                        # transient scratch: claimed during the run ...
    b.in_use -= b.scratch                          # ... and released before the outputs are handed back
    return ReactantServer.MockBuffer[(_sb_bump!(b, sizeof(o)); ReactantServer.MockBuffer(o)) for o in outs]
end
ReactantServer.device_memory_stats(b::StatsBackend, pool) =
    (in_use = b.in_use, limit = 1_000_000, free = 1_000_000 - b.in_use, peak_in_use = b.peak, pool_bytes = 0, peak_pool_bytes = 0)
ReactantServer.clear_memory_stats!(b::StatsBackend, pool) = (b.resettable && (b.peak = b.in_use); b.resettable)
ReactantServer.compiled_memory_stats(b::StatsBackend, ::ReactantServer.MockExecutable) =
    b.static_temp == 0 ? nothing : (temp = b.static_temp, outputs = 0)

function _stats_scheduler(backend::StatsBackend, entries::ReactantServer.ModelEntry...; budget::Int = 1_000_000)
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, budget)
    ReactantServer.preload_pinned!(sched.weight_cache, reg)
    return sched
end

@testset "memory probe: resettable peak measures run scratch, not compile-time pollution" begin
    # Two on-demand models plus a device-pinned one; the run scratch is what the probe must find.
    scratch = 5_000
    b = StatsBackend(; scratch = scratch, resettable = true)
    sched = _stats_scheduler(
        b, _od_entry("light"; pinned = false, nbytes = 8), _od_entry("heavy"; pinned = false, nbytes = 8),
        _od_entry("pinned"; pinned = true, nbytes = _WBYTES)
    )
    # Simulate autotuning during compile: a huge transient that already set the session peak.
    b.peak = 900_000
    got = _probe_max_scratch!(sched, ReactantServer.pinned_weight_bytes(sched.registry))
    # Exactly the run transient: scratch plus the output buffer (8 bytes) that is live at the peak.
    @test got == scratch + 8
    # Pinned models are probed too under the resettable regime (their cost estimate got seeded).
    @test haskey(sched.registry.by_name["pinned"].sched.cost_estimate, 1)
end

@testset "memory probe: the monotone-peak fallback stays ordering-based and inherits pollution" begin
    scratch = 5_000
    b = StatsBackend(; scratch = scratch, resettable = false)
    sched = _stats_scheduler(b, _od_entry("light"; pinned = false, nbytes = 8), _od_entry("heavy"; pinned = false, nbytes = 8))
    # Clean session: the ordering trick recovers the true scratch (peak - pinned - maxweight).
    got = _probe_max_scratch!(sched, 0)
    @test got == scratch + 8           # scratch plus the 8-byte input upload live at the peak
    # Polluted session: the fallback cannot see past the inflated peak; documents why reset matters.
    b.peak = 900_000
    @test _probe_max_scratch!(sched, 0) >= 900_000 - 8
end

@testset "memory probe: static compiled stats stand in when a model cannot be run" begin
    b = StatsBackend(; scratch = 5_000, resettable = true, static_temp = 40_000)
    boom = _od_entry("boom"; pinned = false, nbytes = 8)
    boom.executable.execs[ReactantServer.VariantKey()][1] = ReactantServer.MockExecutable(_ -> error("no run"), 1)
    sched = _stats_scheduler(b, boom)
    ReactantServer.acquire!(sched.weight_cache, boom)
    @test _probe_entry_scratch!(sched, boom, true, 0) == ceil(Int, 1.5 * 40_000)
end

@testset "memory probe: hot-loaded model re-resolves the weight budget without a restart" begin
    b = StatsBackend(; scratch = 3_000, resettable = true)
    a = _od_entry("a"; pinned = false, nbytes = 8)
    sched = _stats_scheduler(b, a)
    arena, fraction, wiggle = 100_000, 1.0, 0.1
    ReactantServer._autosize_weight_cache!(sched, arena, fraction, wiggle)
    base_scratch = sched.weight_cache.max_scratch
    @test base_scratch == 3_000 + 8
    @test sched.autosize == (arena = arena, fraction = fraction, wiggle = wiggle)
    # A heavier-scratch model arrives via the watcher path: the budget shrinks accordingly.
    b.scratch = 20_000
    b.peak = 800_000                                   # its compile polluted the peak; irrelevant
    n = _od_entry("n"; pinned = false, nbytes = 8)
    sched.registry.by_name["n"] = n
    n.sched = ReactantServer.ModelSchedState("n", ReactantServer.ModelSchedConfig(1.0), 0.0)
    _reprobe_after_load!(sched, n)
    @test sched.weight_cache.max_scratch == 20_000 + 8
    expect = weight_budget(; arena, fraction, wiggle, max_scratch = 20_000 + 8, pinned_bytes = 0)
    @test sched.weight_cache.max_bytes == expect.on_demand_budget
    @test sched.weight_cache.weight_pool == expect.weight_pool
    # A lighter model later never lowers the ceiling.
    b.scratch = 100
    _reprobe_after_load!(sched, a)
    @test sched.weight_cache.max_scratch == 20_000 + 8
end
