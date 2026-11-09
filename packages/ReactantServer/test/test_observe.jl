# Lifecycle observability helpers: shape/dtype formatting, compiled-size rendering, the
# resident-weight accounting, the multi-angle memory report, and the structured log records.
# All Reactant-free (MockBackend), so fast and deterministic.

using ReactantServer: TensorSpec, Dim, FIXED, BATCH, VARIABLE, F32, BF16, ModelSignature,
    LoadedModel, ModelEntry, ModelRegistry, Manifest, BatchingSpec, Provenance,
    MockBackend, MockClient, MockDevice, MemoryPool, WeightCache, UNPINNED, PINNED_SYSTEM,
    PINNED_DEVICE, _format_specs, _compiled_sizes, resident_weight_bytes, memory_report,
    log_model_loaded, log_model_unloaded, log_residency_change

_obs_sig() = ModelSignature(String[], DataType[], String[], 0, String[], DataType[], 0)

_obs_manifest(name) = Manifest(
    "2.0", name, "",
    TensorSpec[
        TensorSpec("x", F32, Dim[Dim(FIXED, 3), Dim(FIXED, 224), Dim(BATCH)], 3),
        TensorSpec("mask", BF16, Dim[Dim(VARIABLE), Dim(BATCH)], 2),
    ],
    TensorSpec[TensorSpec("y", F32, Dim[Dim(FIXED, 10), Dim(BATCH)], 2)],
    nothing, nothing, BatchingSpec(Int[]), Provenance(Dict{String, Any}()), nothing
)

_obs_model(sizes::Vector{Int}; weights, nbytes, state = UNPINNED) =
    LoadedModel(_obs_sig(), Dict{Int, Any}(s => nothing for s in sizes), weights, state, nbytes, nothing)

_obs_entry(name, model) = ModelEntry(
    name, _obs_manifest(name), Dict{Int, Vector{UInt8}}(),
    "", nothing, model, nothing, identity, identity
)

_obs_pool() = MemoryPool(MockBackend(), MockClient(), MockDevice(0), "mock", nothing)

@testset "observe: shape and dtype formatting" begin
    m = _obs_manifest("m")
    @test _format_specs(m.executable_inputs) == "x: f32[3,224,n], mask: bf16[?,n]"
    @test _format_specs(m.executable_outputs) == "y: f32[10,n]"
    @test _format_specs(TensorSpec[]) == "(none)"
end

@testset "observe: compiled batch sizes" begin
    @test _compiled_sizes(_obs_model([0]; weights = nothing, nbytes = 0)) == "unbatched"
    @test _compiled_sizes(_obs_model([4, 1, 8]; weights = nothing, nbytes = 0)) == "[1, 4, 8]"
end

@testset "observe: resident weight accounting" begin
    reg = ModelRegistry()
    reg.by_name["resident"] = _obs_entry("resident", _obs_model([1]; weights = Any[1], nbytes = 1000))
    reg.by_name["evicted"] = _obs_entry("evicted", _obs_model([1]; weights = nothing, nbytes = 2000))
    rw = resident_weight_bytes(reg)
    @test rw.bytes == 1000        # only the device-resident model counts
    @test rw.count == 1
end

@testset "observe: multi-angle memory report" begin
    pool = _obs_pool()
    reg = ModelRegistry()
    reg.by_name["a"] = _obs_entry("a", _obs_model([1]; weights = Any[1], nbytes = 4096))
    cache = WeightCache(MockBackend(), pool, reg, 1 << 20)

    # MockBackend reports no device stats -> the device angle degrades to "n/a".
    @test occursin("device n/a", memory_report(MockBackend(), pool))
    full = memory_report(MockBackend(), pool; registry = reg, weight_cache = cache)
    @test occursin("device n/a", full)
    @test occursin("resident weights", full)
    @test occursin("1 models", full)
    @test occursin("on-demand budget", full)
end

@testset "observe: structured log records" begin
    entry = _obs_entry("scale", _obs_model([1, 4]; weights = Any[1], nbytes = 2048, state = PINNED_SYSTEM))
    @test_logs (:info, "model loaded") log_model_loaded(entry, entry.executable; source = :startup, memory = "device n/a")
    @test_logs (:info, "model unloaded") log_model_unloaded("scale", 2048; memory = "device n/a")
    # Residency moves are request-path churn and log at debug level only.
    @test_logs min_level = Base.CoreLogging.Debug (:debug, "residency: model moved") log_residency_change("scale", PINNED_SYSTEM, PINNED_DEVICE, 2048; memory = "device n/a")
end

@testset "observe: nvidia-smi is bounded and keeps off the worker's stdio" begin
    dir = mktempdir()
    fake(name, body) = (p = joinpath(dir, name); write(p, "#!/bin/sh\n" * body); chmod(p, 0o755); p)
    # Run `f` with the worker's stderr redirected to a pipe; return its result, how long it took,
    # and what reached that stderr.
    function with_stderr(f)
        errpipe = Pipe()
        Base.link_pipe!(errpipe; reader_supports_async = true, writer_supports_async = true)
        captured = @async read(errpipe.out, String)
        t0 = time()
        result = redirect_stderr(f, errpipe.in)
        elapsed = time() - t0
        close(errpipe.in)
        return result, elapsed, fetch(captured)
    end
    # A wedged nvidia-smi: before, the scrape waited for it as long as it took (here 60 s), and it
    # wrote onto the worker's stderr, which under the supervisor is the supervisor's pipe.
    wedged = fake("wedged", "echo 'nvidia-smi complains' >&2\nexec sleep 60\n")
    result, elapsed, err = @test_logs (:warn, r"nvidia-smi did not answer within 1.0s") match_mode = :any with_stderr(
        () -> ReactantServer.process_device_used_bytes(; smi = wedged, timeout = 1.0)
    )
    @test result === nothing
    @test elapsed < 10
    @test !occursin("nvidia-smi complains", err)
    # A working one: its line for this process is read, and its stderr still goes nowhere.
    working = fake("working", "echo 'note on stderr' >&2\necho '1, 5'\necho '$(getpid()), 300'\n")
    result, _, err = with_stderr(() -> ReactantServer.process_device_used_bytes(; smi = working))
    @test result == 300 * 1024 * 1024
    @test !occursin("note on stderr", err)
    # One that fails yields nothing.
    @test ReactantServer.process_device_used_bytes(; smi = fake("failing", "exit 2\n")) === nothing
end
