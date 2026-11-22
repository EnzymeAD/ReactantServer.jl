# The startup determinism check (determinism_canary.jl): every program of every model runs twice on
# one synthesized input; any difference fails the worker closed (not ready, requests refused, still
# live). All on MockBackend, whose programs are plain Julia functions, so a nondeterministic one is
# a closure that counts its calls.

using ReactantServer: Scheduler, SchedulerConfig, ModelRegistry, ModelEntry, ModelSignature,
    LoadedModel, Manifest, TensorSpec, Dim, BatchingSpec, Provenance, FIXED, BATCH, F32,
    MockBackend, MockClient, MockDevice, MockExecutable, MemoryPool, InferRequest, NamedTensor,
    WorkerUnavailableError, WorkerMetrics, ServerConfig, RuntimeConfig, EndpointsConfig, CPU_BACKEND,
    determinism_failed, determinism_outcomes
import Prometheus

# A model compiled at batch sizes 1 and 4 whose program is `run` (args -> outputs): input "x" and
# output "y", both (2, n) with the batch axis last. `inputs = false` gives it no executable inputs.
function _dc_entry(name, run; inputs = true)
    inx = TensorSpec("x", F32, Dim[Dim(FIXED, 2), Dim(BATCH)], 2)
    outy = TensorSpec("y", F32, Dim[Dim(FIXED, 2), Dim(BATCH)], 2)
    man = Manifest(
        "2.0", name, "", inputs ? TensorSpec[inx] : TensorSpec[], TensorSpec[outy], nothing, nothing,
        BatchingSpec(Int[]), Provenance(Dict{String, Any}()), 1
    )
    sig = ModelSignature(["x"], DataType[Float32], String[], 1, ["y"], DataType[Float32], 1)
    execs = Dict{Int, Any}(1 => MockExecutable(run, 1), 4 => MockExecutable(run, 1))
    return ModelEntry(name, man, Dict{Int, Vector{UInt8}}(), "", nothing, LoadedModel(sig, execs, Any[]), nothing, identity, identity)
end

function _dc_sched(entries...)
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    reg = ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
    end
    return Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
end

_dc_double(args) = [args[1] .* 2]

@testset "determinism check: deterministic programs pass, with one digest per program" begin
    calls = Threads.Atomic{Int}(0)
    s = _dc_sched(_dc_entry("a", args -> (Threads.atomic_add!(calls, 1); _dc_double(args))))
    ReactantServer.start!(s; determinism_check = true)
    try
        o = determinism_outcomes(s)["a"]
        @test o.status === :passed
        @test first.(o.digests) == ["b1", "b4"]
        @test all(d -> length(last(d)) == 64, o.digests)
        @test !determinism_failed(s)
        # Two canary runs per program, on top of the cost warm-up's runs.
        @test calls[] >= 4
        out = ReactantServer.infer(s, InferRequest("a", ["y"], [NamedTensor("x", ones(Float32, 2, 1))], Int64(0)))
        @test only(out).data == fill(2.0f0, 2, 1)
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "determinism check: a program that differs between runs fails the worker closed" begin
    calls = Threads.Atomic{Int}(0)
    # Every call adds its call number: the same input never gives the same output twice.
    flaky(args) = [args[1] .+ Float32(Threads.atomic_add!(calls, 1))]
    s = _dc_sched(_dc_entry("flaky", flaky), _dc_entry("good", _dc_double))
    @test_logs (:error, r"DETERMINISM CHECK FAILED: flaky") match_mode = :any ReactantServer.start!(s; determinism_check = true)
    try
        outcomes = determinism_outcomes(s)
        @test outcomes["flaky"].status === :failed
        @test occursin("different outputs on run 2", outcomes["flaky"].detail)
        @test outcomes["good"].status === :passed
        @test determinism_failed(s)

        # Requests are refused, for the good model too: the whole worker fails closed.
        @test_throws WorkerUnavailableError ReactantServer.infer(
            s, InferRequest("good", ["y"], [NamedTensor("x", ones(Float32, 2, 1))], Int64(0))
        )
        # Not ready, with the reason in the /readyz body; still live, so it is not restarted.
        ready = ReactantServer._RegistryReady(s.registry, s)()
        @test ready isa String && occursin("determinism check failed for flaky", ready)
        @test ReactantServer.dispatch_alive(s)

        # The metrics carry the outcome.
        cfg = ServerConfig(["."], "", RuntimeConfig(CPU_BACKEND, 0, 0.9, true, true), s.cfg, EndpointsConfig("127.0.0.1", 0))
        io = IOBuffer()
        Prometheus.expose(io, WorkerMetrics(s, s.backend, s.pool, cfg).registry)
        text = String(take!(io))
        @test occursin("worker_determinism_check_failed 1", text)
        @test occursin("worker_determinism_check{model=\"flaky\",outcome=\"failed\"} 1", text)
        @test occursin("worker_determinism_check{model=\"good\",outcome=\"passed\"} 1", text)
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "determinism check: unchecked models are reported but do not fail the worker" begin
    # No inputs to synthesize, and a program that raises on the canary input.
    s = _dc_sched(_dc_entry("noinputs", _dc_double; inputs = false), _dc_entry("raises", args -> error("boom")))
    ReactantServer.start!(s; determinism_check = true)
    try
        outcomes = determinism_outcomes(s)
        @test outcomes["noinputs"].status === :unchecked
        @test outcomes["raises"].status === :unchecked
        @test occursin("boom", outcomes["raises"].detail)
        @test !determinism_failed(s)
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "determinism check: off by default" begin
    calls = Threads.Atomic{Int}(0)
    flaky(args) = [args[1] .+ Float32(Threads.atomic_add!(calls, 1))]
    s = _dc_sched(_dc_entry("flaky", flaky))
    ReactantServer.start!(s)
    try
        @test isempty(determinism_outcomes(s))
        @test !determinism_failed(s)
        @test ReactantServer._RegistryReady(s.registry, s)() === true
    finally
        ReactantServer.shutdown!(s)
    end
end

@testset "determinism check: canary inputs" begin
    # Floating-point inputs get a fixed, non-constant sequence in [-1, 1) (zeros would sum the same in
    # any order); integer inputs stay zero, since they are often indices.
    a = ReactantServer._canary_array(Float32, [2, 3])
    @test a == ReactantServer._canary_array(Float32, [2, 3])
    @test length(unique(a)) > 1 && all(x -> -1 <= x < 1, a)
    @test ReactantServer._canary_array(Int32, [4]) == zeros(Int32, 4)
    @test ReactantServer._canary_array(Bool, [2]) == falses(2)
end
