# Worker Prometheus export: the pull collector emits the expected series from live scheduler /
# weight-cache state, hot-path request counters appear, device-memory series are absent under
# MockBackend, dynamic load/unload is reflected, and the HTTP endpoint serves the exposition.
# All Reactant-free (MockBackend), so fast.

using ReactantServer: WorkerMetrics, inc_request!, observe_request!, start_worker_metrics,
    Scheduler, ModelRegistry, ModelEntry, ModelSchedState, ModelSchedConfig, SchedulerConfig,
    ServerConfig, RuntimeConfig, EndpointsConfig, CPU_BACKEND, CUDA_BACKEND, ModelSignature,
    LoadedModel, Manifest, TensorSpec, Dim, BatchingSpec, Provenance, FIXED, BATCH, F32,
    UNPINNED, MockBackend, MockClient, MockDevice, MemoryPool, WeightCache,
    inject_metric_labels, _gpu_identity
import Prometheus
import HTTP

function _wm_sched()
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    reg = ModelRegistry()
    sig = ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    inx = TensorSpec("x", F32, Dim[Dim(FIXED, 2), Dim(BATCH)], 2)
    outy = TensorSpec("y", F32, Dim[Dim(FIXED, 2), Dim(BATCH)], 2)
    man = Manifest(
        "2.0", "m", "", TensorSpec[inx], TensorSpec[outy], nothing, nothing,
        BatchingSpec(Int[]), Provenance(Dict{String, Any}()), 1
    )
    # weights set (Any[1]) ⇒ device-resident; execs values are unused by the metrics collector.
    model = LoadedModel(sig, Dict{Int, Any}(1 => nothing, 4 => nothing), Any[1], UNPINNED, 4096, nothing)
    reg.by_name["m"] = ModelEntry("m", man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
    sched = Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
    reg.by_name["m"].sched = ModelSchedState("m", ModelSchedConfig(1.0), 0.0)
    cfg = ServerConfig(
        ["."], "", RuntimeConfig(CPU_BACKEND, 0, 0.9, true, true), sched.cfg,
        EndpointsConfig("127.0.0.1", 0)
    )
    return sched, backend, pool, cfg
end

_expose(wm) = (io = IOBuffer(); Prometheus.expose(io, wm.registry); String(take!(io)))

@testset "worker metrics: pull collector series" begin
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg; worker_name = "worker0")
    s = _expose(wm)
    @test occursin("worker_dispatch_total{model=\"m\"}", s)
    @test occursin("worker_queue_depth{model=\"m\"}", s)
    @test occursin("worker_model_resident{model=\"m\"} 1", s)
    @test occursin("worker_models_loaded", s)
    @test occursin("worker_resident_weight_bytes", s)
    # identity labels
    @test occursin("worker_info{", s)
    @test occursin("worker=\"worker0\"", s)
    @test occursin("control_mode=\"static\"", s)
    # device memory is unavailable under MockBackend; the weight cache is disabled here.
    @test !occursin("worker_device_memory_in_use_bytes", s)
    @test !occursin("worker_weight_cache_resident_bytes", s)
end

@testset "worker metrics: weight cache series when enabled" begin
    sched, backend, pool, cfg = _wm_sched()
    sched.weight_cache = WeightCache(backend, pool, sched.registry, 1 << 20)
    wm = WorkerMetrics(sched, backend, pool, cfg)
    s = _expose(wm)
    @test occursin("worker_weight_cache_resident_bytes", s)
    @test occursin("worker_weight_cache_max_bytes", s)
    for r in ("bytes", "contiguity", "scratch")
        @test occursin("worker_weight_evicts_total{reason=\"$r\"} 0", s)
    end
    for name in (
            "worker_weight_compaction_seconds_total", "worker_weight_stages_total",
            "worker_weight_stage_failures_total", "worker_weight_stage_seconds_total",
            "worker_weight_stalled_reads_total", "worker_weight_stalled_reads",
        )
        @test occursin("$name 0", s)
    end
    @test occursin("worker_weight_staging_models{state=\"reading\"} 0", s)
    @test occursin("worker_weight_staging_models{state=\"staged\"} 0", s)
end

# A backend whose allocator reports a largest free block, to exercise the fragmentation series.
struct FragStatsBackend <: ReactantServer.AbstractBackend end
ReactantServer.device_memory_stats(::FragStatsBackend, pool) = (
    in_use = 600, limit = 1000, free = 400, peak_in_use = 700, pool_bytes = 1000,
    peak_pool_bytes = 1000, largest_free = 100,
)

@testset "worker metrics: fragmentation and footprint series" begin
    sched, _, pool, cfg = _wm_sched()
    backend = FragStatsBackend()
    sched.weight_cache = WeightCache(backend, pool, sched.registry, 1 << 20)
    fp = ReactantServer.model_footprint(sched.weight_cache, sched.registry.by_name["m"])
    fp.scratch_bytes = 12345
    ReactantServer.record_load!(fp, 0.5)
    wm = WorkerMetrics(sched, backend, pool, cfg)
    s = _expose(wm)
    @test occursin("worker_device_memory_largest_free_block_bytes 100", s)
    @test occursin("worker_device_memory_fragmentation 0.75", s)
    @test occursin("worker_model_scratch_bytes{model=\"m\"} 12345", s)
    @test occursin("worker_model_load_seconds{model=\"m\"} 0.5", s)
    @test occursin("worker_model_loads_total{model=\"m\"} 1", s)
    @test occursin("worker_model_reads_total{model=\"m\"} 0", s)
    @test occursin("worker_model_evicts_total{model=\"m\"} 0", s)
    @test occursin("worker_weight_load_bytes_total", s)
end

@testset "worker metrics: request counters" begin
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg)
    inc_request!(wm, "m", "OK")
    inc_request!(wm, "m", "DEADLINE_EXCEEDED")
    observe_request!(wm, "m", 0.012)
    s = _expose(wm)
    @test occursin("worker_requests_total{model=\"m\",status=\"OK\"}", s)
    @test occursin("status=\"DEADLINE_EXCEEDED\"", s)
    @test occursin("worker_request_latency_seconds", s)
end

@testset "worker metrics: the interactive-thread monitor sees the thread held" begin
    if Threads.nthreads(:interactive) == 0
        @test_skip "needs an interactive thread"
    else
        reg = Prometheus.CollectorRegistry()
        m = ReactantServer.InteractiveMonitor(reg; interval = 0.02, warn_after = 0.2)
        ReactantServer.start_interactive_monitor!(m)
        sleep(0.2)                                   # free ticks
        # Hold the interactive thread without yielding, as un-offloaded work would.
        wait(Threads.@spawn :interactive (t = time(); while time() - t < 0.4 end))
        sleep(0.2)
        ReactantServer.stop_interactive_monitor!(m)
        io = IOBuffer()
        Prometheus.expose(io, reg)
        body = String(take!(io))
        stalls = parse(Float64, match(r"^worker_interactive_stalls_total (\S+)$"m, body)[1])
        @test stalls >= 1
        count_ = parse(Float64, match(r"^worker_interactive_lag_seconds_count (\S+)$"m, body)[1])
        under = parse(Float64, match(r"^worker_interactive_lag_seconds_bucket\{le=\"0.256\"\} (\S+)$"m, body)[1])
        @test count_ >= 2 && under < count_          # some ticks on time, at least one 0.4 s late
        # The gauge holds the longest delay of the last 30 s: the 0.4 s hold, not an average.
        longest = parse(Float64, match(r"^worker_interactive_lag_max_seconds (\S+)$"m, body)[1])
        @test 0.3 <= longest < 5.0
    end
end

@testset "worker metrics: the dispatch loop's activity reads as a sentence for a stall report" begin
    p = ReactantServer.DispatchProgress()
    @test ReactantServer.dispatch_activity(p, 10.0) == "now: idle; before: idle"
    ReactantServer._begin_unit!(p, :dispatch, "m", 4)
    ReactantServer._note_in_place!(p)
    t = (@atomic p.current).started
    @test ReactantServer.dispatch_activity(p, t + 1.3) ==
        "now: dispatch of m at batch 4, run in place on this thread, running for 1.3 s; before: idle"
    ReactantServer._begin_unit!(p, :control)          # the dispatch ended as the command began
    u = @atomic p.previous
    txt = ReactantServer.dispatch_activity(p, u.ended + 0.5)
    @test startswith(txt, "now: control command, running for 0.5 s; before: dispatch of m at batch 4, run in place on this thread, took ")
    @test endswith(txt, ", ended 0.5 s ago")
    ReactantServer._end_unit!(p)
    @test (@atomic p.current).kind === :idle && (@atomic p.previous).kind === :control
    ReactantServer._begin_unit!(p, :dispatch, "n", 1)  # not in place unless noted
    @test occursin("dispatch of n at batch 1, run on a default-pool task", ReactantServer.dispatch_activity(p))
    ReactantServer._begin_unit!(p, :dispatch, "u", 0)  # a program with no batch axis
    @test occursin("now: dispatch of u unbatched, run on a default-pool task", ReactantServer.dispatch_activity(p))
end

@testset "worker metrics: a stall's log line says what held the thread" begin
    if Threads.nthreads(:interactive) == 0
        @test_skip "needs an interactive thread"
    else
        reg = Prometheus.CollectorRegistry()
        logger = Test.TestLogger()
        Base.CoreLogging.with_logger(logger) do
            m = ReactantServer.InteractiveMonitor(reg; interval = 0.02, warn_after = 0.2, context = () -> "the loop's report")
            ReactantServer.start_interactive_monitor!(m)
            sleep(0.2)
            wait(Threads.@spawn :interactive (t = time(); while time() - t < 0.4 end))
            sleep(0.2)
            ReactantServer.stop_interactive_monitor!(m)
        end
        stalls = filter(r -> occursin("interactive thread was blocked", string(r.message)), logger.logs)
        @test !isempty(stalls)
        kw = Dict(last(stalls).kwargs)
        @test all(k -> haskey(kw, k), (:blocked_s, :gc_s, :safepoint_wait_s, :collections, :compile_s))
        @test kw[:activity] == "the loop's report"
        io = IOBuffer()
        Prometheus.expose(io, reg)
        @test occursin(r"^julia_gc_time_to_safepoint_seconds_total \S+$"m, String(take!(io)))
    end
end

@testset "worker metrics: the monitor's longest delay covers its window, then is forgotten" begin
    reg = Prometheus.CollectorRegistry()
    m = ReactantServer.InteractiveMonitor(reg; window = 30)
    longest() = (io = IOBuffer(); Prometheus.expose(io, reg); parse(Float64, match(r"^worker_interactive_lag_max_seconds (\S+)$"m, String(take!(io)))[1]))
    observe(late, second) = ReactantServer._observe_lag!(m, late, second)
    observe(0.5, 1000)
    observe(0.1, 1010)
    @test longest() == 0.5
    observe(0.05, 1029)
    @test longest() == 0.5                           # second 1000 is still within the last 30 s
    observe(0.05, 1030)
    @test longest() == 0.1                           # 1000 has left the window (its slot is reused); 1010 has not
    observe(0.02, 1040)
    @test longest() == 0.05                          # 1010 has left too
    observe(0.2, 1040)
    @test longest() == 0.2                           # a second keeps its longest tick
    observe(0.01, 1100)
    @test longest() == 0.01                          # after a quiet stretch only the new tick counts
end

@testset "worker metrics: latency buckets reach past a minute" begin
    # A 50 s request lands in a finite bucket (and matches the gateway's buckets), not in +Inf.
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg)
    observe_request!(wm, "m", 50.0)
    s = _expose(wm)
    @test occursin("worker_request_latency_seconds_bucket{model=\"m\",le=\"32.768\"} 0", s)
    @test occursin("worker_request_latency_seconds_bucket{model=\"m\",le=\"65.536\"} 1", s)
    @test occursin("worker_request_latency_seconds_bucket{model=\"m\",le=\"131.072\"} 1", s)
end

@testset "worker metrics: unloading a model drops its snapshot series" begin
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg)
    @test occursin("worker_dispatch_total{model=\"m\"}", _expose(wm))
    delete!(sched.registry.by_name, "m")          # simulate an unload
    @test !occursin("worker_dispatch_total{model=\"m\"}", _expose(wm))
end

@testset "worker metrics: HTTP endpoint" begin
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg; worker_name = "worker0")
    port = grpc_free_port()
    server = start_worker_metrics(wm, "127.0.0.1", port; ready_fn = () -> true)
    try
        resp = HTTP.get("http://127.0.0.1:$port/metrics"; retry = false, readtimeout = 60)
        @test resp.status == 200
        @test occursin("worker_info", String(resp.body))
        @test HTTP.get("http://127.0.0.1:$port/healthz"; retry = false, readtimeout = 60).status == 200
        @test HTTP.get("http://127.0.0.1:$port/readyz"; retry = false, readtimeout = 60).status == 200
    finally
        close(server)
    end
end

@testset "worker metrics: label injection" begin
    text = """
    # HELP worker_dispatch_total Total dispatches per model.
    # TYPE worker_dispatch_total counter
    worker_dispatch_total{model="m"} 3
    # TYPE worker_models_loaded gauge
    worker_models_loaded 1
    worker_request_latency_seconds_bucket{model="m",le="0.001"} 0
    worker_info{worker="worker0",device_ordinal="0"} 1
    """
    out = inject_metric_labels(text, ["worker" => "worker0", "gpu" => "1"])
    @test occursin("worker_dispatch_total{worker=\"worker0\",gpu=\"1\",model=\"m\"} 3", out)
    @test occursin("worker_models_loaded{worker=\"worker0\",gpu=\"1\"} 1", out)
    @test occursin("worker_request_latency_seconds_bucket{worker=\"worker0\",gpu=\"1\",model=\"m\",le=\"0.001\"} 0", out)
    # An already-present key is not duplicated; missing keys are still added.
    @test occursin("worker_info{gpu=\"1\",worker=\"worker0\",device_ordinal=\"0\"} 1", out)
    # Comment lines pass through untouched.
    @test occursin("# HELP worker_dispatch_total Total dispatches per model.", out)
    @test inject_metric_labels(text, Pair{String, String}[]) == text

    # The HTTP endpoint applies the labels to the real exposition.
    sched, backend, pool, cfg = _wm_sched()
    wm = WorkerMetrics(sched, backend, pool, cfg; worker_name = "worker0")
    port = grpc_free_port()
    server = start_worker_metrics(
        wm, "127.0.0.1", port; ready_fn = () -> true,
        worker_name = "worker0", gpu = "2"
    )
    try
        body = String(HTTP.get("http://127.0.0.1:$port/metrics"; retry = false, readtimeout = 60).body)
        @test occursin("worker_dispatch_total{worker=\"worker0\",gpu=\"2\",model=\"m\"}", body)
        @test occursin("worker_models_loaded{worker=\"worker0\",gpu=\"2\"}", body)
    finally
        close(server)
    end
end

@testset "worker metrics: gpu identity" begin
    cuda_cfg(ord) = ServerConfig(
        ["."], "", RuntimeConfig(CUDA_BACKEND, ord, 0.9, true, true),
        SchedulerConfig(30.0, 64, 30.0), EndpointsConfig("127.0.0.1", 0)
    )

    # The supervisor / per-GPU-container case: a single selector is the physical identity.
    @test _gpu_identity(cuda_cfg(0), Dict("CUDA_VISIBLE_DEVICES" => "3")) == "3"
    @test _gpu_identity(cuda_cfg(0), Dict("CUDA_VISIBLE_DEVICES" => "GPU-aaaa")) == "GPU-aaaa"
    # Bare metal, several visible devices: the addressed ordinal's token.
    @test _gpu_identity(cuda_cfg(1), Dict("CUDA_VISIBLE_DEVICES" => "4,5")) == "5"
    # No CUDA_VISIBLE_DEVICES: the device ordinal is the best available identity.
    @test _gpu_identity(cuda_cfg(2), Dict{String, String}()) == "2"
    # CPU backend: no gpu label.
    cpu = ServerConfig(
        ["."], "", RuntimeConfig(CPU_BACKEND, 0, 0.9, true, true),
        SchedulerConfig(30.0, 64, 30.0), EndpointsConfig("127.0.0.1", 0)
    )
    @test _gpu_identity(cpu, Dict("CUDA_VISIBLE_DEVICES" => "3")) == ""
end
