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
    @test occursin("worker_weight_evicts_total", s)
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
        resp = HTTP.get("http://127.0.0.1:$port/metrics"; retry = false, readtimeout = 5)
        @test resp.status == 200
        @test occursin("worker_info", String(resp.body))
        @test HTTP.get("http://127.0.0.1:$port/healthz"; retry = false, readtimeout = 5).status == 200
        @test HTTP.get("http://127.0.0.1:$port/readyz"; retry = false, readtimeout = 5).status == 200
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
        body = String(HTTP.get("http://127.0.0.1:$port/metrics"; retry = false, readtimeout = 5).body)
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
