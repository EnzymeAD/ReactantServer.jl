# Integration tests for the pure-Julia gateway. Two mock worker gRPC servers (typed handlers)
# sit behind a real serve_gateway; the tests drive it with a typed gRPCClient and assert routing,
# round-robin, failover, NotFound, SHM fan-out with rollback, and the admin endpoints. Reuses the
# free-port and grpc_call helpers from grpc_helpers.jl.

import gRPCServer
import gRPCClient
import HTTP

const GWInf = ReactantServerCore.inference

mutable struct MockWorker
    name::String
    models::Vector{String}     # models this worker reports READY via RepositoryIndex
    fail_infer::Bool
    fail_shm::Bool
end

function _mock_router()
    router = gRPCServer.gRPCRouter()
    ReactantServerGateway.register_GRPCInferenceService!(router;
        ServerReady = (req, c) -> GWInf.ServerReadyResponse(; ready = true),
        RepositoryIndex = (req, c) -> GWInf.RepositoryIndexResponse(; models = [
            GWInf.var"RepositoryIndexResponse.ModelIndex"(; name = m, version = "", state = "READY", reason = "")
            for m in c.payload.models]),
        ModelInfer = (req, c) -> begin
            w = c.payload
            w.fail_infer && throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_UNAVAILABLE, "mock $(w.name) down"))
            # A real worker NOT_FOUNDs a model it no longer serves; mirror that so the gateway's
            # unloaded-model handling (route refresh on worker NOT_FOUND) can be exercised.
            (req.model_name in w.models) ||
                throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_NOT_FOUND, "model $(req.model_name) not on $(w.name)"))
            GWInf.ModelInferResponse(; model_name = w.name, id = req.id)
        end,
        SystemSharedMemoryRegister = (req, c) -> begin
            c.payload.fail_shm && throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_FAILED_PRECONDITION, "mock shm fail"))
            GWInf.SystemSharedMemoryRegisterResponse()
        end,
        SystemSharedMemoryUnregister = (req, c) -> GWInf.SystemSharedMemoryUnregisterResponse(),
    )
    return router
end

_start_mock(worker::MockWorker, port::Integer) =
    gRPCServer.serve!(_mock_router(), "127.0.0.1", port; context = worker)

# Send a typed ModelInfer through the gateway and return the response (or rethrow).
_infer(port, model) = grpc_call(GWInf.ModelInferRequest, GWInf.ModelInferResponse, "ModelInfer",
    port, GWInf.ModelInferRequest(; model_name = model))

function _http_get(port, path)
    try
        return HTTP.get("http://127.0.0.1:$port$path"; retry = false, status_exception = false)
    catch e
        return e
    end
end

@testset "gateway" begin
    # worker0 serves replicated + only0; worker1 serves replicated. The gateway discovers this
    # from each worker's RepositoryIndex rather than from any config.
    w0 = MockWorker("worker0", ["replicated", "only0"], false, false)
    w1 = MockWorker("worker1", ["replicated"], false, false)
    p0 = grpc_free_port()
    p1 = grpc_free_port()
    gw_port = grpc_free_port()
    admin_port = grpc_free_port()

    s0 = _start_mock(w0, p0)
    s1 = _start_mock(w1, p1)

    gatewayfile = tempname() * ".yaml"
    write(gatewayfile, """
    listen:
      grpc: "127.0.0.1:$gw_port"
      metrics: "127.0.0.1:$admin_port"
    endpoints:
      - "127.0.0.1:$p0"
      - "127.0.0.1:$p1"
    """)

    gw = ReactantServerGateway.serve_gateway(gatewayfile; blocking = false)

    # Wait for the first discovery round to populate routes before asserting.
    routed = false
    for _ in 1:40
        try
            _infer(gw_port, "only0")
            routed = true
            break
        catch
            sleep(0.1)
        end
    end
    @test routed

    try
        @testset "routing" begin
            r = _infer(gw_port, "only0")
            @test r.model_name == "worker0"
        end

        @testset "round-robin across replicas" begin
            seen = Set{String}()
            for _ in 1:4
                push!(seen, _infer(gw_port, "replicated").model_name)
            end
            @test seen == Set(["worker0", "worker1"])
        end

        @testset "unknown model is NotFound" begin
            err = try
                _infer(gw_port, "nope")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_NOT_FOUND
        end

        @testset "failover to healthy replica" begin
            w0.fail_infer = true
            for _ in 1:4
                @test _infer(gw_port, "replicated").model_name == "worker1"
            end
            # only0 has no healthy replica -> error surfaced
            err = try
                _infer(gw_port, "only0")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            w0.fail_infer = false
        end

        @testset "SHM register fan-out and rollback" begin
            reg(name) = grpc_call(GWInf.SystemSharedMemoryRegisterRequest, GWInf.SystemSharedMemoryRegisterResponse,
                "SystemSharedMemoryRegister", gw_port, GWInf.SystemSharedMemoryRegisterRequest(; name = name))
            # all workers ok
            @test reg("region-ok") isa GWInf.SystemSharedMemoryRegisterResponse
            # one worker fails -> FailedPrecondition
            w1.fail_shm = true
            err = try
                reg("region-bad")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_FAILED_PRECONDITION
            w1.fail_shm = false
        end

        @testset "admin endpoints" begin
            @test _http_get(admin_port, "/healthz").status == 200
            # readiness flips to 200 once a worker reports ready
            ready = false
            for _ in 1:20
                _http_get(admin_port, "/readyz").status == 200 && (ready = true; break)
                sleep(0.25)
            end
            @test ready
            metrics = _http_get(admin_port, "/metrics")
            @test metrics.status == 200
            body = String(metrics.body)
            @test occursin("gateway_requests_total", body)
            @test occursin("gateway_worker_ready", body)
            @test occursin("gateway_routing_table_size", body)
        end

        @testset "on-demand refresh routes a newly loaded model" begin
            # 'lazy' is unknown to every worker initially, so the gateway answers NOT_FOUND.
            err = try; _infer(gw_port, "lazy"); nothing; catch e; e; end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_NOT_FOUND
            # Load it on worker0. Past the refresher's min_interval the next request rescans on
            # demand and routes it, well before the 10s health tick.
            push!(w0.models, "lazy")
            sleep(1.1)
            @test _infer(gw_port, "lazy").model_name == "worker0"
        end

        @testset "on-demand refresh drops an unloaded model" begin
            @test ReactantServerGateway.pick(gw.routes, "only0") !== nothing   # routed before unload
            filter!(!=("only0"), w0.models)                                    # unload on the worker
            # A request to the stale route gets NOT_FOUND from the worker, which kicks an async
            # refresh; within a couple of seconds the route is dropped from the gateway's table.
            dropped = false
            for _ in 1:40
                try; _infer(gw_port, "only0"); catch; end
                if ReactantServerGateway.pick(gw.routes, "only0") === nothing
                    dropped = true
                    break
                end
                sleep(0.2)
            end
            @test dropped
        end
    finally
        ReactantServerGateway.stop!(gw)
        close(s0)
        close(s1)
        rm(gatewayfile; force = true)
    end
end

@testset "env-only gateway config (no gateway.yml)" begin
    # The node supervisor starts an embedded gateway with no config file: defaults plus
    # REACTANT_GATEWAY_* environment variables, with the endpoint list synthesized into
    # REACTANT_GATEWAY_WORKERS.
    withenv("REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080,127.0.0.1:8081",
            "REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:9001") do
        cfg = ReactantServerGateway.load_gateway(nothing)
        @test cfg.workers == ["127.0.0.1:8080", "127.0.0.1:8081"]
        @test cfg.listen_grpc == "0.0.0.0:9001"
        @test cfg.listen_metrics == "0.0.0.0:8002"       # default
        # gRPC message-size limits default to 512 MiB and honor the env override.
        @test cfg.max_recv_msg_bytes == 512 * 1024 * 1024
        @test cfg.max_send_msg_bytes == 512 * 1024 * 1024
    end

    withenv("REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
            "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => "4096") do
        cfg = ReactantServerGateway.load_gateway(nothing)
        @test cfg.max_recv_msg_bytes == 4096
        @test cfg.max_send_msg_bytes == 512 * 1024 * 1024   # unset -> default
    end

    # Without an endpoint list the env-only path keeps the existing guard.
    withenv("REACTANT_GATEWAY_WORKERS" => nothing) do
        @test_throws ReactantServerCore.ConfigError ReactantServerGateway.load_gateway(nothing)
    end

    # An explicit path that does not exist still fails loudly.
    @test_throws ReactantServerCore.ConfigError ReactantServerGateway.load_gateway(tempname())
end

@testset "metrics aggregation" begin
    @testset "merge_expositions" begin
        gw = """
        # HELP gateway_requests_total Count of gateway RPCs.
        # TYPE gateway_requests_total counter
        gateway_requests_total{rpc="ModelInfer",model="m",status="OK"} 2
        """
        w0 = """
        # HELP worker_dispatch_total Total dispatches per model.
        # TYPE worker_dispatch_total counter
        worker_dispatch_total{worker="worker0",gpu="0",model="m"} 3
        # TYPE worker_request_latency_seconds histogram
        worker_request_latency_seconds_bucket{worker="worker0",model="m",le="+Inf"} 1
        worker_request_latency_seconds_count{worker="worker0",model="m"} 1
        """
        w1 = """
        # HELP worker_dispatch_total Total dispatches per model.
        # TYPE worker_dispatch_total counter
        worker_dispatch_total{worker="worker1",gpu="1",model="m"} 5
        """
        merged = ReactantServerGateway.merge_expositions([gw, w0, w1])
        # One header per family, both workers' samples grouped under it.
        @test count("# TYPE worker_dispatch_total counter", merged) == 1
        @test occursin("worker_dispatch_total{worker=\"worker0\",gpu=\"0\",model=\"m\"} 3", merged)
        @test occursin("worker_dispatch_total{worker=\"worker1\",gpu=\"1\",model=\"m\"} 5", merged)
        @test occursin("gateway_requests_total{rpc=\"ModelInfer\",model=\"m\",status=\"OK\"} 2", merged)
        # Histogram suffix lines stay under their family header.
        i_type = findfirst("# TYPE worker_request_latency_seconds histogram", merged)
        i_count = findfirst("worker_request_latency_seconds_count", merged)
        @test i_type !== nothing && i_count !== nothing && first(i_type) < first(i_count)
        # The worker0 block's samples come before any re-declaration would; exposition parses
        # as one block per family (no duplicate headers anywhere).
        @test count("# HELP worker_dispatch_total", merged) == 1
    end

    @testset "aggregated /metrics endpoint" begin
        # Two fake workers serving pre-tagged expositions, one dead endpoint.
        function fake(text)
            p = grpc_free_port()
            s = HTTP.serve!("127.0.0.1", p) do req
                HTTP.Response(200; body = text)
            end
            return s, "127.0.0.1:$p"
        end
        s0, ep0 = fake("worker_models_loaded{worker=\"worker0\",gpu=\"0\"} 1\n")
        s1, ep1 = fake("worker_models_loaded{worker=\"worker1\",gpu=\"1\"} 1\n")
        dead = "127.0.0.1:$(grpc_free_port())"
        admin_port = grpc_free_port()
        metrics = ReactantServerGateway.GatewayMetrics()
        admin = ReactantServerGateway.start_admin(metrics, "127.0.0.1:$admin_port";
            worker_metrics = [ep0, ep1, dead])
        try
            body = String(HTTP.get("http://127.0.0.1:$admin_port/metrics"; retry = false).body)
            @test occursin("worker_models_loaded{worker=\"worker0\",gpu=\"0\"} 1", body)
            @test occursin("worker_models_loaded{worker=\"worker1\",gpu=\"1\"} 1", body)
            @test count("# TYPE gateway_worker_metrics_up gauge", body) == 1
            @test occursin("gateway_worker_metrics_up{worker=\"$ep0\"} 1", body)
            @test occursin("gateway_worker_metrics_up{worker=\"$dead\"} 0", body)
        finally
            close(admin.server)
            close(s0)
            close(s1)
        end
    end

    @testset "metrics endpoints config" begin
        withenv("REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
                "REACTANT_GATEWAY_WORKER_METRICS" => "127.0.0.1:9100, 127.0.0.1:9101") do
            cfg = ReactantServerGateway.load_gateway(nothing)
            @test cfg.worker_metrics == ["127.0.0.1:9100", "127.0.0.1:9101"]
        end
        withenv("REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
                "REACTANT_GATEWAY_WORKER_METRICS" => nothing) do
            @test ReactantServerGateway.load_gateway(nothing).worker_metrics == String[]
        end
    end
end
