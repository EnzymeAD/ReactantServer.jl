# Precompile workload. The gateway runs every task on one thread, so code compiling on first use
# blocks every request, probe and scrape until it finishes. Measured before this existed (CPU worker
# behind a fresh gateway): about 8 s of compile during startup, 5.7 s on the first DescribeModels call
# (the event loop blocked for several seconds) and 0.7 s on the first ModelInfer.
#
# The workload runs the real gateway against an in-process mock worker on loopback ports the OS
# assigns: startup and the first probe round, ModelInfer, DescribeModels, ModelMetadata and
# RepositoryIndex over HTTP/2, and the admin /metrics fan-out and /readyz. It runs once per scheduling
# mode that changes the compiled types (round_robin and lpt_packing), then stops everything. The
# gRPC calls go over HTTP.jl's HTTP/2 client.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    _pc_encode(msg) = (io = IOBuffer(); PB.encode(PB.ProtoEncoder(io), msg); take!(io))
    _pc_frame(msg::Vector{UInt8}) = vcat(UInt8[0], reinterpret(UInt8, [hton(UInt32(length(msg)))]), msg)

    # A worker that serves one model and answers everything the gateway asks at startup and per request.
    function _pc_mock_worker()
        server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", 0)
        register_GRPCInferenceService!(
            server;
            ServerReady = (ctx, req) -> inference.ServerReadyResponse(; ready = true),
            RepositoryIndex = (ctx, req) -> inference.RepositoryIndexResponse(;
                models = [inference.var"RepositoryIndexResponse.ModelIndex"(; name = "m", version = "", state = "READY", reason = "")]
            ),
            ModelInfer = (ctx, req) -> inference.ModelInferResponse(; model_name = req.model_name, id = req.id),
            ModelMetadata = (ctx, req) -> inference.ModelMetadataResponse(; name = req.name, platform = "mock"),
        )
        register_ControlService!(
            server;
            ModelControlStatus = (ctx, req) -> control.ModelControlStatusResponse(;
                residency_mode = "self_managed", discipline = "fifo",
                models = [
                    control.ModelStatus(;
                        name = "m", weight_nbytes = Int64(1 << 20), total_compute_seconds = 1.0,
                        requests_served = UInt64(10), rows_served = UInt64(10), dispatch_count = UInt64(10),
                        max_batch_size = Int64(4), batch_input_name = "x", batch_axis = Int64(1)
                    ),
                ],
                weight_cache_max_bytes = UInt64(1) << 30
            ),
            CompactMemory = (ctx, req) -> control.CompactMemoryResponse(; reloaded_models = Int64(0)),
            DescribeModels = (ctx, req) -> control.DescribeModelsResponse(;
                models = [control.ModelDescription(; name = "m", kind = "model")]
            ),
        )
        gRPCServer.start!(server)
        return server, something(gRPCServer.bound_port(server))
    end

    function _pc_until(f; timeout = 15.0)
        t_end = monotonic_time() + timeout
        while !f() && monotonic_time() < t_end
            sleep(0.05)
        end
        return nothing
    end

    # One gateway in `mode` over the mock worker: start it, wait for the prober, exercise it, stop it.
    function _pc_gateway(mode, wport, mport, client)
        mktempdir() do dir
            path = joinpath(dir, "gateway.yaml")
            write(
                path, """
                listen:
                  grpc: "127.0.0.1:0"
                  metrics: "127.0.0.1:0"
                scheduling:
                  mode: $mode
                  rebalance_compute_seconds: 0.001
                endpoints:
                  - "127.0.0.1:$wport"
                metrics_endpoints:
                  - "127.0.0.1:$mport"
                worker_names: [worker0]
                """
            )
            gw = serve_gateway(path; blocking = false)
            try
                gport, aport = bound_ports(gw)
                ok(r) = r.status == 200
                _pc_until(() -> ok(HTTP.request("GET", "http://127.0.0.1:$aport/readyz"; client = client, protocol = :h1, status_exception = false)))
                headers = ["content-type" => "application/grpc", "te" => "trailers"]
                infer = inference.ModelInferRequest(;
                    model_name = "m",
                    inputs = [inference.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[1, 2])],
                    raw_input_contents = [zeros(UInt8, 8)]
                )
                for (rpc, msg) in (
                        ("inference.GRPCInferenceService/ModelInfer", infer),
                        ("reactant_control.ControlService/DescribeModels", control.DescribeModelsRequest()),
                        ("inference.GRPCInferenceService/ModelMetadata", inference.ModelMetadataRequest(; name = "m")),
                        ("inference.GRPCInferenceService/RepositoryIndex", inference.RepositoryIndexRequest()),
                    )
                    HTTP.request(
                        "POST", "http://127.0.0.1:$gport/$rpc"; client = client, protocol = :h2,
                        headers = headers, body = _pc_frame(_pc_encode(msg))
                    )
                end
                for p in ("/metrics", "/readyz")
                    HTTP.request("GET", "http://127.0.0.1:$aport$p"; client = client, protocol = :h1, status_exception = false)
                end
            finally
                stop!(gw)
            end
        end
        return nothing
    end

    # Every REACTANT_GATEWAY_* variable unset for the workload: they override the config file, so a
    # build (or a precompile triggered at run time, under a deployment's environment) would otherwise
    # point the workload's gateway at real workers and real listen ports.
    _pc_clean_env() = [k => nothing for k in keys(ENV) if startswith(k, GW_ENV_PREFIX)]

    @compile_workload withenv(_pc_clean_env()...) do
        worker, wport = _pc_mock_worker()
        mserver = HTTP.serve!(
            req -> HTTP.Response(200; body = "worker_models_loaded{worker=\"worker0\",gpu=\"0\"} 1\n"),
            "127.0.0.1", 0
        )
        client = HTTP.Client()
        try
            for mode in ("round_robin", "lpt_packing")
                _pc_gateway(mode, wport, HTTP.port(mserver), client)
            end
        finally
            HTTP.close_idle_connections!(client)
            close(mserver)
            gRPCServer.stop!(worker; force = true)
        end
    end
end

# The blocking entry point (the supervisor's), which the workload's non-blocking runs do not compile.
precompile(Tuple{typeof(serve_gateway)})
precompile(Tuple{typeof(serve_gateway), String})
