# End-to-end: start the real server (CPU backend) hosting a bundle, drive it over gRPC with
# the KServe V2 GRPCInferenceService, and verify the result. This is the slice's definition of
# done.

const _EInf = ReactantServer.inference

@testset "server end-to-end (CPU)" begin
    mktempdir() do root
        manifest = """
        format_version: "2.0"
        name: scale4
        executable_inputs:
          - name: x
            dtype: f32
            shape: c
            dims:
              c: 4
        executable_outputs:
          - name: y
            dtype: f32
            shape: c
            dims:
              c: 4
        batching:
          compiled_batch_sizes: [1]
        """
        mlir = """
        module {
          func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.multiply %x, %w : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        write_bundle(root, "scale4";
            manifest_yaml=manifest, mlir_text=mlir,
            weights=Dict("w" => Float32[2, 2, 2, 2]), argument_order=["w"])

        port = grpc_free_port()
        cfg = ReactantServer.ServerConfig([root], "",
            ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", port))

        srv = ReactantServer.serve(cfg; backend=ReactantServer.ReactantBackend(), blocking=false)
        sleep(0.3)
        try
            # health / readiness
            @test grpc_call(_EInf.ServerLiveRequest, _EInf.ServerLiveResponse, "ServerLive", port,
                            _EInf.ServerLiveRequest()).live
            @test grpc_call(_EInf.ServerReadyRequest, _EInf.ServerReadyResponse, "ServerReady", port,
                            _EInf.ServerReadyRequest()).ready
            @test grpc_call(_EInf.ModelReadyRequest, _EInf.ModelReadyResponse, "ModelReady", port,
                            _EInf.ModelReadyRequest(; name="scale4")).ready
            @test !grpc_call(_EInf.ModelReadyRequest, _EInf.ModelReadyResponse, "ModelReady", port,
                             _EInf.ModelReadyRequest(; name="nope")).ready

            # model metadata
            md = grpc_call(_EInf.ModelMetadataRequest, _EInf.ModelMetadataResponse, "ModelMetadata", port,
                           _EInf.ModelMetadataRequest(; name="scale4"))
            @test md.name == "scale4"
            @test md.inputs[1].name == "x" && md.inputs[1].datatype == "FP32"
            @test md.outputs[1].name == "y"

            # unknown model metadata -> NOT_FOUND
            try
                grpc_call(_EInf.ModelMetadataRequest, _EInf.ModelMetadataResponse, "ModelMetadata", port,
                          _EInf.ModelMetadataRequest(; name="nope"))
                @test false
            catch ex
                @test ex isa gRPCClient.gRPCServiceCallException
                @test ex.grpc_status == ReactantServer._G.GRPC_NOT_FOUND
            end

            # inference
            x = Float32[1, 2, 3, 4]
            inp = _EInf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[4])
            reqmsg = _EInf.ModelInferRequest(; model_name="scale4", inputs=[inp],
                raw_input_contents=[collect(reinterpret(UInt8, x))])
            rmsg = grpc_call(_EInf.ModelInferRequest, _EInf.ModelInferResponse, "ModelInfer", port, reqmsg)
            @test rmsg.model_name == "scale4"
            @test collect(reinterpret(Float32, rmsg.raw_output_contents[1])) == Float32[2, 4, 6, 8]

            # inference against an unknown model -> NOT_FOUND
            try
                grpc_call(_EInf.ModelInferRequest, _EInf.ModelInferResponse, "ModelInfer", port,
                          _EInf.ModelInferRequest(; model_name="nope", inputs=[inp],
                              raw_input_contents=[collect(reinterpret(UInt8, x))]))
                @test false
            catch ex
                @test ex isa gRPCClient.gRPCServiceCallException
                @test ex.grpc_status == ReactantServer._G.GRPC_NOT_FOUND
            end

            # repository index lists the loaded model (direct RepositoryIndex introspection)
            idx = grpc_call(_EInf.RepositoryIndexRequest, _EInf.RepositoryIndexResponse, "RepositoryIndex", port,
                            _EInf.RepositoryIndexRequest(; ready=true))
            @test "scale4" in [m.name for m in idx.models]

            # shared-memory status with nothing registered returns an empty region set
            shmst = grpc_call(_EInf.SystemSharedMemoryStatusRequest, _EInf.SystemSharedMemoryStatusResponse,
                              "SystemSharedMemoryStatus", port, _EInf.SystemSharedMemoryStatusRequest(; name=""))
            @test isempty(shmst.regions)
        finally
            ReactantServer.stop!(srv)
        end
    end
end
