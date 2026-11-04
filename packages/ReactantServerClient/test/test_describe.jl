# describe_models / describe_model against a stub ControlService that answers DescribeModels the way a
# worker does: Core's encode_model_description over a manifest, NOT_FOUND for a name it does not
# serve. Checks that the row-major wire shapes come back in Julia column-major order, axis kinds and
# variants included.

using Test
using ReactantServerClient
import ReactantServerCore
import gRPCServer

let ctl = ReactantServerClient.control
    manifest = ReactantServerCore.parse_manifest(
        Dict{String, Any}(
            "format_version" => "2.0", "name" => "det",
            "executable_inputs" => [
                Dict("name" => "img", "dtype" => "f32", "shape" => "whcn", "dims" => Dict("w" => -1, "h" => -1, "c" => 3)),
            ],
            "executable_outputs" => [
                Dict("name" => "boxes", "dtype" => "f32", "shape" => "kn", "dims" => Dict("k" => 4)),
            ],
            "batching" => Dict{String, Any}("compiled_batch_sizes" => [1, 4]),
            "input_shapes" => [Dict("w" => 640, "h" => 480)],
        )
    )

    function describe_handler(ctx, req)
        names = isempty(req.names) ? ["det"] : collect(String, req.names)
        unknown = filter(!=("det"), names)
        isempty(unknown) ||
            throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "unknown model: $(join(unknown, ", "))"))
        return ctl.DescribeModelsResponse(;
            models = [ReactantServerCore.encode_model_description(ctl, "det", manifest; max_batch_size = 4)]
        )
    end

    @testset "describe_models over a live DescribeModels" begin
        server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", 0)
        # The client ships no server stubs, so register the one RPC through the runtime primitive.
        gRPCServer.register_method!(
            server.dispatcher, "reactant_control.ControlService",
            gRPCServer.MethodDescriptor(
                "DescribeModels", gRPCServer.MethodType.UNARY,
                ctl.DescribeModelsRequest, ctl.DescribeModelsResponse, describe_handler
            )
        )
        gRPCServer.start!(server)
        kserve_init()
        try
            url = "grpc://127.0.0.1:$(something(gRPCServer.bound_port(server)))"
            d = only(describe_models(url))
            @test d isa ModelDescription
            @test d.name == "det" && d.kind === :model
            # Julia column-major, the manifest's own `whcn` order, batch last.
            @test d.inputs[1].shape == [-1, -1, 3, -1]
            @test d.inputs[1].axes == [:variable, :variable, :fixed, :batch]
            @test d.outputs[1].shape == [4, -1]
            @test d.batch_sizes == [1, 4] && d.max_batch_size == 4
            @test d.variants == [Dict("img" => [640, 480, 3, -1])]
            @test isempty(d.calls)

            @test describe_model(KServeModel(url, "det")).inputs[1].name == "img"
            err = try
                describe_models(url; names = ["det", "nope"])
                nothing
            catch e
                e
            end
            @test err isa ReactantServerClient.gRPCClient.gRPCServiceCallException
            @test err.grpc_status == ReactantServerClient.gRPCClient.GRPC_NOT_FOUND
        finally
            kserve_shutdown()
            close(server)
        end
    end
end
