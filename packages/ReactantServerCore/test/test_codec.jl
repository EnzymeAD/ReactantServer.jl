# Codec round-trips: build a ModelInferRequest message, translate to the boundary type, build
# a response message, and inspect it. The codec now translates between decoded protobuf
# messages and the boundary types (the transport owns wire framing), so these tests work with
# message objects directly. Exercises both raw_input_contents and inline typed contents.

const _Inf = ReactantServer.inference

@testset "codec round-trip (raw contents)" begin
    x = Float32[1, 2, 3, 4]
    raw = collect(reinterpret(UInt8, x))
    inp = _Inf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[4])
    out = _Inf.var"ModelInferRequest.InferRequestedOutputTensor"(; name="y")
    msg = _Inf.ModelInferRequest(; model_name="scale4", inputs=[inp], outputs=[out],
                                 raw_input_contents=[raw])

    decoded = ReactantServer.decode_infer_request(msg)
    req, id = decoded.request, decoded.id
    @test req.model_name == "scale4"
    @test req.requested_outputs == ["y"]
    @test length(req.inputs) == 1
    @test req.inputs[1].name == "x"
    @test req.inputs[1].dtype == ReactantServer.F32
    @test req.inputs[1].data == x

    # build a response message and inspect it directly
    rmsg = ReactantServer.encode_infer_response("scale4", id, [ReactantServer.NamedTensor("y", Float32[2, 4, 6, 8])])
    @test rmsg isa _Inf.ModelInferResponse
    @test rmsg.model_name == "scale4"
    @test length(rmsg.outputs) == 1
    @test rmsg.outputs[1].name == "y"
    @test rmsg.outputs[1].datatype == "FP32"
    @test collect(reinterpret(Float32, rmsg.raw_output_contents[1])) == Float32[2, 4, 6, 8]
end

@testset "codec deadline KV param round-trip" begin
    x = Float32[1, 2, 3, 4]
    inp = ReactantServer.NamedTensor("x", x)
    # encode with a remaining-budget timeout -> decode converts it to an absolute local deadline.
    budget = Int64(5_000_000_000)
    msg = ReactantServer.encode_infer_request("m", [inp]; parameters=ReactantServer.deadline_params(budget))
    @test haskey(msg.parameters, ReactantServer.TIMEOUT_NS_PARAM)
    before = Int64(time_ns())
    dec = ReactantServer.decode_infer_request(msg).request
    after = Int64(time_ns())
    # The absolute deadline lands within [now+budget] of the decode instant (relative->absolute).
    @test before + budget <= dec.deadline_ns <= after + budget

    # Absent param -> no deadline.
    @test ReactantServer.decode_infer_request(ReactantServer.encode_infer_request("m", [inp])).request.deadline_ns == 0
    # A non-positive budget produces an empty params map (no deadline carried).
    @test isempty(ReactantServer.deadline_params(0))
    @test isempty(ReactantServer.deadline_params(-5))

    # The SHM encoder carries the param too (meta fan-out path).
    shmmsg = ReactantServer.encode_infer_request_shm("m", [inp], "region", [0];
                                                     parameters=ReactantServer.deadline_params(budget))
    @test haskey(shmmsg.parameters, ReactantServer.TIMEOUT_NS_PARAM)
end

@testset "codec inline typed contents" begin
    contents = _Inf.InferTensorContents(; fp32_contents=Float32[5, 6, 7])
    inp = _Inf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[3], contents=contents)
    msg = _Inf.ModelInferRequest(; model_name="m", inputs=[inp])
    req = ReactantServer.decode_infer_request(msg).request
    @test req.inputs[1].data == Float32[5, 6, 7]
end

@testset "codec honors requested_outputs" begin
    rawx = collect(reinterpret(UInt8, Float32[1, 2]))
    inp = _Inf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[2])
    produced = [ReactantServer.NamedTensor("a", Float32[1, 1]), ReactantServer.NamedTensor("b", Float32[2, 2])]

    # a subset request returns only the requested outputs, in the requested order
    sel = [_Inf.var"ModelInferRequest.InferRequestedOutputTensor"(; name="b"),
           _Inf.var"ModelInferRequest.InferRequestedOutputTensor"(; name="a")]
    msg = _Inf.ModelInferRequest(; model_name="m", inputs=[inp], outputs=sel, raw_input_contents=[rawx])
    decoded = ReactantServer.decode_infer_request(msg)
    rmsg = ReactantServer.encode_infer_response("m", decoded, produced, nothing)
    @test [o.name for o in rmsg.outputs] == ["b", "a"]
    @test collect(reinterpret(Float32, rmsg.raw_output_contents[1])) == Float32[2, 2]

    # an empty request returns all outputs in model order
    allmsg = _Inf.ModelInferRequest(; model_name="m", inputs=[inp], raw_input_contents=[rawx])
    alldec = ReactantServer.decode_infer_request(allmsg)
    rall = ReactantServer.encode_infer_response("m", alldec, produced, nothing)
    @test [o.name for o in rall.outputs] == ["a", "b"]

    # requesting an output the model does not produce is an error
    bad = [_Inf.var"ModelInferRequest.InferRequestedOutputTensor"(; name="zzz")]
    badmsg = _Inf.ModelInferRequest(; model_name="m", inputs=[inp], outputs=bad, raw_input_contents=[rawx])
    baddec = ReactantServer.decode_infer_request(badmsg)
    @test_throws Exception ReactantServer.encode_infer_response("m", baddec, produced, nothing)
end

@testset "codec model metadata" begin
    manifest = ReactantServer.parse_manifest(Dict{String,Any}(
        "format_version" => "2.0", "name" => "scale4",
        "executable_inputs" => [Dict("name" => "x", "dtype" => "f32",
                                     "shape" => "c", "dims" => Dict("c" => 4))],
        "executable_outputs" => [Dict("name" => "y", "dtype" => "f32",
                                      "shape" => "c", "dims" => Dict("c" => 4))],
        "batching" => Dict{String,Any}(),
    ))
    md = ReactantServer.encode_model_metadata("scale4", manifest, "xla")
    @test md isa _Inf.ModelMetadataResponse
    @test md.name == "scale4"
    @test md.platform == "xla"
    @test md.inputs[1].name == "x"
    @test md.inputs[1].datatype == "FP32"
    @test md.inputs[1].shape == Int64[4]
    @test md.outputs[1].name == "y"
end

@testset "codec repository index" begin
    idx = ReactantServer.encode_repository_index(["a", "b"])
    @test idx isa _Inf.RepositoryIndexResponse
    @test [m.name for m in idx.models] == ["a", "b"]
    @test all(m.state == "READY" for m in idx.models)
end

@testset "codec rejects hostile shapes before allocating" begin
    _msg(shape, raw) = _Inf.ModelInferRequest(; model_name="m",
        inputs=[_Inf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=shape)],
        raw_input_contents=[raw])

    # A shape whose byte size dwarfs the payload is rejected from the shape product alone,
    # before any allocation of the claimed size is attempted.
    @test_throws Exception ReactantServer.decode_infer_request(_msg(Int64[2^30, 2^30], UInt8[0, 0, 0, 0]))

    # A shape whose element count overflows Int64 is rejected with a clean error.
    @test_throws Exception ReactantServer.decode_infer_request(_msg(Int64[2^62, 2^62], UInt8[0, 0, 0, 0]))

    # Negative dimensions are rejected.
    @test_throws Exception ReactantServer.decode_infer_request(_msg(Int64[-4], UInt8[0, 0, 0, 0]))

    # Inline typed contents take the same validated path.
    c = _Inf.InferTensorContents(; fp32_contents=Float32[1, 2])
    badc = _Inf.ModelInferRequest(; model_name="m",
        inputs=[_Inf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32",
                                                             shape=Int64[2^62, 2^62], contents=c)])
    @test_throws Exception ReactantServer.decode_infer_request(badc)

    # Sane shapes still round-trip.
    good = ReactantServer.decode_infer_request(_msg(Int64[2, 2],
        collect(reinterpret(UInt8, Float32[1, 2, 3, 4]))))
    @test size(good.request.inputs[1].data) == (2, 2)
end
