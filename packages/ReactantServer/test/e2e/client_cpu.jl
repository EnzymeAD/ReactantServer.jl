# Client for the CPU supervisor e2e (run_e2e_cpu.sh): drives the scale4 model through the
# embedded gateway over the inline (TCP) data path and asserts exact results. Runs in the
# ReactantServerGateway project (ReactantServerCore message types + gRPCClient), so it never
# loads Reactant.
#
#   julia --project=packages/ReactantServerGateway client_cpu.jl <gateway_port> <worker0_port>

import gRPCClient
import ProtoBuf
using ReactantServerCore
const Inf = ReactantServerCore.inference

const GATEWAY_PORT = parse(Int, ARGS[1])
const WORKER0_PORT = parse(Int, ARGS[2])
const _GRPC_SERVICE = "/inference.GRPCInferenceService"

function grpc_call(::Type{Req}, ::Type{Resp}, rpc::AbstractString, port::Integer, request) where {Req, Resp}
    client = gRPCClient.gRPCServiceClient{Req, false, Resp, false}("127.0.0.1", port, "$_GRPC_SERVICE/$rpc"; deadline = 30)
    return gRPCClient.grpc_sync_request(client, request)
end

const FAILURES = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    return if cond
        println("PASS  ", name)
    else
        println("FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
        push!(FAILURES, name)
    end
end

# Worker0 serves full model metadata (the gateway only forwards inference and SHM RPCs).
md = grpc_call(
    Inf.ModelMetadataRequest, Inf.ModelMetadataResponse, "ModelMetadata",
    WORKER0_PORT, Inf.ModelMetadataRequest(; name = "scale4")
)
check("worker0 ModelMetadata", md.name == "scale4")
in_shape = Int64[d <= 0 ? 1 : d for d in md.inputs[1].shape]

# Two inferences through the gateway: with both CPU workers serving scale4, round-robin lands
# on each of them.
x = Float32[1, 2, 3, 4]
for i in 1:2
    inp = Inf.var"ModelInferRequest.InferInputTensor"(;
        name = md.inputs[1].name, datatype = md.inputs[1].datatype, shape = in_shape
    )
    req = Inf.ModelInferRequest(;
        model_name = "scale4", inputs = [inp],
        raw_input_contents = [collect(reinterpret(UInt8, x))]
    )
    resp = grpc_call(Inf.ModelInferRequest, Inf.ModelInferResponse, "ModelInfer", GATEWAY_PORT, req)
    y = collect(reinterpret(Float32, resp.raw_output_contents[1]))
    check("gateway ModelInfer scale4 #$i", y == 2 .* x; detail = "got $y")
end

if isempty(FAILURES)
    println("CPU e2e client: all checks passed")
else
    println("CPU e2e client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
