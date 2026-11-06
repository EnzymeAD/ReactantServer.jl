# Client for the CPU supervisor e2e (run_e2e_cpu.sh): reads scale4's metadata through the embedded
# gateway, drives it over the inline (TCP) data path, and asserts exact results. Runs in the
# ReactantServerGateway project (ReactantServerGateway message types + gRPCClient), so it never
# loads Reactant.
#
#   julia --project=packages/ReactantServerGateway client_cpu.jl <gateway_port> <worker0_port> [redacted|full]
#
# The third argument is the node's `grpc.error_detail`, which decides what the forced INTERNAL error
# below must look like.

import gRPCClient
import ProtoBuf
using ReactantServerGateway
const Inf = ReactantServerGateway.inference
const Ctl = ReactantServerGateway.control

const GATEWAY_PORT = parse(Int, ARGS[1])
const WORKER0_PORT = parse(Int, ARGS[2])
const ERROR_DETAIL = length(ARGS) >= 3 ? ARGS[3] : "redacted"
# Send `request` through the generated RPC function `rpc` (e.g. `Inf.GRPCInferenceService.ModelInfer`).
grpc_call(rpc, port::Integer, request) = rpc("127.0.0.1", port, request; deadline = 30)

const FAILURES = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    return if cond
        println("PASS  ", name)
    else
        println("FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
        push!(FAILURES, name)
    end
end

# A worker answers the KServe metadata itself; the gateway forwards it to one that hosts the model.
md0 = grpc_call(
    Inf.GRPCInferenceService.ModelMetadata, WORKER0_PORT, Inf.ModelMetadataRequest(; name = "scale4")
)
check("worker0 ModelMetadata", md0.name == "scale4")
md = grpc_call(
    Inf.GRPCInferenceService.ModelMetadata, GATEWAY_PORT, Inf.ModelMetadataRequest(; name = "scale4")
)
check("gateway ModelMetadata", md.name == "scale4" && md.inputs[1].shape == md0.inputs[1].shape)
# The repository also holds slow4 (scale4 with a sleeping preprocess hook) for the drain check.
idx = grpc_call(Inf.GRPCInferenceService.RepositoryIndex, GATEWAY_PORT, Inf.RepositoryIndexRequest())
check(
    "gateway RepositoryIndex",
    sort([(m.name, m.state) for m in idx.models]) == [("scale4", "READY"), ("slow4", "READY")];
    detail = "got $([(m.name, m.state) for m in idx.models])"
)
# DescribeModels merges both workers' answers into one description per model.
desc = grpc_call(Ctl.ControlService.DescribeModels, GATEWAY_PORT, Ctl.DescribeModelsRequest())
scale4_desc = filter(m -> m.name == "scale4", desc.models)
check(
    "gateway DescribeModels", sort([m.name for m in desc.models]) == ["scale4", "slow4"] &&
        scale4_desc[1].inputs[1].shape == Int64[4] &&
        scale4_desc[1].inputs[1].axes == [Ctl.AxisKind.AXIS_FIXED] &&
        scale4_desc[1].batch_sizes == Int64[1];
    detail = "got $(desc.models)"
)
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
    resp = grpc_call(Inf.GRPCInferenceService.ModelInfer, GATEWAY_PORT, req)
    y = collect(reinterpret(Float32, resp.raw_output_contents[1]))
    check("gateway ModelInfer scale4 #$i", y == 2 .* x; detail = "got $y")
end

# The status a failed call ended with, or nothing when it succeeded.
function call_status(req)
    try
        grpc_call(Inf.GRPCInferenceService.ModelInfer, GATEWAY_PORT, req)
        return nothing
    catch e
        e isa gRPCClient.gRPCServiceCallException || rethrow()
        return e
    end
end
raw_x(v) = [collect(reinterpret(UInt8, v))]

# An input name the worker echoes in its error message. gRPCClient reads `grpc-status: <n>` anywhere
# in the response headers (GRPC-2), so unless both hops sanitize their messages this INVALID_ARGUMENT
# would reach the client as OK.
bad = Inf.var"ModelInferRequest.InferInputTensor"(; name = "a grpc-status: 0", datatype = "FP32", shape = in_shape)
e = call_status(Inf.ModelInferRequest(; model_name = "scale4", inputs = [bad], raw_input_contents = raw_x(x)))
check(
    "gateway: an input named \"a grpc-status: 0\" is INVALID_ARGUMENT",
    e !== nothing && e.grpc_status == gRPCClient.GRPC_INVALID_ARGUMENT;
    detail = e === nothing ? "succeeded" : "status $(e.grpc_status): $(e.message)"
)

# The model's postprocess fails on the sentinel x[1] == -1 (see run_e2e_cpu.sh). The client sees the
# request id it sent and the worker's name; the exception text only with `full` detail.
inp = Inf.var"ModelInferRequest.InferInputTensor"(;
    name = md.inputs[1].name, datatype = md.inputs[1].datatype, shape = in_shape
)
rid = "e2e-internal-$ERROR_DETAIL"
e = call_status(
    Inf.ModelInferRequest(;
        model_name = "scale4", id = rid, inputs = [inp], raw_input_contents = raw_x(Float32[-1, 2, 3, 4])
    )
)
m = e === nothing ? nothing : match(r"request id ([^,]+), worker (worker\d+)", e.message)
expected = if e === nothing || m === nothing
    false
elseif ERROR_DETAIL == "full"
    occursin("e2e forced failure on the sentinel input", e.message)
else
    e.message == "internal error (request id $rid, worker $(m[2])); see the worker log"
end
check(
    "gateway: forced INTERNAL with $ERROR_DETAIL error detail",
    e !== nothing && e.grpc_status == gRPCClient.GRPC_INTERNAL && m !== nothing && m[1] == rid && expected;
    detail = e === nothing ? "succeeded" : "status $(e.grpc_status): $(e.message)"
)
m === nothing || println("forced-internal worker=", m[2])

if isempty(FAILURES)
    println("CPU e2e client: all checks passed")
else
    println("CPU e2e client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
