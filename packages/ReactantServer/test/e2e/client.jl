# End-to-end client for the full serving stack. Drives the running reactant-gateway over both the
# inline (TCP) data path and the POSIX shared-memory data path, for a tiny exact-arithmetic
# model (scale4) and the real bit_resnet50 model, and asserts correctness.
#
# Run after the stack is up (see test/e2e/run_e2e.sh):
#   julia --project=test/e2e test/e2e/client.jl
#
# Inference and SHM register/unregister go through the gateway (127.0.0.1:GATEWAY_PORT). The
# gateway does not serve ModelMetadata, so shapes/dtypes are fetched directly from worker0
# (127.0.0.1:WORKER0_PORT). Workers run with ipc:host, so a region created here in /dev/shm is
# visible to whichever worker handles the request.

using ReactantServer
import InterProcessCommunication as IPC
import gRPCClient
include(joinpath(@__DIR__, "..", "grpc_helpers.jl"))   # grpc_call(Req, Resp, rpc, port, request)

const Inf = ReactantServer.inference
const GATEWAY_PORT = parse(Int, get(ENV, "E2E_GATEWAY_PORT", "8001"))
const WORKER0_PORT = parse(Int, get(ENV, "E2E_WORKER0_PORT", "8080"))

_sp(s) = Inf.InferParameter(; parameter_choice = ReactantServer.ProtoBuf.OneOf(:string_param, String(s)))
_ip(i) = Inf.InferParameter(; parameter_choice = ReactantServer.ProtoBuf.OneOf(:int64_param, Int64(i)))

const FAILURES = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        println("PASS  ", name)
    else
        println("FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
        push!(FAILURES, name)
    end
end

# Create a POSIX shm region; returns (handle, key, byte-view). own=false so finalize unlinks it.
function make_region(nbytes::Int)
    key = "/reactantserver-e2e-$(getpid())-$(rand(UInt32))"
    shm = IPC.SharedMemory(key, nbytes)
    view = unsafe_wrap(Array, convert(Ptr{UInt8}, pointer(shm)), nbytes; own = false)
    fill!(view, 0x00)
    return shm, key, view
end

# Fetch model metadata from worker0; batch dims (<= 0) are filled to 1.
function metadata(model::AbstractString)
    md = grpc_call(Inf.ModelMetadataRequest, Inf.ModelMetadataResponse, "ModelMetadata",
        WORKER0_PORT, Inf.ModelMetadataRequest(; name = model))
    fixshape(s) = Int64[d <= 0 ? 1 : d for d in s]
    return (; in_name = md.inputs[1].name, in_dt = md.inputs[1].datatype, in_shape = fixshape(md.inputs[1].shape),
        out_name = md.outputs[1].name, out_dt = md.outputs[1].datatype, out_shape = fixshape(md.outputs[1].shape))
end

elcount(shape) = Int(prod(shape))

# ModelInfer over the inline (TCP) path through the gateway.
function infer_tcp(model, m, in_bytes::Vector{UInt8})
    inp = Inf.var"ModelInferRequest.InferInputTensor"(; name = m.in_name, datatype = m.in_dt, shape = m.in_shape)
    req = Inf.ModelInferRequest(; model_name = model, inputs = [inp], raw_input_contents = [in_bytes])
    return grpc_call(Inf.ModelInferRequest, Inf.ModelInferResponse, "ModelInfer", GATEWAY_PORT, req)
end

# ModelInfer over the shared-memory path through the gateway: input from one region, output to
# another. Registers/unregisters both regions via the gateway (which fans out to all workers).
# Returns (response, output-bytes-read-back-from-the-output-region).
function infer_shm(model, m, in_bytes::Vector{UInt8}, out_nbytes::Int)
    in_n = length(in_bytes)
    shm_in, key_in, view_in = make_region(in_n)
    shm_out, key_out, view_out = make_region(out_nbytes)
    reg_in = "e2e-in-$(getpid())-$(rand(UInt32))"
    reg_out = "e2e-out-$(getpid())-$(rand(UInt32))"
    register(name, key, n) = grpc_call(Inf.SystemSharedMemoryRegisterRequest, Inf.SystemSharedMemoryRegisterResponse,
        "SystemSharedMemoryRegister", GATEWAY_PORT,
        Inf.SystemSharedMemoryRegisterRequest(; name = name, key = key, offset = 0, byte_size = n))
    try
        register(reg_in, key_in, in_n)
        register(reg_out, key_out, out_nbytes)
        copyto!(view_in, in_bytes)
        inp = Inf.var"ModelInferRequest.InferInputTensor"(; name = m.in_name, datatype = m.in_dt, shape = m.in_shape,
            parameters = Dict("shared_memory_region" => _sp(reg_in),
                "shared_memory_offset" => _ip(0), "shared_memory_byte_size" => _ip(in_n)))
        outp = Inf.var"ModelInferRequest.InferRequestedOutputTensor"(; name = m.out_name,
            parameters = Dict("shared_memory_region" => _sp(reg_out),
                "shared_memory_offset" => _ip(0), "shared_memory_byte_size" => _ip(out_nbytes)))
        resp = grpc_call(Inf.ModelInferRequest, Inf.ModelInferResponse, "ModelInfer", GATEWAY_PORT,
            Inf.ModelInferRequest(; model_name = model, inputs = [inp], outputs = [outp]))
        return resp, copy(view_out)
    finally
        for r in (reg_in, reg_out)
            try
                grpc_call(Inf.SystemSharedMemoryUnregisterRequest, Inf.SystemSharedMemoryUnregisterResponse,
                    "SystemSharedMemoryUnregister", GATEWAY_PORT,
                    Inf.SystemSharedMemoryUnregisterRequest(; name = r))
            catch
            end
        end
        finalize(shm_in)
        finalize(shm_out)
    end
end

println("== e2e client: gateway=127.0.0.1:$GATEWAY_PORT worker0=127.0.0.1:$WORKER0_PORT ==")

# ---------- scale4: exact arithmetic over both transports, through the gateway ----------
let
    m = metadata("scale4")
    xb = collect(reinterpret(UInt8, Float32[1, 2, 3, 4]))
    expected = Float32[2, 4, 6, 8]

    r = infer_tcp("scale4", m, xb)
    got = collect(reinterpret(Float32, r.raw_output_contents[1]))
    check("scale4 / TCP exact output", r.model_name == "scale4" && got == expected; detail = "got=$got")

    resp, outb = infer_shm("scale4", m, xb, 16)
    gots = collect(reinterpret(Float32, outb))
    check("scale4 / SHM exact output (output written to region)",
        isempty(resp.raw_output_contents) && gots == expected; detail = "got=$gots")
end

# ---------- bit_resnet50: real model; shape/dtype, determinism, transport equivalence ----------
let
    m = metadata("bit_resnet50")
    in_count = elcount(m.in_shape)
    out_nbytes = elcount(m.out_shape) * 4
    x = Float32[(i % 251) / 251 for i in 0:(in_count - 1)]   # deterministic, non-degenerate
    xb = collect(reinterpret(UInt8, x))

    # Relative error max|a-b| / max|b| (the project's numerical-comparison convention). The model
    # is served replicated round-robin across two GPUs, and XLA may select different conv
    # algorithms per GPU, so results are numerically equivalent but not bit-identical. A tight
    # relative tolerance is the correct cross-replica assertion: a gross error (misroute, wrong
    # output) is O(1) relative, while fp/algorithm differences are tiny.
    f32(b) = collect(reinterpret(Float32, b))
    relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(Float32))
    TOL = 1.0f-2

    r1 = infer_tcp("bit_resnet50", m, xb)
    r2 = infer_tcp("bit_resnet50", m, xb)
    tcp = r1.raw_output_contents[1]
    o1, o2 = f32(tcp), f32(r2.raw_output_contents[1])
    check("bit_resnet50 / TCP model_name + dtype", r1.model_name == "bit_resnet50" && m.out_dt == "FP32")
    check("bit_resnet50 / TCP output byte length", length(tcp) == out_nbytes;
        detail = "len=$(length(tcp)) want=$out_nbytes")
    e_det = relerr(o2, o1)
    check("bit_resnet50 / TCP determinism across replicas (rel<$TOL)", e_det < TOL; detail = "relerr=$e_det")

    resp, outb = infer_shm("bit_resnet50", m, xb, out_nbytes)
    os = f32(collect(outb))
    check("bit_resnet50 / SHM output to region (inline empty)", isempty(resp.raw_output_contents))
    e_eq = relerr(os, o1)
    check("bit_resnet50 / SHM vs TCP equivalence (rel<$TOL)", e_eq < TOL; detail = "relerr=$e_eq")
    println("    [bit_resnet50 relerr] determinism=$e_det  shm_vs_tcp=$e_eq")
end

if isempty(FAILURES)
    println("\nALL CHECKS PASSED")
    exit(0)
else
    println("\nFAILURES: ", join(FAILURES, ", "))
    exit(1)
end
