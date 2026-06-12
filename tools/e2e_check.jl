# End-to-end numeric check of a recovered bundle through the real server:
# load the on-disk bundle, serve it, run a KServe infer, and compare against a
# direct TorchScript forward pass. Validates the full weight-ordering path
# (bundle safetensors -> runtime argument_order -> compiled StableHLO).

using HTTP, Sockets, Random, PythonCall

const HAS_TORCH = try
    pyimport("torch"); pyimport("torch.export"); pyimport("torchax.export")
    pyimport("torchax.ops.jaten"); pyimport("triton._C.libtriton"); pyimport("numpy"); true
catch err
    @info "skip: torch not importable" error=err; false
end
HAS_TORCH || exit(0)

using ReactantServerExport
using ReactantServer
ReactantServerExport._pyimports()   # initialize ReactantServerExport's numpy/jax refs used by the conversion helpers

const _Inf = ReactantServer.inference
const torch = pyimport("torch")
const np = pyimport("numpy")

_free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)
_encode(m) = (io = IOBuffer(); ReactantServer.ProtoBuf.encode(ReactantServer.ProtoBuf.ProtoEncoder(io), m); take!(io))
_decode(::Type{T}, b) where {T} = ReactantServer.ProtoBuf.decode(ReactantServer.ProtoBuf.ProtoDecoder(IOBuffer(b)), T)
torch_to_julia(t) = (a = t.detach().cpu().contiguous().numpy();
                     ReactantServerExport._numpy_to_julia(a, ReactantServerExport._numpy_dtype_to_julia(pyconvert(String, a.dtype.name))))
function julia_to_torch(arr::AbstractArray{T}) where {T}
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(arr))))
    np_arr = np.frombuffer(pybytes(bytes), dtype=ReactantServerExport._julia_to_numpy_dtype(T)).reshape(collect(reverse(size(arr))))
    return torch.from_numpy(np_arr.copy())
end

const MODEL = "inconsistent_anatomy_c2c7"
const PT = "/docker/triton/dynamic/$MODEL/1/model.pt"

# Isolate the single bundle into its own root so the server loads only it.
root = mktempdir()
symlink("/docker/reactantserver/models/$MODEL", joinpath(root, MODEL))

port = _free_port()
cfg = ReactantServer.ServerConfig([root], "",
    ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 64, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port))
srv = ReactantServer.serve(cfg; backend=ReactantServer.ReactantBackend(), blocking=false)
try
    sleep(0.5)
    base = "http://127.0.0.1:$port"
    r = HTTP.get("$base/v2/models/$MODEL/ready"; status_exception=false)
    r.status == 200 || error("not ready: $(r.status)")

    # UINT8 input (2,153,356) at batch 1: Julia col-major (356,153,2,1).
    x = rand(Random.Xoshiro(1), UInt8, 356, 153, 2, 1)
    wire = collect(Int64, reverse(size(x)))
    inp = _Inf.var"ModelInferRequest.InferInputTensor"(; name="INPUT__0", datatype="UINT8", shape=wire)
    req = _Inf.ModelInferRequest(; model_name=MODEL, inputs=[inp], raw_input_contents=[Vector{UInt8}(vec(x))])
    resp = HTTP.post("$base/v2/models/$MODEL/infer",
        ["Content-Type" => "application/x-protobuf"], _encode(req); status_exception=false, readtimeout=300)
    resp.status == 200 || error("infer failed: $(resp.status) $(String(copy(resp.body)))")
    rmsg = _decode(_Inf.ModelInferResponse, resp.body)
    yshape = reverse(collect(Int, rmsg.outputs[1].shape))
    y_server = reshape(Vector{Float32}(reinterpret(Float32, rmsg.raw_output_contents[1])), yshape...)

    ref = torch.jit.load(PT, map_location="cpu"); ref.eval()
    y_ref = torch_to_julia(ref(julia_to_torch(x)))

    maxd = maximum(abs, y_server .- y_ref); maxr = maximum(abs, y_ref)
    println("\n=== $MODEL e2e ===")
    println("server : ", vec(y_server))
    println("ref    : ", vec(y_ref))
    println("rel err (max|diff|/max|ref|): ", maxd / maxr)
    println(isapprox(y_server, y_ref; rtol=1f-3, atol=1f-3) ? "MATCH" : "MISMATCH")
finally
    ReactantServer.stop!(srv)
end
