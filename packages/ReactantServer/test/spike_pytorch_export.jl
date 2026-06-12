# De-risk spike for the PyTorch conversion frontend.
#
# Drives a tiny torch.nn.Module through torch.export -> torchax StableHLO ->
# ReactantServerExport, then loads the bundle through the current server runtime and
# confirms the output matches a native PyTorch forward pass at multiple batch
# sizes. The spike is self-gated: if torch / torchax aren't installed in the
# active project's Python environment, it logs a skip and exits 0.
#
# Run: julia --project=packages/ReactantServerExport/test packages/ReactantServer/test/spike_pytorch_export.jl

using PythonCall

const HAS_TORCH = try
    pyimport("torch"); pyimport("torch.export"); pyimport("torchax.export"); pyimport("numpy")
    true
catch err
    @info "Skipping spike: torch/torchax not importable" error=err
    false
end

if !HAS_TORCH
    exit(0)
end

using ReactantServerExport
using ReactantServer

const torch = pyimport("torch")
const np = pyimport("numpy")

pyexec("""
import torch
class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2, bias=True)
    def forward(self, x):
        return self.fc(x)
""", @__MODULE__)

function torch_to_julia(py_tensor)
    arr = py_tensor.detach().cpu().contiguous().numpy()
    T = ReactantServerExport._numpy_dtype_to_julia(pyconvert(String, arr.dtype.name))
    return ReactantServerExport._numpy_to_julia(arr, T)
end

function julia_to_torch(arr::AbstractArray{T}) where {T}
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(arr))))
    np_dtype = ReactantServerExport._julia_to_numpy_dtype(T)
    np_arr = np.frombuffer(pybytes(bytes), dtype=np_dtype).reshape(collect(reverse(size(arr))))
    return torch.from_numpy(np_arr.copy())
end

function main()
    torch.manual_seed(0)
    model = pyeval("TinyLinear()", @__MODULE__)

    mktempdir() do root
        example = randn(Float32, 4, 1)
        dir = joinpath(root, "tiny_linear")
        ReactantServerExport.export_bundle(:pytorch, model, (example,);
            dir=dir, name="tiny_linear",
            input_names=["x"], output_name="y",
            batch_sizes=[1, 4])

        @assert isfile(joinpath(dir, "model.b1.mlir"))
        @assert isfile(joinpath(dir, "model.b4.mlir"))
        @assert isfile(joinpath(dir, "weights.safetensors"))
        @assert isfile(joinpath(dir, "manifest.yaml"))
        println("bundle layout looks correct")

        backend = ReactantServer.ReactantBackend()
        pool = ReactantServer.resolve_client(backend,
            ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
        reg = ReactantServer.load_bundles([root])
        entry = ReactantServer.get_model(reg, "tiny_linear")
        entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)

        for b in (1, 4)
            x = randn(Float32, 4, b)
            yref = torch_to_julia(model(julia_to_torch(x)))
            out = ReactantServer.run_model(backend, pool, entry.executable,
                [ReactantServer.NamedTensor("x", x)])
            got = out[1].data
            @assert isapprox(got, yref; rtol=1e-4, atol=1e-5) "round-trip mismatch at batch=$b"
            println("batch=$b: got=$(vec(got))  ref=$(vec(yref))  OK")
        end
        println("\nSPIKE PASSED: torch.export -> torchax -> bundle -> server round trip is correct")
    end
end

main()
