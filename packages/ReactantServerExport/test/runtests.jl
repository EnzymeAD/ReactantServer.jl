# Round-trip tests for ReactantServerExport: export a model to a bundle, then load and run it
# through the ReactantServer runtime and compare to a native forward pass. Covers the Reactant
# tracing frontend (the former LuxExport, always run) and the PyTorch extension (run only when
# torch/torchax are importable; skipped gracefully otherwise). CPU only, no GPU.
#
#   julia --project=packages/ReactantServerExport/test packages/ReactantServerExport/test/runtests.jl

using Test
using Random

# Load PythonCall (and probe torch) BEFORE Reactant: torch's native libraries must initialize
# ahead of Reactant's MLIR/LLVM to avoid a static-initialization SIGSEGV. The PyTorch export
# methods live in the PythonCall-triggered extension, so loading PythonCall here also enables
# them once ReactantServerExport is loaded below.
#
# Set REACTANTSERVER_SKIP_PYTORCH=true to skip the PyTorch path without loading PythonCall at all,
# so no conda environment is provisioned. The `export-lux` CI job sets this for a fast Lux-only
# run; `export-pytorch` leaves it unset to exercise the full round-trip. Default (unset) keeps the
# original behavior: load PythonCall and run the torch tests when torch is importable.
const SKIP_PYTORCH = get(ENV, "REACTANTSERVER_SKIP_PYTORCH", "false") == "true"

# Keep the load in its own top-level statement (via @eval, since `using` cannot be nested inside an
# expression) so the methods are visible in the next world age when we probe below. Loading
# PythonCall here also provisions the CondaPkg environment declared in CondaPkg.toml.
SKIP_PYTORCH || @eval using PythonCall, CondaPkg

# Force the CPU build of PyTorch. The default PyPI torch wheel on Linux is a CUDA build whose
# libtorch_cuda.so clashes with Reactant/XLA and segfaults on a CPU-only host. CondaPkg cannot pin
# a per-package index, so we let it resolve torch (and torchax) normally from CondaPkg.toml, then
# reinstall the resolved torch as its CPU variant from the PyTorch CPU index (this mirrors the
# workaround used in Reactant.jl). The round-trip runs entirely on CPU, so the CPU build is correct
# regardless of host. The resolved version is read from package metadata, not by importing torch,
# because the CUDA build may fail to import; the matching CPU wheel is then force-reinstalled with
# --no-deps since torch's dependencies are already present. The pixi/CondaPkg env ships no pip, so
# ensurepip provisions it first.
if !SKIP_PYTORCH
    CondaPkg.withenv() do
        run(`python -m ensurepip --upgrade`)
        torch_version = readchomp(
            `python -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])"`,
        )
        run(
            `python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps --force-reinstall "torch==$(torch_version)"`,
        )
        # The CUDA torch resolve also pulled in triton, whose bundled libtriton.so statically links
        # an LLVM that collides with Reactant/XLA and segfaults when torch's export path imports it.
        # The CPU build does not need triton, so remove it (pip exits 0 if it is already absent).
        run(`python -m pip uninstall -y triton pytorch-triton`)
    end
end

const HAS_TORCH = if SKIP_PYTORCH
    @info "Skipping PyTorch export tests: REACTANTSERVER_SKIP_PYTORCH is set"
    false
else
    try
        pyimport("torch")
        pyimport("torch.export")
        pyimport("torchax.export")
        pyimport("numpy")
        true
    catch err
        @info "Skipping PyTorch export tests: required Python module not importable" error = err
        false
    end
end

using ReactantServerExport
using ReactantServer
using Lux

# Load a bundle and run it through the ReactantServer runtime (CPU backend).
function run_bundle(root, name, inputs::Vector{<:Pair})
    backend = ReactantServer.ReactantBackend()
    pool = ReactantServer.resolve_client(backend,
        ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
    reg = ReactantServer.load_bundles([root])
    entry = ReactantServer.get_model(reg, name)
    entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)
    tensors = [ReactantServer.NamedTensor(String(first(p)), last(p)) for p in inputs]
    return ReactantServer.run_model(backend, pool, entry.executable, tensors)
end

_load_manifest(dir) = ReactantServer.parse_manifest(
    ReactantServer.YAML.load_file(joinpath(dir, "manifest.yaml"); dicttype = Dict{String,Any}))

@testset "ReactantServerExport" begin
    @testset "Reactant model -> bundle -> server (multi-batch)" begin
        rng = Random.Xoshiro(0)
        model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
        ps, st = Lux.setup(rng, model)

        mktempdir() do root
            example = randn(Float32, 4, 1)            # (features, batch); batch is the last Julia axis
            export_bundle(:lux, model, ps, st, example;
                dir = joinpath(root, "mlp"), name = "mlp", batch_sizes = [1, 4])

            @test isfile(joinpath(root, "mlp", "model.b1.mlir"))
            @test isfile(joinpath(root, "mlp", "model.b4.mlir"))
            @test isfile(joinpath(root, "mlp", "weights.safetensors"))
            @test isfile(joinpath(root, "mlp", "manifest.yaml"))

            man = _load_manifest(joinpath(root, "mlp"))
            @test man.name == "mlp"
            @test man.batching.compiled_batch_sizes == [1, 4]
            @test man.executable_inputs[1].shape[end] == ReactantServer.Dim(ReactantServer.BATCH)
            @test man.input_batch_dim == ndims(randn(Float32, 4, 1)) - 1

            for b in (1, 4)
                x = randn(Float32, 4, b)
                yref = first(model(x, ps, st))        # (3, b) Julia
                out = run_bundle(root, "mlp", ["input" => x])
                @test length(out) == 1
                @test isapprox(out[1].data, yref; rtol = 1e-4, atol = 1e-5)
            end
        end
    end

    @testset "generic Reactant function -> bundle -> server" begin
        g(x, W, b) = W * x .+ b
        W = Float32[1 0 0; 0 1 0]
        bvec = Float32[10, 20]
        mktempdir() do root
            x0 = reshape(collect(Float32, 1:3), 3, 1)
            export_bundle(:reactant, g, (x0,), ["W" => W, "b" => bvec];
                dir = joinpath(root, "affine"), name = "affine", input_names = ["x"])
            @test isfile(joinpath(root, "affine", "model.mlir"))

            x = reshape(Float32[2, 3, 4], 3, 1)
            out = run_bundle(root, "affine", ["x" => x])
            @test isapprox(vec(out[1].data), vec(W * x .+ bvec); rtol = 1e-5)
        end
    end

    if HAS_TORCH
        np = pyimport("numpy")
        torch = pyimport("torch")

        # PyTorch (row-major) tensor -> Julia col-major Array with reversed shape (same bytes).
        function torch_to_julia(py_tensor)
            np_arr = py_tensor.detach().cpu().contiguous().numpy()
            T = ReactantServerExport._numpy_dtype_to_julia(pyconvert(String, np_arr.dtype.name))
            return ReactantServerExport._numpy_to_julia(np_arr, T)
        end

        pyexec("""
import torch

class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 3)
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

class UInt8In(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x):
        xf = x.to(torch.float32) / 255.0 - 0.5
        return self.fc(xf)

# TinyConv exercises the aten._convolution.default delegation patch in the TorchScript path.
class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(4 * 4 * 4, 2)
    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.fc(h.flatten(1))
""", @__MODULE__)

        @testset "PyTorch nn.Module -> bundle -> server (Float32, multi-batch)" begin
            torch.manual_seed(0)
            model = pyeval("TinyMLP()", @__MODULE__)

            mktempdir() do root
                example = randn(Float32, 4, 1)   # (features, batch) Julia = (batch, features) PyTorch
                export_bundle(:pytorch, model, (example,);
                    dir = joinpath(root, "mlp"), name = "mlp",
                    input_names = ["input"], batch_sizes = [1, 4])

                @test isfile(joinpath(root, "mlp", "model.b1.mlir"))
                @test isfile(joinpath(root, "mlp", "model.b4.mlir"))

                man = _load_manifest(joinpath(root, "mlp"))
                @test man.batching.compiled_batch_sizes == [1, 4]
                @test man.executable_inputs[1].shape[end] == ReactantServer.Dim(ReactantServer.BATCH)
                @test man.input_batch_dim == 1
                @test man.executable_inputs[1].dtype == ReactantServer.F32

                for b in (1, 4)
                    x = randn(Float32, 4, b)
                    py_x = torch.from_numpy(np.frombuffer(
                        pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float32"
                    ).reshape(reverse(size(x))...).copy())
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "mlp", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1e-4, atol = 1e-5)
                end
            end
        end

        @testset "PyTorch nn.Module -> bundle -> server (Float64)" begin
            torch.manual_seed(0)
            model = pyeval("TinyMLP()", @__MODULE__)
            model.double()
            model.eval()

            mktempdir() do root
                example = randn(Float64, 4, 1)
                export_bundle(:pytorch, model, (example,);
                    dir = joinpath(root, "mlp64"), name = "mlp64",
                    input_names = ["input"], batch_sizes = [1, 4])

                man = _load_manifest(joinpath(root, "mlp64"))
                @test man.executable_inputs[1].dtype == ReactantServer.F64
                @test man.executable_outputs[1].dtype == ReactantServer.F64

                for b in (1, 4)
                    x = randn(Float64, 4, b)
                    py_x = torch.from_numpy(np.frombuffer(
                        pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float64"
                    ).reshape(reverse(size(x))...).copy())
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "mlp64", ["input" => x])
                    @test length(out) == 1
                    @test eltype(out[1].data) == Float64
                    @test isapprox(out[1].data, yref; rtol = 1e-12, atol = 1e-12)
                end
            end
        end

        @testset "PyTorch nn.Module -> bundle -> server (UInt8 input cast)" begin
            torch.manual_seed(1)
            model = pyeval("UInt8In()", @__MODULE__)

            mktempdir() do root
                example = zeros(UInt8, 4, 1)
                export_bundle(:pytorch, model, (example,);
                    dir = joinpath(root, "u8m"), name = "u8m",
                    input_names = ["input"], batch_sizes = [1, 2])

                man = _load_manifest(joinpath(root, "u8m"))
                @test man.executable_inputs[1].dtype == ReactantServer.U8
                @test man.executable_outputs[1].dtype == ReactantServer.F32
                @test ReactantServerExport.DTYPE_TOKENS[UInt8] == "u8"

                for b in (1, 2)
                    x = rand(UInt8, 4, b)
                    py_x = torch.from_numpy(np.frombuffer(
                        pybytes(Vector{UInt8}(vec(x))), dtype = "uint8"
                    ).reshape(reverse(size(x))...).copy())
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "u8m", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1e-4, atol = 1e-5)
                end
            end
        end

        @testset "PyTorch TorchScript .pt -> bundle -> server" begin
            torch.manual_seed(2)
            model = pyeval("TinyConv()", @__MODULE__)
            model.eval()

            mktempdir() do root
                pt_path = joinpath(root, "tinyconv.pt")
                trace_example = torch.zeros(1, 3, 4, 4)
                scripted = torch.jit.trace(model, trace_example)
                scripted.save(pt_path)
                jit_model = torch.jit.load(pt_path, map_location = "cpu")
                jit_model.eval()

                example = zeros(Float32, 4, 4, 3, 1)   # (W, H, C, batch) Julia = (batch, C, H, W) PyTorch
                export_torchscript_bundle(pt_path, (example,);
                    dir = joinpath(root, "tinyconv"), name = "tinyconv",
                    input_names = ["input"], batch_sizes = [1, 2])

                @test isfile(joinpath(root, "tinyconv", "model.b1.mlir"))
                @test isfile(joinpath(root, "tinyconv", "model.b2.mlir"))

                man = _load_manifest(joinpath(root, "tinyconv"))
                @test man.provenance.fields["source_subframework"] == "torchscript"
                @test man.provenance.fields["torchscript_path"] == pt_path

                for b in (1, 2)
                    x = randn(Float32, 4, 4, 3, b)
                    py_x = torch.from_numpy(np.frombuffer(
                        pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float32"
                    ).reshape(reverse(size(x))...).copy())
                    yref = torch_to_julia(jit_model(py_x))

                    out = run_bundle(root, "tinyconv", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1e-4, atol = 1e-5)
                end
            end
        end
    end
end
