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
# `Reactant` for building a device-resident value in the entry-arity test below. Imported HERE and
# not at the top: the PythonCall block above must initialize torch's native libraries before
# Reactant's MLIR/LLVM, and this line is past it.
using Reactant

# Load a bundle and run it through the ReactantServer runtime (CPU backend).
function run_bundle(root, name, inputs::Vector{<:Pair})
    backend = ReactantServer.ReactantBackend()
    pool = ReactantServer.resolve_client(
        backend,
        ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
    )
    reg = ReactantServer.load_bundles([root])
    entry = ReactantServer.get_model(reg, name)
    entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)
    tensors = [ReactantServer.NamedTensor(String(first(p)), last(p)) for p in inputs]
    return ReactantServer.run_model(backend, pool, entry.executable, tensors)
end

_load_manifest(dir) = ReactantServer.parse_manifest(
    ReactantServer.YAML.load_file(joinpath(dir, "manifest.yaml"); dicttype = Dict{String, Any})
)

@testset "ReactantServerExport" begin
    @testset "collect_provenance" begin
        # Inside a git work tree: all fields present.
        mktempdir() do repo
            run(`git -C $repo init -q`)
            write(joinpath(repo, "f.txt"), "x")
            run(`git -C $repo add f.txt`)
            run(`git -C $repo -c user.name=t -c user.email=t@t commit -qm init`)
            prov = collect_provenance(repo; extra = Dict("model" => "M"))
            for k in (
                    "exported_at", "julia_version", "reactantserverexport_version",
                    "git_commit", "git_tree_sha1", "git_branch", "git_dirty",
                )
                @test haskey(prov, k)
            end
            @test prov["model"] == "M"
            @test prov["git_dirty"] == false
            @test length(prov["git_commit"]) == 40
            @test length(prov["git_tree_sha1"]) == 40
            @test !haskey(prov, "repo_remote")   # no origin in a fresh init
            @test !haskey(prov, "git_diff")      # clean tree carries no patch
            write(joinpath(repo, "f.txt"), "y")
            dirty = collect_provenance(repo)
            @test dirty["git_dirty"] == true
            @test occursin("f.txt", dirty["git_diff"])
            @test endswith(dirty["git_diff"], "\n")   # patch must keep its trailing newline
        end
        # Outside a work tree: git fields omitted, base fields still present, no throw.
        mktempdir() do plain
            prov = collect_provenance(plain)
            @test haskey(prov, "exported_at")
            @test haskey(prov, "julia_version")
            @test !haskey(prov, "git_commit")
        end
        # `extra` overrides a collected field.
        @test collect_provenance(; extra = Dict("julia_version" => "override"))["julia_version"] == "override"
    end

    @testset "weights layout" begin
        W = Float32[1 2 3; 4 5 6]
        V = reshape(Float32.(1:24), 2, 3, 4)
        b = Float32[1, 2]
        mlir = "module {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    return %x : tensor<2xf32>\n  }\n}\n"
        program = ["W" => permutedims(W), "V" => permutedims(V, (3, 2, 1)), "b" => b]
        for layout in (:xla, :legacy)
            mktempdir() do root
                dir = joinpath(root, "lay")
                write_bundle(
                    dir; name = "lay", executable_inputs = [IOSpec("x", Float32, [2])],
                    executable_outputs = [IOSpec("y", Float32, [2])], modules = Dict(0 => mlir),
                    weights = ["W" => W, "V" => V, "b" => b], layout
                )
                @test ReactantServer.YAML.load_file(joinpath(dir, "manifest.yaml"))["format_version"] == (layout === :xla ? "2.1" : "2.0")
                w, _ = ReactantServer.open_weights_file(joinpath(dir, "weights.safetensors"))
                @test w.xla_order == (layout === :xla)
                for (n, a) in ("W" => W, "V" => V, "b" => b)
                    @test ReactantServer.read_weight(w, n) == a                 # what the server uploads
                    layout === :xla &&                                          # stored unpermuted
                        @test ReactantServer._read_raw(w, w.tensors[n]) == vec(a)
                end
                ReactantServer.close_weights!(w)
                @test read_bundle_weights(dir) == program
            end
        end
        @test_throws "layout must be" write_bundle(
            mktempdir(); name = "x", executable_inputs = IOSpec[], executable_outputs = IOSpec[],
            modules = Dict(0 => mlir), weights = Pair{String, Any}[], layout = :bogus
        )
    end

    @testset "Reactant model -> bundle -> server (multi-batch)" begin
        rng = Random.Xoshiro(0)
        model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
        ps, st = Lux.setup(rng, model)

        mktempdir() do root
            example = randn(Float32, 4, 1)            # (features, batch); batch is the last Julia axis
            export_bundle(
                :lux, model, ps, st, example;
                dir = joinpath(root, "mlp"), name = "mlp", batch_sizes = [1, 4],
                provenance = Dict{String, Any}("git_diff" => "--- fake patch\n")
            )

            # write_bundle extracts git_diff into a patch file instead of the manifest.
            @test read(joinpath(root, "mlp", "working_tree.patch"), String) == "--- fake patch\n"
            manifest_text = read(joinpath(root, "mlp", "manifest.yaml"), String)
            @test occursin("git_diff_file", manifest_text)
            @test !occursin("fake patch", manifest_text)

            @test isfile(joinpath(root, "mlp", "model.b1.mlir"))
            @test isfile(joinpath(root, "mlp", "model.b4.mlir"))
            @test isfile(joinpath(root, "mlp", "weights.safetensors"))
            @test isfile(joinpath(root, "mlp", "manifest.yaml"))

            man = _load_manifest(joinpath(root, "mlp"))
            @test man.name == "mlp"
            @test man.batching.compiled_batch_sizes == [1, 4]
            @test occursin("reactant_version", read(joinpath(root, "mlp", "manifest.yaml"), String))
            @test man.executable_inputs[1].shape[end] == ReactantServer.Dim(ReactantServer.BATCH)
            @test man.input_batch_dim == ndims(randn(Float32, 4, 1)) - 1

            for b in (1, 4)
                x = randn(Float32, 4, b)
                yref = first(model(x, ps, st))        # (3, b) Julia
                out = run_bundle(root, "mlp", ["input" => x])
                @test length(out) == 1
                @test isapprox(out[1].data, yref; rtol = 1.0e-4, atol = 1.0e-5)
            end
        end
    end

    @testset "generic Reactant function -> bundle -> server" begin
        g(x, W, b) = W * x .+ b
        W = Float32[1 0 0; 0 1 0]
        bvec = Float32[10, 20]
        for layout in (:xla, :legacy)                  # the frontend passes `layout` through
            mktempdir() do root
                x0 = reshape(collect(Float32, 1:3), 3, 1)
                export_bundle(
                    :reactant, g, (x0,), ["W" => W, "b" => bvec];
                    dir = joinpath(root, "affine"), name = "affine", input_names = ["x"], layout
                )
                @test isfile(joinpath(root, "affine", "model.mlir"))
                @test _load_manifest(joinpath(root, "affine")).format_version == (layout === :xla ? "2.1" : "2.0")

                x = reshape(Float32[2, 3, 4], 3, 1)
                out = run_bundle(root, "affine", ["x" => x])
                @test isapprox(vec(out[1].data), vec(W * x .+ bvec); rtol = 1.0e-5)
            end
        end
    end


    @testset "compile options reach the traced program" begin
        g(x, W, b) = tanh.(W * x .+ b) .* 2
        W = Float32[1 0 0; 0 1 0]
        bvec = Float32[10, 20]
        x0 = reshape(collect(Float32, 1:3), 3, 1)
        x = reshape(Float32[2, 3, 4], 3, 1)
        mktempdir() do root
            export_with(name, opts) = export_bundle(
                :reactant, g, (x0,), ["W" => W, "b" => bvec];
                dir = joinpath(root, name), name, input_names = ["x"], compile_options = opts
            )
            mlir(name) = read(joinpath(root, name, "model.mlir"))
            export_with("dflt", (;))
            export_with("raised", (; raise = true))
            export_with("raised_obj", Reactant.CompileOptions(; raise = true))
            # Skipping Reactant's passes leaves its internal ops in the program: refused at export,
            # which also shows the option reached tracing, by keywords or as an object.
            @test_throws "UNSERVABLE" export_with("noopt", (; optimize = false))
            @test_throws "enzyme." export_with("noopt", Reactant.CompileOptions(; optimization_passes = false))
            @test !isdir(joinpath(root, "noopt"))
            @test mlir("raised_obj") == mlir("raised")              # an object means the same thing
            for name in ("dflt", "raised", "raised_obj")
                out = run_bundle(root, name, ["x" => x])
                @test isapprox(vec(out[1].data), vec(tanh.(W * x .+ bvec) .* 2); rtol = 1.0e-5)
            end
            prov(name) = get(_load_manifest(joinpath(root, name)).provenance.fields, "compile_options", nothing)
            @test prov("dflt") === nothing                          # defaults are not recorded
            @test prov("raised") == Dict("raise" => "true")
            @test prov("raised_obj") isa AbstractString
            @test_throws "unknown compile option `bogus`" export_with("bad", (; bogus = 1))
            @test_throws "set by the exporter" export_with("bad", (; client = nothing))
            @test_throws "compile_options must be" export_with("bad", Dict(:raise => true))
            @test !isdir(joinpath(root, "bad"))                     # refused before anything is written
        end
    end

    @testset "a Julia callback is refused before anything is written" begin
        double!(out, x) = (out .= 2 .* x; nothing)
        # The frontend also runs `g` on host arrays, where there is nothing to call back into.
        cb(y::Reactant.TracedRArray) = Reactant.Ops.julia_callback(double!, ((Float32, size(y)),), y)
        cb(y::AbstractArray) = 2 .* y
        g(x, W) = cb(W * x)
        W = Float32[1 0 0; 0 1 0]
        x0 = reshape(collect(Float32, 1:3), 3, 1)
        nofiles(dir) = !isfile(joinpath(dir, "weights.safetensors")) &&
            (!isdir(dir) || !any(endswith(".mlir"), readdir(dir)))

        # The op survives Reactant's default passes, and the exporter's attribute dropping, with
        # its call target intact.
        ctx = Reactant.ReactantContext()
        mod, _ = Reactant.Compiler.compile_mlir(
            ctx, g, (Reactant.to_rarray(x0), Reactant.to_rarray(W)); drop_unsupported_attributes = true
        )
        text, bytes = GC.@preserve ctx begin
            @test ReactantServerExport._has_julia_callback(mod)
            string(mod), ReactantServerExport._serialize_module(mod)   # the second lowers `mod` to VHLO
        end
        @test ReactantServerExport._has_julia_callback(text)
        @test ReactantServerExport._has_julia_callback(bytes)

        mktempdir() do root
            dir = joinpath(root, "cb")
            @test_throws "reactant_julia_callback" export_bundle(
                :reactant, g, (x0,), ["W" => W]; dir, name = "cb", input_names = ["x"]
            )
            @test nofiles(dir)
            spec(n) = [IOSpec(n, Float32, [3, 1]; batch_axis = 1)]
            for m in (text, bytes)
                err = try
                    write_bundle(
                        dir; name = "cb", executable_inputs = spec("x"), executable_outputs = spec("y"),
                        modules = Dict(4 => m), weights = ["W" => W]
                    )
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("batch size 4", err.msg)
                @test occursin("Ops.julia_callback", err.msg)
                @test nofiles(dir)
            end
        end
    end

    @testset "entry-arity gate refuses an unservable bundle before writing it" begin
        # THE 2026-08-11 AND 2026-08-19 DEFECT, reproduced. `export_bundle(:lux, ...)` CAPTURES `st`
        # in the closure it compiles rather than passing it, and Reactant lifts every device-resident
        # value reachable from a traced callable into an MLIR argument whether the program reads it or
        # not. The `st` below stands in for the real case, which is `st.<layer>.rng.seed` on any Lux
        # model carrying a `Dropout`: two `UInt64`s, never read in eval mode, and the difference
        # between a servable bundle and one that fails every inference.
        lifting_model(x, ps, st) = (ps.W * x, st)
        W = Float32[1 0 0; 0 1 0]
        st_dev = (; seed = Reactant.to_rarray(UInt64[1, 2]))

        mktempdir() do root
            dir = joinpath(root, "lifted")
            err = try
                export_bundle(
                    :lux, lifting_model, (; W = W), st_dev, randn(Float32, 3, 1);
                    dir = dir, name = "lifted"
                )
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("UNSERVABLE", err.msg)
            # x + W + the lifted seed = 3, against one declared input plus one weight.
            @test occursin("compiled entry arguments : 3", err.msg)
            @test occursin("expected                 : 2", err.msg)
            @test occursin("rng", err.msg)                 # the message names the known cause
            # Refusing BEFORE the write is the point: the old failure wrote a complete-looking bundle
            # and was diagnosed from a server that could report two numbers and no cause.
            @test !isfile(joinpath(dir, "weights.safetensors"))
            @test !isfile(joinpath(dir, "manifest.yaml"))
        end
    end

    @testset "auditing a bundle that is already on disk" begin
        rng = Random.Xoshiro(0)
        model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
        ps, st = Lux.setup(rng, model)

        mktempdir() do root
            dir = joinpath(root, "audited")
            export_bundle(
                :lux, model, ps, st, randn(Float32, 4, 1);
                dir = dir, name = "audited", batch_sizes = [1, 4]
            )

            r = bundle_arity_report(dir)
            @test r.servable
            @test r.name == "audited"
            @test r.n_inputs == 1
            @test r.n_weights == 4                       # two Dense layers, weight and bias each
            @test r.expected == r.n_inputs + r.n_weights
            @test length(r.modules) == 2                 # one per compiled batch size
            @test all(m -> m.entry_args == r.expected, r.modules)
            @test bundle_entry_arity(joinpath(dir, "model.b1.mlir")) == r.expected
            @test assert_bundle_arity(dir).servable

            # A bundle whose graph and manifest disagree is exactly what neither readable half of a
            # bundle shows, so give the manifest a second declared input and check the audit notices.
            # This is the shape of the incident, reached from the artifact rather than from a trace.
            man = joinpath(dir, "manifest.yaml")
            m = ReactantServer.YAML.load_file(man)
            push!(m["executable_inputs"], deepcopy(m["executable_inputs"][1]))
            ReactantServer.YAML.write_file(man, m)

            bad = bundle_arity_report(dir)
            @test !bad.servable
            @test bad.expected == r.expected + 1
            e = try
                assert_bundle_arity(dir)
                nothing
            catch err
                err
            end
            @test e isa ErrorException
            @test occursin("UNSERVABLE", e.msg)
            @test occursin("audited", e.msg)
        end
    end

    include("test_detection.jl")

    if HAS_TORCH
        np = pyimport("numpy")
        torch = pyimport("torch")

        # PyTorch (row-major) tensor -> Julia col-major Array with reversed shape (same bytes).
        function torch_to_julia(py_tensor)
            np_arr = py_tensor.detach().cpu().contiguous().numpy()
            T = ReactantServerExport._numpy_dtype_to_julia(pyconvert(String, np_arr.dtype.name))
            return ReactantServerExport._numpy_to_julia(np_arr, T)
        end

        pyexec(
            """
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
            """, @__MODULE__
        )

        @testset "PyTorch nn.Module -> bundle -> server (Float32, multi-batch)" begin
            torch.manual_seed(0)
            model = pyeval("TinyMLP()", @__MODULE__)

            mktempdir() do root
                example = randn(Float32, 4, 1)   # (features, batch) Julia = (batch, features) PyTorch
                export_bundle(
                    :pytorch, model, (example,);
                    dir = joinpath(root, "mlp"), name = "mlp",
                    input_names = ["input"], batch_sizes = [1, 4]
                )

                @test isfile(joinpath(root, "mlp", "model.b1.mlir"))
                @test isfile(joinpath(root, "mlp", "model.b4.mlir"))

                man = _load_manifest(joinpath(root, "mlp"))
                @test man.batching.compiled_batch_sizes == [1, 4]
                @test man.executable_inputs[1].shape[end] == ReactantServer.Dim(ReactantServer.BATCH)
                @test man.input_batch_dim == 1
                @test man.executable_inputs[1].dtype == ReactantServer.F32

                for b in (1, 4)
                    x = randn(Float32, 4, b)
                    py_x = torch.from_numpy(
                        np.frombuffer(
                            pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float32"
                        ).reshape(reverse(size(x))...).copy()
                    )
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "mlp", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1.0e-4, atol = 1.0e-5)
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
                export_bundle(
                    :pytorch, model, (example,);
                    dir = joinpath(root, "mlp64"), name = "mlp64",
                    input_names = ["input"], batch_sizes = [1, 4]
                )

                man = _load_manifest(joinpath(root, "mlp64"))
                @test man.executable_inputs[1].dtype == ReactantServer.F64
                @test man.executable_outputs[1].dtype == ReactantServer.F64

                for b in (1, 4)
                    x = randn(Float64, 4, b)
                    py_x = torch.from_numpy(
                        np.frombuffer(
                            pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float64"
                        ).reshape(reverse(size(x))...).copy()
                    )
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "mlp64", ["input" => x])
                    @test length(out) == 1
                    @test eltype(out[1].data) == Float64
                    @test isapprox(out[1].data, yref; rtol = 1.0e-12, atol = 1.0e-12)
                end
            end
        end

        @testset "PyTorch nn.Module -> bundle -> server (UInt8 input cast)" begin
            torch.manual_seed(1)
            model = pyeval("UInt8In()", @__MODULE__)

            mktempdir() do root
                example = zeros(UInt8, 4, 1)
                export_bundle(
                    :pytorch, model, (example,);
                    dir = joinpath(root, "u8m"), name = "u8m",
                    input_names = ["input"], batch_sizes = [1, 2]
                )

                man = _load_manifest(joinpath(root, "u8m"))
                @test man.executable_inputs[1].dtype == ReactantServer.U8
                @test man.executable_outputs[1].dtype == ReactantServer.F32
                @test ReactantServerExport.DTYPE_TOKENS[UInt8] == "u8"

                for b in (1, 2)
                    x = rand(UInt8, 4, b)
                    py_x = torch.from_numpy(
                        np.frombuffer(
                            pybytes(Vector{UInt8}(vec(x))), dtype = "uint8"
                        ).reshape(reverse(size(x))...).copy()
                    )
                    yref = torch_to_julia(model(py_x))

                    out = run_bundle(root, "u8m", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1.0e-4, atol = 1.0e-5)
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
                export_bundle(
                    :torchscript, pt_path, (example,);
                    dir = joinpath(root, "tinyconv"), name = "tinyconv",
                    input_names = ["input"], batch_sizes = [1, 2]
                )

                @test isfile(joinpath(root, "tinyconv", "model.b1.mlir"))
                @test isfile(joinpath(root, "tinyconv", "model.b2.mlir"))

                man = _load_manifest(joinpath(root, "tinyconv"))
                @test man.provenance.fields["source_subframework"] == "torchscript"
                @test man.provenance.fields["torchscript_path"] == pt_path

                for b in (1, 2)
                    x = randn(Float32, 4, 4, 3, b)
                    py_x = torch.from_numpy(
                        np.frombuffer(
                            pybytes(Vector{UInt8}(reinterpret(UInt8, vec(x)))), dtype = "float32"
                        ).reshape(reverse(size(x))...).copy()
                    )
                    yref = torch_to_julia(jit_model(py_x))

                    out = run_bundle(root, "tinyconv", ["input" => x])
                    @test length(out) == 1
                    @test isapprox(out[1].data, yref; rtol = 1.0e-4, atol = 1.0e-5)
                end
            end
        end
    end
end
