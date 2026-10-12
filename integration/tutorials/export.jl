# # Export a Model to a Bundle
#
# ReactantServer serves *bundles*: directories that hold a compiled StableHLO program, its weights,
# and a manifest describing the tensors. This page builds a small Lux model and exports it into a
# bundle with `ReactantServerExport`. Export runs on the CPU and needs no server; the server never
# loads the export tooling or the model framework, only the bundle it writes.
#
# ## Install
#
# Add the export package and Lux to an environment:
#
# ```julia
# using Pkg
# Pkg.add(["ReactantServerExport", "Lux"])
# ```
#
# ## Build a model
#
# Any Reactant-traceable `model(x, ps, st)` works; Lux is just the convenient way to build one.
# This MLP maps 4 features to 3. Lux puts the batch on the last axis, so a batch of inputs is a
# `(4, batch)` matrix.

using Lux, Random, ReactantServerExport
using Test #src

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)
example = randn(Float32, 4, 1)                     # (features, batch)

# ## Export it
#
# `export_bundle(:lux, ...)` traces the model once per requested batch size and writes the bundle
# into `dir`. The bundle's directory name is the model's name on the server.

export_bundle(
    :lux, model, ps, st, example;
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1, 8]
)

# The bundle is a directory with a `manifest.yaml`, one compiled program per batch size
# (`model.b1.mlir` and `model.b8.mlir`), and the shared `weights.safetensors`. The manifest names the
# input `input` and the output `output` (override with `input_name` and `output_name`), records
# their dtypes and shapes, and lists the compiled batch sizes.

@test isfile(joinpath("models", "mlp", "manifest.yaml")) #src
@test isfile(joinpath("models", "mlp", "model.b1.mlir")) #src
@test isfile(joinpath("models", "mlp", "model.b8.mlir")) #src
@test isfile(joinpath("models", "mlp", "weights.safetensors")) #src

# A server answers requests of any size up to the largest compiled batch by padding a request to
# the next compiled size, and splits larger ones. Compiling more sizes trades startup compile time
# for less padding.
#
# ## Check the bundle
#
# A program must take exactly as many arguments as the manifest declares inputs plus the weights
# file holds tensors. `export_bundle` already refuses to write a bundle that disagrees, and
# `bundle_arity_report` reads the numbers back from any bundle, including one written elsewhere:

report = bundle_arity_report(joinpath("models", "mlp"))
report.servable                                    # true

@test report.servable #src

# `models/` is now a model repository: every subdirectory with a `manifest.yaml` is a servable
# model. Next, [serve it](serve.md). [Bundles](../../bundles.md) documents the manifest, custom pre-
# and postprocessing through a bundle's `model.jl`, and the multi-input and PyTorch frontends.
