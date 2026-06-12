# Generate the tiny `scale4` bundle into test/e2e/models/scale4 for the end-to-end harness.
# Run from the repo root with the package env:  julia --project=. test/e2e/gen_scale4.jl
# Idempotent: skips if the bundle is already present. Reuses the test fixtures (write_bundle +
# stablehlo_artifact), which pull in Reactant to serialize the StableHLO portable artifact.

include(joinpath(@__DIR__, "..", "stablehlo_fixtures.jl"))

const MODELS_DIR = joinpath(@__DIR__, "models")
const BUNDLE = joinpath(MODELS_DIR, "scale4")

if isfile(joinpath(BUNDLE, "model.mlir")) && isfile(joinpath(BUNDLE, "weights.safetensors"))
    @info "scale4 bundle already present; skipping" dir = BUNDLE
else
    mkpath(MODELS_DIR)
    manifest = """
    format_version: "2.0"
    name: scale4
    executable_inputs:
      - {name: x, dtype: f32, shape: c, dims: {c: 4}}
    executable_outputs:
      - {name: y, dtype: f32, shape: c, dims: {c: 4}}
    batching: {compiled_batch_sizes: [1]}
    """
    mlir = """
    module {
      func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.multiply %x, %w : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """
    write_bundle(MODELS_DIR, "scale4"; manifest_yaml = manifest, mlir_text = mlir,
        weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"])
    @info "wrote scale4 bundle" dir = BUNDLE
end
