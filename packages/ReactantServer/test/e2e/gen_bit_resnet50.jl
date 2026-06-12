# Generate the `bit_resnet50` bundle into test/e2e/models/bit_resnet50 for the end-to-end
# harness. Run with the e2e env:  julia --project=test/e2e test/e2e/gen_bit_resnet50.jl
# Idempotent: skips if the bundle is already present.
#
# The model is Luximm's ResNetV2 BiT 50x1 (:resnetv2_50x1_bit_goog_in21k) with random-init
# weights (pretrained checkpoints cannot be checked into the repo, and the e2e assertions are
# determinism and transport equivalence, not accuracy). The weights are seeded so regenerating
# the bundle is reproducible, but note the e2e checks never depend on specific values: both
# workers mount this one generated bundle.

using Luximm
using Lux
using Random
using ReactantServerExport

const MODELS_DIR = joinpath(@__DIR__, "models")
const BUNDLE = joinpath(MODELS_DIR, "bit_resnet50")

if isfile(joinpath(BUNDLE, "manifest.yaml")) && isfile(joinpath(BUNDLE, "weights.safetensors"))
    @info "bit_resnet50 bundle already present; skipping" dir = BUNDLE
else
    model = Luximm.create_model(:resnetv2_50x1_bit_goog_in21k)
    ps, st = Lux.setup(Xoshiro(0), model)
    st = Lux.testmode(st)
    example = randn(Xoshiro(1), Float32, 224, 224, 3, 1)   # Lux WHCN; batch is the last axis
    export_bundle(:lux, model, ps, st, example;
        dir = BUNDLE, name = "bit_resnet50", batch_sizes = [1],
        provenance = Dict("source" => "Luximm.jl :resnetv2_50x1_bit_goog_in21k (random init)"))
    @info "wrote bit_resnet50 bundle" dir = BUNDLE
end
