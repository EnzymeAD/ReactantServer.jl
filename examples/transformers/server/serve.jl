# Step 2 of 3: serve the exported bundles (blocks until Ctrl-C).
#
#   CUDA_VISIBLE_DEVICES=0 julia --project=examples/transformers/server examples/transformers/server/serve.jl
#   julia --project=examples/transformers/server examples/transformers/server/serve.jl --cpu   # GPU-free smoke test
#
# Serves splade, embedding, cross_encoder, and sentiment on 127.0.0.1:$TX_PORT (default 8080).
# Run the export step first. Leave this running and drive it from a second terminal with
# client/query.jl.

using ReactantServer

const BUNDLES = abspath(normpath(joinpath(@__DIR__, "..", "bundles")))
const EXPECTED = ("splade", "embedding", "cross_encoder", "sentiment")
all(m -> isdir(joinpath(BUNDLES, m)), EXPECTED) ||
    error("Missing bundles under $BUNDLES — run the export step first " *
          "(examples/transformers/export/export_stablehlo.jl).")

use_cpu = "--cpu" in ARGS
port = parse(Int, get(ENV, "TX_PORT", "8080"))
backend = use_cpu ? ReactantServer.CPU_BACKEND : ReactantServer.CUDA_BACKEND

cfg = ReactantServer.ServerConfig([BUNDLES], "",
    ReactantServer.RuntimeConfig(backend, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 64, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port))

@info "Compiling and serving transformer bundles (Ctrl-C to stop)" port backend models = EXPECTED bundles = BUNDLES
ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend())  # blocking
