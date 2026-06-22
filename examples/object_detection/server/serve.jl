# Step 2 of 3: serve the exported bundles (blocks until Ctrl-C).
#
#   CUDA_VISIBLE_DEVICES=3 julia --project=examples/object_detection/server examples/object_detection/server/serve.jl
#   julia --project=examples/object_detection/server examples/object_detection/server/serve.jl --cpu   # GPU-free smoke test
#
# Serves object_detector on 127.0.0.1:$OD_PORT (default 8080). Run the export step first. Leave this
# running and drive it from a second terminal with client/detect.jl.

using ReactantServer

const BUNDLES = abspath(normpath(joinpath(@__DIR__, "..", "bundles")))
isdir(joinpath(BUNDLES, "object_detector")) ||
    error("No bundles at $BUNDLES — run the export step first (examples/object_detection/export/export.jl).")

use_cpu = "--cpu" in ARGS
port = parse(Int, get(ENV, "OD_PORT", "8080"))
backend = use_cpu ? ReactantServer.CPU_BACKEND : ReactantServer.CUDA_BACKEND

cfg = ReactantServer.ServerConfig([BUNDLES], "",
    ReactantServer.RuntimeConfig(backend, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 64, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port))

@info "Compiling and serving object_detector (Ctrl-C to stop)" port backend bundles = BUNDLES
ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend())  # blocking
