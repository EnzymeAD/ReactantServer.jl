# # Serve a Model
#
# This page serves a bundle with `ReactantServer` from a Julia session and sends it a request. It
# runs on the CPU, so it works on any machine; the same steps serve on a GPU by changing one line
# of the node file.
#
# ## Install
#
# Add the server, the client, and (for the setup step) the export tooling and Lux:
#
# ```julia
# using Pkg
# Pkg.add(["ReactantServer", "ReactantServerClient", "ReactantServerExport", "Lux"])
# ```
#
# Start Julia with `--threads=auto,1`. The server runs each request's pre- and postprocessing on the
# default thread pool and keeps its device dispatch loop on the interactive thread, so the two
# overlap. With a single thread the server still works, just without the overlap.
#
# ## Setup: a bundle
#
# The server needs a model repository: a directory whose subdirectories are bundles. This exports
# the small MLP from [Export a Model to a Bundle](../basics/export.md) into `models/mlp`:

using Lux, Random, ReactantServerExport
using Test #src

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)
export_bundle(
    :lux, model, ps, st, randn(Float32, 4, 1);
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1, 8]
)

# ## Describe the node
#
# A server is configured by a *node file*: the model repository, the ports, the runtime backend,
# and the workers. A single worker on the CPU needs only this:

write(
    "node.yaml", """
    model_repo: $(abspath("models"))
    base_port: 8001            # KServe V2 gRPC
    metrics_base_port: 8002    # /readyz, /healthz, /metrics
    global:
      runtime:
        backend: cpu           # cuda to serve on a GPU
      endpoints:
        host: 127.0.0.1
    workers:
      - { name: worker0 }
    """
)

# [Node Configuration](../../node_config.md) covers every key, including scheduler tuning, on-demand
# weights, and environment overrides.
#
# ## Start the server
#
# `serve` loads the node file, compiles every bundle in the repository, and then starts the gRPC
# server, so it accepts traffic only once its models are ready. With `blocking = false` it returns
# a handle instead of serving until interrupted:

using ReactantServer

server = serve("node.yaml"; blocking = false)

# Health and metrics are plain HTTP on the metrics port. `/readyz` answers 200 once the worker is
# serving:

using Downloads

Downloads.request("http://127.0.0.1:8002/readyz").status    # 200

@test Downloads.request("http://127.0.0.1:8002/readyz").status == 200 #src

# ## Send a request
#
# The server speaks the KServe V2 inference API over gRPC, so any KServe or Triton client can talk
# to it. `ReactantServerClient` is the Julia one. [`KServeModel`](@ref) binds a model on a server,
# [`InferInput`](@ref) names an input tensor, [`infer_sync`](@ref) sends one request, and
# [`InferOutput`](@ref) reads a named output back. Tensors keep Julia's layout on both ends, batch
# axis last, so one item for this model is a `(4, 1)` matrix:

using ReactantServerClient

kserve_init()
mlp = KServeModel("grpc://127.0.0.1:8001", "mlp")
x = reshape(Float32[1, 2, 3, 4], 4, 1)                      # (features, batch)
response = infer_sync(mlp, [InferInput("input", x)])
y = InferOutput("output", response, Float32)                # (3, 1)

# The served program is the traced Lux model, so it agrees with running the model in this session:

y_local = first(model(x, ps, st))
maximum(abs, y .- y_local)                                  # ≈ 0

@test y ≈ y_local #src

# This one-shot form suits occasional requests. [Query a Server](../basics/client.md) covers batched inference
# over a dataset and the shared-memory transport.
#
# ## Stop
#
# [`kserve_shutdown`](@ref) releases the client, and [`stop!`](@ref) shuts the server down:

kserve_shutdown()
stop!(server)

# From a shell, the blocking form serves until interrupted with Ctrl-C:
#
# ```sh
# julia --project --threads=auto,1 -e 'using ReactantServer; serve("node.yaml")'
# ```
#
# A single `serve` is one worker on one device. To run every GPU on a machine from one node file,
# with restarts and one client endpoint, use the node supervisor: [Run a Node](node.md).
