# # Run Inference
#
# `ReactantServerClient` is a KServe V2 gRPC client with no Reactant or XLA dependency, so it
# installs quickly on any machine that needs to talk to a server. This page runs inference over a
# whole dataset: the client stages inputs through a buffer pool, sends many requests concurrently,
# and uses shared memory instead of the network when it shares a host with the server.
#
# ## Install
#
# A client machine needs only the client:
#
# ```julia
# using Pkg
# Pkg.add("ReactantServerClient")
# ```
#
# This page also starts a server to query, so it adds the server, the export tooling, and Lux too:
#
# ```julia
# Pkg.add(["ReactantServer", "ReactantServerExport", "Lux"])
# ```
#
# ## Setup: a server
#
# The client talks to any server, whichever way it was started. To keep this page self-contained,
# it exports the MLP from [Export a Model to a Bundle](export.md) and serves it from Julia as in
# [Serve a Model](../julia/serve.md). It takes 4 features and returns 3, and is compiled for
# batches of 1 and 8. If you already run the container from
# [Serve on One GPU](../container/single_gpu.md) with this bundle, skip this step: it listens on
# the same address.

using Lux, Random, ReactantServerExport, ReactantServer
using Test #src

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)
export_bundle(
    :lux, model, ps, st, randn(Float32, 4, 1);
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1, 8]
)
write(
    "node.yaml", """
    model_repo: $(abspath("models"))
    base_port: 8001
    metrics_base_port: 8002
    global:
      runtime: { backend: cpu }
      endpoints: { host: 127.0.0.1 }
    workers:
      - { name: worker0 }
    """
)
server = serve("node.yaml"; blocking = false)

# ## Describe your data as an inference IO
#
# Batched inference is driven by an [`AbstractInferenceIO`](@ref): a type that knows how many items
# there are, how to write a range of them into request buffers, and what to do with the response.
# The driver splits the dataset into chunks of at most `max_batch_size` items and calls back into
# the IO for each one.

using ReactantServerClient

struct FeatureIO <: AbstractInferenceIO
    inputs::Matrix{Float32}     # (4, items)
    outputs::Matrix{Float32}    # (3, items), filled in as responses arrive
end

Base.length(io::FeatureIO) = size(io.inputs, 2)

# The bytes one item adds to a request, summed over every input:
ReactantServerClient.item_input_bytes(::FeatureIO) = 4 * sizeof(Float32)

# `infer_encode_chunk!` writes items `r` into a buffer from the chunk's staging slot. [`scratch`](@ref)
# allocates the named input with the batch axis last, and `pool_view` is the array to fill:

function ReactantServerClient.infer_encode_chunk!(io::FeatureIO, r::UnitRange, slot::PoolSlot)
    input = scratch(slot, "input", (4, length(r)), Float32)
    pool_view(input) .= view(io.inputs, :, r)
    return input
end

# `infer_decode_chunk!` reads the response for the same items:

function ReactantServerClient.infer_decode_chunk!(io::FeatureIO, r::UnitRange, response)
    io.outputs[:, r] .= InferOutput("output", response, Float32)
    return nothing
end

# ## Run it
#
# [`kserve_init`](@ref) creates the staging pool. `max_batch_size` matches the bundle's largest
# compiled batch, so each request fills one execution:

kserve_init()
mlp = KServeModel("grpc://127.0.0.1:8001", "mlp"; max_batch_size = 8)

data = randn(Random.Xoshiro(1), Float32, 4, 100)
io = FeatureIO(data, zeros(Float32, 3, 100))
infer_async(mlp, io)

# [`infer_async`](@ref) sends the chunks concurrently and returns once every response is decoded;
# [`infer_sync`](@ref)`(mlp, io)` sends them one at a time. The results match the model run locally:

maximum(abs, io.outputs .- first(model(data, ps, st)))      # ≈ 0

@test io.outputs ≈ first(model(data, ps, st)) #src

# ## Shared memory
#
# On its first request to a server, the client asks whether the server can see its shared memory.
# When it can, as here with both in one process or a server on the same host, inputs travel through
# POSIX shared memory and only small descriptors cross the network. Otherwise they are sent inline.
# The `shared_memory` keyword of [`KServeModel`](@ref) picks: `:auto` (the default) probes, `:on`
# requires shared memory and fails loudly without it, and `:off` always sends inline. Both paths
# give the same results:

@test ReactantServerClient.is_shm_backed(ReactantServerClient.get_or_create_pool!(mlp)) #src

inline = KServeModel("grpc://127.0.0.1:8001", "mlp"; max_batch_size = 8, shared_memory = :off)
io_inline = FeatureIO(data, zeros(Float32, 3, 100))
infer_async(inline, io_inline)
io_inline.outputs == io.outputs                              # true

@test io_inline.outputs == io.outputs #src

# ## Check an IO against the model
#
# An IO hard-codes tensor names, dtypes, and shapes, so it can drift from the model it targets.
# [`validate_io`](@ref) runs its encode and decode methods once over synthetic data shaped by the
# model's real spec and reports any mismatch, without sending an inference request. It reads the
# spec from a running server, or offline from a bundle's manifest:

validate_io(mlp, FeatureIO(zeros(Float32, 4, 8), zeros(Float32, 3, 8)))
validate_io(
    manifest_io_spec(joinpath("models", "mlp", "manifest.yaml")),
    FeatureIO(zeros(Float32, 4, 8), zeros(Float32, 3, 8))
)

# ## Stop
#
# Close the client's connections, and stop the server if this page started it:

kserve_shutdown()
stop!(server)

# [Client Usage](../../client.md) is the reference for the rest: output tensors read back through shared
# memory, retries when a server sheds load, bounds checks under `validate_io`, and the lower-level
# buffer interface.
