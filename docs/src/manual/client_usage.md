```@meta
CurrentModule = ReactantServerClient
```

# Client Usage

`ReactantServerClient` is the inference client for a `ReactantServer` worker or the
[gateway](../api/gateway.md). It builds KServe V2 requests against `ReactantServerCore`'s protobuf
messages and forwards them over gRPC. It depends only on `ReactantServerCore` and the gRPC
layer, so it carries **no Reactant/XLA dependency** and installs quickly on a plain client
machine. (The same KServe V2 wire protocol means any Triton/KServe gRPC client also works; this
package is the convenient Julia option.)

## Installation

The client is a workspace member. From this repository, activate its project:

```
julia --project=packages/ReactantServerClient -e 'using Pkg; Pkg.instantiate()'
```

You can confirm it pulls no Reactant:

```julia
using ReactantServerClient
@assert Base.identify_package(ReactantServerClient, "Reactant") === nothing
```

## One-shot inference

Point a [`KServeModel`](@ref) at a running server, build inputs with [`InferInput`](@ref),
call the one-shot [`infer_sync`](@ref), and read outputs with [`InferOutput`](@ref):

```julia
using ReactantServerClient

kserve_init()
try
    model = KServeModel("grpc://127.0.0.1:8080", "scale4"; max_batch_size = 1)
    x = Float32[1, 2, 3, 4]
    response = infer_sync(model, [InferInput("INPUT__0", x)])
    y = InferOutput("OUTPUT__0", response, Float32)
    @show vec(collect(y))
finally
    kserve_shutdown()
end
```

Run the snippet under the client project against a server you have started (for example a
worker on the e2e `scale4` bundle).

## Batched inference over a dataset

For many items, implement [`AbstractInferenceIO`](@ref) and use [`infer_async`](@ref) (concurrent)
or [`infer_sync`](@ref)`(model, io)` (serial). The driver stages each chunk through a shared
`ReactantServerCore.BufferPool`. The first call to each server probes it with the
`IsSameIPCNamespace` RPC: if the server confirms it shares the client's IPC namespace, inputs
travel over the system shared-memory data plane (zero extra copies on the wire); otherwise, or if
the server does not implement the probe, the driver uses inline transport. The `shared_memory`
keyword on [`KServeModel`](@ref) controls this: `:auto` (default) probes and uses shared memory
when available, `:on` forces it (and errors loudly if the server is in a different namespace), and
`:off` always sends inline. There is no silent runtime fallback once shared memory is chosen. Your
IO implements `length`,
`item_input_bytes`, `infer_encode_chunk!` (request the chunk's input buffers with `scratch`, write
into them, and return them tagged with their tensor names), and `infer_decode_chunk!` (consume the
response). The pool's fixed-slot allocator hands every chunk a disjoint slot, so concurrent
`infer_async` calls are safe.

`kserve_init(; pool_bytes, n_slots)` sizes the staging pool and the number of slots
(the dispatch concurrency); `kserve_shutdown()` tears it down.

A concrete IO for a model with input `INPUT__0` and output `OUTPUT__0`, each a length-4 Float32
vector per item:

```julia
using ReactantServerClient

struct VectorIO <: AbstractInferenceIO
    inputs::Vector{Vector{Float32}}    # each item: 4 Float32
    outputs::Vector{Vector{Float32}}   # filled in by infer_decode_chunk!
end

Base.length(io::VectorIO) = length(io.inputs)

# Bytes one item contributes to the request, summed over all inputs.
ReactantServerClient.item_input_bytes(::VectorIO) = 4 * sizeof(Float32)

function ReactantServerClient.infer_encode_chunk!(io::VectorIO, r::UnitRange, slot::PoolSlot)
    n = length(r)
    # Request all input buffers up front by name (Julia column-major: per-item dims, batch axis
    # last). `scratch` carves them from the chunk's slot and returns wire descriptors; write into
    # each with pool_view. The SHM-vs-inline transport is handled by the driver, so this is the same
    # code either way.
    input = scratch(slot, "INPUT__0", (4, n), Float32)  # single input: scalar form returns one descriptor
    buf = pool_view(input)                            # multiple inputs: feats, mask = pool_view(inputs...)
    @infer_inbounds for (k, i) in enumerate(r)        # hot path: elided normally, checked under validate_io
        buf[:, k] .= io.inputs[i]
    end
    return input                                      # a lone PoolInferInput is accepted; no vector needed
end

function ReactantServerClient.infer_decode_chunk!(io::VectorIO, r::UnitRange, response)
    out = InferOutput("OUTPUT__0", response, Float32) # Julia column-major (4, n)
    @infer_inbounds for (j, i) in enumerate(r)
        io.outputs[i] = collect(out[:, j])
    end
    return nothing
end

io = VectorIO([Float32[i, i, i, i] for i in 1:100], [Float32[] for _ in 1:100])
infer_async(model, io)                                # results land in io.outputs
```

`scratch` is the same buffer-request interface meta models use through `call.scratch` (see the meta
models manual). The lower-level path is still supported: carve a `subslot`, `pool_view` it, and
return an `InferInput(name, sub, shape, T)` descriptor; both forms may be mixed in the returned
vector.

## Validating an IO against a model

Because an IO hard-codes a model's tensor names, dtypes, and shapes, it can silently drift from the
served model. [`validate_io`](@ref) dry-runs an IO against the model's true I/O spec without sending
an inference request: it runs `infer_encode_chunk!` and `infer_decode_chunk!` once over synthetic,
spec-shaped data and surfaces name, dtype, shape, and indexing mismatches as errors. Get the spec
from [`model_io_spec`](@ref)`(model)` against a running server, or from
[`manifest_io_spec`](@ref)`(path)` offline against a model's `manifest.yaml` (suitable for a
build-time check). The harness runs your real methods, so call it on a representative or dummy IO.

```julia
validate_io(model, io)                            # online: fetches the spec from the server
validate_io(manifest_io_spec("manifest.yaml"), io)      # offline: no server needed
```

For example, validating the `VectorIO` above against a model whose `OUTPUT__0` is actually a
length-3 vector would report the shape mismatch, and an `infer_decode_chunk!` that indexed
`out[:, j]` past the real batch size would surface as an indexing error, both without sending a
request.

### Bounds checks in hot paths

`validate_io` can only catch an out-of-range index if the access is bounds-checked. Julia's
built-in `@inbounds` elides the check unconditionally, so an `@inbounds` access stays unchecked even
inside the dry run. If you want `@inbounds` performance in production but a robust check during
validation, write the access with [`@infer_inbounds`](@ref) instead, as the `VectorIO` example does.
It elides the check in normal use, but because `validate_io` runs your methods inside
[`with_bounds_checks`](@ref), an `@infer_inbounds` access is bounds-checked during the dry run and an
out-of-range index is caught rather than silently read. This is opt-in: using `@infer_inbounds` is
the user's responsibility, and bare `@inbounds` remains unchecked unless Julia is started with
`--check-bounds=yes` (which `Pkg.test` does).

```julia
# Bounds-checked under validate_io, elided in production:
@infer_inbounds buf[(4k - 3):(4k)] .= io.inputs[i]

# Force the checked behavior anywhere, not just in validate_io:
with_bounds_checks() do
    run_my_io_step()
end
```
