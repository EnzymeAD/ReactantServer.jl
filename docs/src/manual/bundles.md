```@meta
CurrentModule = ReactantServer
```

# Bundles & model.jl

A model is delivered to the server as a self-contained bundle: a directory holding a compiled
StableHLO program, its weights, and a manifest. Bundles are produced offline by the conversion
tooling and loaded at server startup.

## Bundle layout

A bundle is a directory containing:

- `manifest.yaml` — the metadata parsed into a [`Manifest`](@ref): I/O specs, dtypes, shapes,
  and the compiled batch sizes.
- `model.mlir` — a serialized StableHLO portable artifact (single batch size), or one module
  per size as `model.b{N}.mlir` sharing a single `weights.safetensors`.
- `weights.safetensors` — the model weights, memory-mapped at load time.
- `model.jl` — optional; registers custom pre/post-processing (see below).

The directory name is the model name and must match the manifest's `name`. Each immediate
subdirectory of `model_repo` that contains a `manifest.yaml` is a bundle.

## Manifest shape encoding

Each tensor's shape is an einsum-style string of single ASCII letters, one per axis, with a
companion `dims:` map giving the size of every non-batch letter:

```yaml
executable_inputs:
  - name: input
    dtype: f32
    shape: "chwn"     # channel, height, width, batch
    dims:
      c: 3
      h: 224
      w: 224
```

The letters `n` and `b` are reserved batch markers (at most one occurrence per tensor). Other
letters are tensor-scoped (no implicit cross-tensor equality) and must be unique within a
single shape. A size of `-1` in `dims` marks a [variable axis](@ref Dim) (used today only in
`client_outputs` that pass through `model.jl`). The per-input batch axis is derived from the
position of `n`/`b`; at inference the request's size along that axis must equal one of
`batching.compiled_batch_sizes`. Each tensor parses into a [`TensorSpec`](@ref) with a
[`Dim`](@ref) per axis, and the compiled sizes form the [`BatchingSpec`](@ref).

Shapes use the Julia column-major convention (the batch dimension is the last axis), which is
the reverse of the row-major form a Python/XLA exporter would write. The wire codec handles the
conversion, so KServe clients see canonical row-major shapes.

Datatypes are written as manifest tokens (`f32`, `bf16`, `i64`, `bool`, and so on); see
[`DType`](@ref) for the full mapping between tokens, Julia types, and KServe wire strings.
Client-facing tensors must use a dtype that has a KServe wire mapping (FP8 is executable-only).

## Custom pre/post-processing with model.jl

A bundle may include a `model.jl` that calls [`register_model`](@ref) to attach `preprocess`
and `postprocess` hooks. Both hooks receive and return a `Vector{NamedTensor}` (see
[`NamedTensor`](@ref)); omitted hooks default to identity.

```julia
# model.jl, inside the bundle directory
using ReactantServer

function normalize(inputs)
    # inputs :: Vector{NamedTensor}; transform and return a Vector{NamedTensor}
    return inputs
end

function to_classes(outputs)
    # e.g. map logits to class ids
    return outputs
end

register_model("resnet50"; preprocess=normalize, postprocess=to_classes)
```

The worker runs the hooks on each request's own task (preprocess before the request is queued,
postprocess on the result), crossing the world-age boundary with `invokelatest`. This means the
hooks for different requests run **concurrently, on multiple threads**, overlapping the GPU
execution: keep them free of shared mutable state (or guard it yourself). When `model.jl`
transforms the I/O, declare the client-facing tensors via `client_inputs` / `client_outputs` in
the manifest; without a `model.jl` those keys are not permitted and the executable specs are the
client-facing specs. See [`register_model`](@ref) in the API reference for the exact hook
signatures.

## Producing bundles

`ReactantServerExport` produces bundles offline and is kept out of the server's dependency
graph. It is not part of the server runtime.

A project that owns a Lux model (or any Reactant-traceable function) uses
`ReactantServerExport`; Lux itself is not a dependency of the package:

```julia
using ReactantServerExport
export_bundle(:lux, model, ps, st, example_input;
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
```

A PyTorch project also loads `PythonCall`, which triggers the package extension driving
`torch.export.export` and torchax:

```julia
using ReactantServerExport, PythonCall
export_bundle(:pytorch, model, (example_input,);
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
```

Both frontends trace once per requested batch size and write a server-loadable bundle. The
batch dimension is the last Julia axis (the leading PyTorch axis after the row-major /
column-major reversal). The test suite also builds small bundles directly; see
`test/stablehlo_fixtures.jl`.
