# Bundles & model.jl

A model is delivered to the server as a self-contained bundle: a directory holding a compiled
model (an MLIR module Reactant compiles, a StableHLO program today), its weights, and a manifest.
Bundles are produced offline by the conversion tooling and loaded at server startup.

## Bundle layout

A bundle is a directory containing:

- `manifest.yaml` - the metadata parsed into a [`Manifest`](@ref): I/O specs, dtypes, shapes,
  and the compiled batch sizes.
- `model.mlir` - the MLIR module Reactant compiles, currently a serialized StableHLO portable
  artifact (single batch size), or one module per size as `model.b{N}.mlir` sharing a single
  `weights.safetensors`.
- `weights.safetensors` - the model weights, memory-mapped at load time.
- `model.jl` - optional; registers custom pre/post-processing (see below).

The directory name is the model name: renaming the directory renames the model, with no edits to
the manifest or `model.jl` (a `name` declared in either is informational and ignored). Each
immediate subdirectory of `model_repo` that contains a `manifest.yaml` is a bundle. In `dynamic`
mode a renamed bundle directory (same contents) is detected by the watcher and renamed in place,
keeping the compiled executables and resident weights; only new or changed bundles are compiled.

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
letters are tensor-scoped (no implicit cross-tensor equality) and must be unique within a single
shape. A size of `-1` in `dims` marks a variable axis, a [`Dim`](@ref) with kind `VARIABLE`: a
dynamic non-batch axis. On `client_outputs` that pass through `model.jl` that is how a result with
a data-dependent extent is declared, for example the detection count of a postprocessed detector.
On executable inputs a variable axis is servable only when the compiled variants are enumerated in
a top-level `input_shapes` block (see `write_bundle` below); the
[Object Detection](object_detection.md) example compiles one weight set for several image shapes
this way.

The per-input batch axis is derived from the position of `n`/`b`; at inference the request's size
along that axis must equal one of `batching.compiled_batch_sizes`. Each tensor parses into a
[`TensorSpec`](@ref) with a [`Dim`](@ref) per axis, and the compiled sizes form the
[`BatchingSpec`](@ref). The writer stamps `format_version: "2.0"`; the loader accepts `2.0` or `2`.

Shapes use the Julia column-major convention (the batch dimension is the last axis), which is the
reverse of the row-major form a Python/XLA exporter would write. The wire codec handles the
conversion, so KServe clients see canonical row-major shapes.

Datatypes are written as manifest tokens (`f32`, `bf16`, `i64`, `bool`, and so on); see
[`DType`](@ref) for the full mapping between tokens, Julia types, and KServe wire strings.
Client-facing tensors must use a dtype that has a KServe wire mapping (FP8 is executable-only).

## Custom pre/post-processing with model.jl

A bundle may include a `model.jl` that calls [`register_model`](@ref) to attach `preprocess` and
`postprocess` hooks. Both hooks receive and return a `Vector{NamedTensor}` (see
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
hooks for different requests run concurrently, on multiple threads, overlapping the GPU execution:
keep them free of shared mutable state (or guard it yourself). When `model.jl` transforms the I/O,
declare the client-facing tensors via `client_inputs` / `client_outputs` in the manifest; without a
`model.jl` those keys are not permitted and the executable specs are the client-facing specs. See
[`register_model`](@ref) in the API reference for the exact hook signatures.

For a bundle whose `model.jl` chains several models with data-dependent logic rather than wrapping
one executable, see [Meta Models](meta_models.md).

## Producing bundles

`ReactantServerExport` produces bundles offline and is kept out of the server's dependency graph.
It is not part of the server runtime.

A project that owns a Lux model (or any Reactant-traceable function) uses `ReactantServerExport`;
Lux itself is not a dependency of the package:

```julia
using ReactantServerExport
export_bundle(:lux, model, ps, st, example_input;
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
```

The `:lux` frontend traces `model(x, ps, st)` at each requested batch size, taking the first return
as the output; the batch dimension is the last Julia axis (the Lux convention) and the leading
network axis. A multi-input / multi-output variant takes `example_inputs::Tuple` and `output_select`
mapping the raw model output to the ordered tuple of arrays to export; per-tensor batch axes default
to each array's last Julia axis, overridable with `input_batch_axes` / `output_batch_axes` (1-based
Julia axes, or `nothing` to opt one tensor out of batching).

The `:reactant` frontend is the generic form: any Reactant-traceable `f(inputs..., weights...)`
with explicit `name => array` weight pairs, producing a single unbatched `model.mlir`:

```julia
export_bundle(:reactant, f, (x,), [("w1", w1), ("b1", b1)];
    dir="bundles/mlp", name="mlp")
```

A PyTorch project also loads `PythonCall`, which triggers the package extension driving
`torch.export.export` and torchax:

```julia
using ReactantServerExport, PythonCall
export_bundle(:pytorch, model, (example_input,);
    dir="bundles/mlp", name="mlp", batch_sizes=[1, 8])
```

The `:lux` and `:pytorch` frontends trace once per requested batch size and write a server-loadable
bundle. The batch dimension is the last Julia axis (the leading PyTorch axis after the row-major /
column-major reversal). `export_bundle(:torchscript, ...)` exports a TorchScript artifact (a `.pt`
file or a loaded `ScriptModule`) through the same extension; without `PythonCall` the `:pytorch`
and `:torchscript` calls fail with a message directing you to load it.

### write_bundle and IOSpec

`write_bundle` is the low-level writer: it takes the executable `IOSpec`s, the StableHLO modules
(one per batch size, or text or serialized bytes), and an ordered `weights` collection whose order
becomes the safetensors `argument_order`, and emits `manifest.yaml`, the `model[.b{N}].mlir` files,
and `weights.safetensors`. An `IOSpec` names a tensor, its network (row-major) shape, its dtype,
the 0-based `batch_axis`, and optional explicit `letters` for the non-batch axes (the reserved
markers `n`/`b` are rejected); a `-1` axis size encodes a variable axis. The bundle directory's
basename must equal `name` (the identity rule above).

With `input_shapes` given, the writer marks the variable executable-input axes `-1` in the manifest
and emits one module set per compiled variant as `model.v{i}[.b{N}].mlir`, all sharing the single
`weights.safetensors`; each variant lists the variable-axis sizes in (input, axis) order, and the
variants must share one set of batch sizes. `client_inputs` / `client_outputs` (each `nothing` or a
`Vector{IOSpec}`) declare the wire-facing spec when it differs from the executable spec, for
bundles whose `model.jl` transforms between the two; they are emitted only when given.

`collect_provenance` captures best-effort reproducibility metadata for a bundle (repo remote,
commit, tree, branch, dirty flag, Julia and exporter versions, and an `exported_at` timestamp).
`export_bundle` merges it into the manifest's `provenance` block; on a dirty work tree the
uncommitted diff is written to the bundle as `working_tree.patch` rather than embedded in the
manifest, and `git_commit` plus `git apply --binary working_tree.patch` reconstructs the exported
code exactly.

The test suite also builds small bundles directly; see `test/stablehlo_fixtures.jl`.

## Checking that a bundle is servable

A bundle is served as `executable(inputs..., weights...)`, so the compiled program must take exactly
as many arguments as the manifest declares inputs plus the weights file holds tensors. Both of those
halves are readable, and the number the executable actually wants is in neither: it lives inside
`model*.mlir`, which is a serialized `vhlo` artifact rather than text. A bundle can therefore be
internally inconsistent while every readable part of it looks correct, and the symptom is that the
model registers and serves and then fails every inference with

```
INVALID_ARGUMENT: Execution supplied 216 arguments but compiled program expected 217
```

`export_bundle` now checks this itself and refuses to write a bundle whose graph disagrees with it,
naming both numbers and the usual cause. The usual cause is a device-resident value reachable from
the traced closure: Reactant lifts every one of those into an argument whether the program reads it
or not, so an RNG in layer state (which appears as a leading `tensor<2xui64>`) or a device-resident
configuration value captured by the model becomes an argument no client can supply. Reading such
values back to the host before tracing bakes them into the graph as constants instead.

For bundles that were written before that check existed, `assert_bundle_arity` reads the same three
numbers back out of the artifact:

```julia
using ReactantServerExport

assert_bundle_arity("export_out/my_model_v1")          # raises if the bundle is unservable
r = bundle_arity_report("export_out/my_model_v1")      # the numbers, without raising
r.servable, r.n_inputs, r.n_weights, r.expected
```

`bundle_arity_report` never raises, so it can be run across a directory of bundles to triage them,
and `bundle_entry_arity` reads one module's arity on its own. All three read the artifact rather than
the process that produced it, so they hold for a bundle from any writer.

## Related pages

The manifest and boundary types are documented on the [API](api.md) page: [`Manifest`](@ref),
[`TensorSpec`](@ref), [`Dim`](@ref), [`BatchingSpec`](@ref), [`load_manifest`](@ref),
[`DType`](@ref), and [`NamedTensor`](@ref). `export_bundle`, `write_bundle`, `IOSpec`,
`collect_provenance`, `assert_bundle_arity`, `bundle_arity_report`, and `bundle_entry_arity` are
documented in the `ReactantServerExport` docstrings. The
[Tutorial](tutorial.md) walks the full export-to-serve path, and
[Node Configuration](node_config.md) covers how the server loads and watches a repository of
bundles.
