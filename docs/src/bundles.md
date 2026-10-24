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
- `weights.safetensors` - the model weights, read with `pread` at load time (see
  [Weights file layout](#Weights-file-layout)). A manifest may name another file in the bundle with
  `weights_file` (a migrated copy is `weights.T.safetensors`).
- `model.jl` - optional; registers custom pre/post-processing (see below).
- `.cache/` - written by the worker, never by you: the serialized-executable cache (compiled
  programs keyed by MLIR content, Reactant_jll build and device) plus `mlir_hashes.json`. Safe to
  delete at any time; the watcher ignores it. See `runtime.executable_cache` in [Node Configuration](node_config.md).

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

The per-input batch axis is derived from the position of `n`/`b`. When every executable input and
output has a batch axis, the scheduler coalesces queued requests along it and pads the dispatch
with zero rows up to the smallest compiled size that fits, so a request may carry any number of
rows up to the largest of `batching.compiled_batch_sizes`; otherwise the request's size along that
axis must equal one of them. `runtime.batch_sizes: largest` loads only the largest size (see
[Node Configuration](node_config.md)). Each tensor parses into a
[`TensorSpec`](@ref) with a [`Dim`](@ref) per axis, and the compiled sizes form the
[`BatchingSpec`](@ref). The writer stamps `format_version: "2.1"` (`"2.0"` with `layout = :legacy`);
the loader accepts `2.1`, `2.0` or `2`. Version `2.1` exists so that servers older than 0.2, which
would misread weights in the program's axis order, refuse such a bundle instead.

Shapes use the Julia column-major convention (the batch dimension is the last axis), which is the
reverse of the row-major form a Python/XLA exporter would write. The wire codec handles the
conversion, so KServe clients see canonical row-major shapes.

## Weights file layout

`weights.safetensors` stores each weight as the Julia array the server passes to the program. A
file has one of two layouts, declared in its own metadata, never inferred from the manifest or the
file name:

- **Program axis order** (`axis_order: xla` in the safetensors metadata, written by
  ReactantServerExport 2.0). Each tensor's header shape is the program's shape (the reversed Julia
  shape) and its data is row-major in that shape, which is byte for byte the Julia array. The
  server reads each tensor straight into its destination with one `pread`. numpy, PyTorch and
  SafeTensors.jl read these tensors in the program's shape.
- **Legacy** (no `axis_order` key, written before ReactantServerExport 2.0). Each header shape is
  the Julia shape and the data is permuted to row-major, so every load permutes it back, an extra
  full copy of every tensor.

Both layouts load to identical arrays. Any other `axis_order` value is refused.

## Migrating older bundles

`runtime.weights_migration` (see [Node Configuration](node_config.md)) converts legacy bundles at
load. In `copy` mode the worker writes `weights.T.safetensors` beside the original and adds
`weights_file: weights.T.safetensors` to the manifest, leaving `weights.safetensors` untouched,
so rolling back to an older server needs nothing. In `replace` mode it first does the same, then
sets `format_version: "2.1"`, renames the copy over `weights.safetensors` in one step, and removes
the extra name and the `weights_file` line.

Nothing is accepted without proof. A copy must hold the same tensor names, dtypes, byte counts and
metadata as the original, each header shape reversed, and every tensor equal byte for byte (so NaN
payloads and signed zeros count) to the array the legacy path reads, which is read twice, once by
the server and once by SafeTensors.jl. The copy records the original's size, modification time and
SHA-256. Before `replace` swaps it in, the copy is verified again against the original, and the
file left under the original's name is verified once more against the original (still open, so
its bytes remain readable) before anything else is removed.

Files are written under temporary names, fsynced, verified from disk, and renamed into place, so a
crash at any point leaves a bundle that loads correctly and the migration resumes on the next start.
Workers on one node take turns on a lock in the bundle's `.cache/`, so a bundle is migrated once.
A migration that cannot run (no free space, a read-only directory, an invalid copy) is logged and
the bundle loads as it was.

If you redeploy `weights.safetensors` after a `copy` migration, the copy is stale: the server
notices that the original changed since the copy was made, loads the original, and, with migration
on, rebuilds the copy. Comments and formatting in `manifest.yaml` are kept; only the
`weights_file` and `format_version` lines change.

`tools/migrate_weights.jl` runs the same migration offline, for model directories the workers
cannot write or to convert ahead of a deploy.

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
one executable, see [Meta Models](meta_models.md). That feature is experimental and its design is
subject to change; prefer a single program with the fixed-size result pattern below where possible.

## Variable-length results

A StableHLO program has static output shapes, so a model whose result count depends on the data
(detections, matches, tokens above a threshold) returns a **fixed-size buffer plus a count**: the
program fills the first `n` rows of a buffer sized to the configured maximum, zeroes the rest, and
returns `n` as a second output. A `model.jl` postprocess hook trims the buffer, so clients receive
exactly the rows that exist:

```julia
# model.jl: DETECTIONS is (columns, max_detections), NUM_DETECTIONS is Int64[n]
function _trim(outputs)
    by = Dict(t.name => t.data for t in outputs)
    n = Int(by["NUM_DETECTIONS"][1])
    return [ReactantServer.NamedTensor("OUTPUT__0", by["DETECTIONS"][:, 1:n])]
end

register_model(basename(@__DIR__); postprocess = _trim)
```

The manifest declares the executable outputs at their fixed sizes and the trimmed client output with
a variable axis (`-1`) under `client_outputs`. Inside the program, variable selections become masks
over fixed-size arrays rather than `findall`; the two-stage detectors are the worked example, see
[Object Detection](object_detection.md).

## Producing bundles

`ReactantServerExport` produces bundles offline and is kept out of the server's dependency graph.
It is not part of the server runtime.

A project that owns a Lux model (or any Reactant-traceable function) uses `ReactantServerExport`;
Lux itself is not a dependency of the package. [Export a Model](tutorials/export.md) walks
through exporting a Lux model end to end.

The `:lux` frontend, `export_bundle(:lux, model, ps, st, example_input; dir, name, batch_sizes)`,
traces `model(x, ps, st)` at each requested batch size, taking the first return as the output; the
batch dimension is the last Julia axis (the Lux convention) and the leading
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

The Reactant frontends (`:lux`, `:reactant`, and `export_two_stage_detector`) take
`compile_options`, the options `Reactant.@compile` accepts, applied when the model is traced to
StableHLO. Pass them as a `NamedTuple` or a `Reactant.CompileOptions`:

```julia
export_bundle(:lux, model, ps, st, x; dir="bundles/net", name="net",
    compile_options=(; raise=true))   # raise KernelAbstractions / CUDA.jl kernels to StableHLO
```

Non-default options are recorded under `compile_options` in the manifest's `provenance`. An unknown
option name is an error, and the exporter keeps `client` and `drop_unsupported_attributes` for
itself. A traced program that still contains Reactant-internal `enzyme.*` ops, which XLA cannot
compile (for example with `optimize = false`), is refused before anything is written. The PyTorch
frontends lower through JAX and do not take `compile_options`.

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
and `weights.safetensors` (in the program's axis order unless `layout = :legacy`; see
[Weights file layout](#Weights-file-layout)). `read_bundle_weights` reads a bundle's weights back in
the program's axis order, whichever layout they use. An `IOSpec` names a tensor, its Julia (column-major) shape, its dtype,
the 0-based `batch_axis` in that Julia shape (by convention the last axis), and optional explicit `letters` for the non-batch axes (the reserved
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

`export_bundle` checks this itself and refuses to write a bundle whose graph disagrees with it,
naming both numbers and the usual cause. The usual cause is a device-resident value reachable from
the traced closure: Reactant lifts every one of those into an argument whether the program reads it
or not, so an RNG in layer state (which appears as a leading `tensor<2xui64>`) or a device-resident
configuration value captured by the model becomes an argument no client can supply. Reading such
values back to the host before tracing bakes them into the graph as constants instead.

For a bundle written by other tooling, `assert_bundle_arity` reads the same three numbers back out
of the artifact:

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
[tutorials](tutorials/export.md) walk the full export-to-serve path, and
[Node Configuration](node_config.md) covers how the server loads and watches a repository of
bundles.
