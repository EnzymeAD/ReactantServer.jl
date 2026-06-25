"""
    ReactantServerExport

Offline tooling that assembles ReactantServer model bundles (manifest.yaml, model[.b{N}].mlir,
weights.safetensors) from StableHLO modules and weights, and the Reactant tracing frontend that
turns a model + parameters into a bundle. This package is the bundle-format authority for
producers; it is not depended on by the server runtime. The format is kept in sync with the
server by the round-trip tests.

`write_bundle`/`IOSpec` are the low-level writer. `export_bundle` traces a Reactant model (the
former `LuxExport`; it needs only Reactant, not Lux). PyTorch support
(`export_bundle(model, ::Tuple)` and `export_torchscript_bundle`) lives in a package extension
that loads when `PythonCall` is present — `using ReactantServerExport, PythonCall` enables it.
"""
module ReactantServerExport

using Reactant
using SafeTensors
using YAML
using JSON3

const MLIR = Reactant.MLIR
const Compiler = Reactant.Compiler

export IOSpec, write_bundle, export_bundle, export_torchscript_bundle

# ============================================================================
# Bundle writer (the former BundleWriter)
# ============================================================================

# Manifest dtype tokens, mirroring the server's dtypes. bf16 and the f8 types are intentionally
# omitted here until a frontend needs them (they require extra deps).
const DTYPE_TOKENS = Dict{DataType,String}(
    Float16 => "f16", Float32 => "f32", Float64 => "f64",
    Int8 => "i8", Int16 => "i16", Int32 => "i32", Int64 => "i64",
    UInt8 => "u8", UInt16 => "u16", UInt32 => "u32", UInt64 => "u64",
    Bool => "bool",
)
dtype_token(::Type{T}) where {T} =
    get(() -> error("ReactantServerExport: no manifest dtype token for Julia type $T"), DTYPE_TOKENS, T)

"""
    IOSpec(name, dtype, shape; batch_axis=nothing, letters=nothing)

A tensor's client-facing spec. `shape` is the network (row-major) shape with a concrete
value at the batch axis; `batch_axis` is the 0-based network axis that carries the batch
dimension, or `nothing`. The manifest serialization encodes the shape as an einsum-style
letter string (`"chwn"`) plus a `dims` map of letter→size; the batch axis is emitted as
`n`. Non-batch letters are auto-allocated from `_AXIS_LETTERS` and carry no semantic
meaning across tensors.

`letters` optionally overrides that auto-allocation with explicit non-batch axis letters,
one per non-batch axis in `shape` order (the batch axis is still emitted as `n`). Pass it to
give axes meaningful names, e.g. `['w','h']` for an image input so its manifest reads `whn`.
The reserved markers `n`/`b` are rejected.
"""
struct IOSpec
    name::String
    dtype::DataType
    shape::Vector{Int}
    batch_axis::Union{Int,Nothing}
    letters::Union{Nothing,Vector{Char}}
end
IOSpec(name, dtype, shape; batch_axis=nothing, letters=nothing) =
    IOSpec(String(name), dtype, Int[shape...], batch_axis,
           letters === nothing ? nothing : Char[letters...])

# 'n' and 'b' are reserved batch markers; everything else is fair game for non-batch axes.
const _AXIS_LETTERS = "acdefghijklmopqrstuvwxyz"

# The letters for `s`'s non-batch axes, in `shape` order: the explicit `s.letters` if given (validated),
# otherwise auto-allocated from `_AXIS_LETTERS`.
function _nonbatch_letters(s::IOSpec)
    nax = count(i -> !(s.batch_axis !== nothing && (i - 1) == s.batch_axis), eachindex(s.shape))
    if s.letters !== nothing
        length(s.letters) == nax ||
            error("ReactantServerExport: tensor '$(s.name)' has $nax non-batch axes but $(length(s.letters)) letters $(s.letters)")
        any(c -> c in ('n', 'b'), s.letters) &&
            error("ReactantServerExport: tensor '$(s.name)' axis letters may not use the reserved batch markers 'n'/'b'")
        allunique(s.letters) ||
            error("ReactantServerExport: tensor '$(s.name)' has duplicate axis letters $(s.letters)")
        return s.letters
    end
    nax <= length(_AXIS_LETTERS) ||
        error("ReactantServerExport: tensor '$(s.name)' has more non-batch axes than available axis letters")
    return collect(Char, _AXIS_LETTERS[1:nax])
end

function _spec_dict(s::IOSpec)
    letters = _nonbatch_letters(s)
    shape_chars = Char[]
    dims = Dict{String,Any}()
    k = 0
    for (i, d) in enumerate(s.shape)               # network axis = i - 1
        if s.batch_axis !== nothing && (i - 1) == s.batch_axis
            push!(shape_chars, 'n')
        else
            k += 1
            c = letters[k]
            push!(shape_chars, c)
            dims[string(c)] = Int(d)
        end
    end
    return Dict{String,Any}("name" => s.name, "dtype" => dtype_token(s.dtype),
                            "shape" => String(shape_chars), "dims" => dims)
end

# The shape letters `_spec_dict` assigns to the variable (size -1) axes of each input, in
# (input, axis) order. This is the order the manifest `input_shapes` variants and the runtime
# variant key are both built in, so the emitted letters line up with what the server reads back.
function _variable_letters(specs::AbstractVector{IOSpec})
    out = Char[]
    for s in specs
        letters = _nonbatch_letters(s)
        k = 0
        for (i, d) in enumerate(s.shape)
            (s.batch_axis !== nothing && (i - 1) == s.batch_axis) && continue
            k += 1
            d == -1 && push!(out, letters[k])
        end
    end
    return out
end

# --- StableHLO serialization to a portable artifact ---

function _capture(f)
    cb = @cfunction(MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any))
    ref = Ref(IOBuffer())
    res = f(cb, ref)
    return take!(ref[]), res
end

function _serialize_module(mod::MLIR.IR.Module)
    vbytes, _ = _capture((cb, ref) -> MLIR.API.stablehloGetCurrentVersion(cb, ref))
    ver = String(vbytes)
    bytes, sres = _capture((cb, ref) ->
        MLIR.API.stablehloSerializePortableArtifactFromModule(mod, ver, cb, ref, true))
    MLIR.IR.isfailure(MLIR.IR.LogicalResult(sres)) && error("ReactantServerExport: failed to serialize StableHLO")
    return bytes
end

# Accept an already-serialized artifact, an MLIR module, or MLIR text.
_to_artifact(x::AbstractVector{UInt8}) = Vector{UInt8}(x)
_to_artifact(mod::MLIR.IR.Module) = _serialize_module(mod)
function _to_artifact(text::AbstractString)
    return MLIR.IR.@with_context Reactant.ReactantContext() begin
        _serialize_module(parse(MLIR.IR.Module, String(text)))
    end
end

# Write one variant's StableHLO module(s) under a file prefix (`model` or `model.v{i}`) and return
# the sorted batch-size list. A single entry under key 0 writes `<prefix>.mlir`; otherwise each
# writes `<prefix>.b{N}.mlir`. The returned sizes feed `batching.compiled_batch_sizes`.
function _write_variant_modules(dir::AbstractString, prefix::AbstractString, modules::AbstractDict)
    if length(modules) == 1 && haskey(modules, 0)
        write(joinpath(dir, "$prefix.mlir"), _to_artifact(modules[0]))
        return Int[]
    end
    sizes = sort!(collect(keys(modules)))
    for s in sizes
        write(joinpath(dir, "$prefix.b$s.mlir"), _to_artifact(modules[s]))
    end
    return sizes
end

"""
    write_bundle(dir; name, executable_inputs, executable_outputs, modules, weights,
                 client_inputs=nothing, client_outputs=nothing, input_shapes=nothing,
                 provenance=Dict()) -> dir

Write a bundle. With `input_shapes === nothing`, `modules` is a `Dict` keyed by batch size of
StableHLO modules (or text or bytes); a single entry under key `0` writes `model.mlir`, otherwise
each writes `model.b{N}.mlir`. `weights` is an ordered collection of `name => array` pairs whose
order becomes the safetensors `argument_order`. The per-input batch axis is recorded in each
input's `IOSpec.batch_axis`; the manifest derives `batching.batch_dim` from there.

`input_shapes` (a `Vector{Vector{Int}}` of compiled input-shape variants) turns on the multi-shape
layout: the variable executable-input axes are marked `-1` in `executable_inputs`, and each variant
gives the concrete sizes of those axes in (input, axis) order. `modules` is then keyed by variant
(each key a `Vector{Int}` equal to one `input_shapes` entry) and maps to that variant's batch-size
module dict; the files are written as `model.v{i}.*.mlir` (`i` indexing `input_shapes`), all sharing
the single `weights.safetensors`. The variants must share one set of batch sizes.

`client_inputs`/`client_outputs` (each `nothing` or a `Vector{IOSpec}`) declare the wire-facing
spec when it differs from the executable spec, for bundles that ship a `model.jl` whose
preprocess/postprocess transform between the two. They are emitted only when given. A variable
(non-batch) axis is encoded by passing `-1` for that axis size (e.g. the variable detection count
of a postprocessed detector). The server requires these only when a `model.jl` is present, so the
caller is responsible for also shipping `model.jl` into the bundle dir (see the converter handlers).
"""
function write_bundle(dir::AbstractString; name::AbstractString,
                      executable_inputs::AbstractVector{IOSpec},
                      executable_outputs::AbstractVector{IOSpec},
                      modules::AbstractDict, weights,
                      client_inputs::Union{Nothing,AbstractVector{IOSpec}}=nothing,
                      client_outputs::Union{Nothing,AbstractVector{IOSpec}}=nothing,
                      input_shapes::Union{Nothing,AbstractVector}=nothing,
                      provenance=Dict{String,Any}())
    basename(normpath(dir)) == String(name) ||
        error("ReactantServerExport: bundle dir basename must equal name '$name' (got '$(basename(normpath(dir)))')")
    mkpath(dir)

    wnames = String[String(first(p)) for p in weights]
    wdata = Dict{String,AbstractArray}(String(first(p)) => collect(last(p)) for p in weights)
    SafeTensors.serialize(joinpath(dir, "weights.safetensors"), wdata,
                          Dict("argument_order" => JSON3.write(wnames)))

    manifest = Dict{String,Any}(
        "format_version" => "2.0",
        "name" => String(name),
        "executable_inputs" => [_spec_dict(s) for s in executable_inputs],
        "executable_outputs" => [_spec_dict(s) for s in executable_outputs],
        "provenance" => Dict{String,Any}(string(k) => v for (k, v) in provenance),
    )

    if input_shapes === nothing
        sizes = _write_variant_modules(dir, "model", modules)
        manifest["batching"] = Dict{String,Any}("compiled_batch_sizes" => sizes)
    else
        vletters = _variable_letters(executable_inputs)
        sizes = Int[]
        for (vi, vk) in enumerate(input_shapes)
            vkey = Int[Int(x) for x in vk]
            length(vkey) == length(vletters) ||
                error("ReactantServerExport: input_shapes[$vi] has $(length(vkey)) sizes but the inputs have $(length(vletters)) variable axes")
            haskey(modules, vkey) ||
                error("ReactantServerExport: no modules for input_shapes variant $vkey")
            vsizes = _write_variant_modules(dir, "model.v$(vi - 1)", modules[vkey])
            vi == 1 ? (sizes = vsizes) :
                (vsizes == sizes || error("ReactantServerExport: variant $vkey has batch sizes $vsizes, expected $sizes"))
        end
        manifest["batching"] = Dict{String,Any}("compiled_batch_sizes" => sizes)
        manifest["input_shapes"] =
            [Dict{String,Any}(string(vletters[j]) => Int(vk[j]) for j in eachindex(vletters)) for vk in input_shapes]
    end

    client_inputs === nothing || (manifest["client_inputs"] = [_spec_dict(s) for s in client_inputs])
    client_outputs === nothing || (manifest["client_outputs"] = [_spec_dict(s) for s in client_outputs])
    YAML.write_file(joinpath(dir, "manifest.yaml"), manifest)
    return dir
end

# ============================================================================
# Reactant tracing frontend (the former LuxExport; needs Reactant, not Lux)
# ============================================================================

# Ordered (name, array) leaves of a parameter tree (NamedTuple/Tuple/Vector/Array).
function _named_leaves(x, prefix="", out=Tuple{String,Any}[])
    if x isa AbstractArray{<:Number}
        push!(out, (isempty(prefix) ? "param" : prefix, x))
    elseif x isa NamedTuple
        for k in keys(x)
            _named_leaves(getfield(x, k), isempty(prefix) ? String(k) : "$prefix.$k", out)
        end
    elseif x isa Tuple || x isa AbstractVector
        for (i, v) in enumerate(x)
            _named_leaves(v, isempty(prefix) ? string(i) : "$prefix.$i", out)
        end
    end
    return out
end

# Rebuild a parameter tree shaped like `template`, drawing leaves in order from `ws`.
function _rebuild(template, ws, idx=Ref(0))
    if template isa AbstractArray{<:Number}
        idx[] += 1
        return ws[idx[]]
    elseif template isa NamedTuple
        return NamedTuple{keys(template)}(map(k -> _rebuild(getfield(template, k), ws, idx), keys(template)))
    elseif template isa Tuple
        return map(v -> _rebuild(v, ws, idx), template)
    elseif template isa AbstractVector
        return [_rebuild(v, ws, idx) for v in template]
    else
        return template
    end
end

# A representative array shaped like `x` but with `axis` resized to `s` (values irrelevant,
# only the shape is traced / used to derive output shape).
function _with_batch(x::AbstractArray, axis::Int, s::Integer)
    sz = collect(size(x))
    sz[axis] = Int(s)
    return zeros(eltype(x), sz...)
end

"""
    export_bundle(frontend::Symbol, args...; kwargs...) -> dir

Export a model to a bundle, naming the frontend explicitly so the call site is unambiguous:

- `export_bundle(:lux, model, ps, st, example_input; ...)` — a Lux-style model with separate
  parameters/state (any Reactant-traceable `model(x, ps, st)`; Lux itself is not required).
- `export_bundle(:reactant, f, inputs::Tuple, weights; ...)` — any Reactant-traceable function
  `f(inputs..., weights...)` with explicit `name => array` weight pairs.
- `export_bundle(:pytorch, model, example_inputs::Tuple; ...)` — a `torch.nn.Module`; provided
  by the package extension, so it requires `using PythonCall`.
"""
export_bundle(frontend::Symbol, args...; kwargs...) = export_bundle(Val(frontend), args...; kwargs...)

# Clear error when the PyTorch extension has not been loaded (less specific than the method the
# extension adds, so the real implementation wins once PythonCall is present).
export_bundle(::Val{:pytorch}, args...; kwargs...) =
    error("ReactantServerExport: `export_bundle(:pytorch, ...)` requires `using PythonCall` to load the extension")

"""
    export_bundle(:lux, model, ps, st, example_input; dir, name, input_name="input",
                  output_name="output", batch_sizes=[1], provenance=Dict()) -> dir

Trace `model(x, ps, st)` (taking the first return as the output) at each batch size and
write a bundle. The batch dimension is the last Julia axis (Lux convention) and the leading
network axis. Works for any Reactant-traceable model; Lux itself is not required.
"""
function export_bundle(::Val{:lux}, model, ps, st, example_input::AbstractArray;
                       dir::AbstractString, name::AbstractString,
                       input_name::AbstractString="input", output_name::AbstractString="output",
                       batch_sizes::AbstractVector{<:Integer}=[1], provenance=Dict{String,Any}())
    leaves = _named_leaves(ps)
    isempty(leaves) && error("ReactantServerExport: model has no array parameters to export")
    wnames = String[p[1] for p in leaves]
    warrays = Any[p[2] for p in leaves]

    batch_axis = ndims(example_input)
    in_T = eltype(example_input)
    y0 = first(model(_with_batch(example_input, batch_axis, first(batch_sizes)), ps, st))

    g = (x, ws...) -> first(model(x, _rebuild(ps, collect(ws)), st))

    ctxs = Any[]                                   # keep contexts alive through serialization
    modules = Dict{Int,Any}()
    in_shape_julia = Int[]
    for s in batch_sizes
        x = _with_batch(example_input, batch_axis, s)
        ctx = Reactant.ReactantContext()
        push!(ctxs, ctx)
        args = (Reactant.to_rarray(x), map(Reactant.to_rarray, warrays)...)
        mod, _ = Compiler.compile_mlir(ctx, g, args; drop_unsupported_attributes=true)
        modules[Int(s)] = mod
        in_shape_julia = collect(Int, size(x))
    end

    # Manifest is Julia order; batch axis is the 0-based Julia axis (Lux: last axis).
    in_batch_axis = ndims(example_input) - 1
    out_batch_axis = ndims(y0) - 1
    inputs = [IOSpec(input_name, in_T, in_shape_julia; batch_axis=in_batch_axis)]
    outputs = [IOSpec(output_name, eltype(y0), collect(Int, size(y0)); batch_axis=out_batch_axis)]
    prov = merge(Dict{String,Any}("source_framework" => "reactant", "converter" => "ReactantServerExport.jl"),
                 Dict{String,Any}(provenance))

    GC.@preserve ctxs begin
        write_bundle(dir; name=name, executable_inputs=inputs, executable_outputs=outputs,
            modules=modules, weights=[wnames[i] => warrays[i] for i in eachindex(wnames)],
            provenance=prov)
    end
    return dir
end

"""
    export_bundle(:lux, model, ps, st, example_inputs::Tuple; dir, name,
                  input_names=nothing, output_names=nothing,
                  output_select = y -> (y isa Tuple ? y : (y,)),
                  input_batch_axes=nothing, output_batch_axes=nothing,
                  client_inputs=nothing, client_outputs=nothing,
                  batch_sizes=[1], provenance=Dict()) -> dir

Multi-input / multi-output Lux export. `client_inputs`/`client_outputs` (each `nothing` or a
`Vector{IOSpec}`) declare the wire-facing spec when a shipped `model.jl` postprocess transforms
the executable tensors into different client tensors; they are passed through to `write_bundle`. Traces `model(x_tuple, ps, st)` where `x_tuple` is the
tuple of array inputs the model's forward expects as its single positional argument.
`output_select(first(model(...)))` maps the raw model output to the ordered tuple of arrays to
export; non-array returns (e.g. an `Int` step count) are dropped by the selector and must not
appear in its result. Weights are extracted from `ps` automatically (same as the single-array
method). Per-tensor batch axes default to each array's last Julia axis (Lux convention) but can
be overridden with `input_batch_axes`/`output_batch_axes` (1-based Julia axes), which is required
when the batch axis is not last (e.g. a `(D, N, K+1)` output whose batch axis N is the middle one).
"""
function export_bundle(::Val{:lux}, model, ps, st, example_inputs::Tuple;
                       dir::AbstractString, name::AbstractString,
                       input_names=nothing, output_names=nothing,
                       output_select = y -> (y isa Tuple ? y : (y,)),
                       input_batch_axes::Union{Nothing,AbstractVector}=nothing,
                       output_batch_axes::Union{Nothing,AbstractVector}=nothing,
                       client_inputs::Union{Nothing,AbstractVector{IOSpec}}=nothing,
                       client_outputs::Union{Nothing,AbstractVector{IOSpec}}=nothing,
                       batch_sizes::AbstractVector{<:Integer}=[1], provenance=Dict{String,Any}())
    leaves = _named_leaves(ps)
    isempty(leaves) && error("ReactantServerExport: model has no array parameters to export")
    wnames = String[p[1] for p in leaves]
    warrays = Any[p[2] for p in leaves]

    nin = length(example_inputs)
    in_axes = input_batch_axes === nothing ? [ndims(x) for x in example_inputs] :
              collect(Int, input_batch_axes)
    length(in_axes) == nin ||
        error("ReactantServerExport: input_batch_axes has $(length(in_axes)) entries but $nin inputs")
    innames = input_names === nothing ? ["input_$(i - 1)" for i in 1:nin] :
              collect(String, input_names)

    # `output_select` may return a single array or a tuple of arrays; normalize to a tuple.
    _as_tuple(y) = y isa Tuple ? y : (y,)
    # Lux convention: a model takes ONE input object. With several inputs that object is the tuple
    # itself (`do (a, b)`); with one input it is that array directly (`do x`). Unwrap accordingly so
    # both shapes work. (Reactant still flattens a tuple arg into one MLIR input per element.)
    _modelarg(t) = nin == 1 ? t[1] : t
    mk_inputs(s) = ntuple(i -> _with_batch(example_inputs[i], in_axes[i], s), nin)
    y0 = _as_tuple(output_select(first(model(_modelarg(mk_inputs(first(batch_sizes))), ps, st))))
    all(o -> o isa AbstractArray, y0) ||
        error("ReactantServerExport: output_select must return only arrays; got $(map(typeof, y0))")
    nout = length(y0)
    outnames = output_names === nothing ? ["output_$(i - 1)" for i in 1:nout] :
               collect(String, output_names)
    out_axes = output_batch_axes === nothing ? [ndims(o) for o in y0] :
               collect(Int, output_batch_axes)
    length(out_axes) == nout ||
        error("ReactantServerExport: output_batch_axes has $(length(out_axes)) entries but $nout outputs")

    # ws... are the rebuilt parameter leaves. Returning a Tuple makes the compiled MLIR func emit
    # one result per selected output.
    g = (a, ws...) -> _as_tuple(output_select(first(model(a, _rebuild(ps, collect(ws)), st))))

    ctxs = Any[]                                   # keep contexts alive through serialization
    modules = Dict{Int,Any}()
    in_shapes = [Int[] for _ in 1:nin]
    for s in batch_sizes
        xs = mk_inputs(s)
        ctx = Reactant.ReactantContext()
        push!(ctxs, ctx)
        args = (_modelarg(map(Reactant.to_rarray, xs)), map(Reactant.to_rarray, warrays)...)
        mod, _ = Compiler.compile_mlir(ctx, g, args; drop_unsupported_attributes=true)
        modules[Int(s)] = mod
        for i in 1:nin
            in_shapes[i] = collect(Int, size(xs[i]))
        end
    end

    # Manifest is Julia order; batch axis is the 0-based Julia axis.
    in_specs = [IOSpec(innames[i], eltype(example_inputs[i]), in_shapes[i]; batch_axis=in_axes[i] - 1)
                for i in 1:nin]
    out_specs = [IOSpec(outnames[i], eltype(y0[i]), collect(Int, size(y0[i])); batch_axis=out_axes[i] - 1)
                 for i in 1:nout]
    prov = merge(Dict{String,Any}("source_framework" => "reactant", "converter" => "ReactantServerExport.jl"),
                 Dict{String,Any}(provenance))

    GC.@preserve ctxs begin
        write_bundle(dir; name=name, executable_inputs=in_specs, executable_outputs=out_specs,
            modules=modules, weights=[wnames[i] => warrays[i] for i in eachindex(wnames)],
            client_inputs=client_inputs, client_outputs=client_outputs, provenance=prov)
    end
    return dir
end

"""
    export_bundle(:reactant, f, inputs::Tuple, weights; dir, name, input_names=nothing,
                  output_name="output", provenance=Dict()) -> dir

Generic single-size export for any Reactant-traceable `f(inputs..., weights...)`. `weights`
is an ordered collection of `name => array` pairs. Produces one unbatched `model.mlir`.
"""
function export_bundle(::Val{:reactant}, f, inputs::Tuple, weights::AbstractVector{<:Pair};
                       dir::AbstractString, name::AbstractString,
                       input_names=nothing, output_name::AbstractString="output",
                       provenance=Dict{String,Any}())
    wnames = String[String(first(p)) for p in weights]
    warrays = Any[last(p) for p in weights]
    innames = input_names === nothing ? ["input_$(i - 1)" for i in 1:length(inputs)] : collect(String, input_names)

    y = f(inputs..., warrays...)
    yarr = y isa Tuple ? first(y) : y

    ctx = Reactant.ReactantContext()
    args = (map(Reactant.to_rarray, inputs)..., map(Reactant.to_rarray, warrays)...)
    mod, _ = Compiler.compile_mlir(ctx, f, args; drop_unsupported_attributes=true)

    in_specs = [IOSpec(innames[i], eltype(inputs[i]), collect(Int, size(inputs[i])))
                for i in 1:length(inputs)]
    out_specs = [IOSpec(output_name, eltype(yarr), collect(Int, size(yarr)))]
    prov = merge(Dict{String,Any}("source_framework" => "reactant", "converter" => "ReactantServerExport.jl"),
                 Dict{String,Any}(provenance))

    GC.@preserve ctx begin
        write_bundle(dir; name=name, executable_inputs=in_specs, executable_outputs=out_specs,
            modules=Dict(0 => mod), weights=[wnames[i] => warrays[i] for i in eachindex(wnames)],
            provenance=prov)
    end
    return dir
end

# ============================================================================
# Extension seams: defined here (so callers reach them as ReactantServerExport.X) and given
# methods by the PythonCall-triggered PyTorchExportExt. Calling these without `using PythonCall`
# raises a MethodError directing you to load PythonCall.
# ============================================================================

"""
    export_torchscript_bundle(pt_path_or_module, example_inputs::Tuple; ...) -> dir

Export a TorchScript artifact (a `.pt` file or a loaded `ScriptModule`) to a bundle. Provided
by the package extension; load `PythonCall` to enable it.
"""
export_torchscript_bundle(args...; kwargs...) =
    error("ReactantServerExport: `export_torchscript_bundle` requires `using PythonCall` to load the extension")

function _pyimports end
function _numpy_to_julia end
function _numpy_dtype_to_julia end
function _julia_to_numpy_dtype end

end # module ReactantServerExport
