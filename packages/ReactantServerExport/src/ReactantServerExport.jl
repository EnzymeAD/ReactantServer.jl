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
    IOSpec(name, dtype, shape; batch_axis=nothing)

A tensor's client-facing spec. `shape` is the network (row-major) shape with a concrete
value at the batch axis; `batch_axis` is the 0-based network axis that carries the batch
dimension, or `nothing`. The manifest serialization encodes the shape as an einsum-style
letter string (`"chwn"`) plus a `dims` map of letter→size; the batch axis is emitted as
`n`. Non-batch letters are auto-allocated from `_AXIS_LETTERS` and carry no semantic
meaning across tensors.
"""
struct IOSpec
    name::String
    dtype::DataType
    shape::Vector{Int}
    batch_axis::Union{Int,Nothing}
end
IOSpec(name, dtype, shape; batch_axis=nothing) = IOSpec(String(name), dtype, Int[shape...], batch_axis)

# 'n' and 'b' are reserved batch markers; everything else is fair game for non-batch axes.
const _AXIS_LETTERS = "acdefghijklmopqrstuvwxyz"

function _spec_dict(s::IOSpec)
    shape_chars = Char[]
    dims = Dict{String,Any}()
    next_idx = 1
    for (i, d) in enumerate(s.shape)               # network axis = i - 1
        if s.batch_axis !== nothing && (i - 1) == s.batch_axis
            push!(shape_chars, 'n')
        else
            next_idx <= length(_AXIS_LETTERS) ||
                error("ReactantServerExport: tensor '$(s.name)' has more non-batch axes than available axis letters")
            c = _AXIS_LETTERS[next_idx]
            next_idx += 1
            push!(shape_chars, c)
            dims[string(c)] = Int(d)
        end
    end
    return Dict{String,Any}("name" => s.name, "dtype" => dtype_token(s.dtype),
                            "shape" => String(shape_chars), "dims" => dims)
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

"""
    write_bundle(dir; name, executable_inputs, executable_outputs, modules, weights,
                 provenance=Dict()) -> dir

Write a bundle. `modules` is a `Dict` keyed by batch size of StableHLO modules (or text or
bytes); a single entry under key `0` writes `model.mlir`, otherwise each writes
`model.b{N}.mlir`. `weights` is an ordered collection of `name => array` pairs whose order
becomes the safetensors `argument_order`. The per-input batch axis is recorded in each
input's `IOSpec.batch_axis`; the manifest derives `batching.batch_dim` from there.
"""
function write_bundle(dir::AbstractString; name::AbstractString,
                      executable_inputs::AbstractVector{IOSpec},
                      executable_outputs::AbstractVector{IOSpec},
                      modules::AbstractDict, weights,
                      provenance=Dict{String,Any}())
    basename(normpath(dir)) == String(name) ||
        error("ReactantServerExport: bundle dir basename must equal name '$name' (got '$(basename(normpath(dir)))')")
    mkpath(dir)

    wnames = String[String(first(p)) for p in weights]
    wdata = Dict{String,AbstractArray}(String(first(p)) => collect(last(p)) for p in weights)
    SafeTensors.serialize(joinpath(dir, "weights.safetensors"), wdata,
                          Dict("argument_order" => JSON3.write(wnames)))

    sizes = Int[]
    if length(modules) == 1 && haskey(modules, 0)
        write(joinpath(dir, "model.mlir"), _to_artifact(modules[0]))
    else
        sizes = sort!(collect(keys(modules)))
        for s in sizes
            write(joinpath(dir, "model.b$s.mlir"), _to_artifact(modules[s]))
        end
    end

    manifest = Dict{String,Any}(
        "format_version" => "2.0",
        "name" => String(name),
        "executable_inputs" => [_spec_dict(s) for s in executable_inputs],
        "executable_outputs" => [_spec_dict(s) for s in executable_outputs],
        "batching" => Dict{String,Any}("compiled_batch_sizes" => sizes),
        "provenance" => Dict{String,Any}(string(k) => v for (k, v) in provenance),
    )
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
