# Parsing and validation of a bundle's manifest.yaml.
#
# Two phases: parse_manifest performs structural parsing, validate_manifest enforces
# the cross-field rules from the bundle specification. Validation against the actual
# StableHLO signature is a separate concern handled through the SignatureValidator seam.
#
# Shape encoding (format_version 2.0):
#   Each tensor's shape is an einsum-style string of single ASCII letters, one per
#   axis. A companion `dims:` map gives the size of every non-batch letter:
#       shape: "chwn"
#       dims:
#         c: 3
#         h: 224
#         w: 224
#   Letters `n` and `b` are reserved batch markers; at most one occurrence per tensor.
#   Other letters are tensor-scoped (no implicit cross-tensor equality) and must be
#   unique within a single shape. A size of -1 in `dims` marks a variable axis (used
#   only on non-batch axes; today only client_outputs can carry one). The per-input
#   batch_dim is derived from the position of n/b in the shape string; the top-level
#   `batching` block carries only `compiled_batch_sizes`.

const SUPPORTED_FORMAT_VERSIONS = ("2.0", "2")

# A bundle is either a regular "model" (a StableHLO executable + weights) or a "meta" model: a
# user-authored Julia workflow (model.jl) that chains other models with data-dependent logic in
# between. A meta bundle carries no executable/weights and declares the models it calls. The call
# list may be empty: a compute-only meta model does all its work in Julia and invokes no sub-model
# (e.g. a small model reimplemented in pure Julia, shipping its own weights.safetensors).
const SUPPORTED_KINDS = ("model", "meta")

const _RESERVED_BATCH_LETTERS = ('n', 'b')

struct ManifestError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ManifestError) = print(io, "ManifestError: ", e.msg)

@enum DimKind FIXED BATCH VARIABLE

"""
    Dim

A single axis of a tensor shape. `kind` is one of `FIXED`, `BATCH`, or `VARIABLE`; `size` is
meaningful only when `kind == FIXED` (it is `0` otherwise). A `FIXED` dim has a concrete size,
a `BATCH` dim is the batch axis (from the reserved `n`/`b` shape letters), and a `VARIABLE`
dim (a `-1` in the manifest `dims` map) is a dynamic non-batch axis.
"""
struct Dim
    kind::DimKind
    size::Int            # meaningful only when kind == FIXED
end
Dim(kind::DimKind) = Dim(kind, 0)

"""
    TensorSpec

One tensor in a [`Manifest`](@ref): its `name`, [`DType`](@ref), `shape` (a `Vector{Dim}`),
and `batch_axis` (the 1-based index of the batch dim, or `nothing` when the tensor is not
batched).
"""
struct TensorSpec
    name::String
    dtype::DType
    shape::Vector{Dim}
    batch_axis::Union{Int,Nothing}   # 1-based index of the batch dim, or nothing
end

"""
    BatchingSpec

The set of batch sizes a bundle was compiled for (`compiled_batch_sizes`). At inference the
request's size along the batch axis must equal one of these; the scheduler coalesces requests
up to a compiled size and selects the matching executable.
"""
struct BatchingSpec
    compiled_batch_sizes::Vector{Int}
end

struct Provenance
    fields::Dict{String,Any}
end

"""
    Manifest

The parsed and validated `manifest.yaml` of a model bundle. It records the `format_version`,
the bundle `name` and `description`, the executable input/output specs
(`executable_inputs`/`executable_outputs`), the optional client-facing specs
(`client_inputs`/`client_outputs`, present only when a `model.jl` transforms the I/O), the
[`BatchingSpec`](@ref), provenance metadata, and the derived 0-based `input_batch_dim`. Tensor
specs are [`TensorSpec`](@ref) values; see [`TensorSpec`](@ref) and [`Dim`](@ref) for the
einsum-style shape encoding.
"""
struct Manifest
    format_version::String
    name::String
    description::String
    executable_inputs::Vector{TensorSpec}
    executable_outputs::Vector{TensorSpec}
    client_inputs::Union{Vector{TensorSpec},Nothing}
    client_outputs::Union{Vector{TensorSpec},Nothing}
    batching::BatchingSpec
    provenance::Provenance
    input_batch_dim::Union{Int,Nothing}   # derived: 0-based batch axis of the inputs (or nothing)
    kind::String                          # "model" (default) or "meta"
    meta_calls::Vector{String}            # for kind=="meta": declared sub-model names; empty otherwise
end

# Backward-compatible 10-arg constructor: a regular model with no meta metadata. Keeps existing
# positional callers (tests, fixtures) working after the kind/meta_calls fields were appended.
Manifest(fv, name, desc, exin, exout, clin, clout, batching, prov, ibd) =
    Manifest(fv, name, desc, exin, exout, clin, clout, batching, prov, ibd, "model", String[])

"""
    is_meta(m::Manifest) -> Bool

True when the bundle is a meta model (a Julia orchestration over other models) rather than a
regular StableHLO executable.
"""
is_meta(m::Manifest) = m.kind == "meta"

_is_ascii_letter(c::Char) = ('a' <= c <= 'z') || ('A' <= c <= 'Z')
_is_reserved_batch(c::Char) = c in _RESERVED_BATCH_LETTERS

function parse_shape(shape_str::AbstractString, dims_map, tensor_name::AbstractString)
    dims_map isa AbstractDict ||
        throw(ManifestError("tensor '$tensor_name' has 'dims' that is not a mapping (got $(typeof(dims_map)))"))

    chars = collect(shape_str)
    for c in chars
        _is_ascii_letter(c) ||
            throw(ManifestError("tensor '$tensor_name' shape '$shape_str' contains non-letter character '$c'"))
    end
    length(unique(chars)) == length(chars) ||
        throw(ManifestError("tensor '$tensor_name' shape '$shape_str' has duplicate axis letters"))

    batch_letters = filter(_is_reserved_batch, chars)
    length(batch_letters) <= 1 ||
        throw(ManifestError("tensor '$tensor_name' has more than one batch axis ('n'/'b') in shape '$shape_str'"))

    # Normalize dims_map keys to single-char strings for lookup.
    dims_lookup = Dict{Char,Int}()
    for (k, v) in dims_map
        ks = string(k)
        length(ks) == 1 && _is_ascii_letter(ks[1]) ||
            throw(ManifestError("tensor '$tensor_name' dims key '$k' is not a single ASCII letter"))
        c = ks[1]
        _is_reserved_batch(c) &&
            throw(ManifestError("tensor '$tensor_name' dims key '$c' is reserved for the batch axis"))
        v isa Integer ||
            throw(ManifestError("tensor '$tensor_name' dims['$c'] is not an integer (got $(typeof(v)))"))
        size = Int(v)
        (size > 0 || size == -1) ||
            throw(ManifestError("tensor '$tensor_name' dims['$c']=$size; must be > 0 or -1"))
        dims_lookup[c] = size
    end

    shape_letters = Set(c for c in chars if !_is_reserved_batch(c))
    orphan_keys = setdiff(keys(dims_lookup), shape_letters)
    isempty(orphan_keys) ||
        throw(ManifestError("tensor '$tensor_name' has dims entries not in shape: $(sort(collect(orphan_keys)))"))

    result = Dim[]
    for c in chars
        if _is_reserved_batch(c)
            push!(result, Dim(BATCH))
        else
            haskey(dims_lookup, c) ||
                throw(ManifestError("tensor '$tensor_name' shape uses '$c' but dims has no entry for it"))
            sz = dims_lookup[c]
            push!(result, sz == -1 ? Dim(VARIABLE) : Dim(FIXED, sz))
        end
    end
    return result
end

function _find_batch_axis(name::AbstractString, dims::Vector{Dim})
    idxs = findall(d -> d.kind == BATCH, dims)
    length(idxs) <= 1 ||
        throw(ManifestError("tensor '$name' has more than one batch axis ('n'/'b')"))
    return isempty(idxs) ? nothing : idxs[1]
end

function parse_tensor_spec(d)
    d isa AbstractDict || throw(ManifestError("tensor entry must be a mapping, got $(typeof(d))"))
    name = get(d, "name", nothing)
    name isa AbstractString || throw(ManifestError("tensor entry missing string 'name'"))
    tok = get(d, "dtype", nothing)
    tok isa AbstractString || throw(ManifestError("tensor '$name' missing string 'dtype'"))
    haskey(DTYPE_FROM_TOKEN, tok) || throw(ManifestError("tensor '$name' has unknown dtype '$tok'"))
    shp = get(d, "shape", nothing)
    shp isa AbstractString || throw(ManifestError("tensor '$name' missing string 'shape' (einsum letters)"))
    dims_map = get(d, "dims", Dict{String,Any}())
    dims = parse_shape(String(shp), dims_map, String(name))
    return TensorSpec(String(name), DTYPE_FROM_TOKEN[tok], dims, _find_batch_axis(name, dims))
end

function parse_tensor_list(v, section::AbstractString)
    v isa AbstractVector || throw(ManifestError("'$section' must be a list"))
    return TensorSpec[parse_tensor_spec(x) for x in v]
end

function parse_batching(b)
    b isa AbstractDict || throw(ManifestError("'batching' must be a mapping"))
    sizes = get(b, "compiled_batch_sizes", Int[])
    sizes isa AbstractVector || throw(ManifestError("'batching.compiled_batch_sizes' must be a list"))
    return BatchingSpec(Int[Int(x) for x in sizes])
end

# Cross-input consistency: every executable input that carries a batch axis must
# place it at the same 0-based position. The derived input_batch_dim is that
# position, or nothing if no input has a batch axis.
function _derive_input_batch_dim(inputs::Vector{TensorSpec})
    bd::Union{Int,Nothing} = nothing
    for t in inputs
        t.batch_axis === nothing && continue
        axis0 = t.batch_axis - 1
        if bd === nothing
            bd = axis0
        elseif bd != axis0
            throw(ManifestError("executable_inputs disagree on batch axis position: " *
                                "tensor '$(t.name)' has batch at $axis0, expected $bd"))
        end
    end
    return bd
end

# Parse the `meta` block of a meta manifest into the declared list of called sub-model names.
function _parse_meta_calls(d::AbstractDict, name::AbstractString)
    blk = get(d, "meta", nothing)
    blk isa AbstractDict || throw(ManifestError("meta manifest '$name' missing 'meta' mapping"))
    calls = get(blk, "calls", nothing)
    calls isa AbstractVector || throw(ManifestError("meta manifest '$name' 'meta.calls' must be a list"))
    out = String[]
    for c in calls
        c isa AbstractString ||
            throw(ManifestError("meta manifest '$name' 'meta.calls' entries must be strings (got $(typeof(c)))"))
        push!(out, String(c))
    end
    return out
end

function parse_manifest(d::AbstractDict)
    name = get(d, "name", nothing)
    name isa AbstractString || throw(ManifestError("manifest missing string 'name'"))
    fv = get(d, "format_version", nothing)
    fv === nothing && throw(ManifestError("manifest '$name' missing 'format_version'"))
    desc = get(d, "description", "")

    kind = get(d, "kind", "model")
    kind isa AbstractString || throw(ManifestError("manifest '$name' field 'kind' must be a string"))
    kind = String(kind)
    meta = kind == "meta"
    meta_calls = meta ? _parse_meta_calls(d, String(name)) : String[]

    # A meta bundle has no StableHLO executable, so its executable specs are optional (default
    # empty); its external I/O is carried by client_inputs/client_outputs instead. A regular model
    # must declare both executable sections.
    if meta
        exin = haskey(d, "executable_inputs") ? parse_tensor_list(d["executable_inputs"], "executable_inputs") : TensorSpec[]
        exout = haskey(d, "executable_outputs") ? parse_tensor_list(d["executable_outputs"], "executable_outputs") : TensorSpec[]
    else
        haskey(d, "executable_inputs") || throw(ManifestError("manifest '$name' missing 'executable_inputs'"))
        haskey(d, "executable_outputs") || throw(ManifestError("manifest '$name' missing 'executable_outputs'"))
        exin = parse_tensor_list(d["executable_inputs"], "executable_inputs")
        exout = parse_tensor_list(d["executable_outputs"], "executable_outputs")
    end

    clin = haskey(d, "client_inputs") ? parse_tensor_list(d["client_inputs"], "client_inputs") : nothing
    clout = haskey(d, "client_outputs") ? parse_tensor_list(d["client_outputs"], "client_outputs") : nothing

    batching = parse_batching(get(d, "batching", Dict{String,Any}()))
    prov_raw = get(d, "provenance", Dict{String,Any}())
    prov = Provenance(prov_raw isa AbstractDict ? Dict{String,Any}(prov_raw) : Dict{String,Any}())

    input_batch_dim = _derive_input_batch_dim(exin)

    return Manifest(string(fv), String(name), desc === nothing ? "" : string(desc),
                    exin, exout, clin, clout, batching, prov, input_batch_dim, kind, meta_calls)
end

function _check_unique_names(tensors::Vector{TensorSpec}, section::AbstractString)
    seen = Set{String}()
    for t in tensors
        t.name in seen && throw(ManifestError("duplicate tensor name '$(t.name)' in '$section'"))
        push!(seen, t.name)
    end
end

function validate_manifest(m::Manifest, dir::AbstractString, has_model_jl::Bool)
    m.format_version in SUPPORTED_FORMAT_VERSIONS ||
        throw(ManifestError("unsupported format_version '$(m.format_version)'; " *
                            "supported: $(join(SUPPORTED_FORMAT_VERSIONS, ", "))"))

    m.kind in SUPPORTED_KINDS ||
        throw(ManifestError("unsupported kind '$(m.kind)'; supported: $(join(SUPPORTED_KINDS, ", "))"))

    expected = basename(normpath(String(dir)))
    m.name == expected ||
        throw(ManifestError("manifest name '$(m.name)' does not match directory name '$expected'"))

    if is_meta(m)
        has_model_jl ||
            throw(ManifestError("meta model '$(m.name)' requires a model.jl that calls register_meta_model"))
        (m.client_inputs !== nothing && !isempty(m.client_inputs)) ||
            throw(ManifestError("meta model '$(m.name)' must declare non-empty client_inputs"))
        (m.client_outputs !== nothing && !isempty(m.client_outputs)) ||
            throw(ManifestError("meta model '$(m.name)' must declare non-empty client_outputs"))
        # An empty meta.calls list is allowed: a compute-only meta model invokes no sub-model.
        allunique(m.meta_calls) ||
            throw(ManifestError("meta model '$(m.name)' has duplicate entries in meta.calls"))
        m.name in m.meta_calls &&
            throw(ManifestError("meta model '$(m.name)' lists itself in meta.calls"))
    end

    _check_unique_names(m.executable_inputs, "executable_inputs")
    _check_unique_names(m.executable_outputs, "executable_outputs")
    # Cross-input batch-axis consistency is enforced at parse time via
    # _derive_input_batch_dim. Outputs are deliberately not cross-checked:
    # classifiers and detectors routinely have lower-rank outputs than inputs.

    all(>(0), m.batching.compiled_batch_sizes) ||
        throw(ManifestError("batching.compiled_batch_sizes must be positive integers"))

    if !has_model_jl
        (m.client_inputs === nothing && m.client_outputs === nothing) ||
            throw(ManifestError("client_inputs/client_outputs are only valid when model.jl is present"))
    end
    m.client_inputs === nothing || _check_unique_names(m.client_inputs, "client_inputs")
    m.client_outputs === nothing || _check_unique_names(m.client_outputs, "client_outputs")

    # The KServe V2 wire protocol can only advertise dtypes that have a wire mapping, so
    # client-facing tensors using an unmappable dtype (e.g. FP8) would fail when a response or
    # ModelMetadata message is encoded. Reject them at load time rather than mid-request.
    # Executable-internal dtypes are unconstrained.
    for (section, specs) in (("client_inputs", client_input_spec(m)),
                             ("client_outputs", client_output_spec(m)))
        for t in specs
            haskey(DTYPE_TO_KSERVE, t.dtype) ||
                throw(ManifestError("$section tensor '$(t.name)' has dtype $(dtype_token(t.dtype)) " *
                                    "which has no KServe wire datatype mapping"))
        end
    end
    return m
end

# Client-facing specs default to the executable specs when no model.jl transforms them.
client_input_spec(m::Manifest) = m.client_inputs === nothing ? m.executable_inputs : m.client_inputs
client_output_spec(m::Manifest) = m.client_outputs === nothing ? m.executable_outputs : m.client_outputs

"""
    load_manifest(path) -> Manifest

Parse a manifest YAML file at `path` into a [`Manifest`](@ref). This runs the structural parsing
and validation of `parse_manifest` but not the bundle-directory checks in
`validate_manifest`, so it is usable wherever only the manifest's contents are needed, for example
a client deriving a model's I/O spec offline.
"""
function load_manifest(path::AbstractString)
    raw = YAML.load_file(String(path); dicttype = Dict{String,Any})
    raw isa AbstractDict || throw(ManifestError("manifest at $path is not a mapping"))
    return parse_manifest(raw)
end
