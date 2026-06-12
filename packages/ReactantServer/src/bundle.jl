# Discovery and loading of model bundles from the configured model directories.
#
# A bundle is a directory containing manifest.yaml, model.mlir, weights.safetensors, and
# an optional model.jl. This module turns each into a registered ModelEntry. The runtime
# compiles the StableHLO and fills the executable slot afterwards.

struct BundleError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BundleError) = print(io, "BundleError: ", e.msg)

# Include a bundle's model.jl in an isolated module so it cannot clobber server globals.
# Only register_model and the ReactantServer module are injected.
function _include_model_jl(path::AbstractString, expected_name::AbstractString)
    _CURRENT_REGISTRATION[] = nothing
    sandbox = Module(gensym(:bundle))
    Core.eval(sandbox, :(const register_model = $register_model))
    Core.eval(sandbox, :(const ReactantServer = $(@__MODULE__)))
    Base.include(sandbox, String(path))
    reg = _CURRENT_REGISTRATION[]
    _CURRENT_REGISTRATION[] = nothing
    reg === nothing && throw(BundleError("model.jl for '$expected_name' did not call register_model"))
    reg.name == expected_name ||
        throw(BundleError("model.jl registered '$(reg.name)' but bundle directory is '$expected_name'"))
    return reg
end

# Discover the StableHLO module(s). A bundle has either per-batch-size files
# `model.b{N}.mlir` (keyed by N) or a single `model.mlir` (keyed by 0, used for any request).
function _discover_modules(dir::AbstractString, m::Manifest)
    modules = Dict{Int,Vector{UInt8}}()
    for f in readdir(dir)
        mt = match(r"^model\.b(\d+)\.mlir$", f)
        mt === nothing && continue
        modules[parse(Int, mt.captures[1])] = read(joinpath(dir, f))
    end
    if isempty(modules)
        single = joinpath(dir, "model.mlir")
        isfile(single) || throw(BundleError("bundle '$(m.name)' has no model.mlir or model.b{N}.mlir"))
        modules[0] = read(single)
        return modules
    end
    for sz in m.batching.compiled_batch_sizes
        haskey(modules, sz) ||
            throw(BundleError("bundle '$(m.name)' declares batch size $sz but has no model.b$sz.mlir"))
    end
    return modules
end

"""
    load_bundle_entry(dir; validator=NullSignatureValidator()) -> ModelEntry

Parse and validate the bundle directory `dir` into an uncompiled `ModelEntry` (its `executable`
and `sched` slots are `nothing`). The directory name is *not* enforced to equal the manifest
`name` here; that check is `load_bundles`'s responsibility (it filters by directory name). Used by
both `load_bundles` and the directory watcher (see watcher.jl) to load a single bundle.
"""
function load_bundle_entry(dir::AbstractString; validator::SignatureValidator=NullSignatureValidator())
    manifest_path = joinpath(dir, "manifest.yaml")
    raw = YAML.load_file(manifest_path; dicttype=Dict{String,Any})
    raw isa AbstractDict || throw(BundleError("manifest in $dir is not a mapping"))
    m = parse_manifest(raw)

    model_jl = joinpath(dir, "model.jl")
    has_jl = isfile(model_jl)
    validate_manifest(m, dir, has_jl)

    mlir_bytes = _discover_modules(dir, m)

    weights_path = joinpath(dir, "weights.safetensors")
    isfile(weights_path) || throw(BundleError("bundle '$(m.name)' missing weights.safetensors"))
    weights = SafeTensors.deserialize(weights_path; mmap=true)

    validate_against_signature(validator, m, first(values(mlir_bytes)))

    pre, post = identity, identity
    if has_jl
        r = _include_model_jl(model_jl, m.name)
        pre, post = r.preprocess, r.postprocess
    end

    return ModelEntry(m.name, m, mlir_bytes, weights_path, weights, nothing, nothing, pre, post)
end

function _load_one_bundle!(reg::ModelRegistry, dir::AbstractString, validator::SignatureValidator)
    entry = load_bundle_entry(dir; validator=validator)
    haskey(reg.by_name, entry.name) && throw(BundleError("duplicate model name '$(entry.name)'"))
    reg.by_name[entry.name] = entry
    return nothing
end

"""
    load_bundles(model_dirs; validator=NullSignatureValidator(), include=nothing) -> ModelRegistry

Discover every subdirectory containing a manifest.yaml under each model dir, load and
validate it, and register it. The runtime fills each entry's executable slot afterwards.

When `include` is a non-empty collection of model names, only bundles whose directory name
is in the set are loaded. The directory name equals the manifest `name` (enforced by
`validate_manifest`), so filtering by directory avoids parsing skipped manifests. Names in
`include` that are not found in any model dir produce a warning.
"""
function load_bundles(model_dirs::AbstractVector{<:AbstractString};
                      validator::SignatureValidator=NullSignatureValidator(),
                      include=nothing)
    want = include === nothing ? nothing : Set{String}(String(x) for x in include)
    reg = ModelRegistry()
    found = Set{String}()
    for root in model_dirs
        isdir(root) || throw(BundleError("model dir does not exist: $root"))
        for child in readdir(root; join=true)
            isdir(child) || continue
            isfile(joinpath(child, "manifest.yaml")) || continue
            name = basename(normpath(child))
            if want !== nothing && !(name in want)
                continue
            end
            _load_one_bundle!(reg, child, validator)
            push!(found, name)
        end
    end
    if want !== nothing
        missing_names = setdiff(want, found)
        isempty(missing_names) ||
            @warn "requested models not found in any model dir" missing = sort!(collect(missing_names)) model_dirs = model_dirs
    end
    return reg
end
