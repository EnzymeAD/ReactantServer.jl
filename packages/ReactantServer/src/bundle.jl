# Discovery and loading of model bundles from the configured model directories.
#
# A bundle is a directory containing manifest.yaml, model.mlir, weights.safetensors, and
# an optional model.jl. This module turns each into a registered ModelEntry. The runtime
# compiles the StableHLO and fills the executable slot afterwards.

struct BundleError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BundleError) = print(io, "BundleError: ", e.msg)

# Include a bundle's model.jl in an isolated module so it cannot clobber server globals. Only
# register_model, register_meta_model, and the ReactantServer module are injected. When `meta` is
# true the bundle must call register_meta_model; otherwise it must call register_model.
function _include_model_jl(path::AbstractString, expected_name::AbstractString; meta::Bool=false)
    _CURRENT_REGISTRATION[] = nothing
    _CURRENT_META_REGISTRATION[] = nothing
    sandbox = Module(gensym(:bundle))
    Core.eval(sandbox, :(const register_model = $register_model))
    Core.eval(sandbox, :(const register_meta_model = $register_meta_model))
    Core.eval(sandbox, :(const ReactantServer = $(@__MODULE__)))
    Base.include(sandbox, String(path))
    reg = _CURRENT_REGISTRATION[]
    mreg = _CURRENT_META_REGISTRATION[]
    _CURRENT_REGISTRATION[] = nothing
    _CURRENT_META_REGISTRATION[] = nothing
    if meta
        mreg === nothing && throw(BundleError("model.jl for meta model '$expected_name' did not call register_meta_model"))
        reg === nothing || throw(BundleError("meta model '$expected_name' must call register_meta_model, not register_model"))
        mreg.name == expected_name ||
            throw(BundleError("model.jl registered meta model '$(mreg.name)' but bundle directory is '$expected_name'"))
        return mreg
    end
    reg === nothing && throw(BundleError("model.jl for '$expected_name' did not call register_model"))
    mreg === nothing || throw(BundleError("model '$expected_name' must call register_model, not register_meta_model"))
    reg.name == expected_name ||
        throw(BundleError("model.jl registered '$(reg.name)' but bundle directory is '$expected_name'"))
    return reg
end

# Discover the per-batch-size StableHLO modules for one variant prefix. The prefix is `model` for
# a single-shape bundle and `model.v{i}` for variant `i` of a multi-shape bundle. A variant has
# either per-batch-size files `<prefix>.b{N}.mlir` (keyed by N) or a single `<prefix>.mlir`
# (keyed by 0, used for any batch size).
function _discover_batch_modules(dir::AbstractString, m::Manifest, prefix::AbstractString)
    modules = Dict{Int,Vector{UInt8}}()
    rx = Regex("^" * replace(prefix, "." => "\\.") * "\\.b(\\d+)\\.mlir\$")
    for f in readdir(dir)
        mt = match(rx, f)
        mt === nothing && continue
        modules[parse(Int, mt.captures[1])] = read(joinpath(dir, f))
    end
    if isempty(modules)
        single = joinpath(dir, prefix * ".mlir")
        isfile(single) ||
            throw(BundleError("bundle '$(m.name)' has no $(prefix).mlir or $(prefix).b{N}.mlir"))
        modules[0] = read(single)
        return modules
    end
    for sz in m.batching.compiled_batch_sizes
        haskey(modules, sz) ||
            throw(BundleError("bundle '$(m.name)' declares batch size $sz but has no $(prefix).b$sz.mlir"))
    end
    return modules
end

# Discover every variant's StableHLO module(s), keyed by variant. A single-shape bundle (no
# `input_shapes`) yields one default variant `Int[]` from `model.mlir`/`model.b{N}.mlir`. A
# multi-shape bundle yields one entry per declared `input_shapes` variant `i`, read from
# `model.v{i}.*.mlir`; the variant key is the same variable-axis size vector the manifest resolved
# and the runtime derives from a request, so dispatch lines up with what was compiled.
function _discover_modules(dir::AbstractString, m::Manifest)
    if isempty(m.input_shapes)
        return Dict{VariantKey,Dict{Int,Vector{UInt8}}}(VariantKey() => _discover_batch_modules(dir, m, "model"))
    end
    out = Dict{VariantKey,Dict{Int,Vector{UInt8}}}()
    for (i, vkey) in enumerate(m.input_shapes)
        out[vkey] = _discover_batch_modules(dir, m, "model.v$(i - 1)")
    end
    return out
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

    # A meta bundle has no StableHLO module or weights: it is just a manifest + model.jl
    # orchestration. validate_manifest already enforced model.jl presence and the meta block.
    if is_meta(m)
        mreg = _include_model_jl(model_jl, m.name; meta=true)
        return MetaEntry(m.name, m, m.meta_calls, mreg.run)
    end

    mlir_bytes = _discover_modules(dir, m)

    weights_path = joinpath(dir, "weights.safetensors")
    isfile(weights_path) || throw(BundleError("bundle '$(m.name)' missing weights.safetensors"))
    weights = SafeTensors.deserialize(weights_path; mmap=true)

    # Every variant/batch module shares one signature; validate against any one of them.
    validate_against_signature(validator, m, first(values(first(values(mlir_bytes)))))

    pre, post = identity, identity
    if has_jl
        r = _include_model_jl(model_jl, m.name)
        pre, post = r.preprocess, r.postprocess
    end

    return ModelEntry(m.name, m, mlir_bytes, weights_path, weights, nothing, nothing, pre, post)
end

function _load_one_bundle!(reg::ModelRegistry, dir::AbstractString, validator::SignatureValidator)
    entry = load_bundle_entry(dir; validator=validator)
    (haskey(reg.by_name, entry.name) || haskey(reg.meta, entry.name)) &&
        throw(BundleError("duplicate model name '$(entry.name)'"))
    if entry isa MetaEntry
        reg.meta[entry.name] = entry
    else
        reg.by_name[entry.name] = entry
    end
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
    _validate_meta_calls(reg)
    return reg
end

# Cross-bundle validation of every meta model's declared calls. A meta model may not call another
# meta model (prevents server-side pipeline recursion); a backbone that is not loaded locally is
# allowed (in a multi-worker deployment it lives on another worker, reached via the gateway), but
# is logged so a typo is visible.
function _validate_meta_calls(reg::ModelRegistry)
    for entry in values(reg.meta)
        for callee in entry.calls
            haskey(reg.meta, callee) &&
                throw(BundleError("meta model '$(entry.name)' calls '$callee', which is also a meta model; " *
                                  "meta models may not call other meta models"))
            haskey(reg.by_name, callee) ||
                @info "meta model declares a call to a model not loaded on this worker (expected in multi-worker deployments)" meta = entry.name callee = callee
        end
    end
    return nothing
end
