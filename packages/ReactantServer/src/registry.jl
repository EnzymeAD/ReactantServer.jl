# Model registry and the register_model extension API used by a bundle's model.jl.
#
# A ModelEntry is the single per-model object: its static bundle data plus the compiled runtime
# (`executable`) and the scheduling state (`sched`), which are both filled after construction
# (`executable` by the runtime once compiled, `sched` when the scheduler prepares the entry).
# The registry maps name -> entry and is the single source of truth for which models exist;
# the scheduler reads `entry.sched` rather than keeping a parallel map. Mutations to the map go
# through the scheduler's condition lock (see `admit!`/`evict!`).

# Collected when a bundle's model.jl calls register_model. The bundle loader sets up a
# fresh slot, includes model.jl, then reads the result back.
struct Registration
    name::String
    preprocess::Function
    postprocess::Function
end

const _CURRENT_REGISTRATION = Ref{Union{Registration,Nothing}}(nothing)

"""
    register_model(name; preprocess=identity, postprocess=identity)

Called from a bundle's model.jl to register custom pre/post-processing. Both hooks
receive and return a `Vector{NamedTensor}`. Omitted hooks default to identity.
"""
function register_model(name::AbstractString; preprocess::Function=identity, postprocess::Function=identity)
    _CURRENT_REGISTRATION[] = Registration(String(name), preprocess, postprocess)
    return nothing
end

# Collected when a meta bundle's model.jl calls register_meta_model. Mirrors Registration but
# carries the orchestration function instead of pre/post hooks.
struct MetaRegistration
    name::String
    run::Function
end

const _CURRENT_META_REGISTRATION = Ref{Union{MetaRegistration,Nothing}}(nothing)

"""
    register_meta_model(name; run)

Called from a meta bundle's model.jl to register the orchestration function. `run` has the form
`run(inputs::Vector{NamedTensor}, call) -> Vector{NamedTensor}`, where `call(model_name, inputs)`
invokes another model (routed to the gateway in multi-worker mode, or the local worker otherwise).
"""
function register_meta_model(name::AbstractString; run::Function)
    _CURRENT_META_REGISTRATION[] = MetaRegistration(String(name), run)
    return nothing
end

mutable struct ModelEntry
    name::String
    manifest::Manifest
    mlir_bytes::Dict{Int,Vector{UInt8}}  # batch size -> StableHLO portable artifact; key 0 = single unbatched module
    weights_path::String
    weights::Any                          # SafeTensors handle (mmap), kept lazy; backend-opaque
    executable::Union{LoadedModel,Nothing}   # compiled runtime + residency; `nothing` until compiled
    sched::Union{ModelSchedState,Nothing}    # scheduling state; `nothing` until the scheduler prepares it
    preprocess::Function
    postprocess::Function
end

# A meta model: a Julia orchestration over other models. It has no executable, weights, or
# scheduling state; the gRPC layer runs `run` directly on the request task (see meta.jl).
mutable struct MetaEntry
    name::String
    manifest::Manifest
    calls::Vector{String}   # declared sub-model names (manifest meta.calls)
    run::Function
end

struct ModelRegistry
    by_name::Dict{String,ModelEntry}
    meta::Dict{String,MetaEntry}   # meta models, kept separate so the compile/scheduler paths never see them
end
ModelRegistry() = ModelRegistry(Dict{String,ModelEntry}(), Dict{String,MetaEntry}())

get_model(reg::ModelRegistry, name::AbstractString) = get(reg.by_name, name, nothing)
get_meta(reg::ModelRegistry, name::AbstractString) = get(reg.meta, name, nothing)
is_meta_name(reg::ModelRegistry, name::AbstractString) = haskey(reg.meta, name)
# Every servable name: regular models plus meta models. The two namespaces are disjoint
# (enforced at load), so a simple union is sufficient.
model_names(reg::ModelRegistry) = sort!(collect(union(keys(reg.by_name), keys(reg.meta))))
