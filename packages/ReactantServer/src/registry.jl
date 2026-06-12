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

struct ModelRegistry
    by_name::Dict{String,ModelEntry}
end
ModelRegistry() = ModelRegistry(Dict{String,ModelEntry}())

get_model(reg::ModelRegistry, name::AbstractString) = get(reg.by_name, name, nothing)
model_names(reg::ModelRegistry) = sort!(collect(keys(reg.by_name)))
