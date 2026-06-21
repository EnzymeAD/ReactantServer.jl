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
invokes another model. The meta runs as a scheduled unit holding the GPU exclusively, and `call`
invokes the sub-model's compiled executable directly in-process (no queue re-entry, no gateway hop).
"""
function register_meta_model(name::AbstractString; run::Function)
    _CURRENT_META_REGISTRATION[] = MetaRegistration(String(name), run)
    return nothing
end

# Both a regular model and a meta model are units the dispatch loop can schedule (both carry a
# `sched::ModelSchedState` queue once prepared). The abstract supertype lets selection compare them
# on the same scheduling key without forcing a Union into the type-stable per-entry hot loops, which
# iterate the concrete `by_name` / `meta` dicts separately.
abstract type AbstractDispatchEntry end

mutable struct ModelEntry <: AbstractDispatchEntry
    name::String
    manifest::Manifest
    mlir_bytes::Dict{VariantKey,Dict{Int,Vector{UInt8}}}  # variant -> (batch size -> StableHLO artifact); batch key 0 = single unbatched module
    weights_path::String
    weights::Any                          # SafeTensors handle (mmap), kept lazy; backend-opaque
    executable::Union{LoadedModel,Nothing}   # compiled runtime + residency; `nothing` until compiled
    sched::Union{ModelSchedState,Nothing}    # scheduling state; `nothing` until the scheduler prepares it
    preprocess::Function
    postprocess::Function
end

# Convenience constructor accepting a flat batch-size -> bytes map, wrapped as the single default
# input-shape variant `Int[]`. Keeps hand-built entries (tests, fixtures) working while the field
# is variant-nested; the loader passes the nested map directly.
ModelEntry(name, manifest, mlir_bytes::Dict{Int,Vector{UInt8}}, weights_path, weights,
           executable, sched, preprocess, postprocess) =
    ModelEntry(name, manifest, Dict{VariantKey,Dict{Int,Vector{UInt8}}}(VariantKey() => mlir_bytes),
               weights_path, weights, executable, sched, preprocess, postprocess)

# A meta model: a Julia orchestration over other models. It owns no compiled executable or weights and
# is not scheduled on the dispatch loop. Its `run` executes on the gRPC request task under a per-worker
# gate (one meta at a time), and each sub-call re-enters the scheduler in-process for the sub-model (see
# meta.jl `QueueingCaller`, scheduler.jl `_run_meta_request`). `sched` is filled when the scheduler
# prepares the entry and holds only its serving counters for the control plane (it has no dispatch queue).
mutable struct MetaEntry <: AbstractDispatchEntry
    name::String
    manifest::Manifest
    calls::Vector{String}   # declared sub-model names (manifest meta.calls)
    run::Function
    sched::Union{ModelSchedState,Nothing}   # scheduling state; `nothing` until the scheduler prepares it
end
# Existing call sites (the loader, tests) build a MetaEntry without scheduling state.
MetaEntry(name, manifest, calls, run) = MetaEntry(name, manifest, calls, run, nothing)

struct ModelRegistry
    by_name::Dict{String,ModelEntry}
    meta::Dict{String,MetaEntry}   # meta models, kept separate so the compile/scheduler paths never see them
end
ModelRegistry() = ModelRegistry(Dict{String,ModelEntry}(), Dict{String,MetaEntry}())

get_model(reg::ModelRegistry, name::AbstractString) = get(reg.by_name, name, nothing)
get_meta(reg::ModelRegistry, name::AbstractString) = get(reg.meta, name, nothing)
is_meta_name(reg::ModelRegistry, name::AbstractString) = haskey(reg.meta, name)
# Every registered name: regular models (including the internal sub-models a meta calls) plus meta
# models. The two namespaces are disjoint (enforced at load).
model_names(reg::ModelRegistry) = sort!(collect(union(keys(reg.by_name), keys(reg.meta))))

# Names of regular models that exist ONLY as a meta's internal stage (declared in some meta's
# `calls`). A meta calls these directly in-process, so they are never independently routed, placed,
# or discovered; the gateway should not see them. The union covers a sub-model shared by several metas.
function internal_submodels(reg::ModelRegistry)
    subs = Set{String}()
    for m in values(reg.meta)
        union!(subs, m.calls)
    end
    return subs
end

# Names the gateway may route to: meta models and standalone regular models, with internal
# sub-models hidden. Used by the RepositoryIndex discovery RPC.
function routable_model_names(reg::ModelRegistry)
    subs = internal_submodels(reg)
    return sort!(collect(Iterators.filter(n -> !(n in subs),
                                          union(keys(reg.by_name), keys(reg.meta)))))
end
