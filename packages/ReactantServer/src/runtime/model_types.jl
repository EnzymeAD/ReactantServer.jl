# Per-model value types, defined before `registry.jl` so `ModelEntry` can hold them with precise
# `Union{T,Nothing}` fields rather than `Any`. The behavior lives elsewhere: `build_loaded_model`
# in `runtime/model.jl`, and the `ModelSchedState` scheduling methods in `scheduler.jl`.

# A compiled model's signature: its executable input/output names and element types, the weight
# argument order, and the network batch axis.
struct ModelSignature
    input_names::Vector{String}
    input_eltypes::Vector{DataType}
    weight_names::Vector{String}        # StableHLO argument order, following the inputs
    num_outputs::Int
    output_names::Vector{String}
    output_eltypes::Vector{DataType}
    batch_dim::Int                      # 0-based network batch axis (derived from input shape letters)
    variant_spec::Vector{Tuple{String,Int}}   # (input name, 1-based Julia axis) of each variable input
                                               # axis, in the manifest's input_shapes order; empty => one shape
end

# Single-shape convenience: no variable input axes. Preserves the 7-argument form used widely in
# tests and hand-built models.
ModelSignature(in_names, in_et, w_names, n_out, out_names, out_et, batch_dim) =
    ModelSignature(in_names, in_et, w_names, n_out, out_names, out_et, batch_dim, Tuple{String,Int}[])

num_parameters(s::ModelSignature) = length(s.input_names) + length(s.weight_names)

# A compiled, ready-to-serve model: its signature, its compiled executables (one per batch size),
# and its resident weight buffers.
#
# weights is the resident device (GPU) buffers (shared across sizes, in weight_names order), or
# `nothing` when the model's weights are currently evicted from the GPU. `state` is the residency
# floor an operator or control plane has set (see `ResidencyState`): `PINNED_DEVICE` models are
# kept resident and never evicted; `PINNED_SYSTEM` keeps host_weights resident and loads to the
# device on demand; `UNPINNED` keeps no floor. `nbytes` is the device footprint used for the
# cache budget. host_weights holds the materialized weight Arrays resident in host RAM; when
# present, an on-demand GPU load is a pure host->device transfer rather than a re-materialization
# from the mmap. It is `nothing` when no host floor is held (unpinned, the non-on-demand path,
# and the hand-built models used in tests). `execs` and the buffer vectors stay `Any`-typed
# because their elements are backend-opaque (mock vs Reactant executables and device buffers).
#
# `execs` is keyed first by input-shape variant (the variable-axis sizes, in `sig.variant_spec`
# order; the empty key `Int[]` is the single-shape default) and then by batch size (key 0 = a
# single unbatched module). Every variant and batch size shares the one `weights`/`host_weights`
# set, so compiling a model for several input shapes (e.g. detector aspect ratios) does not
# duplicate weights on the device.
const VariantKey = Vector{Int}

mutable struct LoadedModel
    sig::ModelSignature
    execs::Dict{VariantKey,Dict{Int,Any}}   # variant -> (batch size -> backend executable)
    weights::Union{Vector{Any},Nothing}
    state::ResidencyState
    nbytes::Int
    host_weights::Union{Vector{Any},Nothing}
end

# True when the model's weights are guaranteed resident on the device for the server's lifetime
# (exempt from eviction).
is_device_pinned(m::LoadedModel) = m.state == PINNED_DEVICE

# Wrap a flat batch-size -> executable map as the single default variant `Int[]`. Lets tests and
# hand-built models keep passing a `Dict{Int,Any}` while the field is variant-nested.
_as_variant_execs(execs::Dict{VariantKey,Dict{Int,Any}}) = execs
_as_variant_execs(execs::Dict{Int,Any}) = Dict{VariantKey,Dict{Int,Any}}(VariantKey() => execs)

# True when the model is a single unbatched module (its default variant carries the key-0 module).
_has_unbatched(m::LoadedModel) = any(inner -> haskey(inner, 0), values(m.execs))

# All compiled batch sizes across every variant (union), sorted. `[0]` means a single unbatched
# module.
_all_batch_sizes(m::LoadedModel) =
    sort!(unique(Iterators.flatten(keys(inner) for inner in values(m.execs))))

# Preserve the original three-argument form: resident, unpinned, footprint unknown (0), no host
# pinning. Used by tests that hand-build a model with weights already in place.
LoadedModel(sig::ModelSignature, execs::Union{Dict{Int,Any},Dict{VariantKey,Dict{Int,Any}}}, weights::Vector{Any}) =
    LoadedModel(sig, _as_variant_execs(execs), weights, UNPINNED, 0, nothing)

# Bool shims map the old `pinned` flag onto the residency state (true => PINNED_DEVICE,
# false => UNPINNED), preserving the five- and six-argument forms used by tests.
LoadedModel(sig::ModelSignature, execs::Union{Dict{Int,Any},Dict{VariantKey,Dict{Int,Any}}},
            weights::Union{Vector{Any},Nothing}, pinned::Bool, nbytes::Integer) =
    LoadedModel(sig, _as_variant_execs(execs), weights, pinned ? PINNED_DEVICE : UNPINNED, Int(nbytes), nothing)

LoadedModel(sig::ModelSignature, execs::Union{Dict{Int,Any},Dict{VariantKey,Dict{Int,Any}}},
            weights::Union{Vector{Any},Nothing}, pinned::Bool, nbytes::Integer,
            host_weights::Union{Vector{Any},Nothing}) =
    LoadedModel(sig, _as_variant_execs(execs), weights, pinned ? PINNED_DEVICE : UNPINNED, Int(nbytes), host_weights)

# Wrap a flat batch-size map passed with an explicit ResidencyState (the form hand-built models and
# observability tests use); the nested-execs form hits the default constructor directly.
LoadedModel(sig::ModelSignature, execs::Dict{Int,Any}, weights::Union{Vector{Any},Nothing},
            state::ResidencyState, nbytes::Integer, host_weights::Union{Vector{Any},Nothing}) =
    LoadedModel(sig, _as_variant_execs(execs), weights, state, Int(nbytes), host_weights)

# Per-model scheduler state. The EMA and cost fields drive the fair discipline only; under FIFO
# they are left untouched. The EMA fields are always decayed to "now" before being read or
# written so the value is current as of the read time. The scheduling methods live in
# `scheduler.jl`.
mutable struct ModelSchedState
    name::String
    weight::Float64                       # relative compute share (fair discipline); share = weight / Σ weights
    max_batch_size::Union{Int,Nothing}    # coalescing cap (rows per dispatch); nothing = uncapped
    queue::Vector{QueuedRequest}          # FIFO: front is oldest
    recent_compute_ema::Float64
    ema_last_update::Float64
    cost_estimate::Dict{Int,Float64}      # batch size -> estimated GPU time (seconds)
    # observability
    dispatch_count::Int                   # coalesced batch executions
    requests_served::Int                  # individual requests served (>= dispatch_count under coalescing)
    rows_served::Int                      # batch-axis rows processed; rows/dispatch = effective batch size (counts client-prebatched rows, not just server-coalesced requests)
    total_compute::Float64
    wait_samples::Vector{Float64}
    batch_size_hist::Dict{Int,Int}
end

function ModelSchedState(name::AbstractString, mc::ModelSchedConfig, now::Float64)
    return ModelSchedState(String(name), mc.weight, mc.max_batch_size, QueuedRequest[],
        0.0, now, Dict{Int,Float64}(), 0, 0, 0, 0.0, Float64[], Dict{Int,Int}())
end
