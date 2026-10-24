# Per-model device footprint and allocator fragmentation: the measurement layer that weight
# admission reasons with.
#
# A model's footprint is what admitting it costs the device: its weights (`W`, rounded the way the
# allocator rounds), its largest single weight allocation (`wmax`, the contiguity it needs), and the
# transient bytes one execution claims (`S`, measured by the startup scratch probe). Load time is
# tracked alongside, since it is what an eviction costs to undo. Reactant-free, like observe.jl.

# XLA's BFC allocator rounds every allocation up to a multiple of 256 bytes, so a model's device
# footprint is the sum of rounded tensor sizes, not the raw byte count.
const ALLOC_GRANULE = 256

_alloc_rounded(n::Integer) = cld(Int(n), ALLOC_GRANULE) * ALLOC_GRANULE

# Weight of the newest load-time sample in the moving average: recent enough to follow a change in
# host or link load, smooth enough that one slow transfer does not dominate.
const LOAD_SECONDS_ALPHA = 0.25

mutable struct ModelFootprint
    owner::LoadedModel          # the model this was measured for; a rebuilt model gets a fresh one
    weight_bytes::Int           # W: allocator-rounded total of the weight tensors
    max_weight_bytes::Int       # wmax: the largest single weight allocation, rounded
    scratch_bytes::Int          # S: transient bytes of the largest compiled batch (0 = unmeasured)
    load_seconds::Float64       # moving average of measured on-demand load time (0.0 = never loaded)
    loads::Int
    # Eviction policy state (GreedyDual-Size-Frequency, see admission.jl), for the current residency.
    hits::Int                   # dispatches since the weights were last loaded
    priority::Float64           # H: inflation at last use + hits * reload seconds / weight bytes
    loaded_at::Float64          # time() of the last load, for the minimum-residency guard
end

"""
    weight_footprint(sizes) -> (; weight_bytes, max_weight_bytes)

Allocator-rounded total and largest single allocation for a model whose weight tensors are `sizes`
bytes each.
"""
function weight_footprint(sizes)
    total = 0
    largest = 0
    for n in sizes
        r = _alloc_rounded(n)
        total += r
        largest = max(largest, r)
    end
    return (weight_bytes = total, max_weight_bytes = largest)
end

# Raw per-tensor byte sizes, from the host floor when there is one and otherwise from the
# safetensors header (no data read). `nothing` when neither can be read; the caller then falls back to
# `model.nbytes`.
function _weight_tensor_sizes(entry::ModelEntry, model::LoadedModel)
    model.host_weights === nothing || return (sizeof(h) for h in model.host_weights)
    st = entry.weights
    names = model.sig.weight_names
    try
        return [(s = weight_spec(st, n); prod(s[2]; init = 1) * sizeof(s[1])) for n in names]
    catch
        return nothing
    end
end

function _new_footprint(entry::ModelEntry, model::LoadedModel)
    sizes = _weight_tensor_sizes(entry, model)
    w = sizes === nothing ? (weight_bytes = model.nbytes, max_weight_bytes = model.nbytes) :
        weight_footprint(sizes)
    return ModelFootprint(model, w.weight_bytes, w.max_weight_bytes, 0, 0.0, 0, 0, 0.0, 0.0)
end

"""
    record_load!(fp, seconds) -> fp

Fold one measured on-demand load time into the footprint's moving average.
"""
function record_load!(fp::ModelFootprint, seconds::Real)
    fp.load_seconds = fp.loads == 0 ? Float64(seconds) :
        (1 - LOAD_SECONDS_ALPHA) * fp.load_seconds + LOAD_SECONDS_ALPHA * Float64(seconds)
    fp.loads += 1
    return fp
end

"""
    largest_free_bytes(stats) -> Union{Int, Nothing}

The largest contiguous allocation the device allocator can satisfy right now, from a
[`device_memory_stats`](@ref) snapshot, or `nothing` when the backend does not report it.
"""
largest_free_bytes(stats) = stats === nothing ? nothing : get(stats, :largest_free, nothing)

"""
    fragmentation(stats) -> Union{Float64, Nothing}

`1 - largest_free / free`: 0 when all free memory is one contiguous block, approaching 1 as free
memory splinters into pieces too small to use. `nothing` when the backend cannot report the
largest free block.
"""
function fragmentation(stats)
    lf = largest_free_bytes(stats)
    lf === nothing && return nothing
    stats.free <= 0 && return 0.0
    return clamp(1 - lf / stats.free, 0.0, 1.0)
end
