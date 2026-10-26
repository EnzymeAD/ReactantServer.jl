# On-demand residency for unpinned model weights.
#
# Pinned models load their weights at startup and keep them on the device for the server's
# lifetime; they are exempt from the budget here. Every other model starts evicted and has
# its weights transferred to the device on first dispatch. Weights are kept resident after a
# dispatch (so back-to-back requests to the same model still coalesce) and evicted LRU only
# when a different model needs room and `max_bytes` would be exceeded.
#
# `max_bytes` budgets only the on-demand pool, i.e. the GPU memory left after the pinned
# models. It is sized by the operator with that in mind.
#
# Only the single dispatch-loop thread mutates the cache (plus startup, before that loop is
# spawned), so the read-decide-write of victim selection is internally consistent without a
# lock. The lock guards the brief bookkeeping commits against the read-only metrics snapshot;
# the slow device transfers run outside it.

mutable struct WeightCache
    backend::AbstractBackend
    pool::MemoryPool
    registry::ModelRegistry
    mode::ResidencyMode            # who owns device residency on this worker
    store::WeightStore             # host-weight residency (private, or node-shared)
    max_bytes::Int                 # byte budget for non-device-pinned resident weights (> 0)
    resident_bytes::Int            # current non-device-pinned resident total
    lru::Vector{String}            # non-device-pinned resident model names, LRU at front, MRU at back
    loads::Int                     # observability: on-demand loads performed
    evicts::Int                    # observability: evictions performed
    load_seconds::Float64          # observability: cumulative time spent loading weights
    compactions::Int               # observability: compaction passes performed
    pinned_bytes::Int              # observability: device-pinned footprint reserved (set at startup)
    max_scratch::Int               # observability: measured worst-case execution scratch (set at startup)
    weight_pool::Int               # observability: arena bytes allotted to all weights (pinned + on-demand)
    load_bytes::Int                # observability: weight bytes transferred by on-demand loads
    footprints::Dict{String, ModelFootprint}   # per-model W / wmax / S / load time, by model name
    # Admission (admission.jl).
    inflation::Float64             # GreedyDual-Size-Frequency L: raised to each evicted model's priority
    min_residency_seconds::Float64 # a model loaded more recently than this is evicted only as a last resort
    search::VictimSearch           # reusable victim-search arrays (dispatch thread only)
    fallback_compactions::Int      # observability: compactions admission ran because eviction alone could not fit
    admission_failures::Int        # observability: models admission could not fit even after compacting
    # Host weight cache (host_cache below). With the shared store the budget and the cache live in
    # /dev/shm (`store.budget`); with the private store they are this worker's, below.
    host_budget::Union{Int, Nothing}       # this worker's share of the host budget (private store); nothing = unbudgeted
    host_cached::Dict{String, Vector{Any}} # private store: cached host copies of unpinned models
    host_lru::Vector{String}               # private store: cached names, least recently loaded at front
    host_cached_bytes::Int                 # private store: bytes in host_cached
    host_hits::Int                         # observability: loads served from a cached host copy
    host_misses::Int                       # observability: loads that read the weights file
    serving::Bool                          # startup is over: a reload from disk now warns (once per model)
    disk_warned::Set{String}               # models whose reload from disk has been warned about
    lock::ReentrantLock
end

WeightCache(
    backend::AbstractBackend, pool::MemoryPool, registry::ModelRegistry, max_bytes::Integer;
    mode::ResidencyMode = SELF_MANAGED, store::WeightStore = PrivateWeightStore(),
    min_residency_seconds::Real = 0.0, host_budget::Union{Integer, Nothing} = nothing
) =
    WeightCache(
    backend, pool, registry, mode, store, Int(max_bytes), 0, String[], 0, 0, 0.0, 0, 0, 0, 0, 0,
    Dict{String, ModelFootprint}(), 0.0, Float64(min_residency_seconds), VictimSearch(), 0, 0,
    host_budget === nothing ? nothing : Int(host_budget), Dict{String, Vector{Any}}(), String[], 0, 0, 0,
    false, Set{String}(), ReentrantLock()
)

"""
    model_footprint(cache, entry) -> ModelFootprint

The model's device footprint (`W`, `wmax`, `S`, load time), computed on first use and cached. A
model rebuilt under the same name (a reload) gets a fresh footprint, since its weights and
compiled programs may differ. Runs on the dispatch thread, the sole residency mutator; the lock
covers the dictionary against the metrics snapshot.
"""
function model_footprint(cache::WeightCache, entry::ModelEntry)
    model = entry.executable::LoadedModel
    return lock(cache.lock) do
        fp = get(cache.footprints, entry.name, nothing)
        if fp === nothing || fp.owner !== model
            fp = _new_footprint(entry, model)
            cache.footprints[entry.name] = fp
        end
        return fp
    end
end

# A numeric device-memory snapshot for the load / evict / compact logs, or `nothing` without
# device stats. One allocator query (well under a microsecond on CUDA).
function _memory_point(cache::WeightCache)
    dm = device_memory_stats(cache.backend, cache.pool)
    dm === nothing && return nothing
    return (in_use = dm.in_use, largest_free = largest_free_bytes(dm), fragmentation = fragmentation(dm))
end

"""
    weight_budget(; arena, fraction, wiggle, max_scratch, pinned_bytes) -> (; on_demand_budget, weight_pool, scratch_ceiling, pinned_over_commit)

Solve for the on-demand weight-cache byte budget from the unified memory model. The arena holds, at
the worst instant, pinned weights + on-demand weights + one model's execution `max_scratch` + a
`wiggle` fraction of free slack:

    scratch_ceiling  = (1 - wiggle) * arena
    weight_pool      = min(fraction * arena, scratch_ceiling - max_scratch)   # for pinned + on-demand
    on_demand_budget = clamp(weight_pool - pinned_bytes, 0, weight_pool)

`fraction` caps the share of the arena devoted to weights (1.0 = all of it, minus scratch + wiggle).
Pinned models reserve their footprint off the top, so the operator never subtracts it by hand.
`pinned_over_commit` (pinned exceed the pool) flags a genuine over-commit no sizing can fix. Pure
arithmetic, no device access, so it is unit testable.
"""
function weight_budget(;
        arena::Integer, fraction::Real, wiggle::Real,
        max_scratch::Integer, pinned_bytes::Integer
    )
    scratch_ceiling = floor(Int, (1 - Float64(wiggle)) * arena)
    weight_pool = max(0, min(floor(Int, Float64(fraction) * arena), scratch_ceiling - Int(max_scratch)))
    on_demand_budget = clamp(weight_pool - Int(pinned_bytes), 0, weight_pool)
    return (
        on_demand_budget = on_demand_budget, weight_pool = weight_pool,
        scratch_ceiling = scratch_ceiling, pinned_over_commit = Int(pinned_bytes) > weight_pool,
    )
end

# Free every resident non-pinned device buffer (no reload, no logging, no counters). Used by the
# startup scratch probe to isolate one model at a time; pinned weights and host floors are untouched.
function _free_nonpinned!(cache::WeightCache)
    to_free = Any[]
    lock(cache.lock) do
        for entry in values(cache.registry.by_name)
            model = entry.executable
            (model === nothing || model.weights === nothing || is_device_pinned(model)) && continue
            push!(to_free, model.weights)
            model.weights = nothing
        end
        empty!(cache.lru)
        cache.resident_bytes = 0
    end
    for fb in to_free
        free_weights!(cache.backend, fb)
    end
    return nothing
end

_touch_mru!(cache::WeightCache, name::AbstractString) = begin
    i = findfirst(==(name), cache.lru)
    i !== nothing && (deleteat!(cache.lru, i); push!(cache.lru, name))
    nothing
end

_remove_lru!(cache::WeightCache, name::AbstractString) = begin
    i = findfirst(==(name), cache.lru)
    i !== nothing && deleteat!(cache.lru, i)
    nothing
end

# Transfer a model's weights to GPU buffers. When the weights are pinned in host RAM
# (host_weights set, the normal on-demand case) this is a pure host->device transfer; under a host
# weight budget it goes through the host cache; otherwise it reads the weights file.
function _device_buffers(cache::WeightCache, entry::ModelEntry, model::LoadedModel)
    if model.host_weights !== nothing
        return transfer_to_device(cache.backend, cache.pool, model.host_weights)
    end
    host_budgeted(cache) && return _load_via_host_cache(cache, cache.store, entry, model)
    _note_disk_load!(cache, entry)
    return load_pinned_weights(cache.backend, cache.pool, entry.weights, model.sig.weight_names)
end

"""
    preload_pinned!(cache, registry) -> nothing

Ensure every pinned model's weights are resident. `build_loaded_model` already loads them
when on-demand mode is on, so this is normally a no-op; it loads defensively otherwise.
"""
function preload_pinned!(cache::WeightCache, registry::ModelRegistry)
    for entry in values(registry.by_name)
        model = entry.executable
        if is_device_pinned(model) && model.weights === nothing
            model.weights = _device_buffers(cache, entry, model)
        end
    end
    return nothing
end

"""
    NotResidentError

Raised by [`acquire!`](@ref) in externally-managed mode when a request targets a model whose
weights are not currently resident. A control plane is authoritative for residency in that mode,
so the worker does not autonomously load; the model must be pinned first.
"""
struct NotResidentError <: Exception
    model::String
end
Base.showerror(io::IO, e::NotResidentError) =
    print(io, "model '", e.model, "' is not resident on the device; the control plane must pin it before inference")

"""
    acquire!(cache, entry) -> nothing

Guarantee `entry.executable.weights` is resident before the model runs. Device-pinned and
already-resident models return immediately (the latter is bumped to most-recently-used and its
eviction priority raised). In self-managed mode an evicted model is admitted autonomously by
[`admit_weights!`](@ref): it evicts the cheapest set of resident models that makes the weights
and the execution scratch fit, compacting only as a fallback. Without device memory statistics
(CPU, mocks) admission falls back to the byte budget alone, and a model larger than the whole
budget is loaded anyway after evicting everything, with a warning, since it cannot run
otherwise. In externally-managed mode the worker does not autonomously load: an evicted model
raises [`NotResidentError`](@ref).
"""
function acquire!(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    is_device_pinned(model) && return nothing
    if model.weights !== nothing
        lock(cache.lock) do
            _touch_mru!(cache, entry.name)
        end
        record_hit!(cache, entry)
        return nothing
    end

    cache.mode == EXTERNALLY_MANAGED && throw(NotResidentError(entry.name))
    admit_weights!(cache, entry)
    return nothing
end

# Load a model only to measure it (the startup scratch probe, which frees it again). Counted as a
# load like any other, and also as a probe load, so it is not mistaken for the model's first load
# in service (see `_note_disk_load!`).
function probe_acquire!(cache::WeightCache, entry::ModelEntry)
    acquire!(cache, entry)
    fp = model_footprint(cache, entry)
    lock(() -> fp.probe_loads += 1, cache.lock)
    return nothing
end

# Upload a model's weights and commit residency. Used by admission for both a fresh admission
# (`fresh = true`: its eviction priority starts over) and a fallback-compaction reload (`fresh =
# false`: it keeps the priority it had earned). On a failed transfer the partial upload has
# already been freed (see `transfer_to_device`) and the error propagates.
function _upload_commit!(cache::WeightCache, entry::ModelEntry; fresh::Bool = true)
    model = entry.executable
    fp = model_footprint(cache, entry)
    before = _memory_point(cache)
    # With weights pinned in host RAM this is a pure host->device copy, not a re-materialization.
    t0 = time()
    bufs = _device_buffers(cache, entry, model)
    dt = time() - t0
    lock(cache.lock) do
        model.weights = bufs
        push!(cache.lru, entry.name)
        cache.resident_bytes += model.nbytes
        cache.loads += 1
        cache.load_seconds += dt
        cache.load_bytes += fp.weight_bytes
        record_load!(fp, dt)
        fp.loaded_at = time()
        if fresh
            fp.hits = 1
            fp.priority = cache.inflation + _priority_gain(cache, fp)
        end
    end
    log_memory_event(:load, entry.name, fp, before, _memory_point(cache))
    log_residency_change(
        entry.name, :system, :device, model.nbytes;
        memory = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
    )
    return nothing
end

# Evict resident non-pinned models by name: commit the bookkeeping under the lock, then release the
# device buffers outside it. Raises the policy's inflation to the highest evicted priority, which
# is what ages every model that was not touched since (GreedyDual). Returns the weight bytes freed.
function _evict!(cache::WeightCache, names)
    to_free = Any[]
    freed = 0
    lock(cache.lock) do
        for name in names
            entry = cache.registry.by_name[name]
            vm = entry.executable
            push!(to_free, vm.weights)
            vm.weights = nothing
            _remove_lru!(cache, name)
            cache.resident_bytes -= vm.nbytes
            cache.evicts += 1
            fp = get(cache.footprints, name, nothing)
            if fp !== nothing
                cache.inflation = max(cache.inflation, fp.priority)
                freed += fp.weight_bytes
            end
        end
    end
    for (name, fb) in zip(names, to_free)
        before = _memory_point(cache)
        free_weights!(cache.backend, fb)
        log_memory_event(:evict, name, model_footprint(cache, cache.registry.by_name[name]), before, _memory_point(cache))
    end
    return freed
end

"""
    set_residency_state!(cache, entry, target) -> ResidencyState

Move a model to the `target` residency floor (see `ResidencyState`). Like [`acquire!`](@ref)
this runs on the dispatch-loop thread, the sole mutator of residency. It materializes or drops
the host floor and, for `PINNED_DEVICE`, ensures the weights are resident on the device (exempt
from the budget). In externally-managed mode, leaving the device floor releases the device
buffers (there is no autonomous evictor to reclaim them); in self-managed mode they stay
resident but become evictable. The slow host/device work runs outside the lock; only the
bookkeeping commit is locked.
"""
function set_residency_state!(cache::WeightCache, entry::ModelEntry, target::ResidencyState)
    model = entry.executable
    cur = model.state
    cur == target && return target

    # Slow work outside the lock: materialize the host floor and/or transfer to the device.
    new_host = model.host_weights
    old_host = model.host_weights
    drop_host = false
    if target == PINNED_SYSTEM && model.host_weights === nothing
        # A cached host copy (private store) becomes the floor without rereading the file.
        new_host = _take_cached_host!(cache, entry.name)
        new_host === nothing && (new_host = host_materialize(cache.store, entry, model.sig.weight_names))
    elseif target == UNPINNED && model.host_weights !== nothing
        new_host = nothing        # drop the host floor
        drop_host = true
    end

    new_dev = model.weights
    free_dev = nothing
    if target == PINNED_DEVICE && model.weights === nothing
        new_dev = _device_buffers(cache, entry, model)
    elseif target != PINNED_DEVICE && cache.mode == EXTERNALLY_MANAGED && model.weights !== nothing
        # Externally-managed has no evictor, so unpinning from the device must free it here.
        free_dev = model.weights
        new_dev = nothing
    end

    lock(cache.lock) do
        # Uncount if presently counted (non-device-pinned and device-resident).
        if entry.name in cache.lru
            _remove_lru!(cache, entry.name)
            cache.resident_bytes -= model.nbytes
        end
        model.host_weights = new_host
        model.weights = new_dev
        model.state = target
        # Recount if it is now device-resident but not device-pinned (evictable in self-managed).
        if target != PINNED_DEVICE && model.weights !== nothing
            push!(cache.lru, entry.name)
            cache.resident_bytes += model.nbytes
        end
    end
    free_dev === nothing || free_weights!(cache.backend, free_dev)
    # The host arrays are no longer referenced (model.host_weights was set to nothing); release the
    # store entry (detach + last-one-out unlink for the shared store).
    drop_host && host_release!(cache.store, entry.name)
    # Under a host budget a dropped floor stays cached (the shared store keeps the region; the
    # private store keeps the arrays if they fit), so the next load does not read the file.
    if host_budgeted(cache) && !(cache.store isa SharedWeightStore)
        drop_host && _cache_host!(cache, entry.name, old_host)
        target == PINNED_SYSTEM && _fit_floors!(cache)
    end
    log_residency_change(
        entry.name, cur, target, model.nbytes;
        memory = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
    )
    return target
end

"""
    release_all!(cache, entry) -> nothing

Release every residency a model holds: free its device buffers, drop it from the LRU budget, and
release any shared host floor. Used by `evict!` when a model is removed from the worker. Runs on
the dispatch thread (sole residency mutator); the device free runs outside the lock.
"""
function release_all!(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    model === nothing && return nothing
    to_free = nothing
    lock(cache.lock) do
        if entry.name in cache.lru
            _remove_lru!(cache, entry.name)
            cache.resident_bytes -= model.nbytes
        end
        to_free = model.weights
        model.weights = nothing
        delete!(cache.footprints, entry.name)
    end
    to_free === nothing || free_weights!(cache.backend, to_free)
    if model.host_weights !== nothing
        model.host_weights = nothing
        host_release!(cache.store, entry.name)
    end
    _uncache_host!(cache, entry.name)
    return nothing
end

"""
    rename!(cache, old, new) -> nothing

Rekey a model's residency bookkeeping from `old` to `new` after a model rename: the LRU entry
(when the model is device-resident and non-pinned) and the host-weight store key. No device or
host memory moves; the weights themselves are untouched. Runs on the dispatch thread (sole
residency mutator), like every other cache mutation.
"""
function rename!(cache::WeightCache, old::AbstractString, new::AbstractString)
    lock(cache.lock) do
        i = findfirst(==(String(old)), cache.lru)
        i === nothing || (cache.lru[i] = String(new))
        fp = pop!(cache.footprints, String(old), nothing)
        fp === nothing || (cache.footprints[String(new)] = fp)
        arrays = pop!(cache.host_cached, String(old), nothing)
        arrays === nothing || (cache.host_cached[String(new)] = arrays)
        j = findfirst(==(String(old)), cache.host_lru)
        j === nothing || (cache.host_lru[j] = String(new))
    end
    host_rename!(cache.store, old, new)
    return nothing
end

"""
    compact!(cache, registry; reload) -> Int

Defragment the device arena. Frees every resident *non-pinned* device weight buffer at once so the
BFC allocator coalesces its free list, then reloads each non-pinned model named in `reload`.
Models not in `reload` are left freed; in self-managed mode they reload lazily through
[`acquire!`](@ref) on their next dispatch. Host floors (`host_weights`) are never touched, so the
reload is a pure host->device copy where a floor exists.

Device-pinned models are deliberately left in place: they are loaded once at startup, before any
on-demand traffic, so they sit at the base of the arena, and compaction neither frees them nor
pays to re-read them from disk. Only the on-demand churn region above them is freed and coalesced.

Runs on the dispatch-loop thread (the sole residency mutator), so no execution reads a buffer
while it is being freed. In externally-managed mode `acquire!` will not autonomously load, so the
`reload` list is ignored there (the control plane re-pins what it needs). Returns the number of
models reloaded.
"""
function compact!(cache::WeightCache, registry::ModelRegistry; reload::Vector{String} = String[])
    before = memory_report(cache.backend, cache.pool; registry = registry, weight_cache = cache)
    before_point = _memory_point(cache)

    # Phase A: free every resident non-pinned device buffer. Reset the on-demand budget bookkeeping
    # (which counts only non-pinned residency, so this is exact); keep the host floors. The slow
    # device frees run outside the lock.
    to_free = Any[]
    lock(cache.lock) do
        for entry in values(registry.by_name)
            model = entry.executable
            (model === nothing || model.weights === nothing || is_device_pinned(model)) && continue
            push!(to_free, model.weights)
            model.weights = nothing
        end
        empty!(cache.lru)
        cache.resident_bytes = 0
        cache.compactions += 1
    end
    for fb in to_free
        free_weights!(cache.backend, fb)
    end

    # Phase B: reload the requested non-pinned models into the coalesced region, through `acquire!`
    # so its budget/LRU bookkeeping (and `loads`/`load_seconds`) is restored. Largest first: on an
    # empty region a descending-size fill packs each model against the last and cannot fragment,
    # whereas small models placed first leave gaps a later large model may not fit. Skipped in
    # externally-managed mode, where `acquire!` does not autonomously load.
    reloaded = 0
    if cache.mode != EXTERNALLY_MANAGED
        for name in _largest_first(cache, registry, reload)
            entry = get(registry.by_name, name, nothing)
            entry === nothing && continue
            model = entry.executable
            (model === nothing || is_device_pinned(model) || model.weights !== nothing) && continue
            acquire!(cache, entry)
            reloaded += 1
        end
    end

    after = memory_report(cache.backend, cache.pool; registry = registry, weight_cache = cache)
    after_point = _memory_point(cache)
    @info "memory compacted" reloaded = reloaded requested = length(reload) before = before after = after largest_free_before = _point_field(before_point, :largest_free) largest_free_after = _point_field(after_point, :largest_free) fragmentation_before = _point_field(before_point, :fragmentation) fragmentation_after = _point_field(after_point, :fragmentation)
    return reloaded
end

_point_field(p, k::Symbol) = p === nothing ? nothing : getfield(p, k)

# `names` ordered by descending weight footprint; names that are not loaded models keep their
# relative order at the end (the reload loop skips them).
function _largest_first(cache::WeightCache, registry::ModelRegistry, names::Vector{String})
    function wbytes(name)
        entry = get(registry.by_name, name, nothing)
        (entry === nothing || entry.executable === nothing) && return -1
        return model_footprint(cache, entry).weight_bytes
    end
    return sort(names; by = wbytes, rev = true)
end

"""
    weight_cache_stats(cache) -> NamedTuple

Snapshot the cache counters under its lock for observability.
"""
function weight_cache_stats(cache::WeightCache)
    return lock(cache.lock) do
        return (
            resident_bytes = cache.resident_bytes, max_bytes = cache.max_bytes,
            resident_models = copy(cache.lru), loads = cache.loads, evicts = cache.evicts,
            load_seconds = cache.load_seconds, compactions = cache.compactions,
            pinned_bytes = cache.pinned_bytes, max_scratch = cache.max_scratch,
            weight_pool = cache.weight_pool, load_bytes = cache.load_bytes,
            fallback_compactions = cache.fallback_compactions,
            admission_failures = cache.admission_failures,
            host_budget = host_budget(cache), host_cached_bytes = host_cached_bytes(cache),
            host_hits = cache.host_hits, host_misses = cache.host_misses,
            footprints = Dict(
                name => (
                    weight_bytes = fp.weight_bytes, max_weight_bytes = fp.max_weight_bytes,
                    scratch_bytes = fp.scratch_bytes, load_seconds = fp.load_seconds, loads = fp.loads,
                ) for (name, fp) in cache.footprints
            ),
        )
    end
end
