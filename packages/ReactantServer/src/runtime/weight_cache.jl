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
    lock::ReentrantLock
end

WeightCache(backend::AbstractBackend, pool::MemoryPool, registry::ModelRegistry, max_bytes::Integer;
            mode::ResidencyMode=SELF_MANAGED, store::WeightStore=PrivateWeightStore()) =
    WeightCache(backend, pool, registry, mode, store, Int(max_bytes), 0, String[], 0, 0, 0.0, 0, 0, 0, 0, ReentrantLock())

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
function weight_budget(; arena::Integer, fraction::Real, wiggle::Real,
                       max_scratch::Integer, pinned_bytes::Integer)
    scratch_ceiling = floor(Int, (1 - Float64(wiggle)) * arena)
    weight_pool = max(0, min(floor(Int, Float64(fraction) * arena), scratch_ceiling - Int(max_scratch)))
    on_demand_budget = clamp(weight_pool - Int(pinned_bytes), 0, weight_pool)
    return (on_demand_budget = on_demand_budget, weight_pool = weight_pool,
            scratch_ceiling = scratch_ceiling, pinned_over_commit = Int(pinned_bytes) > weight_pool)
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
# (host_weights set, the normal on-demand case) this is a pure host->device transfer; otherwise
# it falls back to materializing from the mmap (used by tests that do not pin in RAM).
function _device_buffers(cache::WeightCache, entry::ModelEntry, model::LoadedModel)
    if model.host_weights !== nothing
        return transfer_to_device(cache.backend, cache.pool, model.host_weights)
    end
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
already-resident models return immediately (the latter is bumped to most-recently-used). In
self-managed mode an evicted model is loaded autonomously, evicting LRU victims until it fits
the budget (a model larger than the whole budget is loaded anyway after evicting everything,
with a warning, since it cannot run otherwise). In externally-managed mode the worker does not
autonomously load: an evicted model raises [`NotResidentError`](@ref).
"""
function acquire!(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    is_device_pinned(model) && return nothing
    if model.weights !== nothing
        lock(cache.lock) do
            _touch_mru!(cache, entry.name)
        end
        return nothing
    end

    cache.mode == EXTERNALLY_MANAGED && throw(NotResidentError(entry.name))

    need = model.nbytes
    # Select LRU victims (front first) until the target would fit.
    victims = String[]
    projected = cache.resident_bytes
    for name in cache.lru
        projected + need <= cache.max_bytes && break
        push!(victims, name)
        projected -= cache.registry.by_name[name].executable.nbytes
    end
    if projected + need > cache.max_bytes
        @warn "weight cache: model exceeds budget even after evicting all unpinned models; loading anyway" model = entry.name nbytes = need budget = cache.max_bytes
    end

    # Evict victims first so loading the target does not transiently exceed the budget. Commit
    # the residency bookkeeping under the lock, then release the device buffers outside it.
    if !isempty(victims)
        to_free = Any[]
        lock(cache.lock) do
            for name in victims
                vm = cache.registry.by_name[name].executable
                push!(to_free, vm.weights)
                vm.weights = nothing
                _remove_lru!(cache, name)
                cache.resident_bytes -= vm.nbytes
                cache.evicts += 1
            end
        end
        for fb in to_free
            free_weights!(cache.backend, fb)
        end
        @debug "residency: evicted for headroom" loading = entry.name evicted = victims
    end

    # Transfer the target's weights to the GPU (outside the lock), then commit residency. With
    # weights pinned in host RAM this is a pure host->device copy, not a re-materialization.
    t0 = time()
    bufs = _device_buffers(cache, entry, model)
    dt = time() - t0
    lock(cache.lock) do
        model.weights = bufs
        push!(cache.lru, entry.name)
        cache.resident_bytes += model.nbytes
        cache.loads += 1
        cache.load_seconds += dt
    end
    log_residency_change(entry.name, :system, :device, model.nbytes;
        memory=memory_report(cache.backend, cache.pool; registry=cache.registry, weight_cache=cache))
    return nothing
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
    drop_host = false
    if target == PINNED_SYSTEM && model.host_weights === nothing
        new_host = host_materialize(cache.store, entry.name, entry.weights, model.sig.weight_names)
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
    log_residency_change(entry.name, cur, target, model.nbytes;
        memory=memory_report(cache.backend, cache.pool; registry=cache.registry, weight_cache=cache))
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
    end
    to_free === nothing || free_weights!(cache.backend, to_free)
    if model.host_weights !== nothing
        model.host_weights = nothing
        host_release!(cache.store, entry.name)
    end
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
function compact!(cache::WeightCache, registry::ModelRegistry; reload::Vector{String}=String[])
    before = memory_report(cache.backend, cache.pool; registry=registry, weight_cache=cache)

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
    # so its budget/LRU bookkeeping (and `loads`/`load_seconds`) is restored. Skipped in
    # externally-managed mode, where `acquire!` does not autonomously load.
    reloaded = 0
    if cache.mode != EXTERNALLY_MANAGED
        for name in reload
            entry = get(registry.by_name, name, nothing)
            entry === nothing && continue
            model = entry.executable
            (model === nothing || is_device_pinned(model) || model.weights !== nothing) && continue
            acquire!(cache, entry)
            reloaded += 1
        end
    end

    after = memory_report(cache.backend, cache.pool; registry=registry, weight_cache=cache)
    @info "memory compacted" reloaded = reloaded requested = length(reload) before = before after = after
    return reloaded
end

"""
    weight_cache_stats(cache) -> NamedTuple

Snapshot the cache counters under its lock for observability.
"""
function weight_cache_stats(cache::WeightCache)
    lock(cache.lock) do
        return (resident_bytes = cache.resident_bytes, max_bytes = cache.max_bytes,
                resident_models = copy(cache.lru), loads = cache.loads, evicts = cache.evicts,
                load_seconds = cache.load_seconds, compactions = cache.compactions,
                pinned_bytes = cache.pinned_bytes, max_scratch = cache.max_scratch,
                weight_pool = cache.weight_pool)
    end
end
