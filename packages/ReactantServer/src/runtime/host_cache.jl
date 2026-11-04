# The host weight cache: copies of unpinned models' weights kept in host RAM between GPU loads,
# bounded by `runtime.host_weight_budget`.
#
# Without a budget nothing here runs: an unpinned model reads its weights file on every load, as
# before. With one, a load first looks for a cached copy and, on a miss, keeps the copy it read when
# the budget has room, evicting the least recently LOADED copies first. Only a load from the cache
# (or the copy's creation) counts as use; requests served by a model already on the device do not,
# so a model that stays on a GPU ages out of host RAM and the most often loaded models stay cached.
# Pinned floors (`residency: system`) count against the budget but are never evicted; when they
# alone exceed it they load anyway, with a warning.
#
# With the shared store the budget is one node-wide pool in /dev/shm shared by every worker (see
# "Node budget" in ReactantServerCore's weight_store.jl). With the private store each worker manages
# its own share, `budget ÷ host_weight_budget_workers`, here, with no coordination. Loads run for
# the dispatch loop, the sole residency mutator, and the background reads that stage weights ahead
# of a load (below) only read the private cache; `cache.lock` covers the bookkeeping against both and
# against the metrics snapshot.
#
# A miss reads the whole model and queues every upload, exactly like an unbudgeted load; the arrays
# it read are left to the GC when they are not cached, and do not count against the budget.

"""
    host_budgeted(cache) -> Bool

Whether loads of unpinned models go through the host weight cache.
"""
host_budgeted(cache::WeightCache) =
    cache.store isa SharedWeightStore ? cache.store.budget !== nothing : cache.host_budget !== nothing

"""
    host_budget(cache) -> Union{Int, Nothing}

The host weight budget this worker enforces: the node-wide budget with the shared store, this
worker's share with the private store, `nothing` when unbudgeted.
"""
host_budget(cache::WeightCache) = cache.store isa SharedWeightStore ? cache.store.budget : cache.host_budget

"""
    host_cached_bytes(cache) -> Int

Host RAM the budget currently counts: every region on the node with the shared store; this
worker's pinned floors plus its cached copies with the private store.
"""
host_cached_bytes(cache::WeightCache) =
    cache.store isa SharedWeightStore ? host_cache_bytes(cache.store) :
    _host_floor_bytes(cache) + cache.host_cached_bytes

_host_bytes(arrays) = sum(sizeof, arrays; init = 0)

# Bytes held by this worker's pinned host floors (private store).
function _host_floor_bytes(cache::WeightCache)
    total = 0
    for entry in values(cache.registry.by_name)
        m = entry.executable
        (m === nothing || m.host_weights === nothing) && continue
        total += _host_bytes(m.host_weights)
    end
    return total
end

function _load_via_host_cache(
        cache::WeightCache, store::SharedWeightStore, entry::ModelEntry, model::LoadedModel; timing = nothing
    )
    st = entry.weights
    names = model.sig.weight_names
    specs = [weight_spec(st, n) for n in names]
    # The same key and digest as the model's pinned floor (`host_materialize`), so a model pinned by
    # one worker and unpinned on another shares one region.
    digest = weights_digest(entry.name, specs; content = weights_file_token(entry.weights_id))
    fill! = function (arrays)
        t = time()
        for (i, n) in enumerate(names)
            read_weight!(arrays[i], st, n)
        end
        _check_unmodified(entry)
        _timed_read!(timing, time() - t)
        return
    end
    r = try
        with_cached_host_weights(store, entry.name, digest, specs, fill!) do arrays
            _timed_transfer(cache, arrays, timing)
        end
    catch err
        err isa SharedStoreFull || rethrow()
        @warn "host weight cache: the node cannot back this model's region; loading without it" model = entry.name exception = err
        nothing
    end
    if r === nothing                        # no room in the budget: load without caching
        lock(() -> cache.host_misses += 1, cache.lock)
        _note_disk_load!(cache, entry)
        return load_pinned_weights(cache.backend, cache.pool, st, names; timing = timing)
    end
    hit, bufs = r
    lock(cache.lock) do
        hit ? (cache.host_hits += 1) : (cache.host_misses += 1)
    end
    hit || _note_disk_load!(cache, entry)
    return bufs
end

function _load_via_host_cache(
        cache::WeightCache, ::WeightStore, entry::ModelEntry, model::LoadedModel; timing = nothing
    )
    arrays = lock(() -> get(cache.host_cached, entry.name, nothing), cache.lock)
    if arrays !== nothing
        bufs = _timed_transfer(cache, arrays, timing)
        lock(cache.lock) do
            cache.host_hits += 1
            j = findfirst(==(entry.name), cache.host_lru)
            j === nothing || (deleteat!(cache.host_lru, j); push!(cache.host_lru, entry.name))
        end
        return bufs
    end
    _note_disk_load!(cache, entry)
    t = time()
    arrays = materialize_host_weights(entry.weights, model.sig.weight_names)
    _check_unmodified(entry)
    _timed_read!(timing, time() - t)
    bufs = _timed_transfer(cache, arrays, timing)
    lock(() -> cache.host_misses += 1, cache.lock)
    _cache_host!(cache, entry.name, arrays)
    return bufs
end

# Keep `arrays` as `name`'s cached host copy (private store) if the budget can make room for them,
# evicting the least recently loaded copies. Evicts nothing unless that makes them fit.
function _cache_host!(cache::WeightCache, name::String, arrays::Vector{Any})
    budget = cache.host_budget::Int
    nb = _host_bytes(arrays)
    floors = _host_floor_bytes(cache)
    return lock(cache.lock) do
        haskey(cache.host_cached, name) && return true
        floors + nb > budget && return false          # would not fit even with every copy evicted
        while floors + cache.host_cached_bytes + nb > budget
            victim = popfirst!(cache.host_lru)
            cache.host_cached_bytes -= _host_bytes(pop!(cache.host_cached, victim))
        end
        cache.host_cached[name] = arrays
        push!(cache.host_lru, name)
        cache.host_cached_bytes += nb
        return true
    end
end

# Drop `name`'s cached host copy (private store), if any: the model was removed or reloaded.
function _uncache_host!(cache::WeightCache, name::AbstractString)
    lock(cache.lock) do
        arrays = pop!(cache.host_cached, String(name), nothing)
        arrays === nothing && return
        cache.host_cached_bytes -= _host_bytes(arrays)
        j = findfirst(==(String(name)), cache.host_lru)
        j === nothing || deleteat!(cache.host_lru, j)
    end
    return nothing
end

# Take `name`'s cached host copy out of the cache (private store), or `nothing`.
function _take_cached_host!(cache::WeightCache, name::AbstractString)
    cache.store isa SharedWeightStore && return nothing
    return lock(cache.lock) do
        arrays = pop!(cache.host_cached, String(name), nothing)
        arrays === nothing && return nothing
        cache.host_cached_bytes -= _host_bytes(arrays)
        j = findfirst(==(String(name)), cache.host_lru)
        j === nothing || deleteat!(cache.host_lru, j)
        return arrays
    end
end

# After a new pinned floor (private store): evict cached copies, least recently loaded first, until
# floors and copies fit the budget again, and warn if the floors alone do not.
function _fit_floors!(cache::WeightCache)
    budget = cache.host_budget::Int
    floors = _host_floor_bytes(cache)
    lock(cache.lock) do
        while !isempty(cache.host_lru) && floors + cache.host_cached_bytes > budget
            victim = popfirst!(cache.host_lru)
            cache.host_cached_bytes -= _host_bytes(pop!(cache.host_cached, victim))
        end
    end
    check_host_budget(cache)
    return nothing
end

# Warn, once per model per process, when a model that has been on the device before is reloaded
# from its weights file after startup: with no host copy, every reload reads the disk, far slower
# than a copy from RAM, and an operator seeing slow hot swaps should learn why from the log. A
# model's first load (startup, or a model the watcher just added) is expected and not warned. The
# startup scratch probe's loads do not count: the probe frees each model again, so a model's first
# load in service is still its first, even when the probe pushed its host copy out of the budget.
function _note_disk_load!(cache::WeightCache, entry::ModelEntry)
    cache.serving || return nothing
    fp = model_footprint(cache, entry)
    fp.loads > fp.probe_loads || return nothing
    fresh = lock(cache.lock) do
        entry.name in cache.disk_warned && return false
        push!(cache.disk_warned, entry.name)
        return true
    end
    fresh || return nothing
    budget = host_budget(cache)
    @warn "model weights reloaded from disk: no host RAM copy was available, so this reload reads the weights file, which is much slower than a copy from RAM. If models are swapped on and off the GPU, give host copies room with runtime.host_weight_budget or pin this model with residency: system (see Use cases in the docs). Logged once per model." model = entry.name weights = Base.format_bytes(entry.executable.nbytes) host_weight_budget = budget === nothing ? "unlimited" : Base.format_bytes(budget)
    return nothing
end

"""
    check_host_budget(cache) -> nothing

Warn when pinned floors (`residency: system`) alone exceed the host weight budget, which leaves
nothing for cached copies. Called at startup; the shared store also warns as each floor is
admitted.
"""
function check_host_budget(cache::WeightCache)
    budget = host_budget(cache)
    budget === nothing && return nothing
    held = cache.store isa SharedWeightStore ? host_cache_bytes(cache.store) : _host_floor_bytes(cache)
    held > budget &&
        @warn "host weight budget overcommitted: pinned (residency: system) weights exceed it, so no other model's weights stay cached in host RAM" budget = Base.format_bytes(budget) pinned = Base.format_bytes(held)
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Staging: reading a model's weights into host memory ahead of its upload
# ---------------------------------------------------------------------------------------------
#
# A load has a host half (waiting on the shared store's locks, reserving the region, reading the
# weights file, the legacy-layout permute) and a device half (eviction and the upload). The host half
# touches no device memory, so the scheduler runs it on a background task while the dispatch loop
# keeps serving the models already on the device (see `_kick_staging!` in scheduler.jl), and the
# dispatch that finally runs the model only has to upload. Several workers stage at once: the shared
# store serializes only its bookkeeping, and two workers wanting the same model share one read.

"""
    StagedWeights

A model's weights read into host memory for one upload: `arrays` in executable order and whether
they came from a host copy (`hit`) or from the weights file. `cached` is the shared-store region,
held (so no worker evicts it) until `release_staged!`. `fresh` marks private-store arrays
read from the file, which are offered to the host cache once uploaded, as an unstaged load does.
"""
mutable struct StagedWeights
    arrays::Vector{Any}
    hit::Bool
    cached::Union{CachedHostWeights, Nothing}
    fresh::Bool
    host_seconds::Float64                 # time the host half took, lock waits included
    released::Bool
end
StagedWeights(arrays, hit; cached = nothing, fresh = false) =
    StagedWeights(arrays, hit, cached, fresh, 0.0, false)

"""
    needs_host_stage(cache, entry) -> Bool

Whether loading `entry` has host work to do ahead of time: a self-managed, unpinned model that is
off the device and has no host floor (`residency: system`) to upload from.
"""
function needs_host_stage(cache::WeightCache, entry::ModelEntry)
    cache.mode == SELF_MANAGED || return false
    model = entry.executable
    model === nothing && return false
    return !is_device_pinned(model) && model.weights === nothing && model.host_weights === nothing
end

"""
    stage_host_weights(cache, entry) -> StagedWeights

Do every host step of loading `entry` and no device step: take its region from the node cache
(shared store, reading the file into it on a miss), its cached copy (private store), or read the
file. Runs on a background task, concurrently with the dispatch loop: it reads the weights file
with `pread`, which keeps no file position, and takes `cache.lock` for the cache bookkeeping. Hit and
miss counters are left to the upload (`_device_buffers`), so they count loads, as before, and the
time taken is recorded on the result so the load's full cost still reaches the eviction policy.
"""
function stage_host_weights(cache::WeightCache, entry::ModelEntry)
    t0 = time()
    staged = _stage_host_weights(cache, entry)
    staged.host_seconds = time() - t0
    return staged
end

function _stage_host_weights(cache::WeightCache, entry::ModelEntry)
    model = entry.executable::LoadedModel
    names = model.sig.weight_names
    st = entry.weights
    if host_budgeted(cache) && cache.store isa SharedWeightStore
        specs = [weight_spec(st, n) for n in names]
        digest = weights_digest(entry.name, specs; content = weights_file_token(entry.weights_id))
        fill! = function (arrays)
            for (i, n) in enumerate(names)
                read_weight!(arrays[i], st, n)
            end
            _check_unmodified(entry)
            return
        end
        h = try
            acquire_cached_host_weights(cache.store, entry.name, digest, specs, fill!)
        catch err
            err isa SharedStoreFull || rethrow()
            @warn "host weight cache: the node cannot back this model's region; loading without it" model = entry.name exception = err
            nothing
        end
        h === nothing || return StagedWeights(h.arrays, h.hit; cached = h)
        # No room in the node budget: a private copy for this one upload, not cached.
        return StagedWeights(_read_checked(entry, names), false)
    end
    if host_budgeted(cache)
        arrays = lock(() -> get(cache.host_cached, entry.name, nothing), cache.lock)
        arrays === nothing || return StagedWeights(arrays, true)
        return StagedWeights(_read_checked(entry, names), false; fresh = true)
    end
    return StagedWeights(_read_checked(entry, names), false)
end

function _read_checked(entry::ModelEntry, names::Vector{String})
    arrays = materialize_host_weights(entry.weights, names)
    _check_unmodified(entry)
    return arrays
end

# The host-cache bookkeeping of one upload from staged weights, the same as an unstaged load records
# (see `_load_via_host_cache`): a hit or a miss under a budget, the most recent load for a private
# copy, and the once-per-model warning for a reload from disk.
function _note_staged_load!(cache::WeightCache, entry::ModelEntry, staged::StagedWeights)
    if host_budgeted(cache)
        lock(cache.lock) do
            staged.hit ? (cache.host_hits += 1) : (cache.host_misses += 1)
            if staged.hit && !(cache.store isa SharedWeightStore)
                j = findfirst(==(entry.name), cache.host_lru)
                j === nothing || (deleteat!(cache.host_lru, j); push!(cache.host_lru, entry.name))
            end
        end
    end
    staged.hit || _note_disk_load!(cache, entry)
    return nothing
end

"""
    release_staged!(cache, entry, staged; uploaded) -> nothing

Let go of staged weights once their upload is done (`uploaded`) or abandoned: drop the region's lock
(it stays cached in the node budget), and keep a private copy read from the file if the budget has
room, as an unstaged load does. Idempotent.
"""
function release_staged!(cache::WeightCache, entry::ModelEntry, staged::StagedWeights; uploaded::Bool)
    staged.released && return nothing
    staged.released = true
    arrays = staged.arrays
    staged.arrays = Any[]                 # a region's arrays alias its mapping, which goes away below
    if staged.cached !== nothing
        arrays = nothing
        release_cached_host_weights!(cache.store, staged.cached)
    elseif uploaded && staged.fresh
        _cache_host!(cache, entry.name, arrays)
    end
    return nothing
end

"""
    has_host_copy(cache, entry) -> Bool

Whether reloading `entry` would find its weights in host RAM rather than read its weights file: a
host floor (`residency: system`), a private cached copy, or its region in the shared store. For the
shared store this is a snapshot of `/dev/shm` (another worker can evict the region at any moment),
which is what an eviction estimate needs, not a guarantee. Admission asks it once per resident model
it considers, not on every dispatch.
"""
function has_host_copy(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    model === nothing && return false
    model.host_weights !== nothing && return true
    host_budgeted(cache) || return false
    store = cache.store
    if store isa SharedWeightStore
        names = model.sig.weight_names
        specs = [weight_spec(entry.weights, n) for n in names]
        digest = weights_digest(entry.name, specs; content = weights_file_token(entry.weights_id))
        return cached_region_exists(store, entry.name, digest)
    end
    return lock(() -> haskey(cache.host_cached, entry.name), cache.lock)
end
