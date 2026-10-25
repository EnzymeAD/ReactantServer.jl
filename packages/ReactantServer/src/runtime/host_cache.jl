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
# its own share, `budget ÷ host_weight_budget_workers`, here, with no coordination. Both run on the
# dispatch thread, the sole residency mutator; `cache.lock` covers the bookkeeping against the
# metrics snapshot.
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

function _load_via_host_cache(cache::WeightCache, store::SharedWeightStore, entry::ModelEntry, model::LoadedModel)
    st = entry.weights
    names = model.sig.weight_names
    specs = [weight_spec(st, n) for n in names]
    # The same key and digest as the model's pinned floor (`host_materialize`), so a model pinned by
    # one worker and unpinned on another shares one region.
    digest = weights_digest(entry.name, specs; content = weights_file_token(entry.weights_id))
    fill! = function (arrays)
        for (i, n) in enumerate(names)
            read_weight!(arrays[i], st, n)
        end
        _check_unmodified(entry)
        return
    end
    r = try
        with_cached_host_weights(store, entry.name, digest, specs, fill!) do arrays
            transfer_to_device(cache.backend, cache.pool, arrays)
        end
    catch err
        err isa SharedStoreFull || rethrow()
        @warn "host weight cache: the node cannot back this model's region; loading without it" model = entry.name exception = err
        nothing
    end
    if r === nothing                        # no room in the budget: load without caching
        lock(() -> cache.host_misses += 1, cache.lock)
        return load_pinned_weights(cache.backend, cache.pool, st, names)
    end
    hit, bufs = r
    lock(cache.lock) do
        hit ? (cache.host_hits += 1) : (cache.host_misses += 1)
    end
    return bufs
end

function _load_via_host_cache(cache::WeightCache, ::WeightStore, entry::ModelEntry, model::LoadedModel)
    arrays = lock(() -> get(cache.host_cached, entry.name, nothing), cache.lock)
    if arrays !== nothing
        bufs = transfer_to_device(cache.backend, cache.pool, arrays)
        lock(cache.lock) do
            cache.host_hits += 1
            j = findfirst(==(entry.name), cache.host_lru)
            j === nothing || (deleteat!(cache.host_lru, j); push!(cache.host_lru, entry.name))
        end
        return bufs
    end
    arrays = materialize_host_weights(entry.weights, model.sig.weight_names)
    _check_unmodified(entry)
    bufs = transfer_to_device(cache.backend, cache.pool, arrays)
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
