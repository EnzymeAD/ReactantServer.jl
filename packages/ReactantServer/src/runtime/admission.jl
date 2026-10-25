# Fragmentation-aware weight admission: making room for a model's weights and for the scratch of
# the heaviest resident execution, by evicting the resident models whose loss costs least.
#
# Admission decides from allocator statistics before it uploads anything. A refused device
# allocation is expensive to discover (measured on an A6000: XLA's allocator waits about 10
# seconds for frees before it reports one), so trying and rolling back is never the normal path.
# The steps, for model m with weight bytes W, largest weight tensor wmax and scratch S:
#
#   0. Feasibility: m cannot fit even with every evictable model gone. Reject at once rather
#      than evicting everything first.
#   A. Make room for the weights: free bytes >= W + the scratch the resident set needs, the
#      largest free block >= W, and the operator's byte budget. Evict the cheapest covering set
#      for a byte deficit; evict the cheapest single model when only contiguity is short.
#      Requiring a block of W rather than wmax is deliberate: a free block holding the largest
#      tensor does not guarantee the other tensors find room, and the allocator reports only its
#      largest block, so wmax alone would let a multi-tensor upload be refused partway (10 s).
#      A block of exactly W is not enough either: BFC splits a chunk only when it is at least
#      twice the request or the remainder is at least 128 MiB, so one tensor can swallow the rest
#      of the block. The check replays that rule over the model's tensors (`bfc_block_fits`).
#   B. Upload. A failed transfer frees its partial upload and falls back to compaction.
#   C. Guarantee the scratch: the largest free block must hold the largest scratch of any resident
#      model, m included, so every resident model can still run. Evict until it does.
#
# When eviction runs out of candidates, compaction is the fallback (`_compact_and_retry!`):
# free every evictable model, reload the ones worth keeping largest first with m, retry once.
#
# The pinned set is m plus the device-pinned models. Executions are synchronous on the dispatch
# thread, which also runs admission, so no other model can have an execution in flight here.
#
# Victims are ranked by GreedyDual-Size-Frequency: priority H = L + hits * reload seconds / W,
# refreshed on every dispatch, with the inflation L raised to each evicted model's priority. The
# cost of evicting a model is (H - L) * W, its expected reload time discounted by how long ago it
# was last used, so a costly, frequently used model stays and a stale one goes, with recency
# breaking ties as LRU would. Models with queued requests or loaded within
# `min_residency_seconds` are evicted only when nothing else covers the deficit.
#
# Without device statistics (CPU, mocks) only the byte budget applies, as before admission existed.

# Host-to-device bandwidth assumed before any load has been timed. The first-transfer rate measured
# on an A6000 from pageable memory was 0.6 to 1.2 GiB/s; once loads are measured the worker's own
# rate replaces it.
const DEFAULT_LOAD_BANDWIDTH = 1.0 * 2^30

# How long to wait for an eviction's memory to show as freed. Frees of settled buffers are
# immediate in practice; the wait covers deferred frees of buffers with work still pending.
const FREE_WAIT_SECONDS = 0.05

"""
    AdmissionError

Raised when a model cannot be placed on the device: either it cannot fit even with every
evictable model gone, or eviction and compaction together could not make room. Carries the
reason and a memory report for the log.
"""
struct AdmissionError <: Exception
    model::String
    reason::String
end
Base.showerror(io::IO, e::AdmissionError) =
    print(io, "cannot admit model '", e.model, "' to the device: ", e.reason)

"""
    reload_seconds(cache, fp) -> Float64

What evicting this model costs to undo: its measured average load time, or, before its first
load, its weight bytes over the worker's measured load bandwidth (the default before any load).
"""
function reload_seconds(cache::WeightCache, fp::ModelFootprint)
    fp.loads > 0 && return fp.load_seconds
    bw = cache.load_seconds > 0 ? cache.load_bytes / cache.load_seconds : DEFAULT_LOAD_BANDWIDTH
    return fp.weight_bytes / bw
end

_priority_gain(cache::WeightCache, fp::ModelFootprint) =
    fp.hits * reload_seconds(cache, fp) / max(fp.weight_bytes, 1)

"""
    record_hit!(cache, entry) -> nothing

Count a dispatch to a resident model toward its eviction priority. The field writes are not
locked: only the dispatch thread writes them, and the metrics snapshot only reads them.
"""
function record_hit!(cache::WeightCache, entry::ModelEntry)
    fp = get(cache.footprints, entry.name, nothing)
    fp === nothing && (fp = model_footprint(cache, entry))
    fp.hits += 1
    fp.priority = cache.inflation + _priority_gain(cache, fp)
    return nothing
end

# A model's scratch for admission: measured, or the worker's worst case when it could not be.
_scratch_of(cache::WeightCache, fp::ModelFootprint) = fp.scratch_bytes > 0 ? fp.scratch_bytes : cache.max_scratch

# The contiguous block that must stay available for executions: the largest scratch among the
# device-pinned models, the resident on-demand models, and `fp` (the model being admitted).
function _scratch_need(cache::WeightCache, fp::ModelFootprint)
    need = _scratch_of(cache, fp)
    for entry in values(cache.registry.by_name)
        model = entry.executable
        (model === nothing || model.weights === nothing) && continue
        f = get(cache.footprints, entry.name, nothing)
        f === nothing && (f = model_footprint(cache, entry))
        need = max(need, _scratch_of(cache, f))
    end
    return need
end

# Device-pinned weight bytes and the largest pinned scratch: what no eviction can remove.
function _pinned_floor(cache::WeightCache)
    bytes = 0
    scratch = 0
    for entry in values(cache.registry.by_name)
        model = entry.executable
        (model === nothing || !is_device_pinned(model)) && continue
        f = model_footprint(cache, entry)
        bytes += f.weight_bytes
        scratch = max(scratch, _scratch_of(cache, f))
    end
    return bytes, scratch
end

# Fill the cache's victim search with every resident on-demand model except `exclude`, in LRU order
# (front first, so cost ties evict the least recently used).
function _fill_candidates!(cache::WeightCache, exclude::String, now::Float64)
    s = cache.search
    n = length(cache.lru) - (exclude in cache.lru ? 1 : 0)
    reset_candidates!(s, n)
    i = 0
    for name in cache.lru
        name == exclude && continue
        entry = cache.registry.by_name[name]
        fp = get(cache.footprints, name, nothing)
        fp === nothing && (fp = model_footprint(cache, entry))
        cost = max(fp.priority - cache.inflation, 0.0) * fp.weight_bytes
        queued = entry.sched !== nothing && !isempty(entry.sched.queue)
        young = now - fp.loaded_at < cache.min_residency_seconds
        i += 1
        set_candidate!(s, i, name, fp.weight_bytes, cost, queued || young)
    end
    return s
end

# What still stands between model `model` (footprint `fp`) and its weights fitting: a byte deficit
# (budget or device) and whether the largest free block is too small to hold all its weights.
function _weights_shortfall(cache::WeightCache, model::LoadedModel, fp::ModelFootprint, stats)
    deficit = cache.resident_bytes + model.nbytes - cache.max_bytes
    contiguous_short = false
    if stats !== nothing
        deficit = max(deficit, fp.weight_bytes + _scratch_need(cache, fp) - stats.free)
        lf = largest_free_bytes(stats)
        contiguous_short = lf !== nothing && !bfc_block_fits(lf, fp.tensor_bytes)
    end
    return deficit, contiguous_short
end

# After freeing `freed` weight bytes, wait (briefly) until the allocator shows them freed, so the
# next decision does not read stale statistics and over-evict.
function _await_frees!(cache::WeightCache, in_use_before::Int, freed::Int)
    target = in_use_before - freed
    deadline = time() + FREE_WAIT_SECONDS
    while true
        st = device_memory_stats(cache.backend, cache.pool)
        (st === nothing || st.in_use <= target || time() >= deadline) && return st
        yield()
    end
    return
end

# Evict the chosen candidates of the cache's search and wait for the frees; returns fresh stats.
function _evict_chosen!(cache::WeightCache, stats, chosen)
    in_use = stats === nothing ? 0 : stats.in_use
    freed = _evict!(cache, chosen)
    return stats === nothing ? nothing : _await_frees!(cache, in_use, freed)
end

_chosen_names(s::VictimSearch) = [s.names[i] for i in 1:s.n if s.best[i]]

"""
    admit_weights!(cache, entry) -> nothing

Make the evicted model `entry` resident, evicting what it must (see the file header for the
steps). Throws [`AdmissionError`](@ref) when the model cannot be placed. Runs on the dispatch
thread, the sole residency mutator.
"""
function admit_weights!(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    fp = model_footprint(cache, entry)
    stats = device_memory_stats(cache.backend, cache.pool)

    # 0. Feasibility: the device-pinned floor is all that can never be evicted.
    if stats !== nothing
        pinned_bytes, pinned_scratch = _pinned_floor(cache)
        need = fp.weight_bytes + pinned_bytes + max(_scratch_of(cache, fp), pinned_scratch)
        if need > stats.limit
            cache.admission_failures += 1
            throw(AdmissionError(entry.name, "needs $(Base.format_bytes(need)) with the device-pinned models, the device allows $(Base.format_bytes(stats.limit))"))
        end
    end

    # A. Make room for the weights.
    while true
        deficit, contiguous_short = _weights_shortfall(cache, model, fp, stats)
        (deficit <= 0 && !contiguous_short) && break
        s = _fill_candidates!(cache, entry.name, time())
        chosen = if deficit > 0
            (select_cover!(s, deficit) || select_cover!(s, deficit; allow_protected = true)) ?
                _chosen_names(s) : String[]
        else
            i = select_cheapest!(s)
            i == 0 && (i = select_cheapest!(s; allow_protected = true))
            i == 0 ? String[] : [s.names[i]]
        end
        if isempty(chosen)
            if stats === nothing
                # Budget only (no device statistics): the pre-admission behavior, load anyway.
                _evict!(cache, copy(cache.lru))
                @warn "weight cache: model exceeds budget even after evicting all unpinned models; loading anyway" model = entry.name nbytes = model.nbytes budget = cache.max_bytes
                break
            end
            return _compact_and_retry!(cache, entry)
        end
        stats = _evict_chosen!(cache, stats, chosen)
    end

    # B. Upload; a failed transfer has freed its partial upload already.
    try
        _upload_commit!(cache, entry)
    catch err
        stats === nothing && rethrow()
        @warn "weight upload failed despite admission; compacting and retrying" model = entry.name exception = err
        return _compact_and_retry!(cache, entry)
    end

    # C. Guarantee the scratch block for every resident model, this one included.
    stats === nothing && return nothing
    while true
        stats = device_memory_stats(cache.backend, cache.pool)
        lf = largest_free_bytes(stats)
        (lf === nothing || lf >= _scratch_need(cache, fp)) && return nothing
        s = _fill_candidates!(cache, entry.name, time())
        i = select_cheapest!(s)
        i == 0 && (i = select_cheapest!(s; allow_protected = true))
        i == 0 && return _compact_and_retry!(cache, entry)
        _evict_chosen!(cache, stats, [s.names[i]])
    end
    return
end

# The fallback: free every evictable model (and `entry` if it got uploaded), then reload the most
# valuable of the previously resident models that fit alongside `entry`, largest first so the
# refill packs tightly, and check that the weights and the scratch block both fit. Runs at most
# once per admission; a second failure is an AdmissionError.
function _compact_and_retry!(cache::WeightCache, entry::ModelEntry)
    fp = model_footprint(cache, entry)
    keep = [name for name in cache.lru if name != entry.name]
    sort!(keep; by = name -> cache.footprints[name].priority, rev = true)
    cache.fallback_compactions += 1
    before = device_memory_stats(cache.backend, cache.pool)
    freed = sum((cache.footprints[name].weight_bytes for name in cache.lru); init = 0)
    compact!(cache, cache.registry)
    # A model uploaded moments ago (step C's fallback) may free late; wait so the room is real.
    stats = before === nothing ? nothing : _await_frees!(cache, before.in_use, freed)
    if stats === nothing
        _upload_commit!(cache, entry)
        return nothing
    end
    # Room left once the pinned floor, `entry` and the scratch block are accounted for: fill it
    # with the most valuable former residents, within the operator's byte budget too.
    scratch = _scratch_need(cache, fp)
    room = stats.free - fp.weight_bytes - scratch
    budget = cache.max_bytes - entry.executable.nbytes
    reload = ModelEntry[entry]
    for name in keep
        e = cache.registry.by_name[name]
        f = cache.footprints[name]
        extra = max(0, _scratch_of(cache, f) - scratch)    # a larger scratch block it would need
        (f.weight_bytes + extra <= room && e.executable.nbytes <= budget) || continue
        push!(reload, e)
        room -= f.weight_bytes + extra
        budget -= e.executable.nbytes
        scratch += extra
    end
    sort!(reload; by = e -> cache.footprints[e.name].weight_bytes, rev = true)
    for e in reload
        try
            _upload_commit!(cache, e; fresh = e === entry)
        catch err
            cache.admission_failures += 1
            report = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
            @error "admission: reload after compaction failed" model = entry.name reloading = e.name memory = report exception = err
            throw(AdmissionError(entry.name, "upload failed after compaction ($(report))"))
        end
    end
    stats = device_memory_stats(cache.backend, cache.pool)
    lf = largest_free_bytes(stats)
    if lf !== nothing && lf < _scratch_need(cache, fp)
        cache.admission_failures += 1
        report = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
        @error "admission: scratch block does not fit after compaction" model = entry.name memory = report
        throw(AdmissionError(entry.name, "the scratch block does not fit after compaction ($(report))"))
    end
    return nothing
end
