# Fragmentation-aware weight admission: making room for a model's weights and for the scratch of
# the heaviest resident execution, by evicting the resident models whose loss costs least.
#
# Admission decides from allocator statistics before it uploads anything. A refused device
# allocation is expensive to discover (measured on an A6000: XLA's allocator waits about 10
# seconds for frees before it reports one), so trying and rolling back is never the normal path.
# The steps, for model m with weight bytes W, largest weight tensor wmax and scratch S:
#
#   0. Feasibility: m cannot fit even with every evictable model gone. Reject at once rather
#      than evicting everything first. What stays is measured, not assumed: the allocator's bytes
#      in use less what the evictable models hold (each one's in-use growth at its upload), so the
#      device-pinned weights at their allocated size count, and so does any other buffer no
#      eviction frees.
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
# free every evictable model, check from the statistics that m fits alone, reload the ones worth
# keeping largest first with m (as many as the split rule says fit), retry once.
#
# A failed admission leaves nothing behind: m is not resident and holds no device memory (an
# upload that step C then finds without room for the scratch is undone), and what compaction
# freed is reloaded as far as it fits. It is also remembered (`AdmissionRefusal`): until something
# that could make m fit changes (device memory freed that eviction cannot free, a device-pinned
# model unloaded, a residency changed, m reloaded), m's admissions fail at once, so its requests
# neither flush the working set nor wait on the allocator, and none of its weights are read into
# host memory for them (`needs_host_stage`). The transport answers them UNAVAILABLE.
#
# The pinned set is m plus the device-pinned models. Executions are synchronous on the dispatch
# loop, which also runs admission (on a default-pool task it waits for, see `off_interactive`), so
# no other model can have an execution in flight here.
#
# Victims are ranked by GreedyDual-Size-Frequency: priority H = L + hits * reload seconds / W,
# refreshed on every dispatch, with the inflation L raised to each evicted model's priority. The
# cost of evicting a model is (H - L) * W, its expected reload time discounted by how long ago it
# was last used, so a costly, frequently used model stays and a stale one goes, with recency
# breaking ties as LRU would. The reload seconds are what a reload would cost now: the upload, plus
# the weights-file read only when no host copy (a RAM floor, a cached copy, or a shared-memory
# region) is there to upload from, so of two equally used models the one still in host RAM goes
# first. Models with queued requests or loaded within
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
evictable model gone, or eviction and compaction together could not make room, or one of those
happened before and nothing that could make it fit has changed since. Carries the reason and a
memory report for the log. The transport answers a request that meets it with `UNAVAILABLE`: the
model cannot be placed on this worker now, and another replica may have room.
"""
struct AdmissionError <: Exception
    model::String
    reason::String
end
Base.showerror(io::IO, e::AdmissionError) =
    print(io, "cannot admit model '", e.model, "' to the device: ", e.reason)

# Reading a weights file before any read has been timed. Host page cache to RAM, at the order of
# the host-to-device default; once reads are measured the worker's own rate replaces it.
const DEFAULT_READ_BANDWIDTH = 1.0 * 2^30

"""
    reload_seconds(cache, fp) -> Float64

The device time that evicting this model costs to undo: uploading its weights again. The model's
measured average, or before it was measured, its bytes over the worker's measured upload bandwidth
(the default before any).

Reading the weights file is not counted. That read runs on a background task before the dispatch
(`stage_host_weights`), never on the dispatch loop, so it holds no device time; what it costs is
latency for the requests that wait for it, and that cost is incurred when the model's host copy is
evicted, not its device copy. See [`read_seconds`](@ref).
"""
function reload_seconds(cache::WeightCache, fp::ModelFootprint)
    return fp.upload_seconds > 0 ? fp.upload_seconds :
        fp.weight_bytes / _bandwidth(cache.upload_bytes, cache.upload_seconds, DEFAULT_LOAD_BANDWIDTH)
end

"""
    read_seconds(cache, fp) -> Float64

How long reading this model's weights file takes: the latency a request for it waits when no host
copy is left to upload from. The model's measured average, or its bytes over the worker's measured
read bandwidth (the default before any). Off the device's time; reported to the gateway, which
weighs it when planning which models to move.
"""
function read_seconds(cache::WeightCache, fp::ModelFootprint)
    return fp.read_seconds > 0 ? fp.read_seconds :
        fp.weight_bytes / _bandwidth(cache.read_bytes, cache.read_seconds, DEFAULT_READ_BANDWIDTH)
end

_bandwidth(bytes, seconds, default) = (seconds > 0 && bytes > 0) ? bytes / seconds : default

# Device-seconds of upload saved per weight byte held, per use: the GreedyDual-Size-Frequency gain.
# With uploads at a fixed bandwidth the size cancels, so the most often used models stay resident
# whatever their size; a fixed cost per load tilts it slightly toward small models.
_priority_gain(cache::WeightCache, fp::ModelFootprint) =
    fp.hits * reload_seconds(cache, fp) / max(fp.weight_bytes, 1)

# Recompute a resident model's priority from its current reload cost, keeping the inflation it was
# last used at. Called where an eviction decision reads priorities, since measured upload times
# change between a model's uses.
function _refresh_priority!(cache::WeightCache, entry::ModelEntry, fp::ModelFootprint)
    fp.priority = fp.prio_base + _priority_gain(cache, fp)
    return fp.priority
end

"""
    record_hit!(cache, entry) -> nothing

Count a dispatch to a resident model toward its eviction priority. The field writes are not
locked: only the dispatch thread writes them, and the metrics snapshot only reads them.
"""
function record_hit!(cache::WeightCache, entry::ModelEntry)
    fp = get(cache.footprints, entry.name, nothing)
    fp === nothing && (fp = model_footprint(cache, entry))
    fp.hits += 1
    fp.prio_base = cache.inflation
    fp.priority = fp.prio_base + _priority_gain(cache, fp)
    return nothing
end

# A model's scratch for admission: measured, or the worker's worst case when it could not be.
_scratch_of(cache::WeightCache, fp::ModelFootprint) = fp.scratch_bytes > 0 ? fp.scratch_bytes : cache.max_scratch

# The contiguous block that must stay available for executions: the largest scratch among the
# device-pinned models, the resident on-demand models, and `fp` (the model being admitted), if any.
function _scratch_need(cache::WeightCache, fp::Union{ModelFootprint, Nothing} = nothing)
    need = fp === nothing ? 0 : _scratch_of(cache, fp)
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

# Device bytes no eviction can free, measured: what the allocator holds less what the evictable
# (resident on-demand) models hold. The device-pinned weights at their allocated size count, and so
# does any buffer the weight cache does not own, such as one awaiting its finalizer. Read on the
# dispatch loop, where no program runs meanwhile, so no execution's scratch is counted.
function _unevictable_bytes(cache::WeightCache, stats)
    held = 0
    for name in cache.lru
        fp = get(cache.footprints, name, nothing)
        fp === nothing && (fp = model_footprint(cache, cache.registry.by_name[name]))
        held += evictable_bytes(fp)
    end
    return max(stats.in_use - held, 0)
end

_fmt_bytes(n) = n === nothing ? "unknown" : Base.format_bytes(n)

"""
    admission_refused(cache, entry) -> Bool

Whether admission refused `entry` (see `AdmissionRefusal`) and, as far as can be told without
the device statistics, nothing that would let it in has changed since: the model is the same
version, no device-pinned model was unloaded and no residency changed. Its requests are then
refused without reading its weights. Other memory freed since is noticed at its next admission
(`admit_weights!`). Takes the cache lock, so any task can ask.
"""
admission_refused(cache::WeightCache, entry::ModelEntry) = _standing_refusal(cache, entry) !== nothing

function _standing_refusal(cache::WeightCache, entry::ModelEntry)
    return lock(cache.lock) do
        r = get(cache.refusals, entry.name, nothing)
        (r === nothing || r.owner !== entry.executable || r.generation != cache.generation) && return nothing
        return r
    end
end

# Fail the admission of `entry` for `reason` and remember it, with the device memory that eviction
# cannot free right now, so its next admissions fail at once until something changes.
function _refuse!(cache::WeightCache, entry::ModelEntry, reason::String)
    stats = device_memory_stats(cache.backend, cache.pool)
    unevictable = stats === nothing ? 0 : _unevictable_bytes(cache, stats)
    lock(cache.lock) do
        cache.admission_failures += 1
        cache.refusals[entry.name] = AdmissionRefusal(entry.executable, cache.generation, unevictable, reason)
    end
    throw(AdmissionError(entry.name, reason))
end

# Before admitting `entry`: a refusal that stands fails the admission at once, with nothing evicted
# or uploaded. One that less unevictable memory has ended is dropped, and the admission goes ahead,
# unless it has no weights in host memory to upload: the dispatch loop never reads a weights file,
# so this request fails (UNAVAILABLE, retryable) and the model's next request has them read first.
function _check_refusal!(cache::WeightCache, entry::ModelEntry, stats, staged)
    r = _standing_refusal(cache, entry)
    if r !== nothing && _unevictable_bytes(cache, stats) >= r.unevictable
        lock(() -> cache.admission_refusals += 1, cache.lock)
        throw(
            AdmissionError(
                entry.name, "$(r.reason) (an earlier attempt; not retried until device memory is " *
                    "freed, a device-pinned model is unloaded, a residency changes, or the model is reloaded)"
            )
        )
    end
    lock(() -> delete!(cache.refusals, entry.name), cache.lock)
    if r !== nothing && staged === nothing && needs_host_stage(cache, entry)
        throw(
            AdmissionError(
                entry.name, "device memory was freed since it last failed admission; its weights " *
                    "are read into host memory for its next request, which tries again"
            )
        )
    end
    return nothing
end

# Undo the upload of `entry` an admission committed before failing: free its device buffers and
# drop it from the LRU and the budget (not counted as an eviction), then wait for the free to show.
function _unadmit!(cache::WeightCache, entry::ModelEntry)
    model = entry.executable
    fp = model_footprint(cache, entry)
    before = _memory_point(cache)
    bufs = lock(cache.lock) do
        b = model.weights
        model.weights = nothing
        if entry.name in cache.lru
            _remove_lru!(cache, entry.name)
            cache.resident_bytes -= model.nbytes
        end
        return b
    end
    bufs === nothing && return nothing
    free_weights!(cache.backend, bufs)
    before === nothing || _await_frees!(cache, before.in_use, evictable_bytes(fp))
    log_memory_event(:evict, entry.name, fp, before, _memory_point(cache))
    log_residency_change(
        entry.name, :device, :system, model.nbytes;
        memory = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
    )
    return nothing
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
        cost = max(_refresh_priority!(cache, entry, fp) - cache.inflation, 0.0) * fp.weight_bytes
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
function _evict_chosen!(cache::WeightCache, stats, chosen; reason::Symbol)
    in_use = stats === nothing ? 0 : stats.in_use
    freed = _evict!(cache, chosen; reason = reason)
    return stats === nothing ? nothing : _await_frees!(cache, in_use, freed)
end

_chosen_names(s::VictimSearch) = [s.names[i] for i in 1:s.n if s.best[i]]

"""
    admit_weights!(cache, entry; staged=nothing) -> nothing

Make the evicted model `entry` resident, evicting what it must (see the file header for the
steps). Throws [`AdmissionError`](@ref) when the model cannot be placed, leaving it off the device,
and at once, without evicting anything, while an earlier failure stands. Runs for the dispatch
loop, the sole residency mutator. `staged` is the model's weights already read into host memory
(see `stage_host_weights`), which every upload of `entry` here uses; the caller releases it.
"""
function admit_weights!(cache::WeightCache, entry::ModelEntry; staged = nothing)
    model = entry.executable
    fp = model_footprint(cache, entry)
    stats = device_memory_stats(cache.backend, cache.pool)
    stats === nothing || _check_refusal!(cache, entry, stats, staged)

    # 0. Feasibility: only the memory no eviction frees stays. A serving worker measures it. At
    # startup, where a failed admission stops the worker (the probe loads each model in turn), it
    # stays judged by the device-pinned models alone, as before measuring.
    if stats !== nothing
        pinned_bytes, pinned_scratch = _pinned_floor(cache)
        held = cache.serving ? max(_unevictable_bytes(cache, stats), pinned_bytes) : pinned_bytes
        need = fp.weight_bytes + held + max(_scratch_of(cache, fp), pinned_scratch)
        need > stats.limit && _refuse!(
            cache, entry, "needs $(_fmt_bytes(need)) with the device-pinned models and other memory no " *
                "eviction frees ($(_fmt_bytes(held))), the device allows $(_fmt_bytes(stats.limit))"
        )
    end

    # A. Make room for the weights. `victims` collects what this admission evicts, which a failure
    # puts back as far as it fits (`_compact_and_retry!`).
    victims = String[]
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
                _evict!(cache, copy(cache.lru); reason = :bytes)
                @warn "weight cache: model exceeds budget even after evicting all unpinned models; loading anyway" model = entry.name nbytes = model.nbytes budget = cache.max_bytes
                break
            end
            return _compact_and_retry!(cache, entry; staged = staged, victims = victims)
        end
        append!(victims, chosen)
        stats = _evict_chosen!(cache, stats, chosen; reason = deficit > 0 ? :bytes : :contiguity)
    end

    # B. Upload; a failed transfer has freed its partial upload already.
    try
        _upload_commit!(cache, entry; staged = staged)
    catch err
        stats === nothing && rethrow()
        @warn "weight upload failed despite admission; compacting and retrying" model = entry.name exception = err
        return _compact_and_retry!(cache, entry; staged = staged, victims = victims)
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
        i == 0 && return _compact_and_retry!(cache, entry; staged = staged, victims = victims)
        push!(victims, s.names[i])
        _evict_chosen!(cache, stats, [s.names[i]]; reason = :scratch)
    end
    return
end

# The bytes left in a free chunk of `block` bytes after uploading `entries` into it in order, under
# BFC's split rule, or `nothing` if they do not all fit.
function _reload_left(cache::WeightCache, block::Integer, entries)
    left = Int(block)
    for e in entries
        left = bfc_place(left, cache.footprints[e.name].tensor_bytes)
        left === nothing && return nothing
    end
    return left
end

# The fallback: free every evictable model (and `entry` if it got uploaded), then decide from the
# statistics whether `entry` fits on its own: all its weights in the largest free block under the
# split rule, which guarantees the upload (`bfc_block_fits`), and the scratch block in the bytes
# left. If it does, reload the most valuable former residents that fit alongside it, upload them
# and it largest first (`_refill!`), and check that the scratch block fits; if it does not after
# all, the upload is undone. If `entry` does not fit, nothing is uploaded for it: the former
# residents, and the models this admission evicted (`victims`), are reloaded as far as they fit.
# Either failure is refused (`_refuse!`). Runs at most once per admission.
function _compact_and_retry!(cache::WeightCache, entry::ModelEntry; staged = nothing, victims = String[])
    fp = model_footprint(cache, entry)
    former = String[name for name in cache.lru if name != entry.name]
    cache.fallback_compactions += 1
    before = device_memory_stats(cache.backend, cache.pool)
    freed = sum((evictable_bytes(model_footprint(cache, cache.registry.by_name[name])) for name in cache.lru); init = 0)
    compact!(cache, cache.registry)
    # A model uploaded moments ago (step C's fallback) may free late; wait so the room is real.
    stats = before === nothing ? nothing : _await_frees!(cache, before.in_use, freed)
    if stats === nothing
        _upload_commit!(cache, entry; staged = staged)
        return nothing
    end
    scratch = _scratch_need(cache, fp)
    lf = largest_free_bytes(stats)
    if fp.weight_bytes + scratch > stats.free || (lf !== nothing && !bfc_block_fits(lf, fp.tensor_bytes))
        _refill!(cache, stats, unique!(vcat(former, victims)), entry, _scratch_need(cache); with_entry = false)
        _refuse!(
            cache, entry, "its weights ($(_fmt_bytes(fp.weight_bytes))) and scratch " *
                "($(_fmt_bytes(scratch))) do not fit even with every evictable model gone: " *
                "$(_fmt_bytes(stats.free)) free, the largest free block $(_fmt_bytes(lf))"
        )
    end
    _refill!(cache, stats, former, entry, scratch; staged = staged)
    stats = device_memory_stats(cache.backend, cache.pool)
    lf = largest_free_bytes(stats)
    if lf !== nothing && lf < _scratch_need(cache, fp)
        report = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
        @error "admission: scratch block does not fit after compaction; the upload is undone" model = entry.name memory = report
        _unadmit!(cache, entry)
        _refuse!(cache, entry, "the scratch block does not fit after compaction ($(report))")
    end
    return nothing
end

# Reload, largest first so the refill packs tightly, the most valuable of the evicted models `names`
# that fit in the free memory `stats` describes, together with the admitted model `entry` (uploaded
# with them, unless `with_entry` is false), the scratch block `scratch`, and the operator's byte
# budget. Only a model with a host copy is reloaded: this runs on the dispatch loop, which never
# reads a weights file. One without stays evicted and is read in the background before its next
# dispatch. A failed reload of one of them leaves it evicted; a failed upload of `entry` is refused.
function _refill!(
        cache::WeightCache, stats, names::Vector{String}, entry::ModelEntry, scratch::Int;
        staged = nothing, with_entry::Bool = true
    )
    fps = Dict(name => model_footprint(cache, cache.registry.by_name[name]) for name in names)
    for name in names
        _refresh_priority!(cache, cache.registry.by_name[name], fps[name])
    end
    sort!(names; by = name -> fps[name].priority, rev = true)
    room = stats.free - scratch - (with_entry ? model_footprint(cache, entry).weight_bytes : 0)
    budget = cache.max_bytes - (with_entry ? entry.executable.nbytes : 0)
    # Bytes alone over-commit: BFC can hand a tensor a whole chunk (`bfc_block_fits`), and an upload
    # it refuses costs a 10 s allocator retry. So a model is reloaded only if the whole reload,
    # simulated into the largest free block in upload order (largest first), still leaves the scratch
    # block. Other free chunks can only add room, so the upload succeeds whenever the simulation does.
    lf = largest_free_bytes(stats)
    by_size = e -> cache.footprints[e.name].weight_bytes
    reload = with_entry ? ModelEntry[entry] : ModelEntry[]
    for name in names
        e = cache.registry.by_name[name]
        f = fps[name]
        has_host_copy(cache, e) || continue
        extra = max(0, _scratch_of(cache, f) - scratch)    # a larger scratch block it would need
        (f.weight_bytes + extra <= room && e.executable.nbytes <= budget) || continue
        if lf !== nothing
            left = _reload_left(cache, lf, sort!(push!(copy(reload), e); by = by_size, rev = true))
            (left !== nothing && left >= scratch + extra) || continue
        end
        push!(reload, e)
        room -= f.weight_bytes + extra
        budget -= e.executable.nbytes
        scratch += extra
    end
    sort!(reload; by = by_size, rev = true)
    for e in reload
        try
            _upload_commit!(cache, e; fresh = e === entry, staged = e === entry ? staged : nothing)
        catch err
            report = memory_report(cache.backend, cache.pool; registry = cache.registry, weight_cache = cache)
            if e !== entry
                # A former resident that does not fit just stays evicted.
                @warn "admission: reload of a former resident after compaction failed" model = entry.name reloading = e.name memory = report exception = err
                continue
            end
            # Its partial upload is freed already (`transfer_to_device`).
            @error "admission: reload after compaction failed" model = entry.name memory = report exception = err
            _refuse!(cache, entry, "upload failed after compaction ($(report))")
        end
    end
    return nothing
end
