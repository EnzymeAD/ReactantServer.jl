# Dynamic model-directory watching (Triton-style POLL mode).
#
# When `model_poll_seconds > 0` a background task polls the worker's `model_dirs` on that interval
# and hot-swaps bundles whose files changed on disk: a new bundle directory is loaded, a removed
# one is unloaded, and a changed one is reloaded. This avoids a full server restart (which would
# recompile every model at every batch size) when a single model is added or updated.
#
# All filesystem work (scanning, signatures, parsing) happens on this task; the actual swap goes
# through the scheduler's control queue (`load_model!`/`evict!`), which runs the compile and the
# device frees on the dispatch thread, stopping the world only for the duration of that one swap.
#
# A change is acted on only once its signature is observed identical across two consecutive polls
# (debounce), so a bundle still being copied in is not loaded half-written; `load_bundle_entry`'s
# validation is the backstop. A failed load/reload is logged and retried on the next poll.

# A bundle's on-disk fingerprint: (filename, mtime, size) for each file that defines it. Two equal
# signatures mean the bundle is unchanged.
const BundleSig = Vector{Tuple{String,Float64,Int}}

# The files that actually define a bundle (and thus affect compilation/serving). Other files in the
# directory are ignored so unrelated touches do not trigger a reload.
_is_bundle_file(f::AbstractString) =
    f == "manifest.yaml" || f == "weights.safetensors" || f == "model.jl" ||
    occursin(r"^model(\.b\d+)?\.mlir$", f)

function bundle_signature(dir::AbstractString)::BundleSig
    sig = Tuple{String,Float64,Int}[]
    for f in readdir(dir; sort=true)
        _is_bundle_file(f) || continue
        p = joinpath(dir, f)
        isfile(p) || continue
        push!(sig, (f, mtime(p), filesize(p)))
    end
    return sig
end

"""
    scan_repository(model_dirs, include) -> Dict{String,Tuple{String,BundleSig}}

Discover the current bundle directories under `model_dirs`, mapping model name to its directory and
[`BundleSig`](@ref). Uses the same discovery rule as `load_bundles` (a subdirectory containing
`manifest.yaml`; the model name is the directory's basename) and honors the `include` allowlist
(`nothing` = load all). Unlike `load_bundles`, a missing root is skipped rather than raising, since
the watcher must keep running across a transiently absent mount.
"""
function scan_repository(model_dirs, include::Union{Set{String},Nothing})
    out = Dict{String,Tuple{String,BundleSig}}()
    for root in model_dirs
        isdir(root) || continue
        for child in readdir(root; join=true)
            isdir(child) || continue
            isfile(joinpath(child, "manifest.yaml")) || continue
            name = basename(normpath(child))
            include === nothing || name in include || continue
            out[name] = (child, bundle_signature(child))
        end
    end
    return out
end

"""
    BundleWatcher

Background poller for one worker's model repository. Construct it with the running [`Scheduler`](@ref),
the runtime `backend`/`pool`, the resolved [`ServerConfig`](@ref), and the residency knobs
(`on_demand`, `store`) the worker resolved at startup, then [`start_watching!`](@ref) it. Stop it
with [`stop_watching!`](@ref).
"""
mutable struct BundleWatcher
    scheduler::Scheduler
    backend::AbstractBackend
    pool::MemoryPool
    cfg::ServerConfig
    model_dirs::Vector{String}
    include::Union{Set{String},Nothing}
    on_demand::Bool
    store::WeightStore
    interval::Float64
    seen::Dict{String,BundleSig}                       # last-applied signatures (loaded models)
    pending::Dict{String,Union{BundleSig,Nothing}}     # candidate change awaiting a stable second poll
    running::Bool
    task::Union{Task,Nothing}
end

function BundleWatcher(sched::Scheduler, backend::AbstractBackend, pool::MemoryPool,
                       cfg::ServerConfig; interval::Real, on_demand::Bool,
                       store::WeightStore=PrivateWeightStore(),
                       include::Union{Nothing,AbstractVector}=nothing)
    inc = include === nothing ? nothing : Set{String}(String(x) for x in include)
    # Seed `seen` from what is on disk now: it matches what `_bring_up` just loaded, so the first
    # poll is a no-op for the startup models and only later changes are acted on.
    initial = scan_repository(cfg.model_dirs, inc)
    seen = Dict{String,BundleSig}(name => cs[2] for (name, cs) in initial)
    return BundleWatcher(sched, backend, pool, cfg, copy(cfg.model_dirs), inc, on_demand, store,
                         Float64(interval), seen, Dict{String,Union{BundleSig,Nothing}}(), false, nothing)
end

# Apply a single confirmed change. `desired === nothing` means the bundle is gone (unload); a
# BundleSig means load or reload from `dir`. Failures are logged and not propagated: `seen` is left
# untouched on a load failure so the next poll retries.
function _apply_change!(w::BundleWatcher, name::AbstractString,
                        desired::Union{BundleSig,Nothing}, dir::Union{String,Nothing})
    # The detected change and the plan of action; the resulting "model loaded"/"model unloaded"
    # summaries carry the per-model detail.
    action = desired === nothing ? :unload : (haskey(w.seen, name) ? :reload : :load)
    @info "watcher: change detected" name = name action = action dir = something(dir, "")
    try
        if desired === nothing
            # A meta model has no executable/device state, so unload it from the registry directly;
            # otherwise go through the scheduler's evict path (frees device memory).
            is_meta_name(w.scheduler.registry, name) ? remove_meta!(w.scheduler, name) :
                unload_model!(w.scheduler, name)
            delete!(w.seen, name)
        else
            entry = load_bundle_entry(dir)
            # `load_bundles` enforces directory name == manifest name; keep that invariant here so a
            # mismatched bundle is rejected rather than served under the wrong name.
            entry.name == name ||
                throw(BundleError("bundle directory '$name' has manifest name '$(entry.name)'"))
            if entry isa MetaEntry
                # Meta bundles need no compilation; register them straight into the meta map.
                put_meta!(w.scheduler, entry)
            else
                state = _resolve_residency(w.cfg, name, w.on_demand)
                load_model!(w.scheduler, w.backend, w.pool, entry;
                            state=state, on_demand=w.on_demand, store=w.store)
            end
            w.seen[name] = desired
        end
    catch err
        @warn "watcher: failed to apply model change; will retry next poll" name exception = (err, catch_backtrace())
    end
    return nothing
end

# One poll: diff the current repository against the last-applied state and act on each change that
# has been stable for two consecutive polls.
function _watch_once!(w::BundleWatcher)
    current = scan_repository(w.model_dirs, w.include)
    desired = Dict{String,BundleSig}(name => cs[2] for (name, cs) in current)
    next_pending = Dict{String,Union{BundleSig,Nothing}}()
    for name in union(keys(desired), keys(w.seen))
        d = get(desired, name, nothing)   # nothing => should be unloaded
        s = get(w.seen, name, nothing)    # nothing => not currently loaded
        d == s && continue                # in sync; nothing to debounce
        # A change is pending. Act only when the same desired state was already seen last poll, so a
        # bundle mid-write (signature still changing) waits until it settles.
        if haskey(w.pending, name) && w.pending[name] == d
            _apply_change!(w, name, d, d === nothing ? nothing : current[name][1])
        else
            next_pending[name] = d
        end
    end
    w.pending = next_pending
    return nothing
end

"""
    start_watching!(w::BundleWatcher) -> BundleWatcher

Spawn the background poll loop. Polls immediately (a no-op against the seeded startup state), then
every `interval` seconds until [`stop_watching!`](@ref). Each poll round is wrapped so a transient
filesystem error does not kill the watcher.
"""
function start_watching!(w::BundleWatcher)
    w.running = true
    w.task = Threads.@spawn begin
        @info "Model directory watcher started" interval = w.interval model_dirs = w.model_dirs
        while w.running
            try
                _watch_once!(w)
            catch err
                @warn "watcher: poll round failed" exception = (err, catch_backtrace())
            end
            # Sleep in small steps so stop_watching! is responsive even with a long interval.
            slept = 0.0
            while w.running && slept < w.interval
                dt = min(0.25, w.interval - slept)
                sleep(dt)
                slept += dt
            end
        end
        @info "Model directory watcher stopped"
    end
    return w
end

"""
    stop_watching!(w::BundleWatcher)

Signal the poll loop to stop. Returns once the flag is set; the background task exits after its
current poll/sleep step.
"""
stop_watching!(w::BundleWatcher) = (w.running = false; nothing)
