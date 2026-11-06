# Dynamic model-directory watching (Triton-style POLL mode).
#
# When `model_poll_seconds > 0` a background task polls the worker's `model_dirs` on that interval
# and hot-swaps bundles whose files changed on disk: a new bundle directory is loaded, a removed
# one is unloaded, and a changed one is reloaded. A renamed directory is an unload of the old name
# and a load of the new one. This avoids a full server restart (which would recompile every model
# at every batch size) when a single model is added or updated.
#
# All filesystem work (scanning, signatures, parsing) happens on this task; the actual swap goes
# through the scheduler's control queue (`load_model!`/`evict!`), which runs the compile and the
# device frees on the dispatch loop. That stops the whole worker for the swap, not just the model:
# no model dispatches, and the compile holds the interactive thread, so request intake and health
# probes wait too. An executable-cache hit shortens it to a load; an uncached compile takes seconds
# per compiled program.
#
# A change is acted on only once its signature is observed identical across two consecutive polls
# (debounce), so a bundle still being copied in is not loaded half-written; `load_bundle_entry`'s
# validation is the backstop. A failed load/reload is logged and retried every second poll: the
# change is not recorded as applied, so the next poll sees it again as a new change and the one
# after confirms and retries it.
#
# A load records the signature the files had when they were read, not when the load ended, and a
# new version is admitted only if the files still have that signature once it is built. A bundle
# that changed while it loaded (a deploy landing mid-compile) is not admitted; the poll after loads
# it again from the files as they are then.

# A bundle's on-disk fingerprint: (filename, mtime, size) for each file that defines it. Two equal
# signatures mean the bundle is unchanged.
const BundleSig = Vector{Tuple{String, Float64, Int}}

# The files that actually define a bundle (and thus affect compilation/serving). Other files in the
# directory are ignored so unrelated touches do not trigger a reload; in particular everything the
# server itself writes under `.cache/` (the serialized-executable cache) is invisible here, so a
# cache write can never cause the model that produced it to be reloaded. Multi-shape bundles carry
# per-variant modules (`model.v{i}.b{N}.mlir`), which count as bundle files too.
_is_bundle_file(f::AbstractString) =
    f == "manifest.yaml" || occursin(r"^weights[A-Za-z0-9_.-]*\.safetensors$", f) || f == "model.jl" ||
    occursin(r"^model(\.v\d+)?(\.b\d+)?\.mlir$", f)

function bundle_signature(dir::AbstractString)::BundleSig
    sig = Tuple{String, Float64, Int}[]
    for f in readdir(dir; sort = true)
        _is_bundle_file(f) || continue
        p = joinpath(dir, f)
        isfile(p) || continue
        push!(sig, (f, mtime(p), filesize(p)))
    end
    return sig
end

"""
    scan_repository(model_dirs, include) -> Dict{String,Tuple{String,BundleSig}}

Discover the current bundle directories under `model_dirs`, mapping model name to its directory
and [`BundleSig`](@ref). Uses the same discovery rule as `load_bundles` (a subdirectory containing `manifest.yaml`; the model name is the
directory's basename) and honors the `include` allowlist (`nothing` = load all). Unlike
`load_bundles`, a missing root is skipped rather than raising, since the watcher must keep running
across a transiently absent mount.
"""
function scan_repository(model_dirs, include::Union{Set{String}, Nothing})
    out = Dict{String, Tuple{String, BundleSig}}()
    for root in model_dirs
        isdir(root) || continue
        for child in readdir(root; join = true)
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
with [`stop_watching!`](@ref). `seen` is what the scheduler has loaded: each model's bundle
signature as it was read (`load_bundles`' `signatures`), so that a bundle changed, added or removed
since is a change. Without it, the repository as it is now is taken to be what is loaded.
"""
mutable struct BundleWatcher
    scheduler::Scheduler
    backend::AbstractBackend
    pool::MemoryPool
    cfg::ServerConfig
    model_dirs::Vector{String}
    include::Union{Set{String}, Nothing}
    on_demand::Bool
    store::WeightStore
    interval::Float64
    seen::Dict{String, BundleSig}                       # last-applied signatures (loaded models)
    pending::Dict{String, Union{BundleSig, Nothing}}     # candidate change awaiting a stable second poll
    running::Bool
    task::Union{Task, Nothing}
end

function BundleWatcher(
        sched::Scheduler, backend::AbstractBackend, pool::MemoryPool,
        cfg::ServerConfig; interval::Real, on_demand::Bool,
        store::WeightStore = PrivateWeightStore(),
        include::Union{Nothing, AbstractVector} = nothing,
        seen::Union{Nothing, AbstractDict} = nothing
    )
    inc = include === nothing ? nothing : Set{String}(String(x) for x in include)
    # `_bring_up` passes the signatures its bundles had when they were read, before compiling them:
    # what is on disk by now can be newer, and seeding from it would take a bundle that changed
    # during the startup compile as loaded.
    loaded = if seen === nothing
        Dict{String, BundleSig}(name => cs[2] for (name, cs) in scan_repository(cfg.model_dirs, inc))
    else
        Dict{String, BundleSig}(seen)
    end
    return BundleWatcher(
        sched, backend, pool, cfg, copy(cfg.model_dirs), inc, on_demand, store,
        Float64(interval), loaded,
        Dict{String, Union{BundleSig, Nothing}}(), false, nothing
    )
end

# The signature of the bundle in `dir` now, or `nothing` when it is gone (no manifest), as
# `scan_repository` would see it. Never raises: a directory that disappears midway reads as gone.
function _signature_now(dir::AbstractString)
    try
        isfile(joinpath(dir, "manifest.yaml")) || return nothing
        return bundle_signature(dir)
    catch
        return nothing
    end
end

# Apply a single confirmed change. `desired === nothing` means the bundle is gone (unload); a
# BundleSig means load or reload from `dir`. Failures are logged and not propagated: `seen` is left
# untouched on a load failure, so the change is detected again on the next poll and retried on the
# one after (the two-poll debounce). A load whose files changed before the new version was admitted
# is not admitted; it is left pending at what is on disk then, so the next poll loads it again once
# that has held still.
function _apply_change!(
        w::BundleWatcher, name::AbstractString,
        desired::Union{BundleSig, Nothing}, dir::Union{String, Nothing}
    )
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
            # named by its directory basename, i.e. `name`. `read_sig` is the signature of the files
            # the entry is built from: taken after a weights migration rewrites them (so the
            # migration is not a change of its own) and before they are read.
            read_sig = Ref{BundleSig}()
            entry = load_bundle_entry(
                dir; batch_sizes = w.cfg.runtime.batch_sizes,
                weights_migration = w.cfg.runtime.weights_migration, signature = read_sig
            )
            unchanged = () -> _signature_now(dir) == read_sig[]
            admitted = if entry isa MetaEntry
                # Meta bundles need no compilation; register them straight into the meta map.
                unchanged() && (put_meta!(w.scheduler, entry); true)
            else
                state = _resolve_residency(w.cfg, name, w.on_demand)
                load_model!(
                    w.scheduler, w.backend, w.pool, entry;
                    state = state, on_demand = w.on_demand, store = w.store,
                    executable_cache = w.cfg.runtime.executable_cache, unchanged = unchanged
                ) !== nothing
            end
            if !admitted
                @warn "watcher: bundle files changed while the model loaded; not admitted, loading it again on the next poll" name
                w.pending[name] = _signature_now(dir)
                return nothing
            end
            w.seen[name] = read_sig[]
        end
        # On the heartbeat as well as in the log: a hot-load is an event a supervisor wants to see
        # (it is the server doing work), and a log line is not something another process can
        # subscribe to. One squad run proved a hot-load only by reading /proc/<pid>/maps.
        _beat(action === :unload ? :unload : :load, String(name))
    catch err
        @warn "watcher: failed to apply model change; will retry next poll" name exception = (err, catch_backtrace())
    end
    return nothing
end

# One poll: diff the current repository against the last-applied state and act on each change that
# has been stable for two consecutive polls.
function _watch_once!(w::BundleWatcher)
    current = scan_repository(w.model_dirs, w.include)
    desired = Dict{String, BundleSig}(name => cs[2] for (name, cs) in current)
    next_pending = Dict{String, Union{BundleSig, Nothing}}()
    confirmed = Tuple{String, Union{BundleSig, Nothing}}[]
    for name in union(keys(desired), keys(w.seen))
        d = get(desired, name, nothing)   # nothing => should be unloaded
        s = get(w.seen, name, nothing)    # nothing => not currently loaded
        d == s && continue                # in sync; nothing to debounce
        # A change is pending. Act only when the same desired state was already seen last poll, so a
        # bundle mid-write (signature still changing) waits until it settles.
        if haskey(w.pending, name) && w.pending[name] == d
            push!(confirmed, (name, d))
        else
            next_pending[name] = d
        end
    end
    w.pending = next_pending           # before applying: a load that saw its files change adds to it
    for (name, d) in confirmed
        _apply_change!(w, name, d, d === nothing ? nothing : current[name][1])
    end
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
