# Per-bundle serialized-executable cache: pay XLA compilation once, then load the compiled program
# in tens of milliseconds on every later start.
#
# Layout, inside the bundle directory the model was loaded from:
#
#   <bundle>/.cache/mlir_hashes.json                      sha256 of every model*.mlir the bundle had
#   <bundle>/.cache/exec/<target>/<source>.<src16>.<key16>.pjrtexe
#
# `<target>` partitions by everything the serialized program is only valid for and that XLA does not
# check on load (the Reactant_jll build, the CUDA runtime, the platform, the device kind and compute
# capability). `<source>` is the MLIR file the program came from, `<src16>` the prefix of that file's
# sha256, and `<key16>` the prefix of the backend's full key (post-numerics artifact, normalized
# compile options, numerics policy). Two workers on one bundle directory write the same bytes to the
# same names, and every write is temp-plus-rename, so concurrent workers are safe.
#
# Invalidation is by MLIR content: `sync_mlir_hashes!` compares the bundle's current source hashes
# with the recorded ones and deletes every entry of a source that changed or disappeared. Weights
# are deliberately not part of the record, so a weights-only update keeps the compiled programs.
# Entries are also content-addressed, so a changed source would simply miss even before the sweep;
# the sweep is what keeps the directory from accumulating dead programs.
#
# The watcher ignores `.cache/` by construction (it fingerprints only the files that define a
# bundle), so writing here never triggers a reload in dynamic mode.
#
# Everything fails open: any I/O or format error means "compile as if there were no cache".

using SHA: SHA

const EXEC_CACHE_DIRNAME = ".cache"
const EXEC_CACHE_HASHES_FILE = "mlir_hashes.json"
const EXEC_CACHE_EXT = ".pjrtexe"
const EXEC_CACHE_FORMAT = 1

"""
    ExecutableCacheSlot

Where one compiled module's cache entries live: the bundle's cache directory, the MLIR source file
the module came from (`model.b1.mlir`), and that file's sha256. A backend that supports the cache
receives one per `compile_artifact` call.
"""
struct ExecutableCacheSlot
    dir::String
    source::String
    source_sha::String
end

exec_cache_dir(bundle_dir::AbstractString) = joinpath(bundle_dir, EXEC_CACHE_DIRNAME)
sha256hex(bytes) = bytes2hex(SHA.sha256(bytes))

# The MLIR file a (variant, batch size) module was read from; mirrors the discovery rule in
# bundle.jl (`model` or `model.v{i}` prefix, `.b{N}` suffix for a per-batch-size module).
function module_filename(m::Manifest, vkey::VariantKey, sz::Int)
    prefix = if isempty(m.input_shapes)
        "model"
    else
        i = findfirst(==(vkey), m.input_shapes)
        i === nothing && error("variant $vkey is not one of the manifest's input_shapes")
        "model.v$(i - 1)"
    end
    return sz == 0 ? prefix * ".mlir" : prefix * ".b$(sz).mlir"
end

# Filesystem-safe slug for a target component ("NVIDIA RTX A6000" -> "nvidia-rtx-a6000").
target_slug(s::AbstractString) = lowercase(replace(strip(String(s)), r"[^A-Za-z0-9.+]+" => "-"))

function entry_path(slot::ExecutableCacheSlot, target::AbstractString, key_sha::AbstractString)
    name = string(slot.source, ".", first(slot.source_sha, 16), ".", first(key_sha, 16), EXEC_CACHE_EXT)
    return joinpath(slot.dir, "exec", String(target), name)
end

_hashes_path(cache_dir) = joinpath(cache_dir, EXEC_CACHE_HASHES_FILE)

function _read_hashes(cache_dir::AbstractString)
    p = _hashes_path(cache_dir)
    isfile(p) || return Dict{String, String}()
    try
        raw = JSON3.read(read(p, String), Dict{String, Any})
        get(raw, "format", nothing) == EXEC_CACHE_FORMAT || return Dict{String, String}()
        srcs = get(raw, "sources", nothing)
        srcs isa AbstractDict || return Dict{String, String}()
        return Dict{String, String}(String(k) => String(v) for (k, v) in srcs if v isa AbstractString)
    catch err
        @warn "executable cache: unreadable hash record; treating every entry as stale" path = p exception = err
        return Dict{String, String}()
    end
end

function _write_hashes(cache_dir::AbstractString, sources::Dict{String, String})
    mkpath(cache_dir)
    p = _hashes_path(cache_dir)
    tmp = p * ".tmp." * string(getpid()) * "." * string(rand(UInt32); base = 16)
    open(tmp, "w") do io
        JSON3.write(io, Dict("format" => EXEC_CACHE_FORMAT, "sources" => sources, "updated_unix" => floor(Int, time())))
    end
    mv(tmp, p; force = true)
    return nothing
end

# Delete every cached program derived from `source` under every target.
function _sweep_entries!(cache_dir::AbstractString, source::AbstractString)
    root = joinpath(cache_dir, "exec")
    isdir(root) || return 0
    n = 0
    prefix = source * "."
    for (dir, _, files) in walkdir(root)
        for f in files
            (startswith(f, prefix) && endswith(f, EXEC_CACHE_EXT)) || continue
            rm(joinpath(dir, f); force = true)
            n += 1
        end
    end
    return n
end

"""
    sync_mlir_hashes!(cache_dir, sources::Dict{String,String}) -> Vector{String}

Record the bundle's current MLIR source hashes (file name to sha256) in the cache directory and
delete every cached executable whose source changed hash or is no longer part of the bundle.
Returns the names of the invalidated sources. Weights are not part of the record. Fails open:
an I/O error logs a warning and returns an empty list.
"""
function sync_mlir_hashes!(cache_dir::AbstractString, sources::Dict{String, String})
    try
        prev = _read_hashes(cache_dir)
        stale = String[get(sources, src, nothing) == sha ? "" : src for (src, sha) in prev]
        filter!(!isempty, stale)
        for src in stale
            n = _sweep_entries!(cache_dir, src)
            @info "executable cache: MLIR source changed; dropped cached programs" cache_dir source = src entries = n
        end
        prev == sources || _write_hashes(cache_dir, sources)
        return stale
    catch err
        @warn "executable cache: could not sync MLIR hashes (cache disabled for this bundle)" cache_dir exception = err
        return String[]
    end
end

"Bytes of a cached program, or `nothing` when absent or unreadable."
function lookup_entry(path::AbstractString)
    isfile(path) || return nothing
    try
        return read(path)
    catch err
        @warn "executable cache: unreadable entry; recompiling" path exception = err
        return nothing
    end
end

"Atomically write a cached program (temp file plus rename). Returns true on success; failures warn."
function store_entry(path::AbstractString, bytes::Vector{UInt8})
    tmp = path * ".tmp." * string(getpid()) * "." * string(rand(UInt32); base = 16)
    try
        mkpath(dirname(path))
        write(tmp, bytes)
        mv(tmp, path; force = true)
        return true
    catch err
        rm(tmp; force = true)
        @warn "executable cache: could not write entry (is the bundle directory writable?)" path exception = err
        return false
    end
end

"Remove an entry that failed to load, so the next start recompiles instead of retrying it."
drop_entry(path::AbstractString) = (rm(path; force = true); nothing)

# Build the per-module cache slots for a bundle entry: sync the hash record, then one slot per
# (variant, batch size). Returns `nothing` when the bundle cannot host a cache.
function executable_cache_slots(entry::ModelEntry)
    bundle_dir = dirname(entry.weights_path)
    isdir(bundle_dir) || return nothing
    cache_dir = exec_cache_dir(bundle_dir)
    sources = Dict{String, String}()
    shas = Dict{Tuple{VariantKey, Int}, Tuple{String, String}}()
    for (vkey, inner) in entry.mlir_bytes, (sz, bytes) in inner
        f = module_filename(entry.manifest, vkey, sz)
        sha = sha256hex(bytes)
        sources[f] = sha
        shas[(vkey, sz)] = (f, sha)
    end
    sync_mlir_hashes!(cache_dir, sources)
    return Dict{Tuple{VariantKey, Int}, ExecutableCacheSlot}(
        k => ExecutableCacheSlot(cache_dir, f, sha) for (k, (f, sha)) in shas
    )
end

# ── Counters (worker Prometheus gauges; see transport/metrics.jl) ─────────────────────────────────

mutable struct ExecCacheStats
    hits::Int
    misses::Int
    stores::Int
    failures::Int          # cached programs that failed to load and were dropped
    load_seconds::Float64  # cumulative deserialize-and-load time on hits
    compile_seconds::Float64  # cumulative compile time on misses
end
const EXEC_CACHE_STATS = ExecCacheStats(0, 0, 0, 0, 0.0, 0.0)
const _EXEC_CACHE_STATS_LOCK = ReentrantLock()

function record_exec_cache!(event::Symbol, seconds::Real)
    lock(_EXEC_CACHE_STATS_LOCK) do
        s = EXEC_CACHE_STATS
        if event === :hit
            s.hits += 1
            s.load_seconds += seconds
        elseif event === :miss
            s.misses += 1
            s.compile_seconds += seconds
        elseif event === :store
            s.stores += 1
        elseif event === :failure
            s.failures += 1
        end
    end
    return nothing
end

"A snapshot of the executable-cache counters as a NamedTuple."
function exec_cache_snapshot()
    return lock(_EXEC_CACHE_STATS_LOCK) do
        s = EXEC_CACHE_STATS
        return (
            hits = s.hits, misses = s.misses, stores = s.stores, failures = s.failures,
            load_seconds = s.load_seconds, compile_seconds = s.compile_seconds,
        )
    end
end

"Reset the counters (tests)."
function reset_exec_cache_stats!()
    lock(_EXEC_CACHE_STATS_LOCK) do
        s = EXEC_CACHE_STATS
        s.hits = 0; s.misses = 0; s.stores = 0; s.failures = 0
        s.load_seconds = 0.0; s.compile_seconds = 0.0
    end
    return nothing
end
