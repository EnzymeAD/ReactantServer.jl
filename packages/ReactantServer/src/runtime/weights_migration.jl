# Converting a bundle's legacy-order weights file to the program's axis order (see weights_file.jl
# for the two layouts), under `runtime.weights_migration`.
#
# `copy` writes `weights.T.safetensors` beside the original and points the manifest's
# `weights_file` at it, leaving `weights.safetensors` untouched (servers that predate the field
# ignore it and keep loading the original). `replace` then makes the copy THE weights file: it marks
# the manifest `format_version: "2.1"` first (so older servers refuse the bundle rather than misread
# it), hard-links the copy over `weights.safetensors` with one rename, proves the result against the
# still-open original, and only then drops the pointer and the extra name.
#
# Nothing is accepted without proof: every tensor of a copy must equal, byte for byte, the array the
# legacy path makes of the original, which is computed twice, once by our reader and once by
# SafeTensors.jl, so a bug in our reader cannot approve its own output. Every file is written to a
# temporary name, fsynced, read back from disk and verified, then renamed; the directory is fsynced
# after each rename. Each step leaves a bundle that a 0.2.0 server loads correctly (the loader takes
# the layout from the file, not the manifest), so a crash anywhere resumes on the next load.
#
# `weights.safetensors` is what the operator deploys, so a copy stands in for it only while it is
# still the file the copy was made from, judged by content (see `_source_sha256`). A redeployed
# legacy original is loaded itself (and, with migration on, copied again). Once
# `weights.safetensors` is in the program's axis order (replace's swap, or an export deployed over
# the original) it is the weights file in every mode, and copy and replace drop the stale copy
# (replace also marking the manifest 2.1).
#
# Workers on one node serialize on an flock in the bundle's `.cache/`, so one bundle is migrated
# once. A migration never fails a model: errors are logged and the bundle loads as it was.

using SHA: SHA

const LEGACY_WEIGHTS_FILE = ReactantServerCore.DEFAULT_WEIGHTS_FILE
const XLA_COPY_FILE = "weights.T.safetensors"
const MIGRATED_FROM_KEY = "migrated_from"                 # source "size:mtime" (what 0.2.0 servers compare)
const MIGRATED_FROM_SHA_KEY = "migrated_from_sha256"      # source content hash (decides whether the copy is current)
const _MIGRATION_KEYS = (AXIS_ORDER_KEY, MIGRATED_FROM_KEY, MIGRATED_FROM_SHA_KEY)
const _MIGRATION_SPACE_MARGIN = 64 << 20

struct WeightsMigrationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::WeightsMigrationError) = print(io, "WeightsMigrationError: ", e.msg)
_refuse(msg) = throw(WeightsMigrationError(msg))

# Test seam: called with each step's name before it runs; a test makes it throw to simulate a crash.
const _MIGRATION_STEP_HOOK = Ref{Any}(nothing)
_step!(name::Symbol) = (h = _MIGRATION_STEP_HOOK[]; h === nothing || h(name); nothing)
# Test seam: free bytes on the filesystem holding a directory.
const _AVAILABLE_BYTES = Ref{Any}(dir -> Base.Filesystem.diskstat(dir).available)

_source_token(st) = string(st.size, ":", st.mtime)
_source_token(w::WeightsFile) = _source_token(stat(w.io))

function _sha256_hex(w::WeightsFile)
    seekstart(w.io)
    return bytes2hex(SHA.sha256(w.io))
end

# Whether a copy is current is decided by the SHA-256 of `weights.safetensors` against the one the
# copy records, never by size and mtime, which a redeploy can keep (reproducible builds, normalized
# timestamps, `rsync -t`, `cp -p`). Hashing is slow (SHA.jl runs at about 150 MB/s), so the result
# is kept in `.cache/`, keyed by the file's inode, size, mtime and ctime: a file renamed into place
# has another inode, and a rewrite in place moves the ctime, which no user-space call can set back.
# The device number is left out because NFS clients number one export differently, which would make
# workers on different nodes overwrite each other's record. Timestamps are coarse (a kernel tick
# locally, up to a second or two on some network and FAT filesystems), so a file changed less than
# `_SHA_RECORD_MIN_AGE` seconds before it is hashed is not recorded: a second rewrite within the same
# tick would keep its key. The record is a cache: a mismatched, missing or unwritable one costs a
# hash, never a wrong answer.
const _SOURCE_SHA_FILE = "weights_sha256.json"
const _SHA_RECORD_MIN_AGE = Ref(2.0)                       # test seam

_stat_identity(st) = string(st.inode, ":", st.size, ":", st.mtime, ":", st.ctime)

# The SHA-256 of the open `weights.safetensors` `w` of bundle `dir`.
function _source_sha256(dir, w::WeightsFile)
    st = stat(w.io)
    id = _stat_identity(st)
    record = joinpath(dir, ".cache", _SOURCE_SHA_FILE)
    rec = try
        isfile(record) ? JSON3.read(read(record, String), Dict{String, Any}) : nothing
    catch
        nothing                                            # unreadable: hash again and rewrite it
    end
    rec !== nothing && get(rec, "identity", nothing) == id && get(rec, "sha256", nothing) isa String &&
        return rec["sha256"]
    settled = time() - st.ctime >= _SHA_RECORD_MIN_AGE[]
    sha = _sha256_hex(w)
    (settled && _stat_identity(stat(w.io)) == id) || return sha     # changed lately, or while hashed
    tmp = record * ".tmp." * string(getpid()) * "." * string(rand(UInt32); base = 16)
    try
        mkpath(dirname(record))
        write(tmp, JSON3.write(Dict("identity" => id, "sha256" => sha)))
        Base.Filesystem.rename(tmp, record)
    catch e
        rm(tmp; force = true)
        @warn "could not record the SHA-256 of $(LEGACY_WEIGHTS_FILE); each load hashes it again" bundle = dir exception = e
    end
    return sha
end

_bytes(a) = reinterpret(UInt8, vec(a))
_same_bytes(a, b) = sizeof(a) == sizeof(b) && _bytes(a) == _bytes(b)

# ---------------------------------------------------------------------------------------------
# Durable file operations

_cfd(io::IOStream) = reinterpret(Cint, fd(io))

function _fsync(fdc::Cint, what)
    @ccall(gc_safe = true, fsync(fdc::Cint)::Cint) == 0 || throw(SystemError("fsync $what", Libc.errno()))
    return nothing
end
_fsync(io::IOStream, what) = (flush(io); _fsync(_cfd(io), what))

function _fsync_dir(dir)
    fdc = @ccall open(dir::Cstring, Base.Filesystem.JL_O_RDONLY::Cint)::Cint
    fdc < 0 && throw(SystemError("opening $dir", Libc.errno()))
    try
        _fsync(fdc, dir)
    finally
        @ccall close(fdc::Cint)::Cint
    end
    return nothing
end

_tmp_name(dir, name) = joinpath(dir, "." * name * ".tmp")

# Write `name` in `dir` through a temporary file with `like`'s permission bits: `fill(io)` writes
# it, then it is fsynced and `check(tmp)` must accept it before it is renamed into place.
function _durable_write(fill, dir, name; like, check = _ -> nothing)
    tmp = _tmp_name(dir, name)
    try
        open(tmp, "w") do io
            fill(io)
            _fsync(io, tmp)
        end
        chmod(tmp, filemode(like) & 0o7777)
        check(tmp)
        Base.Filesystem.rename(tmp, joinpath(dir, name))      # atomic; `mv(; force)` removes first
    catch
        rm(tmp; force = true)
        rethrow()
    end
    _fsync_dir(dir)
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Writing and verifying

_dtype_name(T) = string(Symbol(SafeTensors.type2tag(T)))

# Write `src` (a legacy file) in the program's axis order: each header shape reversed, each tensor's
# data the Julia array's memory. Streams one tensor at a time. The header is padded with spaces to a
# multiple of 8 bytes, so every tensor's data stays aligned.
function _write_xla_weights(io::IO, src::WeightsFile, metadata::Dict{String, String})
    names = sort!(collect(keys(src.tensors)); by = n -> src.tensors[n].offset)
    header = Dict{String, Any}("__metadata__" => metadata)
    off = 0
    for n in names
        info = src.tensors[n]
        header[n] = Dict{String, Any}(
            "dtype" => _dtype_name(info.eltype), "shape" => reverse(info.shape),
            "data_offsets" => [off, off + info.nbytes]
        )
        off += info.nbytes
    end
    json = JSON3.write(header)
    json *= " "^((8 - sizeof(json) % 8) % 8)
    write(io, htol(UInt64(sizeof(json))))
    write(io, json)
    for n in names
        write(io, read_weight(src, n))
    end
    return nothing
end

"""
    verify_transposed(old, new; reference = nothing) -> nothing

Prove that `new` is an exact copy of the legacy file `old` in the program's axis order.

`new` must be marked `axis_order: xla` and hold the same tensor names, dtypes and byte counts, each
header shape reversed, the same metadata apart from the migration keys, and every tensor's data
equal byte for byte (so NaN payloads and signed zeros count) to the array the legacy path reads.
With `reference` (the path of `old`), each legacy array is also read by SafeTensors.jl and must
match ours. Throws `WeightsMigrationError` naming the first difference.
"""
function verify_transposed(old::WeightsFile, new::WeightsFile; reference = nothing)
    old.xla_order && _refuse("$(old.path) is not in the legacy axis order")
    new.xla_order || _refuse("$(new.path) is not marked axis_order \"xla\"")
    om = something(old.metadata, Dict{String, String}())
    nm = something(new.metadata, Dict{String, String}())
    for (k, v) in om
        get(nm, k, nothing) == v || _refuse("metadata '$k' differs")
    end
    for k in keys(nm)
        (haskey(om, k) || k in _MIGRATION_KEYS) || _refuse("unexpected metadata '$k'")
    end
    keys(old.tensors) == keys(new.tensors) || _refuse("the files hold different tensors")
    st = reference === nothing ? nothing : SafeTensors.deserialize(reference; mmap = true)
    for (n, oi) in old.tensors
        ni = new.tensors[n]
        (oi.eltype == ni.eltype && oi.shape == ni.shape && oi.nbytes == ni.nbytes) ||
            _refuse("tensor '$n' differs in dtype or shape")
        served = read_weight(old, n)
        st === nothing || _same_bytes(served, collect(st[n])) ||
            _refuse("tensor '$n': our reader and SafeTensors.jl disagree on $(old.path)")
        _same_bytes(served, _read_raw(new, ni)) || _refuse("tensor '$n' differs")
    end
    return nothing
end

verify_transposed(old_path::AbstractString, new_path::AbstractString) = _with_weights(
    old -> _with_weights(new -> verify_transposed(old, new; reference = old_path), new_path), old_path
)

# Two files in the program's axis order with identical tensors and data.
function _same_content(a::WeightsFile, b::WeightsFile)
    (a.xla_order && b.xla_order && keys(a.tensors) == keys(b.tensors)) || return false
    for (n, ai) in a.tensors
        bi = b.tensors[n]
        (ai.eltype == bi.eltype && ai.shape == bi.shape) || return false
        _same_bytes(_read_raw(a, ai), _read_raw(b, bi)) || return false
    end
    return true
end

# ---------------------------------------------------------------------------------------------
# Manifest edits

# Set (or, with `nothing`, remove) top-level manifest keys, editing only those lines so comments
# and formatting survive. The edited text is parsed back and must equal the original with exactly
# these changes before it is written.
function _edit_manifest!(dir, changes::Pair{String}...)
    path = joinpath(dir, "manifest.yaml")
    text = read(path, String)
    expected = YAML.load(text; dicttype = Dict{String, Any})
    for (k, v) in changes
        rx = Regex("^" * k * ":.*(\\n|\$)", "m")
        if v === nothing
            text = replace(text, rx => "")
            delete!(expected, k)
        else
            line = k * ": \"" * v * "\"\n"
            text = occursin(rx, text) ? replace(text, rx => line) :
                (isempty(text) || endswith(text, "\n") ? text : text * "\n") * line
            expected[k] = v
        end
    end
    YAML.load(text; dicttype = Dict{String, Any}) == expected ||
        _refuse("editing $(first.(changes)) in $path would change other fields")
    _durable_write(io -> write(io, text), dir, "manifest.yaml"; like = path)
    return nothing
end

_manifest_dict(dir) = YAML.load_file(joinpath(dir, "manifest.yaml"); dicttype = Dict{String, Any})

# ---------------------------------------------------------------------------------------------
# The migration

function _with_migration_lock(f, dir)
    cache = joinpath(dir, ".cache")
    mkpath(cache)
    io = open(joinpath(cache, "migrate.lock"), "a")
    try
        fdc = _cfd(io)
        while ReactantServerCore._flock(fdc, ReactantServerCore._LOCK_EX) != 0
            Libc.errno() == Libc.EINTR || throw(SystemError("locking $(dir)/.cache/migrate.lock", Libc.errno()))
        end
        return f()
    finally
        close(io)                        # releases the lock
    end
end

function _with_weights(f, path)
    w, _ = open_weights_file(path)
    try
        return f(w)
    finally
        close_weights!(w)
    end
end

# The copy at `tpath` is usable for `legacy`: in the program's axis order and made from this
# version of it, whose SHA-256 is `sha`. With `prove`, every tensor is verified too.
function _copy_matches(tpath, legacy::WeightsFile, sha; prove::Bool)
    isfile(tpath) || return false
    return _with_weights(tpath) do t
        md = something(t.metadata, Dict{String, String}())
        (t.xla_order && get(md, MIGRATED_FROM_SHA_KEY, nothing) == sha) || return false
        prove || return true
        try
            verify_transposed(legacy, t; reference = legacy.path)
            return true
        catch e
            e isa WeightsMigrationError || rethrow()
            @warn "existing weights copy does not match its source; rebuilding it" path = tpath reason = e.msg
            return false
        end
    end
end

function _build_copy!(dir, legacy::WeightsFile, sha)
    need = filesize(legacy.path) + _MIGRATION_SPACE_MARGIN
    avail = _AVAILABLE_BYTES[](dir)
    avail >= need || _refuse("not enough free space in $dir: need $(Base.format_bytes(need)), have $(Base.format_bytes(avail))")
    md = Dict{String, String}(something(legacy.metadata, Dict{String, String}()))
    md[AXIS_ORDER_KEY] = AXIS_ORDER_XLA
    md[MIGRATED_FROM_KEY] = _source_token(legacy)
    md[MIGRATED_FROM_SHA_KEY] = sha
    _step!(:write_copy)
    _durable_write(
        io -> _write_xla_weights(io, legacy, md), dir, XLA_COPY_FILE; like = legacy.path,
        check = tmp -> _with_weights(t -> verify_transposed(legacy, t; reference = legacy.path), tmp)
    )
    return nothing
end

"""
    migrate_bundle_weights!(dir, mode::WeightsMigration) -> Symbol

Migrate one bundle's weights file to the program's axis order (see [`WeightsMigration`](@ref)).
Returns `:off`, `:skipped` (nothing to migrate: a meta bundle, or a custom `weights_file`),
`:current` (already in the program's axis order, or already copied under `copy`), `:copied`,
`:replaced`, or `:failed` (logged; the bundle is left loadable as it was). When a
`weights.safetensors` in the program's axis order was deployed over a copy-migrated bundle, the
manifest is pointed back at it and the copy, made from the file it replaced, is removed: under
`copy` that is `:current` and `format_version` is left as it is; under `replace` the manifest is
also marked `format_version: "2.1"`, as after any replace, and the result is `:replaced`.
"""
function migrate_bundle_weights!(dir::AbstractString, mode::WeightsMigration)
    mode == MIGRATE_OFF && return :off
    isfile(joinpath(dir, LEGACY_WEIGHTS_FILE)) || isfile(joinpath(dir, XLA_COPY_FILE)) || return :skipped
    t0 = monotonic_time()
    try
        result = _with_migration_lock(() -> _migrate!(String(dir), mode), dir)
        result in (:copied, :replaced) &&
            @info "migrated weights to the program's axis order" bundle = dir result seconds = round(monotonic_time() - t0; digits = 2)
        return result
    catch e
        @error "weights migration failed; loading the bundle as it is" bundle = dir exception = (e, catch_backtrace())
        for name in (XLA_COPY_FILE, LEGACY_WEIGHTS_FILE, "manifest.yaml")
            rm(_tmp_name(dir, name); force = true)
        end
        return :failed
    end
end

function _migrate!(dir::String, mode::WeightsMigration)
    for name in (XLA_COPY_FILE, LEGACY_WEIGHTS_FILE, "manifest.yaml")   # left by a crash
        rm(_tmp_name(dir, name); force = true)
    end
    d = _manifest_dict(dir)
    get(d, "kind", "model") == "meta" && return :skipped
    wf = get(d, "weights_file", LEGACY_WEIGHTS_FILE)
    wf in (LEGACY_WEIGHTS_FILE, XLA_COPY_FILE) || return :skipped
    lpath, tpath = joinpath(dir, LEGACY_WEIGHTS_FILE), joinpath(dir, XLA_COPY_FILE)

    # Already in the program's axis order: finish a replace that stopped after its rename, drop a
    # copy that an export deployed over the original made stale, or done.
    if isfile(lpath) && _with_weights(w -> w.xla_order, lpath)
        wf == XLA_COPY_FILE || return _drop_copy!(dir, lpath, tpath, :current)
        (isfile(tpath) && samefile(lpath, tpath)) || return _drop_stale_copy!(dir, lpath, tpath, mode)
        mode == MIGRATE_REPLACE || return :current
        return _finish_replace!(dir, lpath, tpath)
    end
    # The original is gone: nothing to prove a copy against, so leave the bundle as it is.
    isfile(lpath) || return wf == XLA_COPY_FILE ? :current : :skipped

    return _with_weights(lpath) do legacy
        sha = _source_sha256(dir, legacy)
        changed = false
        if wf == LEGACY_WEIGHTS_FILE || !_copy_matches(tpath, legacy, sha; prove = false)
            # No copy yet, a copy from a crash, or a stale copy (the original was redeployed).
            _copy_matches(tpath, legacy, sha; prove = true) || _build_copy!(dir, legacy, sha)
            _step!(:point_manifest)
            _edit_manifest!(dir, "weights_file" => XLA_COPY_FILE)
            changed = true
        end
        mode == MIGRATE_COPY && return changed ? :copied : :current
        _replace!(dir, legacy, lpath, tpath)
        return :replaced
    end
end

# `legacy` is the open original; the manifest points at a verified copy.
function _replace!(dir, legacy::WeightsFile, lpath, tpath)
    _with_weights(tpath) do t
        get(something(t.metadata, Dict{String, String}()), MIGRATED_FROM_SHA_KEY, nothing) == _sha256_hex(legacy) ||
            _refuse("$(tpath) was not made from the current $(LEGACY_WEIGHTS_FILE)")
        verify_transposed(legacy, t; reference = lpath)
    end
    _step!(:bump_format)
    _edit_manifest!(dir, "format_version" => "2.1")
    _step!(:swap)
    tmp = _tmp_name(dir, LEGACY_WEIGHTS_FILE)
    rm(tmp; force = true)
    hardlink(tpath, tmp)
    Base.Filesystem.rename(tmp, lpath)
    _fsync_dir(dir)
    # `legacy` still reads the original inode: prove the file now named weights.safetensors.
    _with_weights(w -> verify_transposed(legacy, w), lpath)
    _step!(:unpoint_manifest)
    return _finish_replace!(dir, lpath, tpath)
end

# weights.safetensors is in the program's axis order and the manifest still points at the copy.
function _finish_replace!(dir, lpath, tpath)
    if isfile(tpath) && !samefile(lpath, tpath)
        same = _with_weights(l -> _with_weights(t -> _same_content(l, t), tpath), lpath)
        same || _refuse("$(LEGACY_WEIGHTS_FILE) and $(XLA_COPY_FILE) differ; leaving both")
    end
    _edit_manifest!(dir, "format_version" => "2.1", "weights_file" => nothing)
    return _drop_copy!(dir, lpath, tpath, :replaced)
end

# weights.safetensors is in the program's axis order but is not the copy the manifest points at: an
# export was deployed over the original, so the copy, made from the file it replaced, is stale.
# Point the manifest back at weights.safetensors, then remove the copy. `replace` also marks the
# manifest 2.1 in the same edit, ending where `_finish_replace!` does (so `:replaced`: older servers
# now refuse the bundle); `copy` leaves `format_version` as it is (`:current`).
function _drop_stale_copy!(dir, lpath, tpath, mode)
    isfile(tpath) && !_made_by_migration(tpath) && return :skipped      # the manifest's own file
    if mode == MIGRATE_REPLACE
        _edit_manifest!(dir, "format_version" => "2.1", "weights_file" => nothing)
    else
        _edit_manifest!(dir, "weights_file" => nothing)
    end
    @info "$(LEGACY_WEIGHTS_FILE) is in the program's axis order; dropping $(XLA_COPY_FILE), which was made from an earlier version of it" bundle = dir
    return _drop_copy!(dir, lpath, tpath, mode == MIGRATE_REPLACE ? :replaced : :current)
end

# A copy written by `_build_copy!`, whatever weights.safetensors is now.
_made_by_migration(path) = _with_weights(w -> haskey(something(w.metadata, Dict{String, String}()), MIGRATED_FROM_SHA_KEY), path)

# Remove the copy's name once weights.safetensors is the same file, or once the copy is a stale one
# the manifest no longer names (no server loads it then).
function _drop_copy!(dir, lpath, tpath, result)
    if isfile(tpath)
        if samefile(lpath, tpath) || _made_by_migration(tpath)
            _step!(:unlink_copy)
            rm(tpath)
            _fsync_dir(dir)
        else
            @warn "leaving $(XLA_COPY_FILE), which is not the bundle's weights file" bundle = dir
        end
    end
    return result
end

# ---------------------------------------------------------------------------------------------
# Choosing the file to load, and saying when it costs a copy

const _LEGACY_WARNED = Set{String}()
const _LEGACY_WARNED_LOCK = ReentrantLock()

# Warn, once per model per process, that a model's weights are in the legacy axis order and so
# cost an extra transposed copy of every tensor on each load. Files whose tensors are all 0- or
# 1-dimensional cost nothing extra and are not mentioned.
function warn_legacy_layout(w, name::AbstractString)
    (w isa WeightsFile && !w.xla_order) || return nothing
    any(i -> length(i.shape) >= 2, values(w.tensors)) || return nothing
    first_time = @lock _LEGACY_WARNED_LOCK (name in _LEGACY_WARNED ? false : (push!(_LEGACY_WARNED, name); true))
    first_time || return nothing
    @warn "model '$name' stores its weights in the legacy axis order, so every load makes an extra " *
        "transposed copy of each tensor. To convert it, set runtime.weights_migration to `copy` " *
        "(keeps the original; older servers still load it) or `replace` (once every server is 0.2 " *
        "or later), run tools/migrate_weights.jl, or re-export it with ReactantServerExport 2.0." path = w.path
    return nothing
end

"""
    resolve_weights_path(dir, m::Manifest) -> String

The weights file to load for a bundle. Normally the manifest's `weights_file`; but a migrated copy
is loaded only while `weights.safetensors` is still the file it was made from, compared by SHA-256
(not by size and modification time, which a redeploy can keep). A `weights.safetensors` in the
program's axis order (an export deployed over the original) is loaded itself, and a redeployed
legacy original is loaded instead of the stale copy, so a deploy is never ignored, whatever
`runtime.weights_migration` is.
"""
function resolve_weights_path(dir::AbstractString, m::Manifest)
    path = joinpath(dir, m.weights_file)
    m.weights_file == XLA_COPY_FILE || return path
    lpath = joinpath(dir, LEGACY_WEIGHTS_FILE)
    (isfile(path) && isfile(lpath)) || return path
    made_from = _with_weights(w -> get(something(w.metadata, Dict{String, String}()), MIGRATED_FROM_SHA_KEY, nothing), path)
    made_from === nothing && return path           # not a migrated copy: the manifest's own file
    return _with_weights(lpath) do l
        if l.xla_order                             # replaced mid-way (the same file), or redeployed
            samefile(lpath, path) ||
                @warn "$(LEGACY_WEIGHTS_FILE) is in the program's axis order; loading it rather than $(XLA_COPY_FILE), which was made from an earlier version of it" bundle = dir
            return lpath
        end
        made_from == _source_sha256(dir, l) && return path
        @warn "$(XLA_COPY_FILE) predates the current $(LEGACY_WEIGHTS_FILE); loading the original" bundle = dir
        return lpath
    end
end
