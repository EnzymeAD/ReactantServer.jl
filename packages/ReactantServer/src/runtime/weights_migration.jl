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
# Workers on one node serialize on an flock in the bundle's `.cache/`, so one bundle is migrated
# once. A migration never fails a model: errors are logged and the bundle loads as it was.

using SHA: SHA

const LEGACY_WEIGHTS_FILE = ReactantServerCore.DEFAULT_WEIGHTS_FILE
const XLA_COPY_FILE = "weights.T.safetensors"
const MIGRATED_FROM_KEY = "migrated_from"                 # source "size:mtime" (cheap staleness check)
const MIGRATED_FROM_SHA_KEY = "migrated_from_sha256"      # source content hash (checked before replace)
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
# version of it. With `prove`, every tensor is verified too.
function _copy_matches(tpath, legacy::WeightsFile; prove::Bool)
    isfile(tpath) || return false
    return _with_weights(tpath) do t
        md = something(t.metadata, Dict{String, String}())
        (t.xla_order && get(md, MIGRATED_FROM_KEY, nothing) == _source_token(legacy)) || return false
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

function _build_copy!(dir, legacy::WeightsFile)
    need = filesize(legacy.path) + _MIGRATION_SPACE_MARGIN
    avail = _AVAILABLE_BYTES[](dir)
    avail >= need || _refuse("not enough free space in $dir: need $(Base.format_bytes(need)), have $(Base.format_bytes(avail))")
    md = Dict{String, String}(something(legacy.metadata, Dict{String, String}()))
    md[AXIS_ORDER_KEY] = AXIS_ORDER_XLA
    md[MIGRATED_FROM_KEY] = _source_token(legacy)
    md[MIGRATED_FROM_SHA_KEY] = _sha256_hex(legacy)
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
`:replaced`, or `:failed` (logged; the bundle is left loadable as it was).
"""
function migrate_bundle_weights!(dir::AbstractString, mode::WeightsMigration)
    mode == MIGRATE_OFF && return :off
    isfile(joinpath(dir, LEGACY_WEIGHTS_FILE)) || isfile(joinpath(dir, XLA_COPY_FILE)) || return :skipped
    t0 = time()
    try
        result = _with_migration_lock(() -> _migrate!(String(dir), mode), dir)
        result in (:copied, :replaced) &&
            @info "migrated weights to the program's axis order" bundle = dir result seconds = round(time() - t0; digits = 2)
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

    # Already in the program's axis order: finish a replace that stopped after its rename, or done.
    if isfile(lpath) && _with_weights(w -> w.xla_order, lpath)
        wf == XLA_COPY_FILE || return _drop_copy!(dir, lpath, tpath, :current)
        mode == MIGRATE_REPLACE || return :current
        return _finish_replace!(dir, lpath, tpath)
    end
    # The original is gone: nothing to prove a copy against, so leave the bundle as it is.
    isfile(lpath) || return wf == XLA_COPY_FILE ? :current : :skipped

    return _with_weights(lpath) do legacy
        changed = false
        if wf == LEGACY_WEIGHTS_FILE || !_copy_matches(tpath, legacy; prove = false)
            # No copy yet, a copy from a crash, or a stale copy (the original was redeployed).
            _copy_matches(tpath, legacy; prove = true) || _build_copy!(dir, legacy)
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

# Remove the copy's name once weights.safetensors is the same file.
function _drop_copy!(dir, lpath, tpath, result)
    if isfile(tpath)
        if samefile(lpath, tpath)
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
whose source `weights.safetensors` has changed since (a redeploy of the original) is stale, and
the original is loaded instead, so a deploy is never ignored.
"""
function resolve_weights_path(dir::AbstractString, m::Manifest)
    path = joinpath(dir, m.weights_file)
    m.weights_file == XLA_COPY_FILE || return path
    lpath = joinpath(dir, LEGACY_WEIGHTS_FILE)
    (isfile(path) && isfile(lpath)) || return path
    recorded = _with_weights(w -> get(something(w.metadata, Dict{String, String}()), MIGRATED_FROM_KEY, nothing), path)
    recorded === nothing && return path
    recorded == _source_token(stat(lpath)) && return path
    _with_weights(w -> w.xla_order, lpath) && return path   # replaced mid-way: both current
    @warn "$(XLA_COPY_FILE) predates the current $(LEGACY_WEIGHTS_FILE); loading the original" bundle = dir
    return lpath
end
