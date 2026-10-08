# The Reactant/PJRT backend. This is the ONLY file in the server that imports Reactant.
#
# It maps the backend protocol onto Reactant's runtime execution interface, exactly the
# call sequence validated by test/spike_reactant.jl. Reactant's tracing, compilation, and
# autodiff machinery are not used; only client/buffer/executable runtime calls and the
# StableHLO portable-artifact deserialization.

import Reactant

const _RXLA = Reactant.XLA
const _RMLIR = Reactant.MLIR

struct ReactantBackend <: AbstractBackend end

# Apply the persistent autotune-cache config (runtime.autotune_cache / autotune_cache_dir) to
# Reactant's compile cache. These settings live in mutable module-level Refs in
# Reactant.PersistentCompileCache, populated once from Preferences at its __init__ but read LIVE per
# compile by get_debug_options, so assigning them here (after `import Reactant`, before the first
# GPU compile) takes effect. They are unexported internals, so we guard with isdefined and no-op
# with a warning if a future Reactant renames them. `nothing` / "" mean "inherit LocalPreferences".
function _apply_compile_cache_prefs(autotune_cache::Union{Bool, Nothing}, autotune_cache_dir::AbstractString)
    (autotune_cache === nothing && isempty(autotune_cache_dir)) && return nothing
    if !isdefined(Reactant, :PersistentCompileCache)
        @warn "runtime.autotune_cache* set but Reactant.PersistentCompileCache not found; ignoring"
        return nothing
    end
    pcc = Reactant.PersistentCompileCache
    if !isdefined(pcc, :CACHE_DIR) || !isdefined(pcc, :AUTOTUNE_CACHE_ENABLED)
        @warn "runtime.autotune_cache* set but Reactant.PersistentCompileCache internals not found; ignoring"
        return nothing
    end
    isempty(autotune_cache_dir) || (pcc.CACHE_DIR[] = String(autotune_cache_dir))
    if autotune_cache !== nothing
        # autotune_cache_enabled() also requires CACHE_DIR !== nothing, so enabling without a
        # directory (neither configured here nor already set by LocalPreferences) cannot work.
        if autotune_cache && pcc.CACHE_DIR[] === nothing
            @warn "runtime.autotune_cache=true but no cache directory is set; also set runtime.autotune_cache_dir. Leaving the autotune cache disabled."
        else
            pcc.AUTOTUNE_CACHE_ENABLED[] = autotune_cache
        end
    end
    return nothing
end

function make_client(
        ::ReactantBackend, platform::String; mem_fraction::Float64 = 0.9,
        preallocate::Bool = true, autotune_cache::Union{Bool, Nothing} = nothing,
        autotune_cache_dir::AbstractString = "", kwargs...
    )
    if platform == "cuda" || platform == "gpu"
        # These BFC allocator knobs must be set before the GPU client is first created.
        _RXLA.XLA_REACTANT_GPU_MEM_FRACTION[] = mem_fraction
        _RXLA.XLA_REACTANT_GPU_PREALLOCATE[] = preallocate
        # Persistent autotune cache prefs must be set before the first compile (they are read per
        # compile); do it here, before the client/first executable exists.
        _apply_compile_cache_prefs(autotune_cache, autotune_cache_dir)
        return _RXLA.client("cuda")
    end
    return _RXLA.client("cpu")
end

function make_context(::ReactantBackend)
    if isdefined(Reactant, :registry) && Reactant.registry[] === nothing
        Reactant.initialize_dialect()
    end
    return Reactant.ReactantContext()
end

function select_device(::ReactantBackend, client, ordinal::Int)
    devices = _RXLA.addressable_devices(client)
    0 <= ordinal < length(devices) ||
        error("device ordinal $ordinal out of range; client has $(length(devices)) device(s)")
    return devices[ordinal + 1]
end

device_ordinal(::ReactantBackend, device) = Int(_RXLA.device_ordinal(device))

to_device(::ReactantBackend, client, array::Array, device) = _RXLA.PJRT.Buffer(client, array, device)

buffer_eltype(::ReactantBackend, buffer) = eltype(buffer)
buffer_size(::ReactantBackend, buffer) = size(buffer)

function to_host!(::ReactantBackend, buffer, dest::Array)
    _RXLA.to_host(buffer, dest, Reactant.Sharding.NoSharding())
    return dest
end

# Eager device-buffer release: run the buffer's registered finalizer now and unregister it so
# GC will not double-free it later. This reclaims GPU memory immediately on eviction instead of
# waiting for a stop-the-world GC.gc(). Reactant's XLA.free_buffer is itself the finalizer and
# does not null the pointer after PjRtBufferFree, so Base.finalize (run-once-and-unregister) is
# the safe way to trigger it without reaching into Reactant's field layout.
free_buffer!(::ReactantBackend, buffer) = (Base.finalize(buffer); nothing)

# Eager executable release, same rationale and mechanism as free_buffer!. Reactant's
# PJRT.LoadedExecutable frees the underlying PjRtLoadedExecutable in its GC finalizer
# (free_exec -> ExecutableFree); the executable object itself is tiny, so under low allocation
# pressure that finalizer can be deferred indefinitely while the executable's command buffers
# (CUDA graphs, allocated by the driver outside the BFC arena) stay resident. XLA destroys the
# command buffers when the executable is destroyed, so finalizing now reclaims them now.
free_executable!(::ReactantBackend, exec) = (Base.finalize(exec); nothing)

function _flatten_buffers!(acc, x)
    if x isa _RXLA.AbstractBuffer
        push!(acc, x)
    elseif x isa Tuple || x isa AbstractArray
        for e in x
            _flatten_buffers!(acc, e)
        end
    end
    return acc
end

function execute_single_device(
        ::ReactantBackend, exec, device, buffers::AbstractVector,
        donated::AbstractVector{Bool}, num_outputs::Int
    )
    in_ptrs = (Ptr{Cvoid}[b.buffer for b in buffers]...,)
    don = (UInt8[d ? 0x01 : 0x00 for d in donated]...,)
    outs = _RXLA.execute_sharded(exec, device, in_ptrs, don, Val(num_outputs))
    async = Any[]
    _flatten_buffers!(async, outs)
    return Any[_RXLA.synced_buffer(a) for a in async]
end

# Query the device allocator for memory usage. Only the CUDA client reports this; the CPU client
# (and any platform without an allocator-stats hook) throws, so failures degrade to `nothing`.
# `bytes_limit` is the BFC pool ceiling (mem_fraction of the card); fall back to the card's total
# global memory when the limit is unreported.
function device_memory_stats(::ReactantBackend, pool::MemoryPool)
    try
        stats = _RXLA.allocatorstats(pool.device)
        in_use = Int(stats.bytes_in_use)
        limit = stats.bytes_limit
        limit = (limit === nothing || limit <= 0) ?
            Int(_RXLA.device_properties(pool.device).totalGlobalMem) : Int(limit)
        _orz(x) = x === nothing ? 0 : Int(x)   # the BFC reports pool sizes only once it has allocated
        # `peak_in_use` is the allocator's session high-water mark (the empirical scratch + resident
        # ceiling). The GPU BFC allocator does not populate `largest_free_block_bytes` (it is left 0),
        # so we do not surface it; fragmentation is not directly observable from this allocator.
        return (
            in_use = in_use, limit = limit, free = max(limit - in_use, 0),
            peak_in_use = Int(stats.peak_bytes_in_use),
            pool_bytes = _orz(stats.pool_bytes), peak_pool_bytes = _orz(stats.peak_pool_bytes),
        )
    catch
        return nothing
    end
end

# Executable serialization and allocator-statistics control. The C API is in every Reactant_jll the
# [compat] floor admits (0.0.410+, EnzymeAD/Reactant.jl#3277); the Julia bindings are vendored in
# xla_serialization.jl until a Reactant release carries #3305. Point this at `_RXLA` then.
const _XS = VendoredXLA

supports_executable_cache(::ReactantBackend) = true

# Reset the allocator's high-water mark so the next measurement sees only what runs after it. Only the
# GPU allocator keeps statistics (the CPU device throws), so anything else reports false and callers
# keep the ordering-based fallback.
function clear_memory_stats!(::ReactantBackend, pool::MemoryPool)
    pool.platform == "cuda" || return false
    try
        _XS.clear_memory_stats!(pool.device)
        return true
    catch err
        @warn "could not reset the device allocator statistics; the memory probe uses the ordering-based estimate" exception = err maxlog = 1
        return false
    end
end

# The compiler's static memory accounting for a compiled program, in the shape the probe consumes.
function compiled_memory_stats(::ReactantBackend, exec)
    try
        s = _XS.compiled_memory_stats(exec)
        return (
            temp = Int(s.temp_size_in_bytes), outputs = Int(s.output_size_in_bytes),
            arguments = Int(s.argument_size_in_bytes),
            generated_code = Int(s.generated_code_size_in_bytes),
            peak = Int(s.peak_memory_in_bytes),
        )
    catch
        return nothing
    end
end

# Numerics policy (runtime.numerics; see tf32.jl and NumericsMode), applied to a freshly
# deserialized module before XLA sees it. AUTO follows the hardware: explicit TF32 DotAlgorithms are
# a hard compile error on non-Ampere targets, so strip them there; elsewhere TF32 resolves through
# DEFAULT precision. F32 pins hardware-invariant full-f32 numerics and machine-checks the result.
# TF32 passes the module through untouched; its capability gate ran once at startup (_bring_up).
function _apply_numerics_policy!(
        mod::_RMLIR.IR.Module, pool::MemoryPool, tf32_capable::Bool,
        numerics_stats::Union{NumericsStats, Nothing}
    )
    if pool.numerics == NUMERICS_F32
        st = pin_f32!(mod)
        inv = assert_f32_pinned(mod)
        if numerics_stats !== nothing
            numerics_stats.algorithms_rewritten += st.algorithms_rewritten
            numerics_stats.dots_pinned += st.dots_pinned
            numerics_stats.convs_pinned += st.convs_pinned
            append!(numerics_stats.opaque_ops, inv.opaque_ops)
        end
    elseif pool.numerics == NUMERICS_AUTO && !tf32_capable
        n = maybe_strip_tf32!(mod)
        numerics_stats === nothing || (numerics_stats.tf32_stripped += n)
    end
    return mod
end

# When autotuning is disabled, force xla_gpu_autotune_level=0: XLA uses default gemm/conv algorithm
# selection with no device timing trials. This removes the autotuner's run-to-run non-determinism and
# the compile-time scratch that otherwise inflates the startup memory probe on the first (un-cached)
# start. `runtime.xla_flags` is layered on top, so an explicit xla_gpu_autotune_level there wins.
# With neither, pass no override so the compile is byte-identical to the previous behavior.
function _compile_options(pool::MemoryPool, device_id::Int)
    pool.autotune && isempty(pool.xla_flags) &&
        return _RXLA.make_compile_options(; device_id = Int64(device_id))
    base = pool.autotune ? (;) : (; xla_gpu_autotune_level = Int32(0))
    return _RXLA.make_compile_options(;
        device_id = Int64(device_id), xla_debug_options = merge(base, NamedTuple(pool.xla_flags))
    )
end

# runtime.xla_flags against the DebugOptions proto of the linked XLA: an unknown name or a value
# that does not fit the field's type is a ConfigError at startup, rather than a Setfield error on the
# first compile or, worse, a flag that is silently not what the operator meant.
function resolve_xla_flags(::ReactantBackend, flags::AbstractDict)
    T = Reactant.Proto.xla.DebugOptions
    out = Pair{Symbol, Any}[]
    for (k, v) in flags
        name = Symbol(k)
        hasfield(T, name) ||
            throw(ConfigError("runtime.xla_flags.$k is not an XLA DebugOptions field in Reactant $(pkgversion(Reactant))"))
        push!(out, name => _xla_flag_value(fieldtype(T, name), String(k), v))
    end
    return sort!(out; by = first)
end

_xla_flag_error(k, v, what) = throw(ConfigError("runtime.xla_flags.$k must be $what, got $(repr(v))"))
_xla_flag_value(::Type{Bool}, k, v) = v isa Bool ? v : _xla_flag_error(k, v, "a boolean")
function _xla_flag_value(::Type{T}, k, v) where {T <: Integer}
    (v isa Integer && !(v isa Bool)) || _xla_flag_error(k, v, "an integer")
    typemin(T) <= v <= typemax(T) || _xla_flag_error(k, v, "an integer that fits $T")
    return T(v)
end
_xla_flag_value(::Type{T}, k, v) where {T <: AbstractFloat} =
    (v isa Real && !(v isa Bool)) ? T(v) : _xla_flag_error(k, v, "a number")
_xla_flag_value(::Type{String}, k, v) = v isa AbstractString ? String(v) : _xla_flag_error(k, v, "a string")
# Enum fields take the enumerator's name; repeated and message fields are not supported.
function _xla_flag_value(::Type{T}, k, v) where {T}
    names = try
        [string(Symbol(x)) => x for x in instances(T)]
    catch
        throw(ConfigError("runtime.xla_flags.$k has type $T, which runtime.xla_flags cannot set"))
    end
    if v isa AbstractString
        for (n, x) in names
            n == v && return x
        end
    end
    return _xla_flag_error(k, v, "one of " * join(first.(names), ", "))
end

# ── The executable cache (executable_cache.jl) on the Reactant backend ───────────────────────────

# Serialize a (rewritten) module back to a StableHLO portable artifact at the current version. These
# bytes are what the cache key hashes, so the numerics rewrite is part of the key. Serialization
# lowers `mod` to VHLO in place, so the module handed to XLA afterwards must be deserialized from
# these bytes again rather than reused.
function _portable_artifact_bytes(mod::_RMLIR.IR.Module)
    cb = @cfunction(_RMLIR.IR.print_callback, Cvoid, (_RMLIR.API.MlirStringRef, Any))
    vref = Ref(IOBuffer())
    _RMLIR.API.stablehloGetCurrentVersion(cb, vref)
    ver = String(take!(vref[]))
    ref = Ref(IOBuffer())
    res = _RMLIR.API.stablehloSerializePortableArtifactFromModule(mod, ver, cb, ref, true)
    _RMLIR.IR.isfailure(_RMLIR.IR.LogicalResult(res)) &&
        error("failed to serialize the StableHLO module to a portable artifact")
    return take!(ref[])
end

_reactant_jll_version() =
    isdefined(_RXLA, :Reactant_jll) ? string(pkgversion(_RXLA.Reactant_jll)) : "unknown"

# The cache partition a program is valid for: everything XLA does not verify on load (the library
# build, the platform, the device kind and, on CUDA, the compute capability).
function _cache_target(pool::MemoryPool)
    target = string("jll-", _reactant_jll_version(), "_", pool.platform, "_", target_slug(_RXLA.device_kind(pool.device)))
    if pool.platform == "cuda"
        props = _RXLA.device_properties(pool.device)
        target *= string("_sm", props.major, props.minor)
    end
    return target
end

# Everything besides the source file that determines the compiled program, as a stable string.
# Compile options are described rather than hashed byte-for-byte: their defaults come from the
# Reactant version (covered), the per-host autotune cache path must not split the key, and the
# device ordinal is placed at load time through the compile-options override.
function _cache_policy(pool::MemoryPool, tf32_capable::Bool)
    return string(
        "reactant=", pkgversion(Reactant), ";autotune=", pool.autotune, ";numerics=", pool.numerics,
        ";tf32=", tf32_capable, ";format=", EXEC_CACHE_FORMAT,
        # Only when set, so programs cached before runtime.xla_flags existed keep their keys.
        isempty(pool.xla_flags) ? "" : ";xla=" * join([string(k, "=", v) for (k, v) in pool.xla_flags], ",")
    )
end

# Compile a StableHLO portable artifact, or, with a cache slot, load the compiled program that an
# earlier start stored for this exact module, policy and device kind. Look-up, compile, serialize,
# store: a hit skips XLA entirely; a miss compiles and stores; an entry that fails to load is dropped
# and recompiled. Everything cache-related fails open into a plain compile.
function compile_artifact(
        backend::ReactantBackend, pool::MemoryPool, mlir_bytes,
        n_parameters::Int, n_outputs::Int;
        numerics_stats::Union{NumericsStats, Nothing} = nothing,
        cache::Union{ExecutableCacheSlot, Nothing} = nothing
    )
    ctx = pool.ctx
    _RMLIR.IR.activate(ctx)
    try
        # The C wrapper accepts String/AbstractString but not Vector{UInt8}; pass a binary
        # String that preserves all bytes (including NULs) in the portable artifact.
        artifact = String(copy(Vector{UInt8}(mlir_bytes)))
        mlir_mod = _RMLIR.API.stablehloDeserializePortableArtifactNoError(artifact, ctx)
        mod = _RMLIR.IR.Module(mlir_mod)
        tf32 = tf32_supported(pool.client, pool.device)
        _apply_numerics_policy!(mod, pool, tf32, numerics_stats)
        opts = _compile_options(pool, device_ordinal(backend, pool.device))
        program = (;
            compile_options = opts,
            num_parameters = Int64(n_parameters), num_outputs = Int64(n_outputs),
            is_sharded = false, num_replicas = Int64(1), num_partitions = Int64(1),
        )
        (cache === nothing || !supports_executable_cache(backend)) &&
            return _RXLA.compile(pool.client, mod; program...)

        program_bytes = _portable_artifact_bytes(mod)
        key = sha256hex(vcat(program_bytes, Vector{UInt8}(codeunits(_cache_policy(pool, tf32)))))
        path = entry_path(cache, _cache_target(pool), key)
        # The module the key was taken from is now VHLO; compile exactly the bytes that were hashed.
        compile_module() = _RMLIR.IR.Module(
            _RMLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(program_bytes)), ctx)
        )
        blob = lookup_entry(path)
        if blob !== nothing
            t0 = time()
            try
                exec = _XS.load_serialized_executable(pool.client, blob; program...)
                record_exec_cache!(:hit, time() - t0)
                @info "executable cache: hit" source = cache.source seconds = round(time() - t0; digits = 3) bytes = length(blob) path
                return exec
            catch err
                record_exec_cache!(:failure, 0.0)
                @warn "executable cache: cached program failed to load; recompiling" source = cache.source path exception = err
                drop_entry(path)
            end
        end
        t0 = time()
        exec = _RXLA.compile(pool.client, compile_module(); program...)
        elapsed = time() - t0
        record_exec_cache!(:miss, elapsed)
        try
            bytes = _XS.serialize_executable(exec)
            stored = store_entry(path, bytes)
            stored && record_exec_cache!(:store, 0.0)
            @info "executable cache: miss, compiled and stored" source = cache.source compile_seconds = round(elapsed; digits = 3) bytes = length(bytes) stored path
        catch err
            @warn "executable cache: compiled program could not be serialized; serving the uncached compile" source = cache.source exception = err
        end
        return exec
    finally
        _RMLIR.IR.deactivate(ctx)
    end
end
