# The PJRT C API backend: the CUDA runtime driven through libReactantExtra's exported PJRT_Api
# table (see pjrt_capi.jl) instead of Reactant's C++ shims.
#
# Same XLA, same compiler, same compile options as the Reactant backend (outputs are byte-identical
# on the same build), with two capabilities the shim layer lacks: serialized executables, which
# power the per-bundle executable cache (executable_cache.jl), and a resettable allocator high-water
# mark, which makes the startup memory probe immune to autotuning scratch. Reactant stays in the
# picture for what it does well: the MLIR context and the StableHLO numerics rewrite (tf32.jl),
# compile-option construction, and the autotune-cache preferences.
#
# GPU only: the JLL exports only the GPU plugin's table (the CPU plugin defines the same C symbol
# and is not linked), so `resolve_client` keeps the Reactant backend for CPU. Reactant's own XLA
# client is never created by this backend; any access to `Reactant.XLA.client` or the backend state
# would build a second client and a second BFC arena on the same GPU.

struct PJRTCAPIBackend <: AbstractBackend end

supports_executable_cache(::PJRTCAPIBackend) = true

# What Reactant's GPU client constructor applies before the first compile (accelerators/GPU.jl);
# part of the cache key because it is a global LLVM codegen option.
const _NVPTX_FMA_OPT = "--nvptx-fma-level=1"

function make_client(
        ::PJRTCAPIBackend, platform::String; mem_fraction::Float64 = 0.9,
        preallocate::Bool = true, autotune_cache::Union{Bool, Nothing} = nothing,
        autotune_cache_dir::AbstractString = "", kwargs...
    )
    (platform == "cuda" || platform == "gpu") ||
        error("the PJRT C API backend serves CUDA only (Reactant_jll exports no CPU PJRT table); use the Reactant backend for '$platform'")
    ok, reason = PJRTCAPI.availability()
    ok || error("PJRT C API backend unavailable: $reason")
    _apply_compile_cache_prefs(autotune_cache, autotune_cache_dir)
    _RXLA.LLVMclopts(_NVPTX_FMA_OPT)
    client = PJRTCAPI.create_client(; mem_fraction = mem_fraction, preallocate = preallocate)
    client.platform == "cuda" ||
        error("PJRT C API backend: expected a cuda client, got platform '$(client.platform)'")
    isempty(client.devices) && error("PJRT C API backend: the client has no addressable devices")
    attrs = PJRTCAPI.plugin_attributes()
    @info "PJRT C API backend" api_version = join(string.(PJRTCAPI.api_version()), ".") platform_version = client.platform_version devices = length(client.devices) device_kind = first(client.devices).kind compute_capability = first(client.devices).compute_capability cuda_version = get(attrs, "cuda_version", nothing) reactant_jll = pkgversion(Reactant_jll)
    return client
end

make_context(::PJRTCAPIBackend) = make_context(ReactantBackend())

function select_device(::PJRTCAPIBackend, client::PJRTCAPI.Client, ordinal::Int)
    0 <= ordinal < length(client.devices) ||
        error("device ordinal $ordinal out of range; client has $(length(client.devices)) device(s)")
    return client.devices[ordinal + 1]
end

device_ordinal(::PJRTCAPIBackend, device::PJRTCAPI.Device) = device.ordinal

to_device(::PJRTCAPIBackend, client::PJRTCAPI.Client, array::Array, device::PJRTCAPI.Device) =
    PJRTCAPI.to_device(client, array, device)

buffer_eltype(::PJRTCAPIBackend, buffer::PJRTCAPI.Buffer) = buffer.eltype
buffer_size(::PJRTCAPIBackend, buffer::PJRTCAPI.Buffer) = Tuple(Int.(buffer.dims))
to_host!(::PJRTCAPIBackend, buffer::PJRTCAPI.Buffer, dest::Array) = PJRTCAPI.to_host!(buffer, dest)
free_buffer!(::PJRTCAPIBackend, buffer::PJRTCAPI.Buffer) = PJRTCAPI.destroy!(buffer)
free_executable!(::PJRTCAPIBackend, exec::PJRTCAPI.LoadedExecutable) = PJRTCAPI.destroy!(exec)

function execute_single_device(
        ::PJRTCAPIBackend, exec::PJRTCAPI.LoadedExecutable, device::PJRTCAPI.Device,
        buffers::AbstractVector, donated::AbstractVector{Bool}, num_outputs::Int
    )
    outs = PJRTCAPI.execute(exec, device, PJRTCAPI.Buffer[b for b in buffers], donated)
    length(outs) == num_outputs ||
        @warn "PJRT C API backend: executable returned an unexpected output count" expected = num_outputs got = length(outs)
    return Any[o for o in outs]
end

backend_tf32_capable(::PJRTCAPIBackend, pool::MemoryPool) =
    pool.platform == "cuda" && _tf32_capable("cuda", pool.device.compute_capability[1], pool.device.compute_capability[2])

function device_memory_stats(::PJRTCAPIBackend, pool::MemoryPool)
    try
        s = PJRTCAPI.memory_stats(pool.device)
        limit = s.bytes_limit > 0 ? s.bytes_limit :
            (pool.device.memory_limit > 0 ? pool.device.memory_limit : 0)
        limit > 0 || return nothing
        orz(x) = x < 0 ? 0 : x
        return (
            in_use = s.bytes_in_use, limit = limit, free = max(limit - s.bytes_in_use, 0),
            peak_in_use = orz(s.peak_bytes_in_use),
            pool_bytes = orz(s.pool_bytes), peak_pool_bytes = orz(s.peak_pool_bytes),
        )
    catch
        return nothing
    end
end

function clear_memory_stats!(::PJRTCAPIBackend, pool::MemoryPool)
    try
        PJRTCAPI.clear_memory_stats!(pool.device)
        return true
    catch err
        @warn "PJRT C API backend: could not reset allocator statistics" exception = err maxlog = 1
        return false
    end
end

function compiled_memory_stats(::PJRTCAPIBackend, exec::PJRTCAPI.LoadedExecutable)
    try
        return PJRTCAPI.compiled_memory_stats(exec)
    catch
        return nothing
    end
end

# ── Compilation and the executable cache ─────────────────────────────────────────────────────────

# Serialize a (rewritten) module back to a StableHLO portable artifact at the current version. The
# bytes are what PJRT compiles and what the cache key hashes, so the numerics rewrite is in both.
function _portable_artifact_bytes(mod::_RMLIR.IR.Module)
    cb = @cfunction(_RMLIR.IR.print_callback, Cvoid, (_RMLIR.API.MlirStringRef, Any))
    vref = Ref(IOBuffer())
    _RMLIR.API.stablehloGetCurrentVersion(cb, vref)
    ver = String(take!(vref[]))
    ref = Ref(IOBuffer())
    res = _RMLIR.API.stablehloSerializePortableArtifactFromModule(mod, ver, cb, ref, true)
    _RMLIR.IR.isfailure(_RMLIR.IR.LogicalResult(res)) && error("failed to serialize the StableHLO module to a portable artifact")
    return take!(ref[])
end

function _compile_options_bytes(pool::MemoryPool, device_id::Int)
    opts = pool.autotune ?
        _RXLA.make_compile_options(; device_id = Int64(device_id)) :
        _RXLA.make_compile_options(; device_id = Int64(device_id), xla_debug_options = (; xla_gpu_autotune_level = Int32(0)))
    return Reactant.ProtoUtils.proto_to_bytes(opts)
end

# The cache partition a program is valid for: everything XLA does not verify on load.
function _cache_target(pool::MemoryPool)
    d = pool.device
    return string(
        "jll-", pkgversion(Reactant_jll), "_", target_slug(pool.client.platform_version), "_",
        target_slug(d.kind), "_sm", d.compute_capability[1], d.compute_capability[2]
    )
end

# Everything besides the source file that determines the compiled program, as a stable string.
# Compile options are described rather than hashed byte-for-byte: their defaults come from the
# Reactant version (covered), the per-host autotune cache path must not split the key, and the
# device ordinal is placed at load time through the compile-options override.
function _cache_policy(pool::MemoryPool, tf32_capable::Bool)
    return string(
        "reactant=", pkgversion(Reactant), ";autotune=", pool.autotune, ";numerics=", pool.numerics,
        ";tf32=", tf32_capable, ";llvm=", _NVPTX_FMA_OPT, ";format=", EXEC_CACHE_FORMAT
    )
end

function compile_artifact(
        backend::PJRTCAPIBackend, pool::MemoryPool, mlir_bytes,
        n_parameters::Int, n_outputs::Int;
        numerics_stats::Union{NumericsStats, Nothing} = nothing,
        cache::Union{ExecutableCacheSlot, Nothing} = nothing
    )
    ctx = pool.ctx
    _RMLIR.IR.activate(ctx)
    try
        artifact = String(copy(Vector{UInt8}(mlir_bytes)))
        mlir_mod = _RMLIR.API.stablehloDeserializePortableArtifactNoError(artifact, ctx)
        mod = _RMLIR.IR.Module(mlir_mod)
        tf32 = backend_tf32_capable(backend, pool)
        _apply_numerics_policy!(mod, pool, tf32, numerics_stats)
        program = _portable_artifact_bytes(mod)
        opts = _compile_options_bytes(pool, device_ordinal(backend, pool.device))
        if cache !== nothing
            key = sha256hex(vcat(program, Vector{UInt8}(codeunits(_cache_policy(pool, tf32)))))
            path = entry_path(cache, _cache_target(pool), key)
            blob = lookup_entry(path)
            if blob !== nothing
                t0 = time()
                try
                    exec = PJRTCAPI.deserialize_and_load(pool.client, blob, opts)
                    exec.num_outputs == n_outputs ||
                        error("cached program has $(exec.num_outputs) outputs, expected $n_outputs")
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
            exec = PJRTCAPI.compile(pool.client, program, opts)
            elapsed = time() - t0
            record_exec_cache!(:miss, elapsed)
            try
                bytes = PJRTCAPI.serialize(exec)
                stored = store_entry(path, bytes)
                stored && record_exec_cache!(:store, 0.0)
                @info "executable cache: miss, compiled and stored" source = cache.source compile_seconds = round(elapsed; digits = 3) bytes = length(bytes) stored path
            catch err
                @warn "executable cache: compiled program could not be serialized; serving the uncached compile" source = cache.source exception = err
            end
            return exec
        end
        return PJRTCAPI.compile(pool.client, program, opts)
    finally
        _RMLIR.IR.deactivate(ctx)
    end
end
