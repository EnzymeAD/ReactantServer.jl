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
function _apply_compile_cache_prefs(autotune_cache::Union{Bool,Nothing}, autotune_cache_dir::AbstractString)
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

function make_client(::ReactantBackend, platform::String; mem_fraction::Float64=0.9,
                     preallocate::Bool=true, autotune_cache::Union{Bool,Nothing}=nothing,
                     autotune_cache_dir::AbstractString="", kwargs...)
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

function execute_single_device(::ReactantBackend, exec, device, buffers::AbstractVector,
                               donated::AbstractVector{Bool}, num_outputs::Int)
    in_ptrs = (Ptr{Cvoid}[b.buffer for b in buffers]...,)
    don = (UInt8[d ? 0x1 : 0x0 for d in donated]...,)
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
        return (in_use = in_use, limit = limit, free = max(limit - in_use, 0),
                peak_in_use = Int(stats.peak_bytes_in_use),
                pool_bytes = _orz(stats.pool_bytes), peak_pool_bytes = _orz(stats.peak_pool_bytes))
    catch
        return nothing
    end
end

function compile_artifact(backend::ReactantBackend, pool::MemoryPool, mlir_bytes,
                          n_parameters::Int, n_outputs::Int)
    ctx = pool.ctx
    _RMLIR.IR.activate(ctx)
    try
        # The C wrapper accepts String/AbstractString but not Vector{UInt8}; pass a binary
        # String that preserves all bytes (including NULs) in the portable artifact.
        artifact = String(copy(Vector{UInt8}(mlir_bytes)))
        mlir_mod = _RMLIR.API.stablehloDeserializePortableArtifactNoError(artifact, ctx)
        mod = _RMLIR.IR.Module(mlir_mod)
        # Portable artifacts may carry TF32 baked into dot_general as an explicit DotAlgorithm, which
        # is a hard compile error on non-Ampere targets. Strip it when this device cannot run it,
        # before XLA sees the module; on a TF32-capable GPU leave it intact.
        tf32_supported(pool.client, pool.device) || maybe_strip_tf32!(mod)
        # When autotuning is disabled, force xla_gpu_autotune_level=0: XLA uses default gemm/conv
        # algorithm selection with no device timing trials. This removes the autotuner's run-to-run
        # non-determinism and the compile-time scratch that otherwise inflates the startup memory
        # probe on the first (un-cached) start. When enabled, pass no override so the compile is
        # byte-identical to the previous behavior (XLA's default autotune level).
        opts = pool.autotune ?
            _RXLA.make_compile_options(; device_id=Int64(device_ordinal(backend, pool.device))) :
            _RXLA.make_compile_options(; device_id=Int64(device_ordinal(backend, pool.device)),
                xla_debug_options=(; xla_gpu_autotune_level=Int32(0)))
        return _RXLA.compile(pool.client, mod;
            compile_options=opts,
            num_parameters=Int64(n_parameters),
            num_outputs=Int64(n_outputs),
            is_sharded=false,
            num_replicas=Int64(1),
            num_partitions=Int64(1),
        )
    finally
        _RMLIR.IR.deactivate(ctx)
    end
end
