# The single shared memory pool: one client and one device for the whole server.
#
# resolve_client selects the configured backend, falling back to CPU when CUDA is
# unavailable and the config permits it. All models share this one pool, which is the
# central efficiency property of the architecture.

mutable struct MemoryPool
    backend::AbstractBackend
    client::Any
    device::Any
    platform::String
    ctx::Any                 # backend compilation context (Reactant) or nothing
    autotune::Bool           # runtime.autotune; false => compile with the GPU autotuner disabled
    numerics::NumericsMode   # runtime.numerics; f32 matmul/conv precision policy (see NumericsMode)
end

# Back-compat constructors: default autotune on and hardware-adaptive numerics, so existing call
# sites and test mocks that pass the original five or six fields keep the current behavior.
MemoryPool(backend::AbstractBackend, client, device, platform::String, ctx) =
    MemoryPool(backend, client, device, platform, ctx, true, NUMERICS_AUTO)
MemoryPool(backend::AbstractBackend, client, device, platform::String, ctx, autotune::Bool) =
    MemoryPool(backend, client, device, platform, ctx, autotune, NUMERICS_AUTO)

function resolve_client(backend::AbstractBackend, cfg::RuntimeConfig)
    platform = cfg.backend == CUDA_BACKEND ? "cuda" : "cpu"
    try
        client = make_client(backend, platform; mem_fraction=cfg.mem_fraction, preallocate=cfg.preallocate,
                             autotune_cache=cfg.autotune_cache, autotune_cache_dir=cfg.autotune_cache_dir)
        device = select_device(backend, client, cfg.device_ordinal)
        return MemoryPool(backend, client, device, platform, make_context(backend), cfg.autotune, cfg.numerics)
    catch err
        if cfg.backend == CUDA_BACKEND && cfg.allow_cpu_fallback
            @warn "CUDA backend unavailable; falling back to CPU" exception=(err, catch_backtrace())
            client = make_client(backend, "cpu"; mem_fraction=cfg.mem_fraction, preallocate=cfg.preallocate,
                                 autotune_cache=cfg.autotune_cache, autotune_cache_dir=cfg.autotune_cache_dir)
            device = select_device(backend, client, 0)
            return MemoryPool(backend, client, device, "cpu", make_context(backend), cfg.autotune, cfg.numerics)
        end
        rethrow()
    end
end
