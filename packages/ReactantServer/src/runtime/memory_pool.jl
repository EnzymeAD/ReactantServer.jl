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
end

# Back-compat constructor: default autotune on, so existing call sites and test mocks that pass the
# original five fields keep the current behavior.
MemoryPool(backend::AbstractBackend, client, device, platform::String, ctx) =
    MemoryPool(backend, client, device, platform, ctx, true)

function resolve_client(backend::AbstractBackend, cfg::RuntimeConfig)
    platform = cfg.backend == CUDA_BACKEND ? "cuda" : "cpu"
    try
        client = make_client(backend, platform; mem_fraction=cfg.mem_fraction, preallocate=cfg.preallocate)
        device = select_device(backend, client, cfg.device_ordinal)
        return MemoryPool(backend, client, device, platform, make_context(backend), cfg.autotune)
    catch err
        if cfg.backend == CUDA_BACKEND && cfg.allow_cpu_fallback
            @warn "CUDA backend unavailable; falling back to CPU" exception=(err, catch_backtrace())
            client = make_client(backend, "cpu"; mem_fraction=cfg.mem_fraction, preallocate=cfg.preallocate)
            device = select_device(backend, client, 0)
            return MemoryPool(backend, client, device, "cpu", make_context(backend), cfg.autotune)
        end
        rethrow()
    end
end
