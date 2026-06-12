# Client staging pool + Triton/KServe shared-memory registration.
#
# The byte buffer and the concurrency-safe slot allocator live in ReactantServerCore
# (BufferPool, acquire_slot!/release_slot!, subslot, pool_view/pool_memory/pool_fsa). This file
# wraps a BufferPool with the server-registration bookkeeping and the per-URL routing that
# decides, per (host, port), whether to use the SHM-backed pool or the inline fallback.
#
# Concurrency: the old driver carved slots by a fixed index and relied on lockstep task
# awaiting to avoid reuse, which only held while a single top-level inference call owned the
# pool. The slot allocator in Core replaces that: every chunk draws a disjoint slot (possibly
# spanning several contiguous physical slots) from one shared allocator, so concurrent
# infer_async/infer_sync calls can never overlap.

function triton_unregister_shm()
    @lock _pools_lock begin
        shm = _shm_pool[]
        shm === nothing || unregister_pool!(shm)
    end
end

# ============================================================================
# InferenceBufferPool: a Core BufferPool plus server-registration bookkeeping.
#
# At most two pools exist per client:
#   - one SHM-backed pool registered with every server that shares this client's IPC namespace,
#   - one inline (Memory{UInt8}) pool for every server that doesn't.
# Membership is decided per (host, port) by probing the SHM pool with a
# SystemSharedMemoryRegister; the result is cached in `_pool_routes`.
# ============================================================================

mutable struct InferenceBufferPool
    pool::BufferPool
    registered_models::Vector{KServeModel}
    registered_keys::Set{Tuple{String,UInt16}}
    register_lock::ReentrantLock
end

function InferenceBufferPool(n_bytes::Integer; n_slots::Integer = 8, use_shm::Bool = true,
                             name::AbstractString = "reactant_server_client_pool")
    pool = BufferPool(n_bytes; n_slots = n_slots, use_shm = use_shm, name = name)
    return InferenceBufferPool(pool, KServeModel[], Set{Tuple{String,UInt16}}(), ReentrantLock())
end

Base.sizeof(p::InferenceBufferPool) = sizeof(p.pool)
is_shm_backed(p::InferenceBufferPool) = is_shm_backed(p.pool)
pool_name(p::InferenceBufferPool) = p.pool.name
slot_bytes(p::InferenceBufferPool) = p.pool.slot_bytes
n_slots(p::InferenceBufferPool) = p.pool.n_slots

# Slot acquisition delegates to the Core allocator; every caller draws from the same
# allocator. `span` requests that many physically contiguous slots as one range.
acquire_slot!(p::InferenceBufferPool, span::Integer = 1) = acquire_slot!(p.pool, span)

# ---- Triton/KServe SHM registration ----

# Register the pool's SHM region with the model's server. No-op for inline pools and for models
# whose (host, port) is already registered. Register failures propagate so the lazy-creation
# path can fall back to an inline pool. The pre-emptive unregister is best-effort and quiet.
function register_pool_with_model!(p::InferenceBufferPool, model::KServeModel)
    is_shm_backed(p) || return

    key = (model.host, model.port)
    lock(p.register_lock) do
        key in p.registered_keys && return

        client_register = grpc_shm_register_client(model)
        client_unregister = grpc_shm_unregister_client(model)

        try
            grpc_sync_request(client_unregister, SystemSharedMemoryUnregisterRequest(name = pool_name(p)))
        catch ex
            @info ex
        end

        grpc_sync_request(
            client_register,
            SystemSharedMemoryRegisterRequest(
                name = pool_name(p),
                key = shmid(p.pool.backing),
                offset = 0,
                byte_size = sizeof(p),
            ),
        )
        push!(p.registered_keys, key)
        push!(p.registered_models, model)
    end
    nothing
end

function unregister_pool!(p::InferenceBufferPool)
    is_shm_backed(p) || return
    lock(p.register_lock) do
        for m in p.registered_models
            try
                grpc_sync_request(grpc_shm_unregister_client(m),
                                  SystemSharedMemoryUnregisterRequest(name = pool_name(p)))
            catch ex
                @info ex
            end
        end
        empty!(p.registered_keys)
        empty!(p.registered_models)
    end
    nothing
end

# ---- Pool registry (two singletons + per-URL routing) ----

const PoolKey = Tuple{String,UInt16}
const _shm_pool = Ref{Union{InferenceBufferPool,Nothing}}(nothing)
const _inline_pool = Ref{Union{InferenceBufferPool,Nothing}}(nothing)
const _pool_routes = Dict{PoolKey,InferenceBufferPool}()
const _route_locks = Dict{PoolKey,ReentrantLock}()
const _pools_lock = ReentrantLock()
const _pool_bytes = Ref{Int}(DEFAULT_POOL_BYTES)
const _pool_slots = Ref{Int}(DEFAULT_POOL_SLOTS)

function _route_lock_for(key::PoolKey)
    @lock _pools_lock get!(() -> ReentrantLock(), _route_locks, key)
end

function _get_shm_pool!()
    @lock _pools_lock begin
        p = _shm_pool[]
        p === nothing || return p
        p = InferenceBufferPool(_pool_bytes[]; n_slots = _pool_slots[], use_shm = true)
        _shm_pool[] = p
        return p
    end
end

function _get_inline_pool!()
    @lock _pools_lock begin
        p = _inline_pool[]
        p === nothing || return p
        p = InferenceBufferPool(_pool_bytes[]; n_slots = _pool_slots[], use_shm = false)
        _inline_pool[] = p
        return p
    end
end

# Unregister the current SHM pool from every server it registered with and unlink its /dev/shm
# region. Caller must hold _pools_lock. Errors during unlink are logged and swallowed.
function _teardown_shm_pool!()
    shm = _shm_pool[]
    shm === nothing && return
    unregister_pool!(shm)
    if is_shm_backed(shm)
        try
            rm(shm.pool.backing)
        catch ex
            @warn "Failed to unlink SHM region $(pool_name(shm))" exception = ex
        end
    end
    _shm_pool[] = nothing
    return
end

# Route a model to the SHM pool if its server can map our region, otherwise to the inline pool.
# The decision is made by attempting a SystemSharedMemoryRegister against the SHM pool the first
# time we see this (host, port); the result is cached. The server's register call only stores
# metadata, so the actual shm_open at inference time can still fail; callers handle that via
# migrate_to_inline!. The per-URL lock prevents N concurrent first-time callers from each
# sending their own register RPC and racing to overwrite _pool_routes.
function get_or_create_pool!(model::KServeModel)
    key = (model.host, model.port)
    cached = @lock _pools_lock get(_pool_routes, key, nothing)
    cached === nothing || return cached

    lock(_route_lock_for(key)) do
        cached = @lock _pools_lock get(_pool_routes, key, nothing)
        cached === nothing || return cached

        shm_pool = _get_shm_pool!()
        try
            register_pool_with_model!(shm_pool, model)
            @lock _pools_lock _pool_routes[key] = shm_pool
            return shm_pool
        catch ex
            @info "SHM probe failed for $(model.host):$(model.port); using inline transport" exception = ex
            inline_pool = _get_inline_pool!()
            @lock _pools_lock _pool_routes[key] = inline_pool
            return inline_pool
        end
    end
end

# Predicate for the at-inference SHM mmap failure: the server reports NOT_FOUND when our
# register call recorded metadata but the actual shm_open at ModelInfer time could not see our
# region. Scoped narrowly so other gRPC errors (deadline, model-execution INTERNAL) propagate.
function _is_shm_not_found_error(ex)
    occursin("Unable to find shared memory region", sprint(showerror, ex))
end

# Force a model's URL onto the inline pool and tell its server to forget our SHM region.
# Idempotent and serialized per URL by _route_lock_for.
function migrate_to_inline!(model::KServeModel)
    key = (model.host, model.port)
    inline_pool = _get_inline_pool!()

    lock(_route_lock_for(key)) do
        already = @lock _pools_lock get(_pool_routes, key, nothing)
        already === inline_pool && return inline_pool
        @lock _pools_lock _pool_routes[key] = inline_pool

        shm = _shm_pool[]
        if shm !== nothing
            lock(shm.register_lock) do
                if key in shm.registered_keys
                    try
                        grpc_sync_request(grpc_shm_unregister_client(model),
                                          SystemSharedMemoryUnregisterRequest(name = pool_name(shm)))
                    catch ex
                        @info ex
                    end
                    delete!(shm.registered_keys, key)
                    filter!(m -> (m.host, m.port) != key, shm.registered_models)
                end
            end
        end
        inline_pool
    end
end
