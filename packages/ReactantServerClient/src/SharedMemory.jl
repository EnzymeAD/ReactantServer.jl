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
# Membership is decided per (host, port) by an explicit IsSameIPCNamespace probe (see
# get_or_create_pool!); the result is cached in `_pool_routes`. There is no silent runtime
# fallback: once a URL is routed to the SHM pool, a register or inference failure surfaces to
# the caller.
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

# Routes are keyed by (host, port, shared_memory mode): the chosen transport depends not only on
# the server but on the model's mode, so two models hitting the same endpoint with different modes
# (e.g. one :off and one :on) must not share a cached route -- otherwise :on could silently inherit
# an inline route instead of failing loudly.
const PoolKey = Tuple{String,UInt16,Symbol}
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

# Send the IsSameIPCNamespace probe and classify the answer. :yes / :no are the server's boolean;
# :unknown means the server does not implement the RPC (UNIMPLEMENTED, e.g. stock Triton). Any
# other gRPC error propagates: a probe that fails for an unrelated reason must not be silently
# read as "no shared memory".
function query_same_ipc_namespace(model::KServeModel, name::AbstractString)
    client = grpc_is_same_ipc_namespace_client(model)
    try
        resp = grpc_sync_request(client, IsSameIPCNamespaceRequest(name = name))
        return resp.same ? :yes : :no
    catch ex
        if ex isa gRPCClient.gRPCServiceCallException && ex.grpc_status == gRPCClient.GRPC_UNIMPLEMENTED
            return :unknown
        end
        rethrow()
    end
end

# Decide which pool a model's (host, port) routes to, per its `shared_memory` mode. See the
# KServeModel docstring for the full matrix. Register failures and the :on + different-namespace
# case surface to the caller; there is no silent fallback once SHM is chosen.
function _decide_pool!(model::KServeModel)
    mode = model.shared_memory
    mode === :off && return _get_inline_pool!()

    shm_pool = _get_shm_pool!()
    verdict = query_same_ipc_namespace(model, shmid(shm_pool.pool.backing))

    if verdict === :yes
        register_pool_with_model!(shm_pool, model)
        return shm_pool
    elseif verdict === :no
        if mode === :on
            error("shared_memory=:on for $(model.host):$(model.port), but the server reports it is " *
                  "not in this client's IPC namespace, so system shared memory cannot work. Use " *
                  "shared_memory=:auto to fall back to inline transport, or run the client and " *
                  "server in the same IPC namespace.")
        end
        return _get_inline_pool!()
    else  # :unknown -- the server does not implement IsSameIPCNamespace
        if mode === :on
            # Explicit opt-in (e.g. stock Triton): attempt SHM via the legacy register path.
            # Making it work across namespaces is the caller's responsibility.
            register_pool_with_model!(shm_pool, model)
            return shm_pool
        end
        return _get_inline_pool!()
    end
end

# Route a model to the SHM or inline pool the first time we see its (host, port); cache the
# result so every later call to the same URL skips the probe. The per-URL lock prevents N
# concurrent first-time callers from each probing and racing to overwrite _pool_routes.
function get_or_create_pool!(model::KServeModel)
    key = (model.host, model.port, model.shared_memory)
    cached = @lock _pools_lock get(_pool_routes, key, nothing)
    cached === nothing || return cached

    lock(_route_lock_for(key)) do
        cached = @lock _pools_lock get(_pool_routes, key, nothing)
        cached === nothing || return cached

        pool = _decide_pool!(model)
        @lock _pools_lock _pool_routes[key] = pool
        return pool
    end
end
