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
    # Per-endpoint registration generation, bumped every time the pool is (re-)registered with an
    # endpoint. Used to coalesce concurrent re-registration after a server restart: the first failed
    # request bumps it, all other in-flight requests that observed the same generation skip the
    # re-register and just retry (see recover_registration!).
    reg_gen::Dict{Tuple{String,UInt16},Int}
    register_lock::ReentrantLock
end

function InferenceBufferPool(n_bytes::Integer; n_slots::Integer = 8, use_shm::Bool = true,
                             name::AbstractString = "reactant_server_client_pool")
    pool = BufferPool(n_bytes; n_slots = n_slots, use_shm = use_shm, name = name)
    return InferenceBufferPool(pool, KServeModel[], Set{Tuple{String,UInt16}}(),
                               Dict{Tuple{String,UInt16},Int}(), ReentrantLock())
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
        # One representative model per endpoint drives the teardown fan-out; re-registration reuses
        # the same key, so guard against pushing a duplicate on the recovery path.
        model in p.registered_models || push!(p.registered_models, model)
        p.reg_gen[key] = get(p.reg_gen, key, 0) + 1
    end
    nothing
end

# Current registration generation for an endpoint (0 if never registered). A request snapshots this
# before it sends, then hands it to recover_registration! so exactly one re-registration happens per
# detected restart no matter how many requests fail at once.
registration_gen(p::InferenceBufferPool, key::Tuple{String,UInt16}) =
    @lock p.register_lock get(p.reg_gen, key, 0)

# Re-register the pool with `model`'s endpoint after a stale-registration failure, coalescing
# concurrent callers. `observed_gen` is the generation the caller saw before its failed request:
# if another caller has already re-registered since (current generation moved on), this is a no-op
# and the caller simply retries; otherwise this caller performs the single re-registration. Register
# failures propagate so the caller can escalate to the inline fallback.
function recover_registration!(p::InferenceBufferPool, model::KServeModel, observed_gen::Integer)
    is_shm_backed(p) || return
    key = (model.host, model.port)
    lock(p.register_lock) do
        get(p.reg_gen, key, 0) == observed_gen || return   # someone already re-registered; just retry
        delete!(p.registered_keys, key)                    # force register_pool_with_model! to act
        register_pool_with_model!(p, model)                # re-registers and bumps reg_gen[key]
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

# Endpoints (PoolKey) whose route has been latched to the inline pool because shared memory could
# not be recovered in-band (see latch_inline!). A single background task (below) re-probes these and
# unlatches them back to SHM if the server recovers. Guarded by _pools_lock; the value is a
# representative model the poller re-probes with.
const _latched = Dict{PoolKey,KServeModel}()
const _reprobe_interval = Ref{Float64}(60.0)      # seconds; <= 0 disables the poller
const _reprobe_task = Ref{Union{Task,Nothing}}(nothing)
# The running loop is identified by a generation number. Bumping it (in _stop_reprobe!) invalidates
# the current loop so it returns at its next checkpoint, WITHOUT anyone blocking on the task -- the
# task may be parked in a gRPC probe, and waiting on that is the shutdown hang we must avoid.
const _reprobe_gen = Ref{Int}(0)
# Short per-probe deadline for the poller's own IsSameIPCNamespace call. A restarted-but-not-yet-ready
# or wedged server must not pin the background task; the normal request path keeps the 10s default.
const _REPROBE_RPC_DEADLINE = 3

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
function query_same_ipc_namespace(model::KServeModel, name::AbstractString; deadline = 10)
    client = grpc_is_same_ipc_namespace_client(model; deadline = deadline)
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

# ---- Inline latch + background re-probe (last-resort SHM fallback) ----
#
# When shared memory cannot be recovered in-band (re-registration failed, or the retried request was
# still stale), the endpoint is *latched* to the inline pool: its route is swapped so every later
# request goes inline with no per-request SHM probing. The only thing that re-attempts SHM for a
# latched endpoint is the background poller, which unlatches it once the server can register again.

# Route `model`'s endpoint to the inline pool and record it for the re-probe poller. Idempotent.
function latch_inline!(model::KServeModel)
    key = (model.host, model.port, model.shared_memory)
    already = @lock _pools_lock begin
        was = haskey(_latched, key)
        _pool_routes[key] = _get_inline_pool!()
        _latched[key] = model
        was
    end
    if !already
        @warn "shared memory unrecoverable for $(model.host):$(model.port); falling back to inline " *
              "transport (a background probe will restore SHM if the server recovers)"
    end
    _ensure_reprobe_running!()
    return _get_inline_pool!()
end

# Route a recovered endpoint back to the SHM pool and drop it from the latch set.
function unlatch_shm!(key::PoolKey)
    recovered = @lock _pools_lock begin
        haskey(_latched, key) || return
        _pool_routes[key] = _get_shm_pool!()
        delete!(_latched, key)
        true
    end
    recovered && @info "shared memory recovered for $(key[1]):$(key[2]); routing back to SHM"
    return
end

# Start the re-probe loop once (idempotent), unless the interval is non-positive (poller disabled).
# Each loop carries the generation it was started under; _stop_reprobe! bumps the generation to
# retire it.
function _ensure_reprobe_running!()
    _reprobe_interval[] > 0 || return
    @lock _pools_lock begin
        t = _reprobe_task[]
        (t !== nothing && !istaskdone(t)) && return
        g = (_reprobe_gen[] += 1)
        _reprobe_task[] = errormonitor(Threads.@spawn _shm_reprobe_loop(g))
    end
    return
end

# Retire the re-probe loop. Non-blocking by design: we bump the generation (so the loop returns at
# its next checkpoint) and drop our handle to it, but we do NOT wait for the task. The task may be
# parked in a blocking gRPC probe against a down/wedged server, and blocking shutdown on that is
# exactly the hang this avoids; the orphaned task observes the generation change and exits on its own.
# Safe to call with or without _pools_lock held.
function _stop_reprobe!()
    @lock _pools_lock begin
        _reprobe_gen[] += 1
        _reprobe_task[] = nothing
    end
    return
end

# One re-probe pass over the latched endpoints. For each, ask whether the server can see our SHM
# object again (IsSameIPCNamespace, under a short deadline); only on :yes do we attempt a real
# re-registration and, on success, unlatch back to SHM. A :no/:unknown answer or any error leaves the
# endpoint latched. Failed probes log at @debug only, so a persistently-degraded endpoint does not
# spam the log every tick.
function _reprobe_once()
    latched = @lock _pools_lock collect(_latched)   # snapshot of (key => model)
    isempty(latched) && return
    shm = _get_shm_pool!()
    for (key, model) in latched
        try
            query_same_ipc_namespace(model, shmid(shm.pool.backing);
                                     deadline = _REPROBE_RPC_DEADLINE) === :yes || continue
            # Force an actual re-register even if the endpoint is still in registered_keys.
            @lock shm.register_lock delete!(shm.registered_keys, (model.host, model.port))
            register_pool_with_model!(shm, model)
            unlatch_shm!(key)
        catch ex
            @debug "SHM re-probe failed; staying on inline" endpoint = "$(model.host):$(model.port)" exception = ex
        end
    end
    return
end

# Background loop: re-probe every `interval` seconds until this generation is retired. Sleeps in short
# slices so a stop (generation bump) is picked up within a slice even while idle.
function _shm_reprobe_loop(mygen::Int)
    while _reprobe_gen[] == mygen
        slept = 0.0
        while slept < _reprobe_interval[] && _reprobe_gen[] == mygen
            dt = min(0.5, _reprobe_interval[] - slept)
            sleep(dt)
            slept += dt
        end
        _reprobe_gen[] == mygen || break
        try
            _reprobe_once()
        catch ex
            @debug "SHM re-probe pass errored" exception = ex
        end
    end
    return
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
