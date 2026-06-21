# Meta-model execution: a meta model is a user-authored Julia workflow (a bundle's model.jl that
# calls register_meta_model) that chains several models with data-dependent logic in between. Its
# orchestration runs on the gRPC request task (off the GPU dispatch loop, exactly like the
# pre/post hooks), and issues ordinary inference calls through an injected ModelCaller. The caller
# abstracts the destination: in single-worker mode it calls the local scheduler in-process; in
# multi-worker mode it calls back into the gateway over gRPC, so a backbone on another worker is
# reached transparently. The scheduler, dispatch loop, and weight cache never see meta models.

# Environment variable naming the loopback gRPC endpoint (the gateway in multi-worker mode). When
# unset, the worker has no gateway and meta sub-calls go in-process through the local scheduler.
const LOOPBACK_ENV = "REACTANT_LOOPBACK_GRPC"

# Concurrent-stream cap for the loopback handle all of a worker's meta sub-calls share. The gRPCClient
# default (GRPC_MAX_STREAMS=16) is far too small here: a worker admits up to `max_concurrent_requests`
# (default 64) metas, each issuing a sub-call, and the acquire of a stream slot (`take!(sem)`) is an
# UNBOUNDED wait done before any deadline applies — so a saturated handle parks metas past their
# deadline. Size it to the worker's inbound cap so the acquire effectively never blocks.
const LOOPBACK_MAX_STREAMS_ENV = "REACTANT_LOOPBACK_MAX_STREAMS"
_loopback_max_streams() = something(tryparse(Int, strip(get(ENV, LOOPBACK_MAX_STREAMS_ENV, ""))), 64)

abstract type ModelCaller end

# In-process caller: routes a sub-call straight through the local scheduler's `infer`. No gRPC, no
# serialization. Used when no loopback gateway is configured (single-worker deployments).
struct LocalCaller <: ModelCaller
    sched::Scheduler
end

# `deadline_ns` is an absolute local `time_ns()` deadline (0 = none). In-process, the sub-call shares
# this worker's clock, so the meta's absolute deadline is passed straight through to the sub-request;
# the scheduler then drops it at admission if it has already expired.
call_model(c::LocalCaller, name::AbstractString, inputs::Vector{NamedTensor};
           requested_outputs::Vector{String}=String[], deadline_ns::Integer=0) =
    infer(c.sched, InferRequest(String(name), requested_outputs, inputs, Int64(deadline_ns)))

# Loopback caller: routes a sub-call to the gateway over gRPC. The gateway's existing routing places
# the backbone on whichever worker hosts it, so device placement stays abstracted from the meta
# author. One client over one libcurl multi handle, built once at startup and reused concurrently.
# `pool` is this worker's shared-memory fan-out region (or nothing): inputs whose bytes already live
# in it are sent by SHM reference instead of inlined (see `_build_subcall_request`).
struct GatewayCaller{C} <: ModelCaller
    url::String
    client::C
    pool::Union{BufferPool,Nothing}
end

# The fan-out pool a caller can stage sub-call inputs into; nothing means inline-only (no wire).
caller_pool(::LocalCaller) = nothing
caller_pool(c::GatewayCaller) = c.pool

# Split "host:port" (optionally with a grpc:// / grpcs:// scheme) into (host, port).
function _split_loopback(url::AbstractString)
    s = String(url)
    for scheme in ("grpc://", "grpcs://", "http://", "https://")
        startswith(s, scheme) && (s = s[(length(scheme) + 1):end])
    end
    i = findlast(==(':'), s)
    i === nothing && throw(ArgumentError("loopback endpoint '$url' is not host:port"))
    host = s[1:(i - 1)]
    port = tryparse(Int, s[(i + 1):end])
    port === nothing && throw(ArgumentError("loopback endpoint '$url' has a non-numeric port"))
    return host, port
end

function GatewayCaller(url::AbstractString; deadline::Real=300,
                       max_msg_bytes::Integer=_MAX_MESSAGE_BYTES,
                       pool::Union{BufferPool,Nothing}=nothing)
    host, port = _split_loopback(url)
    # sticky=false: meta orchestrations run on the worker's default (compute) thread pool, so the
    # multi handle's driving tasks must be schedulable on whichever thread issues the call.
    grpc = gRPCClient.gRPCCURL(; sticky=false, max_streams=_loopback_max_streams())
    client = GRPCInferenceService_ModelInfer_Client(host, port; grpc=grpc, deadline=deadline,
        TRequest=ModelInferRequest, TResponse=ModelInferResponse,
        max_send_message_length=max_msg_bytes, max_recieve_message_length=max_msg_bytes)
    return GatewayCaller{typeof(client)}(String(url), client, pool)
end

# Byte offset of `a`'s data within the pool region [base, base+nbytes), or nothing if `a` is not a
# contiguous (dense) array whose bytes lie entirely inside the region. Detection is by memory
# location, not object identity, so a reshape/contiguous prefix of a scratch buffer is still found.
function _region_offset(a::AbstractArray, base::Ptr, nbytes::Integer)
    a isa DenseArray || return nothing            # DenseArray guarantees a contiguous pointer range
    p = UInt(pointer(a)); b = UInt(base)
    (b <= p && p + UInt(sizeof(a)) <= b + UInt(nbytes)) ? Int(p - b) : nothing
end

# Build a sub-call request. All-or-nothing per call (the decode path treats raw_input_contents as
# parallel-to-inputs): if every input's bytes live in `pool`, send by SHM reference; otherwise inline.
# `parameters` carries the request-level KV map (the remaining-budget timeout), which rides through
# the gateway's raw-byte forwarding unchanged to the destination worker.
function _build_subcall_request(pool::Union{BufferPool,Nothing}, name::AbstractString,
                                inputs::Vector{NamedTensor}, requested_outputs::Vector{String},
                                parameters::Dict{String,inference.InferParameter})
    if pool !== nothing && !isempty(inputs)
        base = pool_base_pointer(pool); nbytes = sizeof(pool)
        offs = Int[]
        ok = true
        for t in inputs
            o = _region_offset(t.data, base, nbytes)
            o === nothing ? (ok = false; break) : push!(offs, o)
        end
        ok && return encode_infer_request_shm(name, inputs, pool_region_name(pool), offs;
                                              requested_outputs=requested_outputs, parameters=parameters)
        # A pool exists but some input is not a contiguous pool buffer -> inline. Diagnostic only (at
        # @debug): a meta legitimately passes large NON-scratch inputs through (e.g. the stage1 image),
        # so this is normal, not a warning; it's useful only when chasing why a buffer that WAS meant
        # to be scratch ended up inlined (a copy/strided view of a scratch buffer).
        for t in inputs
            sizeof(t.data) > 1_000_000 && _region_offset(t.data, base, nbytes) === nothing &&
                @debug "meta sub-call input inlined (not a contiguous pool buffer)" model = name input = t.name bytes = sizeof(t.data)
        end
    end
    return encode_infer_request(name, inputs; requested_outputs=requested_outputs, parameters=parameters)
end

# `deadline_ns` is an absolute local `time_ns()` deadline (0 = none). The sub-call crosses a process
# boundary (loopback gRPC, possibly via the gateway to another worker), so the absolute deadline is
# carried two ways, both as a RELATIVE remaining budget recomputed here: (1) the request-level timeout
# KV param (rides through the gateway's raw-byte forwarding) and (2) the per-call gRPC deadline (the
# gateway decrements this grpc-timeout further for transit). The destination worker honors the
# tightest. We also bound the WHOLE sub-call by the budget locally (`_await_within`) so the meta never
# parks past its deadline on the one wait a deadline cannot reach: the stream-semaphore acquire.
function call_model(c::GatewayCaller, name::AbstractString, inputs::Vector{NamedTensor};
                    requested_outputs::Vector{String}=String[], deadline_ns::Integer=0)
    if deadline_ns == 0
        req = _build_subcall_request(c.pool, name, inputs, requested_outputs, deadline_params(0))
        return decode_infer_response(gRPCClient.grpc_sync_request(c.client, req))
    end
    budget_ns = Int64(deadline_ns) - Int64(time_ns())
    budget_ns <= 0 && throw(DeadlineExceeded(String(name)))
    req = _build_subcall_request(c.pool, name, inputs, requested_outputs, deadline_params(budget_ns))
    return decode_infer_response(_await_within(c.client, req, budget_ns, String(name)))
end

# Issue one sub-call bounded by `budget_s`. The request runs on its own task with a per-call gRPC
# deadline of `budget_s` (so curl times out, and the grpc-timeout it sends carries the budget
# downstream). A timer races it; on expiry we interrupt the request task and raise DeadlineExceeded,
# so the meta stops waiting and frees its worker handler slot at the deadline instead of parking. The
# interrupt unwinds a task stuck in the stream-semaphore acquire (`take!(sem)`, which holds no slot);
# a task already past the acquire keeps its in-flight curl request, which the `budget_s` curl timeout
# then completes, releasing the slot in gRPCClient's own cleanup. `out` is buffered so neither the
# request task nor the timer task can block on `put!` (and an interrupted task's catch still fits).
function _await_within(client, req, budget_ns::Int64, name::AbstractString)
    budget_s = budget_ns / 1e9
    # Per-call gRPC deadline in WHOLE SECONDS (grpc_timeout_header_val only encodes integer seconds
    # cleanly; a fractional value becomes a >8-digit nanosecond string the receiver rejects). Ceil so
    # it is never 0 (which disables curl's timeout) and stays >= the precise timer below, so the timer
    # is what actually bounds us at the true budget.
    deadline_s = max(1, cld(budget_ns, 1_000_000_000))
    out = Channel{Tuple{Symbol,Any}}(2)
    # Threads.@spawn, not @async: these must stay migratable across the worker's compute pool. The
    # GatewayCaller handle is built sticky=false on purpose so its gRPC driving floats to any thread;
    # an @async task here would be pinned to the issuing thread and serialize meta sub-calls onto it.
    Threads.@spawn begin
        try
            r = gRPCClient.grpc_async_request(client, req; deadline=deadline_s)
            put!(out, (:ok, gRPCClient.grpc_async_await(client, r)))
        catch e
            put!(out, (:err, e))
        end
    end
    timer = Timer(budget_s)
    Threads.@spawn (try; wait(timer); put!(out, (:timeout, nothing)); catch; end)
    tag, val = take!(out)
    close(timer)
    if tag === :timeout
        # Do NOT interrupt the request task. With the shared loopback handle that task may be driving
        # libcurl's multi-handle callbacks, and throwing an InterruptException into it there corrupts
        # the request and risks wedging the handle every meta sub-call on this worker shares. Instead
        # abandon it: its per-call curl deadline (`deadline_s`) terminates the in-flight request and
        # releases the stream slot in gRPCClient's own cleanup, and a task still parked on the
        # semaphore acquire holds no slot and drains harmlessly once one frees. The meta's worker
        # handler slot is freed the moment we return here, which is the point.
        throw(DeadlineExceeded(name))
    elseif tag === :err
        throw(val)
    end
    return val
end

"""
    build_caller(sched) -> ModelCaller

Construct the worker's process-wide meta-model caller from the environment: a [`GatewayCaller`](@ref)
when `REACTANT_LOOPBACK_GRPC` names a gateway (multi-worker mode), otherwise a [`LocalCaller`](@ref)
over the local scheduler (single-worker mode).
"""
# Env naming the worker's own fan-out region ("<key>:<bytes>:<n_slots>") and its peers' regions
# ("<key>:<bytes>,..."), injected by the supervisor's pool mesh. Absent => no fan-out pool (meta
# sub-calls inline, as before).
const FANOUT_SELF_ENV = "REACTANT_FANOUT_SELF"
const FANOUT_PEERS_ENV = "REACTANT_FANOUT_PEERS"

_region_basename(key::AbstractString) = startswith(key, "/") ? String(key[2:end]) : String(key)

# Attach a peer's region read-only into `shm`, retrying while it may not be created yet (all workers
# create their own region early, so a few seconds covers staggered startup). Best-effort: a peer that
# never attaches just means producers targeting this worker fall back to inline (still correct).
function _attach_peer!(shm::SharedMemoryRegistry, name::AbstractString, key::AbstractString,
                       bytes::Integer; budget_s::Real=60.0)
    t0 = time()
    while true
        try
            shm_register!(shm, name, key, 0, bytes)
            return true
        catch err
            if time() - t0 > budget_s
                @warn "fan-out: peer region never attached; sub-calls to it will inline" region = name exception = err
                return false
            end
            sleep(0.25)
        end
    end
end

# Build this worker's fan-out producer pool from the injected mesh and attach all peer regions
# read-only into `shm` (so a sub-call routed here can read a peer's staged tensor). Returns the own
# pool, or nothing if no mesh is configured or setup fails (meta sub-calls then inline — safe).
function setup_fanout(shm::SharedMemoryRegistry)
    self = strip(get(ENV, FANOUT_SELF_ENV, ""))
    isempty(self) && return nothing
    parts = split(self, ":")
    if length(parts) != 3
        @warn "fan-out: REACTANT_FANOUT_SELF malformed; meta sub-calls will inline" value = self
        return nothing
    end
    self_key = String(parts[1])
    own = try
        BufferPool(parse(Int, parts[2]); n_slots=parse(Int, parts[3]), use_shm=true, key=self_key)
    catch err
        @warn "fan-out: could not create own region; meta sub-calls will inline" exception = err
        return nothing
    end
    # Register our OWN region for READING too. The gateway can route a meta's sub-call back to the
    # SAME worker that produced it; there decode_infer_request -> shm_read must find this region. The
    # BufferPool (write) and the registry (read) are separate mappings of the same physical region.
    try
        shm_register!(shm, own.name, self_key, 0, sizeof(own))
    catch err
        @warn "fan-out: could not register own region for read-back; same-worker sub-calls will error on SHM" exception = err
    end
    for ent in split(strip(get(ENV, FANOUT_PEERS_ENV, "")), ","; keepempty=false)
        kb = split(ent, ":")
        length(kb) == 2 && _attach_peer!(shm, _region_basename(kb[1]), String(kb[1]), parse(Int, kb[2]))
    end
    @info "fan-out pool ready" region = own.name bytes = sizeof(own) slots = own.n_slots
    return own
end

function build_caller(sched::Scheduler; pool::Union{BufferPool,Nothing}=nothing)
    url = strip(get(ENV, LOOPBACK_ENV, ""))
    if isempty(url)
        return LocalCaller(sched)
    end
    @info "Meta models will route sub-calls through the loopback gateway" endpoint = url has_pool = pool !== nothing
    return GatewayCaller(String(url); pool=pool)
end

# The injected `call` handed to a meta model's `run(inputs, call)`. Callable to dispatch a sub-call
# (`call(name, inputs)`), and exposes `call.scratch(...)` to request fan-out buffers. Tracks the pool
# slots acquired this request so `run_meta` can release them when the orchestration returns.
mutable struct MetaCall{C<:ModelCaller}
    caller::C
    name::String
    declared::Set{String}
    slots::Vector{PoolSlot}
    scratched::Bool          # call.scratch may be used at most once per request (see _scratch)
    deadline_ns::Int64       # absolute local time_ns() deadline for the whole orchestration (0 = none)
end
MetaCall(caller::ModelCaller, name::AbstractString, declared; deadline_ns::Integer=0) =
    MetaCall(caller, String(name), Set{String}(declared), PoolSlot[], false, Int64(deadline_ns))

# `call(name, inputs)` — dispatch a sub-call, rejecting undeclared callees. Before issuing the
# sub-call, bail if the orchestration's deadline has already passed: there is no point starting more
# GPU work the caller has abandoned (the sub-call's own admission would drop it anyway, but bailing
# here also stops the meta from issuing the remaining stages). The deadline is propagated to the
# sub-call so its worker drops it at admission if it expires in flight.
function (mc::MetaCall)(name::AbstractString, inputs::Vector{NamedTensor};
                        requested_outputs::Vector{String}=String[])
    String(name) in getfield(mc, :declared) ||
        error("meta model '$(getfield(mc, :name))' called undeclared model '$name'; add it to meta.calls")
    dl = getfield(mc, :deadline_ns)
    dl != 0 && Int64(time_ns()) >= dl && throw(DeadlineExceeded(getfield(mc, :name)))
    return call_model(getfield(mc, :caller), name, inputs;
                      requested_outputs=requested_outputs, deadline_ns=dl)
end

# `call.scratch(...)` resolves to a closure over this MetaCall; every other field reads normally.
function Base.getproperty(mc::MetaCall, s::Symbol)
    s === :scratch && return (args...) -> _scratch(mc, args...)
    return getfield(mc, s)
end

"""
    call.scratch(dims, T) -> Array{T}
    call.scratch([dims1 => T1, dims2 => T2, ...]) -> Vector{Array}

Request fan-out scratch buffers for this request. With a pool (multi-worker), all requested buffers
are carved from ONE contiguous acquired block (released when the orchestration returns) and writing
into them then passing them to `call(...)` sends them by shared-memory reference. Without a pool
(single-worker, in-process), returns plain heap arrays — identical model.jl code either way.

MUST be called at most once per request: ask for ALL buffers up front in a single call (pass a vector
of `dims => T`). Two reasons, both enforced by rejecting a second call:
  1. Deadlock-freedom: a request only ever holds one block, so it never waits on the pool while
     holding a slot — there is no acquire-while-holding circular wait.
  2. Allocation efficiency: one call carves N buffers from ONE block (waste = the unused tail of the
     final slot, < slot_bytes, O(1) in N); N separate calls would grab N blocks (O(n) waste).
A second call throws — enforced in both modes so the anti-pattern surfaces in single-worker testing,
not just under multi-worker load.
"""
_scratch(mc::MetaCall, dims, ::Type{T}) where {T} =
    first(_scratch(mc, Pair[(dims isa Tuple ? dims : (dims,)) => T]))

function _scratch(mc::MetaCall, reqs::AbstractVector)
    getfield(mc, :scratched) && error(
        "meta model '$(getfield(mc, :name))': call.scratch may be called only once per request; " *
        "request all buffers up front in a single call, e.g. call.scratch([dims1 => T1, dims2 => T2]). " *
        "Repeated acquisition while holding a slot is rejected to keep the fan-out pool deadlock-free.")
    setfield!(mc, :scratched, true)
    specs = [(Tuple(first(r)), last(r)) for r in reqs]   # dims => T pairs
    pool = caller_pool(getfield(mc, :caller))
    if pool === nothing
        return Any[Array{T}(undef, d...) for (d, T) in specs]
    end
    # One contiguous block fits everything (simple by design; revisit for fragmentation later).
    total = sum(sizeof(T) * prod(d) for (d, T) in specs)
    span = max(1, cld(total, pool_slot_bytes(pool)))
    slot = acquire_slot!(pool, span)
    push!(getfield(mc, :slots), slot)
    return Any[pool_view(subslot(slot, sizeof(T) * prod(d)), T, d...) for (d, T) in specs]
end

"""
    run_meta(entry, caller, inputs; deadline_ns=0) -> Vector{NamedTensor}

Run a meta model's orchestration. The injected `call` (a [`MetaCall`](@ref)) dispatches sub-calls
through `caller`, rejecting any callee not declared in `meta.calls`, and offers `call.scratch` for
fan-out buffers. `deadline_ns` is an absolute local `time_ns()` deadline for the whole orchestration
(0 = none): once it passes, the next `call(...)` bails with [`DeadlineExceeded`](@ref) rather than
issuing more GPU work, and every sub-call carries the remaining budget so its worker drops it at
admission if it expires in flight. Any pool slots the orchestration acquired via `scratch` are
released here when it returns (request-scoped), robust to the author reusing a buffer across sub-calls.
"""
function run_meta(entry::MetaEntry, caller::ModelCaller, inputs::Vector{NamedTensor};
                  deadline_ns::Integer=0)
    mc = MetaCall(caller, entry.name, entry.calls; deadline_ns=deadline_ns)
    try
        # The orchestration is defined in a sandboxed model.jl (a newer world age), so cross it with
        # invokelatest, exactly as infer() does for pre/post hooks.
        out = Base.invokelatest(entry.run, inputs, mc)
        out isa Vector{NamedTensor} ||
            error("meta model '$(entry.name)' returned $(typeof(out)); expected Vector{NamedTensor}")
        return out
    finally
        for s in getfield(mc, :slots)
            release_slot!(s)
        end
    end
end
