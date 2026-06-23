# Meta-model execution: a meta model is a user-authored Julia workflow (a bundle's model.jl that
# calls register_meta_model) that chains several models with data-dependent logic in between. Its
# orchestration runs on the gRPC request task (off the GPU dispatch loop, like the pre/post hooks),
# under a per-worker gate that admits only one meta at a time (see scheduler.jl `infer`). Each sub-call
# goes through an injected `QueueingCaller` that re-enters the local scheduler in-process: the sub-model
# dispatches on the loop (so the GPU is free for other models during the meta's CPU glue), but a meta's
# in-flight sub-call is COMMITTED, so it jumps the queue for the next GPU slot and is never shed on the
# EDF laxity prediction (only on the base deadline-passed check). There is no loopback gRPC and no
# shared-memory transport. The sub-models are internal to the meta (the gateway never routes to them);
# placement keeps them resident with their meta.

abstract type ModelCaller end

# In-process caller. A sub-call re-enters the local scheduler's `infer` as a COMMITTED request: it is
# queued (so it coalesces and the GPU stays serial), jumps the line for the next dispatch, and is
# exempt from the laxity drop. Carries the worker's local reuse pool for `call.scratch`. The scheduler
# itself carries the backend/pool/weight-cache, so the sub-call needs only a handle to it.
struct QueueingCaller <: ModelCaller
    sched::Scheduler
    scratch::Union{BufferPool,Nothing}
end

# The local reuse pool a meta's `call.scratch` allocates from (or nothing -> plain heap arrays).
caller_pool(c::QueueingCaller) = c.scratch

# `deadline_ns` is an absolute local `time_ns()` deadline (0 = none). In-process, the sub-call shares
# this worker's clock, so the deadline is passed straight through to the sub-request; the scheduler
# drops it at admission only if it has already expired (committed requests skip the laxity drop). The
# sub-model's own preprocess/postprocess run inside `infer` on this task.
call_model(c::QueueingCaller, name::AbstractString, inputs::Vector{NamedTensor};
           requested_outputs::Vector{String}=String[], deadline_ns::Integer=0) =
    infer(c.sched, InferRequest(String(name), requested_outputs, inputs, Int64(deadline_ns));
          committed=true)

# The injected `call` handed to a meta model's `run(inputs, call)`. Callable to dispatch a sub-call
# (`call(name, inputs)`), and exposes `call.scratch(...)` to request reuse buffers. Tracks the pool
# slots acquired this request so `run_meta` releases them when the orchestration returns.
mutable struct MetaCall{C<:ModelCaller}
    caller::C
    name::String
    declared::Set{String}
    slots::Vector{PoolSlot}
    scratched::Bool          # call.scratch may be used at most once per request (see _scratch)
    deadline_ns::Int64       # absolute local time_ns() deadline for the whole orchestration (0 = none)
    call_ns::Int64           # nanoseconds spent inside sub-calls this request; the Julia glue between
                             # calls is deliberately NOT counted (see `run_meta` / `_run_meta_request`)
end
MetaCall(caller::ModelCaller, name::AbstractString, declared; deadline_ns::Integer=0) =
    MetaCall(caller, String(name), Set{String}(declared), PoolSlot[], false, Int64(deadline_ns), Int64(0))

# `call(name, inputs)` — dispatch a sub-call, rejecting undeclared callees. Bail before issuing if
# the orchestration's deadline has passed (no point starting more GPU work the caller has abandoned),
# and pass the deadline down so the in-process sub-call short-circuits too.
function (mc::MetaCall)(name::AbstractString, inputs::Vector{NamedTensor};
                        requested_outputs::Vector{String}=String[])
    String(name) in getfield(mc, :declared) ||
        error("meta model '$(getfield(mc, :name))' called undeclared model '$name'; add it to meta.calls")
    dl = getfield(mc, :deadline_ns)
    dl != 0 && Int64(time_ns()) >= dl && throw(DeadlineExceeded(getfield(mc, :name)))
    # Time only the sub-call itself: this is the meta's GPU/model-call cost. The data-dependent Julia
    # glue between calls runs outside this window and is intentionally excluded from the meta's compute.
    t0 = time_ns()
    result = call_model(getfield(mc, :caller), name, inputs;
                        requested_outputs=requested_outputs, deadline_ns=dl)
    setfield!(mc, :call_ns, getfield(mc, :call_ns) + Int64(time_ns() - t0))
    return result
end

# `call.scratch(...)` resolves to a closure over this MetaCall; every other field reads normally.
function Base.getproperty(mc::MetaCall, s::Symbol)
    s === :scratch && return (args...) -> _scratch(mc, args...)
    return getfield(mc, s)
end

"""
    call.scratch(dims, T) -> Array{T}
    call.scratch([dims1 => T1, dims2 => T2, ...]) -> Vector{Array}

Request reuse buffers for this request. With a pool, all requested buffers are carved from ONE
contiguous acquired block (released when the orchestration returns); without a pool, plain heap
arrays are returned. Identical model.jl code either way: each buffer is an `Array` that can be
handed straight to the next stage as a `NamedTensor` by reference. The pool exists purely to keep large
intermediates (e.g. an ROI feature tensor) off the per-request allocation path and hold down GC
pressure: a meta writes into the pooled buffer and hands it to the next stage by reference (the call
is in-process, so there is no copy). The pool is plain `Memory`, local to the worker, never shared.

Call this at most once per request — ask for ALL buffers up front in a single call (pass a vector of
`dims => T`). One call carves N buffers from ONE block (O(1) waste); N separate calls would grab N
blocks. A second call throws.
"""
_scratch(mc::MetaCall, dims, ::Type{T}) where {T} =
    first(_scratch(mc, Pair[(dims isa Tuple ? dims : (dims,)) => T]))

function _scratch(mc::MetaCall, reqs::AbstractVector)
    getfield(mc, :scratched) && error(
        "meta model '$(getfield(mc, :name))': call.scratch may be called only once per request; " *
        "request all buffers up front in a single call, e.g. call.scratch([dims1 => T1, dims2 => T2]).")
    setfield!(mc, :scratched, true)
    specs = [(Tuple(first(r)), last(r)) for r in reqs]   # dims => T pairs
    pool = caller_pool(getfield(mc, :caller))
    if pool === nothing
        return Any[Array{T}(undef, d...) for (d, T) in specs]
    end
    # One contiguous block fits everything (simple by design; revisit for fragmentation later). Since
    # metas run exclusively, the pool sees no cross-meta contention; the deadline-bounded acquire is a
    # harmless guard rather than a load-bearing back-pressure mechanism.
    total = sum(sizeof(T) * prod(d) for (d, T) in specs)
    span = max(1, cld(total, pool_slot_bytes(pool)))
    slot = try
        acquire_slot!(pool, span; deadline_ns = getfield(mc, :deadline_ns))
    catch e
        e isa PoolAcquireTimeout && throw(DeadlineExceeded(getfield(mc, :name)))
        rethrow()
    end
    push!(getfield(mc, :slots), slot)
    # Carve all N buffers from the one acquired block (advances the slot cursor in order).
    return scratch(slot, [d => T for (d, T) in specs])
end

"""
    run_meta(entry, caller, inputs; deadline_ns=0) -> Vector{NamedTensor}

Run a meta model's orchestration. The injected `call` (a [`MetaCall`](@ref)) dispatches sub-calls
through `caller` (a [`QueueingCaller`](@ref) that re-enters the local scheduler), rejecting any callee
not declared in `meta.calls`, and offers `call.scratch` for reuse buffers. `deadline_ns` is an absolute
local `time_ns()` deadline for the whole orchestration (0 = none): once it passes, the next
`call(...)` (or a slot acquire) bails with [`DeadlineExceeded`](@ref) rather than starting more GPU
work. Any pool slots acquired via `scratch` are released here when it returns (request-scoped). If
`call_ns_out` is given, the total nanoseconds spent inside sub-calls (the meta's GPU/model-call cost,
excluding the Julia glue between calls) is written to it, even when the orchestration throws.
"""
function run_meta(entry::MetaEntry, caller::ModelCaller, inputs::Vector{NamedTensor};
                  deadline_ns::Integer=0, call_ns_out::Union{Base.RefValue{Int64},Nothing}=nothing)
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
        call_ns_out === nothing || (call_ns_out[] = getfield(mc, :call_ns))
    end
end
