# ============================================================================
# AbstractInferenceIO contract
#
#   infer_encode_chunk!(io, r::UnitRange, slot::PoolSlot)::Union{PoolInferInput, AbstractVector}
#   item_input_bytes(io)::Int    # total bytes per item summed across all inputs
#   infer_decode_chunk!(io, r::UnitRange, response)::Nothing
#   length(io)::Int
#
# `infer_encode_chunk!` stages a chunk's inputs into the driver-provided `slot` and returns one
# descriptor per network input. The recommended way is `scratch`, which mirrors the meta-model
# `call.scratch`: request all input buffers up front by name, write into them, and return them —
#
#   function infer_encode_chunk!(io, r, slot)
#       n = length(r)
#       inputs = scratch(slot, ["INPUT__0" => ((4, n), Float32), "MASK" => ((n,), Int32)])
#       feats, mask = pool_view(inputs...)
#       for (j, i) in enumerate(r); feats[:, j] .= io.feats[i]; mask[j] = io.labels[i]; end
#       return inputs
#   end
#
# `item_input_bytes` must equal the per-item sum over those buffers, since the driver sizes the
# slot from it before calling `infer_encode_chunk!`. The lower-level path — carve `subslot`s,
# `pool_view` them, and return `InferInput(name, sub, shape, T)` (`PoolInferInput`) — is still
# accepted; both forms may be mixed in the returned vector.
# ============================================================================

# A deferred descriptor produced by infer_encode_chunk! (directly, or by `scratch`). The
# SHM-vs-inline choice is resolved by `materialize_input` at gRPC-request time, based on whether the
# pool that owns the slot is SHM-backed: SHM references the registered region by offset, while
# inline emits one raw little-endian byte blob per input into `raw_input_contents` (no typed
# `InferTensorContents`, no per-element encoding).
struct PoolInferInput
    name::String
    subslot::PoolSlot
    shape::Vector{Int}      # Julia column-major (batch axis last); reversed to row-major at the wire
    dtype::DataType
end

# `shape` is the Julia column-major shape of the chunk's input (batch axis last). It is reversed
# to the network's row-major order only when the wire tensor is materialized.
function InferInput(
        name::String,
        sub::PoolSlot,
        shape::AbstractVector{<:Integer},
        ::Type{T},
    ) where {T <: TritonType}
    return PoolInferInput(name, sub, Int.(shape), T)
end

# Convenience: derive dtype and the Julia column-major shape from a pool view.
function InferInput(
        name::String,
        sub::PoolSlot,
        view::AbstractArray{T},
    ) where {T <: TritonType}
    return PoolInferInput(name, sub, collect(Int, size(view)), T)
end

# Per-IO total bytes per item across every input.
function item_input_bytes end

# Concrete IOs add methods; declared as generic functions so downstream packages can dispatch on
# their concrete IO types. `infer_encode_chunk!` stages a chunk's inputs into the slot and returns
# the input descriptors; `infer_decode_chunk!` consumes the chunk's response. They are the request
# and response sides of the same per-chunk interface.
function infer_encode_chunk! end
function infer_decode_chunk! end

# ---- Output declaration (opt-in shared-memory outputs) ----

"""
    OutputSpec(name, dtype, per_item_dims)

Declares an expected network output so the driver can read it back through shared memory.
`name` is the output tensor name, `dtype` its element type, and `per_item_dims` its Julia
column-major shape for a single item, excluding the leading batch dimension. For each chunk the
driver reserves `sizeof(dtype) * prod(per_item_dims) * items_in_chunk` bytes in the staging slot
and asks the server to write that output there. See [`output_specs`](@ref).
"""
struct OutputSpec
    name::String
    dtype::DataType
    per_item_dims::Vector{Int}

    # Inner constructor (suppresses the default) to avoid ambiguity with the varargs form below.
    OutputSpec(name::AbstractString, dtype::Type{<:TritonType}, dims::AbstractVector{<:Integer}) =
        new(String(name), dtype, Int[d for d in dims])
end

OutputSpec(name::AbstractString, dtype::Type{<:TritonType}, dims::Integer...) =
    OutputSpec(name, dtype, Int[dims...])

"""
    output_specs(io) -> Vector{OutputSpec}

Outputs an [`AbstractInferenceIO`](@ref) declares for shared-memory read-back. Defaults to empty,
which keeps every output inline (`raw_output_contents`) exactly as before: no output declaration
means no shared memory, the safe fallback. Returning a non-empty vector opts the IO into
shared-memory outputs and **explicit-output mode**: the request asks the server for exactly these
outputs in this order, so every output the IO consumes must be declared. Outputs with
data-dependent (dynamic) shapes cannot be sized ahead of time and must stay inline (leave them
out, which means returning empty here).
"""
output_specs(::AbstractInferenceIO) = OutputSpec[]

# Per-item output bytes summed across declared outputs; 0 when none are declared.
item_output_bytes(io::AbstractInferenceIO) =
    sum(s -> sizeof(s.dtype) * prod(s.per_item_dims), output_specs(io); init = 0)

# ---- Materialize PoolInferInput descriptors against a model + pool. ----

"""
    materialize_input(d::PoolInferInput, model, pool) -> (tensor, raw)

Resolve one deferred descriptor against the pool: the wire tensor plus its payload. On an
SHM-backed pool the tensor carries the three `shared_memory_*` parameters and `raw` is empty
(`UInt8[]`); the payload stays in the registered region. On an inline pool the tensor carries
only `name`, `datatype`, and reversed row-major `shape` (no `contents` field), and `raw` is a
`Vector{UInt8}` aliasing the pool slot (zero-copy `unsafe_wrap`). `raw` is parallel to the
tensor: `_materialize_inputs` lines it up with `inputs` so `raw_input_contents[i]` corresponds
to `inputs[i]`. The server codec rejects a request whose `raw_input_contents` is non-empty
but does not match the input count; that all-or-nothing property holds because
`is_shm_backed` is a property of the pool, so every input in a chunk takes the same branch.

The raw blob is little-endian by specification while pool memory is host-native; the two agree
on every supported (little-endian) target. The driven path covers numeric `TritonType` values
only, so no `BYTES` dtype needs per-element length prefixes. Callers must keep the pool rooted
while the blob is alive (see `_run_chunk`'s `GC.@preserve`): the alias does not root the
pool's backing on its own, and it must not outlive the slot it points into.

On either transport, a descriptor whose shape needs more bytes than its subslot holds, or that has
a negative dimension, is refused with an `ArgumentError`.
"""
function materialize_input(
        d::PoolInferInput,
        model::AbstractInferenceModel,
        pool::InferenceBufferPool,
    )
    # Both transports hand the server exactly n_bytes starting at the subslot, so the subslot must
    # hold them (CLI-12). The inline branch's pool_view would catch an overrun, but the shared-memory
    # branch sends only an offset and a size, and the server would read past the subslot into the
    # next input, or into another chunk's data, without an error.
    all(>=(0), d.shape) ||
        throw(ArgumentError("input \"$(d.name)\": shape $(d.shape) has a negative dimension"))
    n_bytes = Base.checked_mul(Int(sizeof(d.dtype)), reduce(Base.checked_mul, d.shape; init = 1))
    n_bytes <= d.subslot.capacity || throw(
        ArgumentError(
            "input \"$(d.name)\": a $(d.dtype) tensor of shape $(d.shape) is $n_bytes bytes, more " *
                "than the $(d.subslot.capacity) bytes of the subslot it was staged in; carve the " *
                "subslot for this shape (scratch does)"
        )
    )
    return if is_shm_backed(pool)
        var"ModelInferRequest.InferInputTensor"(
                name = d.name,
                datatype = KSERVE_OUTPUT_DTYPE_TABLE_REVERSE[d.dtype],
                shape = reverse(d.shape),   # Julia column-major -> network row-major
                parameters = Dict(
                    "shared_memory_region" =>
                    InferParameter(parameter_choice = OneOf(:string_param, pool_name(pool))),
                    "shared_memory_offset" => InferParameter(
                        parameter_choice = OneOf(:int64_param, Int64(d.subslot.offset)),
                    ),
                    "shared_memory_byte_size" => InferParameter(
                        parameter_choice = OneOf(:int64_param, Int64(n_bytes)),
                    ),
                ),
            ), UInt8[]
    else
        # No typed contents: the payload travels in raw_input_contents as one byte blob per
        # input, aliased straight off the pool (zero-copy; the caller keeps the pool rooted
        # until the request is encoded and the response decoded).
        var"ModelInferRequest.InferInputTensor"(
                name = d.name,
                datatype = KSERVE_OUTPUT_DTYPE_TABLE_REVERSE[d.dtype],
                shape = reverse(d.shape),   # Julia column-major -> network row-major
            ), pool_view(d.subslot, UInt8, n_bytes)
    end
end

# Materialize descriptors into the wire tensors plus, for inline pools, the parallel
# raw_input_contents byte blobs. On the SHM branch every payload travels through the registered
# region, so `raws` stays empty (not one empty blob per input): the server codec rejects a
# request whose raw_input_contents is non-empty but does not match the input count, and that
# all-or-nothing property holds because `is_shm_backed` is a property of the pool, not of an
# individual input.
function _materialize_inputs(inputs, model, pool::InferenceBufferPool)
    tensors = var"ModelInferRequest.InferInputTensor"[]
    raws = Vector{Vector{UInt8}}()
    inline = !is_shm_backed(pool)
    for d in inputs
        tensor, raw = materialize_input(d, model, pool)
        push!(tensors, tensor)
        inline && push!(raws, raw)
    end
    return tensors, raws
end

# `infer_encode_chunk!` may return a single `PoolInferInput` (e.g. straight from the scalar
# `scratch(slot, name, dims, T)` form, the common single-input case) or a vector of them.
# Normalize either to an iterable of descriptors so the driver and `validate_io` treat both
# the same; a lone descriptor would otherwise be iterated field-by-field.
_encoded_inputs(d::PoolInferInput) = (d,)
_encoded_inputs(ds) = ds

# ---- scratch: request all of a chunk's input buffers at once (mirrors the meta `call.scratch`) ----

"""
    scratch(slot, name, dims, T) -> PoolInferInput
    scratch(slot, ["name1" => (dims1, T1), "name2" => (dims2, T2), ...]) -> Vector{PoolInferInput}

Carve one input buffer per named spec from the chunk's `slot`, advancing its cursor so the buffers
occupy disjoint, contiguous byte ranges, and return the wire descriptors ready to hand back from
`infer_encode_chunk!`. `dims` is the Julia column-major shape (per-item dims then the batch
axis), or a bare integer for a vector. Get the writable views with `pool_view`: one
descriptor returns one view, and splatting the vector returns all of them to destructure at once.

The scalar form returns one `PoolInferInput`; the vector form returns a `Vector{PoolInferInput}`.
`infer_encode_chunk!` accepts either as its return value, so a single-input IO can return the
scalar form directly without wrapping it in a vector.

```julia
function infer_encode_chunk!(io, r, slot)
    n = length(r)
    inputs = scratch(slot, ["INPUT__0" => ((4, n), Float32), "MASK" => ((n,), Int32)])
    feats, mask = pool_view(inputs...)
    for (j, i) in enumerate(r); feats[:, j] .= io.feats[i]; mask[j] = io.labels[i]; end
    return inputs
end
```

The returned vector is homogeneous (`Vector{PoolInferInput}`), so it never triggers element
promotion the way a literal of differently-typed arrays would. `item_input_bytes(io)` must equal the
per-item sum over these buffers, since the driver sizes the slot from it before `infer_encode_chunk!`
runs. The lower-level path (carve a `subslot`, `pool_view` it, build `InferInput` by hand) is still
supported and may be mixed into the returned vector.
"""
scratch(slot::PoolSlot, name::AbstractString, dims, ::Type{T}) where {T} =
    first(scratch(slot, [String(name) => (dims, T)]))

function scratch(slot::PoolSlot, specs::AbstractVector{<:Pair{<:AbstractString}})
    # Laid out by element alignment, largest first, so every buffer is aligned for its type with no
    # padding between them; the descriptors come back in the requested order (CORE-15).
    out = Vector{PoolInferInput}(undef, length(specs))
    for i in ReactantServerCore._scratch_order([last(last(s)) for s in specs])
        out[i] = _scratch_input(slot, first(specs[i]), last(specs[i]))
    end
    return out
end

function _scratch_input(slot::PoolSlot, name::AbstractString, spec)
    dims, T = first(spec), last(spec)
    d = dims isa Tuple ? dims : (dims,)
    sub = subslot(slot, sizeof(T) * prod(d); align = Base.datatype_alignment(T))
    return PoolInferInput(String(name), sub, Int[d...], T)
end

# Writable view of a scratch-carved input buffer, typed and shaped from its descriptor.
pool_view(d::PoolInferInput) = pool_view(d.subslot, d.dtype, d.shape...)

# Splat the descriptor vector to get every view at once and destructure them:
#   feats, mask = pool_view(inputs...)
pool_view(d1::PoolInferInput, ds::PoolInferInput...) = map(pool_view, (d1, ds...))

# ---- Chunking math ----
#
# Slots have a fixed size (set at pool construction). An item larger than one slot is staged
# across `span` physically contiguous slots acquired as one range; the chunk size is how many
# items fit in that range, capped by the model's max batch size.
function _chunk_geometry(io::AbstractInferenceIO, m::AbstractInferenceModel, pool::InferenceBufferPool)
    ipb = item_input_bytes(io)
    ipb > 0 || error(
        "item_input_bytes($(typeof(io))) returned $ipb; the pool driver needs a positive value",
    )
    # Shared-memory outputs are staged in the same slot as inputs, so they consume slot space.
    # Inline outputs travel in the gRPC response, not the slot, so they cost nothing here.
    opb = is_shm_backed(pool) ? item_output_bytes(io) : 0
    # A negative output size could make the chunk size 0, which never advances (CLI-4).
    opb >= 0 || error(
        "item_output_bytes($(typeof(io))) returned $opb: an OutputSpec in output_specs has a " *
            "negative per-item dimension. An output with a variable axis cannot be sized ahead of " *
            "time; leave it out of output_specs and it comes back inline",
    )
    per_item = ipb + opb
    span = cld(per_item, slot_bytes(pool))
    span <= n_slots(pool) || error(
        "one item needs $per_item bytes ($ipb input + $opb output), i.e. $span slots of " *
            "$(slot_bytes(pool)) bytes, but the pool has only $(n_slots(pool)) slots " *
            "($(sizeof(pool)) bytes total); increase pool_bytes on kserve_init",
    )
    # KServeModel refuses a size below 1 when it is built; this covers other model types. With it,
    # the chunk is at least 1: span * slot_bytes >= per_item > 0.
    mbs = max_batch_size(m)
    mbs >= 1 || throw(ArgumentError("max_batch_size($(typeof(m))) returned $mbs; it must be at least 1"))
    chunk = min(mbs, (span * slot_bytes(pool)) ÷ per_item)
    return chunk, span
end

_chunk_size(io::AbstractInferenceIO, m::AbstractInferenceModel, pool::InferenceBufferPool) =
    first(_chunk_geometry(io, m, pool))

# ---- Drivers ----

"""
    infer_async(model, io::AbstractInferenceIO)

Run inference over every item in `io`, staging inputs through the shared-memory [`BufferPool`]
and dispatching chunks concurrently (bounded by the pool's slot count). Each chunk acquires a
disjoint slot, so this is safe to call from multiple threads against one model. Results are
delivered through `io`'s `infer_decode_chunk!`. Use [`infer_sync`](@ref) for serial dispatch.

When a chunk fails, no further chunk starts, and a chunk already in flight finishes the attempt it
has sent but makes no further retry (a result that still arrives is delivered). The call waits for
those chunks to finish, so once it returns or throws, nothing writes into `io` or holds a pool slot
for it. The error is the first failing chunk's exception as the chunk raised it, not wrapped in a
`TaskFailedException`, so it is caught by type just as from `infer_sync`, and an unrecoverable
shared-memory failure falls back to inline transport the same way. When several chunks fail, an
unrecoverable shared-memory failure takes precedence over the others, so that fallback happens even
if another chunk failed first. The chunk's backtrace stays on the exception stack
(`current_exceptions()`).
"""
function infer_async(m::AbstractInferenceModel, io::AbstractInferenceIO)
    return _infer_pool_driven(m, io; force_serial = false)
end

# Request-level KV params carrying this model's per-request budget as a RELATIVE remaining timeout.
# The worker (reached through the gateway, which forwards the request body verbatim, so the param
# rides along unchanged) converts it to a local absolute deadline and drops the request at admission
# once it passes, instead of spending GPU on work the client has already abandoned. The gRPC client
# also sends `grpc-timeout`, but the gateway replaces that with its own per-call deadline, so the
# in-body KV param is what survives the hop. Models without a deadline send an empty map (no change).
_request_deadline_params(::AbstractInferenceModel) = deadline_params(inference, 0)
_request_deadline_params(m::KServeModel) =
    deadline_params(inference, deadline(m) > 0 ? round(Int64, deadline(m) * 1.0e9) : 0)
# Per-attempt variant: carry an explicit remaining budget (seconds) rather than the model's full
# deadline, so each retry admits against what is actually left of the original budget.
_request_deadline_params(budget_s::Real) =
    deadline_params(inference, budget_s > 0 ? round(Int64, budget_s * 1.0e9) : 0)

# A stale shared-memory registration: the server no longer knows the region our request references,
# because it restarted (or otherwise dropped its registry) since we registered. The worker maps it to
# FAILED_PRECONDITION; the message substring is a fallback for the case where the status is rewritten
# in transit (e.g. an older gateway folding it into UNAVAILABLE) but the worker's message survives.
function _is_stale_registration(e)
    e isa gRPCClient.gRPCServiceCallException || return false
    e.grpc_status == gRPCClient.GRPC_FAILED_PRECONDITION && return true
    return occursin("unregistered shared memory region", e.message)
end

# Thrown when shared memory could not be recovered in-band (re-registration failed, or the retried
# request was still stale). Signals the drive loop to latch the endpoint to inline transport.
struct ShmUnrecoverableError <: Exception
    model::Any
    cause::Any
end

# Name the endpoint and the cause. The default display printed the whole model, gRPC handle and
# connection pool included.
function Base.showerror(io::IO, e::ShmUnrecoverableError)
    m = e.model
    target = m isa KServeModel ? "model \"$(model_name(m))\" at $(m.host):$(m.port)" : string(typeof(m))
    print(io, "ShmUnrecoverableError: shared memory could not be recovered for ", target, ": ")
    return showerror(io, e.cause)
end

# A shed the client should back off and retry: RESOURCE_EXHAUSTED (a worker or the gateway at its
# concurrency cap) always, and UNAVAILABLE (no replica reachable) when the policy allows. Deadline,
# NOT_FOUND, INVALID_ARGUMENT, and INTERNAL are not retried, so a doomed request fails fast. A stale
# shared-memory registration is never retried here: it is handled by _send_with_shm_recovery, which
# re-registers and retries once, so the backoff loop must surface it immediately instead of burning
# the budget re-sending a request the server will keep rejecting.
function _retryable_shed(e, p::RetryPolicy)
    e isa gRPCClient.gRPCServiceCallException || return false
    _is_stale_registration(e) && return false
    e.grpc_status == gRPCClient.GRPC_RESOURCE_EXHAUSTED && return true
    return p.retry_unavailable && e.grpc_status == gRPCClient.GRPC_UNAVAILABLE
end

# For a request that belongs to no larger call: it is never cancelled.
_never() = false

# Wrap a chunk send with one-shot, coalesced shared-memory recovery. On the SHM path, snapshot the
# endpoint's registration generation, send, and on a stale-registration error re-register once (only
# the first concurrent failer actually re-registers; see recover_registration!) and retry the send.
# If re-registration throws, or the retry is still stale, raise ShmUnrecoverableError so the drive
# loop latches this endpoint to inline. Inline pools (and non-KServe models) send with no recovery.
# `cancelled()` reports that the call the chunk belongs to has already failed elsewhere; then the
# stale error is surfaced as is, with no re-registration and no second send.
function _send_with_shm_recovery(send, m, pool::InferenceBufferPool; cancelled = _never)
    (is_shm_backed(pool) && m isa KServeModel) || return send()
    observed = registration_gen(pool, (m.host, m.port))
    try
        return send()
    catch e
        (_is_stale_registration(e) && !cancelled()) || rethrow()
        try
            recover_registration!(pool, m, observed)
        catch reg_err
            throw(ShmUnrecoverableError(m, reg_err))
        end
        try
            return send()
        catch e2
            _is_stale_registration(e2) && throw(ShmUnrecoverableError(m, e2))
            rethrow()
        end
    end
end

# Run `attempt(budget_s)` (one ModelInfer round trip given the remaining budget in seconds),
# retrying with exponential backoff while the server sheds the request under overload. Each attempt
# is handed the remaining slice of the model's original deadline, which `attempt` forwards both as
# the gRPC call timeout and as the in-body admission budget, so the retries never push total wall
# time past that deadline. A shed with no budget left to sleep-and-retry surfaces the last error.
# Full jitter (wait uniform in [0, backoff]) keeps concurrent chunk retries from synchronizing.
#
# `cancelled()` reports that the call the request belongs to has already failed elsewhere (another
# chunk of the same `infer_async` call). From then on no further attempt goes out: the attempt on
# the wire finishes (a shared-memory attempt cannot be abandoned safely), and its shed, or a backoff
# being slept when the call fails, surfaces the last error at once.
function _infer_with_retry(attempt, m::AbstractInferenceModel; cancelled = _never)
    p = retry_policy(m)
    overall = overall_deadline(m)
    (!p.enabled || overall <= 0) && return attempt(overall)
    start = time()
    backoff = p.initial_backoff
    while true
        try
            return attempt(overall - (time() - start))
        catch e
            (_retryable_shed(e, p) && !cancelled()) || rethrow()
            wait_s = min(backoff, p.max_backoff)
            p.jitter && (wait_s *= rand())
            # Give up if a retry could not run after the sleep, so we never sleep away the budget
            # only to send a request the server would immediately deadline-shed.
            (time() - start) + wait_s >= overall - p.min_budget && rethrow()
            _backoff_sleep(wait_s, cancelled) && rethrow()
            backoff *= p.factor
        end
    end
    return
end

# How often a retry backoff checks whether its call has failed elsewhere, in seconds.
const _CANCEL_POLL_SECONDS = 0.05

# Sleep `s` seconds of retry backoff in slices of at most _CANCEL_POLL_SECONDS. Returns true, early,
# once `cancelled()` holds, and false after the full `s`.
function _backoff_sleep(s, cancelled)
    until = time() + s
    while true
        cancelled() && return true
        left = until - time()
        left <= 0 && return false
        sleep(min(left, _CANCEL_POLL_SECONDS))
    end
    return
end

"""
    infer_sync(model, io::AbstractInferenceIO)
    infer_sync(model, network_inputs) -> ModelInferResponse

Synchronous inference. The [`AbstractInferenceIO`](@ref) form drives `io` one chunk at a time
(no concurrency) and stops at the first chunk that fails, throwing its exception. The second
form is a one-shot call: `network_inputs` is a vector of wire
tensors built with [`InferInput`](@ref), sent inline in a single `ModelInferRequest`, and the
decoded `ModelInferResponse` is returned for reading with [`InferOutput`](@ref). The one-shot
form encodes tensor data element by element as typed contents, so it emits a one-time
performance warning (once per session) steering large or repeated workloads to the
`AbstractInferenceIO` drivers, which stage through the buffer pool and ship raw bytes on every
transport.
"""
function infer_sync(m::AbstractInferenceModel, io::AbstractInferenceIO)
    return _infer_pool_driven(m, io; force_serial = true)
end

function _infer_pool_driven(
        m::AbstractInferenceModel,
        io::AbstractInferenceIO;
        force_serial::Bool,
    )
    # Transport is decided up front by get_or_create_pool! (an IsSameIPCNamespace probe for a
    # KServeModel). A stale shared-memory registration self-heals in _send_with_shm_recovery (one
    # re-register + retry, keeping SHM). Only if that fails does the batch surface a
    # ShmUnrecoverableError here, and we latch the endpoint to inline and re-run inline as a last
    # resort; a background poller restores SHM later if the server recovers.
    pool = get_or_create_pool!(m)
    try
        _drive_pool_inference(m, io, pool; force_serial = force_serial)
    catch e
        (e isa ShmUnrecoverableError && is_shm_backed(pool) && m isa KServeModel) || rethrow()
        inline = latch_inline!(m)
        # Re-run the whole batch inline. Chunks decoded before the SHM failure are decoded again;
        # this is benign for the typical preallocated-output IO, and a stale registration normally
        # fails the batch wholesale anyway.
        _drive_pool_inference(m, io, inline; force_serial = force_serial)
    end
    return nothing
end

# Read the int64 `shared_memory_byte_size` parameter the server stamps on an SHM-backed output
# tensor (the actual bytes written, codec.jl), or nothing if absent/mismatched.
function _shm_byte_size(params)
    haskey(params, "shared_memory_byte_size") || return nothing
    p = params["shared_memory_byte_size"].parameter_choice
    return (p !== nothing && p.name === :int64_param) ? Int(p[]) : nothing
end
_is_shm_output(params) = haskey(params, "shared_memory_region")

# Byte size of an output tensor derived from its wire shape and dtype (fallback when the server
# did not stamp shared_memory_byte_size).
function _output_bytes_from_shape(o)
    dt = KSERVE_OUTPUT_DTYPE_TABLE[o.datatype]
    return sizeof(dt) * (isempty(o.shape) ? 1 : prod(o.shape))
end

# Build the requested-output list for one chunk. For an SHM pool, carve an output subslot per
# declared output (continuing the slot cursor past the inputs) and reference it by
# region/offset/byte_size; for an inline pool, request the outputs by name only so the server
# returns them inline. Returns the requested tensors and a name -> output-subslot map (empty for
# inline pools). The slot's cursor is driver-local, so this runs outside `fill_lock`.
function _build_requested_outputs(specs, slot, r, pool)
    requested = var"ModelInferRequest.InferRequestedOutputTensor"[]
    subslots = Dict{String, PoolSlot}()
    isempty(specs) && return requested, subslots

    n = length(r)
    shm = is_shm_backed(pool)
    for s in specs
        if shm
            nbytes = sizeof(s.dtype) * prod(s.per_item_dims) * n
            sub = subslot(slot, nbytes)
            subslots[s.name] = sub
            push!(
                requested,
                var"ModelInferRequest.InferRequestedOutputTensor"(
                    name = s.name,
                    parameters = Dict(
                        "shared_memory_region" =>
                            InferParameter(parameter_choice = OneOf(:string_param, pool_name(pool))),
                        "shared_memory_offset" => InferParameter(
                            parameter_choice = OneOf(:int64_param, Int64(sub.offset)),
                        ),
                        "shared_memory_byte_size" => InferParameter(
                            parameter_choice = OneOf(:int64_param, Int64(nbytes)),
                        ),
                    ),
                ),
            )
        else
            push!(requested, var"ModelInferRequest.InferRequestedOutputTensor"(name = s.name))
        end
    end
    return requested, subslots
end

# Normalize a response so its `raw_output_contents` is full-length and aligned to `outputs`,
# making the SHM transport invisible to `InferOutput` / `infer_decode_chunk!`. For each output
# the server wrote to shared memory (carries a region parameter), copy the bytes it actually
# wrote out of our pool slot (before the slot is released) and strip the SHM parameters.
#
# Inline outputs come from the response's `raw_output_contents`, which arrives in one of two forms.
# The KServe proto requires one entry per output in `outputs` order, with an empty placeholder for
# each shared-memory output (positional; what Triton sends and what Triton clients index). Older
# ReactantServer workers appended an entry for inline outputs only (compressed). The two are told
# apart by length: positional has exactly one entry per output, compressed one per inline output,
# and the lengths coincide only when no output went to shared memory, where the forms are the same.
function _rehydrate_response(response, subslots)
    isempty(subslots) && return response

    outputs = response.outputs
    raw_in = response.raw_output_contents
    positional = length(raw_in) == length(outputs)
    if !positional
        n_inline = count(o -> !_is_shm_output(o.parameters), outputs)
        length(raw_in) == n_inline || error(
            "response has $(length(raw_in)) raw_output_contents entries for $(length(outputs)) " *
                "outputs ($n_inline inline); expected one per output or one per inline output",
        )
    end
    new_outputs = var"ModelInferResponse.InferOutputTensor"[]
    raw = Vector{UInt8}[]
    inline_idx = 1
    for (i, o) in enumerate(outputs)
        if _is_shm_output(o.parameters)
            sub = subslots[o.name]
            nbytes = something(_shm_byte_size(o.parameters), _output_bytes_from_shape(o))
            push!(raw, copy(pool_view(sub, UInt8, nbytes)))
            push!(
                new_outputs,
                var"ModelInferResponse.InferOutputTensor"(
                    name = o.name,
                    datatype = o.datatype,
                    shape = o.shape,
                ),
            )
        else
            push!(raw, raw_in[positional ? i : inline_idx])
            inline_idx += 1
            push!(new_outputs, o)
        end
    end

    return ModelInferResponse(
        model_name = response.model_name,
        model_version = response.model_version,
        id = response.id,
        parameters = response.parameters,
        outputs = new_outputs,
        raw_output_contents = raw,
    )
end

# Add the slot control parameters for one attempt to `params`, arming `slot` with a fresh token
# (CORE-3). A retry re-arms, which closes the previous attempt so its server can no longer write.
function _add_slot_control!(params, pool::InferenceBufferPool, slot::PoolSlot)
    token = slot_arm!(slot)
    params[SLOT_CONTROL_REGION_PARAM] = InferParameter(parameter_choice = OneOf(:string_param, pool_name(pool)))
    params[SLOT_CONTROL_OFFSET_PARAM] =
        InferParameter(parameter_choice = OneOf(:int64_param, Int64(slot_control_offset(slot))))
    params[SLOT_CONTROL_TOKEN_PARAM] = InferParameter(parameter_choice = OneOf(:int64_param, Int64(token)))
    return params
end

# Run one chunk end-to-end on a freshly acquired slot: fill the staging buffer, send the
# request, handle the response, and always return the slot to the pool. `infer_encode_chunk!` is
# serialized by `fill_lock` because concrete IOs commonly read from shared source state.
#
# On a SHM pool every attempt carries the slot control parameters under a fresh token. Shared-memory
# outputs are read only once `slot_seal!` confirms no server can still be writing them, and the
# `finally` releases the slot whatever happened: release closes the attempt, so a server that has
# not begun writing never will, and a slot a server is writing at that moment is quarantined rather
# than reused (see `release_slot!`). Nothing here waits on a server.
#
# `cancelled()` reports that the call the chunk belongs to has already failed elsewhere; from then
# on the chunk sends no retry (see `_infer_with_retry` and `_send_with_shm_recovery`). A response
# that still arrives is decoded as usual.
function _run_chunk(m, io, pool, fill_lock, r, slot; cancelled = _never)
    return try
        reset_slot!(slot)
        inputs = _encoded_inputs(
            lock(fill_lock) do
                infer_encode_chunk!(io, r, slot)
            end
        )
        # Output subslots are carved from the same slot, after the inputs, so a request can stage
        # both through one registered region. Outputs are read back before the slot is released.
        requested, out_subslots = _build_requested_outputs(output_specs(io), slot, r, pool)
        # Materialize inputs once; only the per-attempt deadline (client timeout + in-body budget)
        # changes across retries, and the staged bytes stay valid until the slot is released below.
        # On inline pools the raw blobs alias the pool's backing (unsafe_wrap does not root it), so
        # GC.@preserve keeps the backing alive from encoding through response decode; the slot is
        # released only in the `finally` after this send returns.
        materialized, raws = _materialize_inputs(inputs, m, pool)
        controlled = _slot_controlled(pool)
        response = GC.@preserve pool begin
            _send_with_shm_recovery(m, pool; cancelled = cancelled) do
                _infer_with_retry(m; cancelled = cancelled) do budget_s
                    params = _request_deadline_params(budget_s)
                    controlled && _add_slot_control!(params, pool, slot)
                    GRPCInferenceService.ModelInfer(
                        m.chan,
                        ModelInferRequest(
                            model_name = model_name(m),
                            inputs = materialized,
                            outputs = requested,
                            parameters = params,
                            raw_input_contents = raws,
                        );
                        deadline = budget_s,
                    )
                end
            end
        end
        if controlled && !isempty(out_subslots)
            slot_seal!(slot) || error(
                "the shared-memory outputs of this chunk were claimed by another writer after the " *
                    "response; they cannot be trusted",
            )
        end
        infer_decode_chunk!(io, r, _rehydrate_response(response, out_subslots))
    finally
        release_slot!(slot)
    end
end

# How many chunks bound for a shared-memory pool were sent inline because no slot was free at once.
const _INLINE_FALLBACKS = Threads.Atomic{Int}(0)

"""
    inline_fallback_count() -> Int

How many chunks this process sent inline instead of through its shared-memory pool because no pool
slot was free when the chunk was staged (all in flight, or quarantined or retired behind a server
still writing them). A steadily growing count means the pool has too few slots for the concurrency.
"""
inline_fallback_count() = _INLINE_FALLBACKS[]

# A private one-slot inline pool large enough for a chunk sized for `pool`'s `span` slots.
_ephemeral_inline_pool(pool::InferenceBufferPool, span::Integer) = InferenceBufferPool(
    cld(span * slot_bytes(pool), ReactantServerCore._page_size()) * ReactantServerCore._page_size();
    n_slots = 1, use_shm = false,
)

# Whether chunk `r` fits one gRPC message each way when sent inline: its inputs in the request and
# its declared outputs in the response, with room for the protobuf framing.
function _fits_inline(m::AbstractInferenceModel, io::AbstractInferenceIO, r)
    m isa KServeModel || return true
    margin = 1 << 20
    return length(r) * item_input_bytes(io) + margin <= m.max_send_message_length &&
        length(r) * item_output_bytes(io) + margin <= m.max_receive_message_length
end

# The pool and slot a chunk is staged in. A slot of `pool` when one is free right now; otherwise the
# chunk goes inline at once through a private buffer, so no chunk ever waits on a slot another
# chunk (or an enclosing inference, CLI-10) holds. Only a chunk too large for one gRPC message waits
# for a slot of a SHM pool, and then no longer than the model's deadline.
function _chunk_slot(m::AbstractInferenceModel, io::AbstractInferenceIO, pool::InferenceBufferPool, r, span::Integer)
    slot = try_acquire_slot!(pool.pool, span)
    slot === nothing || return pool, slot
    if !is_shm_backed(pool) || _fits_inline(m, io, r)
        if is_shm_backed(pool)
            n = Threads.atomic_add!(_INLINE_FALLBACKS, 1) + 1
            @debug "no shared-memory slot free; sending this chunk inline" fallbacks = n
            @info "no shared-memory slot was free for a chunk, so it was sent inline; " *
                "`ReactantServerClient.inline_fallback_count()` counts these" maxlog = 1
        end
        eph = _ephemeral_inline_pool(pool, span)
        return eph, acquire_slot!(eph.pool, 1)
    end
    budget = overall_deadline(m)
    deadline_ns = budget > 0 ? Int64(time_ns()) + round(Int64, budget * 1.0e9) : Int64(0)
    slot = try
        acquire_slot!(pool.pool, span; deadline_ns = deadline_ns)
    catch e
        e isa PoolAcquireTimeout || rethrow()
        error(
            "no shared-memory slot became free within the model's deadline for a chunk of " *
                "$(length(r)) items, which is too large to send inline in one gRPC message",
        )
    end
    return pool, slot
end

function _drive_pool_inference(
        m::AbstractInferenceModel,
        io::AbstractInferenceIO,
        pool::InferenceBufferPool;
        force_serial::Bool,
    )
    chunk, span = _chunk_geometry(io, m, pool)
    fill_lock = ReentrantLock()

    if force_serial
        for r in BatchIterator(length(io), chunk)
            chunk_pool, slot = _chunk_slot(m, io, pool, r, span)
            _run_chunk(m, io, chunk_pool, fill_lock, r, slot)
        end
        return nothing
    end

    # Each chunk acquires a disjoint slot run from the pool's shared allocator, so ranges never
    # overlap across tasks or across concurrent top-level inference calls. The in-flight window is
    # how many spans the pool holds, so one call alone never runs out of slots; a chunk that still
    # finds none free (concurrent calls, a quarantined slot) goes inline rather than waiting.
    window = max(1, n_slots(pool) ÷ span)
    tasks = Task[]
    failure = _FirstChunkFailure()

    try
        i = 0
        for r in BatchIterator(length(io), chunk)
            i += 1
            length(tasks) >= window && wait(tasks[i - window])
            _failed(failure) && break   # after the first failure, no further chunk starts
            push!(tasks, Threads.@spawn _run_async_chunk(failure, m, io, pool, fill_lock, r, span))
        end
    finally
        # Whatever ends the loop, every chunk it started finishes before the call returns: one still
        # running would go on writing into the caller's io and holding a pool slot (CLI-3). Chunk
        # tasks record their failure instead of throwing, so these waits do not throw.
        foreach(wait, tasks)
    end
    _throw_first_failure(failure)
    return nothing
end

# The failure one `infer_async` call throws, as (exception, backtrace): see `_record_failure!`.
mutable struct _FirstChunkFailure
    @atomic first::Union{Nothing, Tuple{Any, Any}}
end
_FirstChunkFailure() = _FirstChunkFailure(nothing)
_failed(f::_FirstChunkFailure) = (@atomic f.first) !== nothing

# Record a chunk's failure. The first one is kept, except that the first ShmUnrecoverableError
# displaces any other failure: it is what makes `_infer_pool_driven` latch the endpoint to inline,
# and a failure that merely came earlier must not hide it.
function _record_failure!(f::_FirstChunkFailure, ex, bt)
    seen = @atomic f.first
    while seen === nothing || (ex isa ShmUnrecoverableError && !(first(seen) isa ShmUnrecoverableError))
        seen, done = @atomicreplace f.first seen => (ex, bt)
        done && break
    end
    return nothing
end

# One chunk of an `infer_async` call, run on its own task. A failure is recorded in `failure`
# rather than thrown, so the driver can wait on every chunk without the first failed wait cutting
# the others short. A chunk that had not started when another failed does not start, and one in
# flight finishes the attempt it has sent but makes no further retry.
function _run_async_chunk(failure::_FirstChunkFailure, m, io, pool, fill_lock, r, span)
    _failed(failure) && return nothing
    try
        chunk_pool, slot = _chunk_slot(m, io, pool, r, span)
        _run_chunk(m, io, chunk_pool, fill_lock, r, slot; cancelled = () -> _failed(failure))
    catch ex
        _record_failure!(failure, ex, catch_backtrace())
    end
    return nothing
end

# Throw the recorded chunk failure, if any, as the exception the chunk raised, not wrapped in a
# TaskFailedException: callers that catch by type, and the inline-fallback latch in
# `_infer_pool_driven` (which looks for `ShmUnrecoverableError`), see what `infer_sync` would have
# thrown. Raising it inside a `catch` keeps the chunk's own backtrace on the exception stack below it
# (`current_exceptions()`), which is where Julia prints it, as "caused by".
function _throw_first_failure(failure::_FirstChunkFailure)
    recorded = @atomic failure.first
    recorded === nothing && return nothing
    ex, bt = recorded
    return try
        throw(CapturedException(ex, bt))
    catch
        throw(ex)
    end
end

function infer_sync(m::AbstractInferenceModel, network_inputs)
    @warn "infer_sync(model, inputs) sends tensor data as typed contents, encoded element by " *
        "element, which is slow for large or repeated requests; implement an AbstractInferenceIO " *
        "and use infer_async or infer_sync(model, io) to stage through the buffer pool and send " *
        "raw bytes" maxlog = 1
    return _infer_with_retry(m) do budget_s
        GRPCInferenceService.ModelInfer(
            m.chan,
            ModelInferRequest(
                model_name = model_name(m),
                inputs = network_inputs,
                parameters = _request_deadline_params(budget_s),
            );
            deadline = budget_s,
        )
    end
end


function InferInput(name::String, shape, contents::AbstractVector{UInt8})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT8",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt16})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT16",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt32})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT32",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt64})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT64",
        shape = reverse(shape),
        contents = InferTensorContents(uint64_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int8})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT8",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int16})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT16",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int32})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT32",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int64})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT64",
        shape = reverse(shape),
        contents = InferTensorContents(int64_contents = contents),
    )
end


function InferInput(name::String, shape, contents::AbstractVector{Float32})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "FP32",
        shape = reverse(shape),
        contents = InferTensorContents(fp32_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Float64})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "FP64",
        shape = reverse(shape),
        contents = InferTensorContents(fp64_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Bool})
    return var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "BOOL",
        shape = reverse(shape),
        contents = InferTensorContents(bool_contents = contents),
    )
end

"""
    InferInput(name, array) -> ModelInferRequest.InferInputTensor

Build a wire input tensor named `name` from a Julia `array`, shipping the bytes inline as
typed contents, encoded element by element (the one-shot path warns once about this cost; see
[`infer_sync`](@ref)). You pass the array in its natural Julia column-major shape
`(W, H, …, N)`; the client reverses it to the
network's row-major `(N, …, H, W)` internally (the bytes are unchanged). Pass a vector of these to
the one-shot [`infer_sync`](@ref)`(model, inputs)`. Variants taking an explicit Julia column-major
`shape` and a typed `contents` vector are also provided.
"""
function InferInput(name::String, inp::AbstractArray{T}) where {T <: TritonType}
    return InferInput(name, collect(size(inp)), vec(inp))
end

"""
    InferOutput(name, response, dtype) -> Array
    InferOutput(name, response) -> Array

Extract the output tensor named `name` from a `ModelInferResponse` as a Julia array. The wire
row-major shape `(N, …, H, W)` is reshaped (no copy) to Julia column-major `(W, …, N)`. Pass the
element type as `dtype` for a type-stable result; the two-argument form reads the dtype from the
response metadata.
"""
function InferOutput(
        name::String,
        response::ModelInferResponse,
        dtype::Type{T},
    ) where {T <: TritonType}
    # type stable version
    for (i, output) in enumerate(response.outputs)
        output.name == name || continue
        content = response.raw_output_contents[i]
        content = reinterpret(dtype, content)
        # Network sends row-major (N, ..., H, W); reshape directly to col-major (W, H, ..., N).
        # Dimensions are reversed relative to the network shape — no copy required.
        content = reshape(content, reverse(output.shape)...)
        return content
    end

    error("no such output in KServe response: $(name)")
end


function InferOutput(name::String, response::ModelInferResponse)
    # dynamic dispatch version
    for (i, output) in enumerate(response.outputs)
        output.name == name || continue
        content = response.raw_output_contents[i]

        haskey(KSERVE_OUTPUT_DTYPE_TABLE, output.datatype) ||
            error("output '$(name)' has unsupported KServe datatype '$(output.datatype)'")
        dtype = KSERVE_OUTPUT_DTYPE_TABLE[output.datatype]

        content = reinterpret(dtype, content)
        # Network sends row-major (N, ..., H, W); reshape directly to col-major (W, H, ..., N).
        # Dimensions are reversed relative to the network shape — no copy required.
        content = reshape(content, reverse(output.shape)...)
        return content
    end

    error("no such output in KServe response: $(name)")
end
