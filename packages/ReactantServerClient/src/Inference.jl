# ============================================================================
# AbstractInferenceIO contract
#
#   infer_encode_chunk!(io, r::UnitRange, slot::PoolSlot)::AbstractVector{PoolInferInput}
#   item_input_bytes(io)::Int    # total bytes per item summed across all inputs
#   infer_decode_chunk!(io, r::UnitRange, response)::Nothing
#   length(io)::Int
# ============================================================================

# A deferred descriptor produced by infer_encode_chunk!. The SHM-vs-inline choice is
# resolved by `materialize_input` at gRPC-request time, based on whether the
# pool that owns the slot is SHM-backed.
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
) where {T<:TritonType}
    PoolInferInput(name, sub, Int.(shape), T)
end

# Convenience: derive dtype and the Julia column-major shape from a pool view.
function InferInput(
    name::String,
    sub::PoolSlot,
    view::AbstractArray{T},
) where {T<:TritonType}
    PoolInferInput(name, sub, collect(Int, size(view)), T)
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

function materialize_input(
    d::PoolInferInput,
    model::AbstractInferenceModel,
    pool::InferenceBufferPool,
)
    n_bytes = sizeof(d.dtype) * prod(d.shape)
    if is_shm_backed(pool)
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
        )
    else
        n_elems = prod(d.shape)
        view = pool_view(d.subslot, d.dtype, n_elems)
        InferInput(d.name, d.shape, view)
    end
end

function _materialize_inputs(inputs, model, pool::InferenceBufferPool)
    [materialize_input(d, model, pool) for d in inputs]
end

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
    per_item = ipb + opb
    span = cld(per_item, slot_bytes(pool))
    span <= n_slots(pool) || error(
        "one item needs $per_item bytes ($ipb input + $opb output), i.e. $span slots of " *
        "$(slot_bytes(pool)) bytes, but the pool has only $(n_slots(pool)) slots " *
        "($(sizeof(pool)) bytes total); increase pool_bytes on kserve_init",
    )
    chunk = min(max_batch_size(m), (span * slot_bytes(pool)) ÷ per_item)
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
"""
function infer_async(m::AbstractInferenceModel, io::AbstractInferenceIO)
    _infer_pool_driven(m, io; force_serial = false)
end

# Request-level KV params carrying this model's per-request budget as a RELATIVE remaining timeout.
# The worker (reached through the gateway, which forwards the request body verbatim, so the param
# rides along unchanged) converts it to a local absolute deadline and drops the request at admission
# once it passes, instead of spending GPU on work the client has already abandoned. The gRPC client
# also sends `grpc-timeout`, but the gateway replaces that with its own per-call deadline, so the
# in-body KV param is what survives the hop. Models without a deadline send an empty map (no change).
_request_deadline_params(::AbstractInferenceModel) = deadline_params(0)
_request_deadline_params(m::KServeModel) =
    deadline_params(deadline(m) > 0 ? round(Int64, deadline(m) * 1e9) : 0)

"""
    infer_sync(model, io::AbstractInferenceIO)
    infer_sync(model, network_inputs) -> ModelInferResponse

Synchronous inference. The [`AbstractInferenceIO`](@ref) form drives `io` one chunk at a time
(no concurrency). The second form is a one-shot call: `network_inputs` is a vector of wire
tensors built with [`InferInput`](@ref), sent inline in a single `ModelInferRequest`, and the
decoded `ModelInferResponse` is returned for reading with [`InferOutput`](@ref).
"""
function infer_sync(m::AbstractInferenceModel, io::AbstractInferenceIO)
    _infer_pool_driven(m, io; force_serial = true)
end

function _infer_pool_driven(
    m::AbstractInferenceModel,
    io::AbstractInferenceIO;
    force_serial::Bool,
)
    pool = get_or_create_pool!(m)
    try
        _drive_pool_inference(m, io, pool; force_serial = force_serial)
    catch ex
        # Triton's SHM register only stores metadata, so the real shm_open can
        # still fail at inference time when /dev/shm is not shared. Recover by
        # routing this URL to the inline pool and replaying the call.
        #
        # _is_shm_not_found_error must stay narrow: any other gRPC failure
        # (DEADLINE_EXCEEDED, INTERNAL from a model-execution error, etc.)
        # has to surface to the caller unmodified. A false positive here would
        # silently retry a non-SHM error through the inline pool and mask the
        # real cause.
        if is_shm_backed(pool) && _is_shm_not_found_error(ex)
            @info "Triton at $(m.host):$(m.port) can't map our SHM region; migrating to inline transport" exception = ex
            pool = migrate_to_inline!(m)
            _drive_pool_inference(m, io, pool; force_serial = force_serial)
        else
            rethrow()
        end
    end
    nothing
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
    subslots = Dict{String,PoolSlot}()
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
# wrote out of our pool slot (before the slot is released) and strip the SHM parameters; inline
# outputs are taken from the original `raw_output_contents` in order, because the server's encoder
# only appends a raw entry per inline output (codec.jl), so positions are compressed there.
function _rehydrate_response(response, subslots)
    isempty(subslots) && return response

    new_outputs = var"ModelInferResponse.InferOutputTensor"[]
    raw = Vector{UInt8}[]
    inline_idx = 1
    for o in response.outputs
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
            push!(raw, response.raw_output_contents[inline_idx])
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

# Run one chunk end-to-end on a freshly acquired slot: fill the staging buffer, send the
# request, handle the response, and always return the slot to the pool. `infer_encode_chunk!` is
# serialized by `fill_lock` because concrete IOs commonly read from shared source state.
function _run_chunk(m, io, pool, client, fill_lock, r, slot)
    try
        reset_slot!(slot)
        inputs = lock(fill_lock) do
            infer_encode_chunk!(io, r, slot)
        end
        # Output subslots are carved from the same slot, after the inputs, so a request can stage
        # both through one registered region. Outputs are read back before the slot is released.
        requested, out_subslots = _build_requested_outputs(output_specs(io), slot, r, pool)
        response = grpc_sync_request(
            client,
            ModelInferRequest(
                model_name = model_name(m),
                inputs = _materialize_inputs(inputs, m, pool),
                outputs = requested,
                parameters = _request_deadline_params(m),
            ),
        )
        infer_decode_chunk!(io, r, _rehydrate_response(response, out_subslots))
    finally
        release_slot!(slot)
    end
end

function _drive_pool_inference(
    m::AbstractInferenceModel,
    io::AbstractInferenceIO,
    pool::InferenceBufferPool;
    force_serial::Bool,
)
    chunk, span = _chunk_geometry(io, m, pool)
    client = grpc_infer_client(m)
    fill_lock = ReentrantLock()

    if force_serial
        for r in BatchIterator(length(io), chunk)
            slot = acquire_slot!(pool, span)
            _run_chunk(m, io, pool, client, fill_lock, r, slot)
        end
        return nothing
    end

    # Each chunk acquires a disjoint slot run from the pool's shared allocator, so concurrency
    # is bounded by the slot count and ranges never overlap across tasks or across concurrent
    # top-level inference calls. The in-flight window mirrors how many spans the pool can serve
    # at once so we do not spawn an unbounded number of tasks blocked on acquisition.
    window = max(1, n_slots(pool) ÷ span)
    tasks = Task[]
    err = Ref{Any}(nothing)

    i = 0
    for r in BatchIterator(length(io), chunk)
        i += 1
        if length(tasks) >= window
            wait(tasks[i - window])
        end
        err[] === nothing || break
        local_r = r
        t = Threads.@spawn begin
            slot = acquire_slot!(pool, span)
            try
                _run_chunk(m, io, pool, client, fill_lock, local_r, slot)
            catch ex
                err[] = ex
                rethrow()
            end
        end
        push!(tasks, t)
    end

    foreach(wait, tasks)
    err[] === nothing || throw(err[])
    nothing
end

function infer_sync(m::AbstractInferenceModel, network_inputs)
    client = grpc_infer_client(m)

    grpc_sync_request(
        client,
        ModelInferRequest(
            model_name = model_name(m),
            inputs = network_inputs,
            parameters = _request_deadline_params(m),
        ),
    )
end


function InferInput(name::String, shape, contents::AbstractVector{UInt8})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT8",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt16})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT16",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt32})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT32",
        shape = reverse(shape),
        contents = InferTensorContents(uint_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{UInt64})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "UINT64",
        shape = reverse(shape),
        contents = InferTensorContents(uint64_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int8})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT8",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int16})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT16",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int32})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT32",
        shape = reverse(shape),
        contents = InferTensorContents(int_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Int64})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "INT64",
        shape = reverse(shape),
        contents = InferTensorContents(int64_contents = contents),
    )
end


function InferInput(name::String, shape, contents::AbstractVector{Float32})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "FP32",
        shape = reverse(shape),
        contents = InferTensorContents(fp32_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Float64})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "FP64",
        shape = reverse(shape),
        contents = InferTensorContents(fp64_contents = contents),
    )
end

function InferInput(name::String, shape, contents::AbstractVector{Bool})
    var"ModelInferRequest.InferInputTensor"(
        name = name,
        datatype = "BOOL",
        shape = reverse(shape),
        contents = InferTensorContents(bool_contents = contents),
    )
end

"""
    InferInput(name, array) -> ModelInferRequest.InferInputTensor

Build a wire input tensor named `name` from a Julia `array`, shipping the bytes inline. You pass
the array in its natural Julia column-major shape `(W, H, …, N)`; the client reverses it to the
network's row-major `(N, …, H, W)` internally (the bytes are unchanged). Pass a vector of these to
the one-shot [`infer_sync`](@ref)`(model, inputs)`. Variants taking an explicit Julia column-major
`shape` and a typed `contents` vector are also provided.
"""
function InferInput(name::String, inp::AbstractArray{T}) where {T<:TritonType}
    InferInput(name, collect(size(inp)), vec(inp))
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
) where {T<:TritonType}
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
