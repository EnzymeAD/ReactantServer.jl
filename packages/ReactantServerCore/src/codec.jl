# Translation between the KServe V2 protobuf wire messages and the boundary types.
#
# Tensor data travels either inline (raw_input_contents / raw_output_contents) or through a
# registered shared-memory region referenced by the tensor's parameters
# (shared_memory_region / shared_memory_offset / shared_memory_byte_size). The shared-memory
# path is one copy from the mapped region into a host array for input, and one copy back into
# the region for output. The codec depends only on the generated protobuf, dtypes, and the
# shared-memory registry, never on HTTP.

const _PB_INF = inference

const _SHM_REGION = "shared_memory_region"
const _SHM_OFFSET = "shared_memory_offset"
const _SHM_BYTE_SIZE = "shared_memory_byte_size"

"""
    TIMEOUT_NS_PARAM

Key of the request-level KV parameter carrying the caller's REMAINING budget in nanoseconds
(relative, not an absolute timestamp). Like the shared-memory region parameters, this is an
extension to KServe V2 passed through `ModelInferRequest.parameters`. It is relative so each hop
converts it to its own local absolute deadline (`time_ns() + budget`), which makes it robust to
cross-process monotonic-clock differences and lets it ride unchanged through the gateway's raw-byte
request forwarding. See [`deadline_params`](@ref).
"""
const TIMEOUT_NS_PARAM = "reactant_timeout_ns"

"""
    deadline_params(budget_ns) -> Dict{String,InferParameter}

Build the request-level parameters map carrying a remaining-budget timeout of `budget_ns`
nanoseconds (see [`TIMEOUT_NS_PARAM`](@ref)). A non-positive `budget_ns` yields an empty map (no
deadline). Merge the result into a `ModelInferRequest`'s `parameters`.
"""
function deadline_params(budget_ns::Integer)
    budget_ns > 0 || return Dict{String,_PB_INF.InferParameter}()
    return Dict{String,_PB_INF.InferParameter}(TIMEOUT_NS_PARAM => _int_param(Int64(budget_ns)))
end

# Read the relative timeout budget (ns) from a request's parameters and convert it to an absolute
# local deadline (`time_ns() + budget`), or 0 when absent/non-positive. Saturates instead of
# overflowing on an absurd budget so a hostile value yields a far-future deadline, never a wrapped one.
function _decode_deadline_ns(params)
    budget = _param_int(params, TIMEOUT_NS_PARAM)
    (budget === nothing || budget <= 0) && return Int64(0)
    now = Int64(time_ns())
    return budget > typemax(Int64) - now ? typemax(Int64) : now + Int64(budget)
end

# Element count and byte size of an untrusted wire shape, validated BEFORE any allocation:
# every dim must be non-negative and the products must fit in Int. A hostile shape must be
# rejected here, not by attempting the allocation it describes.
function _checked_elems(::Type{T}, wire_shape::Vector{Int}) where {T}
    n = 1
    nbytes = 0
    try
        for d in wire_shape
            d >= 0 || error("tensor shape $wire_shape has a negative dimension")
            n = Base.checked_mul(n, d)
        end
        nbytes = Base.checked_mul(n, Int(sizeof(T)))
    catch e
        e isa OverflowError && error("tensor shape $wire_shape element/byte count overflows Int64")
        rethrow()
    end
    return n, nbytes
end

# Allocate a Julia col-major Array of the reverse of the wire (row-major) shape. Bytes
# laid out row-major on the wire are the same memory as a Julia column-major array of the
# reversed shape, so we allocate the destination directly in Julia shape and avoid the
# ReshapedArray type (which would force downstream re-specialization). `n` is the validated
# element count from `_checked_elems`.
function _alloc_julia(::Type{T}, wire_shape::Vector{Int}, n::Int) where {T}
    length(wire_shape) <= 1 && return Vector{T}(undef, n)
    return Array{T}(undef, reverse(wire_shape)...)
end

function _array_from_raw(dt::DType, wire_shape::Vector{Int}, raw::AbstractVector{UInt8})
    T = julia_type(dt)
    sizeof(T) == 0 || length(raw) % sizeof(T) == 0 ||
        error("raw content of $(length(raw)) bytes is not a multiple of $(sizeof(T)) for dtype $(dtype_token(dt))")
    n, nbytes = _checked_elems(T, wire_shape)
    nbytes == length(raw) ||
        error("raw content $(length(raw)) bytes does not match shape $wire_shape * sizeof($T)")
    arr = _alloc_julia(T, wire_shape, n)
    GC.@preserve raw arr unsafe_copyto!(convert(Ptr{UInt8}, pointer(arr)), pointer(raw), length(raw))
    return arr
end

function _array_from_contents(dt::DType, wire_shape::Vector{Int}, c)
    T = julia_type(dt)
    vals = if dt == BOOL
        c.bool_contents
    elseif dt in (I8, I16, I32)
        T.(c.int_contents)
    elseif dt == I64
        c.int64_contents
    elseif dt in (U8, U16, U32)
        T.(c.uint_contents)
    elseif dt == U64
        c.uint64_contents
    elseif dt == F32
        c.fp32_contents
    elseif dt == F64
        c.fp64_contents
    else
        error("inline typed contents not supported for dtype $(dtype_token(dt)); use raw or shared memory")
    end
    flat = vals isa Vector{T} ? vals : Vector{T}(vals)
    n, _ = _checked_elems(T, wire_shape)
    length(flat) == n || error("contents has $(length(flat)) elements but shape $wire_shape needs $n")
    length(wire_shape) <= 1 && return flat                      # 1-D: the flat vector is the tensor
    arr = _alloc_julia(T, wire_shape, n)
    copyto!(arr, flat)
    return arr
end

# Read a typed parameter out of a tensor's parameters map, or nothing if absent/mismatched.
function _param_string(params, key)
    haskey(params, key) || return nothing
    p = params[key].parameter_choice
    return (p !== nothing && p.name === :string_param) ? p[]::String : nothing
end
function _param_int(params, key)
    haskey(params, key) || return nothing
    p = params[key].parameter_choice
    return (p !== nothing && p.name === :int64_param) ? Int(p[]) : nothing
end

# Where a requested output's data should be written.
struct OutputTarget
    region::String
    offset::Int
    byte_size::Int
end

struct DecodedRequest
    request::InferRequest
    id::String
    output_targets::Dict{String,OutputTarget}   # by output name; only shm-backed outputs
end

"""
    decode_infer_request(msg, registry=nothing) -> DecodedRequest

Translate a decoded ModelInferRequest message into the boundary InferRequest. The transport
(gRPC) hands us the already-decoded protobuf message, so the codec never touches wire bytes.
Input tensor data is read from a registered shared-memory region (preferred when the tensor
declares one), otherwise from raw_input_contents, otherwise from the typed contents field.
"""
function decode_infer_request(msg::_PB_INF.ModelInferRequest,
                              registry::Union{SharedMemoryRegistry,Nothing}=nothing)
    n = length(msg.inputs)
    use_raw = !isempty(msg.raw_input_contents)
    if use_raw && length(msg.raw_input_contents) != n
        error("raw_input_contents has $(length(msg.raw_input_contents)) entries but request has $n inputs")
    end

    tensors = Vector{NamedTensor}(undef, n)
    for i in 1:n
        t = msg.inputs[i]
        dt = dtype_from_kserve(t.datatype)
        shape = Int[Int(s) for s in t.shape]
        region = _param_string(t.parameters, _SHM_REGION)
        data = if region !== nothing
            registry === nothing && error("input '$(t.name)' references shared memory but the server has no registry")
            offset = something(_param_int(t.parameters, _SHM_OFFSET), 0)
            bsize = _param_int(t.parameters, _SHM_BYTE_SIZE)
            bsize === nothing && error("input '$(t.name)' is missing $_SHM_BYTE_SIZE")
            _array_from_raw(dt, shape, shm_read(registry, region, offset, bsize))
        elseif use_raw
            _array_from_raw(dt, shape, msg.raw_input_contents[i])
        elseif t.contents !== nothing
            _array_from_contents(dt, shape, t.contents)
        else
            error("input '$(t.name)' carries neither shared memory, raw_input_contents, nor contents")
        end
        tensors[i] = NamedTensor(t.name, dt, Tuple(size(data)), data)
    end

    requested = String[o.name for o in msg.outputs]
    targets = Dict{String,OutputTarget}()
    for o in msg.outputs
        region = _param_string(o.parameters, _SHM_REGION)
        region === nothing && continue
        offset = something(_param_int(o.parameters, _SHM_OFFSET), 0)
        bsize = _param_int(o.parameters, _SHM_BYTE_SIZE)
        bsize === nothing && error("output '$(o.name)' references shared memory but is missing $_SHM_BYTE_SIZE")
        targets[o.name] = OutputTarget(region, offset, bsize)
    end

    deadline_ns = _decode_deadline_ns(msg.parameters)
    return DecodedRequest(InferRequest(msg.model_name, requested, tensors, deadline_ns), msg.id, targets)
end

# Copy a Julia col-major array's bytes into a fresh Vector{UInt8}. The bytes are already
# in the wire's row-major order for the reversed shape, so no permutation is needed; the
# direct byte copy avoids ReshapedArray/reinterpret intermediates.
function _raw_from_array(data::AbstractArray{T}) where {T}
    nb = sizeof(T) * length(data)
    out = Vector{UInt8}(undef, nb)
    GC.@preserve data out unsafe_copyto!(pointer(out), convert(Ptr{UInt8}, pointer(data)), nb)
    return out
end

_string_param(s) = _PB_INF.InferParameter(; parameter_choice=ProtoBuf.OneOf(:string_param, String(s)))
_int_param(i) = _PB_INF.InferParameter(; parameter_choice=ProtoBuf.OneOf(:int64_param, Int64(i)))

function _output_tensor(t::NamedTensor; parameters=Dict{String,_PB_INF.InferParameter}())
    # Wire shape is row-major: reverse of the Julia col-major NamedTensor.shape.
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    return _PB_INF.var"ModelInferResponse.InferOutputTensor"(;
        name=t.name, datatype=kserve_string(t.dtype),
        shape=wire_shape, parameters=parameters)
end

# Honor the client's requested_outputs: when the list is non-empty, return exactly those
# outputs in the requested order. Outputs the client did not ask for are dropped; a requested
# name the model does not produce is an error (surfaced to the client as INVALID_ARGUMENT).
function _select_outputs(outputs::Vector{NamedTensor}, requested::Vector{String})
    isempty(requested) && return outputs
    byname = Dict(t.name => t for t in outputs)
    return NamedTensor[get(() -> error("requested output '$name' is not produced by the model"),
                           byname, name) for name in requested]
end

_build_response(model_name, id, out_tensors, raw) =
    _PB_INF.ModelInferResponse(;
        model_name=String(model_name), id=String(id),
        outputs=out_tensors, raw_output_contents=raw)

"""
    encode_infer_response(model_name, id, outputs) -> ModelInferResponse

Build the response message with outputs entirely inline (raw_output_contents). The transport
serializes the returned message.
"""
function encode_infer_response(model_name::AbstractString, id::AbstractString, outputs::Vector{NamedTensor})
    out_tensors = [_output_tensor(t) for t in outputs]
    raw = Vector{UInt8}[_raw_from_array(t.data) for t in outputs]
    return _build_response(model_name, id, out_tensors, raw)
end

"""
    encode_infer_response(model_name, decoded, outputs, registry) -> ModelInferResponse

Build the response message, writing any output whose requested entry named a shared-memory
region into that region (and referencing it in the response) instead of inline.
raw_output_contents holds the inline outputs in order.
"""
function encode_infer_response(model_name::AbstractString, decoded::DecodedRequest,
                               outputs::Vector{NamedTensor},
                               registry::Union{SharedMemoryRegistry,Nothing})
    out_tensors = _PB_INF.var"ModelInferResponse.InferOutputTensor"[]
    raw = Vector{UInt8}[]
    selected = _select_outputs(outputs, decoded.request.requested_outputs)
    for t in selected
        tgt = get(decoded.output_targets, t.name, nothing)
        if tgt === nothing
            push!(out_tensors, _output_tensor(t))
            push!(raw, _raw_from_array(t.data))
        else
            registry === nothing && error("output '$(t.name)' targets shared memory but the server has no registry")
            bytes = _raw_from_array(t.data)
            length(bytes) <= tgt.byte_size ||
                error("output '$(t.name)' produced $(length(bytes)) bytes but region slot is $(tgt.byte_size)")
            shm_write!(registry, tgt.region, tgt.offset, bytes)
            params = Dict(
                _SHM_REGION => _string_param(tgt.region),
                _SHM_OFFSET => _int_param(tgt.offset),
                _SHM_BYTE_SIZE => _int_param(length(bytes)),
            )
            push!(out_tensors, _output_tensor(t; parameters=params))
        end
    end
    return _build_response(model_name, id_of(decoded), out_tensors, raw)
end

id_of(d::DecodedRequest) = d.id

# --- Outbound request / inbound response -----------------------------------------------------
#
# The pair below is the mirror image of decode_infer_request / encode_infer_response: it lets a
# process act as a *client* of a KServe V2 endpoint (used by the worker's meta-model GatewayCaller
# to call back into the gateway). Data travels inline via raw_input_contents / raw_output_contents;
# shared memory is deliberately not used on this path.

function _input_tensor(t::NamedTensor)
    # Wire shape is row-major: the reverse of the Julia col-major NamedTensor.shape.
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    return _PB_INF.var"ModelInferRequest.InferInputTensor"(;
        name=t.name, datatype=kserve_string(t.dtype), shape=wire_shape)
end

"""
    encode_infer_request(model_name, inputs; requested_outputs=String[], id="") -> ModelInferRequest

Build a ModelInferRequest from boundary [`NamedTensor`](@ref) inputs, with tensor data inline in
raw_input_contents. `requested_outputs`, when non-empty, names the outputs to return.
"""
function encode_infer_request(model_name::AbstractString, inputs::Vector{NamedTensor};
                              requested_outputs::Vector{String}=String[], id::AbstractString="",
                              parameters::Dict{String,_PB_INF.InferParameter}=Dict{String,_PB_INF.InferParameter}())
    in_tensors = [_input_tensor(t) for t in inputs]
    raw = Vector{UInt8}[_raw_from_array(t.data) for t in inputs]
    outs = _PB_INF.var"ModelInferRequest.InferRequestedOutputTensor"[
        _PB_INF.var"ModelInferRequest.InferRequestedOutputTensor"(; name=String(n)) for n in requested_outputs]
    return _PB_INF.ModelInferRequest(; model_name=String(model_name), id=String(id),
        inputs=in_tensors, outputs=outs, raw_input_contents=raw, parameters=parameters)
end

# Build an InferInputTensor that references bytes already staged in a shared-memory region rather
# than inlining them (the meta fan-out's transport==scratch path; mirrors the client encoder).
function _shm_input_tensor(t::NamedTensor, region::AbstractString, offset::Integer, byte_size::Integer)
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    params = Dict{String,_PB_INF.InferParameter}(
        _SHM_REGION => _string_param(region),
        _SHM_OFFSET => _int_param(offset),
        _SHM_BYTE_SIZE => _int_param(byte_size))
    return _PB_INF.var"ModelInferRequest.InferInputTensor"(;
        name=t.name, datatype=kserve_string(t.dtype), shape=wire_shape, parameters=params)
end

"""
    encode_infer_request_shm(model_name, inputs, region, offsets; requested_outputs, id)

Encode a request whose inputs are ALL staged in shared-memory `region` at the given byte `offsets`
(parallel to `inputs`); no `raw_input_contents` — the receiver reads each tensor via `shm_read`. The
receiver must have `region` registered. This is all-or-nothing per request: the decode path treats
`raw_input_contents` as parallel-to-inputs, so a request never mixes raw and SHM inputs.
"""
function encode_infer_request_shm(model_name::AbstractString, inputs::Vector{NamedTensor},
                                  region::AbstractString, offsets::Vector{<:Integer};
                                  requested_outputs::Vector{String}=String[], id::AbstractString="",
                                  parameters::Dict{String,_PB_INF.InferParameter}=Dict{String,_PB_INF.InferParameter}())
    length(offsets) == length(inputs) ||
        throw(ArgumentError("encode_infer_request_shm: offsets ($(length(offsets))) != inputs ($(length(inputs)))"))
    in_tensors = [_shm_input_tensor(inputs[i], region, offsets[i], sizeof(inputs[i].data))
                  for i in eachindex(inputs)]
    outs = _PB_INF.var"ModelInferRequest.InferRequestedOutputTensor"[
        _PB_INF.var"ModelInferRequest.InferRequestedOutputTensor"(; name=String(n)) for n in requested_outputs]
    return _PB_INF.ModelInferRequest(; model_name=String(model_name), id=String(id),
        inputs=in_tensors, outputs=outs, parameters=parameters)
end

"""
    decode_infer_response(msg) -> Vector{NamedTensor}

Translate a ModelInferResponse into boundary [`NamedTensor`](@ref) outputs. Data is read from
raw_output_contents when present, otherwise from the typed contents field. Shared-memory-backed
outputs are not supported on this path (the caller never requests them).
"""
function decode_infer_response(msg::_PB_INF.ModelInferResponse)
    n = length(msg.outputs)
    use_raw = !isempty(msg.raw_output_contents)
    if use_raw && length(msg.raw_output_contents) != n
        error("raw_output_contents has $(length(msg.raw_output_contents)) entries but response has $n outputs")
    end
    tensors = Vector{NamedTensor}(undef, n)
    for i in 1:n
        o = msg.outputs[i]
        dt = dtype_from_kserve(o.datatype)
        shape = Int[Int(s) for s in o.shape]
        data = if use_raw
            _array_from_raw(dt, shape, msg.raw_output_contents[i])
        elseif o.contents !== nothing
            _array_from_contents(dt, shape, o.contents)
        else
            error("output '$(o.name)' carries neither raw_output_contents nor contents")
        end
        tensors[i] = NamedTensor(o.name, dt, Tuple(size(data)), data)
    end
    return tensors
end

# The manifest's shape is Julia order (col-major); the wire metadata advertises the
# reverse (row-major), so Triton/KServe-style clients see canonical network dims.
_julia_shape_int64(s::TensorSpec) = Int64[d.kind == FIXED ? Int64(d.size) : Int64(-1) for d in s.shape]

function _tensor_metadata(s::TensorSpec)
    return _PB_INF.var"ModelMetadataResponse.TensorMetadata"(;
        name=s.name, datatype=kserve_string(s.dtype), shape=reverse(_julia_shape_int64(s)))
end

"""
    encode_model_metadata(name, manifest, platform) -> ModelMetadataResponse

Build a ModelMetadataResponse message from the manifest's client-facing I/O spec.
"""
function encode_model_metadata(name::AbstractString, manifest::Manifest, platform::AbstractString)
    return _PB_INF.ModelMetadataResponse(;
        name=String(name), versions=String[], platform=String(platform),
        inputs=[_tensor_metadata(s) for s in client_input_spec(manifest)],
        outputs=[_tensor_metadata(s) for s in client_output_spec(manifest)])
end

"""
    encode_repository_index(names) -> RepositoryIndexResponse
    encode_repository_index(entries::AbstractVector{<:Pair}) -> RepositoryIndexResponse

Build a RepositoryIndexResponse. The first form lists every model as READY (direct-client
introspection). The second takes `name => ready::Bool` pairs and reports `READY` or `UNAVAILABLE`
per model, so the gateway can discover which replicas actually serve a model (readiness reflects
residency on the worker).
"""
function encode_repository_index(names::AbstractVector{<:AbstractString})
    return encode_repository_index([String(n) => true for n in names])
end

function encode_repository_index(entries::AbstractVector{<:Pair})
    models = [_PB_INF.var"RepositoryIndexResponse.ModelIndex"(;
                  name=String(first(p)), version="",
                  state=(last(p) ? "READY" : "UNAVAILABLE"), reason="") for p in entries]
    return _PB_INF.RepositoryIndexResponse(; models=models)
end

# Build the registered regions into a SystemSharedMemoryStatusResponse message.
function encode_shm_status(reg::SharedMemoryRegistry, name::AbstractString)
    regions = shm_regions(reg)
    sel = isempty(name) ? regions : filter(p -> first(p) == name, regions)
    out = Dict{String,_PB_INF.var"SystemSharedMemoryStatusResponse.RegionStatus"}()
    for (rname, r) in sel
        out[rname] = _PB_INF.var"SystemSharedMemoryStatusResponse.RegionStatus"(;
            name=r.name, key=r.key, offset=UInt64(r.offset), byte_size=UInt64(r.byte_size))
    end
    return _PB_INF.SystemSharedMemoryStatusResponse(; regions=out)
end

encode_shm_register_response() = _PB_INF.SystemSharedMemoryRegisterResponse()
encode_shm_unregister_response() = _PB_INF.SystemSharedMemoryUnregisterResponse()

encode_is_same_ipc_namespace_response(same::Bool) =
    _PB_INF.IsSameIPCNamespaceResponse(; same=same)
