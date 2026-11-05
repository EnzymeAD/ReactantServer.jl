# Translation between the KServe V2 protobuf wire messages and the boundary types.
#
# Tensor data travels either inline (raw_input_contents / raw_output_contents) or through a
# registered shared-memory region referenced by the tensor's parameters
# (shared_memory_region / shared_memory_offset / shared_memory_byte_size). The shared-memory
# path is one copy from the mapped region into a host array for input, and one copy back into
# the region for output. The codec depends only on the generated protobuf, dtypes, and the
# shared-memory registry, never on HTTP.

const _PB_INF = inference

using Dates: DateTime

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
    deadline_params(PB::Module, budget_ns) -> Dict{String,InferParameter}

Build the request-level parameters map carrying a remaining-budget timeout of `budget_ns`
nanoseconds (see [`TIMEOUT_NS_PARAM`](@ref)). A non-positive `budget_ns` yields an empty map (no
deadline). Merge the result into a `ModelInferRequest`'s `parameters`. The PB-first form builds the
map from pb module `PB` (consumer packages pass their own generated module; the single-argument
form uses this package's `inference` module).
"""
function deadline_params(budget_ns::Integer)
    return deadline_params(inference, budget_ns)
end

function deadline_params(PB::Module, budget_ns::Integer)
    budget_ns > 0 || return Dict{String, PB.InferParameter}()
    return Dict{String, PB.InferParameter}(TIMEOUT_NS_PARAM => _int_param(PB, Int64(budget_ns)))
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

# Convert a request deadline carried as a wall-clock DateTime (or nothing) plus the still-remaining
# budget in seconds (or nothing) into an absolute deadline on Julia's monotonic time_ns() clock,
# saturating at typemax(Int64). Returns 0 when either input is nothing (no deadline on that
# channel), matching _decode_deadline_ns's convention. The scheduler compares against time_ns()
# (scheduler.jl; the gateway's _post_infer uses deadline_ns - time_ns()), so handlers on the
# merged gRPCServer, which parses grpc-timeout into ctx.deadline::DateTime and offers
# remaining_time(ctx) in seconds, recover the semantics the legacy runtime provided natively via
# ctx.deadline_ns. Transport-agnostic: only the two primitives are touched.
function deadline_to_time_ns(deadline::Union{DateTime, Nothing}, remaining_seconds::Union{Real, Nothing})
    deadline === nothing && return Int64(0)
    remaining_seconds === nothing && return Int64(0)
    # Guard the conversion itself: a NaN or absurdly large remaining budget must
    # saturate to a far-future deadline, never throw InexactError.
    budget = Float64(remaining_seconds) * 1.0e9
    (isnan(budget) || budget >= typemax(Int64)) && return typemax(Int64)
    budget_i = round(Int64, budget)
    now_ns = Int64(time_ns())
    return budget_i > typemax(Int64) - now_ns ? typemax(Int64) : now_ns + budget_i
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

# Read a typed parameter out of a parameters map, or nothing if absent/mismatched.
function _param_int(params, key)
    haskey(params, key) || return nothing
    p = params[key].parameter_choice
    return (p !== nothing && p.name === :int64_param) ? Int(p[]) : nothing
end

# Read `key` from a parameters map, insisting on the parameter type `kind` (:string_param or
# :int64_param). Absent is `nothing`; present with any other type (or no value) is an error, never
# silently "absent". `what` names the tensor for the message.
function _strict_param(params, key::AbstractString, kind::Symbol, what::AbstractString)
    haskey(params, key) || return nothing
    p = params[key].parameter_choice
    (p !== nothing && p.name === kind) ||
        error("invalid value type for '$key' parameter for $what, expected $kind")
    return p[]
end

# A tensor's shared-memory reference `(region, offset, byte_size)`, or `nothing` when it does not
# reference shared memory. Read strictly, as Triton and the KServe system shared-memory extension
# define it: `shared_memory_region` is a string_param, `shared_memory_offset` and
# `shared_memory_byte_size` are int64_params; the offset defaults to 0, the byte size is required,
# and neither may appear without a region. A parameter of any other type used to read as absent, so
# an offset sent as a string read from (or wrote to) offset 0; every such case is now a client error
# (INVALID_ARGUMENT at the transport).
function _shm_reference(params, what::AbstractString)
    region = _strict_param(params, _SHM_REGION, :string_param, what)
    offset = _strict_param(params, _SHM_OFFSET, :int64_param, what)
    bsize = _strict_param(params, _SHM_BYTE_SIZE, :int64_param, what)
    if region === nothing
        offset === nothing ||
            error("'$_SHM_OFFSET' can not be specified without '$_SHM_REGION' for $what")
        bsize === nothing ||
            error("'$_SHM_BYTE_SIZE' can not be specified without '$_SHM_REGION' for $what")
        return nothing
    end
    isempty(region) && error("'$_SHM_REGION' for $what is empty")
    bsize === nothing &&
        error("'$_SHM_BYTE_SIZE' must be specified along with '$_SHM_REGION' for $what")
    off = something(offset, Int64(0))
    off >= 0 || error("'$_SHM_OFFSET' for $what must be non-negative, got $off")
    bsize >= 0 || error("'$_SHM_BYTE_SIZE' for $what must be non-negative, got $bsize")
    return String(region), Int(off), Int(bsize)
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
    output_targets::Dict{String, OutputTarget}   # by output name; only shm-backed outputs
end

"""
    decode_infer_request(msg, registry=nothing) -> DecodedRequest
    decode_infer_request(PB::Module, msg, registry=nothing) -> DecodedRequest

Translate a decoded ModelInferRequest message into the boundary InferRequest. The transport
(gRPC) hands us the already-decoded protobuf message, so the codec never touches wire bytes.
Input tensor data is read from a registered shared-memory region (preferred when the tensor
declares one), otherwise from raw_input_contents, otherwise from the typed contents field.
The PB-first form accepts a message from pb module `PB` (consumer packages pass their own
generated module; the shorter form uses this package's `inference` module).
"""
function decode_infer_request(
        msg, registry::Union{SharedMemoryRegistry, Nothing} = nothing
    )
    return decode_infer_request(inference, msg, registry)
end

function decode_infer_request(
        PB::Module,
        msg,
        registry::Union{SharedMemoryRegistry, Nothing} = nothing
    )
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
        ref = _shm_reference(t.parameters, "input '$(t.name)'")
        data = if ref !== nothing
            registry === nothing && error("input '$(t.name)' references shared memory but the server has no registry")
            region, offset, bsize = ref
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
    targets = Dict{String, OutputTarget}()
    for o in msg.outputs
        ref = _shm_reference(o.parameters, "output '$(o.name)'")
        ref === nothing && continue
        targets[o.name] = OutputTarget(ref...)
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

_string_param(PB::Module, s) =
    PB.InferParameter(; parameter_choice = ProtoBuf.OneOf(:string_param, String(s)))
_int_param(PB::Module, i) =
    PB.InferParameter(; parameter_choice = ProtoBuf.OneOf(:int64_param, Int64(i)))

function _output_tensor(PB::Module, t::NamedTensor; parameters = Dict{String, PB.InferParameter}())
    # Wire shape is row-major: reverse of the Julia col-major NamedTensor.shape.
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    return PB.var"ModelInferResponse.InferOutputTensor"(;
        name = t.name, datatype = kserve_string(t.dtype),
        shape = wire_shape, parameters = parameters
    )
end

# Honor the client's requested_outputs: when the list is non-empty, return exactly those
# outputs in the requested order. Outputs the client did not ask for are dropped; a requested
# name the model does not produce is an error (surfaced to the client as INVALID_ARGUMENT).
function _select_outputs(outputs::Vector{NamedTensor}, requested::Vector{String})
    isempty(requested) && return outputs
    byname = Dict(t.name => t for t in outputs)
    return NamedTensor[
        get(
            () -> error("requested output '$name' is not produced by the model"),
            byname, name
        ) for name in requested
    ]
end

_build_response(PB::Module, model_name, id, out_tensors, raw) =
    PB.ModelInferResponse(;
    model_name = String(model_name), id = String(id),
    outputs = out_tensors, raw_output_contents = raw
)

"""
    encode_infer_response(model_name, id, outputs) -> ModelInferResponse
    encode_infer_response(PB::Module, model_name, id, outputs) -> ModelInferResponse

Build the response message with outputs entirely inline (raw_output_contents). The transport
serializes the returned message. The PB-first form builds the message from pb module `PB`
(consumer packages pass their own generated module; the shorter form uses this package's
`inference` module).
"""
function encode_infer_response(model_name::AbstractString, id::AbstractString, outputs::Vector{NamedTensor})
    return encode_infer_response(inference, model_name, id, outputs)
end

function encode_infer_response(PB::Module, model_name::AbstractString, id::AbstractString, outputs::Vector{NamedTensor})
    out_tensors = [_output_tensor(PB, t) for t in outputs]
    raw = Vector{UInt8}[_raw_from_array(t.data) for t in outputs]
    return _build_response(PB, model_name, id, out_tensors, raw)
end

"""
    encode_infer_response(model_name, decoded, outputs, registry) -> ModelInferResponse
    encode_infer_response(PB::Module, model_name, decoded, outputs, registry) -> ModelInferResponse

Build the response message, writing any output whose requested entry named a shared-memory
region into that region (and referencing it in the response) instead of inline.
raw_output_contents holds one entry per output in `outputs` order, as the KServe proto requires:
the bytes of an inline output, or an empty placeholder for a shared-memory output, so a client
that indexes it by output position (as Triton clients do) reads the right tensor. The PB-first
form builds the message from pb module `PB` (consumer packages pass their own generated module).
"""
function encode_infer_response(
        model_name::AbstractString, decoded::DecodedRequest,
        outputs::Vector{NamedTensor},
        registry::Union{SharedMemoryRegistry, Nothing}
    )
    return encode_infer_response(inference, model_name, decoded, outputs, registry)
end

function encode_infer_response(
        PB::Module, model_name::AbstractString, decoded::DecodedRequest,
        outputs::Vector{NamedTensor},
        registry::Union{SharedMemoryRegistry, Nothing}
    )
    out_tensors = PB.var"ModelInferResponse.InferOutputTensor"[]
    raw = Vector{UInt8}[]
    selected = _select_outputs(outputs, decoded.request.requested_outputs)
    for t in selected
        tgt = get(decoded.output_targets, t.name, nothing)
        if tgt === nothing
            push!(out_tensors, _output_tensor(PB, t))
            push!(raw, _raw_from_array(t.data))
        else
            registry === nothing && error("output '$(t.name)' targets shared memory but the server has no registry")
            bytes = _raw_from_array(t.data)
            length(bytes) <= tgt.byte_size ||
                error("output '$(t.name)' produced $(length(bytes)) bytes but region slot is $(tgt.byte_size)")
            shm_write!(registry, tgt.region, tgt.offset, bytes)
            params = Dict(
                _SHM_REGION => _string_param(PB, tgt.region),
                _SHM_OFFSET => _int_param(PB, tgt.offset),
                _SHM_BYTE_SIZE => _int_param(PB, length(bytes)),
            )
            push!(out_tensors, _output_tensor(PB, t; parameters = params))
            push!(raw, UInt8[])   # placeholder keeps raw_output_contents aligned with outputs
        end
    end
    return _build_response(PB, model_name, id_of(decoded), out_tensors, raw)
end

id_of(d::DecodedRequest) = d.id

# --- Outbound request / inbound response -----------------------------------------------------
#
# The pair below is the mirror image of decode_infer_request / encode_infer_response: it lets a
# process act as a *client* of a KServe V2 endpoint (used by the worker's meta-model GatewayCaller
# to call back into the gateway). Data travels inline via raw_input_contents / raw_output_contents;
# shared memory is deliberately not used on this path.

function _input_tensor(PB::Module, t::NamedTensor)
    # Wire shape is row-major: the reverse of the Julia col-major NamedTensor.shape.
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    return PB.var"ModelInferRequest.InferInputTensor"(;
        name = t.name, datatype = kserve_string(t.dtype), shape = wire_shape
    )
end

"""
    encode_infer_request(model_name, inputs; requested_outputs=String[], id="") -> ModelInferRequest
    encode_infer_request(PB::Module, model_name, inputs; requested_outputs=String[], id="") -> ModelInferRequest

Build a ModelInferRequest from boundary [`NamedTensor`](@ref) inputs, with tensor data inline in
raw_input_contents. `requested_outputs`, when non-empty, names the outputs to return. The PB-first
form builds the message from pb module `PB` (consumer packages pass their own generated module).
"""
function encode_infer_request(
        model_name::AbstractString, inputs::Vector{NamedTensor}; kwargs...
    )
    return encode_infer_request(inference, model_name, inputs; kwargs...)
end

function encode_infer_request(
        PB::Module,
        model_name::AbstractString, inputs::Vector{NamedTensor};
        requested_outputs::Vector{String} = String[], id::AbstractString = "",
        parameters = Dict{String, PB.InferParameter}()
    )
    in_tensors = [_input_tensor(PB, t) for t in inputs]
    raw = Vector{UInt8}[_raw_from_array(t.data) for t in inputs]
    outs = PB.var"ModelInferRequest.InferRequestedOutputTensor"[
        PB.var"ModelInferRequest.InferRequestedOutputTensor"(; name = String(n)) for n in requested_outputs
    ]
    return PB.ModelInferRequest(;
        model_name = String(model_name), id = String(id),
        inputs = in_tensors, outputs = outs, raw_input_contents = raw, parameters = parameters
    )
end

# Build an InferInputTensor that references bytes already staged in a shared-memory region rather
# than inlining them (the meta fan-out's transport==scratch path; mirrors the client encoder).
function _shm_input_tensor(PB::Module, t::NamedTensor, region::AbstractString, offset::Integer, byte_size::Integer)
    wire_shape = Int64[Int64(s) for s in reverse(collect(t.shape))]
    params = Dict{String, PB.InferParameter}(
        _SHM_REGION => _string_param(PB, region),
        _SHM_OFFSET => _int_param(PB, offset),
        _SHM_BYTE_SIZE => _int_param(PB, byte_size)
    )
    return PB.var"ModelInferRequest.InferInputTensor"(;
        name = t.name, datatype = kserve_string(t.dtype), shape = wire_shape, parameters = params
    )
end

"""
    encode_infer_request_shm(model_name, inputs, region, offsets; requested_outputs, id)
    encode_infer_request_shm(PB::Module, model_name, inputs, region, offsets; requested_outputs, id)

Encode a request whose inputs are ALL staged in shared-memory `region` at the given byte `offsets`
(parallel to `inputs`); no `raw_input_contents` (the receiver reads each tensor via `shm_read`). The
receiver must have `region` registered. This is all-or-nothing per request: the decode path treats
`raw_input_contents` as parallel-to-inputs, so a request never mixes raw and SHM inputs. The PB-first
form builds the message from pb module `PB`.
"""
function encode_infer_request_shm(
        model_name::AbstractString, inputs::Vector{NamedTensor},
        region::AbstractString, offsets::Vector{<:Integer}; kwargs...
    )
    return encode_infer_request_shm(inference, model_name, inputs, region, offsets; kwargs...)
end

function encode_infer_request_shm(
        PB::Module,
        model_name::AbstractString, inputs::Vector{NamedTensor},
        region::AbstractString, offsets::Vector{<:Integer};
        requested_outputs::Vector{String} = String[], id::AbstractString = "",
        parameters = Dict{String, PB.InferParameter}()
    )
    length(offsets) == length(inputs) ||
        throw(ArgumentError("encode_infer_request_shm: offsets ($(length(offsets))) != inputs ($(length(inputs)))"))
    in_tensors = [
        _shm_input_tensor(PB, inputs[i], region, offsets[i], sizeof(inputs[i].data))
            for i in eachindex(inputs)
    ]
    outs = PB.var"ModelInferRequest.InferRequestedOutputTensor"[
        PB.var"ModelInferRequest.InferRequestedOutputTensor"(; name = String(n)) for n in requested_outputs
    ]
    return PB.ModelInferRequest(;
        model_name = String(model_name), id = String(id),
        inputs = in_tensors, outputs = outs, parameters = parameters
    )
end

"""
    decode_infer_response(msg) -> Vector{NamedTensor}
    decode_infer_response(PB::Module, msg) -> Vector{NamedTensor}

Translate a ModelInferResponse into boundary [`NamedTensor`](@ref) outputs. Data is read from
raw_output_contents when present, otherwise from the typed contents field. Shared-memory-backed
outputs are not supported on this path (the caller never requests them). The PB-first form accepts
a message from pb module `PB` (consumer packages pass their own generated module).
"""
function decode_infer_response(msg)
    return decode_infer_response(inference, msg)
end

function decode_infer_response(PB::Module, msg)
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

function _tensor_metadata(PB::Module, s::TensorSpec)
    return PB.var"ModelMetadataResponse.TensorMetadata"(;
        name = s.name, datatype = kserve_string(s.dtype), shape = reverse(_julia_shape_int64(s))
    )
end

"""
    encode_model_metadata(name, manifest, platform) -> ModelMetadataResponse
    encode_model_metadata(PB::Module, name, manifest, platform) -> ModelMetadataResponse

Build a ModelMetadataResponse message from the manifest's client-facing I/O spec. The PB-first
form builds the message from pb module `PB` (consumer packages pass their own generated module).
"""
function encode_model_metadata(name::AbstractString, manifest::Manifest, platform::AbstractString)
    return encode_model_metadata(inference, name, manifest, platform)
end

function encode_model_metadata(PB::Module, name::AbstractString, manifest::Manifest, platform::AbstractString)
    return PB.ModelMetadataResponse(;
        name = String(name), versions = String[], platform = String(platform),
        inputs = [_tensor_metadata(PB, s) for s in client_input_spec(manifest)],
        outputs = [_tensor_metadata(PB, s) for s in client_output_spec(manifest)]
    )
end

_axis_kind(PB::Module, d::Dim) =
    d.kind == FIXED ? PB.AxisKind.AXIS_FIXED :
    d.kind == BATCH ? PB.AxisKind.AXIS_BATCH : PB.AxisKind.AXIS_VARIABLE

# Row-major like the KServe metadata, with the kind of each axis alongside, so a batch axis and a
# variable axis (both -1 in the shape) can be told apart.
function _tensor_description(PB::Module, s::TensorSpec)
    return PB.TensorDescription(;
        name = s.name, datatype = kserve_string(s.dtype),
        shape = reverse(_julia_shape_int64(s)),
        axes = reverse([_axis_kind(PB, d) for d in s.shape])
    )
end

# The compiled input shapes, one concrete shape per executable input for each variant. A variant key
# lists the variable-axis sizes in (input, axis) order (see `_variable_axes` in manifest.jl), so they
# are consumed in that same order. Only reported when the client sends the executable inputs: a
# model.jl that transforms its inputs decides for itself which client shapes map onto a variant.
function _shape_variants(PB::Module, m::Manifest)
    (isempty(m.input_shapes) || m.client_inputs !== nothing) && return PB.ShapeVariant[]
    return map(m.input_shapes) do key
        k = 0
        inputs = map(m.executable_inputs) do sp
            dims = map(sp.shape) do d
                d.kind == FIXED && return Int64(d.size)
                d.kind == BATCH && return Int64(-1)
                k += 1
                return Int64(key[k])
            end
            PB.TensorShape(; name = sp.name, shape = reverse(dims))
        end
        PB.ShapeVariant(; inputs = inputs)
    end
end

"""
    encode_model_description(name, manifest; max_batch_size = 0) -> ModelDescription
    encode_model_description(PB::Module, name, manifest; max_batch_size = 0) -> ModelDescription

Build the control plane's `ModelDescription` for a served model from its manifest: the client-facing
tensors with the kind of every axis, the compiled batch sizes and input-shape variants, and a meta
model's sub-models. `max_batch_size` is the worker's effective coalescing limit, which the manifest
alone does not know. The PB-first form builds the message from pb module `PB`.
"""
function encode_model_description(name::AbstractString, manifest::Manifest; max_batch_size::Integer = 0)
    return encode_model_description(control, name, manifest; max_batch_size = max_batch_size)
end

function encode_model_description(
        PB::Module, name::AbstractString, manifest::Manifest; max_batch_size::Integer = 0
    )
    return PB.ModelDescription(;
        name = String(name), kind = manifest.kind, description = manifest.description,
        inputs = [_tensor_description(PB, s) for s in client_input_spec(manifest)],
        outputs = [_tensor_description(PB, s) for s in client_output_spec(manifest)],
        batch_sizes = Int64.(sort(unique(manifest.batching.compiled_batch_sizes))),
        max_batch_size = Int64(max_batch_size),
        calls = copy(manifest.meta_calls),
        variants = _shape_variants(PB, manifest)
    )
end

"""
    encode_repository_index(names) -> RepositoryIndexResponse
    encode_repository_index(entries::AbstractVector{<:Pair}) -> RepositoryIndexResponse
    encode_repository_index(PB::Module, names) -> RepositoryIndexResponse
    encode_repository_index(PB::Module, entries::AbstractVector{<:Pair}) -> RepositoryIndexResponse

Build a RepositoryIndexResponse. The first form lists every model as READY (direct-client
introspection). The second takes `name => ready::Bool` pairs and reports `READY` or `UNAVAILABLE`
per model, so the gateway can discover which replicas actually serve a model (readiness reflects
residency on the worker). The PB-first forms build the message from pb module `PB`.
"""
function encode_repository_index(names::AbstractVector{<:AbstractString})
    return encode_repository_index(inference, names)
end

function encode_repository_index(entries::AbstractVector{<:Pair})
    return encode_repository_index(inference, entries)
end

function encode_repository_index(PB::Module, names::AbstractVector{<:AbstractString})
    return encode_repository_index(PB, [String(n) => true for n in names])
end

function encode_repository_index(PB::Module, entries::AbstractVector{<:Pair})
    models = [
        PB.var"RepositoryIndexResponse.ModelIndex"(;
            name = String(first(p)), version = "",
            state = (last(p) ? "READY" : "UNAVAILABLE"), reason = ""
        ) for p in entries
    ]
    return PB.RepositoryIndexResponse(; models = models)
end

# Build the registered regions into a SystemSharedMemoryStatusResponse message.
function encode_shm_status(reg::SharedMemoryRegistry, name::AbstractString)
    return encode_shm_status(inference, reg, name)
end

function encode_shm_status(PB::Module, reg::SharedMemoryRegistry, name::AbstractString)
    regions = shm_regions(reg)
    sel = isempty(name) ? regions : filter(p -> first(p) == name, regions)
    out = Dict{String, PB.var"SystemSharedMemoryStatusResponse.RegionStatus"}()
    for (rname, r) in sel
        out[rname] = PB.var"SystemSharedMemoryStatusResponse.RegionStatus"(;
            name = r.name, key = r.key, offset = UInt64(r.offset), byte_size = UInt64(r.byte_size)
        )
    end
    return PB.SystemSharedMemoryStatusResponse(; regions = out)
end

encode_shm_register_response() = encode_shm_register_response(inference)
encode_shm_register_response(PB::Module) = PB.SystemSharedMemoryRegisterResponse()

encode_shm_unregister_response() = encode_shm_unregister_response(inference)
encode_shm_unregister_response(PB::Module) = PB.SystemSharedMemoryUnregisterResponse()

encode_is_same_ipc_namespace_response(same::Bool) =
    encode_is_same_ipc_namespace_response(inference, same)

encode_is_same_ipc_namespace_response(PB::Module, same::Bool) =
    PB.IsSameIPCNamespaceResponse(; same = same)
