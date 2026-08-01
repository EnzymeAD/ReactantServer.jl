# Partial-schema decoding of the routing fields the gateway needs, without decoding a whole
# message. ProtoBuf's generated decode reads only the fields a type declares and skips the rest
# with `Base.skip`, so a struct that declares only `model_name` and `id` seeks past the tensor
# payloads of a ModelInferRequest rather than allocating them. The forwarded bytes are never
# decoded; only these small headers are.

import ProtoBuf as PB

# Minimal partial schema for ModelInferRequest: the routing key (field 1), the client correlation id
# (field 3, used for the audit log), and the leading dimension of the first input tensor (field 5,
# `inputs[0].shape[0]`), which is how many items the request carries. Every other field, including
# every tensor payload, is skipped.
#
# Reading the leading dimension does not cost a payload decode. `inputs` is field 5 and the bulk
# `raw_input_contents` is field 7, and within `InferInputTensor` the `shape` is field 3 while the
# inline `contents` is field 5, so the dimension always precedes the data on the wire whichever
# transport the client uses. It is present even on the shared-memory path, where the body carries
# shapes and no contents at all. `Base.skip` seeks over what it skips rather than copying it.
struct ModelInferHeader
    model_name::String
    id::String
    batch::Int          # inputs[0].shape[0]; 0 when there are no inputs or no shape
end
PB.default_values(::Type{ModelInferHeader}) = (; model_name = "", id = "", batch = 0)
PB.field_numbers(::Type{ModelInferHeader}) = (; model_name = 1, id = 3, batch = 5)

# Partial schema for one InferInputTensor: only `shape` (field 3). The name, datatype, parameters,
# and any inline `contents` are skipped, so entering this submessage never touches tensor data. The
# repeated-scalar decode is the library's, which handles both the packed and unpacked encodings of
# `repeated int64`; hand-rolling that is the kind of thing that works until a client encodes the
# other way.
struct InferInputShape
    shape::Vector{Int64}
end
PB.default_values(::Type{InferInputShape}) = (; shape = Int64[])
PB.field_numbers(::Type{InferInputShape}) = (; shape = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InferInputShape}, _endpos::Int = 0, _group::Bool = false)
    shape = PB.BufferedVector{Int64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 3
            PB.decode!(d, wire_type, shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return InferInputShape(shape[])
end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ModelInferHeader}, _endpos::Int = 0, _group::Bool = false)
    model_name = ""
    id = ""
    batch = 0
    have_batch = false
    tensor = Ref{Union{Nothing,InferInputShape}}(nothing)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            model_name = PB.decode(d, String)
        elseif field_number == 3
            id = PB.decode(d, String)
        elseif field_number == 5 && !have_batch
            # Only the FIRST input tensor is inspected. Inputs that disagree on their leading
            # dimension are a model-authoring bug, not something the gateway should adjudicate; the
            # remaining `inputs` entries fall through to the skip below.
            PB.decode!(d, tensor)
            t = tensor[]
            batch = (t === nothing || isempty(t.shape)) ? 0 : Int(t.shape[1])
            have_batch = true
        else
            Base.skip(d, wire_type)
        end
    end
    return ModelInferHeader(model_name, id, batch)
end

"""
    peek_model_header(body) -> (model_name, id, batch)

Read the routing fields from a serialized `ModelInferRequest` without decoding its payload. `batch`
is the leading dimension of the first input tensor, or 0 when the request declares no shaped input.

Whether that dimension actually means "items" is the scheduler's call, not this function's: an
unbatched model's leading dimension is a real axis (channels, say), so the caller gates on the
worker-reported max batch. See `resolve_fill_plan`.
"""
function peek_model_header(body::AbstractVector{UInt8})
    h = PB.decode(PB.ProtoDecoder(IOBuffer(body)), ModelInferHeader)
    return h.model_name, h.id, h.batch
end

# Read the region `name` (field 1) from a serialized SystemSharedMemory{Register,Unregister}Request.
# SystemSharedMemoryUnregisterRequest is already a name-only message, so it doubles as the partial
# schema for the register request (key/offset/byte_size are skipped).
function peek_shm_name(body::AbstractVector{UInt8})
    msg = PB.decode(PB.ProtoDecoder(IOBuffer(body)), SystemSharedMemoryUnregisterRequest)
    return msg.name
end
