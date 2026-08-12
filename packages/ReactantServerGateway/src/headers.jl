# Partial-schema decoding of the routing fields the gateway needs, without decoding a whole
# message. ProtoBuf's generated decode reads only the fields a type declares and skips the rest
# with `Base.skip`, so a struct that declares only `model_name` and `id` seeks past the tensor
# payloads of a ModelInferRequest rather than allocating them. The forwarded bytes are never
# decoded; only these small headers are.

import ProtoBuf as PB

# Minimal partial schema for ModelInferRequest: only the routing key (field 1) and the client
# correlation id (field 3, used for the audit log). Every other field, including the inline
# tensor payloads, is skipped.
struct ModelInferHeader
    model_name::String
    id::String
end
PB.default_values(::Type{ModelInferHeader}) = (; model_name = "", id = "")
PB.field_numbers(::Type{ModelInferHeader}) = (; model_name = 1, id = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ModelInferHeader}, _endpos::Int = 0, _group::Bool = false)
    model_name = ""
    id = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            model_name = PB.decode(d, String)
        elseif field_number == 3
            id = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ModelInferHeader(model_name, id)
end

# Read model_name and id from a serialized ModelInferRequest without touching the payload.
function peek_model_name_and_id(body::AbstractVector{UInt8})
    h = PB.decode(PB.ProtoDecoder(IOBuffer(body)), ModelInferHeader)
    return h.model_name, h.id
end

# Partial schema for one InferInputTensor: its `name` (field 1) and `shape` (field 3). The datatype,
# parameters, and any inline `contents` are skipped, so entering this submessage never touches
# tensor data. The repeated-scalar decode is the library's, which handles both the packed and
# unpacked encodings of `repeated int64`; hand-rolling that works until a client encodes the other
# way.
struct InferInputShape
    name::String
    shape::Vector{Int64}
end
PB.default_values(::Type{InferInputShape}) = (; name = "", shape = Int64[])
PB.field_numbers(::Type{InferInputShape}) = (; name = 1, shape = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InferInputShape}, _endpos::Int = 0, _group::Bool = false)
    name = ""
    shape = PB.BufferedVector{Int64}()
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, wire_type, shape)
        else
            Base.skip(d, wire_type)
        end
    end
    return InferInputShape(name, shape[])
end

"""
    peek_batch_size(body, input_name, axis) -> Int

The extent of `input_name`'s `axis`-th dimension in a serialized `ModelInferRequest`, or 0 when the
request has no such input or its shape is too short. `input_name` and `axis` come from the worker
over the control plane (`ModelStatus.batch_input_name` / `batch_axis`), which is the only party that
reads the manifest and therefore the only one that knows where a bundle puts its batch axis.

Matching by NAME rather than position is deliberate: KServe inputs are name-addressed and nothing
requires a client to send them in manifest order.

Reading this costs no payload decode. `inputs` is field 5 while the bulk `raw_input_contents` is
field 7, and within a tensor `shape` is field 3 while any inline `contents` is field 5, so the
shapes always precede the data on the wire; `Base.skip` seeks over the rest rather than copying it.
The shape is present even on the shared-memory path, where the body carries no contents at all.
"""
function peek_batch_size(body::AbstractVector{UInt8}, input_name::AbstractString, axis::Integer)
    (isempty(input_name) || axis < 1) && return 0
    d = PB.ProtoDecoder(IOBuffer(body))
    while !PB.message_done(d, 0, false)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 5
            # A FRESH Ref per input: `decode!` into a singular-message Ref MERGES successive
            # occurrences (proto3 semantics for a repeated field decoded as singular), so reusing one
            # concatenates the shapes of every input and silently reports the wrong extent.
            tensor = Ref{Union{Nothing, InferInputShape}}(nothing)
            PB.decode!(d, tensor)
            t = tensor[]
            if t !== nothing && t.name == input_name
                return axis <= length(t.shape) ? Int(t.shape[axis]) : 0
            end
        else
            Base.skip(d, wire_type)
        end
    end
    return 0
end

# Read the region `name` (field 1) from a serialized SystemSharedMemory{Register,Unregister}Request.
# SystemSharedMemoryUnregisterRequest is already a name-only message, so it doubles as the partial
# schema for the register request (key/offset/byte_size are skipped).
function peek_shm_name(body::AbstractVector{UInt8})
    msg = PB.decode(PB.ProtoDecoder(IOBuffer(body)), SystemSharedMemoryUnregisterRequest)
    return msg.name
end
