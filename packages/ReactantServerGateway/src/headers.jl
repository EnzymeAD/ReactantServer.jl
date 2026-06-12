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

# Read the region `name` (field 1) from a serialized SystemSharedMemory{Register,Unregister}Request.
# SystemSharedMemoryUnregisterRequest is already a name-only message, so it doubles as the partial
# schema for the register request (key/offset/byte_size are skipped).
function peek_shm_name(body::AbstractVector{UInt8})
    msg = PB.decode(PB.ProtoDecoder(IOBuffer(body)), SystemSharedMemoryUnregisterRequest)
    return msg.name
end
