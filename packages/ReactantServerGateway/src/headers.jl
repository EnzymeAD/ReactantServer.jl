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

const _LENGTH_DELIMITED = PB.Codecs.LENGTH_DELIMITED

# The tensor parameters that place a tensor in a registered shared-memory region, and the bytes it
# occupies there (KServe's system shared-memory extension).
const _SHM_REGION_PARAM = "shared_memory_region"
const _SHM_BYTE_SIZE_PARAM = "shared_memory_byte_size"

# The value of an `InferParameter` when it is an `int64_param` (field 2), else `nothing`. The bool and
# string choices are skipped.
struct InferParameterInt
    value::Union{Nothing, Int64}
end
PB.default_values(::Type{InferParameterInt}) = (; value = nothing)
PB.field_numbers(::Type{InferParameterInt}) = (; value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InferParameterInt}, _endpos::Int = 0, _group::Bool = false)
    value = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2 && wire_type == PB.Codecs.VARINT
            value = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return InferParameterInt(value)
end

# Partial schema for one entry of a tensor's `parameters` map (key field 1, value field 2): the key,
# and the value when it is an int64 parameter.
struct ParameterEntryInt
    key::String
    value::Union{Nothing, Int64}
end
PB.default_values(::Type{ParameterEntryInt}) = (; key = "", value = nothing)
PB.field_numbers(::Type{ParameterEntryInt}) = (; key = 1, value = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ParameterEntryInt}, _endpos::Int = 0, _group::Bool = false)
    key = ""
    value = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            key = PB.decode(d, String)
        elseif field_number == 2 && wire_type == _LENGTH_DELIMITED
            v = Ref{Union{Nothing, InferParameterInt}}(nothing)
            PB.decode!(d, v)
            value = v[] === nothing ? nothing : v[].value
        else
            Base.skip(d, wire_type)
        end
    end
    return ParameterEntryInt(key, value)
end

# Partial schema for one InferInputTensor: its `name` (field 1), its `shape` (field 3), whether a
# parameter (field 4) places it in shared memory and the byte size declared there, and the encoded
# length of any inline `contents` (field 5), which is skipped rather than read. The datatype and
# every tensor payload are skipped, so entering this submessage never touches tensor data. The
# repeated-scalar decode is the library's, which handles both the packed and unpacked encodings of
# `repeated int64`; hand-rolling that works until a client encodes the other way.
struct InferInputData
    name::String
    shape::Vector{Int64}
    shm::Bool                  # a `shared_memory_region` parameter is present
    shm_bytes::Int64           # its `shared_memory_byte_size` (0 when absent or not an int64)
    contents_bytes::Int64      # encoded bytes of the inline typed `contents`
end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:InferInputData}, _endpos::Int = 0, _group::Bool = false)
    name = ""
    shape = PB.BufferedVector{Int64}()
    shm = false
    shm_bytes = Int64(0)
    contents_bytes = Int64(0)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 3
            PB.decode!(d, wire_type, shape)
        elseif field_number == 4 && wire_type == _LENGTH_DELIMITED
            entry = Ref{Union{Nothing, ParameterEntryInt}}(nothing)   # a fresh Ref per entry
            PB.decode!(d, entry)
            e = entry[]
            if e !== nothing
                e.key == _SHM_REGION_PARAM && (shm = true)
                (e.key == _SHM_BYTE_SIZE_PARAM && e.value !== nothing) && (shm_bytes = e.value)
            end
        elseif field_number == 5 && wire_type == _LENGTH_DELIMITED
            n = PB.decode(d, UInt64)
            skip(PB.Codecs.get_stream(d), n)
            contents_bytes += Int64(min(n, UInt64(typemax(Int32))))
        else
            Base.skip(d, wire_type)
        end
    end
    return InferInputData(name, shape[], shm, shm_bytes, contents_bytes)
end

"""
    peek_batch_input(body, input_name, axis) -> (extent, data_bytes)

The extent of `input_name`'s `axis`-th dimension in a serialized `ModelInferRequest`, and how many
bytes of data the request carries for that input: the `shared_memory_byte_size` it declares when it
is in shared memory, else the length of its entry in `raw_input_contents`, else the encoded length of
its inline typed `contents`. `(0, 0)` when the request has no such input; the extent is 0 when the
shape is too short. `input_name` and `axis` come from the worker over the control plane
(`ModelStatus.batch_input_name` / `batch_axis`), which is the only party that reads the manifest and
therefore the only one that knows where a bundle puts its batch axis.

Matching by NAME rather than position is deliberate: KServe inputs are name-addressed and nothing
requires a client to send them in manifest order. `raw_input_contents` is positional, so the input's
position among `inputs` picks its entry there.

Reading this costs no payload decode: every `raw_input_contents` entry and every inline `contents` is
skipped by its length prefix (`Base.skip` seeks rather than copying), and the shape is present even
on the shared-memory path, where the body carries no contents at all.
"""
function peek_batch_input(body::AbstractVector{UInt8}, input_name::AbstractString, axis::Integer)
    (isempty(input_name) || axis < 1) && return (0, 0)
    io = IOBuffer(body)
    d = PB.ProtoDecoder(io)
    seen = 0                                     # inputs read so far
    pos = 0                                      # the batch input's position among them
    found = nothing
    raw = Int64[]                                # raw_input_contents entry lengths, in order
    while !PB.message_done(d, 0, false)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 5 && wire_type == _LENGTH_DELIMITED
            # A FRESH Ref per input: `decode!` into a singular-message Ref MERGES successive
            # occurrences (proto3 semantics for a repeated field decoded as singular), so reusing one
            # concatenates the shapes of every input and silently reports the wrong extent.
            tensor = Ref{Union{Nothing, InferInputData}}(nothing)
            PB.decode!(d, tensor)
            seen += 1
            t = tensor[]
            if found === nothing && t !== nothing && t.name == input_name
                found = t
                pos = seen
            end
        elseif field_number == 7 && wire_type == _LENGTH_DELIMITED
            n = PB.decode(d, UInt64)
            skip(io, n)
            push!(raw, Int64(min(n, UInt64(typemax(Int32)))))
        else
            Base.skip(d, wire_type)
        end
    end
    found === nothing && return (0, 0)
    extent = axis <= length(found.shape) ? Int(found.shape[axis]) : 0
    bytes = found.shm ? max(found.shm_bytes, 0) :
        !isempty(raw) ? (pos <= length(raw) ? raw[pos] : 0) : found.contents_bytes
    return extent, Int(bytes)
end

"""
    peek_batch_size(body, input_name, axis) -> Int

The extent of `input_name`'s `axis`-th dimension as the request states it, or 0 when the request has
no such input or its shape is too short (see [`peek_batch_input`](@ref)). This is the client's claim;
[`request_item_count`](@ref) is what the gateway believes.
"""
peek_batch_size(body::AbstractVector{UInt8}, input_name::AbstractString, axis::Integer) =
    first(peek_batch_input(body, input_name, axis))

# The most items the gateway counts one request as, whatever it claims. Far above any real request (a
# trillion rows), and low enough that a load counter can add tens of thousands of such requests
# without overflowing (see `MAX_CHARGE` in scheduler.jl).
const MAX_REQUEST_ITEMS = 1 << 40

"""
    request_item_count(body, input_name, axis) -> Int

How many items the gateway counts a request as: the extent of its batch axis, but no more than the
bytes of data it carries for that input (every row takes at least one byte), at least 1, and at most
`MAX_REQUEST_ITEMS`. The extent and the byte count are both the client's, but the bytes are what the
request actually carries inline, so a request that claims 2^62 rows in a few bytes of data counts as
a few items. A shared-memory input's byte count is the size the client declares for it, which the
worker checks against the registered region; the ceiling bounds that case.

The gateway never rejects a request over its shape (the worker is the authority on what is valid);
a request it cannot read this far, or one with no such input, counts as one item.
"""
function request_item_count(body::AbstractVector{UInt8}, input_name::AbstractString, axis::Integer)
    (isempty(input_name) || axis < 1) && return 1
    extent, bytes = try
        peek_batch_input(body, input_name, axis)
    catch
        (0, 0)
    end
    return clamp(min(extent, bytes), 1, MAX_REQUEST_ITEMS)
end

# Read the region `name` (field 1) from a serialized SystemSharedMemory{Register,Unregister}Request.
# SystemSharedMemoryUnregisterRequest is already a name-only message, so it doubles as the partial
# schema for the register request (key/offset/byte_size are skipped).
function peek_shm_name(body::AbstractVector{UInt8})
    msg = PB.decode(PB.ProtoDecoder(IOBuffer(body)), SystemSharedMemoryUnregisterRequest)
    return msg.name
end

# Partial schema for one entry of a tensor's `parameters` map (`map<string, InferParameter>` is a
# repeated entry message with the key as field 1 and the value as field 2): its key only.
struct ParameterKey
    key::String
end
PB.default_values(::Type{ParameterKey}) = (; key = "")
PB.field_numbers(::Type{ParameterKey}) = (; key = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ParameterKey}, _endpos::Int = 0, _group::Bool = false)
    key = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            key = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ParameterKey(key)
end

# Partial schema for an input tensor (`parameters` is field 4) or a requested output (field 2, so
# `F` is the field number): whether any parameter names a shared-memory region. Everything else in
# the tensor, its inline contents included, is skipped.
struct TensorShmRef{F}
    shm::Bool
end
PB.default_values(::Type{TensorShmRef{F}}) where {F} = (; shm = false)
PB.field_numbers(::Type{TensorShmRef{F}}) where {F} = (; shm = F)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TensorShmRef{F}}, _endpos::Int = 0, _group::Bool = false) where {F}
    shm = false
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == F
            entry = Ref{Union{Nothing, ParameterKey}}(nothing)   # a fresh Ref per entry (see peek_batch_size)
            PB.decode!(d, entry)
            entry[] !== nothing && entry[].key == _SHM_REGION_PARAM && (shm = true)
        else
            Base.skip(d, wire_type)
        end
    end
    return TensorShmRef{F}(shm)
end

"""
    peek_uses_shared_memory(body) -> Bool

Whether a serialized `ModelInferRequest` reads any input from, or writes any requested output to,
a registered shared-memory region (a `shared_memory_region` parameter on the tensor). Only the
tensors' parameter keys are read; the payloads are skipped, as everywhere in the gateway. A body
that cannot be read this far counts as using shared memory, the safe answer for the caller (which
then never runs the request twice).
"""
function peek_uses_shared_memory(body::AbstractVector{UInt8})
    try
        d = PB.ProtoDecoder(IOBuffer(body))
        while !PB.message_done(d, 0, false)
            field_number, wire_type = PB.decode_tag(d)
            if field_number == 5          # inputs
                t = Ref{Union{Nothing, TensorShmRef{4}}}(nothing)
                PB.decode!(d, t)
                t[] !== nothing && t[].shm && return true
            elseif field_number == 6      # outputs
                o = Ref{Union{Nothing, TensorShmRef{2}}}(nothing)
                PB.decode!(d, o)
                o[] !== nothing && o[].shm && return true
            else
                Base.skip(d, wire_type)
            end
        end
        return false
    catch
        return true
    end
end
