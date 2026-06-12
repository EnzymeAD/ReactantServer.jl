# Transport-agnostic boundary between the wire layer and the rest of the server.
#
# The codec decodes a wire ModelInferRequest into an InferRequest; the scheduler and
# runtime consume only these types. Swapping the transport (HTTP to gRPC) changes how
# these are produced, not the types themselves.

"""
    NamedTensor(name, dtype, shape, data)
    NamedTensor(name, data)

A named host tensor carried across the transport boundary as both an input and an output. It
pairs a tensor `name` with its [`DType`](@ref), its `shape` (Julia column-major dimensions),
and the backing `data` array. The two-argument form derives `dtype` and `shape` from a typed
host `Array`.
"""
struct NamedTensor
    name::String
    dtype::DType
    shape::Dims
    data::Array
end

# Derive dtype and shape from a typed host array.
NamedTensor(name::AbstractString, data::Array) =
    NamedTensor(String(name), dtype_of(eltype(data)), size(data), data)

"""
    InferRequest

A decoded inference request, the scheduler's unit of work. It names the target model
(`model_name`), the `requested_outputs` the caller wants returned, and the input tensors
(`inputs`, a `Vector{NamedTensor}`). The codec produces it from a wire `ModelInferRequest`;
the scheduler and runtime consume only this transport-agnostic form.
"""
struct InferRequest
    model_name::String
    requested_outputs::Vector{String}
    inputs::Vector{NamedTensor}
end

struct QueuedRequest
    req::InferRequest
    enqueued_at::Float64
    reply::Channel{Any}      # buffered size 1; holds the result or a captured exception
end
QueuedRequest(req::InferRequest) = QueuedRequest(req, time(), Channel{Any}(1))
