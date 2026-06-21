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
    DeadlineExceeded(model_name)

Raised when a request's deadline has already passed at dispatch admission: the scheduler
refuses to *begin* GPU work that is already expired, and a meta orchestration refuses to issue
a further sub-call once its budget is gone. It never interrupts a running PJRT/GPU call; it only
declines to start new work. The gRPC layer maps it to `DEADLINE_EXCEEDED`.
"""
struct DeadlineExceeded <: Exception
    model_name::String
end
Base.showerror(io::IO, e::DeadlineExceeded) =
    print(io, "deadline exceeded before dispatch for model '", e.model_name, "'")

"""
    InferRequest

A decoded inference request, the scheduler's unit of work. It names the target model
(`model_name`), the `requested_outputs` the caller wants returned, and the input tensors
(`inputs`, a `Vector{NamedTensor}`). `deadline_ns` is an absolute local `time_ns()` deadline
(0 means none): a remaining-budget timeout carried over the wire is converted to this local
absolute form at decode, so cross-process monotonic-clock differences never matter. The codec
produces it from a wire `ModelInferRequest`; the scheduler and runtime consume only this
transport-agnostic form.
"""
struct InferRequest
    model_name::String
    requested_outputs::Vector{String}
    inputs::Vector{NamedTensor}
    deadline_ns::Int64
end

# Deadline defaults to 0 (none): the form the scheduler unit tests and most in-process callers
# build. The codec and meta sub-call path pass an explicit deadline.
InferRequest(model_name::AbstractString, requested_outputs::Vector{String},
             inputs::Vector{NamedTensor}) =
    InferRequest(String(model_name), requested_outputs, inputs, Int64(0))

struct QueuedRequest
    req::InferRequest
    prepared::Vector{NamedTensor}   # preprocess(req.inputs), the executable-ready inputs;
                                    # computed on the caller's task before the request is queued
    enqueued_at::Float64
    reply::Channel{Any}      # buffered size 1; holds the raw sliced outputs or a captured exception
end
# `prepared` defaults to the request's own inputs (the identity-preprocess case, and the form the
# scheduler unit tests build). The caller passes the preprocessed inputs explicitly when a
# bundle's `preprocess` hook is non-trivial; the dispatch loop coalesces and executes `prepared`,
# never re-running the hook.
QueuedRequest(req::InferRequest, prepared::Vector{NamedTensor}=req.inputs) =
    QueuedRequest(req, prepared, time(), Channel{Any}(1))
