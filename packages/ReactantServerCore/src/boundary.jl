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
transport-agnostic form. `slot_control` is the client's shared-memory slot control for the
request (see [`SlotControl`](@ref)), or `nothing`; the scheduler consults it at dispatch through
[`request_cancelled`](@ref).
"""
struct InferRequest
    model_name::String
    requested_outputs::Vector{String}
    inputs::Vector{NamedTensor}
    deadline_ns::Int64
    slot_control::Union{SlotControl, Nothing}
end

# No slot control: every in-process caller (the scheduler unit tests, meta sub-calls).
InferRequest(
    model_name::AbstractString, requested_outputs::Vector{String},
    inputs::Vector{NamedTensor}, deadline_ns::Integer
) = InferRequest(String(model_name), requested_outputs, inputs, Int64(deadline_ns), nothing)

# Deadline defaults to 0 (none): the form the scheduler unit tests and most in-process callers
# build. The codec and meta sub-call path pass an explicit deadline.
InferRequest(
    model_name::AbstractString, requested_outputs::Vector{String},
    inputs::Vector{NamedTensor}
) =
    InferRequest(String(model_name), requested_outputs, inputs, Int64(0))

"""
    request_cancelled(req::InferRequest) -> Bool

Whether `req`'s client has given up on it, as its shared-memory slot control word shows (CORE-3):
the predicate the scheduler checks before starting a request's work. Never waits; `false` for a
request without slot control.
"""
request_cancelled(req::InferRequest) =
    req.slot_control !== nothing && slot_control_cancelled(req.slot_control)

struct QueuedRequest
    req::InferRequest
    prepared::Vector{NamedTensor}   # preprocess(req.inputs), the executable-ready inputs;
    # computed on the caller's task before the request is queued
    enqueued_at::Float64
    reply::Channel{Any}      # buffered; holds the raw sliced outputs or a captured exception
    committed::Bool          # a meta's in-flight continuation sub-call: jumps the queue for the next
    # GPU slot and is exempt from the EDF laxity drop (but not the base
    # deadline-passed drop). False for every client-originated request.
    withdrawn::Threads.Atomic{Bool}   # set by a caller that stopped waiting for the reply (its
    # dispatch timeout passed, or its client cancelled): the
    # dispatch loop drops the request instead of running it
end
# `prepared` defaults to the request's own inputs (the identity-preprocess case, and the form the
# scheduler unit tests build). The caller passes the preprocessed inputs explicitly when a
# bundle's `preprocess` hook is non-trivial; the dispatch loop coalesces and executes `prepared`,
# never re-running the hook. `committed` defaults false: only the meta sub-call path sets it. The
# reply channel holds two values: the scheduler puts exactly one reply, and a caller that stops
# waiting puts its own wake-up beside it, so neither side can ever block on the other.
QueuedRequest(req::InferRequest, prepared::Vector{NamedTensor} = req.inputs; committed::Bool = false) =
    QueuedRequest(req, prepared, time(), Channel{Any}(2), committed)
# Four-positional form (scheduler unit tests, fixtures) defaults to a non-committed request.
QueuedRequest(
    req::InferRequest, prepared::Vector{NamedTensor}, enqueued_at::Float64,
    reply::Channel{Any}
) = QueuedRequest(req, prepared, enqueued_at, reply, false)
# Five-positional form: a request nobody has withdrawn.
QueuedRequest(
    req::InferRequest, prepared::Vector{NamedTensor}, enqueued_at::Float64,
    reply::Channel{Any}, committed::Bool
) = QueuedRequest(req, prepared, enqueued_at, reply, committed, Threads.Atomic{Bool}(false))
