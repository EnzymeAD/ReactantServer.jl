# The worker's typed request failures. The scheduler and the request path raise these instead of
# plain `ErrorException`s so the transport can give each its own gRPC status by type, never by
# matching message text (see `_grpc_status` in transport/grpc.jl, where the whole mapping lives).
# A client and a gateway act on the status: they back off on RESOURCE_EXHAUSTED, try another
# replica on UNAVAILABLE or NOT_FOUND, give up on DEADLINE_EXCEEDED, and re-register shared memory
# on FAILED_PRECONDITION, so a failure must never carry another failure's code.

"""
    QueueFullError(model, depth)

Raised when a request is rejected at submission because its model's queue already holds
`scheduler.max_queue_depth` requests. The gRPC layer maps it to `RESOURCE_EXHAUSTED`: transient
backpressure, retryable after a backoff.
"""
struct QueueFullError <: Exception
    model::String
    depth::Int
end
Base.showerror(io::IO, e::QueueFullError) =
    print(io, "queue for model '", e.model, "' is full (", e.depth, "); request rejected")

"""
    WorkerUnavailableError(msg)

Raised when the worker cannot serve any request right now: it is shutting down or draining, or its
dispatch loop has died. The gRPC layer maps it to `UNAVAILABLE`, so a gateway fails the request over
to another replica and a client may retry.
"""
struct WorkerUnavailableError <: Exception
    msg::String
end
Base.showerror(io::IO, e::WorkerUnavailableError) = print(io, e.msg)

"""
    WeightsReadStalledError(model, msg)

Raised to a request for a model this worker cannot load now because reading weights stalled: the
background read of the model's weights made no progress for `scheduler.weight_read_stall_seconds`
and was abandoned, an earlier read of them stalled and has not returned, or so many reads have
stalled that the worker starts no new ones. Models already on the device keep serving. The gRPC
layer maps it to `UNAVAILABLE`, so a gateway fails the request over to another replica and a
client may retry.
"""
struct WeightsReadStalledError <: Exception
    model::String
    msg::String
end
Base.showerror(io::IO, e::WeightsReadStalledError) = print(io, e.msg)

"""
    ModelNotFoundError(model[, msg])

Raised when a request names a model this worker does not serve: it was never loaded, or it was
unloaded or replaced while the request waited. The gRPC layer maps it to `NOT_FOUND`, which a
gateway treats as a stale route (it refreshes its routing table and tries another replica).
"""
struct ModelNotFoundError <: Exception
    model::String
    msg::String
end
ModelNotFoundError(model::AbstractString) = ModelNotFoundError(String(model), "unknown model: $model")
Base.showerror(io::IO, e::ModelNotFoundError) = print(io, e.msg)

"""
    DispatchTimeoutError(model, seconds)

Raised to a caller that waited `scheduler.dispatch_timeout_seconds` for its request without a
result. The request is withdrawn: if it is still queued it never runs, and if it is already
executing its result is discarded. The gRPC layer maps it to `DEADLINE_EXCEEDED`.
"""
struct DispatchTimeoutError <: Exception
    model::String
    seconds::Float64
end
Base.showerror(io::IO, e::DispatchTimeoutError) =
    print(
    io, "request for model '", e.model, "' did not complete within the dispatch timeout (",
    e.seconds, " s); it was withdrawn"
)

"""
    RequestCancelledError(model)

Raised to a caller whose client cancelled the call while the request waited. The request is
withdrawn exactly as on a dispatch timeout. The gRPC layer maps it to `CANCELLED`.
"""
struct RequestCancelledError <: Exception
    model::String
end
Base.showerror(io::IO, e::RequestCancelledError) =
    print(io, "request for model '", e.model, "' was cancelled by the client; it was withdrawn")

"""
    InvalidInputError(msg)

Throw this from a bundle's `preprocess` hook (see [`register_model`](@ref)) to reject a request
whose data the hook cannot accept, such as text that is not valid UTF-8 or a length tensor that
does not match its batch. The gRPC layer maps it to `INVALID_ARGUMENT`, so the client is told
the request itself is wrong and the gateway does not retry it on another replica. Any other
exception a hook throws is a server-side failure and maps to `INTERNAL`. [`set_policy!`](@ref)
raises it too, for a fair weight that is not positive and finite.
"""
struct InvalidInputError <: Exception
    msg::String
end
Base.showerror(io::IO, e::InvalidInputError) = print(io, e.msg)
