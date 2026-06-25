"""
    AbstractInferenceIO

Interface for streaming a dataset through batched inference with [`infer_async`](@ref) /
[`infer_sync`](@ref). A concrete subtype implements `length(io)`, `item_input_bytes(io)`,
`infer_encode_chunk!(io, range, slot)` (stage a chunk's inputs into the pool slot and return the input
descriptors), and `infer_decode_chunk!(io, range, response)` (consume the response). The pool
owns the staging bytes; the IO must not retain slot references past `infer_encode_chunk!`.

A subtype may optionally implement [`output_specs`](@ref) to read its outputs back through shared
memory. The default is empty, which keeps every output inline (the response carries
`raw_output_contents` as before). Declaring outputs is transparent to `infer_decode_chunk!`:
the driver reads the shared-memory results back into the response before handing it over, so
`InferOutput` works the same on either transport.
"""
abstract type AbstractInferenceIO end
#=
AbstractInferenceIO children *MUST* implement:

    infer_encode_chunk!(io, r::UnitRange, slot::PoolSlot) ::Union{PoolInferInput, AbstractVector{PoolInferInput}}
        Carve subslots from `slot`, write the inputs for items `r` into them,
        and return the network input descriptors. Return a single PoolInferInput
        for a single-input model (e.g. straight from `scratch(slot, name, dims, T)`),
        or a vector of them for multiple inputs. The pool owns the bytes; the IO
        must not retain references to the slot or subslots past this call's return.

    item_input_bytes(io) ::Int
        Bytes consumed per item, summed across every input the IO stages.
        The driver uses this to derive chunk size and slot count.

    infer_decode_chunk!(io, r::UnitRange, response::ModelInferResponse) ::Nothing
        Process the network's response for items `r`.

    length(io) ::Int
        Total number of items to infer over.

AbstractInferenceIO children *MAY* implement:

    output_specs(io) ::Vector{OutputSpec}
        Declare expected outputs to read them back through shared memory. Defaults to empty,
        which leaves outputs inline (current behavior). When non-empty, the request asks the
        server for exactly these outputs, so list every output the IO consumes; outputs with
        dynamic shapes cannot be sized and must stay inline.
=#
model(x::AbstractInferenceIO) = x.impl

abstract type AbstractInferenceModel end

function parse_grpc_url(url::String)
    url = lowercase(url)
    parts = split(url, ":")
    _bad() = error("invalid gRPC URL '$url'; expected [grpc[s]|http[s]://]host:port")
    function _port(s)
        p = tryparse(UInt16, s)
        p === nothing && _bad()
        return p
    end

    host, port, secure = if length(parts) == 3
        # "protocol://host:port"
        prefix, host, port = parts
        prefix in ("http", "https", "grpc", "grpcs") || _bad()
        host = replace(host, "/" => "")
        secure = prefix in ("https", "grpcs")
        host, _port(port), secure
    elseif length(parts) == 2
        # "host:port"
        host, port = parts
        host, _port(port), false
    else
        _bad()
    end

    host, port, secure
end

# Per-message wire caps. The 4 MiB gRPC default is far below what inference tensors routinely
# need (a single 768x768x4 f32 output is ~9.4 MB), and the rest of the stack already sizes for
# this: the worker router allows 512 MiB and the gateway 256 MiB. Match the gateway, the
# endpoint clients normally talk to. These are decode-time caps, not allocations; raising them
# costs nothing until a message that large actually arrives.
const DEFAULT_MAX_MESSAGE_BYTES = 256 * 1024 * 1024

"""
    KServeModel(host, port, model_name; secure=false, max_batch_size=1, deadline=10.0, ...)
    KServeModel(url, model_name; max_batch_size=1, deadline=10.0, ...)

A handle to one model served by a KServe V2 gRPC endpoint (a `ReactantServer` worker or the
gateway). The second form parses a `url` of the form `grpc://host:port` (or `host:port`);
`grpcs`/`https` selects a secure channel. `max_batch_size` caps how many items the batched
[`infer_async`](@ref) / [`infer_sync`](@ref) drivers coalesce per request; `deadline` is the
per-request timeout in seconds. `max_send_message_length` / `max_receive_message_length`
bound a single gRPC message (default 256 MiB, matching the gateway's limits).

`shared_memory` controls system shared-memory transport for staged inputs/outputs:
- `:auto` (default): probe the server with `IsSameIPCNamespace`; use shared memory only if the
  server confirms it shares this client's IPC namespace. If the server returns false, or does
  not implement the RPC, fall back to inline transport. There is no silent runtime fallback.
- `:on`: force shared memory. The server confirming a different namespace is a hard error (fail
  loudly, no fallback). If the server does not implement `IsSameIPCNamespace` (e.g. stock
  Triton), shared memory is still attempted via `SystemSharedMemoryRegister`; making that work
  is then the caller's responsibility.
- `:off`: never use shared memory; always send inline. The probe is not sent.
"""
struct KServeModel <: AbstractInferenceModel
    host::String
    port::UInt16
    secure::Bool
    model_name::String
    max_batch_size::Int64
    deadline::Float64
    max_send_message_length::Int64
    max_receive_message_length::Int64
    shared_memory::Symbol
    # The gRPCCURL handle (libcurl multi handle + connection pool + concurrent-stream semaphore) all
    # of this model's client calls share. Defaults to the process-global handle (GRPC_MAX_STREAMS=16
    # concurrent requests). Pass a dedicated handle with a larger `max_streams` to drive more
    # concurrency than 16 (e.g. a load generator sizing it to its request concurrency).
    grpc::gRPCClient.gRPCCURL

    function KServeModel(
        host,
        port,
        model_name;
        secure = false,
        max_batch_size = 1,
        deadline = 10.0,
        max_send_message_length = DEFAULT_MAX_MESSAGE_BYTES,
        max_receive_message_length = DEFAULT_MAX_MESSAGE_BYTES,
        shared_memory = :auto,
        grpc = gRPCClient.grpc_global_handle(),
    )
        shared_memory in (:auto, :on, :off) ||
            throw(ArgumentError("shared_memory must be :auto, :on, or :off, got $(repr(shared_memory))"))
        new(
            host,
            port,
            secure,
            model_name,
            max_batch_size,
            deadline,
            max_send_message_length,
            max_receive_message_length,
            shared_memory,
            grpc,
        )
    end

    function KServeModel(
        url,
        model_name;
        max_batch_size = 1,
        deadline = 10.0,
        max_send_message_length = DEFAULT_MAX_MESSAGE_BYTES,
        max_receive_message_length = DEFAULT_MAX_MESSAGE_BYTES,
        shared_memory = :auto,
        grpc = gRPCClient.grpc_global_handle(),
    )
        host, port, secure = parse_grpc_url(url)

        KServeModel(
            host,
            port,
            model_name;
            secure = secure,
            max_batch_size = max_batch_size,
            max_send_message_length = max_send_message_length,
            max_receive_message_length = max_receive_message_length,
            deadline = deadline,
            shared_memory = shared_memory,
            grpc = grpc,
        )
    end
end

model_name(x::KServeModel) = x.model_name
max_batch_size(x::KServeModel) = x.max_batch_size
deadline(x::KServeModel) = x.deadline
timeout(x::KServeModel) = deadline(x)

function grpc_infer_client(x::KServeModel)
    GRPCInferenceService_ModelInfer_Client(
        x.host,
        x.port;
        secure = x.secure,
        deadline = x.deadline,
        max_send_message_length = x.max_send_message_length,
        # The generated client stub's keyword spells it 'recieve'; the public field does not.
        max_recieve_message_length = x.max_receive_message_length,
        grpc = x.grpc,
    )
end

function grpc_shm_unregister_client(x::KServeModel)
    GRPCInferenceService_SystemSharedMemoryUnregister_Client(
        x.host,
        x.port;
        secure = x.secure,
        grpc = x.grpc,
    )
end

function grpc_shm_register_client(x::KServeModel)
    GRPCInferenceService_SystemSharedMemoryRegister_Client(
        x.host,
        x.port;
        secure = x.secure,
        grpc = x.grpc,
    )
end

function grpc_is_same_ipc_namespace_client(x::KServeModel)
    GRPCInferenceService_IsSameIPCNamespace_Client(
        x.host,
        x.port;
        secure = x.secure,
        grpc = x.grpc,
    )
end

function grpc_metadata_client(x::KServeModel)
    GRPCInferenceService_ModelMetadata_Client(
        x.host,
        x.port;
        secure = x.secure,
        deadline = x.deadline,
        grpc = x.grpc,
    )
end
