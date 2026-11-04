# Targeted variant probe for a multi-shape model (e.g. a detector compiled for several image sizes).
#
# Sends one zero-filled request per compiled input shape, largest first, so every compiled variant
# actually executes and allocates its full activation and scratch memory, confirming the largest
# programs fit on the card. The compiled shapes come from the gateway (ControlService/DescribeModels),
# so they always match what is deployed. Run it WHILE watching `nvidia-smi` on the target GPU to see
# the pool grow to each variant's peak; a worker OOM would surface here as a failed request and a
# supervisor restart.
#
# Under edf there is NO execution warmup at startup, only compilation, so the large variants are
# compiled but unexercised until a real request of that shape arrives.
#
# Env knobs:
#   PROBE_GATEWAY   grpc URL of the gateway   (default grpc://localhost:8001)
#   PROBE_MODEL     model to probe            (default text_fuse_net_gpu)
#   PROBE_DEADLINE  per-request seconds       (default 300; first run may compile/autotune)
#
# Run it with the same project as the loadgen (ReactantServerClient pulls in gRPCClient):
#   julia --project=packages/ReactantServerClient tools/loadgen/probe_large_variants.jl

using ReactantServerClient
import gRPCClient

env(k, d) = get(ENV, k, d)
const GATEWAY = env("PROBE_GATEWAY", "grpc://localhost:8001")
const MODEL = env("PROBE_MODEL", "text_fuse_net_gpu")
const DEADLINE = parse(Float64, env("PROBE_DEADLINE", "300"))

const DTYPE = Dict(
    "BOOL" => Bool, "UINT8" => UInt8, "UINT16" => UInt16, "INT8" => Int8, "INT16" => Int16,
    "INT32" => Int32, "INT64" => Int64, "FP32" => Float32, "FP64" => Float64,
)

# One zero-filled item of one compiled shape: (name, element type, Julia column-major dims) per
# input, the batch axis set to 1. The response is kept so its output shape can be printed.
mutable struct ZeroIO <: AbstractInferenceIO
    inputs::Vector{Tuple{String, DataType, Vector{Int}}}
    response::Any
end
Base.length(::ZeroIO) = 1
ReactantServerClient.item_input_bytes(io::ZeroIO) = sum(sizeof(T) * prod(dims) for (_, T, dims) in io.inputs)
function ReactantServerClient.infer_encode_chunk!(io::ZeroIO, r, slot)
    return map(io.inputs) do (name, T, dims)
        nbytes = sizeof(T) * prod(dims)
        sub = subslot(slot, nbytes)
        fill!(pool_view(sub, UInt8, nbytes), 0x00)
        InferInput(name, sub, dims, T)
    end
end
ReactantServerClient.infer_decode_chunk!(io::ZeroIO, r, response) = (io.response = response; nothing)

function main()
    grpc = gRPCClient.gRPCCURL(; sticky = false, max_streams = 1)
    d = only(describe_models(GATEWAY; names = [MODEL], grpc = grpc))
    isempty(d.variants) && (println("$MODEL has no compiled input shapes to probe"); exit(2))
    dtype = Dict(t.name => DTYPE[t.datatype] for t in d.inputs)
    inputs(v) = [(t.name, dtype[t.name], [s == -1 ? 1 : s for s in v[t.name]]) for t in d.inputs]
    # Largest first (by bytes), so a card that cannot hold the biggest program fails immediately.
    order = sort(d.variants; by = v -> -sum(sizeof(T) * prod(dims) for (_, T, dims) in inputs(v)))
    pool_bytes = max(ReactantServerClient.DEFAULT_POOL_BYTES, 2 * sum(sizeof(T) * prod(dims) for (_, T, dims) in inputs(order[1])))
    kserve_init(; pool_bytes = pool_bytes, n_slots = 1)
    m = KServeModel(GATEWAY, MODEL; max_batch_size = 1, deadline = DEADLINE, grpc = grpc, shared_memory = :off)
    println("== probing $MODEL on $GATEWAY: $(length(order)) variants, largest first, deadline $(DEADLINE)s ==")
    nfail = 0
    for v in order
        io = ZeroIO(inputs(v), nothing)
        shape = join(("$name=$(dims)" for (name, _, dims) in io.inputs), " ")
        t0 = time()
        try
            infer_sync(m, io)
            dt = round(time() - t0, digits = 2)
            outs = io.response === nothing ? "" : join(("$(o.name)=$(collect(o.shape))" for o in io.response.outputs), " ")
            println("  OK   $shape  $(dt)s  out_wire_shapes: $outs")
        catch err
            nfail += 1
            println("  FAIL $shape  ", sprint(showerror, err))
        end
        flush(stdout)
    end
    println("== done: $(length(order) - nfail)/$(length(order)) variants ran ==")
    kserve_shutdown()
    return exit(nfail == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
