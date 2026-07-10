# Targeted variant probe for the multi-shape detection models (text_fuse_net_gpu).
#
# The generic loadgen synthesizes inputs from the manifest and collapses every -1 (variable) axis,
# so it cannot drive a multi-shape model: text_fuse_net's input is {w: -1, h: -1, c: 1} and the
# loadgen would send a degenerate 1x1 image that matches no compiled variant. This script instead
# sends one correctly-sized zero image per declared input shape, so every compiled stage1 variant
# actually executes (allocates its full activation/scratch), confirming the largest programs fit on
# the card. Run it WHILE watching `nvidia-smi` on the target GPU to see the pool grow to each
# variant's peak; a worker OOM would surface here as a failed request and a supervisor restart.
#
# Under edf (this node's discipline) there is NO execution warmup at startup, only compilation, so
# the large variants are compiled but unexercised until a real request of that shape arrives.
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
const GATEWAY  = env("PROBE_GATEWAY", "grpc://localhost:8001")
const MODEL    = env("PROBE_MODEL", "text_fuse_net_gpu")
const DEADLINE = parse(Float64, env("PROBE_DEADLINE", "300"))

# The compiled (W, H) variants, kept in sync with input_shapes in private/convert.yaml. Probed
# largest-first (by pixel count) so a card that cannot hold the biggest program fails immediately.
const SHAPES = [
    (1408, 2816), (1664, 2432), (2432, 1664), (2816, 1408), (1984, 1984),  # ~4 MP: the heavy tier
    (832, 1216), (1216, 832), (704, 1408), (1024, 1024),                   # ~1 MP
    (384, 640), (384, 704), (640, 384), (704, 384), (512, 512),            # small
]

function main()
    kserve_init()
    grpc = gRPCClient.gRPCCURL(; sticky = false, max_streams = 1)
    order = sort(SHAPES; by = wh -> -(wh[1] * wh[2]))
    println("== probing $MODEL on $GATEWAY: $(length(order)) variants, largest first, deadline $(DEADLINE)s ==")
    nfail = 0
    for (w, h) in order
        m = KServeModel(GATEWAY, MODEL; max_batch_size = 1, deadline = DEADLINE, grpc = grpc)
        img = zeros(Float32, w, h, 1)   # (W, H, 1) Julia col-major == manifest whc
        t0 = time()
        try
            resp = infer_sync(m, [InferInput("INPUT__0", img)])
            dt = round(time() - t0, digits = 2)
            wire = isempty(resp.outputs) ? Int[] : collect(resp.outputs[1].shape)
            println("  OK   $(w)x$(h)  ($(w*h) px)  $(dt)s  out_wire_shape=$wire")
        catch err
            nfail += 1
            println("  FAIL $(w)x$(h)  ($(w*h) px)  ", sprint(showerror, err))
        end
        flush(stdout)
    end
    println("== done: $(length(order) - nfail)/$(length(order)) variants ran ==")
    kserve_shutdown()
    exit(nfail == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
