module ReactantServer

# The Reactant-backed inference worker. The shared substrate (dtypes, boundary, manifest,
# config, cluster, codec, the shared-memory registry, and the protobuf messages) lives in
# ReactantServerCore; this package adds the model registry, the runtime (the only Reactant
# consumer), the scheduler, and the KServe V2 gRPC server.

using ReactantServerCore
using ReactantServerCore.inference   # message types in scope for the server gRPC stubs
using ReactantServerCore.control     # control-plane message types for the ControlService stubs

# Re-expose ReactantServerCore's public API through ReactantServer, so the shared substrate is
# reachable as `ReactantServer.X` (and unqualified) exactly as it was before the monorepo split.
# Every Core symbol is defined in Core alone, so none of these collide with worker definitions.
for _n in names(ReactantServerCore)
    _n === :ReactantServerCore && continue
    @eval import ReactantServerCore: $_n
    @eval export $_n
end

using YAML
using SafeTensors
using JSON3
using BFloat16s
using DLFP8Types
using ProtoBuf

# Server-side gRPC service stubs (define register_GRPCInferenceService! and the per-RPC
# Method helpers). Core ships the file but does not compile it; included here so its bare
# message-type references resolve and `import gRPCServer` runs against this package's deps.
import gRPCServer
include(ReactantServerCore.inference_server_stubs_path())
include(ReactantServerCore.control_server_stubs_path())

# Client-side gRPC service stubs (define the per-RPC `_Client` constructors). The worker is a
# client only for the meta-model loopback path (a GatewayCaller calling back into the gateway).
import gRPCClient
include(ReactantServerCore.inference_client_stubs_path())

# Per-model value types (defined before the registry so ModelEntry can hold them precisely).
include("runtime/model_types.jl")

# Model registry and bundle loading.
include("registry.jl")
include("bundle.jl")

# Runtime. reactant_backend.jl is the only file that imports Reactant.
include("runtime/backend.jl")
include("runtime/mock_backend.jl")
include("runtime/memory_pool.jl")
include("runtime/weights.jl")
include("runtime/model.jl")
include("runtime/weight_cache.jl")
include("runtime/execution.jl")
include("runtime/reactant_backend.jl")
# Load-time TF32 stripping for portable artifacts; uses the backend's _RMLIR/_RXLA aliases.
include("runtime/tf32.jl")

# Lifecycle observability helpers (formatting + structured load/unload/residency logs). After the
# backend (for device_memory_stats) and weight_cache (for weight_cache_stats); before scheduler.
include("runtime/observe.jl")

include("scheduler.jl")

# Dynamic model-directory watching (opt-in via model_poll_seconds). Defined after the scheduler
# (it drives load_model!/evict!) and before server.jl (RunningServer holds a BundleWatcher).
include("watcher.jl")

# Self-contained detection glue (anchors/decode/NMS/roi_align) for two-stage detector meta models.
# Referenced from a bundle's model.jl as ReactantServer.DetectionGlue. No deps on the runtime above.
include("postprocess/detection.jl")

# Meta-model execution (the ModelCaller abstraction + run_meta). After the scheduler (LocalCaller
# wraps it) and before grpc.jl (InferContext holds a ModelCaller and _handle_infer dispatches meta).
include("meta.jl")

# Transport assembly: the gRPC control plane (the codec and shared-memory registry come from
# ReactantServerCore) and top-level server. Worker Prometheus metrics are defined before grpc.jl
# (InferContext holds a WorkerMetrics) and reuse the scheduler/observe snapshot functions above.
include("transport/metrics.jl")
include("transport/grpc.jl")
include("transport/control_grpc.jl")
include("server.jl")

export serve, serve_worker, stop!, register_model, register_meta_model

end # module ReactantServer
