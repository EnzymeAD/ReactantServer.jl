module ReactantServerCore

# Shared, Reactant-free substrate for the ReactantServer monorepo: the canonical dtype
# vocabulary, the KServe V2 protobuf messages, transport-agnostic boundary types, the manifest
# parser, server/node config, the wire <-> boundary codec, the server-side shared-memory
# registry, and the concurrency-safe staging BufferPool. The worker, gateway, and client all
# build on this; nothing here depends on Reactant.

using BFloat16s: BFloat16
using DLFP8Types: Float8_E5M2, Float8_E4M3FN
using InterProcessCommunication
using FixedSizeArrays
using ProtoBuf
using YAML

# Canonical dtype vocabulary (no transport, no Reactant).
include("dtypes.jl")

# KServe V2 protobuf messages (ProtoBuf only; gRPC service stubs are package-local).
include("proto/inference/inference.jl")

# ReactantServer ControlService protobuf messages (worker control plane).
include("proto/control/control.jl")

# Shared-memory primitives: aliasing helpers (vendored MMISHM) and the slot-allocated pool.
include("shm_helpers.jl")
include("buffer_pool.jl")

# Host-weight residency store (private, or node-shared via POSIX SHM + flock).
include("weight_store.jl")

# Boundary, manifest, config, node.
include("boundary.jl")
include("manifest.jl")
include("signature.jl")
include("config.jl")
include("node.jl")

# Server-side shared-memory registry and the wire <-> boundary codec.
include("shared_memory.jl")
include("codec.jl")

function __init__()
    _init_shm_naming!()
    return nothing
end

# Paths to the split gRPC service-stub source files. Core ships them but does not compile them
# (so Core stays free of gRPCClient/gRPCServer). Consumer packages `include` the one they need
# into a module that has done `using ReactantServerCore.inference`, so the bare message-type
# references in the stubs resolve.
const _PROTO_DIR = joinpath(@__DIR__, "proto", "inference")
inference_client_stubs_path() = joinpath(_PROTO_DIR, "grpc_client_stubs.jl")
inference_server_stubs_path() = joinpath(_PROTO_DIR, "grpc_server_stubs.jl")

const _CONTROL_PROTO_DIR = joinpath(@__DIR__, "proto", "control")
control_server_stubs_path() = joinpath(_CONTROL_PROTO_DIR, "control_server_stubs.jl")
control_client_stubs_path() = joinpath(_CONTROL_PROTO_DIR, "control_client_stubs.jl")

# ---- dtypes ----
export DType, F16, F32, F64, BF16, F8E5M2, F8E4M3, I8, I16, I32, I64, U8, U16, U32, U64, BOOL
export DTYPE_FROM_TOKEN, DTYPE_TO_TOKEN, DTYPE_TO_JULIA, JULIA_TO_DTYPE, DTYPE_TO_KSERVE, KSERVE_TO_DTYPE
export dtype_from_token, dtype_token, julia_type, dtype_of, dtype_size, kserve_string, dtype_from_kserve
export TritonType, KSERVE_OUTPUT_DTYPE_TABLE, KSERVE_OUTPUT_DTYPE_TABLE_REVERSE

# ---- protobuf modules ----
export inference, control
export inference_client_stubs_path, inference_server_stubs_path
export control_server_stubs_path, control_client_stubs_path

# ---- shared-memory helpers ----
export shm_key, WrappedFArray, WrappedCArray, MemCopySafeArray, memcpy_safe_arr_n_bytes
export memory_from_shm, memory_from_bytes, fsa_from_memory

# ---- buffer pool ----
export BufferPool, PoolSlot, acquire_slot!, release_slot!, subslot, reset_slot!, PoolAcquireTimeout
export pool_view, pool_memory, pool_fsa, is_shm_backed, scratch
export pool_base_pointer, pool_region_name, pool_slot_bytes

# ---- weight store ----
export WeightStore, PrivateWeightStore, SharedWeightStore
export materialize_host_weights!, release_host_weights!, weights_digest

# ---- boundary ----
export NamedTensor, InferRequest, QueuedRequest, DeadlineExceeded

# ---- manifest ----
export ManifestError, DimKind, FIXED, BATCH, VARIABLE, Dim, TensorSpec, BatchingSpec, Provenance, Manifest
export parse_shape, parse_tensor_spec, parse_tensor_list, parse_batching, parse_manifest, validate_manifest
export load_manifest, is_meta
export client_input_spec, client_output_spec

# ---- signature ----
export SignatureValidator, NullSignatureValidator, validate_against_signature

# ---- config ----
export ConfigError, BackendKind, CPU_BACKEND, CUDA_BACKEND
export ResidencyState, UNPINNED, PINNED_SYSTEM, PINNED_DEVICE
export ResidencyMode, SELF_MANAGED, EXTERNALLY_MANAGED
export ModelControlMode, STATIC, DYNAMIC, EXPLICIT
export SchedulingDiscipline, FAIR, FIFO, EDF
export RuntimeConfig, ModelSchedConfig, SchedulerConfig, EndpointsConfig, ServerConfig
export build_config, validate_config, apply_env_overrides!, log_effective_config

# ---- node ----
export load_node, load_node_raw, validate_node, worker_names, worker_raw_config, node_server_config
export node_gpus, materialize_node!

# ---- shared-memory registry ----
export ShmRegion, SharedMemoryRegistry, shm_register!, shm_unregister!, shm_read, shm_write!
export shm_regions, shm_teardown!, same_ipc_namespace

# ---- codec ----
export OutputTarget, DecodedRequest, decode_infer_request, encode_infer_response, encode_model_metadata
export encode_repository_index, encode_shm_status, encode_shm_register_response, encode_shm_unregister_response, id_of
export encode_is_same_ipc_namespace_response
export encode_infer_request, encode_infer_request_shm, decode_infer_response
export deadline_params, TIMEOUT_NS_PARAM

end # module ReactantServerCore
