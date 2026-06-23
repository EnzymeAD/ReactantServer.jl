module ReactantServerClient

# KServe V2 inference client. Builds requests against ReactantServerCore's protobuf messages
# and dtype vocabulary, stages tensor data through the concurrency-safe BufferPool from Core,
# and forwards over gRPC. Ported from the (unpublished) SimpleKServe.jl; the shared-memory
# pool/slot machinery and dtype tables now come from ReactantServerCore. This package never
# depends on Reactant.

using Base.Threads
using ProtoBuf
using ResumableFunctions
using InterProcessCommunication
using FixedSizeArrays
using gRPCClient

using ReactantServerCore
using ReactantServerCore.inference   # KServe message types in scope for the client stubs

# These Core functions get new methods below (on InferenceBufferPool, and the client `scratch`
# overload returning wire descriptors), so they must be imported to extend rather than shadow them.
import ReactantServerCore: acquire_slot!, is_shm_backed, scratch, pool_view

import Base: length, sizeof, rm, elsize

# gRPC client service stubs (define GRPCInferenceService_*_Client). Core ships the file but
# does not compile it; included here so its bare message-type references resolve and
# `import gRPCClient` runs against this package's deps.
include(ReactantServerCore.inference_client_stubs_path())

const DEFAULT_POOL_BYTES = 256 * 1024 * 1024  # 256 MiB
const DEFAULT_POOL_SLOTS = 8                   # fixed slots per pool (the allocator's parallelism)

include("Defines.jl")
include("Model.jl")
include("SharedMemory.jl")
include("Inference.jl")
include("Metadata.jl")

"""
    kserve_init(; pool_bytes=DEFAULT_POOL_BYTES, n_slots=DEFAULT_POOL_SLOTS)

Initialize the gRPC subsystem and (re)set the staging-pool parameters. `n_slots` is the number
of fixed-size slots each pool is divided into, which bounds how many chunks can be in flight
concurrently against a single pool.
"""
function kserve_init(; pool_bytes::Integer = DEFAULT_POOL_BYTES,
                            n_slots::Integer = DEFAULT_POOL_SLOTS)
    grpc_init()
    @lock _pools_lock begin
        _teardown_shm_pool!()
        _pool_bytes[] = Int(pool_bytes)
        _pool_slots[] = Int(n_slots)
        _inline_pool[] = nothing
        empty!(_pool_routes)
        empty!(_route_locks)
    end
    nothing
end

"""
    kserve_shutdown()

Tear down the client: unregister and unlink the shared-memory pool from every server it was
registered with, drop the cached pools and per-URL routes, and shut the gRPC subsystem down.
Pair with [`kserve_init`](@ref).
"""
function kserve_shutdown()
    @lock _pools_lock begin
        _teardown_shm_pool!()
        _inline_pool[] = nothing
        empty!(_pool_routes)
        empty!(_route_locks)
    end
    grpc_shutdown()
end

export kserve_init
export kserve_shutdown

export AbstractInferenceModel
export AbstractInferenceIO

export KServeModel
export InferInput, InferOutput

export infer_async
export infer_sync
export infer_decode_chunk!
export model_name

export InferenceBufferPool, PoolSlot, PoolInferInput
export subslot, pool_view, pool_memory, pool_fsa, scratch, item_input_bytes, infer_encode_chunk!
export OutputSpec, output_specs, item_output_bytes
export acquire_slot!, release_slot!

export ModelIOSpec, TensorMeta
export model_io_spec, manifest_io_spec, validate_io
export with_bounds_checks, @infer_inbounds

export DEFAULT_POOL_BYTES
export DEFAULT_POOL_SLOTS

export triton_unregister_shm

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        # Warm the pool/materialization paths for the common dtypes so callers that write
        # through the FSA-on-Memory{T} signature in infer_encode_chunk! get a precompiled
        # materialization. An inline (Memory{UInt8}-backed) pool keeps this hermetic: no SHM
        # region and no network. End-to-end inference is exercised by the test suite, not here,
        # so precompilation never depends on a reachable server.
        _pc_pool = InferenceBufferPool(1024; n_slots = 4, use_shm = false)
        _pc_slot = acquire_slot!(_pc_pool)
        pool_memory(_pc_slot, UInt8, 256)
        pool_fsa(_pc_slot, UInt8, (16, 16))
        reset_slot!(_pc_slot)
        pool_memory(_pc_slot, Float32, 16)
        pool_fsa(_pc_slot, Float32, (4, 4))
        sub = subslot(_pc_slot, 64)
        pool_view(sub, UInt8, 64)
        release_slot!(_pc_slot)
    end
end

end # module ReactantServerClient
