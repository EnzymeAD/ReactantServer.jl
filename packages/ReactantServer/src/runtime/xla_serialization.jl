# Executable serialization and allocator-statistics control, vendored from EnzymeAD/Reactant.jl#3305.
#
# The C++ half of this API (EnzymeAD/Reactant.jl#3277) is merged and shipped: Reactant_jll 0.0.410 is
# the first build that exports PjRtLoadedExecutableSerialize, PjRtClientLoadSerializedExecutable,
# PjRtDeviceClearMemoryStats, PjRtLoadedExecutableGetCompiledMemoryStats and their IFRT
# counterparts, and the [compat] floor in Project.toml guarantees one. The Julia half (#3305) is still
# in review, so this module carries it until a Reactant release does.
#
# It is a port, not a copy, in two ways that keep it safe to leave in place when #3305 lands:
#
#   - The functions live HERE, not as methods of Reactant.XLA's. Defining `XLA.serialize_executable`
#     from this package would be type piracy, and once Reactant defines the same methods it would
#     overwrite them (and break precompilation).
#   - The C functions are called directly with @ccall rather than through Reactant.MLIR.API, whose
#     generated wrappers for them first appear in Reactant 0.2.288. Calling the symbols ties this file
#     to the JLL, which is the real requirement, and not to a regeneration of Reactant's bindings.
#
# Retiring it once a Reactant release carries #3305: raise the Reactant floor to that release, point
# `_XS` in reactant_backend.jl at `Reactant.XLA`, and delete this file.
module VendoredXLA

using Reactant: Reactant
using Reactant_jll: Reactant_jll

const XLA = Reactant.XLA
const PJRT = Reactant.XLA.PJRT
const IFRT = Reactant.XLA.IFRT

"""
    CompiledMemoryStats

The memory XLA's buffer assignment reserved for an executable, in bytes (see
[`compiled_memory_stats`](@ref)). Field-for-field the C struct `JLCompiledMemoryStats` in
`deps/ReactantExtra/API.cpp`, which is why it can be passed to the C side by reference.
"""
struct CompiledMemoryStats
    generated_code_size_in_bytes::Int64
    argument_size_in_bytes::Int64
    output_size_in_bytes::Int64
    alias_size_in_bytes::Int64
    temp_size_in_bytes::Int64
    host_generated_code_size_in_bytes::Int64
    host_argument_size_in_bytes::Int64
    host_output_size_in_bytes::Int64
    host_alias_size_in_bytes::Int64
    host_temp_size_in_bytes::Int64
    peak_memory_in_bytes::Int64
end

_serialized_compile_options(::Nothing) = UInt8[]
_serialized_compile_options(opts::Reactant.Proto.xla.CompileOptionsProto) =
    Reactant.ProtoUtils.proto_to_bytes(opts)

"""
    serialize_executable(exec) -> Vector{UInt8}

Serialize a compiled PJRT or IFRT executable into bytes that [`load_serialized_executable`](@ref)
turns back into an executable, in this process or a later one, without compiling again.

The bytes load only on the same platform, the same `Reactant_jll` build and the same kind of device
(on GPUs, the same compute capability), and only through the runtime that produced them. XLA checks
the compute capability when it loads a GPU executable but not the XLA, CUDA or cuDNN versions, so a
store of these bytes must key on those itself (executable_cache.jl does).
"""
function serialize_executable(exec::PJRT.LoadedExecutable)
    size = Ref{Csize_t}(0)
    data = GC.@preserve exec begin
        @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableSerialize(
            exec.exec::Ptr{Cvoid}, size::Ptr{Csize_t}
        )::Ptr{UInt8}
    end
    # malloc'd by the C++ side; the array owns the buffer and frees it when collected.
    return unsafe_wrap(Array, data, (Int(size[]),); own = true)
end

function serialize_executable(exec::IFRT.LoadedExecutable)
    size = Ref{Csize_t}(0)
    data = GC.@preserve exec begin
        @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_serialize(
            exec.exec::Ptr{Cvoid}, size::Ptr{Csize_t}
        )::Ptr{UInt8}
    end
    return unsafe_wrap(Array, data, (Int(size[]),); own = true)
end

"""
    load_serialized_executable(
        client, serialized::Vector{UInt8};
        compile_options = nothing, num_parameters, num_outputs,
        is_sharded = false, num_replicas = 1, num_partitions = 1,
    )

Load an executable serialized with [`serialize_executable`](@ref). `compile_options` (a
`CompileOptionsProto`, see `Reactant.XLA.make_compile_options`) replaces the options stored with the
executable, which is how a program compiled for one device ordinal is placed on another; `nothing`
keeps the stored ones. The remaining keywords describe the program and are not part of the bytes:
pass the values the original executable was created with. Invalid bytes throw
`Reactant.XLA.ReactantInternalError`.
"""
function load_serialized_executable(
        client::PJRT.Client, serialized::Vector{UInt8};
        compile_options::Union{Nothing, Reactant.Proto.xla.CompileOptionsProto} = nothing,
        num_parameters::Int64, num_outputs::Int64,
        is_sharded::Bool = false, num_replicas::Int64 = 1, num_partitions::Int64 = 1,
    )
    opts = _serialized_compile_options(compile_options)
    exec = GC.@preserve client serialized opts begin
        @ccall Reactant_jll.libReactantExtra.PjRtClientLoadSerializedExecutable(
            client.client::Ptr{Cvoid}, serialized::Ptr{UInt8}, length(serialized)::Csize_t,
            opts::Ptr{UInt8}, length(opts)::Csize_t,
        )::Ptr{Cvoid}
    end
    return PJRT.LoadedExecutable(exec, num_outputs, num_parameters, is_sharded, num_replicas, num_partitions)
end

function load_serialized_executable(
        client::IFRT.Client, serialized::Vector{UInt8};
        compile_options::Union{Nothing, Reactant.Proto.xla.CompileOptionsProto} = nothing,
        num_parameters::Int64, num_outputs::Int64,
        is_sharded::Bool = false, num_replicas::Int64 = 1, num_partitions::Int64 = 1,
    )
    opts = _serialized_compile_options(compile_options)
    exec = GC.@preserve client serialized opts begin
        @ccall Reactant_jll.libReactantExtra.ifrt_client_load_serialized_executable(
            client.client::Ptr{Cvoid}, serialized::Ptr{UInt8}, length(serialized)::Csize_t,
            opts::Ptr{UInt8}, length(opts)::Csize_t,
        )::Ptr{Cvoid}
    end
    return IFRT.LoadedExecutable(exec, num_outputs, num_parameters, is_sharded, num_replicas, num_partitions)
end

"""
    clear_memory_stats!(device)

Reset the allocator high-water marks reported by `Reactant.XLA.allocatorstats` (`peak_bytes_in_use`,
`peak_bytes_reserved`, `peak_pool_bytes`) to the current usage, so the peak reached by the code that
follows can be measured on its own. Only devices whose allocator keeps statistics (CUDA, ROCm, TPU)
support it; on the CPU device it throws.
"""
function clear_memory_stats!(device::PJRT.Device)
    GC.@preserve device begin
        @ccall Reactant_jll.libReactantExtra.PjRtDeviceClearMemoryStats(device.device::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

function clear_memory_stats!(device::IFRT.Device)
    GC.@preserve device begin
        @ccall Reactant_jll.libReactantExtra.ifrt_device_clear_memory_stats(device.device::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

"""
    compiled_memory_stats(exec) -> CompiledMemoryStats

The memory XLA's buffer assignment reserved for a compiled executable, available without running it.
`temp_size_in_bytes` is the scratch the program needs; the runtime allocator rounds allocations up
and may add its own, so it is a lower bound on the scratch actually allocated at run time.
"""
function compiled_memory_stats(exec::PJRT.LoadedExecutable)
    ref = Ref{CompiledMemoryStats}()
    GC.@preserve exec begin
        @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableGetCompiledMemoryStats(
            exec.exec::Ptr{Cvoid}, ref::Ptr{CompiledMemoryStats}
        )::Cvoid
    end
    return ref[]
end

function compiled_memory_stats(exec::IFRT.LoadedExecutable)
    ref = Ref{CompiledMemoryStats}()
    GC.@preserve exec begin
        @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_get_compiled_memory_stats(
            exec.exec::Ptr{Cvoid}, ref::Ptr{CompiledMemoryStats}
        )::Cvoid
    end
    return ref[]
end

end # module VendoredXLA
