# The PJRT C API, driven directly from Julia.
#
# The CUDA build of Reactant_jll's libReactantExtra.so is itself a PJRT GPU plugin: it exports the
# plain C symbol `GetPjrtApi`, which returns the `PJRT_Api` function table. Everything a serving
# runtime needs (client, buffers, compile, execute, memory stats) is reachable through that table,
# plus two things Reactant's own C++ shims do not expose: `PJRT_Executable_Serialize` /
# `PJRT_Executable_DeserializeAndLoad` (the serialized-executable cache) and
# `PJRT_Device_ClearMemoryStats` (resettable allocator high-water mark).
#
# Struct layouts come from Reactant's generated bindings (`src/xla/PJRT/CAPI.jl`), which Reactant
# ships but does not include; loading them from `pkgdir(Reactant)` keeps them in lockstep with the
# installed Reactant. The C API is versioned: `availability()` compares the library's table size and
# API version against the bindings and refuses the whole backend on a mismatch, so a Reactant whose
# bindings lag its JLL (0.2.270 did) falls back to the Reactant backend instead of corrupting memory.
#
# Handles created here are never mixed with Reactant's C++ client objects; the two are different
# representations of the same XLA objects and are not interchangeable.
module PJRTCAPI

using CEnum: CEnum, @cenum
using Libdl: Libdl
import ..Reactant

const BINDINGS_PATH = joinpath(pkgdir(Reactant), "src", "xla", "PJRT", "CAPI.jl")
Base.include_dependency(BINDINGS_PATH)

# Two generated lines are C preprocessor leftovers that are not valid Julia; everything else is
# plain struct and enum definitions.
let lines = filter(readlines(BINDINGS_PATH)) do l
        !occursin("_PJRT_API_STRUCT_FIELD", l) && !occursin("PJRT_NO_DISCARD", l) &&
            !startswith(strip(l), "using CEnum")
    end
    include_string(@__MODULE__, join(lines, '\n'), "CAPI.jl")
end

struct PJRTError <: Exception
    what::String
    code::Int
    msg::String
end
Base.showerror(io::IO, e::PJRTError) = print(io, "PJRT ", e.what, " failed [", e.code, "]: ", e.msg)

# ── Struct helpers ───────────────────────────────────────────────────────────────────────────────

struct_size(::Type{T}) where {T} = Csize_t(Integer(getfield(@__MODULE__, Symbol(nameof(T), :_STRUCT_SIZE))))

# A zero-initialised args struct with `struct_size` filled in and the given fields overridden.
function mk(::Type{T}; kw...) where {T}
    kwd = Dict{Symbol, Any}(kw)
    vals = Vector{Any}(undef, fieldcount(T))
    for (i, fn) in enumerate(fieldnames(T))
        ft = fieldtype(T, i)
        vals[i] = if fn === :struct_size
            struct_size(T)
        elseif haskey(kwd, fn)
            convert(ft, kwd[fn])
        elseif ft <: Ptr || ft === Cstring
            ft(C_NULL)
        elseif ft === Bool
            false
        elseif ft <: Number
            zero(ft)
        elseif ft <: CEnum.Cenum
            ft(0)
        else
            error("PJRTCAPI.mk: no default for field $fn::$ft of $T")
        end
    end
    return T(vals...)
end

# ── The API table ────────────────────────────────────────────────────────────────────────────────

const API = Ref{Ptr{PJRT_Api}}(C_NULL)
const _AVAILABILITY = Ref{Union{Nothing, Tuple{Bool, String}}}(nothing)
const _PLUGIN_INITIALIZED = Ref(false)
const _LOCK = ReentrantLock()

api() = unsafe_load(API[])

# Look up `name` in the table and call it with `args`. The table struct is opaque bytes with a
# generated `getproperty` offset map, so `getproperty`, never `getfield`.
function pj(name::Symbol, args::Ref)
    f = getproperty(api(), name)
    f == C_NULL && throw(PJRTError(String(name), -1, "not implemented by this PJRT plugin"))
    return ccall(f, Ptr{PJRT_Error}, (Ptr{Cvoid},), args)
end

function check(err::Ptr{PJRT_Error}, what::AbstractString)
    err == C_NULL && return nothing
    m = Ref(mk(PJRT_Error_Message_Args; error = err))
    ccall(getproperty(api(), :PJRT_Error_Message), Cvoid, (Ptr{Cvoid},), m)
    msg = unsafe_string(Ptr{UInt8}(m[].message), m[].message_size)
    c = Ref(mk(PJRT_Error_GetCode_Args; error = err))
    ccall(getproperty(api(), :PJRT_Error_GetCode), Ptr{Cvoid}, (Ptr{Cvoid},), c)
    code = Int(Integer(c[].code))
    d = Ref(mk(PJRT_Error_Destroy_Args; error = err))
    ccall(getproperty(api(), :PJRT_Error_Destroy), Cvoid, (Ptr{Cvoid},), d)
    throw(PJRTError(String(what), code, msg))
end

"""
    availability() -> (ok::Bool, reason::String)

Whether the installed libReactantExtra exposes a PJRT C API table this module can drive: the
`GetPjrtApi` symbol exists, and the table's size and API version match the generated bindings.
Cached after the first call. A false result is the signal to fall back to the Reactant backend.
"""
function availability()
    return lock(_LOCK) do
        _AVAILABILITY[] === nothing || return _AVAILABILITY[]
        result = try
            _probe_availability()
        catch err
            (false, sprint(showerror, err))
        end
        _AVAILABILITY[] = result
        return result
    end
end

function _probe_availability()
    Reactant.Reactant_jll.is_available() || return (false, "Reactant_jll is not available on this platform")
    handle = Reactant.Reactant_jll.libReactantExtra_handle
    sym = Libdl.dlsym_e(handle, :GetPjrtApi)
    sym == C_NULL && return (false, "libReactantExtra.so does not export GetPjrtApi")
    p = ccall(sym, Ptr{PJRT_Api}, ())
    p == C_NULL && return (false, "GetPjrtApi returned NULL")
    a = unsafe_load(p)
    want = struct_size(PJRT_Api)
    a.struct_size == want ||
        return (false, "PJRT_Api struct_size $(a.struct_size) != $(want) expected by the bindings (regenerate CAPI.jl for this Reactant_jll)")
    v = a.pjrt_api_version
    (v.major_version == PJRT_API_MAJOR && v.minor_version == PJRT_API_MINOR) ||
        return (false, "PJRT API version $(v.major_version).$(v.minor_version) != bindings $(PJRT_API_MAJOR).$(PJRT_API_MINOR)")
    API[] = p
    return (true, "")
end

function ensure_initialized!()
    ok, reason = availability()
    ok || throw(PJRTError("availability", -1, reason))
    lock(_LOCK) do
        _PLUGIN_INITIALIZED[] && return nothing
        check(pj(:PJRT_Plugin_Initialize, Ref(mk(PJRT_Plugin_Initialize_Args))), "Plugin_Initialize")
        _PLUGIN_INITIALIZED[] = true
    end
    return nothing
end

api_version() = (v = api().pjrt_api_version; (Int(v.major_version), Int(v.minor_version)))

# ── PJRT_NamedValue ──────────────────────────────────────────────────────────────────────────────
#
# {struct_size, extension_start, name, name_size, type (+pad), value union, value_size}: 56 bytes.
# The generated binding models it as opaque bytes; this is the same layout with typed fields.
struct NamedValue
    struct_size::Csize_t
    ext::Ptr{Cvoid}
    name::Ptr{UInt8}
    name_size::Csize_t
    type::UInt32
    _pad::UInt32
    value::UInt64
    value_size::Csize_t
end
@assert sizeof(NamedValue) == 56

nv_string(n::String, s::String) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 0, 0, UInt64(pointer(s)), sizeof(s))
nv_int(n::String, v::Integer) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 1, 0, reinterpret(UInt64, Int64(v)), 1)
nv_list(n::String, a::Vector{Int64}) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 2, 0, UInt64(pointer(a)), length(a))
nv_float(n::String, f::Real) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 3, 0, UInt64(reinterpret(UInt32, Float32(f))), 1)
nv_bool(n::String, b::Bool) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 4, 0, UInt64(b), 1)

function read_named_values(p::Ptr{PJRT_NamedValue}, n::Integer)
    out = Dict{String, Any}()
    for i in 1:n
        nv = unsafe_load(Ptr{NamedValue}(p), i)
        name = unsafe_string(nv.name, nv.name_size)
        out[name] = if nv.type == 0
            unsafe_string(Ptr{UInt8}(nv.value), nv.value_size)
        elseif nv.type == 1
            reinterpret(Int64, nv.value)
        elseif nv.type == 2
            copy(unsafe_wrap(Array, Ptr{Int64}(nv.value), nv.value_size))
        elseif nv.type == 3
            reinterpret(Float32, UInt32(nv.value & 0xffffffff))
        else
            nv.value != 0
        end
    end
    return out
end

"Plugin-level attributes: `cuda_version`, `stablehlo_current_version`, `stablehlo_minimum_version`."
function plugin_attributes()
    ensure_initialized!()
    a = Ref(mk(PJRT_Plugin_Attributes_Args))
    check(pj(:PJRT_Plugin_Attributes, a), "Plugin_Attributes")
    return read_named_values(a[].attributes, a[].num_attributes)
end

# ── Client and devices ───────────────────────────────────────────────────────────────────────────

struct Device
    ptr::Ptr{PJRT_Device}
    ordinal::Int                        # index among the client's addressable devices, 0-based
    kind::String                        # "NVIDIA RTX A6000"
    compute_capability::Tuple{Int, Int}  # (0, 0) when the plugin does not report it
    memory_limit::Int                   # device_memory_bytes_limit attribute, or -1
    attributes::Dict{String, Any}
end

mutable struct Client
    ptr::Ptr{PJRT_Client}
    platform::String                    # "cuda"
    platform_version::String            # "cuda 13010"
    devices::Vector{Device}
end

function _cstring_field(args, name::Symbol, size::Symbol)
    return unsafe_string(Ptr{UInt8}(getfield(args, name)), getfield(args, size))
end

function _describe_device(ptr::Ptr{PJRT_Device}, ordinal::Int)
    gd = Ref(mk(PJRT_Device_GetDescription_Args; device = ptr))
    check(pj(:PJRT_Device_GetDescription, gd), "Device_GetDescription")
    desc = gd[].device_description
    kd = Ref(mk(PJRT_DeviceDescription_Kind_Args; device_description = desc))
    check(pj(:PJRT_DeviceDescription_Kind, kd), "DeviceDescription_Kind")
    kind = _cstring_field(kd[], :device_kind, :device_kind_size)
    at = Ref(mk(PJRT_DeviceDescription_Attributes_Args; device_description = desc))
    check(pj(:PJRT_DeviceDescription_Attributes, at), "DeviceDescription_Attributes")
    attrs = read_named_values(at[].attributes, at[].num_attributes)
    cc = (0, 0)
    if haskey(attrs, "compute_capability") && attrs["compute_capability"] isa AbstractString
        parts = split(attrs["compute_capability"], '.')
        if length(parts) == 2
            maj = tryparse(Int, parts[1])
            mn = tryparse(Int, parts[2])
            (maj === nothing || mn === nothing) || (cc = (maj, mn))
        end
    end
    lim = get(attrs, "device_memory_bytes_limit", -1)
    return Device(ptr, ordinal, kind, cc, lim isa Integer ? Int(lim) : -1, attrs)
end

"""
    create_client(; mem_fraction, preallocate) -> Client

Create the GPU client through `PJRT_Client_Create`. Every device visible to the process (per
`CUDA_VISIBLE_DEVICES`) is addressable, matching Reactant's GPU client; the allocator is the
plugin default (BFC) sized by `mem_fraction` and `preallocate`, as Reactant's is.
"""
function create_client(; mem_fraction::Real = 0.9, preallocate::Bool = true)
    ensure_initialized!()
    names = ["memory_fraction", "preallocate"]
    opts = [nv_float(names[1], mem_fraction), nv_bool(names[2], preallocate)]
    args = Ref(mk(PJRT_Client_Create_Args; create_options = pointer(opts), num_options = length(opts)))
    GC.@preserve names opts begin
        check(pj(:PJRT_Client_Create, args), "Client_Create")
    end
    ptr = args[].client
    pn = Ref(mk(PJRT_Client_PlatformName_Args; client = ptr))
    check(pj(:PJRT_Client_PlatformName, pn), "Client_PlatformName")
    pv = Ref(mk(PJRT_Client_PlatformVersion_Args; client = ptr))
    check(pj(:PJRT_Client_PlatformVersion, pv), "Client_PlatformVersion")
    ad = Ref(mk(PJRT_Client_AddressableDevices_Args; client = ptr))
    check(pj(:PJRT_Client_AddressableDevices, ad), "Client_AddressableDevices")
    raw = unsafe_wrap(Array, ad[].addressable_devices, ad[].num_addressable_devices)
    devices = [_describe_device(raw[i], i - 1) for i in eachindex(raw)]
    return Client(
        ptr, _cstring_field(pn[], :platform_name, :platform_name_size),
        _cstring_field(pv[], :platform_version, :platform_version_size), devices
    )
end

function destroy!(c::Client)
    c.ptr == C_NULL && return nothing
    check(pj(:PJRT_Client_Destroy, Ref(mk(PJRT_Client_Destroy_Args; client = c.ptr))), "Client_Destroy")
    c.ptr = C_NULL
    return nothing
end

# ── Events ───────────────────────────────────────────────────────────────────────────────────────

# Block until `ev` completes, surface its error if any, and destroy it.
function await_event(ev::Ptr{PJRT_Event})
    ev == C_NULL && return nothing
    err = pj(:PJRT_Event_Await, Ref(mk(PJRT_Event_Await_Args; event = ev)))
    # Destroy before raising so a failed event does not leak.
    derr = pj(:PJRT_Event_Destroy, Ref(mk(PJRT_Event_Destroy_Args; event = ev)))
    check(err, "Event_Await")
    check(derr, "Event_Destroy")
    return nothing
end

# ── Element types ────────────────────────────────────────────────────────────────────────────────
#
# PJRT_Buffer_Type mirrors xla::PrimitiveType. Only the types the server's dtype table can produce
# are mapped (see ReactantServerCore dtypes.jl); anything else is a load-time error.
const _ELTYPE_TO_PJRT = Dict{DataType, PJRT_Buffer_Type}()
const _PJRT_TO_ELTYPE = Dict{PJRT_Buffer_Type, DataType}()

function register_eltype!(T::DataType, pt::PJRT_Buffer_Type)
    _ELTYPE_TO_PJRT[T] = pt
    _PJRT_TO_ELTYPE[pt] = T
    return nothing
end

for (T, pt) in (
        Bool => PJRT_Buffer_Type_PRED, Int8 => PJRT_Buffer_Type_S8, Int16 => PJRT_Buffer_Type_S16,
        Int32 => PJRT_Buffer_Type_S32, Int64 => PJRT_Buffer_Type_S64, UInt8 => PJRT_Buffer_Type_U8,
        UInt16 => PJRT_Buffer_Type_U16, UInt32 => PJRT_Buffer_Type_U32, UInt64 => PJRT_Buffer_Type_U64,
        Float16 => PJRT_Buffer_Type_F16, Float32 => PJRT_Buffer_Type_F32, Float64 => PJRT_Buffer_Type_F64,
        ComplexF32 => PJRT_Buffer_Type_C64, ComplexF64 => PJRT_Buffer_Type_C128,
    )
    register_eltype!(T, pt)
end

function pjrt_type(T::DataType)
    pt = get(_ELTYPE_TO_PJRT, T, nothing)
    pt === nothing && throw(ArgumentError("element type $T has no PJRT buffer type mapping"))
    return pt
end
function julia_eltype(pt::PJRT_Buffer_Type)
    T = get(_PJRT_TO_ELTYPE, pt, nothing)
    T === nothing && throw(ArgumentError("PJRT buffer type $pt has no Julia element type mapping"))
    return T
end

# ── Buffers ──────────────────────────────────────────────────────────────────────────────────────

# `dims` are XLA (row-major) dimensions: the reverse of the Julia array's size, exactly as
# Reactant's `PJRT.Buffer(client, array, device)` passes them, so the mock backend's reversed-shape
# emulation and `run_model`'s reverse back both hold for this backend too.
mutable struct Buffer
    ptr::Ptr{PJRT_Buffer}
    eltype::DataType
    dims::Vector{Int64}
end

function to_device(client::Client, array::Array{T}, device::Device) where {T}
    dims = collect(Int64, reverse(size(array)))
    args = Ref(
        mk(
            PJRT_Client_BufferFromHostBuffer_Args; client = client.ptr, data = Ptr{Cvoid}(pointer(array)),
            type = pjrt_type(T), dims = pointer(dims), num_dims = length(dims),
            host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
            device = device.ptr
        )
    )
    GC.@preserve array dims begin
        check(pj(:PJRT_Client_BufferFromHostBuffer, args), "Client_BufferFromHostBuffer")
        # The host array may be reused or freed by the caller as soon as we return, so wait for
        # the transfer here (the semantics chosen above make the copy complete on this event).
        await_event(args[].done_with_host_buffer)
    end
    return Buffer(args[].buffer, T, dims)
end

# Wrap an output buffer handed back by execute, reading its type and shape from the runtime.
function _wrap_output(ptr::Ptr{PJRT_Buffer})
    et = Ref(mk(PJRT_Buffer_ElementType_Args; buffer = ptr))
    check(pj(:PJRT_Buffer_ElementType, et), "Buffer_ElementType")
    dm = Ref(mk(PJRT_Buffer_Dimensions_Args; buffer = ptr))
    check(pj(:PJRT_Buffer_Dimensions, dm), "Buffer_Dimensions")
    dims = dm[].num_dims == 0 ? Int64[] : copy(unsafe_wrap(Array, dm[].dims, dm[].num_dims))
    return Buffer(ptr, julia_eltype(et[].type), dims)
end

function to_host!(buffer::Buffer, dest::Array)
    buffer.ptr == C_NULL && throw(ArgumentError("buffer already freed"))
    n = sizeof(dest)
    args = Ref(mk(PJRT_Buffer_ToHostBuffer_Args; src = buffer.ptr, dst = Ptr{Cvoid}(pointer(dest)), dst_size = n))
    GC.@preserve dest begin
        check(pj(:PJRT_Buffer_ToHostBuffer, args), "Buffer_ToHostBuffer")
        await_event(args[].event)
    end
    return dest
end

function destroy!(b::Buffer)
    b.ptr == C_NULL && return nothing
    p = b.ptr
    b.ptr = C_NULL
    check(pj(:PJRT_Buffer_Destroy, Ref(mk(PJRT_Buffer_Destroy_Args; buffer = p))), "Buffer_Destroy")
    return nothing
end

# ── Executables ──────────────────────────────────────────────────────────────────────────────────

mutable struct LoadedExecutable
    ptr::Ptr{PJRT_LoadedExecutable}
    num_outputs::Int
end

function _num_outputs(loaded::Ptr{PJRT_LoadedExecutable})
    ex = _get_executable(loaded)
    try
        n = Ref(mk(PJRT_Executable_NumOutputs_Args; executable = ex))
        check(pj(:PJRT_Executable_NumOutputs, n), "Executable_NumOutputs")
        return Int(n[].num_outputs)
    finally
        _destroy_executable(ex)
    end
end

function _get_executable(loaded::Ptr{PJRT_LoadedExecutable})
    g = Ref(mk(PJRT_LoadedExecutable_GetExecutable_Args; loaded_executable = loaded))
    check(pj(:PJRT_LoadedExecutable_GetExecutable, g), "LoadedExecutable_GetExecutable")
    return g[].executable
end
_destroy_executable(ex::Ptr{PJRT_Executable}) =
    check(pj(:PJRT_Executable_Destroy, Ref(mk(PJRT_Executable_Destroy_Args; executable = ex))), "Executable_Destroy")

"""
    compile(client, program::Vector{UInt8}, compile_options::Vector{UInt8}) -> LoadedExecutable

Compile a StableHLO portable artifact (MLIR bytecode; text is accepted too) with a serialized
`CompileOptionsProto`.
"""
function compile(client::Client, program::Vector{UInt8}, compile_options::Vector{UInt8})
    fmt = "mlir"
    prog = Ref(
        mk(
            PJRT_Program; code = Cstring(pointer(program)), code_size = length(program),
            format = Cstring(pointer(fmt)), format_size = sizeof(fmt)
        )
    )
    args = Ref(
        mk(
            PJRT_Client_Compile_Args; client = client.ptr,
            program = Base.unsafe_convert(Ptr{PJRT_Program}, prog),
            compile_options = Cstring(pointer(compile_options)), compile_options_size = length(compile_options)
        )
    )
    GC.@preserve program fmt compile_options prog begin
        check(pj(:PJRT_Client_Compile, args), "Client_Compile")
    end
    loaded = args[].executable
    return LoadedExecutable(loaded, _num_outputs(loaded))
end

"Serialize a loaded executable to bytes that `deserialize_and_load` on the same platform, build and device class can load."
function serialize(exec::LoadedExecutable)
    ex = _get_executable(exec.ptr)
    try
        s = Ref(mk(PJRT_Executable_Serialize_Args; executable = ex))
        check(pj(:PJRT_Executable_Serialize, s), "Executable_Serialize")
        bytes = copy(unsafe_wrap(Array, Ptr{UInt8}(s[].serialized_bytes), s[].serialized_bytes_size))
        if s[].serialized_executable_deleter != C_NULL
            ccall(s[].serialized_executable_deleter, Cvoid, (Ptr{Cvoid},), s[].serialized_executable)
        end
        return bytes
    finally
        _destroy_executable(ex)
    end
end

"""
    deserialize_and_load(client, bytes, compile_options) -> LoadedExecutable

Load a serialized executable. `compile_options` (a serialized `CompileOptionsProto`, or `nothing`
to keep the ones stored in the blob) overrides the stored options, which is how a blob compiled for
one device ordinal is placed on another.
"""
function deserialize_and_load(client::Client, bytes::Vector{UInt8}, compile_options::Union{Nothing, Vector{UInt8}})
    args = if compile_options === nothing
        Ref(
            mk(
                PJRT_Executable_DeserializeAndLoad_Args; client = client.ptr,
                serialized_executable = Cstring(pointer(bytes)), serialized_executable_size = length(bytes)
            )
        )
    else
        Ref(
            mk(
                PJRT_Executable_DeserializeAndLoad_Args; client = client.ptr,
                serialized_executable = Cstring(pointer(bytes)), serialized_executable_size = length(bytes),
                overridden_serialized_compile_options = Cstring(pointer(compile_options)),
                overridden_serialized_compile_options_size = length(compile_options)
            )
        )
    end
    GC.@preserve bytes compile_options begin
        check(pj(:PJRT_Executable_DeserializeAndLoad, args), "Executable_DeserializeAndLoad")
    end
    loaded = args[].loaded_executable
    return LoadedExecutable(loaded, _num_outputs(loaded))
end

"""
    compiled_memory_stats(exec) -> NamedTuple

The compiler's static memory accounting from buffer assignment: `temp` (scratch), `outputs`,
`arguments`, `generated_code`, `peak`, in bytes. Available without executing.
"""
function compiled_memory_stats(exec::LoadedExecutable)
    ex = _get_executable(exec.ptr)
    try
        ms = Ref(mk(PJRT_Executable_GetCompiledMemoryStats_Args; executable = ex))
        check(pj(:PJRT_Executable_GetCompiledMemoryStats, ms), "Executable_GetCompiledMemoryStats")
        m = ms[]
        return (
            temp = Int(m.temp_size_in_bytes), outputs = Int(m.output_size_in_bytes),
            arguments = Int(m.argument_size_in_bytes), generated_code = Int(m.generated_code_size_in_bytes),
            peak = Int(m.peak_memory_in_bytes),
        )
    finally
        _destroy_executable(ex)
    end
end

function destroy!(e::LoadedExecutable)
    e.ptr == C_NULL && return nothing
    p = e.ptr
    e.ptr = C_NULL
    check(pj(:PJRT_LoadedExecutable_Destroy, Ref(mk(PJRT_LoadedExecutable_Destroy_Args; executable = p))), "LoadedExecutable_Destroy")
    return nothing
end

"""
    execute(exec, device, buffers, donated) -> Vector{Buffer}

Run `exec` on its compiled device assignment with `buffers` as arguments, in order. Inputs whose
`donated` flag is false are vetoed from donation (`non_donatable_input_indices`); donation itself
is decided at compile time by aliasing, exactly as in Reactant's `XLAExecuteSharded`. Blocks until
the device has completed, so the returned output buffers are ready.
"""
function execute(exec::LoadedExecutable, device::Device, buffers::AbstractVector{Buffer}, donated::AbstractVector{Bool})
    exec.ptr == C_NULL && throw(ArgumentError("executable already freed"))
    ptrs = Ptr{PJRT_Buffer}[b.ptr for b in buffers]
    nondon = Int64[i - 1 for i in eachindex(donated) if !donated[i]]
    opts = Ref(
        mk(
            PJRT_ExecuteOptions;
            non_donatable_input_indices = isempty(nondon) ? Ptr{Int64}(C_NULL) : pointer(nondon),
            num_non_donatable_input_indices = length(nondon)
        )
    )
    arglist = [pointer(ptrs)]
    outbuf = fill(Ptr{PJRT_Buffer}(C_NULL), exec.num_outputs)
    outlist = [pointer(outbuf)]
    events = Ptr{PJRT_Event}[C_NULL]
    args = Ref(
        mk(
            PJRT_LoadedExecutable_Execute_Args; executable = exec.ptr,
            options = Base.unsafe_convert(Ptr{PJRT_ExecuteOptions}, opts),
            argument_lists = pointer(arglist), num_devices = 1, num_args = length(ptrs),
            output_lists = pointer(outlist), device_complete_events = pointer(events)
        )
    )
    GC.@preserve ptrs nondon opts arglist outbuf outlist events buffers begin
        check(pj(:PJRT_LoadedExecutable_Execute, args), "LoadedExecutable_Execute")
        await_event(events[1])
    end
    return Buffer[_wrap_output(p) for p in outbuf]
end

# ── Memory stats ─────────────────────────────────────────────────────────────────────────────────

"""
    memory_stats(device) -> NamedTuple

The device allocator's statistics (`bytes_in_use`, `peak_bytes_in_use`, `bytes_limit` or -1,
`pool_bytes`, `peak_pool_bytes`; unreported fields are -1).
"""
function memory_stats(device::Device)
    a = Ref(mk(PJRT_Device_MemoryStats_Args; device = device.ptr))
    check(pj(:PJRT_Device_MemoryStats, a), "Device_MemoryStats")
    m = a[]
    return (
        bytes_in_use = Int(m.bytes_in_use),
        peak_bytes_in_use = m.peak_bytes_in_use_is_set ? Int(m.peak_bytes_in_use) : -1,
        bytes_limit = m.bytes_limit_is_set ? Int(m.bytes_limit) : -1,
        pool_bytes = m.pool_bytes_is_set ? Int(m.pool_bytes) : -1,
        peak_pool_bytes = m.peak_pool_bytes_is_set ? Int(m.peak_pool_bytes) : -1,
    )
end

"Reset the allocator's high-water mark (`peak_bytes_in_use`) to the current in-use bytes."
function clear_memory_stats!(device::Device)
    check(pj(:PJRT_Device_ClearMemoryStats, Ref(mk(PJRT_Device_ClearMemoryStats_Args; device = device.ptr))), "Device_ClearMemoryStats")
    return nothing
end

end # module PJRTCAPI
