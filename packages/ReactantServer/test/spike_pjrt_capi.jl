# De-risk spike: drive the PJRT C API directly from Julia to prove a serialized-executable cache.
#
# Everything goes through the PJRT_Api function table that libReactantExtra.so exports via the
# plain C symbol GetPjrtApi (the CUDA build of Reactant_jll is itself a PJRT GPU plugin). Struct
# layouts come from Reactant's generated (but not included) src/xla/PJRT/CAPI.jl, so they track the
# installed Reactant version. Reactant's own XLA client is never created.
#
# Modes:
#   compile <model.mlir> <blob>   create client, compile the StableHLO artifact, serialize to <blob>,
#                                  execute with seeded inputs and save the raw outputs to <blob>.out
#   load    <model.mlir> <blob>   fresh process: deserialize+load <blob>, execute, compare to <blob>.out
#   both    <model.mlir> <blob>   compile and load in one process
#   xdev    <model.mlir> <blob>   two GPUs: compile for device 0, reload with the compile options
#                                  overridden to device 1, execute there, compare
# Env: SPIKE_AUTOTUNE_LEVEL0=1 disables autotuning; SPIKE_FMA=1 applies Reactant's --nvptx-fma-level=1.
# Needs a GPU (CUDA_VISIBLE_DEVICES) and an environment with Reactant >= 0.2.285 plus CEnum, e.g.
#   gpu-lease -need 1 julia --project=<env> test/spike_pjrt_capi.jl both <bundle>/model.b1.mlir /tmp/x.exe
# Companion: spike_pjrt_capi_reactant_ref.jl compiles the same bundle through Reactant's own path and
# checks the outputs are byte-identical. Findings are written up in pjrt-executable-cache-plan.md.
using Reactant, Libdl, Serialization, Random, Printf
const MLIR = Reactant.MLIR
const XLA = Reactant.XLA
const jll = Reactant.Reactant_jll
const CAPI_PATH = joinpath(pkgdir(Reactant), "src", "xla", "PJRT", "CAPI.jl")

module CAPI
    using CEnum: CEnum, @cenum
    let lines = filter(readlines(Main.CAPI_PATH)) do l
            !occursin("_PJRT_API_STRUCT_FIELD", l) && !occursin("PJRT_NO_DISCARD", l) && !startswith(strip(l), "using CEnum")
        end
        include_string(@__MODULE__, join(lines, "\n"), "CAPI.jl")
    end
end

const API = Ref{Ptr{CAPI.PJRT_Api}}(C_NULL)
struct_size(::Type{T}) where {T} = Csize_t(Integer(getfield(CAPI, Symbol(nameof(T), :_STRUCT_SIZE))))

# Zero-initialised args struct with struct_size filled in and the given fields overridden.
function mk(::Type{T}; kw...) where {T}
    kwd = Dict{Symbol, Any}(kw)
    vals = Any[]
    for (i, fn) in enumerate(fieldnames(T))
        ft = fieldtype(T, i)
        if fn === :struct_size
            push!(vals, struct_size(T))
        elseif haskey(kwd, fn)
            push!(vals, convert(ft, kwd[fn]))
        elseif ft <: Ptr || ft === Cstring
            push!(vals, ft(C_NULL))
        elseif ft === Bool
            push!(vals, false)
        elseif ft <: Number
            push!(vals, zero(ft))
        elseif ft <: CAPI.CEnum.Cenum
            push!(vals, ft(0))
        else
            error("mk: no default for field $fn::$ft of $T")
        end
    end
    return T(vals...)
end

api() = unsafe_load(API[])
pj(name::Symbol, args::Ref) = ccall(getproperty(api(), name), Ptr{CAPI.PJRT_Error}, (Ptr{Cvoid},), args)

function check(err::Ptr{CAPI.PJRT_Error}, what)
    err == C_NULL && return nothing
    m = Ref(mk(CAPI.PJRT_Error_Message_Args; error = err))
    ccall(api().PJRT_Error_Message, Cvoid, (Ptr{Cvoid},), m)
    msg = unsafe_string(Ptr{UInt8}(m[].message), m[].message_size)
    c = Ref(mk(CAPI.PJRT_Error_GetCode_Args; error = err))
    ccall(api().PJRT_Error_GetCode, Ptr{Cvoid}, (Ptr{Cvoid},), c)
    code = c[].code
    d = Ref(mk(CAPI.PJRT_Error_Destroy_Args; error = err))
    ccall(api().PJRT_Error_Destroy, Cvoid, (Ptr{Cvoid},), d)
    error("PJRT $what failed [$code]: $msg")
end

# PJRT_NamedValue: {struct_size, extension_start, name, name_size, type(+pad), value union, value_size} = 56 bytes
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
nv_int(n::String, v) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 1, 0, reinterpret(UInt64, Int64(v)), 1)
nv_list(n::String, a::Vector{Int64}) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 2, 0, UInt64(pointer(a)), length(a))
nv_float(n::String, f) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 3, 0, UInt64(reinterpret(UInt32, Float32(f))), 1)
nv_bool(n::String, b::Bool) = NamedValue(56, C_NULL, pointer(n), sizeof(n), 4, 0, UInt64(b), 1)

function read_named_values(p::Ptr{CAPI.PJRT_NamedValue}, n)
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

function init_api()
    lib = jll.libReactantExtra_path
    h = Libdl.dlopen(lib)
    API[] = ccall(Libdl.dlsym(h, :GetPjrtApi), Ptr{CAPI.PJRT_Api}, ())
    a = api()
    @info "PJRT_Api" lib struct_size = a.struct_size expected = struct_size(CAPI.PJRT_Api) major = a.pjrt_api_version.major_version minor = a.pjrt_api_version.minor_version bindings_minor = CAPI.PJRT_API_MINOR
    a.struct_size == struct_size(CAPI.PJRT_Api) || @warn "PJRT_Api struct_size mismatch between library and bindings"
    check(pj(:PJRT_Plugin_Initialize, Ref(mk(CAPI.PJRT_Plugin_Initialize_Args))), "Plugin_Initialize")
    pa = Ref(mk(CAPI.PJRT_Plugin_Attributes_Args))
    check(pj(:PJRT_Plugin_Attributes, pa), "Plugin_Attributes")
    attrs = read_named_values(pa[].attributes, pa[].num_attributes)
    @info "Plugin attributes" attrs
    return attrs
end

function create_client(gpu::Int; mem_fraction = 0.5, preallocate = false, visible::Vector{Int64} = Int64[gpu])
    names = ["visible_devices", "memory_fraction", "preallocate", "allocator"]
    vis = visible
    alloc = "bfc"
    opts = [nv_list(names[1], vis), nv_float(names[2], mem_fraction), nv_bool(names[3], preallocate), nv_string(names[4], alloc)]
    args = Ref(mk(CAPI.PJRT_Client_Create_Args; create_options = pointer(opts), num_options = length(opts)))
    GC.@preserve names vis alloc opts begin
        check(pj(:PJRT_Client_Create, args), "Client_Create")
    end
    client = args[].client
    pn = Ref(mk(CAPI.PJRT_Client_PlatformName_Args; client))
    check(pj(:PJRT_Client_PlatformName, pn), "PlatformName")
    pv = Ref(mk(CAPI.PJRT_Client_PlatformVersion_Args; client))
    check(pj(:PJRT_Client_PlatformVersion, pv), "PlatformVersion")
    @info "client" platform = unsafe_string(Ptr{UInt8}(pn[].platform_name), pn[].platform_name_size) version = unsafe_string(Ptr{UInt8}(pv[].platform_version), pv[].platform_version_size)
    ad = Ref(mk(CAPI.PJRT_Client_AddressableDevices_Args; client))
    check(pj(:PJRT_Client_AddressableDevices, ad), "AddressableDevices")
    devs = unsafe_wrap(Array, ad[].addressable_devices, ad[].num_addressable_devices)
    device = devs[1]
    gd = Ref(mk(CAPI.PJRT_Device_GetDescription_Args; device))
    check(pj(:PJRT_Device_GetDescription, gd), "GetDescription")
    desc = gd[].device_description
    kd = Ref(mk(CAPI.PJRT_DeviceDescription_Kind_Args; device_description = desc))
    check(pj(:PJRT_DeviceDescription_Kind, kd), "Kind")
    at = Ref(mk(CAPI.PJRT_DeviceDescription_Attributes_Args; device_description = desc))
    check(pj(:PJRT_DeviceDescription_Attributes, at), "Attributes")
    dattrs = read_named_values(at[].attributes, at[].num_attributes)
    @info "device" n = length(devs) kind = unsafe_string(Ptr{UInt8}(kd[].device_kind), kd[].device_kind_size)
    for (k, v) in sort(collect(dattrs); by = first)
        println("    devattr ", k, " = ", repr(v))
    end
    return client, device, copy(devs)
end

function compile(client, code::Vector{UInt8}, copts::Vector{UInt8})
    fmt = "mlir"
    prog = Ref(mk(CAPI.PJRT_Program; code = Cstring(pointer(code)), code_size = length(code), format = Cstring(pointer(fmt)), format_size = sizeof(fmt)))
    args = Ref(mk(CAPI.PJRT_Client_Compile_Args; client, program = Base.unsafe_convert(Ptr{CAPI.PJRT_Program}, prog), compile_options = Cstring(pointer(copts)), compile_options_size = length(copts)))
    GC.@preserve code fmt copts prog begin
        check(pj(:PJRT_Client_Compile, args), "Client_Compile")
    end
    return args[].executable
end

function get_executable(loaded)
    g = Ref(mk(CAPI.PJRT_LoadedExecutable_GetExecutable_Args; loaded_executable = loaded))
    check(pj(:PJRT_LoadedExecutable_GetExecutable, g), "GetExecutable")
    return g[].executable
end

function serialize_exec(loaded)
    ex = get_executable(loaded)
    s = Ref(mk(CAPI.PJRT_Executable_Serialize_Args; executable = ex))
    check(pj(:PJRT_Executable_Serialize, s), "Executable_Serialize")
    bytes = copy(unsafe_wrap(Array, Ptr{UInt8}(s[].serialized_bytes), s[].serialized_bytes_size))
    if s[].serialized_executable_deleter != C_NULL
        ccall(s[].serialized_executable_deleter, Cvoid, (Ptr{Cvoid},), s[].serialized_executable)
    end
    f = Ref(mk(CAPI.PJRT_Executable_Fingerprint_Args; executable = ex))
    check(pj(:PJRT_Executable_Fingerprint, f), "Executable_Fingerprint")
    fp = unsafe_string(Ptr{UInt8}(f[].executable_fingerprint), f[].executable_fingerprint_size)
    ms = Ref(mk(CAPI.PJRT_Executable_GetCompiledMemoryStats_Args; executable = ex))
    check(pj(:PJRT_Executable_GetCompiledMemoryStats, ms), "GetCompiledMemoryStats")
    check(pj(:PJRT_Executable_Destroy, Ref(mk(CAPI.PJRT_Executable_Destroy_Args; executable = ex))), "Executable_Destroy")
    return bytes, fp, ms[]
end

function num_outputs(loaded)
    ex = get_executable(loaded)
    n = Ref(mk(CAPI.PJRT_Executable_NumOutputs_Args; executable = ex))
    check(pj(:PJRT_Executable_NumOutputs, n), "NumOutputs")
    check(pj(:PJRT_Executable_Destroy, Ref(mk(CAPI.PJRT_Executable_Destroy_Args; executable = ex))), "Executable_Destroy")
    return Int(n[].num_outputs)
end

function deserialize_exec(client, bytes::Vector{UInt8}, copts::Union{Nothing, Vector{UInt8}})
    args = if copts === nothing
        Ref(mk(CAPI.PJRT_Executable_DeserializeAndLoad_Args; client, serialized_executable = Cstring(pointer(bytes)), serialized_executable_size = length(bytes)))
    else
        Ref(
            mk(
                CAPI.PJRT_Executable_DeserializeAndLoad_Args; client, serialized_executable = Cstring(pointer(bytes)), serialized_executable_size = length(bytes),
                overridden_serialized_compile_options = Cstring(pointer(copts)), overridden_serialized_compile_options_size = length(copts)
            )
        )
    end
    GC.@preserve bytes copts begin
        check(pj(:PJRT_Executable_DeserializeAndLoad, args), "DeserializeAndLoad")
    end
    return args[].loaded_executable
end

const PJRT_TYPES = Dict{Any, Any}(
    Float32 => CAPI.PJRT_Buffer_Type_F32, Float64 => CAPI.PJRT_Buffer_Type_F64, Float16 => CAPI.PJRT_Buffer_Type_F16,
    Int32 => CAPI.PJRT_Buffer_Type_S32, Int64 => CAPI.PJRT_Buffer_Type_S64, Int8 => CAPI.PJRT_Buffer_Type_S8, Int16 => CAPI.PJRT_Buffer_Type_S16,
    UInt8 => CAPI.PJRT_Buffer_Type_U8, UInt32 => CAPI.PJRT_Buffer_Type_U32, UInt64 => CAPI.PJRT_Buffer_Type_U64, Bool => CAPI.PJRT_Buffer_Type_PRED,
)

function await(ev::Ptr{CAPI.PJRT_Event})
    check(pj(:PJRT_Event_Await, Ref(mk(CAPI.PJRT_Event_Await_Args; event = ev))), "Event_Await")
    check(pj(:PJRT_Event_Destroy, Ref(mk(CAPI.PJRT_Event_Destroy_Args; event = ev))), "Event_Destroy")
    return nothing
end

function to_device(client, device, arr::Array, dims::Vector{Int64})
    args = Ref(
        mk(
            CAPI.PJRT_Client_BufferFromHostBuffer_Args; client, data = Ptr{Cvoid}(pointer(arr)), type = PJRT_TYPES[eltype(arr)],
            dims = pointer(dims), num_dims = length(dims), host_buffer_semantics = CAPI.PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes, device
        )
    )
    GC.@preserve arr dims begin
        check(pj(:PJRT_Client_BufferFromHostBuffer, args), "BufferFromHostBuffer")
        args[].done_with_host_buffer != C_NULL && await(args[].done_with_host_buffer)
    end
    return args[].buffer
end

function execute(loaded, device, bufs::Vector{Ptr{CAPI.PJRT_Buffer}}, nout::Int; portable::Bool = false)
    opts = Ref(mk(CAPI.PJRT_ExecuteOptions))
    arglist = [pointer(bufs)]
    outbuf = fill(Ptr{CAPI.PJRT_Buffer}(C_NULL), nout)
    outlist = [pointer(outbuf)]
    events = Ptr{CAPI.PJRT_Event}[C_NULL]
    args = Ref(
        mk(
            CAPI.PJRT_LoadedExecutable_Execute_Args; executable = loaded, options = Base.unsafe_convert(Ptr{CAPI.PJRT_ExecuteOptions}, opts),
            argument_lists = pointer(arglist), num_devices = 1, num_args = length(bufs), output_lists = pointer(outlist),
            device_complete_events = pointer(events), execute_device = portable ? device : Ptr{CAPI.PJRT_Device}(C_NULL)
        )
    )
    GC.@preserve opts bufs arglist outbuf outlist events begin
        check(pj(:PJRT_LoadedExecutable_Execute, args), "Execute")
        events[1] != C_NULL && await(events[1])
    end
    return outbuf
end

function to_host(buf)
    a = Ref(mk(CAPI.PJRT_Buffer_ToHostBuffer_Args; src = buf))
    check(pj(:PJRT_Buffer_ToHostBuffer, a), "ToHostBuffer(size)")
    n = a[].dst_size
    out = Vector{UInt8}(undef, n)
    b = Ref(mk(CAPI.PJRT_Buffer_ToHostBuffer_Args; src = buf, dst = Ptr{Cvoid}(pointer(out)), dst_size = n))
    GC.@preserve out begin
        check(pj(:PJRT_Buffer_ToHostBuffer, b), "ToHostBuffer")
        b[].event != C_NULL && await(b[].event)
    end
    return out
end

function memory_stats(device)
    a = Ref(mk(CAPI.PJRT_Device_MemoryStats_Args; device))
    check(pj(:PJRT_Device_MemoryStats, a), "Device_MemoryStats")
    m = a[]
    return (
        in_use = m.bytes_in_use, peak = m.peak_bytes_in_use, limit = m.bytes_limit_is_set ? m.bytes_limit : -1,
        pool = m.pool_bytes_is_set ? m.pool_bytes : -1, peak_pool = m.peak_pool_bytes_is_set ? m.peak_pool_bytes : -1,
    )
end
function clear_memory_stats(device)
    a = Ref(mk(CAPI.PJRT_Device_ClearMemoryStats_Args; device))
    check(pj(:PJRT_Device_ClearMemoryStats, a), "Device_ClearMemoryStats")
    return nothing
end
function compiled_memory_stats(loaded)
    ex = get_executable(loaded)
    ms = Ref(mk(CAPI.PJRT_Executable_GetCompiledMemoryStats_Args; executable = ex))
    check(pj(:PJRT_Executable_GetCompiledMemoryStats, ms), "GetCompiledMemoryStats")
    check(pj(:PJRT_Executable_Destroy, Ref(mk(CAPI.PJRT_Executable_Destroy_Args; executable = ex))), "Executable_Destroy")
    m = ms[]
    return (;
        generated_code = m.generated_code_size_in_bytes, arguments = m.argument_size_in_bytes, outputs = m.output_size_in_bytes,
        alias = m.alias_size_in_bytes, temp = m.temp_size_in_bytes, peak = m.peak_memory_in_bytes, total = m.total_size_in_bytes,
        total_alloc = m.total_allocation_bytes, indefinite = m.indefinite_allocations, peak_unpadded_heap = m.peak_unpadded_heap_bytes,
    )
end
function buffer_device(buf)
    a = Ref(mk(CAPI.PJRT_Buffer_Device_Args; buffer = buf))
    check(pj(:PJRT_Buffer_Device, a), "Buffer_Device")
    return a[].device
end
destroy_buffer(buf) = check(pj(:PJRT_Buffer_Destroy, Ref(mk(CAPI.PJRT_Buffer_Destroy_Args; buffer = buf))), "Buffer_Destroy")
destroy_loaded(l) = check(pj(:PJRT_LoadedExecutable_Destroy, Ref(mk(CAPI.PJRT_LoadedExecutable_Destroy_Args; executable = l))), "LoadedExecutable_Destroy")

# (eltype, dims) of every @main argument and result, read from the StableHLO artifact.
function main_signature(mlir_bytes::Vector{UInt8})
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    return try
        m = MLIR.IR.Module(MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(mlir_bytes)), ctx))
        for op in MLIR.IR.body(m)
            MLIR.IR.name(op) == "func.func" || continue
            occursin("main", string(MLIR.IR.getattr(op, "sym_name"))) || continue
            ft = MLIR.IR.FunctionType(op)
            shp(t) = (MLIR.IR.julia_type(eltype(t)), Int64[MLIR.IR.size(t, d) for d in 1:MLIR.IR.ndims(t)])
            ins = [shp(MLIR.IR.input(ft, i)) for i in 1:MLIR.IR.ninputs(ft)]
            outs = [shp(MLIR.IR.result(ft, i)) for i in 1:MLIR.IR.nresults(ft)]
            return ins, outs
        end
        error("no @main")
    finally
        MLIR.IR.deactivate(ctx)
    end
end

function make_inputs(ins)
    rng = MersenneTwister(1234)
    arrs = Any[]
    for (T, dims) in ins
        n = prod(dims)
        a = if T <: AbstractFloat
            T.(randn(rng, Float32, n) .* 0.05f0)
        elseif T === Bool
            rand(rng, Bool, n)
        else
            T.(rand(rng, 0:99, n))
        end
        push!(arrs, a)
    end
    return arrs
end

function compile_options_bytes(; autotune_dir::String = "", device_id::Int = 0)
    if !isempty(autotune_dir)
        Reactant.PersistentCompileCache.CACHE_DIR[] = autotune_dir
        Reactant.PersistentCompileCache.AUTOTUNE_CACHE_ENABLED[] = true
        mkpath(autotune_dir)
    else
        Reactant.PersistentCompileCache.AUTOTUNE_CACHE_ENABLED[] = false
    end
    dbg = get(ENV, "SPIKE_AUTOTUNE_LEVEL0", "0") == "1" ? (; xla_gpu_autotune_level = Int32(0)) : (;)
    opts = XLA.make_compile_options(; device_id = Int64(device_id), xla_debug_options = dbg)
    return Reactant.ProtoUtils.proto_to_bytes(opts)
end

function run_and_fetch(client, device, loaded, ins, outs, host_inputs)
    t0 = time()
    bufs = Ptr{CAPI.PJRT_Buffer}[to_device(client, device, host_inputs[i], ins[i][2]) for i in eachindex(ins)]
    t1 = time()
    nout = num_outputs(loaded)
    nout == length(outs) || @warn "num_outputs mismatch" nout length(outs)
    ob = execute(loaded, device, bufs, nout)
    t2 = time()
    ob2 = execute(loaded, device, bufs, nout)
    t3 = time()
    res = [to_host(b) for b in ob2]
    outdev = buffer_device(ob2[1])
    @info "output buffer device" outdev expected = device same_device = (outdev == device) mem = memory_stats(device)
    foreach(destroy_buffer, ob); foreach(destroy_buffer, ob2); foreach(destroy_buffer, bufs)
    @info @sprintf("execute: upload %.3fs, first exec %.3fs, second exec %.4fs", t1 - t0, t2 - t1, t3 - t2)
    return res
end

function main()
    mode, mlir_path, blob = ARGS[1], ARGS[2], ARGS[3]
    gpu = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
    autotune_dir = length(ARGS) >= 5 ? ARGS[5] : ""
    code = read(mlir_path)
    ins, outs = main_signature(code)
    @info "module" n_args = length(ins) n_results = length(outs) mlir_bytes = length(code)
    host_inputs = make_inputs(ins)

    init_api()
    if get(ENV, "SPIKE_FMA", "0") == "1"
        XLA.LLVMclopts("--nvptx-fma-level=1")   # what Reactant's GPU client constructor does (accelerators/GPU.jl)
        @info "set --nvptx-fma-level=1"
    end
    visible = mode == "xdev" ? Int64[0, 1] : Int64[gpu]
    t = time(); client, device, devs = create_client(gpu; visible); tc = time() - t
    @info @sprintf("client create: %.3fs", tc)
    copts = compile_options_bytes(; autotune_dir)

    if mode in ("compile", "both")
        t = time(); loaded = compile(client, code, copts); tcomp = time() - t
        @info @sprintf("COMPILE: %.3fs", tcomp)
        t = time(); bytes, fp, ms = serialize_exec(loaded); tser = time() - t
        @info @sprintf("SERIALIZE: %.3fs, %d bytes (%.1f MB), fingerprint=%s", tser, length(bytes), length(bytes) / 1.0e6, fp) generated_code_size = ms.generated_code_size_in_bytes temp = ms.temp_size_in_bytes
        write(blob, bytes)
        res = run_and_fetch(client, device, loaded, ins, outs, host_inputs)
        serialize(blob * ".out", res)
        destroy_loaded(loaded)
    end
    if mode in ("load", "both")
        bytes = read(blob)
        for (label, o) in (("with overridden compile options", copts), ("without options", nothing))
            t = time(); loaded = deserialize_exec(client, bytes, o); tload = time() - t
            @info @sprintf("DESERIALIZE+LOAD (%s): %.3fs", label, tload)
            res = run_and_fetch(client, device, loaded, ins, outs, host_inputs)
            ref = deserialize(blob * ".out")
            same = length(ref) == length(res) && all(ref[i] == res[i] for i in eachindex(ref))
            @info "PARITY vs compiled outputs (raw bytes)" same nbytes = sum(length, res)
            destroy_loaded(loaded)
        end
    end
    if mode == "memprobe"
        # Does the allocator high-water mark reset, and does the compiler's static temp size predict run scratch?
        mb(x) = round(x / 2^20; digits = 1)
        st(label) = (m = memory_stats(device); @info label in_use_MB = mb(m.in_use) peak_MB = mb(m.peak))
        st("before compile")
        loaded = compile(client, code, copts)
        st("after compile (autotune scratch is in the peak)")
        cms = compiled_memory_stats(loaded)
        @info "compiled memory stats (static, from buffer assignment)" temp_MB = mb(cms.temp) outputs_MB = mb(cms.outputs) arguments_MB = mb(cms.arguments) peak_MB = mb(cms.peak) total_MB = mb(cms.total) generated_code_MB = mb(cms.generated_code)
        clear_memory_stats(device)
        st("after ClearMemoryStats")
        bufs = Ptr{CAPI.PJRT_Buffer}[to_device(client, device, host_inputs[i], ins[i][2]) for i in eachindex(ins)]
        clear_memory_stats(device)
        m0 = memory_stats(device)
        st("inputs+weights resident, stats cleared")
        nout = num_outputs(loaded)
        ob = execute(loaded, device, bufs, nout)
        m1 = memory_stats(device)
        @info "after first execute" in_use_MB = mb(m1.in_use) peak_MB = mb(m1.peak) run_scratch_MB = mb(m1.peak - m0.in_use) predicted_temp_plus_outputs_MB = mb(cms.temp + cms.outputs)
        foreach(destroy_buffer, ob)
        clear_memory_stats(device)
        ob = execute(loaded, device, bufs, nout)
        m2 = memory_stats(device)
        @info "after second execute (cleared in between)" peak_MB = mb(m2.peak) run_scratch_MB = mb(m2.peak - m0.in_use)
        foreach(destroy_buffer, ob); foreach(destroy_buffer, bufs)
        destroy_loaded(loaded)
        st("after teardown")
    elseif mode == "xdev"
        # compile for device 0, serialize, reload with compile options overridden to device 1, run there
        t = time(); loaded = compile(client, code, copts); tcomp = time() - t
        @info @sprintf("COMPILE (device 0): %.3fs", tcomp)
        bytes, fp, _ = serialize_exec(loaded)
        res0 = run_and_fetch(client, devs[1], loaded, ins, outs, host_inputs)
        destroy_loaded(loaded)
        copts1 = compile_options_bytes(; autotune_dir, device_id = 1)
        t = time(); loaded1 = deserialize_exec(client, bytes, copts1); tload = time() - t
        @info @sprintf("DESERIALIZE+LOAD with device_id=1 override: %.3fs", tload)
        res1 = run_and_fetch(client, devs[2], loaded1, ins, outs, host_inputs)
        @info "PARITY device0 vs device1" same = all(res0[i] == res1[i] for i in eachindex(res0))
        destroy_loaded(loaded1)
        t = time(); loaded_noov = deserialize_exec(client, bytes, nothing); tload = time() - t
        @info @sprintf("DESERIALIZE+LOAD without override (should stay on device 0): %.3fs", tload)
        run_and_fetch(client, devs[1], loaded_noov, ins, outs, host_inputs)
        destroy_loaded(loaded_noov)
    end
    check(pj(:PJRT_Client_Destroy, Ref(mk(CAPI.PJRT_Client_Destroy_Args; client))), "Client_Destroy")
    @info "Reactant's own XLA backend state" initialized = Reactant.XLA.global_backend_state.initialized
    return @info "done"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
