# Reference: compile + execute the same bundle through Reactant's own PJRT path (XLA.client("cuda"),
# XLA.compile, execute_sharded), with the same inputs the C API spike used, and compare raw output bytes.
include(joinpath(@__DIR__, "spike_pjrt_capi.jl"))
code = read(ARGS[1]); outfiles = ARGS[2:end]
ins, outs = main_signature(code); host = make_inputs(ins)
XLA.XLA_REACTANT_GPU_PREALLOCATE[] = false
XLA.XLA_REACTANT_GPU_MEM_FRACTION[] = 0.5
t = time(); client = XLA.client("cuda"); dev = first(XLA.addressable_devices(client)); @info @sprintf("Reactant client: %.3fs", time() - t)
ctx = Reactant.ReactantContext(); MLIR.IR.activate(ctx)
exec = try
    mod = MLIR.IR.Module(MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(code)), ctx))
    opts = XLA.make_compile_options(; device_id = Int64(XLA.device_ordinal(dev)), xla_debug_options = (; xla_gpu_autotune_level = Int32(0)))
    local t = time()
    e = XLA.compile(
        client, mod; compile_options = opts, num_parameters = Int64(length(ins)), num_outputs = Int64(length(outs)),
        is_sharded = false, num_replicas = Int64(1), num_partitions = Int64(1)
    )
    @info @sprintf("Reactant COMPILE (autotune level 0): %.3fs", time() - t)
    e
finally
    MLIR.IR.deactivate(ctx)
end
bufs = [XLA.PJRT.Buffer(client, reshape(host[i], reverse(ins[i][2])...), dev) for i in eachindex(ins)]
in_ptrs = (Ptr{Cvoid}[b.buffer for b in bufs]...,)
don = (zeros(UInt8, length(bufs))...,)
res = XLA.execute_sharded(exec, dev, in_ptrs, don, Val(length(outs)))
flat = Any[]
function _flat!(acc, x)
    x isa XLA.AbstractBuffer && return push!(acc, x)
    (x isa Tuple || x isa AbstractArray) && foreach(e -> _flat!(acc, e), x)
    return acc
end
_flat!(flat, res)
bytes = Vector{UInt8}[]
for (i, b) in enumerate(flat)
    sb = XLA.synced_buffer(b)
    T, dims = outs[i]
    dest = Array{T}(undef, reverse(dims)...)
    XLA.to_host(sb, dest, Reactant.Sharding.NoSharding())
    push!(bytes, collect(reinterpret(UInt8, vec(dest))))
end
for f in outfiles
    ref = deserialize(f)
    same = length(ref) == length(bytes) && all(ref[i] == bytes[i] for i in eachindex(ref))
    maxdiff = same ? 0.0 : maximum(maximum(abs.(reinterpret(Float32, ref[i]) .- reinterpret(Float32, bytes[i]))) for i in eachindex(ref) if outs[i][1] === Float32; init = 0.0)
    @info "PARITY Reactant path vs C API outputs" file = basename(f) same maxdiff
end
