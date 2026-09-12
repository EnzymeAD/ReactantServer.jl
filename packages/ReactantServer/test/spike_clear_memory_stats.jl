# De-risk spike. Can the CURRENT Reactant backend reset the BFC high-water mark without a library rebuild?
# Calls the exported (mangled) C++ method xla::StreamExecutorGpuDevice::ClearMemoryStats() on the
# PjRtDevice* Reactant already holds. absl::Status return goes through a hidden sret pointer.
using Reactant, Libdl, Printf
const XLA = Reactant.XLA; const MLIR = Reactant.MLIR
XLA.XLA_REACTANT_GPU_PREALLOCATE[] = false; XLA.XLA_REACTANT_GPU_MEM_FRACTION[] = 0.5
client = XLA.client("cuda"); dev = first(XLA.addressable_devices(client))
peak() = XLA.allocatorstats(dev).peak_bytes_in_use / 2^20
code = read(ARGS[1])
ctx = Reactant.ReactantContext(); MLIR.IR.activate(ctx)
try
    mod = MLIR.IR.Module(MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(code)), ctx))
    opts = XLA.make_compile_options(; device_id = Int64(XLA.device_ordinal(dev)))
    @info @sprintf("peak before compile: %.1f MB", peak())
    XLA.compile(client, mod; compile_options = opts, num_parameters = Int64(152), num_outputs = Int64(1), is_sharded = false, num_replicas = Int64(1), num_partitions = Int64(1))
    @info @sprintf("peak after compile (autotune): %.1f MB", peak())
finally
    MLIR.IR.deactivate(ctx)
end
h = Libdl.dlopen(Reactant.Reactant_jll.libReactantExtra_path)
sym = Libdl.dlsym_e(h, "_ZN3xla23StreamExecutorGpuDevice16ClearMemoryStatsEv")
@info "symbol" sym
if sym != C_NULL
    sret = Ref{UInt64}(0)
    ccall(sym, Ptr{Cvoid}, (Ref{UInt64}, Ptr{Cvoid}), sret, dev.device)
    @info @sprintf("status rep = 0x%x (absl OK inlined rep is 0x1); peak after ClearMemoryStats: %.1f MB", sret[], peak())
end
