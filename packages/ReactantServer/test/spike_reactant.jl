# De-risk spike: prove the server's runtime path against the installed Reactant.
#
# Validates, on the CPU PJRT backend, the exact sequence the server will use:
#   StableHLO text -> portable-artifact bytes -> deserialize -> wrap as Module
#   -> XLA.compile (weights as explicit args) -> execute_sharded -> read host.
#
# Run: julia --project=. test/spike_reactant.jl

using Reactant

const XLA = Reactant.XLA
const MLIR = Reactant.MLIR

# Capture a callback-based MLIR string producer into bytes. Returns (bytes, raw_result).
function capture_bytes(f)
    cb = @cfunction(MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any))
    ref = Ref(IOBuffer())
    res = f(cb, ref)
    return take!(ref[]), res
end

function current_stablehlo_version()
    bytes, _ = capture_bytes((cb, ref) -> MLIR.API.stablehloGetCurrentVersion(cb, ref))
    return String(bytes)
end

# Flatten whatever execute_sharded returns into a flat list of buffers.
function collect_buffers(x)
    x isa XLA.AbstractBuffer && return XLA.AbstractBuffer[x]
    if x isa Tuple || x isa AbstractArray
        out = XLA.AbstractBuffer[]
        for e in x; append!(out, collect_buffers(e)); end
        return out
    end
    return XLA.AbstractBuffer[]
end

const ADD_MLIR = """
module {
  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

function run_spike()
    isdefined(Reactant, :registry) && Reactant.registry[] === nothing &&
        Reactant.initialize_dialect()

    # --- bring-up under an active Reactant MLIR context ---
    artifact, client, dev, exec = MLIR.IR.@with_context Reactant.ReactantContext() begin
        ctx = MLIR.IR.current_context()

        # author + serialize a portable artifact (stands in for an external bundle's model.mlir)
        m = parse(MLIR.IR.Module, ADD_MLIR)
        ver = current_stablehlo_version()
        @info "StableHLO current version" ver
        bytes, sres = capture_bytes((cb, ref) -> MLIR.API.stablehloSerializePortableArtifactFromModule(
            m, ver, cb, ref, true))
        MLIR.IR.isfailure(MLIR.IR.LogicalResult(sres)) && error("failed to serialize module")
        @info "Serialized portable artifact" nbytes=length(bytes)

        # deserialize bytes -> module  (the path the server takes on every bundle load).
        # The C wrapper has a cconvert for String/AbstractString but not Vector{UInt8},
        # so pass the artifact as a binary String (preserves all bytes incl. NULs).
        mlir_mod = MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(bytes)), ctx)
        m2 = MLIR.IR.Module(mlir_mod)

        cl = XLA.client("cpu")
        d = first(XLA.addressable_devices(cl))
        opts = XLA.make_compile_options(; device_id = Int64(XLA.device_ordinal(d)))
        ex = XLA.compile(cl, m2;
            compile_options = opts,
            num_parameters = Int64(2),
            num_outputs = Int64(1),
            is_sharded = false,
            num_replicas = Int64(1),
            num_partitions = Int64(1),
        )
        (bytes, cl, d, ex)
    end

    # --- execute ---
    a = Float32[1, 2, 3, 4]
    b = Float32[10, 20, 30, 40]
    ba = XLA.PJRT.Buffer(client, a, dev)
    bb = XLA.PJRT.Buffer(client, b, dev)

    outs = XLA.execute_sharded(exec, dev, (ba.buffer, bb.buffer), (UInt8(0), UInt8(0)), Val(1))
    @info "execute_sharded returned" typeof=typeof(outs)

    bufs = collect_buffers(outs)
    @assert length(bufs) == 1 "expected 1 output buffer, got $(length(bufs))"
    ob = bufs[1]
    @info "output buffer" eltype=eltype(ob) size=size(ob)

    host = Array{Float32}(undef, 4)
    XLA.to_host(ob, host, Reactant.Sharding.NoSharding())
    @info "result" host

    expected = Float32[11, 22, 33, 44]
    @assert host == expected "spike mismatch: got $host expected $expected"
    println("\nSPIKE PASSED: deserialize -> compile -> execute round trip works on CPU")
    return host
end

run_spike()
