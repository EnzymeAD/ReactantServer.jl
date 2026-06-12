# De-risk spike for the conversion tooling.
#
# Traces a Dense-like g(x,W,b)=W*x.+b with Reactant.Compiler.compile_mlir at two batch
# sizes, confirms the StableHLO entry signature arg order is positional [x, W, b] with
# weights batch-independent, serializes to portable artifacts, and round-trips ONE size
# through the current server runtime to confirm compile_mlir output is server-loadable.
#
# Run: julia --project=packages/ReactantServerExport/test packages/ReactantServer/test/spike_lux_export.jl

using Reactant
using SafeTensors
using JSON3
using ReactantServer

const XLA = Reactant.XLA
const MLIR = Reactant.MLIR
const Compiler = Reactant.Compiler

g(x, W, b) = W * x .+ b

function capture_bytes(f)
    cb = @cfunction(MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any))
    ref = Ref(IOBuffer())
    res = f(cb, ref)
    return take!(ref[]), res
end

function serialize_module(mod)
    vbytes, _ = capture_bytes((cb, ref) -> MLIR.API.stablehloGetCurrentVersion(cb, ref))
    ver = String(vbytes)
    bytes, sres = capture_bytes((cb, ref) ->
        MLIR.API.stablehloSerializePortableArtifactFromModule(mod, ver, cb, ref, true))
    MLIR.IR.isfailure(MLIR.IR.LogicalResult(sres)) && error("serialize failed")
    return bytes
end

# Trace g at a given batch size; return (mlir_text, portable_artifact_bytes). Keep ctx alive
# until serialization completes.
function trace_and_serialize(batch::Int, W, b)
    ctx = Reactant.ReactantContext()
    x = Reactant.to_rarray(reshape(collect(Float32, 1:(3 * batch)), 3, batch))  # (in=3, batch)
    Wr = Reactant.to_rarray(W)
    br = Reactant.to_rarray(b)
    mod, _ = Compiler.compile_mlir(ctx, g, (x, Wr, br); drop_unsupported_attributes=true)
    text = string(mod)
    bytes = serialize_module(mod)
    return text, bytes
end

function main()
    W = Float32[1 0 0; 0 1 0]      # (out=2, in=3)
    b = Float32[10, 20]            # (out=2,)

    text1, bytes1 = trace_and_serialize(1, W, b)
    _, bytes4 = trace_and_serialize(4, W, b)

    println("=== StableHLO (batch=1) entry signature ===")
    for line in split(text1, '\n')
        occursin("func.func", line) && println(strip(line))
    end
    println("batch=1 artifact: $(length(bytes1)) bytes; batch=4 artifact: $(length(bytes4)) bytes")

    # Round-trip the batch=1 module through the CURRENT (single-module) server runtime.
    mktempdir() do root
        dir = joinpath(root, "dense_spike")
        mkpath(dir)
        write(joinpath(dir, "model.mlir"), bytes1)
        SafeTensors.serialize(joinpath(dir, "weights.safetensors"),
            Dict("W" => W, "b" => b), Dict("argument_order" => JSON3.write(["W", "b"])))
        write(joinpath(dir, "manifest.yaml"), """
        format_version: "2.0"
        name: dense_spike
        executable_inputs:
          - {name: x, dtype: f32, shape: nc, dims: {c: 3}}
        executable_outputs:
          - {name: y, dtype: f32, shape: nc, dims: {c: 2}}
        batching: {compiled_batch_sizes: [1]}
        """)

        backend = ReactantServer.ReactantBackend()
        pool = ReactantServer.resolve_client(backend, ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
        reg = ReactantServer.load_bundles([root])
        entry = ReactantServer.get_model(reg, "dense_spike")
        entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)

        xin = reshape(collect(Float32, 1:3), 3, 1)            # (in=3, batch=1)
        out = ReactantServer.run_model(backend, pool, entry.executable, [ReactantServer.NamedTensor("x", xin)])
        got = vec(out[1].data)
        ref = vec(W * xin .+ b)
        println("server output: $got   reference: $ref")
        @assert isapprox(got, ref; rtol=1e-5) "round-trip mismatch: $got vs $ref"
        println("\nSPIKE PASSED: compile_mlir output serializes and loads/runs in the server")
    end
end

main()
