# Export several ResNetV2-50 BiT models (Luximm.jl, random weights, one seed each) into a residency
# stress repository, with a host reference output for each, and add them to its catalog.toml.
#
#   julia tools/residency_stress/export_bit.jl OUT_DIR [N] [LUXIMM_PATH]
#
# N defaults to 4 and LUXIMM_PATH to ~/Git/Luximm.jl. Builds its own environment under
# OUT_DIR/.bit-env (Luximm, Lux, Reactant, and this tree's ReactantServerExport), since Luximm is not
# part of this repository. Needs no GPU: tracing and the reference forward pass run on the CPU.
#
# RS_STABLEHLO_TARGET=<version> re-serializes each exported module as a StableHLO portable artifact
# of that version, for a worker whose Reactant is older than the one Luximm resolves to (print a
# worker's version with `stablehloGetCurrentVersion`, see gen_models.jl's environment).

using Pkg

const OUT = abspath(ARGS[1])
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
const LUXIMM = length(ARGS) >= 3 ? ARGS[3] : joinpath(homedir(), "Git", "Luximm.jl")
const EXPORT_PKG = normpath(joinpath(@__DIR__, "..", "..", "packages", "ReactantServerExport"))

env = joinpath(OUT, ".bit-env")
mkpath(env)
Pkg.activate(env)
if !isfile(joinpath(env, "Manifest.toml"))
    Pkg.develop([PackageSpec(path = LUXIMM), PackageSpec(path = EXPORT_PKG)])
    Pkg.add(["Lux", "Reactant", "Random", "TOML"])
    # NNlib 0.9.54 passes `count_include_pad` to meanpool, which Reactant 0.2.299's NNlib overlay
    # does not accept, so tracing BiT fails; 0.9.34 (Luximm's floor) traces.
    Pkg.add(PackageSpec(name = "NNlib", version = "0.9.34"))
end
Pkg.instantiate()

using Luximm, Lux, Reactant, Random, TOML
using ReactantServerExport: ReactantServerExport, export_bundle

Reactant.set_default_backend("cpu")
const MLIR = Reactant.MLIR

const VARIANT = :resnetv2_50x1_bit_goog_in21k

# Re-serialize a bundle's modules at an older StableHLO version, in place.
function retarget!(dir, version)
    for f in readdir(dir)
        endswith(f, ".mlir") || continue
        p = joinpath(dir, f)
        bytes = read(p)
        ctx = Reactant.ReactantContext()
        out = MLIR.IR.@with_context ctx begin
            m = MLIR.IR.Module(MLIR.API.stablehloDeserializePortableArtifactNoError(String(copy(bytes)), ctx))
            cb = @cfunction(MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any))
            ref = Ref(IOBuffer())
            res = MLIR.API.stablehloSerializePortableArtifactFromModule(m, version, cb, ref, true)
            MLIR.IR.isfailure(MLIR.IR.LogicalResult(res)) && error("cannot serialize $p at StableHLO $version")
            take!(ref[])
        end
        write(p, out)
    end
    return dir
end

function main()
    models_dir = joinpath(OUT, "models")
    mkpath(models_dir)
    entries = Dict{String, Any}[]
    target = get(ENV, "RS_STABLEHLO_TARGET", "")
    for i in 1:N
        name = "bit50-$i"
        model = create_model(VARIANT; num_classes = 1000)
        ps, st = Lux.setup(Xoshiro(1000 + i), model)
        st = Lux.testmode(st)
        x = randn(Xoshiro(7), Float32, 224, 224, 3, 1)
        y, _ = model(x, ps, st)                      # host reference
        dir = joinpath(models_dir, name)
        isdir(dir) && rm(dir; recursive = true)
        export_bundle(:lux, model, ps, st, x[:, :, :, 1:1]; dir, name, batch_sizes = [1])
        isempty(target) || retarget!(dir, target)
        inp = joinpath(OUT, "bit-input.f32")
        write(inp, x)                                # Julia WHCN memory == row-major NCHW
        nbytes = sum(sizeof, ReactantServerExport._named_leaves(ps) .|> last; init = 0)
        push!(
            entries, Dict{String, Any}(
                "kind" => "bit", "name" => name, "dir" => dir, "input" => inp,
                "input_name" => "input", "input_shape" => [1, 3, 224, 224],
                "reference" => Float64.(vec(y)), "rtol" => 2.0e-2, "outputs" => length(y),
                "batch_sizes" => [1], "variants" => Int[], "alts" => Any[], "weight_bytes" => nbytes,
            )
        )
        @info "exported" name weights = Base.format_bytes(nbytes)
    end
    catalog = joinpath(OUT, "catalog.toml")
    old = isfile(catalog) ? TOML.parsefile(catalog) : Dict{String, Any}("profile" => "bit")
    keep = filter(m -> m["kind"] != "bit", get(old, "model", Any[]))
    old["model"] = vcat(keep, entries)
    open(io -> TOML.print(io, old), catalog, "w")
    return 0
end

exit(main())
