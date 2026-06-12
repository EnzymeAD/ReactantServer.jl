# Helpers for building StableHLO portable-artifact bytes and on-disk bundle fixtures.
# These let tests author tiny models without the offline conversion tooling.

using Reactant
using SafeTensors
using JSON3

const _FXLA = Reactant.XLA
const _FMLIR = Reactant.MLIR

# Serialize StableHLO assembly text to portable-artifact bytes (what a bundle's model.mlir holds).
function stablehlo_artifact(text::AbstractString)
    if isdefined(Reactant, :registry) && Reactant.registry[] === nothing
        Reactant.initialize_dialect()
    end
    _FMLIR.IR.@with_context Reactant.ReactantContext() begin
        m = parse(_FMLIR.IR.Module, String(text))
        cb = @cfunction(_FMLIR.IR.print_callback, Cvoid, (_FMLIR.API.MlirStringRef, Any))
        vref = Ref(IOBuffer())
        _FMLIR.API.stablehloGetCurrentVersion(cb, vref)
        ver = String(take!(vref[]))
        ref = Ref(IOBuffer())
        res = _FMLIR.API.stablehloSerializePortableArtifactFromModule(m, ver, cb, ref, true)
        _FMLIR.IR.isfailure(_FMLIR.IR.LogicalResult(res)) && error("failed to serialize fixture")
        return take!(ref[])
    end
end

# Write a complete bundle directory and return its path.
function write_bundle(root::AbstractString, name::AbstractString;
                      manifest_yaml::AbstractString,
                      mlir_text::AbstractString,
                      weights::AbstractDict{String,<:AbstractArray}=Dict{String,Array}(),
                      argument_order::Vector{String}=String[])
    dir = joinpath(root, name)
    mkpath(dir)
    write(joinpath(dir, "manifest.yaml"), manifest_yaml)
    write(joinpath(dir, "model.mlir"), stablehlo_artifact(mlir_text))
    meta = isempty(argument_order) ? nothing : Dict("argument_order" => JSON3.write(argument_order))
    SafeTensors.serialize(joinpath(dir, "weights.safetensors"), weights, meta)
    return dir
end
