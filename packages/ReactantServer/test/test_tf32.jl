# Load-time TF32 stripping (maybe_strip_tf32!) and the device-capability policy. Pure MLIR, no XLA
# compile and no GPU: we generate a genuine TF32-baked dot_general module with Reactant's own
# lowering on CPU, round-trip it through text into a context we own, then run the pass on it.

using Reactant

const _TMLIR = Reactant.MLIR
const RS = ReactantServer

# Lower `x * y` (a dot_general) to StableHLO and return the module text. With `algorithm` set, bake
# that explicit DotAlgorithm; without it, the op relies on precision_config (no algorithm attr).
function _dot_general_hlo(; algorithm=nothing)
    x = Reactant.to_rarray(ones(Float32, 4, 4))
    y = Reactant.to_rarray(ones(Float32, 4, 4))
    mod = if algorithm === nothing
        @code_hlo(*(x, y))
    else
        with_config(; dot_general_algorithm=algorithm) do
            @code_hlo(*(x, y))
        end
    end
    return repr(mod)
end

# A TF32-baked dot_general module text. The TF32_TF32_F32 preset cannot be constructed through
# Reactant's typed DotGeneralAlgorithm in this version, so we lower a real dot_general with a working
# explicit F32 algorithm and retype only its precision-type fields to tf32 (accumulation_type and the
# tensor element types stay f32). Parsing this is pure IR construction, so tf32's runtime
# (un)supportedness is irrelevant to the test.
_tf32_module_text() = replace(
    _dot_general_hlo(; algorithm=DotGeneralAlgorithmPreset.F32_F32_F32),
    "lhs_precision_type = f32" => "lhs_precision_type = tf32",
    "rhs_precision_type = f32" => "rhs_precision_type = tf32",
)

# Parse StableHLO text into a fresh module we own and run `f(mod)` inside a managed context.
function _with_parsed(f, text)
    if isdefined(Reactant, :registry) && Reactant.registry[] === nothing
        Reactant.initialize_dialect()
    end
    _TMLIR.IR.@with_context Reactant.ReactantContext() begin
        return f(parse(_TMLIR.IR.Module, String(text)))
    end
end

@testset "tf32 capability policy" begin
    @test RS._tf32_capable("cuda", 8, 0)         # Ampere
    @test RS._tf32_capable("cuda", 8, 6)         # A6000 / Ampere
    @test RS._tf32_capable("cuda", 9, 0)         # Hopper
    @test !RS._tf32_capable("cuda", 7, 5)        # T4 (Turing)
    @test !RS._tf32_capable("cuda", 7, 0)        # Volta
    @test !RS._tf32_capable("cpu", 0, 0)
    @test !RS._tf32_capable("rocm", 9, 0)
end

@testset "tf32 strip removes the algorithm" begin
    text = _tf32_module_text()
    @test occursin("tf32", text)

    _with_parsed(text) do mod
        n = @test_logs (:warn,) RS.maybe_strip_tf32!(mod)
        @test n == 1
        s = repr(mod)
        @test !occursin("tf32", s)
        @test !occursin("algorithm", s)         # removed -> falls back to precision_config
        @test RS.maybe_strip_tf32!(mod) == 0     # idempotent
    end
end

@testset "tf32 force_rewrite yields an all-f32 algorithm" begin
    text = _tf32_module_text()
    _with_parsed(text) do mod
        n = RS.maybe_strip_tf32!(mod; force_rewrite=true)
        @test n == 1
        s = repr(mod)
        @test !occursin("tf32", s)
        @test occursin("algorithm = <lhs_precision_type = f32", s)
        @test RS.maybe_strip_tf32!(mod) == 0     # idempotent
    end
end

@testset "tf32 leaves precision_config dot_general untouched" begin
    text = _dot_general_hlo()                    # no explicit algorithm
    @test !occursin("tf32", text)
    _with_parsed(text) do mod
        before = repr(mod)
        n = @test_logs RS.maybe_strip_tf32!(mod) # N == 0 emits no warn
        @test n == 0
        @test repr(mod) == before
    end
end
