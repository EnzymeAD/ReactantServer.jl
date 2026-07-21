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

# A convolution module lowered by Reactant's NNlib path (guaranteed-valid stablehlo.convolution).
function _conv_hlo()
    x = Reactant.to_rarray(ones(Float32, 5, 5, 1, 1))
    w = Reactant.to_rarray(ones(Float32, 2, 2, 1, 1))
    return repr(@code_hlo(RS.NNlib.conv(x, w)))
end

@testset "pin_f32!: HIGHEST on an algorithm-free f32 dot_general" begin
    _with_parsed(_dot_general_hlo()) do mod
        # Before the pin, the invariant must reject the module (DEFAULT precision).
        @test_throws ErrorException RS.assert_f32_pinned(mod)
        st = RS.pin_f32!(mod)
        @test st.algorithms_rewritten == 0
        @test st.dots_pinned == 1
        @test st.convs_pinned == 0
        @test occursin("HIGHEST", repr(mod))
        @test RS.assert_f32_pinned(mod).opaque_ops == String[]
        st2 = RS.pin_f32!(mod)                   # idempotent
        @test st2.dots_pinned == 0
    end
end

@testset "pin_f32!: tf32 algorithm rewritten to f32, no precision added" begin
    _with_parsed(_tf32_module_text()) do mod
        st = RS.pin_f32!(mod)
        @test st.algorithms_rewritten == 1
        @test st.dots_pinned == 0                # algorithm and precision are mutually exclusive
        s = repr(mod)
        @test !occursin("tf32", s)
        @test occursin("lhs_precision_type = f32", s)
        @test !occursin("HIGHEST", s)
        RS.assert_f32_pinned(mod)                # an explicit f32 algorithm is an equally hard pin
        @test RS.pin_f32!(mod).algorithms_rewritten == 0   # idempotent
    end
end

@testset "pin_f32!: non-f32 operands are never touched" begin
    # A pure-text f16 retype of the algorithm-free module: HIGHEST on f16 would change semantics.
    text16 = replace(_dot_general_hlo(), "f32" => "f16")
    _with_parsed(text16) do mod
        before = repr(mod)
        st = RS.pin_f32!(mod)
        @test st.dots_pinned == 0
        @test repr(mod) == before
        RS.assert_f32_pinned(mod)                # nothing f32 to pin => invariant holds vacuously
    end
end

@testset "pin_f32!: f32 convolution pinned" begin
    text = _conv_hlo()
    @test occursin("stablehlo.convolution", text)
    _with_parsed(text) do mod
        st = RS.pin_f32!(mod)
        @test st.convs_pinned == 1
        @test occursin("HIGHEST", repr(mod))
        @test RS.assert_f32_pinned(mod).opaque_ops == String[]
        @test RS.pin_f32!(mod).convs_pinned == 0   # idempotent
    end
end

@testset "tf32 probe on CPU: plain f32 detected, pinned leg attests exactly" begin
    backend = RS.ReactantBackend()
    runtime = RS.RuntimeConfig(RS.CPU_BACKEND, 0, 0.9, true, true)
    pool = RS.resolve_client(backend, runtime)

    # auto (the pool default): the DEFAULT-precision leg runs, the attestation leg is skipped.
    res = RS.tf32_probe(backend, pool)
    @test res.tf32_active == false               # CPU DEFAULT is plain f32
    @test res.pinned_exact === nothing

    # f32: the attestation leg compiles through the real pin path and must be bit-exact.
    pool_f32 = RS.MemoryPool(pool.backend, pool.client, pool.device, pool.platform, pool.ctx,
                             pool.autotune, RS.NUMERICS_F32)
    res2 = RS.tf32_probe(backend, pool_f32)
    @test res2.pinned_exact === true
end
