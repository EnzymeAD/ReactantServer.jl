# Load-time TF32 handling for portable StableHLO artifacts: the AUTO-mode strip and the F32-mode
# precision pin (see `NumericsMode`).
#
# AUTO (maybe_strip_tf32!): bundles are lowered once with TF32 baked into stablehlo.dot_general as
# an explicit DotAlgorithm (lhs/rhs_precision_type = tf32, accumulation_type = f32), so one
# artifact is portable across pre- and post-Ampere targets. TF32 as an explicit algorithm is a
# hard XLA compile error on hardware that cannot run it (the StableHLO spec raises rather than
# falling back), so when the compile target is not Ampere+ the worker strips the algorithm at load
# time, before XLA sees the module. Removal lets the op fall back to its precision_config
# (DEFAULT), which is plain f32 on these targets. Numerics therefore FOLLOW THE HARDWARE: DEFAULT
# resolves to TF32 on capable GPUs, both for surviving explicit algorithms and for ops that never
# carried precision info at all (CPU-lowered MLIR).
#
# F32 (pin_f32!): the opposite surgery, for deployments that need hardware-invariant numerics
# (validated medical-device configurations). TF32 algorithms are rewritten to f32 and every
# algorithm-free f32 dot_general/convolution gets precision_config = HIGHEST, which every XLA GPU
# backend honors (cuBLAS selects the f32 compute type, Triton emits allow_tf32=false, cuDNN uses
# FMA math). TF32 is reachable only through tensor-core matmul instructions, and at the StableHLO
# level those originate exclusively from dot_general and convolution, so pinning these two op
# classes is complete coverage; `assert_f32_pinned` machine-checks that claim per module.
#
# Uses the `_RMLIR = Reactant.MLIR` / `_RXLA = Reactant.XLA` aliases from reactant_backend.jl
# (included just before this file).

const _DOT_GENERAL_OP = "stablehlo.dot_general"

# The raw MLIR text of an attribute (e.g. "#stablehlo.dot_algorithm<lhs_precision_type = tf32, …>").
# `show(::Attribute)` wraps this in `Attribute(#= … =#)`, which neither parses back nor is clean to
# substring-match, so go straight to mlirAttributePrint.
function _attr_text(attr)
    cb = @cfunction(_RMLIR.IR.print_callback, Cvoid, (_RMLIR.API.MlirStringRef, Any))
    ref = Ref(IOBuffer())
    _RMLIR.API.mlirAttributePrint(attr, cb, ref)
    return String(take!(ref[]))
end

"""
    _tf32_capable(platform, major, minor) -> Bool

Pure policy: TF32 is supported only on an NVIDIA CUDA GPU with compute capability >= 8.0 (Ampere).
Everything else (CPU, TPU, ROCm, pre-Ampere NVIDIA) is unsupported.
"""
_tf32_capable(platform::AbstractString, major::Integer, minor::Integer) =
    lowercase(platform) == "cuda" && major >= 8

"""
    tf32_supported(client, device) -> Bool

Whether the XLA `client`/`device` this compile targets can run TF32. Queries Reactant for the
platform name and (for CUDA) the device compute capability. Any non-CUDA platform, any failure, or
an undeterminable capability counts as unsupported: when in doubt, we strip rather than risk a hard
compile error.
"""
function tf32_supported(client, device)
    try
        _RXLA.platform_name(client) == "cuda" || return false
        props = _RXLA.device_properties(device)   # CUDA-only; struct carries .major/.minor
        return _tf32_capable("cuda", Int(props.major), Int(props.minor))
    catch
        return false
    end
end

# Backend-protocol hook (declared in backend.jl; default false for CPU/mock backends).
backend_tf32_capable(::ReactantBackend, pool::MemoryPool) = tf32_supported(pool.client, pool.device)

# The explicit TF32 algorithm attribute on a dot_general op, or nothing. A dot_general that relies
# on precision_config (no `algorithm` attribute) returns nothing and is left untouched: DEFAULT
# already resolves to TF32 on capable GPUs and f32 elsewhere, so it needs no surgery.
function _tf32_algorithm(op)
    _RMLIR.IR.name(op) == _DOT_GENERAL_OP || return nothing
    attr = _RMLIR.IR.getattr(op, "algorithm")           # nothing when absent (Operation.jl)
    attr === nothing && return nothing
    occursin("tf32", _attr_text(attr)) || return nothing
    return attr
end

# An all-f32 DotAlgorithm built from `attr` by swapping its tf32 precision types for f32. Replacing
# only the precision-type fields in the op's own printed attribute guarantees a schema-valid
# round-trip without hardcoding the DotAlgorithm field list; parsed in `ctx` so it attaches.
_f32_algorithm(attr, ctx) =
    parse(_RMLIR.IR.Attribute, replace(_attr_text(attr), "tf32" => "f32"); context=ctx)

# Recurse operation -> regions -> blocks -> operations, converting each TF32 dot_general. We only
# mutate an op's attributes (remove/replace), never erase the op, so it stays in its block and the
# block iterator's next handle remains valid throughout. Returns the number of ops changed.
function _convert_tf32_dot!(op, ctx, force_rewrite::Bool)
    n = 0
    attr = _tf32_algorithm(op)
    if attr !== nothing
        if force_rewrite
            _RMLIR.IR.setattr!(op, "algorithm", _f32_algorithm(attr, ctx))
        else
            _RMLIR.IR.rmattr!(op, "algorithm")
        end
        n += 1
    end
    for region in op, block in region, inner in block
        n += _convert_tf32_dot!(inner, ctx, force_rewrite)
    end
    return n
end

"""
    maybe_strip_tf32!(mod; force_rewrite=false) -> Int

Walk a StableHLO `MLIR.IR.Module` and neutralize every `stablehlo.dot_general` whose `algorithm`
attribute specifies TF32. By default the `algorithm` attribute is removed, so the op falls back to
`precision_config` (DEFAULT = plain f32 on non-TF32 targets) and the algorithm/precision
mutual-exclusivity rule is satisfied. With `force_rewrite=true` the attribute is instead replaced by
an all-f32 `DotAlgorithm`. Ops without an explicit TF32 algorithm are left untouched. Returns the
number of ops changed; the pass is idempotent and safe to run on an already-stripped module.

Operates in the module's own context, in place. Call only when the compile target does not support
TF32 (see [`tf32_supported`](@ref)); on a capable device the TF32 algorithm must be preserved.
"""
function maybe_strip_tf32!(mod::_RMLIR.IR.Module; force_rewrite::Bool=false)
    ctx = _RMLIR.IR.context(mod)
    n = 0
    for op in _RMLIR.IR.body(mod)
        n += _convert_tf32_dot!(op, ctx, force_rewrite)
    end
    if n > 0
        @warn "Compile target does not support TF32: converted $n stablehlo.dot_general op(s) from TF32 to F32. Numerics will differ from the TF32 path." force_rewrite
    else
        @debug "maybe_strip_tf32!: no TF32 dot_general algorithms present; nothing to strip"
    end
    return n
end

# ---------------------------------------------------------------------------------------------
# NUMERICS_F32: pin full-f32 precision so numerics are identical across GPU generations.
# ---------------------------------------------------------------------------------------------

const _CONVOLUTION_OP = "stablehlo.convolution"

# Both matmul operands are f32 tensors. Only f32 ops are pinned: HIGHEST on f16/bf16/fp8 operands
# would change their semantics (upcast to f32 compute) and forfeit the performance those dtypes
# were chosen for; TF32 cannot affect them anyway (it is an f32-input tensor-core format).
function _f32_operands(op)
    _RMLIR.IR.noperands(op) >= 2 || return false
    for i in 1:2
        t = _RMLIR.IR.type(_RMLIR.IR.operand(op, i))
        _RMLIR.IR.isshaped(t) || return false
        try
            _RMLIR.IR.julia_type(eltype(t)) == Float32 || return false
        catch
            return false                          # exotic element type: leave the op alone
        end
    end
    return true
end

_highest_precision_attr(ctx) =
    parse(_RMLIR.IR.Attribute, "[#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]";
          context=ctx)

# True when the op's precision_config already names HIGHEST for every operand.
function _is_highest(op)
    attr = _RMLIR.IR.getattr(op, "precision_config")
    attr === nothing && return false
    txt = _attr_text(attr)
    return occursin("HIGHEST", txt) && !occursin("DEFAULT", txt) && !occursin(r"\bHIGH\b", txt)
end

# Recurse and pin one op; returns (algorithms_rewritten, dots_pinned, convs_pinned) increments.
function _pin_f32_op!(op, ctx)
    rewrote = dots = convs = 0
    opname = _RMLIR.IR.name(op)
    if opname == _DOT_GENERAL_OP
        attr = _tf32_algorithm(op)
        if attr !== nothing
            # An explicit TF32 algorithm: rewrite its precision types to f32. Do NOT also set
            # precision_config (an op with an algorithm must keep precision DEFAULT).
            _RMLIR.IR.setattr!(op, "algorithm", _f32_algorithm(attr, ctx))
            rewrote += 1
        elseif _RMLIR.IR.getattr(op, "algorithm") === nothing && _f32_operands(op) && !_is_highest(op)
            # No explicit algorithm: DEFAULT precision resolves to TF32 on capable GPUs, so pin
            # HIGHEST, which every XLA GPU backend maps to true f32.
            _RMLIR.IR.setattr!(op, "precision_config", _highest_precision_attr(ctx))
            dots += 1
        end
        # A non-TF32 explicit algorithm states intent; leave it untouched.
    elseif opname == _CONVOLUTION_OP && _f32_operands(op) && !_is_highest(op)
        _RMLIR.IR.setattr!(op, "precision_config", _highest_precision_attr(ctx))
        convs += 1
    end
    for region in op, block in region, inner in block
        (r, d, c) = _pin_f32_op!(inner, ctx)
        rewrote += r; dots += d; convs += c
    end
    return (rewrote, dots, convs)
end

"""
    pin_f32!(mod) -> (; algorithms_rewritten, dots_pinned, convs_pinned)

Pin full-f32 matmul/convolution precision on a StableHLO `MLIR.IR.Module`, in place: every
`stablehlo.dot_general` carrying a TF32 `DotAlgorithm` is rewritten to the all-f32 algorithm, and
every algorithm-free `dot_general`/`convolution` whose operands are f32 gets
`precision_config = HIGHEST`. Non-f32 ops (f16/bf16/fp8) and ops with a non-TF32 explicit
algorithm are left untouched. Idempotent. This is the `NUMERICS_F32` compile-time pass; follow it
with [`assert_f32_pinned`](@ref) to machine-check the result.
"""
function pin_f32!(mod::_RMLIR.IR.Module)
    ctx = _RMLIR.IR.context(mod)
    rewrote = dots = convs = 0
    for op in _RMLIR.IR.body(mod)
        (r, d, c) = _pin_f32_op!(op, ctx)
        rewrote += r; dots += d; convs += c
    end
    return (; algorithms_rewritten = rewrote, dots_pinned = dots, convs_pinned = convs)
end

# Ops the pin cannot see into: opaque calls, and ops backed by libraries (cuSOLVER) whose internal
# matmuls are not expressed as dot_general. None appear in the current model fleet; they are
# counted so the invariant's caller can surface them rather than silently assume coverage.
const _F32_OPAQUE_OPS = ("stablehlo.custom_call", "stablehlo.cholesky", "stablehlo.triangular_solve")

function _check_pinned_op(op, unpinned::Vector{String}, opaque::Vector{String})
    opname = _RMLIR.IR.name(op)
    if opname in _F32_OPAQUE_OPS
        push!(opaque, opname)
    elseif (opname == _DOT_GENERAL_OP || opname == _CONVOLUTION_OP) && _f32_operands(op)
        pinned = _is_highest(op)
        if !pinned && opname == _DOT_GENERAL_OP
            # An explicit all-f32 algorithm is an equally hard pin (rewritten TF32 case).
            attr = _RMLIR.IR.getattr(op, "algorithm")
            pinned = attr !== nothing && !occursin("tf32", _attr_text(attr))
        end
        pinned || push!(unpinned, opname)
    end
    for region in op, block in region, inner in block
        _check_pinned_op(inner, unpinned, opaque)
    end
    return nothing
end

"""
    assert_f32_pinned(mod) -> (; opaque_ops)

Machine-check the [`pin_f32!`](@ref) invariant: every f32 `dot_general`/`convolution` in `mod`
carries either `precision_config = HIGHEST` or an explicit non-TF32 algorithm. Throws when any
f32 matmul-class op remains at DEFAULT precision (a pass bug must fail the model load loudly, not
serve unpinned numerics). Returns the names of opaque/library-backed ops the pin cannot govern
(`custom_call`, `cholesky`, `triangular_solve`; empty for the current model fleet) so the caller
can log them.
"""
function assert_f32_pinned(mod::_RMLIR.IR.Module)
    unpinned = String[]
    opaque = String[]
    for op in _RMLIR.IR.body(mod)
        _check_pinned_op(op, unpinned, opaque)
    end
    isempty(unpinned) ||
        throw(ErrorException("numerics=f32 invariant violated: $(length(unpinned)) f32 matmul-class op(s) " *
                             "remain at DEFAULT precision after pin_f32! ($(join(unique(unpinned), ", ")))"))
    return (; opaque_ops = opaque)
end

# ---------------------------------------------------------------------------------------------
# Startup attestation probe: does this worker's hardware+stack actually use TF32, and does the
# f32 pin hold? Runs once in _bring_up, on CUDA only, BEFORE any model compile and before the
# scratch high-water probe (`_probe_max_scratch!`). That probe reads the allocator's monotone
# session peak and takes max_i(peak_i - pinned - maxweight_i) over models, so this probe's
# transient footprint (~3 MB: three 1 MB buffers plus workspace) can only perturb the result if
# it exceeded the largest model's pinned + weights + scratch, which is hundreds of MB for real
# models; buffers and the probe executable are freed eagerly regardless.
# ---------------------------------------------------------------------------------------------

const _PROBE_N = 512

# C = A * B with B = I: every f32 summation order yields C bitwise equal to A (all cross terms
# are exact zeros and x * 1.0 is exact), while a TF32 matmul truncates A's inputs first. A's
# sentinels 1 + k*2^-20 (k < 512) are exact in f32 but all land below half a TF32 ulp of 1.0
# (2^-11), so under TF32 every entry becomes exactly 1.0. A is symmetric so the comparison is
# invariant to the XLA/Julia row/column-major transposition.
const _PROBE_MLIR = """
module {
  func.func @main(%a: tensor<$(_PROBE_N)x$(_PROBE_N)xf32>, %b: tensor<$(_PROBE_N)x$(_PROBE_N)xf32>) -> tensor<$(_PROBE_N)x$(_PROBE_N)xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<$(_PROBE_N)x$(_PROBE_N)xf32>, tensor<$(_PROBE_N)x$(_PROBE_N)xf32>) -> tensor<$(_PROBE_N)x$(_PROBE_N)xf32>
    return %0 : tensor<$(_PROBE_N)x$(_PROBE_N)xf32>
  }
}
"""

function _probe_sentinel_matrix()
    A = Matrix{Float32}(undef, _PROBE_N, _PROBE_N)
    for j in 1:_PROBE_N, i in 1:_PROBE_N
        A[i, j] = 1.0f0 + Float32(((i + j) % _PROBE_N) * 2.0^-20)
    end
    return A
end

function _probe_identity_matrix()
    B = zeros(Float32, _PROBE_N, _PROBE_N)
    for i in 1:_PROBE_N
        B[i, i] = 1.0f0
    end
    return B
end

# The probe module as portable-artifact bytes (what compile_artifact consumes), serialized at the
# current StableHLO version. Mirrors the test fixture `stablehlo_artifact`.
function _probe_artifact(ctx)
    _RMLIR.IR.activate(ctx)
    try
        m = parse(_RMLIR.IR.Module, _PROBE_MLIR)
        cb = @cfunction(_RMLIR.IR.print_callback, Cvoid, (_RMLIR.API.MlirStringRef, Any))
        vref = Ref(IOBuffer())
        _RMLIR.API.stablehloGetCurrentVersion(cb, vref)
        ver = String(take!(vref[]))
        ref = Ref(IOBuffer())
        res = _RMLIR.API.stablehloSerializePortableArtifactFromModule(m, ver, cb, ref, true)
        _RMLIR.IR.isfailure(_RMLIR.IR.LogicalResult(res)) && error("failed to serialize the TF32 probe module")
        return take!(ref[])
    finally
        _RMLIR.IR.deactivate(ctx)
    end
end

# Compile and run the probe matmul under `numerics`, returning the C matrix. Everything device-side
# is freed eagerly (see the block comment above on the scratch-probe interaction).
function _run_probe_leg(backend, pool::MemoryPool, numerics::NumericsMode)
    legpool = MemoryPool(pool.backend, pool.client, pool.device, pool.platform, pool.ctx,
                         pool.autotune, numerics)
    exec = compile_artifact(backend, legpool, _probe_artifact(pool.ctx), 2, 1)
    a = b = nothing
    outs = Any[]
    try
        a = to_device(backend, pool.client, _probe_sentinel_matrix(), pool.device)
        b = to_device(backend, pool.client, _probe_identity_matrix(), pool.device)
        outs = execute_single_device(backend, exec, pool.device, Any[a, b], [false, false], 1)
        C = Matrix{Float32}(undef, _PROBE_N, _PROBE_N)
        to_host!(backend, outs[1], C)
        return C
    finally
        for buf in outs
            free_buffer!(backend, buf)
        end
        a === nothing || free_buffer!(backend, a)
        b === nothing || free_buffer!(backend, b)
        free_executable!(backend, exec)
    end
end

"""
    tf32_probe(backend, pool) -> (; tf32_active, pinned_exact)

Startup numerics attestation. Leg 1 (informational) compiles the probe matmul at DEFAULT
precision (`auto` semantics) and reports whether TF32 arithmetic was actually used by this
worker's hardware+stack. Leg 2 runs only under `numerics = f32`: it compiles through the
as-configured pool (the real `pin_f32!` production path) and **throws** unless the result is
bitwise-exact f32; a pin that does not hold is a bug, not a tolerance. Results are logged; the
returned fields are `true`/`false`, or `nothing` for a leg that did not run or was indeterminate.
"""
function tf32_probe(backend, pool::MemoryPool)
    A = _probe_sentinel_matrix()
    tf32_active = nothing
    try
        C = _run_probe_leg(backend, pool, NUMERICS_AUTO)
        tf32_active = C == A ? false : (all(==(1.0f0), C) ? true : nothing)
        tf32_active === nothing &&
            @warn "TF32 probe: DEFAULT-precision leg returned neither exact-f32 nor the TF32 signature" platform = pool.platform
    catch err
        @warn "TF32 probe: DEFAULT-precision leg failed; skipping detection" exception = (err, catch_backtrace())
    end
    pinned_exact = nothing
    if pool.numerics == NUMERICS_F32
        # No try/catch: under numerics=f32 the attestation is load-bearing; a pin that cannot be
        # demonstrated must fail startup loudly.
        C = _run_probe_leg(backend, pool, NUMERICS_F32)
        pinned_exact = C == A
        pinned_exact ||
            throw(ErrorException("numerics=f32 attestation failed: the pinned probe matmul did not " *
                                 "reproduce exact f32 results on this device ($(pool.platform))"))
    end
    @info "TF32 probe" numerics = pool.numerics tf32_active = tf32_active pinned_exact = pinned_exact
    return (; tf32_active, pinned_exact)
end
