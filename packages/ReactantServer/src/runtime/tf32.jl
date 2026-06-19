# Load-time TF32 stripping for portable StableHLO artifacts.
#
# Bundles are lowered once with TF32 baked into stablehlo.dot_general as an explicit DotAlgorithm
# (lhs/rhs_precision_type = tf32, accumulation_type = f32), so one artifact is portable across
# pre- and post-Ampere targets. TF32 as an explicit algorithm is a hard XLA compile error on
# hardware that cannot run it (the StableHLO spec raises rather than falling back), so when the
# compile target is not Ampere+ the worker strips the algorithm at load time, before XLA sees the
# module. Removal lets the op fall back to its precision_config (DEFAULT), which is plain f32 on
# these targets. Uses the `_RMLIR = Reactant.MLIR` / `_RXLA = Reactant.XLA` aliases from
# reactant_backend.jl (included just before this file).

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
