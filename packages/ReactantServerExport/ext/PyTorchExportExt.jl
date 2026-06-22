"""
    PyTorchExport

Convert a `torch.nn.Module` to an ReactantServer bundle. The model is traced with
`torch.export.export` and lowered to StableHLO via
`torchax.export.exported_program_to_jax`. The weights and their names are recorded
in the decomposed graph's placeholder (`input_specs`) order, which is the order the
lowered function binds them in (see `_reorder_weights_to_placeholder_order`), then
filtered through `stablehlo.module_kept_var_idx` so that parameters or buffers
pruned by the JAX-based lowering do not desynchronize from the StableHLO entry
signature.

The PyTorch convention is row-major with batch on the leading axis. The bundle
format uses Julia column-major shapes throughout, which describe the same
underlying bytes after axis reversal. Inputs and outputs are written with shape
`reverse(pytorch_shape)` and `batch_axis = ndims - 1`.

Strategy 1 batching only: the model is re-traced once per requested batch size,
emitting one `model.b{N}.mlir` per size. Weights are captured from the first
trace and reused unchanged (they are batch-independent).

float32 matmul precision follows JAX's platform default unless `matmul_precision`
is passed (`"highest"`, `"high"`, `"default"`). The precision is fixed at export
time by the JAX backend platform, not by the deployment GPU: a GPU-initialized
export may lower float32 matmuls at reduced (TF32-style) precision and bake that
into the StableHLO, while a CPU-initialized export always lowers at full precision
and cannot capture reduced precision in the bundle. `export_bundle` reports the
precision and backend in effect once per session; pass `matmul_precision="highest"`
to force full float32 in every case.

float64 models are exported at double precision. JAX disables 64-bit mode by
default, which would canonicalize float64 weights and inputs to float32, so
`_pyimports` enables `jax_enable_x64` once for the process (set once, not toggled
per export: JAX does not reliably honor flipping it mid-session). Genuine float64
data therefore keeps its dtype. PyTorch's weak-scalar promotion, where a float64
Python-float constant binds to the dtype of the tensor it operates on rather than
upcasting it, is reproduced by `_demote_weak_scalar_constants`, which casts 0-dim
float64 constants to float32. A float32 model's incidental float64 scalars thus stay
float32 instead of colliding with float32 operands, while real float64 tensors are
preserved.

For TorchScript artifacts (`.pt` files produced by `torch.jit.script` or
`torch.jit.trace`), use `export_torchscript_bundle` instead. It applies three
workarounds before delegating to `export_bundle`: it re-wraps parameters as
`torch.nn.Parameter` (TorchScript stores them as bare tensors), registers a
torchax handler for the deprecated `aten._convolution.default` overload that
JIT graphs emit, and wraps the `ScriptModule` as a plain `nn.Module` so
`torch.export` accepts it. Export runs with `strict=false` because Dynamo
cannot trace JIT graphs.
"""
module PyTorchExportExt

using PythonCall
import ReactantServerExport
import ReactantServerExport: export_bundle, export_torchscript_bundle, _pyimports,
    _numpy_to_julia, _numpy_dtype_to_julia, _julia_to_numpy_dtype


# Lazy pyimport caches: the package loads without a working Python ML stack so
# the test suite can skip gracefully when torch/torchax are missing.
const _torch = Ref{Py}()
const _torch_export = Ref{Py}()
const _torchax_export = Ref{Py}()
const _numpy = Ref{Py}()
const _jax = Ref{Py}()
const _to_stablehlo_inputs_first = Ref{Py}()

# torchax's `exported_program_to_stablehlo` emits StableHLO whose entry signature
# is `(weights..., inputs...)`, but the server's execution path passes
# `vcat(in_bufs, model.weights)` (inputs first). Re-trace through jax.export with
# the JAX function wrapped so that the entry signature becomes `(inputs..., weights...)`,
# matching the server's convention and LuxExport's output.
#
# torchax extracts the model's states in `params + buffers + lifted-constants` order,
# but `torch.export` can order the graph placeholders differently (e.g. params,
# constant, buffers). The `func` returned by `exported_program_to_jax` binds the
# weight list to the graph placeholders by position, so feeding the extraction-order
# list to a graph with a different placeholder order binds the wrong tensor to each
# placeholder (a scalar constant where a conv buffer is expected, etc.), silently
# corrupting results for any model whose orders diverge.
#
# `_reorder_weights_to_placeholder_order` fixes this without monkey patching torchax:
# it recovers the decomposed `ExportedProgram` that `func` binds to (captured in
# `func`'s closure) and permutes the already-converted weights into the graph's own
# `input_specs` order. Reading the order from the exact program `func` uses avoids any
# assumption about torchax's decomposition sequence; an incompatible torchax fails
# loudly here rather than producing silently wrong results. The placeholder targets
# double as the weight names the bundle records, so `_to_stablehlo_inputs_first`
# returns them as `names`, paired positionally with the reordered `weights`.
#
# jax 64-bit mode is enabled once for the process (see `_pyimports`) so genuinely
# double-precision models keep their dtype. PyTorch, however, treats a float64 *scalar*
# (a Python float literal) as a weak type that binds to the dtype of the tensor it
# operates on rather than upcasting it; jax materializes it as a strong float64 array
# that would upcast and then collide with float32 operands (e.g. a 0.001 constant
# reaching a float32 conv). `_demote_weak_scalar_constants` reproduces torch's
# weak-scalar promotion by casting 0-dim float64 constants to float32, so the constant's
# precision follows its operands: it stays float32 against float32, and jax promotes it
# back to float64 wherever it meets a genuine float64 tensor. This makes the lowering
# correct independent of the global x64 flag and avoids toggling it per export (jax does
# not reliably honor flipping `jax_enable_x64` mid-session).
const _STABLEHLO_INPUTS_FIRST_PY = """
import jax as _jax
import jax.numpy as _jnp
import torch as _torch
import torchax as _torchax
import torchax.export as _txe

def _reorder_weights_to_placeholder_order(weights, func):
    # `func` is the closure returned by `exported_program_to_jax`; it runs the
    # interpreter over a decomposed `ExportedProgram` it captured. Recover that
    # program by type (independent of closure cell order).
    decomposed = next(
        (cell.cell_contents for cell in (func.__closure__ or [])
         if isinstance(cell.cell_contents, _torch.export.ExportedProgram)),
        None,
    )
    if decomposed is None:
        raise RuntimeError(
            "could not recover the decomposed ExportedProgram from torchax's func "
            "closure; the installed torchax version is incompatible with PyTorchExport"
        )
    _InputKind = _torch.export.graph_signature.InputKind
    gs = decomposed.graph_signature
    # The order torchax extracted `weights` in (mirrors its
    # `_extract_states_from_exported_program`).
    extracted = (list(gs.parameters) + list(gs.buffers)
                 + list(getattr(gs, "lifted_tensor_constants", [])))
    # The order the interpreter consumes the states in: the graph's own placeholders.
    placeholder_order = [s.target for s in gs.input_specs if s.kind != _InputKind.USER_INPUT]
    if sorted(placeholder_order) != sorted(extracted):
        raise RuntimeError(
            "torchax state/placeholder set mismatch; the installed torchax version is "
            "incompatible with PyTorchExport"
        )
    pos = {name: i for i, name in enumerate(extracted)}
    return placeholder_order, [weights[pos[target]] for target in placeholder_order]

def _demote_weak_scalar_constants(weights):
    # Reproduce PyTorch's weak-scalar promotion: a float64 *scalar* constant binds to
    # the dtype of the tensor it operates on instead of upcasting it. Cast 0-dim float64
    # constants to float32; genuine multi-dim float64 data is left alone, and jax
    # promotes a demoted scalar back to float64 wherever it meets a real float64 tensor.
    return [w.astype(_jnp.float32)
            if (getattr(w, "ndim", None) == 0 and getattr(w, "dtype", None) == _jnp.float64)
            else w
            for w in weights]

def _to_stablehlo_inputs_first(exported_program):
    weights, func = _txe.exported_program_to_jax(exported_program)
    # torchax extracts `weights` in params+buffers+constants order, which can diverge
    # from the order `func` binds them to the graph placeholders. Reorder into
    # placeholder order so each weight reaches the correct placeholder; the placeholder
    # targets are the weight names the bundle records.
    names, weights = _reorder_weights_to_placeholder_order(weights, func)
    weights = _demote_weak_scalar_constants(weights)
    jax_avals = _txe.extract_avals(exported_program)
    def reordered(inputs, weights):
        return func(weights, inputs)
    jax_export = _jax.export.export(_jax.jit(reordered))((jax_avals,), weights)
    return weights, jax_export, len(jax_avals), names
"""

function _pyimports()
    isassigned(_torch) && return
    _torch[] = pyimport("torch")
    _torch_export[] = pyimport("torch.export")
    _torchax_export[] = pyimport("torchax.export")
    _numpy[] = pyimport("numpy")
    _jax[] = pyimport("jax")
    # Enable jax 64-bit mode once for the process so genuinely double-precision models
    # (float64 weights or inputs) keep their dtype through the lowering. Set once rather
    # than toggled per export: jax treats `jax_enable_x64` as a process-global startup
    # flag and does not reliably honor flipping it mid-session. Incidental weak float64
    # scalar constants are handled deterministically by `_demote_weak_scalar_constants`,
    # so leaving x64 on does not upcast float32 models.
    _jax[].config.update("jax_enable_x64", true)
    pyexec(_STABLEHLO_INPUTS_FIRST_PY, @__MODULE__)
    _to_stablehlo_inputs_first[] = pyeval("_to_stablehlo_inputs_first", @__MODULE__)
    return
end

# TorchScript-specific shims for export_torchscript_bundle. Concerns:
#   1. JIT modules store parameters as plain `torch.Tensor` with a flag, but
#      `torch.export`'s verifier requires real `torch.nn.Parameter`. Walk the
#      module tree and re-wrap each leaf parameter.
#   2. JIT graphs emit `aten._convolution.default` (deprecated, carries cuDNN
#      flags). torchax only registers `aten.convolution.default`. Install a
#      delegating handler that drops the cuDNN flags. This mutates the global
#      torchax dispatch registry, so guard with `_torchscript_patched` to run
#      it once per process.
#   3. `torch.export` expects an `nn.Module`; a bare `ScriptModule` is not
#      one. `_JitWrapper` is a minimal forwarding `nn.Module`.
#   4. torchax's `full_like`/`fill` handler (`_aten_fill`) does not accept the
#      `layout` allocation kwarg that exported graphs carry, so re-register a
#      handler that ignores `layout` and the other allocation kwargs.
#   5. torchax has no `aten.adaptive_max_pool2d` handler. Register one that
#      reduces each adaptive bin with max and returns the (values, indices)
#      tuple the op schema expects (indices are unused by these graphs).
#
# The state-ordering concern these models also hit is handled without monkey
# patching, in `_STABLEHLO_INPUTS_FIRST_PY` on both export paths: see
# `_reorder_weights_to_placeholder_order`.
const _TORCHSCRIPT_PATCHES_PY = """
import torch as _torch
from torchax.ops import ops_registry as _ops_registry, jaten as _jaten

def _fix_jit_parameters(mod):
    for name in list(mod._parameters.keys()):
        p = mod._parameters[name]
        if p is not None and not isinstance(p, _torch.nn.Parameter):
            mod._parameters[name] = _torch.nn.Parameter(p.detach().clone(), requires_grad=False)
    for child in mod._modules.values():
        if child is not None:
            _fix_jit_parameters(child)

def _register_aten_convolution_default_handler():
    aten = _torch.ops.aten
    def _conv_handler(input, weight, bias, stride, padding, dilation, transposed,
                      output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32):
        return _jaten._aten_convolution(input, weight, bias, stride, padding, dilation,
                                         transposed, output_padding, groups)
    _ops_registry.register_torch_dispatch_op(aten._convolution.default, _conv_handler)

def _register_full_like_layout_handler():
    aten = _torch.ops.aten
    jnp = _jaten.jnp
    mappings = _jaten.mappings
    def _fill_handler(x, value, dtype=None, pin_memory=None, memory_format=None,
                      device=None, layout=None, requires_grad=None):
        dt = x.dtype if dtype is None else mappings.t2j_dtype(dtype)
        return jnp.full(x.shape, value, dt)
    _ops_registry.register_torch_dispatch_op(aten.full_like, _fill_handler)
    _ops_registry.register_torch_dispatch_op(aten.fill, _fill_handler)

def _register_adaptive_max_pool2d_handler():
    aten = _torch.ops.aten
    jnp = _jaten.jnp
    def _amp2d(input, output_size):
        H, W = int(input.shape[-2]), int(input.shape[-1])
        oh, ow = int(output_size[0]), int(output_size[1])
        def bins(n, o):
            return [((i * n) // o, ((i + 1) * n + o - 1) // o) for i in range(o)]
        rows = []
        for hs, he in bins(H, oh):
            cols = [jnp.max(input[..., hs:he, ws:we], axis=(-2, -1)) for ws, we in bins(W, ow)]
            rows.append(jnp.stack(cols, axis=-1))
        out = jnp.stack(rows, axis=-2)
        # Indices are part of the op schema but unused by these graphs; return a
        # placeholder (int32 to match jax's default int width and stay quiet).
        return out, jnp.zeros(out.shape, dtype=jnp.int32)
    _ops_registry.register_torch_dispatch_op(aten.adaptive_max_pool2d, _amp2d)

class _JitWrapper(_torch.nn.Module):
    def __init__(self, jit_mod):
        super().__init__()
        self.jit_mod = jit_mod
    def forward(self, *args):
        return self.jit_mod(*args)
"""

const _torchscript_patched = Ref{Bool}(false)

function _pyimports_torchscript()
    _torchscript_patched[] && return
    pyexec(_TORCHSCRIPT_PATCHES_PY, @__MODULE__)
    pyeval("_register_aten_convolution_default_handler", @__MODULE__)()
    pyeval("_register_full_like_layout_handler", @__MODULE__)()
    pyeval("_register_adaptive_max_pool2d_handler", @__MODULE__)()
    _torchscript_patched[] = true
    return
end

# Julia dtype <-> numpy dtype name. The set is exactly what ReactantServerExport.DTYPE_TOKENS
# covers today; extend both together when adding bf16/f8/etc.
const JULIA_TO_NUMPY = Dict{DataType,String}(
    Float16 => "float16", Float32 => "float32", Float64 => "float64",
    Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64",
    UInt8 => "uint8", UInt16 => "uint16", UInt32 => "uint32", UInt64 => "uint64",
    Bool => "bool",
)
const NUMPY_TO_JULIA = Dict{String,DataType}(v => k for (k, v) in JULIA_TO_NUMPY)

function _julia_to_numpy_dtype(::Type{T}) where {T}
    haskey(JULIA_TO_NUMPY, T) || error("PyTorchExport: no numpy dtype mapping for $T")
    return JULIA_TO_NUMPY[T]
end

function _numpy_dtype_to_julia(np_dtype_name::AbstractString)
    haskey(NUMPY_TO_JULIA, np_dtype_name) || error(
        "PyTorchExport: numpy dtype '$np_dtype_name' has no Julia mapping " *
        "(not in ReactantServerExport.DTYPE_TOKENS). Extend ReactantServerExport and PyTorchExport together to add it.")
    return NUMPY_TO_JULIA[np_dtype_name]
end

# Julia col-major bytes are bit-identical to row-major bytes with reversed shape.
# Build a numpy view over the Julia bytes at the PyTorch shape, then copy into a
# torch.Tensor so Python owns the memory across the trace.
function _julia_to_torch(arr::AbstractArray{T}) where {T}
    np = _numpy[]
    torch = _torch[]
    flat = Vector{UInt8}(reinterpret(UInt8, vec(collect(arr))))
    py_bytes = pybytes(flat)
    np_arr = np.frombuffer(py_bytes, dtype=_julia_to_numpy_dtype(T)).reshape(collect(reverse(size(arr))))
    return torch.from_numpy(np_arr.copy())
end

# Numpy / JAX-materialized array -> Julia Array with reversed shape, same bytes.
function _numpy_to_julia(np_arr_py, ::Type{T}) where {T}
    np = _numpy[]
    contig = np.ascontiguousarray(np_arr_py)
    shape_py = pyconvert(Vector{Int}, contig.shape)
    julia_shape = reverse(shape_py)
    raw = pyconvert(Vector{UInt8}, contig.tobytes())
    expected = sizeof(T) * prod(julia_shape)
    length(raw) == expected ||
        error("PyTorchExport: byte-size mismatch ($(length(raw)) vs expected $expected for shape $julia_shape)")
    arr = Array{T}(undef, julia_shape...)
    copyto!(reinterpret(UInt8, vec(arr)), raw)
    return arr
end

# Representative array shaped like `x` but with the trailing Julia axis set to `s`.
function _with_batch(x::AbstractArray, s::Integer)
    axis = ndims(x)
    sz = collect(size(x))
    sz[axis] = Int(s)
    return zeros(eltype(x), sz...)
end

"""
    export_bundle(:pytorch, model, example_inputs::Tuple; dir, name,
                  input_names=nothing, output_name="output", output_names=nothing,
                  batch_sizes=[1], provenance=Dict()) -> dir

Trace `model` (a `torch.nn.Module`) at each batch size and write a bundle.
`example_inputs` is a tuple of Julia arrays whose trailing Julia axis is the
batch axis (PyTorch leading axis). Element types must be in
`ReactantServerExport.DTYPE_TOKENS`. Returns `dir`.

The model may return a single tensor or a tuple/list of tensors; each output
tensor becomes one entry in the bundle's `executable_outputs`. Output names come
from `output_names` (a vector matching the output count); when it is `nothing` or
its length does not match, names fall back to `output_name` for a single output
and to `output_0`, `output_1`, ... otherwise.

`matmul_precision` sets JAX's `jax_default_matmul_precision` before tracing
(`"highest"`, `"high"`, `"default"`); `nothing` leaves it at the platform default.
The precision in effect, and the JAX backend platform, are reported once per
session. See the module docstring for why the export host's platform, not the
deployment GPU, determines whether reduced (TF32-style) precision is baked in.

`strict` selects `torch.export`'s tracing mode. It defaults to `false` (non-strict):
strict mode drives TorchDynamo, which on recent torch queries the current
accelerator's stream and raises against torchax's registered "jax" device. Non-strict
tracing avoids that path and is the recommended default for `torch.export`.
"""
# The non-batch Julia axes (1-based) of input `i` whose size differs across the shape variants.
# These become the variable (`-1`) axes of the executable input and define the variant key. The
# batch axis is the trailing Julia axis and is excluded.
function _variant_axes_of_input(variants::Vector{<:Tuple}, i::Int)
    base = variants[1][i]
    ndim = ndims(base)
    axes = Int[]
    for ax in 1:(ndim - 1)
        any(v -> size(v[i], ax) != size(base, ax), variants) && push!(axes, ax)
    end
    return axes
end

# The variant key for one example-input tuple: the sizes of the variable input axes, in
# (input, axis) order. Lines up with the manifest input_shapes and the server's runtime key.
function _variant_key(inputs_tuple::Tuple, var_axes::Vector{Vector{Int}})
    key = Int[]
    for i in eachindex(var_axes), ax in var_axes[i]
        push!(key, size(inputs_tuple[i], ax))
    end
    return key
end

function export_bundle(::Val{:pytorch}, model, example_inputs::Tuple;
                       dir::AbstractString, name::AbstractString,
                       input_names=nothing, output_name::AbstractString="output",
                       output_names=nothing,
                       batch_sizes::AbstractVector{<:Integer}=[1],
                       shape_variants::Union{Nothing,AbstractVector}=nothing,
                       strict::Bool=false,
                       matmul_precision::Union{Nothing,AbstractString}=nothing,
                       client_inputs=nothing, client_outputs=nothing,
                       provenance=Dict{String,Any}())
    isempty(example_inputs) && error("PyTorchExport: at least one example input is required")
    isempty(batch_sizes) && error("PyTorchExport: batch_sizes cannot be empty")

    # The shape variants to trace. With none given this is the single base shape (the original
    # single-shape path); otherwise each variant is a full example-input tuple at a distinct shape,
    # all sharing one weight set. `example_inputs` is the first variant.
    variants = shape_variants === nothing ? Tuple[Tuple(example_inputs)] :
               Tuple[Tuple(v) for v in shape_variants]
    multishape = shape_variants !== nothing
    n_in = length(example_inputs)
    for (vi, v) in enumerate(variants)
        length(v) == n_in ||
            error("PyTorchExport: shape variant $vi has $(length(v)) inputs, expected $n_in")
        for (i, x) in enumerate(v)
            haskey(ReactantServerExport.DTYPE_TOKENS, eltype(x)) || error(
                "PyTorchExport: variant $vi input $i has element type $(eltype(x)), " *
                "which is not registered in ReactantServerExport.DTYPE_TOKENS")
            eltype(x) === eltype(variants[1][i]) ||
                error("PyTorchExport: variant $vi input $i dtype differs from variant 1")
            ndims(x) === ndims(variants[1][i]) ||
                error("PyTorchExport: variant $vi input $i rank differs from variant 1")
        end
    end

    _pyimports()
    torch = _torch[]
    torchexport = _torch_export[]
    torchaxexport = _torchax_export[]
    np = _numpy[]
    jax = _jax[]

    _report_matmul_precision(jax, matmul_precision)

    innames = input_names === nothing ?
              ["input_$(i - 1)" for i in 1:n_in] : collect(String, input_names)
    length(innames) == n_in || error("PyTorchExport: input_names length mismatch ($n_in inputs, $(length(innames)) names)")

    model.eval()

    # The variable input axes (those that differ across variants); empty in the single-shape case.
    var_axes = Vector{Int}[_variant_axes_of_input(variants, i) for i in 1:n_in]

    modules_by_variant = Dict{Vector{Int},Dict{Int,String}}()
    out_specs_by_variant = Dict{Vector{Int},Vector{ReactantServerExport.IOSpec}}()
    kept_names = String[]
    kept_weights = Any[]
    captured = false

    for v in variants
        vkey = _variant_key(v, var_axes)
        modules = Dict{Int,String}()
        for s in batch_sizes
            julia_inputs = Tuple(_with_batch(x, s) for x in v)
            py_input_tuple = Tuple(_julia_to_torch(x) for x in julia_inputs)
            py_args = pytuple(py_input_tuple)

            exported = torchexport.export(model, py_args; strict=strict)
            result = _to_stablehlo_inputs_first[](exported)
            weights_py = result[0]
            stablehlo = result[1]
            n_inputs_in_signature = pyconvert(Int, result[2])

            modules[Int(s)] = pyconvert(String, stablehlo.mlir_module())

            # Capture weights once: they are identical across variants and batch sizes, so the one
            # weights.safetensors is shared by every compiled program.
            if !captured
                captured = true
                # Weight names in the exact order of `weights_py` (the graph's
                # placeholder/input_specs order). Both come from
                # `_reorder_weights_to_placeholder_order`, so `all_names[k]` always pairs
                # with `weights_py[k-1]` regardless of how torch.export ordered things.
                all_names = String[pyconvert(String, n) for n in result[3]]

                # In the inputs-first signature, flattened positions are
                # [input_0, ..., input_{N-1}, weight_0, ..., weight_{M-1}].
                # `module_kept_var_idx` indexes into this flat list after DCE.
                # Indices in [0, n_inputs) reference inputs (skipped — those are
                # request-time tensors, not bundle weights). Indices >= n_inputs
                # reference weights at offset `i - n_inputs`.
                kept_idx = Int[pyconvert(Int, i) for i in stablehlo.module_kept_var_idx]
                for i in kept_idx
                    i < n_inputs_in_signature && continue
                    w_idx = i - n_inputs_in_signature
                    w_idx < length(all_names) || continue
                    push!(kept_names, all_names[w_idx + 1])
                    jax_arr = weights_py[w_idx]
                    np_arr = np.asarray(jax_arr)
                    T = _numpy_dtype_to_julia(pyconvert(String, np_arr.dtype.name))
                    push!(kept_weights, _numpy_to_julia(np_arr, T))
                end
            end
        end

        # Sample forward pass (first batch size) to capture each output's shape and dtype for this
        # variant; output spatial dims scale with the input shape, so they are reconciled per axis.
        v0 = Tuple(_with_batch(x, first(batch_sizes)) for x in v)
        py_out = model(Tuple(_julia_to_torch(x) for x in v0)...)
        out_tensors = _flatten_outputs(py_out)
        isempty(out_tensors) && error("PyTorchExport: model produced no output tensors")
        onames = _resolve_output_names(output_names, output_name, length(out_tensors))
        specs = ReactantServerExport.IOSpec[]
        for (j, t) in enumerate(out_tensors)
            pyhasattr(t, "shape") ||
                error("PyTorchExport: output $j is not a tensor (nested/non-tensor outputs unsupported)")
            o_np = t.detach().cpu().numpy()
            o_dtype = _numpy_dtype_to_julia(pyconvert(String, o_np.dtype.name))
            o_shape_julia = collect(Int, reverse(pyconvert(Vector{Int}, t.shape)))
            push!(specs, ReactantServerExport.IOSpec(onames[j], o_dtype, o_shape_julia;
                                                 batch_axis=ndims_from_len(length(o_shape_julia))))
        end
        modules_by_variant[vkey] = modules
        out_specs_by_variant[vkey] = specs
    end

    variant_keys = Vector{Int}[_variant_key(v, var_axes) for v in variants]

    # executable_inputs: the first variant's shape with the variable axes marked -1.
    inputs = ReactantServerExport.IOSpec[]
    for i in 1:n_in
        x_at_s = _with_batch(variants[1][i], first(batch_sizes))
        shp = collect(Int, size(x_at_s))
        for ax in var_axes[i]
            shp[ax] = -1
        end
        bax = ndims(x_at_s) - 1
        push!(inputs, ReactantServerExport.IOSpec(innames[i], eltype(x_at_s), shp; batch_axis=bax))
    end

    # executable_outputs: each output axis that varies across variants is marked -1 (the FPN
    # feature maps scale with the input). The batch axis is left as the batch marker.
    base_specs = out_specs_by_variant[variant_keys[1]]
    outputs = ReactantServerExport.IOSpec[]
    for (j, base) in enumerate(base_specs)
        shp = collect(Int, base.shape)
        for ax in 1:length(shp)
            (base.batch_axis !== nothing && (ax - 1) == base.batch_axis) && continue
            any(vk -> out_specs_by_variant[vk][j].shape[ax] != shp[ax], variant_keys) && (shp[ax] = -1)
        end
        push!(outputs, ReactantServerExport.IOSpec(base.name, base.dtype, shp; batch_axis=base.batch_axis))
    end

    torch_v = _try_version(torch)
    torchax_v = _try_version_of("torchax")

    prov = merge(Dict{String,Any}(
        "source_framework" => "pytorch",
        "converter" => "PyTorchExport.jl",
        "torch_version" => torch_v,
        "torchax_version" => torchax_v,
        "batch_sizes" => collect(Int, batch_sizes),
    ), Dict{String,Any}(provenance))
    multishape && (prov["input_shapes"] = variant_keys)

    if multishape
        ReactantServerExport.write_bundle(dir;
            name=name,
            executable_inputs=inputs,
            executable_outputs=outputs,
            modules=modules_by_variant,
            input_shapes=variant_keys,
            weights=[kept_names[i] => kept_weights[i] for i in eachindex(kept_names)],
            client_inputs=client_inputs, client_outputs=client_outputs,
            provenance=prov)
    else
        ReactantServerExport.write_bundle(dir;
            name=name,
            executable_inputs=inputs,
            executable_outputs=outputs,
            modules=modules_by_variant[variant_keys[1]],
            weights=[kept_names[i] => kept_weights[i] for i in eachindex(kept_names)],
            client_inputs=client_inputs, client_outputs=client_outputs,
            provenance=prov)
    end
    return dir
end

# Apply the requested float32 matmul precision (if any) and report, once per
# session, the precision and JAX backend platform in effect for this export.
#
# The precision JAX uses to *lower* float32 matmuls is fixed at export time by the
# platform JAX initialized for, not by the deployment target: jaxlib advertises the
# `cuda`/`gpu` platform whenever GPUs are visible (even with no usable device), and
# only a fully GPU-hidden run takes the CPU path. A GPU-initialized export may lower
# at reduced (TF32-style) precision and bake that into the StableHLO; a CPU export
# always lowers at full precision and cannot capture reduced precision in the
# bundle. `matmul_precision="highest"` forces full float32 in every case. See
# Reactant.jl/pytorch_precision_check.jl for the empirical basis.
function _report_matmul_precision(jax, matmul_precision)
    if matmul_precision !== nothing
        jax.config.update("jax_default_matmul_precision", matmul_precision)
    end
    builtins = pyimport("builtins")
    precision = pyconvert(String, builtins.str(jax.config.jax_default_matmul_precision))
    backend = pyconvert(String, jax.default_backend())
    if backend == "cpu"
        @warn "PyTorchExport: tracing with jax_default_matmul_precision=$(precision) on \
               the JAX \"cpu\" backend. float32 matmuls lower at full precision, so \
               TF32-style reduced precision cannot be captured in this bundle even when \
               it is later deployed on a GPU. To bake in reduced precision, run the \
               export on a GPU-initialized JAX (GPUs visible; TF32 itself requires an \
               Ampere or newer GPU). The lowered precision is frozen into the exported \
               StableHLO at trace time by the JAX platform, not by the deployment GPU." maxlog = 1
    else
        @warn "PyTorchExport: tracing with jax_default_matmul_precision=$(precision) on \
               the JAX \"$(backend)\" backend. A value of None means JAX's platform \
               default, which can be reduced (TF32-style) precision on a GPU-initialized \
               JAX (TF32 requires an Ampere or newer GPU); that choice is frozen into the \
               exported StableHLO and will differ from a CPU run. Set \
               matmul_precision=\"highest\" for full float32. The lowered precision is \
               fixed at trace time by the JAX platform, not by the deployment GPU." maxlog = 1
    end
    return precision
end

# 0-based "last Julia axis" for an array with `n` dimensions, or `nothing` for scalars.
ndims_from_len(n::Integer) = n >= 1 ? Int(n - 1) : nothing

# A model output is either a single tensor or a tuple/list of tensors. Return the
# output tensors as a flat vector of `Py`; the single-tensor case is detected by
# the presence of a `shape` attribute.
function _flatten_outputs(py_out)
    pyhasattr(py_out, "shape") && return Py[py_out]
    outs = Py[]
    for t in py_out
        push!(outs, t)
    end
    return outs
end

# Resolve output names against the actual output count. `output_names`, when given
# and length-matched, wins; otherwise fall back to `output_name` for a lone output
# and to `output_0`, `output_1`, ... for several.
function _resolve_output_names(output_names, output_name, n::Integer)
    if output_names !== nothing
        names = collect(String, output_names)
        length(names) == n && return names
    end
    n == 1 && return [String(output_name)]
    return ["output_$(i - 1)" for i in 1:n]
end

function _try_version(py_mod)
    try
        return pyconvert(String, py_mod.__version__)
    catch
        return "unknown"
    end
end

function _try_version_of(name::AbstractString)
    try
        return pyconvert(String, pyimport(name).__version__)
    catch
        return "unknown"
    end
end

"""
    export_torchscript_bundle(pt_path::AbstractString, example_inputs::Tuple; ...)
    export_torchscript_bundle(jit_module::Py, example_inputs::Tuple; ...)

Convert a TorchScript model (`.pt` file from `torch.jit.script` or
`torch.jit.trace`) into a server bundle. Applies the three TorchScript-only
workarounds described in the module docstring, then delegates to
`export_bundle` with `strict=false`. The path overload loads the `.pt` via
`torch.jit.load(pt_path; map_location)` and records `torchscript_path` in
provenance. The module overload accepts a pre-loaded `ScriptModule` (useful
for in-memory scripts/traces or custom load options).

`wrap` overrides the module wrapper. By default the JIT module is wrapped in
`_JitWrapper` (forwards `*args` to the module). Pass a Python callable taking the
(parameter-fixed) JIT module and returning an `nn.Module` to export a custom
forward, for example a wrapper that returns an intermediate activation and drops
a trailing data-dependent op. The TorchScript patches (parameter re-wrap, conv
handler) are applied to `jit_module` before `wrap` is called.

`client_inputs`/`client_outputs` (each `nothing` or a `Vector{IOSpec}`) are forwarded to
`write_bundle` to declare the wire-facing spec when a shipped `model.jl` postprocess turns the
dense executable outputs into different client outputs (e.g. a variable detection count, encoded
with `-1` for that axis). When you pass these you must also copy a `model.jl` into `dir`.
"""
function export_torchscript_bundle(pt_path::AbstractString,
                                   example_inputs::Tuple;
                                   dir::AbstractString,
                                   name::AbstractString,
                                   input_names=nothing,
                                   output_name::AbstractString="output",
                                   output_names=nothing,
                                   batch_sizes::AbstractVector{<:Integer}=[1],
                                   shape_variants::Union{Nothing,AbstractVector}=nothing,
                                   matmul_precision::Union{Nothing,AbstractString}=nothing,
                                   client_inputs=nothing, client_outputs=nothing,
                                   provenance=Dict{String,Any}(),
                                   map_location="cpu",
                                   wrap=nothing)
    _pyimports()
    jit_mod = _torch[].jit.load(pt_path, map_location=map_location)
    prov = merge(Dict{String,Any}("torchscript_path" => String(pt_path)),
                 Dict{String,Any}(provenance))
    return export_torchscript_bundle(jit_mod, example_inputs;
        dir=dir, name=name, input_names=input_names,
        output_name=output_name, output_names=output_names,
        batch_sizes=batch_sizes, shape_variants=shape_variants, matmul_precision=matmul_precision,
        client_inputs=client_inputs, client_outputs=client_outputs,
        provenance=prov, wrap=wrap)
end

function export_torchscript_bundle(jit_module::Py, example_inputs::Tuple;
                                   dir::AbstractString,
                                   name::AbstractString,
                                   input_names=nothing,
                                   output_name::AbstractString="output",
                                   output_names=nothing,
                                   batch_sizes::AbstractVector{<:Integer}=[1],
                                   shape_variants::Union{Nothing,AbstractVector}=nothing,
                                   matmul_precision::Union{Nothing,AbstractString}=nothing,
                                   client_inputs=nothing, client_outputs=nothing,
                                   provenance=Dict{String,Any}(),
                                   wrap=nothing)
    _pyimports()
    _pyimports_torchscript()
    # Most scripted modules want eval mode, but some bake training=False as a constant and raise
    # "Can't set constant training" on .eval(); they are already in eval, so
    # ignore that specific failure.
    try
        jit_module.eval()
    catch err
        occursin("training", sprint(showerror, err)) || rethrow()
    end
    pyeval("_fix_jit_parameters", @__MODULE__)(jit_module)
    wrapper = wrap === nothing ? pyeval("_JitWrapper", @__MODULE__)(jit_module) : wrap(jit_module)
    prov = merge(Dict{String,Any}("source_subframework" => "torchscript"),
                 Dict{String,Any}(provenance))
    return export_bundle(Val(:pytorch), wrapper, example_inputs;
        dir=dir, name=name, input_names=input_names,
        output_name=output_name, output_names=output_names,
        batch_sizes=batch_sizes, shape_variants=shape_variants,
        strict=false,
        matmul_precision=matmul_precision,
        client_inputs=client_inputs, client_outputs=client_outputs,
        provenance=prov)
end

end # module PyTorchExportExt
