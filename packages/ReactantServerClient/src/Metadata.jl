# Model I/O metadata and the dry-run IO validator.
#
# The client can fetch a model's true input/output spec either online (the server's ModelMetadata
# RPC) or offline (a manifest YAML, via Core's encode_model_metadata so the wire shapes match the
# server exactly). `validate_io` then runs the user's own infer_encode_chunk! / infer_decode_chunk!
# once against a synthetic, spec-shaped request and response, with no inference round trip, so
# shape and indexing mismatches surface as ordinary Julia exceptions.

# ---- forced bounds checking for the dry run ----
#
# The built-in `@inbounds` is a compile-time elision governed by the --check-bounds startup flag;
# it cannot be toggled per call. So we provide `@infer_inbounds`, a drop-in for `@inbounds` in IO
# code that compiles both a checked and an elided version of the block and picks between them at
# runtime from a scoped flag. Normally (flag unset) it elides like `@inbounds`; inside
# `with_bounds_checks` (which `validate_io` enters) it takes the checked branch, so the dry run
# catches out-of-bounds indexing without requiring --check-bounds=yes. This only protects blocks
# written with `@infer_inbounds`; bare `@inbounds` still needs --check-bounds=yes (auto mode
# bounds-checks un-annotated indexing anyway). --check-bounds=no defeats all of this.

const _FORCE_BOUNDS = Base.ScopedValues.ScopedValue{Bool}(false)

"""
    with_bounds_checks(f)

Run `f` with forced bounds checking enabled for any [`@infer_inbounds`](@ref) blocks reached during
the call (task-local, restored on exit). [`validate_io`](@ref) uses this so the dry run honors
bounds checks even where the IO would normally elide them.
"""
with_bounds_checks(f) = Base.ScopedValues.with(f, _FORCE_BOUNDS => true)

"""
    @infer_inbounds expr

Like `@inbounds`, but the elision is conditional: outside a [`with_bounds_checks`](@ref) context the
wrapped `expr` runs with bounds checks elided (as `@inbounds`); inside one it runs with bounds
checks. Use it instead of `@inbounds` in `infer_encode_chunk!` / `infer_decode_chunk!` so a
[`validate_io`](@ref) dry run still catches out-of-bounds indexing. Wrap a whole loop or block so
the runtime branch is taken once, not per element.

```julia
function ReactantServerClient.infer_decode_chunk!(io::MyIO, r, response)
    out = InferOutput("OUTPUT__0", response, Float32)
    @infer_inbounds for (j, i) in enumerate(r)
        io.results[i] = collect(out[:, j])
    end
    return nothing
end
```
"""
macro infer_inbounds(ex)
    quote
        if $(_FORCE_BOUNDS)[]
            $(esc(ex))
        else
            @inbounds $(esc(ex))
        end
    end
end

struct TensorMeta
    name::String
    datatype::String    # KServe wire string, e.g. "FP32"
    shape::Vector{Int}  # Julia column-major (batch axis last); -1 marks the batch axis and dynamic dims
end

struct ModelIOSpec
    inputs::Dict{String,TensorMeta}
    outputs::Dict{String,TensorMeta}
    input_order::Vector{String}
    output_order::Vector{String}
end

# The wire advertises shapes row-major (KServe); reverse to the Julia column-major view the user
# works in, so an introspected shape matches the order used to build the array.
_tensor_meta(t) = TensorMeta(t.name, t.datatype, reverse(Int[Int(s) for s in t.shape]))

function _parse_io_spec(resp::ModelMetadataResponse)
    inputs = Dict{String,TensorMeta}()
    outputs = Dict{String,TensorMeta}()
    in_order = String[]
    out_order = String[]
    for t in resp.inputs
        inputs[t.name] = _tensor_meta(t)
        push!(in_order, t.name)
    end
    for t in resp.outputs
        outputs[t.name] = _tensor_meta(t)
        push!(out_order, t.name)
    end
    return ModelIOSpec(inputs, outputs, in_order, out_order)
end

"""
    model_io_spec(model::KServeModel) -> ModelIOSpec

Fetch a model's input/output spec from a running server over the ModelMetadata RPC. Throws if the
server is unreachable or does not implement ModelMetadata; the call is explicit, so failing loudly
is intended. Pair with [`validate_io`](@ref) or use it to introspect a model's I/O directly. The
reported shapes are Julia column-major (batch axis last), matching the order you build arrays in;
the row-major KServe wire shape is reversed away by the client.
"""
function model_io_spec(model::KServeModel)
    resp = grpc_sync_request(
        grpc_metadata_client(model),
        ModelMetadataRequest(name = model_name(model)),
    )
    return _parse_io_spec(resp)
end

"""
    manifest_io_spec(path) -> ModelIOSpec

Load a model's input/output spec from a manifest YAML at `path`, with no running server. Reuses the
server's own wire encoding (`encode_model_metadata`), so the result matches [`model_io_spec`](@ref)
for the same model. Suitable for an offline precompile or build-time check. The reported shapes are
Julia column-major (batch axis last), so they read in the same axis order as the manifest's einsum
letters.
"""
function manifest_io_spec(path::AbstractString)
    manifest = load_manifest(path)
    resp = encode_model_metadata(manifest.name, manifest, "")
    return _parse_io_spec(resp)
end

# ---- shape compatibility ----

# Element-wise compatibility; -1 in `meta_wire` is a wildcard (batch or a dynamic axis the user
# fixed). Equal ranks required.
function _shapes_match(user_wire, meta_wire)
    length(user_wire) == length(meta_wire) || return false
    for (u, m) in zip(user_wire, meta_wire)
        m == -1 && continue
        u == m || return false
    end
    return true
end

# Can the user's shape be reconciled with the model's? Both are Julia column-major. Inputs include
# the batch axis on both sides. Outputs are declared per item (no batch), so also try dropping a
# trailing batch axis from the model shape (the batch axis is last in column-major order).
function _alignable(user_shape, meta_shape, user_has_batch)
    user_has_batch && return _shapes_match(user_shape, meta_shape)
    if length(meta_shape) == length(user_shape) + 1 && _shapes_match(user_shape, @view meta_shape[1:end-1])
        return true
    end
    return _shapes_match(user_shape, meta_shape)
end

function _check_shape(name, kind, user_shape, meta_shape; user_has_batch::Bool)
    _alignable(user_shape, meta_shape, user_has_batch) && return nothing
    hint = _alignable(reverse(user_shape), meta_shape, user_has_batch) ?
        " The declared shape matches the model only when reversed; check the axis order (shapes " *
        "here are Julia column-major, with the batch axis last)." : ""
    error("$kind '$name' shape $(collect(user_shape)) is incompatible with the model shape " *
          "$(meta_shape); -1 is a wildcard for batch or dynamic dims.$hint")
end

# ---- descriptor / spec validation ----

function _validate_input_descriptors(spec::ModelIOSpec, descriptors)
    for d in descriptors
        meta = get(spec.inputs, d.name, nothing)
        meta === nothing &&
            error("input '$(d.name)' is not an input of this model; model inputs are $(spec.input_order)")
        want = get(KSERVE_OUTPUT_DTYPE_TABLE_REVERSE, d.dtype, nothing)
        want === nothing && continue
        want == meta.datatype ||
            error("input '$(d.name)' declares dtype $(want) but the model expects $(meta.datatype)")
        _check_shape(d.name, "input", d.shape, meta.shape; user_has_batch = true)
    end
end

function _validate_output_specs(spec::ModelIOSpec, specs::Vector{OutputSpec})
    for s in specs
        meta = get(spec.outputs, s.name, nothing)
        meta === nothing &&
            error("output '$(s.name)' is not an output of this model; model outputs are $(spec.output_order)")
        want = get(KSERVE_OUTPUT_DTYPE_TABLE_REVERSE, s.dtype, nothing)
        want === nothing && continue
        want == meta.datatype ||
            error("output '$(s.name)' declares dtype $(want) but the model produces $(meta.datatype)")
        # per_item_dims and the model shape are both Julia column-major; compare directly.
        _check_shape(s.name, "output", s.per_item_dims, meta.shape; user_has_batch = false)
    end
end

# ---- synthetic response ----

# Resolve a model shape to a concrete row-major wire shape for the synthetic response (which mimics
# a real server response, so it must be row-major for InferOutput to reverse back). meta.shape is
# Julia column-major, so reverse it to wire order first: a leading -1 (batch, by KServe convention)
# becomes `batch`; remaining -1 dims are filled from a declared OutputSpec's per-item dims when
# available, else 1.
function _resolve_wire_shape(meta::TensorMeta, batch::Int, declared::Union{OutputSpec,Nothing})
    shape = reverse(meta.shape)
    isempty(shape) && return shape
    shape[1] == -1 && (shape[1] = batch)
    if declared !== nothing
        per_item = reverse(declared.per_item_dims)            # row-major per item
        offset = length(shape) - length(per_item)             # trailing axes are the non-batch dims
        for i in eachindex(per_item)
            j = offset + i
            1 <= j <= length(shape) && shape[j] == -1 && (shape[j] = per_item[i])
        end
    end
    for i in eachindex(shape)
        shape[i] == -1 && (shape[i] = 1)
    end
    return shape
end

function _synth_response(spec::ModelIOSpec, io, r)
    batch = length(r)
    declared = Dict(s.name => s for s in output_specs(io))
    outs = var"ModelInferResponse.InferOutputTensor"[]
    raw = Vector{UInt8}[]
    for name in spec.output_order
        meta = spec.outputs[name]
        dt = get(KSERVE_OUTPUT_DTYPE_TABLE, meta.datatype, nothing)
        if dt === nothing
            @warn "validate_io: output '$name' has dtype $(meta.datatype) with no client mapping; skipping it in the synthetic response"
            continue
        end
        wire = _resolve_wire_shape(meta, batch, get(declared, name, nothing))
        nbytes = sizeof(dt) * (isempty(wire) ? 1 : prod(wire))
        push!(outs, var"ModelInferResponse.InferOutputTensor"(
            name = name, datatype = meta.datatype, shape = Int64[wire...]))
        push!(raw, zeros(UInt8, nbytes))
    end
    return ModelInferResponse(model_name = "validate_io", outputs = outs, raw_output_contents = raw)
end

# ---- the dry run ----

"""
    validate_io(spec::ModelIOSpec, io::AbstractInferenceIO; items=1)
    validate_io(model::KServeModel, io::AbstractInferenceIO; items=1)

Dry-run `io` against a model's true I/O spec without sending an inference request. Runs the user's
`infer_encode_chunk!` and `infer_decode_chunk!` (see [`AbstractInferenceIO`](@ref)) once for the first `items` items
against a synthetic, spec-shaped request and response, checking input/output names, dtypes, and
shapes and surfacing indexing or shape errors in the user's own code as exceptions.

`spec` comes from [`manifest_io_spec`](@ref) (offline) or [`model_io_spec`](@ref) (online); the
`KServeModel` form fetches it online. The harness runs the user's real methods, so it has side
effects (it may write into the io's buffers at positions `1:items`); call it on a representative or
dummy io. Zeroed synthetic data does not exercise data-dependent branches.

The dry run runs the user methods inside [`with_bounds_checks`](@ref), so indexing written with
[`@infer_inbounds`](@ref) is bounds-checked here even though it elides in normal use. Bare
`@inbounds` is not affected by that context; to catch out-of-bounds in bare-`@inbounds` code, start
Julia with `--check-bounds=yes` (which `Pkg.test` does).
"""
function validate_io(spec::ModelIOSpec, io::AbstractInferenceIO; items::Integer = 1)
    n = length(io)
    if n == 0
        @warn "validate_io: io is empty (length 0); nothing to validate"
        return nothing
    end
    r = 1:min(Int(items), n)

    # Declared outputs are checked even if the encode step never references them.
    _validate_output_specs(spec, output_specs(io))

    # A private single-slot pool sized exactly to the dry run: the one slot already spans the
    # whole pool, so no multi-slot span is needed here.
    bytes = max(length(r) * (item_input_bytes(io) + item_output_bytes(io)), 1)
    pool = InferenceBufferPool(bytes; n_slots = 1, use_shm = false)
    slot = acquire_slot!(pool)
    try
        reset_slot!(slot)
        inputs = try
            with_bounds_checks() do
                infer_encode_chunk!(io, r, slot)
            end
        catch ex
            error("infer_encode_chunk!($(typeof(io)), $r) failed during validate_io: " *
                  "$(sprint(showerror, ex))")
        end
        _validate_input_descriptors(spec, _encoded_inputs(inputs))

        response = _synth_response(spec, io, r)
        try
            with_bounds_checks() do
                infer_decode_chunk!(io, r, response)
            end
        catch ex
            error("infer_decode_chunk!($(typeof(io)), $r) failed against a synthetic response shaped " *
                  "from the model spec; this usually means the result handling assumes a different " *
                  "output shape or index than the model produces: $(sprint(showerror, ex))")
        end
    finally
        release_slot!(slot)
    end
    return nothing
end

validate_io(model::KServeModel, io::AbstractInferenceIO; kwargs...) =
    validate_io(model_io_spec(model), io; kwargs...)