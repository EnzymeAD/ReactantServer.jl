# The per-request execution path: named host inputs to named host outputs.
#
# Inputs are reordered into the StableHLO main argument order (model inputs in manifest
# order, then the pinned weights), transferred to the device, executed, and read back.
# Transient input and output buffers are released after each call; weights stay resident.

# Pick the executable for a request. First select the input-shape variant from the request's
# variable input axes (empty `variant_spec` => the single default variant), then the batch size
# within it (from the first declared input's batch axis). The manifest uses Julia shape order, so
# batch_dim and variant axes are Julia 1-based once offset.
function _variant_key(sig::ModelSignature, byname)
    isempty(sig.variant_spec) && return VariantKey()
    return Int[size(byname[nm].data, ax) for (nm, ax) in sig.variant_spec]
end

function _select_exec(model::LoadedModel, byname)
    sig = model.sig
    vkey = _variant_key(sig, byname)
    inner = get(model.execs, vkey, nothing)
    inner === nothing &&
        error("no compiled program for input shape variant $vkey (have $(sort(collect(keys(model.execs)))))")
    length(inner) == 1 && return first(values(inner))
    t = byname[sig.input_names[1]]
    julia_axis = sig.batch_dim + 1                  # Julia 0-based -> 1-based index
    batch = size(t.data, julia_axis)
    haskey(inner, batch) ||
        error("no compiled executable for batch size $batch (variant $vkey; have $(sort(collect(keys(inner)))))")
    return inner[batch]
end

function run_model(backend::AbstractBackend, pool::MemoryPool, model::LoadedModel,
                   inputs::AbstractVector{NamedTensor})
    sig = model.sig
    byname = Dict(t.name => t for t in inputs)

    # Every transient buffer is freed in the finally below, success or error. An error mid-call
    # (a device OOM during a transfer is the likely case under memory pressure) must not strand
    # the buffers already transferred. Weights are not transient and are never freed here.
    in_bufs = Any[]
    out_bufs = Any[]
    try
        for (i, nm) in enumerate(sig.input_names)
            haskey(byname, nm) || error("missing required input '$nm'")
            t = byname[nm]
            et = sig.input_eltypes[i]
            eltype(t.data) === et || error("input '$nm' has eltype $(eltype(t.data)), expected $et")
            push!(in_bufs, to_device(backend, pool.client, t.data, pool.device))
        end

        exec = _select_exec(model, byname)
        arg_bufs = vcat(in_bufs, model.weights)
        donated = falses(length(arg_bufs))
        append!(out_bufs, execute_single_device(backend, exec, pool.device, arg_bufs, donated, sig.num_outputs))

        outputs = Vector{NamedTensor}(undef, length(out_bufs))
        for k in eachindex(out_bufs)
            b = out_bufs[k]
            xla_shape = buffer_size(backend, b)
            # Allocate the destination already in the Julia (column-major) shape so the result
            # is a concrete Array{T,N} and downstream callers are not re-specialized on a
            # ReshapedArray. The underlying bytes are the same memory regardless of which view
            # interprets them, so copying row-major buffer bytes into a Julia col-major Array
            # of the reversed shape is the correct logical tensor.
            T = buffer_eltype(backend, b)
            julia_shape = length(xla_shape) <= 1 ? xla_shape : reverse(xla_shape)
            dest = Array{T}(undef, julia_shape)
            to_host!(backend, b, dest)
            nm = k <= length(sig.output_names) ? sig.output_names[k] : "output_$(k - 1)"
            outputs[k] = NamedTensor(nm, dest)
        end
        return outputs
    finally
        for b in in_bufs
            free_buffer!(backend, b)
        end
        for b in out_bufs
            free_buffer!(backend, b)
        end
    end
end
