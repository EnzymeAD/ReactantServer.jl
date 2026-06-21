# Compilation of a bundle into a `LoadedModel`. The value types (`ModelSignature`, `LoadedModel`)
# are defined in `runtime/model_types.jl`, included before `registry.jl`.

"""
    build_loaded_model(backend, pool, entry; state=UNPINNED, on_demand=false) -> LoadedModel

Derive the signature from the manifest and compile the StableHLO artifact for the pool's
device. Compilation needs only the parameter count, not the weight buffers, so it always
runs at startup. When on-demand loading is disabled (`weight_cache_bytes == 0`) every model's
weights are loaded to the device and kept resident, the original behavior. When on-demand is
enabled the initial residency follows `state`: `PINNED_DEVICE` loads to the device at startup;
`PINNED_SYSTEM` materializes the host floor and starts evicted from the device; `UNPINNED`
holds no floor and starts fully evicted (the weight cache loads it from the mmap on first
dispatch). Called once per model at startup.
"""
function build_loaded_model(backend::AbstractBackend, pool::MemoryPool, entry::ModelEntry;
                            state::ResidencyState=UNPINNED, on_demand::Bool=false,
                            store::WeightStore=PrivateWeightStore(), source::Symbol=:startup)
    m = entry.manifest
    input_names = String[t.name for t in m.executable_inputs]
    input_eltypes = DataType[julia_type(t.dtype) for t in m.executable_inputs]
    output_names = String[t.name for t in m.executable_outputs]
    output_eltypes = DataType[julia_type(t.dtype) for t in m.executable_outputs]
    n_outputs = length(m.executable_outputs)

    wnames = weight_order(entry.weights)
    # batch_dim is only consulted when dispatching across multiple compiled batch
    # sizes (see _select_exec). When the model has no batch axis at all, default to 0;
    # _select_exec returns the sole executable before reading batch_dim in that case.
    input_batch_dim = m.input_batch_dim === nothing ? 0 : m.input_batch_dim
    # The variable input axes, in (input, axis) order, line up with the manifest's input_shapes and
    # so with the variant keys of entry.mlir_bytes; _select_exec reads the same axes from a request.
    variant_spec = Tuple{String,Int}[]
    for t in m.executable_inputs
        for (ax, dm) in enumerate(t.shape)
            dm.kind == VARIABLE && push!(variant_spec, (t.name, ax))
        end
    end
    # Multiple compiled batch sizes within a variant need a batch axis to dispatch on.
    if any(inner -> length(inner) > 1, values(entry.mlir_bytes)) && m.input_batch_dim === nothing
        error("manifest '$(m.name)' has multiple compiled batch sizes but no input with a batch axis ('n'/'b')")
    end
    sig = ModelSignature(input_names, input_eltypes, wnames, n_outputs, output_names, output_eltypes,
                         input_batch_dim, variant_spec)

    # Weights do not vary with batch size, so they are loaded once and shared.
    nbytes = weights_nbytes(entry.weights, wnames)
    # Initial residency. With on-demand off, every model is resident on the device and never
    # evicted (the original behavior), regardless of state. With on-demand on, the floor follows
    # `state`: PINNED_SYSTEM materializes the host floor and starts device-evicted; UNPINNED holds
    # nothing and loads from the mmap on first dispatch; PINNED_DEVICE loads to the device.
    host_weights = nothing
    weights = nothing
    if !on_demand
        weights = load_pinned_weights(backend, pool, entry.weights, wnames)
    elseif state == PINNED_DEVICE
        host_weights = materialize_host_weights(entry.weights, wnames)
        weights = transfer_to_device(backend, pool, host_weights)
        host_weights = nothing                       # device-pinned never evicts; no host floor needed
    elseif state == PINNED_SYSTEM
        host_weights = host_materialize(store, entry.name, entry.weights, wnames)
    end
    np = num_parameters(sig)
    # One executable per (variant, batch size); all share the single weight set loaded above.
    execs = Dict{VariantKey,Dict{Int,Any}}()
    for (vkey, batchmap) in entry.mlir_bytes
        inner = Dict{Int,Any}()
        for (sz, bytes) in batchmap
            inner[sz] = compile_artifact(backend, pool, bytes, np, n_outputs)
        end
        execs[vkey] = inner
    end
    model = LoadedModel(sig, execs, weights, state, nbytes, host_weights)
    log_model_loaded(entry, model; source=source, memory=memory_report(backend, pool))
    return model
end
