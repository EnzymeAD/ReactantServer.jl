# Lifecycle observability: formatting helpers and the structured load / unload / residency logs.
#
# These are shared so a model is reported consistently no matter what triggered the change
# (startup, the directory watcher, or the control plane), and so memory is shown from several
# angles at once: the live device allocator (total device pressure, including activations and
# compiled executables), the server's own accounting of device-resident weight bytes, and the
# on-demand weight-cache budget. Device stats come through the backend protocol
# (`device_memory_stats`), so this file stays Reactant-free.

# Render one shape axis: a FIXED dim is its size, the BATCH axis is `n`, a VARIABLE axis is `?`.
function _format_dim(d::Dim)
    d.kind == FIXED && return string(d.size)
    d.kind == BATCH && return "n"
    return "?"
end

# Render a tensor list as e.g. "x: f32[3,224,224,n], mask: f32[n]".
function _format_specs(specs::Vector{TensorSpec})
    isempty(specs) && return "(none)"
    return join(("$(s.name): $(dtype_token(s.dtype))[$(join((_format_dim(d) for d in s.shape), ","))]"
                 for s in specs), ", ")
end

# The compiled batch sizes of a built model; the sole key 0 means a single unbatched module. With
# several input-shape variants the variant count is prefixed (each variant shares one weight set).
function _compiled_sizes(model::LoadedModel)
    ks = _all_batch_sizes(model)
    nvar = length(model.execs)
    base = ks == [0] ? "unbatched" : string(ks)
    return nvar > 1 ? "$(nvar) shapes × $(base)" : base
end

"""
    resident_weight_bytes(registry) -> (bytes, count)

The total device footprint of every currently device-resident model (its weights have been
transferred to the GPU), the server's own accounting of how much device memory its model weights
occupy.
"""
function resident_weight_bytes(registry::ModelRegistry)
    bytes = 0
    count = 0
    for e in values(registry.by_name)
        m = e.executable
        if m !== nothing && m.weights !== nothing
            bytes += m.nbytes
            count += 1
        end
    end
    return (bytes = bytes, count = count)
end

_pct(part::Integer, whole::Integer) = whole > 0 ? round(Int, 100 * part / whole) : 0

"""
    memory_report(backend, pool; registry=nothing, weight_cache=nothing) -> String

A multi-angle, human-readable memory snapshot. Includes each angle whose inputs are available: the
live device allocator (or `device n/a` when unsupported, e.g. CPU), the server's resident-weight
accounting (when a `registry` is given), and the on-demand weight-cache budget (when a
`weight_cache` is given).
"""
function memory_report(backend::AbstractBackend, pool::MemoryPool;
                       registry::Union{ModelRegistry,Nothing}=nothing,
                       weight_cache=nothing)
    parts = String[]
    stats = device_memory_stats(backend, pool)
    if stats === nothing
        push!(parts, "device n/a")
    else
        push!(parts, "device $(Base.format_bytes(stats.free)) free / $(Base.format_bytes(stats.limit)) ($(_pct(stats.free, stats.limit))% free)")
    end
    if registry !== nothing
        rw = resident_weight_bytes(registry)
        push!(parts, "resident weights $(Base.format_bytes(rw.bytes)) / $(rw.count) models")
    end
    if weight_cache !== nothing
        st = weight_cache_stats(weight_cache)
        freeb = max(st.max_bytes - st.resident_bytes, 0)
        push!(parts, "on-demand budget $(Base.format_bytes(st.resident_bytes))/$(Base.format_bytes(st.max_bytes)) ($(_pct(freeb, st.max_bytes))% free)")
    end
    return join(parts, " | ")
end

# Consistent model summary, emitted from build_loaded_model so startup and the watcher format a
# model identically. `source` distinguishes the trigger (:startup, :dynamic).
function log_model_loaded(entry::ModelEntry, model::LoadedModel; source::Symbol, memory::AbstractString)
    m = entry.manifest
    @info "model loaded" name = entry.name source = source inputs = _format_specs(m.executable_inputs) outputs = _format_specs(m.executable_outputs) batch_sizes = _compiled_sizes(model) weights = Base.format_bytes(model.nbytes) residency = model.state memory = memory
    return nothing
end

log_model_unloaded(name::AbstractString, nbytes::Integer; memory::AbstractString) =
    (@info "model unloaded" name = name freed = Base.format_bytes(nbytes) memory = memory; nothing)

# Debug level: residency moves happen on the request path (on-demand weight cache churn), which
# is far too chatty for the default log surface. Enable with JULIA_DEBUG=ReactantServer.
log_residency_change(name::AbstractString, from, to, nbytes::Integer; memory::AbstractString) =
    (@debug "residency: model moved" name = name from = from to = to bytes = Base.format_bytes(nbytes) memory = memory; nothing)
