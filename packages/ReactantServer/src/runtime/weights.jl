# Loading model weights from safetensors into resident (pinned) device buffers.
#
# Weights are explicit StableHLO arguments. They are loaded once, in the order declared
# by the safetensors __metadata__ argument_order, and held for the server's lifetime by
# the LoadedModel that owns them.

"""
    weight_order(st) -> Vector{String}

Return weight names in StableHLO argument order. Reads the argument_order field from the
safetensors __metadata__ (a JSON-encoded list). A file with tensors but no argument_order
is an error; a file with no tensors yields an empty order.
"""
function weight_order(st)
    md = st.metadata
    if md !== nothing && haskey(md, "argument_order")
        return collect(String, JSON3.read(md["argument_order"], Vector{String}))
    end
    names = collect(keys(st))
    isempty(names) && return String[]
    throw(ErrorException("weights.safetensors has tensors but no 'argument_order' metadata"))
end

function load_pinned_weights(backend::AbstractBackend, pool::MemoryPool, st, names::Vector{String})
    bufs = Vector{Any}(undef, length(names))
    for (i, name) in enumerate(names)
        haskey(st, name) || throw(ErrorException("weight '$name' missing from safetensors"))
        host = collect(st[name])                 # materialize the lazy mmap view to a contiguous Array
        bufs[i] = to_device(backend, pool.client, host, pool.device)
    end
    return bufs
end

"""
    materialize_host_weights(st, names) -> Vector{Any}

Materialize the named weights from the lazy mmap safetensors views into resident host Arrays,
in weight_names order. This is the expensive step (a layout copy at host-memory speed); doing
it once at startup and keeping the result resident in RAM lets an on-demand GPU load be a pure
host->device transfer (see `transfer_to_device`) rather than a re-materialization.
"""
function materialize_host_weights(st, names::Vector{String})
    hosts = Vector{Any}(undef, length(names))
    for (i, name) in enumerate(names)
        haskey(st, name) || throw(ErrorException("weight '$name' missing from safetensors"))
        hosts[i] = collect(st[name])
    end
    return hosts
end

"""
    host_materialize(store, key, st, names) -> Vector{Any}

Materialize a model's host weight floor through a [`WeightStore`](@ref). The private store
allocates per-worker arrays (identical to `materialize_host_weights`); the shared store backs them
with a node-shared SHM region so same-node workers share one copy. `key` is the model name.
"""
host_materialize(::PrivateWeightStore, key, st, names::Vector{String}; content::UInt64=UInt64(0)) =
    materialize_host_weights(st, names)

# `content` identifies the weights file's on-disk version (see `weights_file_token`); it keys the
# shared region so a weights-only update materializes a NEW region instead of reusing the stale
# one (which persists in /dev/shm across worker restarts by design).
function host_materialize(store::SharedWeightStore, key, st, names::Vector{String};
                          content::UInt64=UInt64(0))
    specs = [(eltype(st[n]), size(st[n])) for n in names]
    fill! = function (arrays)
        for (i, n) in enumerate(names)
            copyto!(arrays[i], st[n])
        end
    end
    return materialize_host_weights!(store, key, weights_digest(String(key), specs; content=content),
                                     specs, fill!)
end

"""
    host_release!(store, key) -> nothing

Release a model's host weight floor previously obtained from `host_materialize`. A no-op for the
private store; for the shared store it detaches and, if last on the node, unlinks the region.
The caller must drop its references to the host arrays first.
"""
host_release!(::PrivateWeightStore, key) = nothing
host_release!(store::SharedWeightStore, key) = release_host_weights!(store, key)

"""
    host_rename!(store, old, new) -> nothing

Rekey a model's host weight floor from `old` to `new` after a model rename (the weights are
unchanged, so nothing is re-materialized). A no-op for the private store.
"""
host_rename!(store::WeightStore, old, new) = rename_host_weights!(store, old, new)

"""
    transfer_to_device(backend, pool, hosts) -> Vector{Any}

Transfer already-materialized host weight Arrays to device buffers, in order. This is the only
cost paid on an on-demand load when weights are pinned in host RAM.
"""
function transfer_to_device(backend::AbstractBackend, pool::MemoryPool, hosts::Vector{Any})
    bufs = Vector{Any}(undef, length(hosts))
    for (i, h) in enumerate(hosts)
        bufs[i] = to_device(backend, pool.client, h, pool.device)
    end
    return bufs
end

"""
    weights_nbytes(st, names) -> Int

Total device footprint of the named weights, summed from the lazy safetensors views' shape
and element type. Reads only metadata (no `collect`), so it is cheap and lets the weight
cache budget against a model before its weights are ever loaded.
"""
function weights_nbytes(st, names::Vector{String})
    total = 0
    for name in names
        haskey(st, name) || throw(ErrorException("weight '$name' missing from safetensors"))
        v = st[name]
        total += prod(size(v)) * sizeof(eltype(v))
    end
    return total
end

"""
    free_weights!(backend, bufs) -> nothing

Release every device buffer in `bufs`. Used to evict an unpinned model's resident weights.
"""
function free_weights!(backend::AbstractBackend, bufs)
    for b in bufs
        free_buffer!(backend, b)
    end
    return nothing
end
