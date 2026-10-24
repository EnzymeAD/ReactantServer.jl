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
    try
        for (i, name) in enumerate(names)
            haskey(st, name) || throw(ErrorException("weight '$name' missing from safetensors"))
            host = read_weight(st, name)             # a contiguous Array of the tensor's Julia shape
            bufs[i] = to_device(backend, pool.client, host, pool.device)
        end
    catch
        _free_partial!(backend, bufs)
        rethrow()
    end
    return bufs
end

# Free the buffers a failed multi-tensor upload had already created, so an allocation failure
# midway (the likely one: out of device memory) does not strand them until GC.
function _free_partial!(backend::AbstractBackend, bufs::Vector{Any})
    for i in eachindex(bufs)
        isassigned(bufs, i) && free_buffer!(backend, bufs[i])
    end
    return nothing
end

"""
    materialize_host_weights(st, names) -> Vector{Any}

Read the named weights from the weights file into resident host Arrays,
in weight_names order. This is the expensive step (a layout copy at host-memory speed); doing
it once at startup and keeping the result resident in RAM lets an on-demand GPU load be a pure
host->device transfer (see `transfer_to_device`) rather than a re-materialization.
"""
function materialize_host_weights(st, names::Vector{String})
    hosts = Vector{Any}(undef, length(names))
    for (i, name) in enumerate(names)
        haskey(st, name) || throw(ErrorException("weight '$name' missing from safetensors"))
        hosts[i] = read_weight(st, name)
    end
    return hosts
end

"""
    host_materialize(store, key, st, names) -> Vector{Any}

Materialize a model's host weight floor through a [`WeightStore`](@ref). The private store
allocates per-worker arrays (identical to `materialize_host_weights`); the shared store backs them
with a node-shared SHM region so same-node workers share one copy. `key` is the model name.
"""
host_materialize(::PrivateWeightStore, key, st, names::Vector{String}; content::UInt64 = UInt64(0)) =
    materialize_host_weights(st, names)

# `content` identifies the weights file's on-disk version (see `weights_file_token`); it keys the
# shared region so a weights-only update materializes a NEW region instead of reusing the stale
# one (which persists in /dev/shm across worker restarts by design).
# `verify` runs after the copy and before the region is published; throwing there leaves it
# unpublished.
function host_materialize(
        store::SharedWeightStore, key, st, names::Vector{String};
        content::UInt64 = UInt64(0), verify = () -> nothing
    )
    specs = [weight_spec(st, n) for n in names]
    fill! = function (arrays)
        for (i, n) in enumerate(names)
            read_weight!(arrays[i], st, n)
        end
        verify()
        return
    end
    return materialize_host_weights!(
        store, key, weights_digest(String(key), specs; content = content),
        specs, fill!
    )
end

"""
    host_materialize(store, entry, names) -> Vector{Any}

Materialize `entry`'s host weight floor, labeled with the version of the weights file `entry`
actually opened (`entry.weights_id`), never with a fresh stat of the path: a deploy that replaces
the file after the bundle loaded must not publish the old bytes under the new version's label for
other workers to reuse. After copying, the file is checked for an in-place modification (same
inode, different size or mtime), which would mean the bytes changed underneath the read; that
fails the load rather than publishing mixed bytes. When the node cannot back a shared region
([`SharedStoreFull`](@ref)), the model gets a private copy instead.
"""
function host_materialize(store::WeightStore, entry::ModelEntry, names::Vector{String})
    st = entry.weights
    if store isa SharedWeightStore
        try
            return host_materialize(
                store, entry.name, st, names; content = weights_file_token(entry.weights_id),
                verify = () -> _check_unmodified(entry)
            )
        catch err
            err isa SharedStoreFull || rethrow()
            @warn "shared host weights: the node cannot back this model's region; using a private copy" model = entry.name exception = err
        end
    end
    arrays = materialize_host_weights(st, names)
    _check_unmodified(entry)
    return arrays
end

# A weights file must be replaced atomically (write a new file, rename it over the old one): the
# open file then keeps the old version's bytes, consistent with `entry.weights_id`. Writing into
# the loaded file in place changes the bytes mid-read, which nothing downstream could detect.
function _check_unmodified(entry::ModelEntry)
    id = entry.weights_id
    id == NO_WEIGHTS_FILE && return nothing
    now = weights_file_id(entry.weights_path)
    if now.device == id.device && now.inode == id.inode && now != id
        throw(
            ErrorException(
                "weights file $(entry.weights_path) was modified in place after it was loaded; " *
                    "deploy weights by writing a new file and renaming it over the old one"
            )
        )
    end
    return nothing
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
    try
        for (i, h) in enumerate(hosts)
            bufs[i] = to_device(backend, pool.client, h, pool.device)
        end
    catch
        _free_partial!(backend, bufs)
        rethrow()
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
        T, dims = weight_spec(st, name)
        total += prod(dims; init = 1) * sizeof(T)
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
