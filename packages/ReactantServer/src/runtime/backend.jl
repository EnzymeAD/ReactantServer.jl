# The narrow backend protocol.
#
# Everything in the server except runtime/reactant_backend.jl talks to a backend through
# these generic functions. A second implementation (a direct PJRT C binding, say) only
# needs to provide methods for these to replace Reactant. MockBackend implements them for
# Reactant-free tests.

abstract type AbstractBackend end

# Lifecycle and device selection.
function make_client end                 # (backend, platform::String; mem_fraction, preallocate, autotune_cache, autotune_cache_dir) -> client
make_context(::AbstractBackend) = nothing  # opaque per-backend compilation context (or nothing)
function select_device end               # (backend, client, ordinal::Int) -> device
function device_ordinal end              # (backend, device) -> Int

# Compilation. compile_artifact parses the StableHLO portable artifact and compiles it.
function compile_artifact end            # (backend, pool, mlir_bytes, num_parameters, num_outputs) -> executable

# Data movement.
function to_device end                   # (backend, client, array::Array, device) -> buffer
function buffer_eltype end               # (backend, buffer) -> Type
function buffer_size end                 # (backend, buffer) -> Dims
function to_host! end                    # (backend, buffer, dest::Array) -> dest
function free_buffer! end                # (backend, buffer) -> nothing

# Execution. inputs are in StableHLO main argument order (model inputs, then weights).
function execute_single_device end       # (backend, exec, device, buffers, donated, num_outputs) -> Vector{buffer}

# Device memory introspection (optional, for observability). Returns a NamedTuple
# `(in_use, limit, free)` of byte counts, or `nothing` when the backend/device cannot report it
# (e.g. the CPU client or MockBackend). Callers degrade gracefully on `nothing`.
device_memory_stats(::AbstractBackend, pool) = nothing
