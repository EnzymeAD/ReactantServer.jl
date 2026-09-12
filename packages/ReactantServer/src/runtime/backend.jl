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

# Compilation. compile_artifact parses the StableHLO portable artifact and compiles it. The
# device backends accept a `numerics_stats` keyword: a NumericsStats accumulator recording what
# the numerics policy (runtime.numerics, see NumericsMode) did to the module, aggregated across a
# model's artifacts by build_loaded_model for the model-loaded log. Backends that
# `supports_executable_cache` also accept `cache::ExecutableCacheSlot` (see executable_cache.jl)
# and load the compiled program from the bundle's `.cache/` when present instead of compiling.
function compile_artifact end            # (backend, pool, mlir_bytes, num_parameters, num_outputs; numerics_stats, cache) -> executable

# Whether compile_artifact can serialize compiled programs into the per-bundle executable cache.
# Default false (MockBackend has nothing to cache); the Reactant backend reports true when its Reactant
# exposes XLA.serialize_executable and XLA.load_serialized_executable.
supports_executable_cache(::AbstractBackend) = false

mutable struct NumericsStats
    algorithms_rewritten::Int   # explicit TF32 DotAlgorithms rewritten to f32 (NUMERICS_F32)
    dots_pinned::Int            # algorithm-free f32 dot_general ops pinned to HIGHEST (NUMERICS_F32)
    convs_pinned::Int           # f32 convolution ops pinned to HIGHEST (NUMERICS_F32)
    tf32_stripped::Int          # explicit TF32 DotAlgorithms stripped for compilability (NUMERICS_AUTO)
    opaque_ops::Vector{String}  # custom_call/cholesky/triangular_solve ops the pin cannot govern
end
NumericsStats() = NumericsStats(0, 0, 0, 0, String[])

# One-line human summary for the model-loaded log: the mode plus what the pass actually did.
function format_numerics(mode::NumericsMode, s::NumericsStats, tf32_capable::Bool)
    if mode == NUMERICS_F32
        parts = String[]
        s.algorithms_rewritten > 0 && push!(parts, "rewrote $(s.algorithms_rewritten) tf32 algorithm(s)")
        s.dots_pinned > 0 && push!(parts, "pinned $(s.dots_pinned) dot_general")
        s.convs_pinned > 0 && push!(parts, "pinned $(s.convs_pinned) convolution")
        isempty(s.opaque_ops) ||
            push!(parts, "UNGOVERNED opaque ops: $(join(sort(unique(s.opaque_ops)), ", "))")
        return isempty(parts) ? "f32 (nothing to pin)" : "f32 (" * join(parts, ", ") * ")"
    elseif mode == NUMERICS_TF32
        return "tf32"
    end
    s.tf32_stripped > 0 &&
        return "auto (stripped $(s.tf32_stripped) tf32 algorithm(s): target lacks TF32)"
    return tf32_capable ? "auto (tf32 active)" : "auto (f32: target lacks TF32)"
end

# Data movement.
function to_device end                   # (backend, client, array::Array, device) -> buffer
function buffer_eltype end               # (backend, buffer) -> Type
function buffer_size end                 # (backend, buffer) -> Dims
function to_host! end                    # (backend, buffer, dest::Array) -> dest
function free_buffer! end                # (backend, buffer) -> nothing

# Execution. inputs are in StableHLO main argument order (model inputs, then weights).
function execute_single_device end       # (backend, exec, device, buffers, donated, num_outputs) -> Vector{buffer}

# Eager release of a compiled executable's device-side state (command buffers / CUDA graphs live
# outside the BFC arena and are otherwise only reclaimed when GC happens to run the finalizer).
# Called by evict on the dispatch thread once no execution can be in flight. Default: no-op
# (backends whose executables hold no device state, e.g. MockBackend).
free_executable!(::AbstractBackend, exec) = nothing

# Whether this backend's compile target can run TF32 (Ampere+ CUDA). Default false (CPU, mock);
# the Reactant override lives in tf32.jl. Drives the AUTO-mode summary in format_numerics and the
# startup gate for NUMERICS_TF32.
backend_tf32_capable(::AbstractBackend, pool) = false

# Device memory introspection (optional, for observability). Returns a NamedTuple
# `(in_use, limit, free)` of byte counts, or `nothing` when the backend/device cannot report it
# (e.g. the CPU client or MockBackend). Callers degrade gracefully on `nothing`.
device_memory_stats(::AbstractBackend, pool) = nothing

# Reset the device allocator's high-water mark (`peak_in_use`) to the current in-use bytes. Returns
# true when the reset happened. With it, a scratch measurement is `peak - in_use_before` for exactly
# the work run in between; without it (default false) the peak is a monotone session value that
# compile-time autotuning scratch inflates, and callers fall back to the ordering-based estimate.
clear_memory_stats!(::AbstractBackend, pool) = false

# The compiler's static memory accounting for a compiled executable, or `nothing` when the backend
# cannot report it: a NamedTuple with at least `temp` (scratch) and `outputs` byte counts. A lower
# bound on the allocator's view of a run (it excludes allocator rounding and library workspaces),
# available before the program is ever executed.
compiled_memory_stats(::AbstractBackend, exec) = nothing
