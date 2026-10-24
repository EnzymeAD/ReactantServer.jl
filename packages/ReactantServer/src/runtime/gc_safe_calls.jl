# The serving hot path's calls into libReactantExtra, made GC-safe.
#
# Julia's garbage collector stops every thread at a safepoint before it runs. A thread inside a
# plain `@ccall` is not at one, so while it runs, any other thread that asks for a collection makes
# EVERY thread wait for the call to return. Reactant's bindings are plain `@ccall`s (and execution
# is a `Base.llvmcall`), and the serving hot path spends most of its time inside them: a weight
# upload copies up to gigabytes, an execution's result wait lasts as long as the program runs. On a
# soak with continuous weight churn (one A6000, 19 uploads a second) that froze the whole process
# for 15 to 120 s at a time, stalling request threads and the GPU lease heartbeat alike.
#
# These are the same C functions Reactant's own wrappers call (same symbols, same arguments, as in
# Reactant 0.2.287's src/mlir/libMLIR_h.jl and src/xla/PJRT), declared `gc_safe = true` so a
# collection can proceed while the call runs. That is sound under two conditions, both met here:
#
#   - The C side must not touch Julia-managed memory the collector could free. Every Julia object
#     a call reads or writes (host arrays, the Refs that receive results, the wrapper objects
#     owning the pointers) is rooted with `GC.@preserve`, and Julia's collector does not move
#     objects, so their addresses stay valid while it runs concurrently.
#   - Errors must return to Julia safely. libReactantExtra reports a failed status by calling the
#     `reactant_err` @cfunction Reactant registers at init, which throws. A cfunction entry
#     switches the thread back to GC-unsafe before running Julia code, and the exception handler
#     that catches the throw restores the thread's GC state, so a throw from inside a gc_safe call
#     unwinds like one from a plain call. test_gc_safe_calls.jl exercises this.
#
# Like VendoredXLA, the functions live here rather than as methods on Reactant's types, and they
# read Reactant's wrapper fields (`client.client`, `device.device`, `exec.exec`, `buffer.buffer`)
# directly. Retire this module once a Reactant release marks its own bindings gc_safe.
module GcSafeXLA

using Reactant: Reactant
using Reactant_jll: Reactant_jll

const XLA = Reactant.XLA
const PJRT = Reactant.XLA.PJRT

"""
    upload(client, array, device) -> PJRT.Buffer

Copy a host array to a new device buffer (`ArrayFromHostBuffer`, as `PJRT.Buffer(client, array,
device)` does). Reactant asks PJRT for "immutable only during call" semantics, so the call returns
once the data is staged and the array may be reused; the device copy completes afterwards.
"""
function upload(client::PJRT.Client, array::Array{T, N}, device::PJRT.Device) where {T, N}
    shape = collect(Int64, reverse(size(array)))
    ptr = GC.@preserve client array device shape begin
        @ccall gc_safe = true Reactant_jll.libReactantExtra.ArrayFromHostBuffer(
            client.client::Ptr{Cvoid}, pointer(array)::Ptr{Cvoid}, XLA.primitive_type(T)::UInt64,
            N::Csize_t, pointer(shape)::Ptr{Int64}, device.device::Ptr{Cvoid}
        )::Ptr{Cvoid}
    end
    return PJRT.Buffer(ptr)
end

"""
    to_host!(buffer, dest) -> dest

Copy a device buffer into the host array `dest` (`BufferToHost`), blocking until the copy is done.
"""
function to_host!(buffer::PJRT.Buffer, dest::Array)
    buffer.buffer == C_NULL && throw(ArgumentError("to_host! on a freed buffer"))
    GC.@preserve buffer dest begin
        @ccall gc_safe = true Reactant_jll.libReactantExtra.BufferToHost(
            buffer.buffer::Ptr{Cvoid}, pointer(dest)::Ptr{Cvoid}
        )::Cvoid
    end
    return dest
end

"""
    free!(buffer) -> nothing

Release a device buffer now (`PjRtBufferFree`, what Reactant's finalizer calls), then null the
pointer and run the finalizer once so it is unregistered and finds nothing left to free. Freeing a
buffer with device work pending can wait for that work, which is why this is gc_safe too.
"""
function free!(buffer::PJRT.Buffer)
    ptr = buffer.buffer
    if ptr != C_NULL && XLA.is_live[]
        GC.@preserve buffer begin
            @ccall gc_safe = true Reactant_jll.libReactantExtra.PjRtBufferFree(ptr::Ptr{Cvoid})::Cvoid
        end
    end
    buffer.buffer = C_NULL
    finalize(buffer)            # Reactant's finalizer skips a null pointer; this just unregisters it
    return nothing
end

"""
    execute(exec, device, inputs, donated, num_outputs) -> Vector{PJRT.Buffer}

Run a compiled program on one device (`XLAExecuteSharded`) and wait for its results
(`FutureAwait`), returning ready output buffers: what `execute_sharded` followed by
`synced_buffer` does, with both the launch and the wait gc_safe.
"""
function execute(
        exec::PJRT.LoadedExecutable, device::PJRT.Device, inputs::Vector{Ptr{Cvoid}},
        donated::Vector{UInt8}, num_outputs::Int
    )
    length(inputs) == length(donated) || throw(ArgumentError("inputs and donated differ in length"))
    outputs = Vector{Ptr{Cvoid}}(undef, num_outputs)
    futures = Vector{Ptr{Cvoid}}(undef, num_outputs)
    has_future = Ref{UInt8}(0)
    GC.@preserve exec device inputs donated outputs futures has_future begin
        @ccall gc_safe = true Reactant_jll.libReactantExtra.XLAExecuteSharded(
            exec.exec::Ptr{Cvoid}, Cint(length(inputs))::Cint, pointer(inputs)::Ptr{Ptr{Cvoid}},
            device.device::Ptr{Cvoid}, pointer(donated)::Ptr{UInt8}, Cint(num_outputs)::Cint,
            pointer(outputs)::Ptr{Ptr{Cvoid}}, has_future::Ptr{UInt8}, pointer(futures)::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    bufs = PJRT.Buffer[PJRT.Buffer(p) for p in outputs]     # owned (finalizers) before any wait can throw
    if has_future[] != 0
        try
            for f in futures
                @ccall gc_safe = true Reactant_jll.libReactantExtra.FutureAwait(f::Ptr{Cvoid})::Cvoid
            end
        finally
            for f in futures
                @ccall Reactant_jll.libReactantExtra.FreeFuture(f::Ptr{Cvoid})::Cvoid
            end
        end
    end
    return bufs
end

end # module GcSafeXLA
