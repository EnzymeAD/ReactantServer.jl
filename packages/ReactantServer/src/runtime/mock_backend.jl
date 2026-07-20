# A Reactant-free backend for testing ordering, lifecycle, and the request path.
#
# A mock executable is a plain Julia function over input/weight arrays. This lets tests
# exercise run_model, the scheduler, and buffer release without a GPU or Reactant.

struct MockBackend <: AbstractBackend end

struct MockClient end
struct MockDevice
    ordinal::Int
end

mutable struct MockBuffer
    data::Array
    freed::Bool
end
MockBuffer(a::Array) = MockBuffer(a, false)

# fn maps a vector of argument arrays (inputs then weights) to a vector of output arrays.
# `freed` mirrors MockBuffer: set by free_executable! so tests can assert eager release on evict.
mutable struct MockExecutable
    fn::Function
    num_outputs::Int
    freed::Bool
end
MockExecutable(fn::Function, num_outputs::Int) = MockExecutable(fn, num_outputs, false)

make_client(::MockBackend, platform::String; kwargs...) = MockClient()
select_device(::MockBackend, ::MockClient, ordinal::Int) = MockDevice(ordinal)
device_ordinal(::MockBackend, d::MockDevice) = d.ordinal

to_device(::MockBackend, ::MockClient, a::Array, ::MockDevice) = MockBuffer(copy(a))
buffer_eltype(::MockBackend, b::MockBuffer) = eltype(b.data)
# Emulate the real backend's reversed (XLA row-major) shape: a Julia (2,3) array reports as
# (3,2), and run_model reverses it back. For 1-D buffers reverse is a no-op. Without this the
# mock would disagree with ReactantBackend on multi-dimensional outputs.
buffer_size(::MockBackend, b::MockBuffer) = reverse(size(b.data))
to_host!(::MockBackend, b::MockBuffer, dest::Array) = (copyto!(dest, b.data); dest)
free_buffer!(::MockBackend, b::MockBuffer) = (b.freed = true; nothing)
free_executable!(::MockBackend, e::MockExecutable) = (e.freed = true; nothing)

function execute_single_device(::MockBackend, exec::MockExecutable, ::MockDevice,
                               buffers::AbstractVector, donated::AbstractVector{Bool}, num_outputs::Int)
    args = [b.data for b in buffers]
    outs = exec.fn(args)
    length(outs) == num_outputs ||
        error("mock executable returned $(length(outs)) outputs, expected $num_outputs")
    return MockBuffer[MockBuffer(o) for o in outs]
end
