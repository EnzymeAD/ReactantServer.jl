# Progress of a read of model weights, for a watcher that tells a slow read from a stalled one.
#
# A worker reads a model's weights on a background task and abandons a read that goes too long
# without progress (`scheduler.weight_read_stall_seconds`). Progress is bytes arriving: every chunk a
# weights file read returns is counted with `note_read_progress!` against the sink the reading task
# runs under (`with_read_progress`), so nothing between the scheduler and the read passes it along. A
# wait on a lock counts as none, except a wait for a peer that fills the same shared region, which
# counts the peer's progress (see `_lock_shared_following_fill` in weight_store.jl).

"""
    ReadProgress()

How far a read of model weights has got: the bytes read so far and when the last of them arrived
(`time_ns()`). Advanced by [`note_read_progress!`](@ref) for the read a task runs under
[`with_read_progress`](@ref).
"""
mutable struct ReadProgress
    @atomic bytes::Int
    @atomic last_ns::Int64
end
ReadProgress() = ReadProgress(0, Int64(time_ns()))

# The sink of the read running on this task: a ReadProgress, a function of the byte count (a shared
# region's fill, which publishes its progress and passes it on), or nothing.
const _READ_PROGRESS = Base.ScopedValues.ScopedValue{Any}(nothing)

"""
    with_read_progress(f, sink)

Run `f()`, counting the bytes it reads of model weights on this task in `sink`: a
[`ReadProgress`](@ref), or a function called with each count.
"""
with_read_progress(f, sink) = Base.ScopedValues.with(f, _READ_PROGRESS => sink)

"""
    note_read_progress!(nbytes)

Count `nbytes` of model weights read (or `0`, for work done that reads nothing) for the read this
task runs under [`with_read_progress`](@ref), if any.
"""
note_read_progress!(nbytes::Integer) = _note_progress!(_READ_PROGRESS[], Int(nbytes))

_note_progress!(::Nothing, ::Int) = nothing
function _note_progress!(p::ReadProgress, n::Int)
    @atomic p.bytes += n
    @atomic p.last_ns = Int64(time_ns())
    return nothing
end
_note_progress!(f, n::Int) = (f(n); nothing)
