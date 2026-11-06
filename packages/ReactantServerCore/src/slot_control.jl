# Per-slot control words (CORE-3): the client and a server agree, through one word per pool slot,
# on whether the server may still write a slot's outputs, so a server can never write into a slot
# the client has given up on and reused. No timeouts decide correctness.
#
# Layout: a SHM pool's object holds, after its data slots, a page-aligned control area with one
# 8-byte word per slot, padded to SLOT_CONTROL_STRIDE (128) bytes (buffer_pool.jl). A run of slots
# uses the word of its first slot. A word is `token << 3 | state`, with a 61-bit token that never
# repeats within the pool (each pool starts its tokens at a random point, and each attempt takes a
# new one). States:
#
#   FREE            no attempt (a new object is zero filled)
#   OWNED(t)        the client staged attempt t and may send it; a server may claim it
#   WRITING(t)      a server is copying attempt t's outputs into the slot
#   DONE(t)         that server finished copying (or gave up mid-copy) and will not touch it again
#   CANCELLED(t)    the client gave up on attempt t (or read its outputs); no server may claim it
#
# Transitions, each by one compare-and-swap or store on the word:
#
#   client  arm      FREE | DONE(x) | CANCELLED(x)  ->  OWNED(t)          fresh t, before sending
#   server  claim    OWNED(t)       ->  WRITING(t)   under an OFD lock on the word, before copying
#   server  finish   WRITING(t)     ->  DONE(t)      after copying (in `finally`), then unlock
#   client  close    OWNED(t)       ->  CANCELLED(t) on giving up, and before reading outputs
#                                                    that no server claimed (one that ignores
#                                                    the protocol)
#
# Only a server moves a word out of WRITING, and only the client moves it out of OWNED, DONE or
# CANCELLED, so a word in WRITING(t) belongs to its writer until DONE(t). The client therefore
# never waits: on giving up it closes OWNED(t) and reuses the slot at once (the server's claim then
# fails and it skips the write), and if the word is already WRITING(t) the run is quarantined until
# DONE(t). A writer holds an open-file-description lock on the word's 8 bytes from before its claim
# until after it stores DONE, and the kernel drops that lock when the writer's process dies; so a
# word still WRITING(t) whose lock can be taken has a dead writer, and is reclaimed. A writer alive
# but stuck past SLOT_RETIRE_SECONDS keeps its run retired (never reused) until it finishes. The
# server also checks the word before every chunk of its copy, and at dispatch it drops a request
# whose word no longer reads OWNED(t).
#
# OFD locks (F_OFD_SETLK) are Linux only. Elsewhere the protocol is off: the client sends no control
# parameters (and the client keeps shared memory off on macOS anyway), and a server refuses them.

const SLOT_FREE = UInt64(0)
const SLOT_OWNED = UInt64(1)
const SLOT_WRITING = UInt64(2)
const SLOT_DONE = UInt64(3)
const SLOT_CANCELLED = UInt64(4)
const _SLOT_TOKEN_MAX = (UInt64(1) << 61) - 1

slot_word(state::UInt64, token::UInt64) = (token << 3) | state
slot_state(word::UInt64) = word & UInt64(7)
slot_token(word::UInt64) = word >> 3

"""
    SLOT_CONTROL_REGION_PARAM, SLOT_CONTROL_OFFSET_PARAM, SLOT_CONTROL_TOKEN_PARAM

Request-level parameters (`ModelInferRequest.parameters`) carrying a request's slot control: the
registered region holding the control word (string_param), the word's byte offset within that
region's registered window (int64_param, 8-byte aligned), and the attempt's token (int64_param).
All three or none. A request with shared-memory outputs and slot control must keep those outputs in
the control word's region.
"""
const SLOT_CONTROL_REGION_PARAM = "reactant_shm_control_region"
const SLOT_CONTROL_OFFSET_PARAM = "reactant_shm_control_offset"
const SLOT_CONTROL_TOKEN_PARAM = "reactant_shm_control_token"

# Bytes a server copies between two checks of the control word.
const SLOT_COPY_CHUNK_BYTES = 1 << 20

# How often a parked `acquire_slot!` re-checks quarantined runs.
const _QUARANTINE_POLL_SECONDS = 0.05
# How often a retired run's writer is probed again.
const _RETIRED_PROBE_NS = Int64(1_000_000_000)

"""
    slot_control_supported() -> Bool

Whether the slot control protocol can run here: it needs open-file-description locks, which only
Linux has.
"""
slot_control_supported() = Sys.islinux()

"""
    RequestCancelled(model_name, reason)

The request was given up by its client: at dispatch its slot control word no longer read OWNED, or
before writing its outputs the server could not claim the slot. The transport maps it to CANCELLED.
"""
struct RequestCancelled <: Exception
    model_name::String
    reason::String
end
Base.showerror(io::IO, e::RequestCancelled) =
    print(io, "request for model '", e.model_name, "' was cancelled: ", e.reason)

"""
    SlotWriteInProgress(index)

Raised by [`slot_arm!`](@ref) when the slot's previous attempt is still being written by a server,
so the slot cannot carry a new attempt (a retry) until that writer is done.
"""
struct SlotWriteInProgress <: Exception
    index::Int
end
Base.showerror(io::IO, e::SlotWriteInProgress) =
    print(io, "slot ", e.index, " is still being written by a server for an earlier attempt")

# ---- open-file-description locks ----

# struct flock as Linux declares it (x86_64 and aarch64 alike; the C layout pads l_whence to 8).
struct _Flock
    l_type::Cshort
    l_whence::Cshort
    l_start::Int64
    l_len::Int64
    l_pid::Cint
end
const _F_OFD_SETLK = Cint(37)
const _F_WRLCK = Cshort(1)
const _F_UNLCK = Cshort(2)

function _ofd_setlk(fd::ShmFd, type::Cshort, start::Integer, len::Integer)
    isopen(fd) || return Libc.EBADF
    fl = Ref(_Flock(type, Cshort(0), Int64(start), Int64(len), Cint(0)))
    r = ccall(:fcntl, Cint, (Cint, Cint, Ptr{_Flock}...), fd.fd, _F_OFD_SETLK, fl)
    return r == 0 ? 0 : Libc.errno()
end

"""
    shm_try_lock(fd, start, len) -> Bool

Take a write lock on bytes `[start, start + len)` of `fd`'s object with `F_OFD_SETLK`, without
waiting: `true` when taken, `false` when another open file description holds an overlapping lock.
The lock belongs to this descriptor (not to the process or thread) and is released by
[`shm_unlock`](@ref), by closing the descriptor, or by the kernel when the process dies. Linux only;
throws `SystemError` for any other failure.
"""
function shm_try_lock(fd::ShmFd, start::Integer, len::Integer)
    slot_control_supported() || error("shm_try_lock: open file description locks need Linux")
    e = _ofd_setlk(fd, _F_WRLCK, start, len)
    e == 0 && return true
    (e == Libc.EAGAIN || e == Libc.EACCES) && return false
    throw(SystemError("fcntl(F_OFD_SETLK)", e))
end

function shm_unlock(fd::ShmFd, start::Integer, len::Integer)
    _ofd_setlk(fd, _F_UNLCK, start, len)
    return nothing
end

# ---- client side (the pool that owns the slots) ----

"""
    has_slot_control(pool) -> Bool

Whether `pool`'s slots carry control words the client can arm: a SHM pool, on Linux.
"""
has_slot_control(pool::BufferPool) = pool.ctl_offset > 0 && slot_control_supported()

"""
    slot_control_offset(slot) -> Int

Byte offset, in the pool's object (and so in a window registered at offset 0), of the control word
of `slot`'s run (the word of its first slot).
"""
slot_control_offset(slot::PoolSlot) = slot.pool.ctl_offset + (slot.index - 1) * SLOT_CONTROL_STRIDE

_word_ptr(pool::BufferPool, index::Integer) =
    Ptr{UInt64}(_pool_base_pointer(pool) + pool.ctl_offset + (index - 1) * SLOT_CONTROL_STRIDE)

_next_control_token!(pool::BufferPool) = Threads.atomic_add!(pool.control_tokens, UInt64(1))

# Move a word out of OWNED(token) so no server can claim the attempt any more. Returns :writing
# when a server is copying that attempt right now (the word is WRITING(token)), otherwise :closed:
# the word is now CANCELLED(token), or was DONE(token) (the writer finished), or holds no claimable
# state for `token` at all.
function _close_control_word!(pool::BufferPool, index::Integer, token::UInt64)
    owned = slot_word(SLOT_OWNED, token)
    return GC.@preserve pool begin
        p = _word_ptr(pool, index)
        while true
            w = unsafe_load(p, 1, :acquire)
            if w == owned
                unsafe_replace!(p, owned, slot_word(SLOT_CANCELLED, token), :acquire_release, :acquire).success &&
                    return :closed
                # A server claimed it between the load and the swap: look again.
            elseif w == slot_word(SLOT_WRITING, token)
                return :writing
            else
                return :closed
            end
        end
    end
end

"""
    slot_arm!(slot) -> UInt64

Start an attempt on `slot` (a top-level slot of a pool with [`has_slot_control`](@ref)): move its
control word to OWNED(t) under a fresh token `t`, record `t` on the slot, and return it, for the
request to carry. Arming again for a retry first closes the previous attempt, so its server can no
longer claim it; if that server is already writing, throws [`SlotWriteInProgress`](@ref) and the
slot keeps its previous attempt (release it as usual).
"""
function slot_arm!(slot::PoolSlot)
    pool = slot.pool
    (slot.index > 0 && has_slot_control(pool)) ||
        throw(ArgumentError("slot_arm!: the slot's pool has no slot control"))
    if slot.token != 0
        _close_control_word!(pool, slot.index, slot.token) === :writing &&
            throw(SlotWriteInProgress(slot.index))
    end
    GC.@preserve pool begin
        p = _word_ptr(pool, slot.index)
        w = unsafe_load(p, 1, :acquire)
        # A run is only handed out once nothing can claim or be writing its word (release_slot!
        # closes it, quarantine waits out WRITING), so neither state can be here.
        st = slot_state(w)
        (st == SLOT_OWNED || st == SLOT_WRITING) &&
            error("slot_arm!: slot $(slot.index)'s control word is in use ($(repr(w)))")
        t = _next_control_token!(pool)
        unsafe_replace!(p, w, slot_word(SLOT_OWNED, t), :acquire_release, :acquire).success ||
            error("slot_arm!: slot $(slot.index)'s control word changed while arming")
        slot.token = t
        return t
    end
end

"""
    slot_seal!(slot) -> Bool

After a successful response, decide whether the outputs in `slot` are final and may be read: `true`
when the server finished writing them under the protocol (DONE(t)), or when no server claimed the
attempt (a server that does not run the protocol wrote them before replying), in which case the
word is closed first so nothing can begin writing during the read. `false` means a writer claimed
the attempt and is still copying, so the bytes cannot be trusted. Always `true` for an unarmed slot.
"""
function slot_seal!(slot::PoolSlot)
    t = slot.token
    t == 0 && return true
    pool = slot.pool
    GC.@preserve pool begin
        p = _word_ptr(pool, slot.index)
        w = unsafe_load(p, 1, :acquire)
        w == slot_word(SLOT_DONE, t) && return true
        w == slot_word(SLOT_OWNED, t) || return false
        return unsafe_replace!(p, w, slot_word(SLOT_CANCELLED, t), :acquire_release, :acquire).success
    end
end

function _free_quarantined!(pool::BufferPool, q::QuarantinedRun)
    for i in q.index:(q.index + q.span - 1)
        pool.quarantined[i] = false
        pool.free[i] = true
    end
    return
end

# Free every quarantined run whose writer is done, reclaim those whose writer died, and retire
# those whose writer is alive but has held them past the retire delay. Caller holds alloc_lock.
function _reclaim_quarantined!(pool::BufferPool)
    (isempty(pool.quarantine) || pool.released) && return false
    now = Int64(time_ns())
    freed = false
    GC.@preserve pool filter!(pool.quarantine) do q
        p = _word_ptr(pool, q.index)
        writing = slot_word(SLOT_WRITING, q.token)
        if unsafe_load(p, 1, :acquire) != writing
            _free_quarantined!(pool, q)          # DONE: the writer finished and let go
            freed = true
            return false
        end
        now - q.since_ns < pool.retire_after_ns && return true
        (q.retired && now - q.probed_ns < _RETIRED_PROBE_NS) && return true
        q.probed_ns = now
        off = pool.ctl_offset + (q.index - 1) * SLOT_CONTROL_STRIDE
        locked = try
            shm_try_lock(pool.fd, off, 8)
        catch err
            @warn "slot control: cannot probe a writer's lock; keeping the slot retired" pool = pool.name index = q.index exception = err maxlog = 1
            false
        end
        if locked
            # The writer's lock is gone while the word still says WRITING: it died mid-copy.
            try
                unsafe_replace!(p, writing, slot_word(SLOT_CANCELLED, q.token), :acquire_release, :acquire)
            finally
                shm_unlock(pool.fd, off, 8)
            end
            @warn "slot control: reclaimed a slot whose writer died while copying" pool = pool.name index = q.index
            _free_quarantined!(pool, q)
            freed = true
            return false
        end
        q.retired || @warn "slot control: retiring a slot whose writer is alive but still copying after $(pool.retire_after_ns / 1.0e9)s; it rejoins the pool when the writer finishes" pool = pool.name index = q.index
        q.retired = true
        return true
    end
    freed && notify(pool.freed)
    return freed
end

# ---- server side (a registered region) ----

"""
    SlotControl(mapping, offset, token)

A request's slot control as decoded: the registration its control word lives in (bound at decode,
like an output target), the word's offset within that registration's window, and the attempt's
token.
"""
struct SlotControl
    mapping::ShmRegion
    offset::Int
    token::UInt64
end

# Absolute offset of the word in the object (the lock range), and its address. Caller holds the
# region's lock with the region attached.
_control_object_offset(c::SlotControl) = c.mapping.offset + c.offset
_control_word_ptr(c::SlotControl) = Ptr{UInt64}(c.mapping.ptr + _control_object_offset(c))

"""
    slot_control_cancelled(c::SlotControl) -> Bool

Whether the client has given up on `c`'s attempt: its word no longer reads OWNED(token). Never
waits: when the region's lock is busy (a copy in progress) or the region is gone, it answers
`false` and leaves the decision to the claim before the output write.
"""
function slot_control_cancelled(c::SlotControl)
    r = c.mapping
    trylock(r.lock) || return false
    try
        r.attached || return false
        return GC.@preserve r unsafe_load(_control_word_ptr(c), 1, :acquire) != slot_word(SLOT_OWNED, c.token)
    finally
        unlock(r.lock)
    end
end

"""
    shm_write_controlled!(c::SlotControl, items, model_name; chunk_bytes, before_chunk = nothing)

Write a request's shared-memory outputs under the slot control protocol. `items` are
`(data::Array, offset, nbytes)` triples in `c`'s region (bounds are checked first). Holding the
region's lock: take an OFD write lock on the control word, claim OWNED(t) -> WRITING(t), copy each
output in `chunk_bytes` pieces, checking before each piece that the word still reads WRITING(t),
then store DONE(t) and unlock (both in `finally`). A lock or a claim that fails means the client gave
up: nothing is written and [`RequestCancelled`](@ref) is thrown. `before_chunk` is a test hook.
"""
function shm_write_controlled!(
        c::SlotControl, items, model_name::AbstractString;
        chunk_bytes::Integer = SLOT_COPY_CHUNK_BYTES, before_chunk = nothing
    )
    r = c.mapping
    @lock r.lock begin
        r.attached || throw(UnregisteredRegionError(r.name))
        dsts = Ptr{UInt8}[first(_region_base(r, off, nb)) for (_, off, nb) in items]
        lock_at = _control_object_offset(c)
        shm_try_lock(r.fd, lock_at, 8) ||
            throw(RequestCancelled(String(model_name), "its slot control word is locked by another party"))
        try
            GC.@preserve r begin
                p = _control_word_ptr(c)
                writing = slot_word(SLOT_WRITING, c.token)
                unsafe_replace!(p, slot_word(SLOT_OWNED, c.token), writing, :acquire_release, :acquire).success ||
                    throw(RequestCancelled(String(model_name), "its client gave up on the shared-memory slot"))
                try
                    for ((data, _, nb), dst) in zip(items, dsts)
                        GC.@preserve data begin
                            src = convert(Ptr{UInt8}, pointer(data))
                            done = 0
                            while done < nb
                                before_chunk === nothing || before_chunk()
                                unsafe_load(p, 1, :acquire) == writing || throw(
                                    RequestCancelled(String(model_name), "its slot control word changed during the copy")
                                )
                                n = min(Int(chunk_bytes), nb - done)
                                unsafe_copyto!(dst + done, src + done, n)
                                done += n
                            end
                        end
                    end
                finally
                    unsafe_replace!(p, writing, slot_word(SLOT_DONE, c.token), :release, :monotonic)
                end
            end
        finally
            shm_unlock(r.fd, lock_at, 8)
        end
    end
    return nothing
end

# Copy `nbytes` of `data` into a region at `offset`, without the protocol (a client that sends no
# slot control).
function _shm_copy_in!(r::ShmRegion, offset::Integer, data::Array, nbytes::Integer)
    @lock r.lock begin
        r.attached || throw(UnregisteredRegionError(r.name))
        base, bs = _region_base(r, offset, nbytes)
        GC.@preserve r data unsafe_copyto!(base, convert(Ptr{UInt8}, pointer(data)), bs)
    end
    return nothing
end
