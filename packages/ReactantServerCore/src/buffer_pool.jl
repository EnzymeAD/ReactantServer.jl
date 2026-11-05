# Concurrency-safe staging buffer with a fixed-slot allocator.
#
# A BufferPool is a single backing region (POSIX shared memory or a plain Memory{UInt8})
# divided into `n_slots` equal, non-overlapping slots of `slot_bytes` each. Slot index i maps
# to offset (i-1)*slot_bytes, so adjacent indices are physically contiguous and a caller can
# acquire a run of `span` slots as one contiguous byte range. Free slots are tracked in a
# bitmask guarded by a lock; `acquire_slot!` blocks until a contiguous run of the requested
# span is free and hands out a disjoint byte range; `release_slot!` returns the whole run.
# Concurrent producers can never be handed overlapping ranges, which is the property the old
# fixed-index/lockstep scheme only guaranteed within a single top-level inference call. The
# allocator also supplies natural back-pressure: a caller blocks when all slots are in flight
# instead of overrunning the buffer.
#
# Waiters are served strictly FIFO: only the head of `waitq` may allocate, and everyone else
# re-parks. Head-of-line blocking is deliberate — it guarantees a large-span waiter cannot be
# starved by a stream of span-1 acquisitions (a bypass scheme can livelock the large waiter
# forever through fragmentation), and it preserves the FIFO order span-1 callers had under the
# previous Channel-based free list. `try_acquire_slot!` never waits and never passes a waiter.
#
# A SHM pool also carries a control area after its data slots: one control word per slot, through
# which the client and a server agree on who may write the slot (see slot_control.jl, CORE-3). A
# run released while a server may still be writing it is quarantined, not freed, until the word
# says the writer is done (or the writer is found dead); one whose writer is alive but stuck is
# retired the same way and rejoins when that writer finishes.

const PoolBacking = Union{SharedMemory, Memory{UInt8}}

# Bytes per slot control word: one 8-byte word padded to 128 bytes, so no two slots' words share a
# cache line or an adjacent-line prefetch pair (Apple M-series lines are 128 bytes; Intel prefetches
# 64-byte lines in pairs). 32 slots fill one 4 KiB page.
const SLOT_CONTROL_STRIDE = 128

"""
    SLOT_RETIRE_SECONDS

How long (seconds) a released slot may stay under a live writer before the client checks whether
that writer still exists: a writer that died is reclaimed at once, one that is alive but stuck has
its slot retired (never reused while it may still write) until it finishes.
"""
const SLOT_RETIRE_SECONDS = 30.0

# A run released while a server held its control word in WRITING(token).
mutable struct QuarantinedRun
    index::Int
    span::Int
    token::UInt64
    since_ns::Int64
    retired::Bool                # a live writer was found still holding it after the retire delay
    probed_ns::Int64             # last liveness probe
end

mutable struct BufferPool
    backing::PoolBacking
    n_bytes::Int                 # data bytes: n_slots * slot_bytes
    name::String          # SHM region name (the basename clients register); "" for inline pools
    slot_bytes::Int
    n_slots::Int
    free::BitVector              # true = the 1-based slot index is free
    alloc_lock::ReentrantLock    # guards free / waitq / token_counter / quarantine
    freed::Threads.Condition     # tied to alloc_lock; signaled on release and waiter exit
    waitq::Vector{UInt64}        # FIFO of waiter tokens; only the head may allocate
    token_counter::UInt64
    closed::Bool                 # close() was called: no new slots (guarded by alloc_lock)
    released::Bool               # the SHM mapping has been released (guarded by alloc_lock)
    fd::Union{ShmFd, Nothing}    # the SHM object's descriptor, kept open; nothing for inline pools
    region_bytes::Int            # the object's whole length (what a client registers)
    ctl_offset::Int              # offset of the slot control area; 0 when the pool has none
    control_tokens::Threads.Atomic{UInt64}   # next slot control token; never repeats in this pool
    quarantine::Vector{QuarantinedRun}       # runs a writer may still write (guarded by alloc_lock)
    quarantined::BitVector       # slot indices held by a quarantined run
    retire_after_ns::Int64
end

# Slot boundaries are aligned (CORE-15): `unsafe_wrap` refuses a pointer misaligned for its element
# type, so a slot at an odd offset (n_bytes ÷ n_slots odd, e.g. 10^6 ÷ 3) failed for every typed
# view in it. A slot of at least a page starts on a page boundary, a smaller one on a 64-byte line,
# and a tiny one on 8 bytes (the largest element alignment). A lone slot sits at the base and keeps
# its exact size.
const _MIN_SLOT_ALIGN = 64
_page_size() = Int(ccall(:getpagesize, Cint, ()))
function _aligned_slot_bytes(per_slot::Int, n_slots::Int)
    n_slots == 1 && return per_slot
    a = per_slot >= _page_size() ? _page_size() : per_slot >= _MIN_SLOT_ALIGN ? _MIN_SLOT_ALIGN : 8
    return per_slot - per_slot % a
end

# How many freshly minted names a SHM pool tries before giving up when each one already exists.
const _SHM_CREATE_ATTEMPTS = 8

# Create a SHM pool's object of `len` bytes, exclusively (O_CREAT | O_EXCL, so a name collision
# never shares memory). An explicit `key` is tried once; otherwise every attempt mints a fresh key
# from `name` (see `_fresh_shm_key`), and a name that already exists, such as one a crashed process
# with the same pid left in /dev/shm, is retried under a new token. The object is mapped through
# the descriptor that created it, which the pool keeps; IPC's `SharedMemory` owns the mapping.
# `mint` is the key generator (a test hook).
function _create_pool_object(
        name::AbstractString, key::Union{AbstractString, Nothing}, len::Int;
        mint = _fresh_shm_key
    )
    attempts = key === nothing ? _SHM_CREATE_ATTEMPTS : 1
    for _ in 1:attempts
        k = key === nothing ? String(mint(String(name))) : String(key)
        made = shm_create_exclusive(k, len)
        made === nothing && continue
        fd, ptr, _ = made
        # volatile: the name is unlinked when the mapping is finalized (also at process exit), as
        # IPC's own constructor does for an object it creates.
        shm = SharedMemory{String}(convert(Ptr{Cvoid}, ptr), len, true, k)
        return shm, (startswith(k, "/") ? k[2:end] : k), fd
    end
    key === nothing || throw(SystemError("shm_open (create) of shared memory object '$key'", Libc.EEXIST))
    error("BufferPool: $attempts freshly minted shared memory names for '$name' all exist already")
end

"""
    BufferPool(n_bytes; n_slots=8, use_shm=true, name="reactant_server_pool", key=nothing)

Allocate a staging pool of `n_bytes` divided into `n_slots` equal slots. `slot_bytes` is fixed
at construction, not recomputed per request, so the allocator can hand disjoint slots to concurrent
callers: `n_bytes ÷ n_slots` rounded down to a multiple of the page size when that is at least a
page, otherwise to a multiple of 64 bytes (8 below that; a single slot keeps its size), so every
slot boundary keeps any element type aligned.
A SHM-backed pool can be registered with a server; an inline pool (`use_shm=false`) is the
fallback transport. A SHM pool's object name is minted per pool from `name` with a fresh random
token (two pools never share a name, even one after the other in one process), unless an explicit
`key` is given.

A SHM pool's object also holds the slot control area after the data slots (page aligned, one
128-byte word per slot), so its length, [`pool_region_bytes`](@ref), is what a client registers.
`retire_after` is [`SLOT_RETIRE_SECONDS`](@ref) unless a test shortens it.
"""
function BufferPool(
        n_bytes::Integer; n_slots::Integer = 8, use_shm::Bool = true,
        name::AbstractString = "reactant_server_pool",
        key::Union{AbstractString, Nothing} = nothing,
        retire_after::Real = SLOT_RETIRE_SECONDS
    )
    n_bytes > 0 || throw(ArgumentError("BufferPool: n_bytes must be positive (got $n_bytes)"))
    n_slots > 0 || throw(ArgumentError("BufferPool: n_slots must be positive (got $n_slots)"))
    slot_bytes = _aligned_slot_bytes(Int(n_bytes) ÷ Int(n_slots), Int(n_slots))
    slot_bytes > 0 ||
        throw(ArgumentError("BufferPool: n_bytes ($n_bytes) too small for $n_slots slots"))
    usable = slot_bytes * Int(n_slots)
    # The control area starts on the page after the data and spans whole pages.
    page = _page_size()
    ctl_offset = use_shm ? cld(usable, page) * page : 0
    region_bytes = use_shm ? ctl_offset + cld(Int(n_slots) * SLOT_CONTROL_STRIDE, page) * page : usable

    backing, region_name, fd = if use_shm
        # An explicit `key` (e.g. one the supervisor minted and injected so peer workers can attach
        # the same region) is used verbatim; otherwise a fresh key is minted from `name`. A new
        # object is zero filled: every control word starts FREE.
        _create_pool_object(name, key, region_bytes)
    else
        Memory{UInt8}(undef, usable), "", nothing
    end
    # Slot offsets are aligned relative to the base, so the base must be aligned too: mmap returns a
    # page, and Julia allocates Memory at least 16-byte aligned, more than any staged element needs.
    UInt(pointer(backing)) % 16 == 0 || error("BufferPool: the backing memory is not 16-byte aligned")

    alloc_lock = ReentrantLock()
    return BufferPool(
        backing, usable, region_name, slot_bytes, Int(n_slots),
        trues(Int(n_slots)), alloc_lock, Threads.Condition(alloc_lock),
        UInt64[], UInt64(0), false, false, fd,
        region_bytes, ctl_offset, Threads.Atomic{UInt64}(_control_token_seed()),
        QuarantinedRun[], falses(Int(n_slots)), round(Int64, Float64(retire_after) * 1.0e9)
    )
end

# A random start for a pool's control tokens, in [1, 2^60), so tokens never repeat within the pool
# and a token from one pool generation is not expected in another.
_control_token_seed() = (reinterpret(UInt64, Libc.getrandom!(Vector{UInt8}(undef, 8)))[1] >> 4) | UInt64(1)

Base.sizeof(pool::BufferPool) = pool.n_bytes
is_shm_backed(pool::BufferPool) = pool.backing isa SharedMemory
pool_region_name(pool::BufferPool) = pool.name
pool_slot_bytes(pool::BufferPool) = pool.slot_bytes

"""
    pool_region_bytes(pool) -> Int

The whole length of a SHM pool's object: the data slots and then the slot control area. This, not
`sizeof(pool)` (the data bytes), is the window a client registers, so a server can reach the
control words. Equal to `sizeof(pool)` for an inline pool.
"""
pool_region_bytes(pool::BufferPool) = pool.region_bytes

function _pool_base_pointer(pool::BufferPool)
    return pool.backing isa SharedMemory ? convert(Ptr{UInt8}, pointer(pool.backing)) : pointer(pool.backing)
end

# Base address of the pool's backing region. Used by the meta fan-out to detect, by pointer range,
# whether a sub-call input's bytes already live in this pool (so it can be sent by SHM reference
# instead of inlined) — robust to the caller reshaping/sub-viewing the scratch buffer.
pool_base_pointer(pool::BufferPool) = _pool_base_pointer(pool)

"""
    close(pool::BufferPool)

Retire a pool. Later `acquire_slot!` calls, and waiters already parked in one, throw a
[`PoolClosedError`](@ref); slots already handed out stay valid until released. For a SHM-backed
pool the name is unlinked at once, and the mapping is released as soon as no slot is outstanding:
here if the pool is idle, otherwise in the `release_slot!` that returns the last slot. Unlinking alone does
not free the pages in /dev/shm while the region is still mapped, and before this the mapping was
only released when the GC got round to finalizing the `SharedMemory`. A live `pool_memory` alias
may be read at any time, so while one exists the mapping is left to the GC as before. Idempotent.
"""
function Base.close(pool::BufferPool)
    @lock pool.alloc_lock begin
        pool.closed && return nothing
        pool.closed = true
        if pool.backing isa SharedMemory
            try
                rm(pool.backing)            # shm_unlink by name; the mapping is untouched
                # Unlinked once, here: the finalizer must not unlink the name a second time, when it
                # could already belong to another object.
                pool.backing.volatile = false
            catch ex
                @warn "Failed to unlink SHM region $(pool.name)" exception = ex
            end
        end
        _release_if_idle!(pool)
        notify(pool.freed)                  # parked waiters re-check and throw
    end
    return nothing
end

Base.isopen(pool::BufferPool) = !(@lock pool.alloc_lock pool.closed)

# Release a closed SHM pool's mapping once nothing can still touch it: no slot outstanding and no
# pool_memory alias alive. A quarantined run counts as idle: nothing in this process reads it, and a
# server's late write lands in its own mapping, not ours. Caller holds alloc_lock. `finalize` runs
# InterProcessCommunication's _destroy (munmap) now instead of at some later GC, and unregisters it
# so it never runs twice.
function _release_if_idle!(pool::BufferPool)
    (pool.closed && !pool.released && pool.backing isa SharedMemory) || return
    all(i -> pool.free[i] || pool.quarantined[i], 1:pool.n_slots) || return
    _has_pinned_alias(pool) && return
    pool.released = true
    empty!(pool.quarantine)
    finalize(pool.backing)
    pool.fd === nothing || close(pool.fd)
    return
end

# ---- PoolSlot: a byte range within a pool, with a cursor for carving subslots ----

mutable struct PoolSlot
    pool::BufferPool
    index::Int       # 1-based first physical slot for release_slot!; 0 for a derived subslot
    offset::Int
    capacity::Int    # span * slot_bytes for top-level slots
    span::Int        # physical slots covered; 0 for a derived subslot
    cursor::Int
    token::UInt64    # the slot control token of the current attempt (`slot_arm!`); 0 when unarmed
end

PoolSlot(pool::BufferPool, index::Integer, offset::Integer, capacity::Integer, span::Integer) =
    PoolSlot(pool, Int(index), Int(offset), Int(capacity), Int(span), 0, UInt64(0))
PoolSlot(pool::BufferPool, index::Integer, offset::Integer, capacity::Integer, span::Integer, cursor::Integer) =
    PoolSlot(pool, Int(index), Int(offset), Int(capacity), Int(span), Int(cursor), UInt64(0))

Base.sizeof(slot::PoolSlot) = slot.capacity

# First index of a free run of `span` slots, or 0 if none exists. Caller holds alloc_lock.
function _find_free_run(free::BitVector, span::Int)
    n = length(free)
    i = 1
    while i <= n - span + 1
        if free[i]
            j = i + 1
            while j < i + span && free[j]
                j += 1
            end
            j == i + span && return i
            i = j + 1   # skip past the occupied slot that broke the run
        else
            i += 1
        end
    end
    return 0
end

"""
    PoolAcquireTimeout(span, waited_ns)

Raised by [`acquire_slot!`](@ref) when a `deadline_ns` was supplied and passed before `span`
contiguous slots became free. The waiter is dequeued before this is thrown, so it never stalls the
line. Callers that carry a request deadline (e.g. a meta model's fan-out) translate this into their
own deadline-exceeded error.
"""
struct PoolAcquireTimeout <: Exception
    span::Int
    waited_ns::Int64
end
Base.showerror(io::IO, e::PoolAcquireTimeout) =
    print(
    io, "acquire_slot!: timed out after ", round(e.waited_ns / 1.0e9, digits = 3),
    "s waiting for ", e.span, " contiguous slot(s)"
)

"""
    PoolClosedError()

Raised by [`acquire_slot!`](@ref) on a pool that has been `close`d, including to a waiter that was
already parked when the pool closed.
"""
struct PoolClosedError <: Exception end
Base.showerror(io::IO, ::PoolClosedError) = print(io, "acquire_slot!: the buffer pool is closed")

"""
    acquire_slot!(pool, span=1; deadline_ns=0) -> PoolSlot

Block until `span` physically contiguous slots are free, then return them as one slot whose
`[offset, offset + span*slot_bytes)` range no other in-flight slot overlaps. Throws an
`ArgumentError` immediately if `span` exceeds the pool's total slot count, since such a
request could never be satisfied and would otherwise deadlock. Waiters are served in FIFO
order. Pair with `release_slot!`, which frees the whole run.

`deadline_ns` is an absolute `time_ns()` deadline (0 = wait indefinitely, the default and prior
behavior). When set, a waiter that has not acquired by the deadline throws [`PoolAcquireTimeout`]
instead of parking past it. A one-shot timer wakes the waiter at the deadline even if no slot is
released in the meantime, so a starved waiter fails fast rather than burning a request's whole
budget in the park.
"""
function acquire_slot!(pool::BufferPool, span::Integer = 1; deadline_ns::Integer = 0)
    s = Int(span)
    s >= 1 || throw(ArgumentError("acquire_slot!: span must be >= 1 (got $s)"))
    s <= pool.n_slots || throw(
        ArgumentError(
            "acquire_slot!: span $s exceeds the pool's $(pool.n_slots) slots of " *
                "$(pool.slot_bytes) bytes each; this request can never be satisfied. " *
                "Increase the pool's total bytes or decrease n_slots so each slot is larger."
        )
    )

    dl = Int64(deadline_ns)
    t0 = Int64(time_ns())
    lock(pool.alloc_lock)
    if pool.closed
        unlock(pool.alloc_lock)
        throw(PoolClosedError())
    end
    token = UInt64(0)   # sentinel keeps the finally safe if we are interrupted pre-enqueue
    timer = nothing     # one-shot wake at the deadline; armed on first park, closed in finally
    return try
        token = (pool.token_counter += 1)
        push!(pool.waitq, token)
        while true
            pool.closed && throw(PoolClosedError())
            if dl != 0 && Int64(time_ns()) >= dl
                throw(PoolAcquireTimeout(s, Int64(time_ns()) - t0))
            end
            if first(pool.waitq) == token
                _reclaim_quarantined!(pool)
                start = _find_free_run(pool.free, s)
                if start != 0
                    for i in start:(start + s - 1)
                        pool.free[i] = false
                    end
                    return PoolSlot(
                        pool, start, (start - 1) * pool.slot_bytes,
                        s * pool.slot_bytes, s
                    )
                end
            end
            # Arm a single timer for the absolute deadline so the park is bounded even with no
            # release. It notifies `freed` (under the lock it guards), waking every waiter to
            # re-check; the expired one then throws on the next loop turn. notify wakes all, so one
            # timer per waiter is enough for each to observe its own deadline.
            if dl != 0 && timer === nothing
                rem = (dl - Int64(time_ns())) / 1.0e9
                rem <= 0 && throw(PoolAcquireTimeout(s, Int64(time_ns()) - t0))
                timer = Timer(
                    _ -> (
                        try
                            @lock pool.alloc_lock notify(pool.freed)
                        catch
                        end
                    ), rem
                )
            end
            # A quarantined run frees when its writer finishes, which signals nothing here, so poll.
            poll = isempty(pool.quarantine) ? nothing : Timer(
                    _ -> (
                        try
                            @lock pool.alloc_lock notify(pool.freed)
                    catch
                    end
                    ), _QUARANTINE_POLL_SECONDS
                )
            try
                wait(pool.freed)   # releases alloc_lock while parked, reacquires on wake
            finally
                poll === nothing || close(poll)
            end
        end
    finally
        timer === nothing || close(timer)
        # Runs on success, on interruption while parked (wait reacquires the lock before
        # rethrowing), and on any other error: dequeue ourselves and wake the next head so an
        # abandoned waiter can never stall the line.
        if token != 0
            i = findfirst(==(token), pool.waitq)
            i === nothing || deleteat!(pool.waitq, i)
        end
        notify(pool.freed)
        unlock(pool.alloc_lock)
    end
end

"""
    try_acquire_slot!(pool, span=1) -> Union{PoolSlot, Nothing}

Like [`acquire_slot!`](@ref) without waiting: the slot when `span` contiguous slots are free right
now, otherwise `nothing`. It never passes a caller parked in `acquire_slot!`, so it returns
`nothing` while anyone waits. Throws [`PoolClosedError`](@ref) on a closed pool and `ArgumentError`
for a span the pool can never satisfy.
"""
function try_acquire_slot!(pool::BufferPool, span::Integer = 1)
    s = Int(span)
    (1 <= s <= pool.n_slots) ||
        throw(ArgumentError("try_acquire_slot!: span $s is outside 1:$(pool.n_slots)"))
    @lock pool.alloc_lock begin
        pool.closed && throw(PoolClosedError())
        _reclaim_quarantined!(pool)
        isempty(pool.waitq) || return nothing
        start = _find_free_run(pool.free, s)
        start == 0 && return nothing
        for i in start:(start + s - 1)
            pool.free[i] = false
        end
        return PoolSlot(pool, start, (start - 1) * pool.slot_bytes, s * pool.slot_bytes, s)
    end
end

"""
    release_slot!(slot)

Return a slot acquired with `acquire_slot!` to the pool's free set, freeing every physical
slot in its span. Releasing a derived subslot (`index == 0`) is a no-op. Throws if any slot
in the span is already free (double release).

A slot armed for the slot control protocol ([`slot_arm!`](@ref)) first has its control word moved
out of OWNED, so no server can begin writing it any more. If a server is writing it at that moment
(WRITING), the run is quarantined instead of freed: it rejoins the free set once the writer is done,
or once the writer is found dead after [`SLOT_RETIRE_SECONDS`](@ref); a live writer stuck past that
keeps it retired until it finishes. Nothing ever waits here.
"""
function release_slot!(slot::PoolSlot)
    slot.index == 0 && return nothing
    pool = slot.pool
    @lock pool.alloc_lock begin
        for i in slot.index:(slot.index + slot.span - 1)
            (pool.free[i] || pool.quarantined[i]) && throw(
                ArgumentError(
                    "release_slot!: slot index $i is already free (double release?)"
                )
            )
        end
        token = slot.token
        slot.token = UInt64(0)
        if token != 0 && !pool.released && _close_control_word!(pool, slot.index, token) === :writing
            pool.quarantined[slot.index:(slot.index + slot.span - 1)] .= true
            push!(pool.quarantine, QuarantinedRun(slot.index, slot.span, token, Int64(time_ns()), false, 0))
            @debug "slot quarantined: a server is still writing it" pool = pool.name index = slot.index
            _release_if_idle!(pool)
            return nothing
        end
        for i in slot.index:(slot.index + slot.span - 1)
            pool.free[i] = true
        end
        _release_if_idle!(pool)
        notify(pool.freed)
    end
    return nothing
end

"""
    subslot(parent, n_bytes; align = 1) -> PoolSlot

Carve a child slot of `n_bytes` from the next cursor position of `parent`, first moving the cursor
forward to a multiple of `align` bytes from the pool's base (a power of two, at most 64). The
parent's cursor advances; the child starts with a fresh cursor at zero and index 0 (not separately
releasable, its lifetime is bounded by the parent's). A typed view of the child (`pool_view(child,
T, ...)`) needs `align` to be at least `T`'s alignment; [`scratch`](@ref) carves that way.
"""
function subslot(parent::PoolSlot, n_bytes::Integer; align::Integer = 1)
    n = Int(n_bytes)
    n >= 0 || throw(ArgumentError("subslot byte count must be non-negative (got $n)"))
    a = Int(align)
    (ispow2(a) && a <= _MIN_SLOT_ALIGN) ||
        throw(ArgumentError("subslot alignment must be a power of two up to $_MIN_SLOT_ALIGN (got $a)"))
    # Slot offsets are multiples of _MIN_SLOT_ALIGN from the base, so aligning the offset aligns
    # the address.
    start = parent.cursor + mod(-(parent.offset + parent.cursor), a)
    # Subtraction form: `start + n` wraps for a huge `n` and would pass the check (CORE-15).
    (start <= parent.capacity && n <= parent.capacity - start) ||
        throw(
        ArgumentError(
            "subslot of $n bytes exceeds parent slot capacity " *
                "($(max(0, parent.capacity - start)) bytes remaining)"
        )
    )
    child = PoolSlot(parent.pool, 0, parent.offset + start, n, 0, 0)
    parent.cursor = start + n
    return child
end

# Reset a slot's cursor so it can be re-partitioned when recycled.
reset_slot!(slot::PoolSlot) = (slot.cursor = 0; slot)

# A typed pointer to `slot`'s first byte, refused with a clear error rather than by `unsafe_wrap`
# when the offset does not suit `T`.
function _typed_slot_pointer(slot::PoolSlot, ::Type{T}, who) where {T}
    ptr = Ptr{T}(_pool_base_pointer(slot.pool) + slot.offset)
    a = Base.datatype_alignment(T)
    UInt(ptr) % a == 0 || throw(
        ArgumentError(
            "$who: offset $(slot.offset) is not aligned for $T; carve typed buffers with `scratch`, " *
                "or with `subslot(slot, n_bytes; align = $a)`"
        )
    )
    return ptr
end

# A Julia array view of the slot's bytes typed as T. The pool owns the memory; the view is
# ephemeral.
function pool_view(slot::PoolSlot, ::Type{T}, dims::Integer...) where {T}
    n_bytes = Base.checked_mul(Int(sizeof(T)), reduce(Base.checked_mul, Int.(dims); init = 1))
    n_bytes <= slot.capacity ||
        throw(ArgumentError("pool_view ($n_bytes bytes) exceeds slot capacity ($(slot.capacity) bytes)"))
    return unsafe_wrap(Array, _typed_slot_pointer(slot, T, "pool_view"), dims)
end

# Pinned pool references keyed by the aliased Memory's objectid. Same rationale as
# _SHM_KEEPALIVE: a closure-only finalizer is not reliable, and keying by Memory itself would
# hold it strongly and prevent its finalizer from ever running.
const _POOL_KEEPALIVE = Dict{UInt, BufferPool}()
const _POOL_KEEPALIVE_LOCK = ReentrantLock()

# A Memory{T} aliasing the slot's bytes (zero-copy, own=false). The pool is pinned in
# _POOL_KEEPALIVE for the Memory's lifetime so its backing cannot be GC'd out from under it.
function pool_memory(slot::PoolSlot, ::Type{T}, n_elem::Integer) where {T}
    n = Int(n_elem)
    n >= 0 || throw(ArgumentError("pool_memory: n_elem must be non-negative (got $n)"))
    n_bytes = Base.checked_mul(Int(sizeof(T)), n)
    n_bytes <= slot.capacity ||
        throw(ArgumentError("pool_memory ($n_bytes bytes) exceeds slot capacity ($(slot.capacity) bytes)"))
    pool = slot.pool
    ptr = _typed_slot_pointer(slot, T, "pool_memory")
    mem = unsafe_wrap(Memory{T}, ptr, n; own = false)
    return _pin_alias!(_POOL_KEEPALIVE, _POOL_KEEPALIVE_LOCK, mem, pool)
end

# Whether any pool_memory alias of `pool` is still pinned. Linear in the live aliases, and only
# consulted for a closed pool. An entry whose finalizer was deferred still counts, which errs
# toward keeping the mapping.
_has_pinned_alias(pool::BufferPool) =
    @lock _POOL_KEEPALIVE_LOCK any(p -> p === pool, values(_POOL_KEEPALIVE))

# FixedSizeArray view over a pool slot, backed by the slot's Memory{T} alias.
pool_fsa(slot::PoolSlot, ::Type{T}, dims::NTuple{N, <:Integer}) where {T, N} =
    fsa_from_memory(pool_memory(slot, T, prod(dims)), dims)

pool_fsa(slot::PoolSlot, ::Type{T}, dims::Vararg{Integer, N}) where {T, N} = pool_fsa(slot, T, dims)

"""
    scratch(slot, dims, T) -> Array{T}
    scratch(slot, [dims1 => T1, dims2 => T2, ...]) -> Vector{Array}

Carve one typed buffer per `dims => T` spec from `slot` in a single call, advancing the slot's
cursor so the buffers occupy disjoint, contiguous byte ranges. Each buffer is an `Array{T}`
aliasing the pool's backing (via `pool_view`; zero-copy, and uniform across SHM- and
`Memory`-backed pools since it is just `pool_base + offset`); write into the returned arrays
directly. `dims` is a shape tuple (or a bare integer for a vector).

This is the buffer-request interface shared by the meta-model `call.scratch` and the client driver:
ask for ALL buffers up front in one call. The carved buffers' lifetime is bounded by `slot` (and by
the pool, which the caller keeps alive); they become invalid once it is released.

Every buffer starts at an offset aligned for its element type. Within one call the buffers are laid
out by element alignment, largest first (the returned order is the requested one): each buffer's
size is a multiple of its element size, so that order leaves no padding between them and the call
uses exactly the sum of the buffer sizes, which is what a caller sizes the slot for.
"""
scratch(slot::PoolSlot, dims, ::Type{T}) where {T} =
    first(scratch(slot, Pair[(dims isa Tuple ? dims : (dims,)) => T]))

function scratch(slot::PoolSlot, specs::AbstractVector)
    out = Vector{Any}(undef, length(specs))
    for i in _scratch_order([last(s) for s in specs])
        out[i] = _carve_scratch(slot, first(specs[i]), last(specs[i]))
    end
    return out
end

# The order to carve buffers of element types `types` in: by alignment, largest first, stable. With
# power-of-two alignments that each divide the element size, every buffer then starts where the
# previous one ended (no padding), given an aligned start (CORE-15).
_scratch_order(types) =
    sortperm(collect(types); by = T -> -Base.datatype_alignment(T), alg = Base.Sort.MergeSort)

function _carve_scratch(slot::PoolSlot, dims, ::Type{T}) where {T}
    d = dims isa Tuple ? dims : (dims,)
    sub = subslot(slot, sizeof(T) * prod(d); align = Base.datatype_alignment(T))
    return pool_view(sub, T, d...)
end
