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
# previous Channel-based free list.

const PoolBacking = Union{SharedMemory, Memory{UInt8}}

mutable struct BufferPool
    backing::PoolBacking
    n_bytes::Int
    name::String          # SHM region name (the basename clients register); "" for inline pools
    slot_bytes::Int
    n_slots::Int
    free::BitVector              # true = the 1-based slot index is free
    alloc_lock::ReentrantLock    # guards free / waitq / token_counter
    freed::Threads.Condition     # tied to alloc_lock; signaled on release and waiter exit
    waitq::Vector{UInt64}        # FIFO of waiter tokens; only the head may allocate
    token_counter::UInt64
    closed::Bool                 # close() was called: no new slots (guarded by alloc_lock)
    released::Bool               # the SHM mapping has been released (guarded by alloc_lock)
    fd::Union{ShmFd, Nothing}    # the SHM object's descriptor, kept open; nothing for inline pools
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
at construction (`n_bytes ÷ n_slots`), not recomputed per request, so the allocator can hand
disjoint slots to concurrent callers. A SHM-backed pool can be registered with a server; an
inline pool (`use_shm=false`) is the fallback transport. A SHM pool's object name is minted per
pool from `name` with a fresh random token (two pools never share a name, even one after the other
in one process), unless an explicit `key` is given.
"""
function BufferPool(
        n_bytes::Integer; n_slots::Integer = 8, use_shm::Bool = true,
        name::AbstractString = "reactant_server_pool",
        key::Union{AbstractString, Nothing} = nothing
    )
    n_bytes > 0 || throw(ArgumentError("BufferPool: n_bytes must be positive (got $n_bytes)"))
    n_slots > 0 || throw(ArgumentError("BufferPool: n_slots must be positive (got $n_slots)"))
    slot_bytes = Int(n_bytes) ÷ Int(n_slots)
    slot_bytes > 0 ||
        throw(ArgumentError("BufferPool: n_bytes ($n_bytes) too small for $n_slots slots"))
    usable = slot_bytes * Int(n_slots)

    backing, region_name, fd = if use_shm
        # An explicit `key` (e.g. one the supervisor minted and injected so peer workers can attach
        # the same region) is used verbatim; otherwise a fresh key is minted from `name`.
        _create_pool_object(name, key, usable)
    else
        Memory{UInt8}(undef, usable), "", nothing
    end

    alloc_lock = ReentrantLock()
    return BufferPool(
        backing, usable, region_name, slot_bytes, Int(n_slots),
        trues(Int(n_slots)), alloc_lock, Threads.Condition(alloc_lock),
        UInt64[], UInt64(0), false, false, fd
    )
end

Base.sizeof(pool::BufferPool) = pool.n_bytes
is_shm_backed(pool::BufferPool) = pool.backing isa SharedMemory
pool_region_name(pool::BufferPool) = pool.name
pool_slot_bytes(pool::BufferPool) = pool.slot_bytes

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
# pool_memory alias alive. Caller holds alloc_lock. `finalize` runs InterProcessCommunication's
# _destroy (munmap) now instead of at some later GC, and unregisters it so it never runs twice.
function _release_if_idle!(pool::BufferPool)
    (pool.closed && !pool.released && pool.backing isa SharedMemory) || return
    all(pool.free) || return
    _has_pinned_alias(pool) && return
    pool.released = true
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
end

PoolSlot(pool::BufferPool, index::Integer, offset::Integer, capacity::Integer, span::Integer) =
    PoolSlot(pool, Int(index), Int(offset), Int(capacity), Int(span), 0)

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
            wait(pool.freed)   # releases alloc_lock while parked, reacquires on wake
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
    release_slot!(slot)

Return a slot acquired with `acquire_slot!` to the pool's free set, freeing every physical
slot in its span. Releasing a derived subslot (`index == 0`) is a no-op. Throws if any slot
in the span is already free (double release).
"""
function release_slot!(slot::PoolSlot)
    slot.index == 0 && return nothing
    pool = slot.pool
    @lock pool.alloc_lock begin
        for i in slot.index:(slot.index + slot.span - 1)
            pool.free[i] && throw(
                ArgumentError(
                    "release_slot!: slot index $i is already free (double release?)"
                )
            )
            pool.free[i] = true
        end
        _release_if_idle!(pool)
        notify(pool.freed)
    end
    return nothing
end

# Carve a child slot of `n_bytes` from the next cursor position of `parent`. The parent's
# cursor advances; the child starts with a fresh cursor at zero and index 0 (not separately
# releasable, its lifetime is bounded by the parent's).
function subslot(parent::PoolSlot, n_bytes::Integer)
    n = Int(n_bytes)
    n >= 0 || throw(ArgumentError("subslot byte count must be non-negative (got $n)"))
    parent.cursor + n <= parent.capacity ||
        throw(
        ArgumentError(
            "subslot of $n bytes exceeds parent slot capacity " *
                "($(parent.capacity - parent.cursor) bytes remaining)"
        )
    )
    child = PoolSlot(parent.pool, 0, parent.offset + parent.cursor, n, 0, 0)
    parent.cursor += n
    return child
end

# Reset a slot's cursor so it can be re-partitioned when recycled.
reset_slot!(slot::PoolSlot) = (slot.cursor = 0; slot)

# A Julia array view of the slot's bytes typed as T. The pool owns the memory; the view is
# ephemeral.
function pool_view(slot::PoolSlot, ::Type{T}, dims::Integer...) where {T}
    n_bytes = Base.checked_mul(Int(sizeof(T)), reduce(Base.checked_mul, Int.(dims); init = 1))
    n_bytes <= slot.capacity ||
        throw(ArgumentError("pool_view ($n_bytes bytes) exceeds slot capacity ($(slot.capacity) bytes)"))
    ptr = Ptr{T}(_pool_base_pointer(slot.pool) + slot.offset)
    return unsafe_wrap(Array, ptr, dims)
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
    ptr = Ptr{T}(_pool_base_pointer(pool) + slot.offset)
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
"""
scratch(slot::PoolSlot, dims, ::Type{T}) where {T} =
    first(scratch(slot, Pair[(dims isa Tuple ? dims : (dims,)) => T]))

function scratch(slot::PoolSlot, specs::AbstractVector)
    return Any[_carve_scratch(slot, first(s), last(s)) for s in specs]
end

function _carve_scratch(slot::PoolSlot, dims, ::Type{T}) where {T}
    d = dims isa Tuple ? dims : (dims,)
    return pool_view(subslot(slot, sizeof(T) * prod(d)), T, d...)
end
