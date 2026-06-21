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

const PoolBacking = Union{SharedMemory,Memory{UInt8}}

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
end

"""
    BufferPool(n_bytes; n_slots=8, use_shm=true, name="reactant_server_pool")

Allocate a staging pool of `n_bytes` divided into `n_slots` equal slots. `slot_bytes` is fixed
at construction (`n_bytes ÷ n_slots`), not recomputed per request, so the allocator can hand
disjoint slots to concurrent callers. A SHM-backed pool can be registered with a server; an
inline pool (`use_shm=false`) is the fallback transport.
"""
function BufferPool(n_bytes::Integer; n_slots::Integer = 8, use_shm::Bool = true,
                    name::AbstractString = "reactant_server_pool",
                    key::Union{AbstractString,Nothing} = nothing)
    n_bytes > 0 || throw(ArgumentError("BufferPool: n_bytes must be positive (got $n_bytes)"))
    n_slots > 0 || throw(ArgumentError("BufferPool: n_slots must be positive (got $n_slots)"))
    slot_bytes = Int(n_bytes) ÷ Int(n_slots)
    slot_bytes > 0 ||
        throw(ArgumentError("BufferPool: n_bytes ($n_bytes) too small for $n_slots slots"))
    usable = slot_bytes * Int(n_slots)

    backing, region_name = if use_shm
        # An explicit `key` (e.g. one the supervisor minted and injected so peer workers can attach
        # the same region) is used verbatim; otherwise derive a per-process key from `name`.
        k = key === nothing ? shm_key(String(name)) : String(key)
        shm = SharedMemory(k, usable)
        shm, (startswith(k, "/") ? k[2:end] : k)
    else
        Memory{UInt8}(undef, usable), ""
    end

    alloc_lock = ReentrantLock()
    return BufferPool(backing, usable, region_name, slot_bytes, Int(n_slots),
                      trues(Int(n_slots)), alloc_lock, Threads.Condition(alloc_lock),
                      UInt64[], UInt64(0))
end

Base.sizeof(pool::BufferPool) = pool.n_bytes
is_shm_backed(pool::BufferPool) = pool.backing isa SharedMemory
pool_region_name(pool::BufferPool) = pool.name
pool_slot_bytes(pool::BufferPool) = pool.slot_bytes

function _pool_base_pointer(pool::BufferPool)
    pool.backing isa SharedMemory ? convert(Ptr{UInt8}, pointer(pool.backing)) : pointer(pool.backing)
end

# Base address of the pool's backing region. Used by the meta fan-out to detect, by pointer range,
# whether a sub-call input's bytes already live in this pool (so it can be sent by SHM reference
# instead of inlined) — robust to the caller reshaping/sub-viewing the scratch buffer.
pool_base_pointer(pool::BufferPool) = _pool_base_pointer(pool)

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
    acquire_slot!(pool, span=1) -> PoolSlot

Block until `span` physically contiguous slots are free, then return them as one slot whose
`[offset, offset + span*slot_bytes)` range no other in-flight slot overlaps. Throws an
`ArgumentError` immediately if `span` exceeds the pool's total slot count, since such a
request could never be satisfied and would otherwise deadlock. Waiters are served in FIFO
order. Pair with `release_slot!`, which frees the whole run.
"""
function acquire_slot!(pool::BufferPool, span::Integer = 1)
    s = Int(span)
    s >= 1 || throw(ArgumentError("acquire_slot!: span must be >= 1 (got $s)"))
    s <= pool.n_slots || throw(ArgumentError(
        "acquire_slot!: span $s exceeds the pool's $(pool.n_slots) slots of " *
        "$(pool.slot_bytes) bytes each; this request can never be satisfied. " *
        "Increase the pool's total bytes or decrease n_slots so each slot is larger."))

    lock(pool.alloc_lock)
    token = UInt64(0)   # sentinel keeps the finally safe if we are interrupted pre-enqueue
    try
        token = (pool.token_counter += 1)
        push!(pool.waitq, token)
        while true
            if first(pool.waitq) == token
                start = _find_free_run(pool.free, s)
                if start != 0
                    for i in start:(start + s - 1)
                        pool.free[i] = false
                    end
                    return PoolSlot(pool, start, (start - 1) * pool.slot_bytes,
                                    s * pool.slot_bytes, s)
                end
            end
            wait(pool.freed)   # releases alloc_lock while parked, reacquires on wake
        end
    finally
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
            pool.free[i] && throw(ArgumentError(
                "release_slot!: slot index $i is already free (double release?)"))
            pool.free[i] = true
        end
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
        throw(ArgumentError("subslot of $n bytes exceeds parent slot capacity " *
                            "($(parent.capacity - parent.cursor) bytes remaining)"))
    child = PoolSlot(parent.pool, 0, parent.offset + parent.cursor, n, 0, 0)
    parent.cursor += n
    return child
end

# Reset a slot's cursor so it can be re-partitioned when recycled.
reset_slot!(slot::PoolSlot) = (slot.cursor = 0; slot)

# A Julia array view of the slot's bytes typed as T. The pool owns the memory; the view is
# ephemeral.
function pool_view(slot::PoolSlot, ::Type{T}, dims::Integer...) where {T}
    n_bytes = Base.checked_mul(Int(sizeof(T)), reduce(Base.checked_mul, Int.(dims); init=1))
    n_bytes <= slot.capacity ||
        throw(ArgumentError("pool_view ($n_bytes bytes) exceeds slot capacity ($(slot.capacity) bytes)"))
    ptr = Ptr{T}(_pool_base_pointer(slot.pool) + slot.offset)
    return unsafe_wrap(Array, ptr, dims)
end

# Pinned pool references keyed by the aliased Memory's objectid. Same rationale as
# _SHM_KEEPALIVE: a closure-only finalizer is not reliable, and keying by Memory itself would
# hold it strongly and prevent its finalizer from ever running.
const _POOL_KEEPALIVE = Dict{UInt,BufferPool}()
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
    key = objectid(mem)
    @lock _POOL_KEEPALIVE_LOCK _POOL_KEEPALIVE[key] = pool
    finalizer(mem) do m
        @lock _POOL_KEEPALIVE_LOCK delete!(_POOL_KEEPALIVE, objectid(m))
    end
    return mem
end

# FixedSizeArray view over a pool slot, backed by the slot's Memory{T} alias.
pool_fsa(slot::PoolSlot, ::Type{T}, dims::NTuple{N,<:Integer}) where {T,N} =
    fsa_from_memory(pool_memory(slot, T, prod(dims)), dims)

pool_fsa(slot::PoolSlot, ::Type{T}, dims::Vararg{Integer,N}) where {T,N} = pool_fsa(slot, T, dims)
