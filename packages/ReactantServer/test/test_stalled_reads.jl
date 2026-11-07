# Weight reads that stall (STG-4). Storage that stops answering (a hard NFS mount, a shared-store lock
# held by a stuck worker) leaves a background weights read in a system call that never returns. The
# worker abandons a read that makes no progress for `scheduler.weight_read_stall_seconds`: the
# requests waiting for it fail with UNAVAILABLE, its read slot passes on, nothing waits for it, and
# whatever it returns with, if it ever does, is released. Real weights files, read through the
# `_PREAD` seam, which these tests make wait on chosen files as a hung mount does (and, for the shared
# store, a real flock held as by another worker). Against the MockBackend; uses the helpers of
# test_weight_cache.jl (_W, _WBYTES, _batched_manifest) and test_weight_staging.jl (_ST_OUT).

const _SL_DIR = mktempdir()

# Reads of the files whose descriptors are registered here wait at the file's gate, as on storage
# that stopped answering; `_SL_SLOW` files return `bytes` per read after `pause` seconds, as on
# storage that is slow but moving. `_SL_READS` counts the reads of each file.
const _SL_HELD = Dict{Cint, Base.Event}()
const _SL_SLOW = Dict{Cint, Tuple{Float64, Int}}()
const _SL_READS = Dict{Cint, Int}()
const _SL_LOCK = ReentrantLock()
const _SL_WAITING = Threads.Atomic{Int}(0)          # reads waiting at a gate now

function _sl_pread(fd, ptr, n, off)
    gate, slow = lock(_SL_LOCK) do
        _SL_READS[fd] = get(_SL_READS, fd, 0) + 1
        return (get(_SL_HELD, fd, nothing), get(_SL_SLOW, fd, nothing))
    end
    if gate !== nothing
        Threads.atomic_add!(_SL_WAITING, 1)
        try
            wait(gate)
        finally
            Threads.atomic_sub!(_SL_WAITING, 1)
        end
    end
    if slow !== nothing
        sleep(slow[1])
        n = min(n, slow[2])
    end
    return ReactantServer._pread(fd, ptr, n, off)
end

_sl_fd(e) = ReactantServer._fd(e.weights)
_sl_reads(e) = lock(() -> get(_SL_READS, _sl_fd(e), 0), _SL_LOCK)

function _sl_hold!(e)
    g = Base.Event()
    lock(() -> (_SL_HELD[_sl_fd(e)] = g), _SL_LOCK)
    return g
end

function _sl_release!(e)
    g = lock(() -> pop!(_SL_HELD, _sl_fd(e), nothing), _SL_LOCK)
    g === nothing || notify(g)
    return nothing
end

function _sl_release_all!()
    gs = lock(_SL_LOCK) do
        out = collect(values(_SL_HELD))
        empty!(_SL_HELD)
        empty!(_SL_SLOW)
        return out
    end
    foreach(notify, gs)
    return nothing
end

# A model whose single weight `_W` is in a weights file of its own, read with pread. With `path`,
# the model opens that file again instead, as a second worker loading the same bundle does.
function _sl_entry(name; path = nothing)
    if path === nothing
        path = joinpath(_SL_DIR, "$name-$(time_ns()).safetensors")
        SafeTensors.serialize(path, Dict("w" => copy(_W)), Dict("argument_order" => JSON3.write(["w"])))
    end
    w, id = ReactantServer.open_weights_file(path)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    lm = ReactantServer.LoadedModel(sig, Dict{Int, Any}(1 => exec), nothing, false, _WBYTES, nothing)
    mlir = Dict{ReactantServer.VariantKey, Dict{Int, Vector{UInt8}}}(ReactantServer.VariantKey() => Dict{Int, Vector{UInt8}}())
    return ReactantServer.ModelEntry(name, _batched_manifest(name), mlir, path, w, lm, nothing, identity, identity, id)
end

# A running scheduler with room on the device for `device_models` models and nothing on it yet (FIFO
# runs no warmup), whose reads go through `_sl_pread`, abandoned after `stall` seconds without
# progress, and which starts no read with `bound` stalled reads outstanding. The read path is run
# once first, unwatched, so that compiling it never counts against a watched read's stall limit.
function _sl_scheduler(
        entries...; stall = 1.0, timeout = 10.0, device_models = 1, bound = 4,
        store = ReactantServer.PrivateWeightStore(), mode = ReactantServer.SELF_MANAGED
    )
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
    end
    cfg = ReactantServer.SchedulerConfig(
        30.0, 1024, timeout; discipline = ReactantServer.FIFO, weight_read_stall_seconds = stall
    )
    sched = ReactantServer.Scheduler(reg, backend, pool, cfg)
    for e in entries
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    wc = ReactantServer.WeightCache(backend, pool, reg, device_models * _WBYTES; store = store, mode = mode)
    ReactantServer.release_staged!(wc, entries[1], ReactantServer.stage_host_weights(wc, entries[1]); uploaded = false)
    sched.weight_cache = wc
    sched.stalled_read_bound = bound
    ReactantServer.start!(sched)
    return sched
end

# The result of one request, or the error it failed with.
function _sl_infer(sched, name)
    req = ReactantServer.InferRequest(name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))])
    return try
        ReactantServer.infer(sched, req)[1].data
    catch err
        err
    end
end

# Whether `err` reaches a client as UNAVAILABLE.
_sl_unavailable(err) = err isa Exception &&
    ReactantServer._grpc_status(err) == ReactantServer.gRPCServer.StatusCode.UNAVAILABLE

function _sl_until(f; timeout::Real = 10.0)
    t_end = time() + timeout
    while !f()
        time() > t_end && return false
        sleep(0.01)
    end
    return true
end

_sl_outstanding(sched) = ReactantServer.weight_cache_metrics(sched).stalled_reads_outstanding

@testset "stalled reads: requests for the model fail UNAVAILABLE soon after the stall limit, then at once until the read returns" begin
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A = _sl_entry("A")
    B = _sl_entry("B")
    sched = _sl_scheduler(A, B; stall = 1.0, timeout = 10.0)
    try
        @test _sl_infer(sched, "A") == _ST_OUT              # A on the device, B off it
        _sl_hold!(B)
        t0 = time()
        tB = Threads.@spawn _sl_infer(sched, "B")
        @test _sl_until(() -> _SL_WAITING[] == 1)           # B's read hangs, off the dispatch loop
        @test _sl_infer(sched, "A") == _ST_OUT              # and A serves meanwhile
        err = fetch(tB)
        dt = time() - t0
        @test _sl_unavailable(err)                          # not the 10 s dispatch timeout's DEADLINE_EXCEEDED
        @test dt < 4.0
        @test err isa ReactantServer.WeightsReadStalledError && occursin("stalled", sprint(showerror, err))
        # The read is still outstanding: B is not read again, and its requests fail at once.
        t1 = time()
        err2 = _sl_infer(sched, "B")
        @test _sl_unavailable(err2) && time() - t1 < 1.0
        @test _SL_WAITING[] == 1                            # no second read was started
        @test _sl_infer(sched, "A") == _ST_OUT
        @test ReactantServer.dispatch_alive(sched)          # the worker still serves (and reports ready)
        m = ReactantServer.weight_cache_metrics(sched)
        @test m.stalled_reads == 1 && m.stalled_reads_outstanding == 1
        # The read returns: what it read is dropped, and B is read again for its next request.
        _sl_release!(B)
        @test _sl_until(() -> _sl_outstanding(sched) == 0)
        @test B.executable.weights === nothing && B.sched.staged === nothing
        @test _sl_infer(sched, "B") == _ST_OUT
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: two stall and a third cold model still serves; at the bound no read starts" begin
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A, B1, B2, C, D, E = (_sl_entry(n) for n in ("A", "B1", "B2", "C", "D", "E"))
    sched = _sl_scheduler(A, B1, B2, C, D, E; stall = 1.0, timeout = 8.0, device_models = 2, bound = 3)
    try
        @test _sl_infer(sched, "A") == _ST_OUT
        _sl_hold!(B1)
        _sl_hold!(B2)
        t1 = Threads.@spawn _sl_infer(sched, "B1")
        t2 = Threads.@spawn _sl_infer(sched, "B2")
        @test _sl_until(() -> _SL_WAITING[] == 2)           # both read slots taken by hung reads
        t0 = time()
        tC = Threads.@spawn _sl_infer(sched, "C")          # waits for a slot
        # The two stalled reads are abandoned, their slots pass on, and C is read and served.
        rC = timedwait(() -> istaskdone(tC), 20.0) === :ok ? fetch(tC) : :still_waiting
        @test rC == _ST_OUT
        @test time() - t0 < 5.0                             # not C's 8 s dispatch timeout
        @test _sl_unavailable(fetch(t1)) && _sl_unavailable(fetch(t2))
        @test _sl_outstanding(sched) == 2
        # A third stalled read reaches the bound: no read starts, so E, whose storage is fine, fails
        # at once, while the models on the device serve.
        _sl_hold!(D)
        @test _sl_unavailable(_sl_infer(sched, "D"))
        @test _sl_outstanding(sched) == 3
        t3 = time()
        errE = _sl_infer(sched, "E")
        @test _sl_unavailable(errE) && time() - t3 < 1.0
        @test occursin("no new read starts", sprint(showerror, errE))
        @test E.executable.weights === nothing
        @test _sl_infer(sched, "A") == _ST_OUT && _sl_infer(sched, "C") == _ST_OUT
        @test ReactantServer.dispatch_alive(sched)
        # One returns: below the bound again, E is read and served.
        _sl_release!(B1)
        @test _sl_until(() -> _sl_outstanding(sched) == 2)
        @test _sl_infer(sched, "E") == _ST_OUT
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: a read waiting on a shared-store lock stalls, and its region is released when it returns" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        Core_ = ReactantServer.ReactantServerCore
        prefix = "tsl" * string(getpid())
        R = Core_._layout([(Float32, (2,))])[2]
        A = _sl_entry("A")
        B = _sl_entry("B")
        digest = ReactantServer.weights_digest(
            "B", [(Float32, (2,))]; content = ReactantServer.weights_file_token(B.weights_id)
        )
        lockpath = "/dev/shm/" * prefix * "-B-" * string(digest; base = 16) * ".lock"
        store = ReactantServer.SharedWeightStore(; budget = 4R, prefix = prefix)
        sched = _sl_scheduler(A, B; stall = 1.0, timeout = 8.0, store = store)
        held = Cint(-1)
        try
            @test _sl_infer(sched, "A") == _ST_OUT
            # Another worker holds B's region exclusively and never finishes: B's read waits on the
            # lock, reading nothing, which is no progress.
            held = Core_._open_lock(lockpath)
            @test Core_._flock(held, Core_._LOCK_EX) == 0
            t0 = time()
            err = _sl_infer(sched, "B")
            @test _sl_unavailable(err) && time() - t0 < 4.0
            m = ReactantServer.weight_cache_metrics(sched)
            @test m.stalled_reads == 1 && m.stalled_reads_outstanding == 1
            # The lock is let go: the read takes it, fills the region and returns holding it, and what
            # it holds is released, so no holder is left on the region.
            Core_._flock(held, Core_._LOCK_UN)
            Core_._close_fd(held)
            held = Cint(-1)
            @test _sl_until(() -> _sl_outstanding(sched) == 0)
            region_free = function ()
                probe = Core_._open_lock(lockpath)
                free = Core_._flock(probe, Core_._LOCK_EX | Core_._LOCK_NB) == 0
                free && Core_._flock(probe, Core_._LOCK_UN)
                Core_._close_fd(probe)
                return free
            end
            @test _sl_until(region_free)
            @test B.executable.weights === nothing && B.sched.staged === nothing
            @test B.sched.host_stage === :none
            @test _sl_infer(sched, "B") == _ST_OUT          # read again, from the region it filled
        finally
            held >= 0 && (Core_._flock(held, Core_._LOCK_UN); Core_._close_fd(held))
            _sl_release_all!()
            ReactantServer.shutdown!(sched)
            for f in readdir("/dev/shm"; join = true)
                startswith(basename(f), prefix * "-") && rm(f; force = true)
            end
        end
    end
end

@testset "stalled reads: StageWeights answers a stalled read with a failed row and leaves no hold" begin
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A = _sl_entry("A")
    B = _sl_entry("B")
    sched = _sl_scheduler(A, B; stall = 1.0, timeout = 8.0)
    try
        @test _sl_infer(sched, "A") == _ST_OUT
        _sl_hold!(B)
        tS = Threads.@spawn ReactantServer.stage_weights!(sched, ["B"])
        ok = timedwait(() -> istaskdone(tS), 6.0) === :ok
        @test ok                                            # within the stall limit, not when the read returns
        ok || _sl_release!(B)
        row = only(fetch(tS))
        @test row.state == "failed" && occursin("stalled", row.error)
        @test !haskey(sched.prefetched, "B")                # no hold keeps B `:pending`
        # The read is still outstanding: requests for B fail at once, and so does another StageWeights.
        @test _sl_unavailable(_sl_infer(sched, "B"))
        @test only(ReactantServer.stage_weights!(sched, ["B"])).state == "failed"
        @test _sl_infer(sched, "A") == _ST_OUT
        # Once it returns, B is read for its requests as usual.
        _sl_release!(B)
        @test _sl_until(() -> _sl_outstanding(sched) == 0)
        @test _sl_infer(sched, "B") == _ST_OUT
        @test isempty(sched.prefetched)
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: unloading a model with a read in progress or stalled does not wait; the read closes its file ($how)" for how in (:reading, :stalled)
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A = _sl_entry("A")
    B = _sl_entry("B")
    sched = _sl_scheduler(A, B; stall = how === :stalled ? 1.0 : 60.0, timeout = 8.0)
    try
        @test _sl_infer(sched, "A") == _ST_OUT
        _sl_hold!(B)
        tB = Threads.@spawn _sl_infer(sched, "B")
        @test _sl_until(() -> _SL_WAITING[] == 1)
        how === :stalled && @test _sl_unavailable(fetch(tB))
        # The unload runs on the dispatch loop, and returns with B's read still held there.
        tE = Threads.@spawn ReactantServer.evict!(sched, "B")
        ok = timedwait(() -> istaskdone(tE), 5.0) === :ok
        @test ok
        ok || _sl_release!(B)
        @test fetch(tE) === B
        how === :reading && @test fetch(tB) isa ReactantServer.ModelNotFoundError
        @test _sl_infer(sched, "A") == _ST_OUT
        # The file stays open while the read may still use its descriptor, and is closed once it returns.
        @test isopen(B.weights.io) == ok
        _sl_release!(B)
        @test _sl_until(() -> !isopen(B.weights.io))
        @test lock(() -> isempty(sched.reads), sched.cond)
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: a read that is slow but moving is never abandoned" begin
    # Two bytes every 0.3 s: the read takes longer than the 1 s stall limit, but it never goes that
    # long without progress.
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A = _sl_entry("A")
    B = _sl_entry("B")
    sched = _sl_scheduler(A, B; stall = 1.0, timeout = 10.0)
    try
        @test _sl_infer(sched, "A") == _ST_OUT
        lock(() -> (_SL_SLOW[_sl_fd(B)] = (0.3, 2)), _SL_LOCK)
        t0 = time()
        @test _sl_infer(sched, "B") == _ST_OUT
        @test time() - t0 > 1.1
        @test ReactantServer.weight_cache_metrics(sched).stalled_reads == 0
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: a residency change whose read stalls fails UNAVAILABLE and holds up no other ($target)" for target in (:device, :system)
    # Under externally managed residency the SetModelResidency RPC reads the weights before its
    # control step, holding the lock that orders residency changes. A read that never returns used to
    # hang that call and every residency change after it.
    old = ReactantServer._PREAD[]
    ReactantServer._PREAD[] = _sl_pread
    A = _sl_entry("A")
    B = _sl_entry("B")
    sched = _sl_scheduler(A, B; stall = 1.0, mode = ReactantServer.EXTERNALLY_MANAGED)
    ctx = ReactantServer.InferContext(sched, sched.registry, ReactantServer.SharedMemoryRegistry(), "mock")
    Ctl = ReactantServer.control
    wire = target === :device ? Ctl.Residency.PINNED_DEVICE : Ctl.Residency.PINNED_SYSTEM
    rpc(name) = try
        ReactantServer._handle_set_model_residency(ctx, Ctl.SetModelResidencyRequest(; name = name, target = wire)).residency
    catch err
        err
    end
    unavailable(err) = err isa ReactantServer.gRPCServer.GRPCError &&
        err.code == ReactantServer.gRPCServer.StatusCode.UNAVAILABLE
    try
        _sl_hold!(B)
        tB = Threads.@spawn rpc("B")
        ok = timedwait(() -> istaskdone(tB), 6.0) === :ok
        @test ok                                            # within the stall limit, not when the read returns
        ok || _sl_release!(B)
        err = fetch(tB)
        @test unavailable(err) && occursin("stalled", err.message)
        @test !islocked(sched.residency_lock)
        # Another model's change goes through; the stalled model's fails at once while its read is out.
        @test rpc("A") == wire
        t0 = time()
        @test unavailable(rpc("B")) && time() - t0 < 1.0
        # Once the read returns, what it read is let go, and the change for B goes through.
        _sl_release!(B)
        @test _sl_until(() -> _sl_outstanding(sched) == 0)
        @test rpc("B") == wire
    finally
        _sl_release_all!()
        ReactantServer._PREAD[] = old
        ReactantServer.shutdown!(sched)
    end
end

@testset "stalled reads: waiting for a peer's fill of a shared region counts its progress; a stuck fill still stalls" begin
    # Two workers on a node share the region of a model both want: the first fills it while holding
    # its lock, and the second waits for the lock. A slow fill that keeps moving must not get the
    # waiter abandoned (a large model on slow storage takes far longer than the stall limit to
    # fill), and a fill stuck on storage that stopped answering must.
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        Core_ = ReactantServer.ReactantServerCore
        old = ReactantServer._PREAD[]
        ReactantServer._PREAD[] = _sl_pread
        prefix = "tsp" * string(getpid())
        R = Core_._layout([(Float32, (2,))])[2]
        A1, B1, C1 = (_sl_entry(n) for n in ("A", "B", "C"))
        A2, B2, C2 = (_sl_entry(e.name; path = e.weights_path) for e in (A1, B1, C1))
        store() = ReactantServer.SharedWeightStore(; budget = 8R, prefix = prefix)
        # A 2 s limit, so that compiling the waiting path on its first use here does not count
        # against it.
        w1 = _sl_scheduler(A1, B1, C1; stall = 2.0, timeout = 10.0, store = store())
        w2 = _sl_scheduler(A2, B2, C2; stall = 2.0, timeout = 10.0, store = store())
        try
            # Slow but moving: worker 1 reads B a byte every 0.4 s, 3.2 s in all, past the limit.
            lock(() -> (_SL_SLOW[_sl_fd(B1)] = (0.4, 1)), _SL_LOCK)
            t1 = Threads.@spawn _sl_infer(w1, "B")
            # Worker 1 has read a byte of B into its region, so it is the one filling it.
            filling() = any(r -> (@atomic r.progress.bytes) > 0, lock(() -> copy(w1.reads), w1.cond))
            @test _sl_until(filling)
            n2 = _sl_reads(B2)
            t0 = time()
            @test _sl_infer(w2, "B") == _ST_OUT             # worker 2 waited for it, and was served
            @test time() - t0 > 2.0                         # for longer than the stall limit
            @test fetch(t1) == _ST_OUT
            @test _sl_reads(B2) == n2                       # from the region worker 1 filled
            @test ReactantServer.weight_cache_metrics(w2).stalled_reads == 0
            @test ReactantServer.weight_cache_metrics(w1).stalled_reads == 0
            # Stuck: worker 1's read of C hangs in the middle of its fill, so worker 2, waiting for
            # C's region, sees no progress and is abandoned like worker 1.
            _sl_hold!(C1)
            t1 = Threads.@spawn _sl_infer(w1, "C")
            @test _sl_until(() -> _SL_WAITING[] == 1)
            t0 = time()
            @test _sl_unavailable(_sl_infer(w2, "C")) && time() - t0 < 5.0
            @test _sl_unavailable(fetch(t1))
            # The fill goes on: worker 1 finishes it, and both reads return and let go of the region.
            _sl_release!(C1)
            @test _sl_until(() -> _sl_outstanding(w1) == 0 && _sl_outstanding(w2) == 0)
            @test _sl_infer(w2, "C") == _ST_OUT && _sl_infer(w1, "C") == _ST_OUT
        finally
            _sl_release_all!()
            ReactantServer._PREAD[] = old
            ReactantServer.shutdown!(w1)
            ReactantServer.shutdown!(w2)
            for f in readdir("/dev/shm"; join = true)
                startswith(basename(f), prefix * "-") && rm(f; force = true)
            end
        end
    end
end
