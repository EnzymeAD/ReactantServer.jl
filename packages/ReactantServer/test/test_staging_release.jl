# Staging slots are given back. A worker reads at most `_MAX_HOST_STAGES` cold models into host memory
# ahead of their dispatch (see `_kick_staging!`), and what a model read stays held until its dispatch
# uploads it. Every way out of a staged model's turn must either upload those weights or release them
# and free the slot: a failure before the upload, a request cancelled, withdrawn or past its deadline,
# an unload, a reload. A slot kept past that is kept for good, and once the slots are gone every other
# cold model waits for one forever. Against the MockBackend; uses the helpers of
# test_weight_staging.jl (GatedWeights, _st_entry, _st_infer, _ST_OUT), test_weight_cache.jl (_W,
# _WBYTES) and test_host_cache.jl (CountingWeights).

# Like `_st_entry`, compiled for batches of 1 and 2, so two queued requests share a dispatch. Its
# preprocess stands in for a model.jl with a bug: for a negative input it emits three features where
# the program takes two. Such a request is queued (its batch extents agree and the model reads no
# shape variant), and coalesced with another it fails the concatenation, which runs before the
# dispatch uploads the model's weights.
function _sr_entry(name, weights)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    lm = ReactantServer.LoadedModel(sig, Dict{Int, Any}(1 => exec, 2 => exec), nothing, false, _WBYTES, nothing)
    pre = function (ins)
        x = ins[1].data
        return [ReactantServer.NamedTensor("x", x[1] < 0 ? zeros(Float32, 3, size(x, 2)) : x)]
    end
    return ReactantServer.ModelEntry(
        name, _batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", weights, lm, nothing, pre, identity
    )
end

# Holds the dispatch loop: once `armed`, the execution of a `_sr_hold_entry` model signals `running`
# and waits at `release`, so a test can stage another model while the loop cannot dispatch it.
mutable struct LoopHold
    armed::Bool
    running::Base.Event
    release::Base.Event
end
LoopHold() = LoopHold(false, Base.Event(), Base.Event())

function _sr_hold_entry(name, hold::LoopHold)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    fn = function (args)
        if hold.armed
            notify(hold.running)
            wait(hold.release)
        end
        return [args[1] .* args[2]]
    end
    lm = ReactantServer.LoadedModel(
        sig, Dict{Int, Any}(1 => ReactantServer.MockExecutable(fn, 1)), nothing, false, _WBYTES, nothing
    )
    return ReactantServer.ModelEntry(
        name, _batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", CountingWeights(), lm, nothing,
        identity, identity
    )
end

# A running scheduler with room on the device for one model. `timeout` is the dispatch timeout
# (0 waits indefinitely).
function _sr_scheduler(entries::ReactantServer.ModelEntry...; timeout::Real = 0.0)
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
    end
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, timeout))
    for e in entries
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, _WBYTES)
    ReactantServer.start!(sched)
    return sched
end

# Control ops for `_run_control`. Defined here, before any test starts a dispatch loop: the loop runs
# an op in the world it was started in, which a closure written inside a testset is newer than.
_sr_unload_op(name) = sch -> ReactantServer._evict_entry!(sch, name)
_sr_reload_op(e) = sch -> (ReactantServer._evict_entry!(sch, e.name); ReactantServer._admit_entry!(sch, e))

_sr_stage(sched, e) = lock(() -> e.sched.host_stage, sched.cond)
_sr_queued(sched, e) = lock(() -> length(e.sched.queue), sched.cond)

function _sr_until(f; timeout::Real = 10.0)
    t_end = time() + timeout
    while !f()
        time() > t_end && return false
        sleep(0.01)
    end
    return true
end

_sr_req(name, v; deadline_ns::Integer = 0) = ReactantServer.InferRequest(
    name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[v, v], 2, 1))], deadline_ns
)

# The result of one request, or the error it failed with.
function _sr_infer(sched, name, v; deadline_ns::Integer = 0, cancelled = nothing)
    return try
        ReactantServer.infer(sched, _sr_req(name, v; deadline_ns); cancelled = cancelled)[1].data
    catch err
        err
    end
end

# Queue a good and a bad request for the cold model `e` while its read is held, then let the read
# land: its dispatch coalesces the two and fails before the upload. Returns both replies.
function _sr_fail_before_upload!(sched, e, gw::GatedWeights)
    gw.armed = true
    good = Threads.@spawn _sr_infer(sched, e.name, 1.0f0)
    wait(gw.started)
    bad = Threads.@spawn _sr_infer(sched, e.name, -1.0f0)
    @test _sr_until(() -> _sr_queued(sched, e) == 2)
    gw.armed = false
    notify(gw.gate)
    return fetch(good), fetch(bad)
end

# Both staging slots are free: two cold models are read at once, neither waits for a slot.
function _sr_both_slots_free(sched, es, gws)
    for g in gws
        g.armed = true
    end
    tasks = [Threads.@spawn(_sr_infer(sched, e.name, 1.0f0)) for e in es]
    foreach(g -> wait(g.started), gws)
    @test all(e -> _sr_stage(sched, e) === :staging, es)
    for g in gws
        g.armed = false
        notify(g.gate)
    end
    @test all(t -> fetch(t) == _ST_OUT, tasks)
    return nothing
end

@testset "staging slots: a staged dispatch that fails before its upload gives its slot back" begin
    # Two models staged and failed before the upload would hold both slots, so a third cold model
    # would never be read: with no dispatch timeout its request hangs, and with one it is withdrawn
    # and fails DEADLINE_EXCEEDED, as does every later request for it.
    for timeout in (0.0, 8.0)
        gws = [GatedWeights() for _ in 1:3]
        A = _st_entry("A", CountingWeights())
        Bs = [_sr_entry("B$i", gws[i]) for i in 1:3]
        sched = _sr_scheduler(A, Bs...; timeout = timeout)
        try
            @test _st_infer(sched, "A") == _ST_OUT              # A on the device, every B off it
            @test all(e -> e.executable.weights === nothing, Bs)
            for i in 1:2
                good, bad = _sr_fail_before_upload!(sched, Bs[i], gws[i])
                @test good isa DimensionMismatch && bad isa DimensionMismatch
                @test Bs[i].executable.weights === nothing      # nothing was uploaded
                @test _sr_until(() -> _sr_stage(sched, Bs[i]) === :none)
                @test Bs[i].sched.staged === nothing
            end
            t3 = Threads.@spawn _sr_infer(sched, "B3", 1.0f0)
            r3 = timedwait(() -> istaskdone(t3), 20.0) === :ok ? fetch(t3) : :still_waiting
            @test r3 == _ST_OUT
            @test _st_infer(sched, "B1") == _ST_OUT             # the failed models still serve
        finally
            foreach(g -> notify(g.gate), gws)
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "staging slots: a request cancelled, withdrawn or past its deadline while its model is staged ($how)" for how in (:cancelled, :timeout, :deadline)
    hold = LoopHold()
    gws = [GatedWeights() for _ in 1:3]
    H = _sr_hold_entry("H", hold)
    Bs = [_st_entry("B$i", gws[i]) for i in 1:3]
    sched = _sr_scheduler(H, Bs...; timeout = how === :timeout ? 2.0 : 0.0)
    try
        @test _st_infer(sched, "H") == _ST_OUT                  # H on the device
        hold.armed = true
        tH = Threads.@spawn _sr_infer(sched, "H", 1.0f0)
        wait(hold.running)                                      # the dispatch loop is busy with H
        gone = Threads.Atomic{Bool}(false)
        deadline = how === :deadline ? Int64(time_ns()) + Int64(500_000_000) : 0
        tB = Threads.@spawn _sr_infer(sched, "B1", 1.0f0; deadline_ns = deadline, cancelled = () -> gone[])
        @test _sr_until(() -> _sr_stage(sched, Bs[1]) === :staged)   # read, and waiting for the loop
        held = lock(() -> Bs[1].sched.staged, sched.cond)
        if how === :cancelled
            gone[] = true
            @test fetch(tB) isa ReactantServer.RequestCancelledError
        elseif how === :timeout
            @test fetch(tB) isa ReactantServer.DispatchTimeoutError
        else
            sleep(0.6)                                          # past the deadline, still queued
        end
        hold.armed = false
        notify(hold.release)
        wait(tH)
        how === :deadline && @test fetch(tB) isa ReactantServer.DeadlineExceeded
        # The loop drops the request and lets go of what B1 read, without uploading it.
        @test _sr_until(() -> _sr_stage(sched, Bs[1]) === :none)
        @test held.released && Bs[1].sched.staged === nothing
        @test Bs[1].executable.weights === nothing
        _sr_both_slots_free(sched, Bs[2:3], gws[2:3])
    finally
        hold.armed = false
        notify(hold.release)
        foreach(g -> notify(g.gate), gws)
        ReactantServer.shutdown!(sched)
    end
end

@testset "staging slots: unloading or reloading a staged model gives its slot back ($how)" for how in (:unload, :reload)
    hold = LoopHold()
    gws = [GatedWeights() for _ in 1:3]
    H = _sr_hold_entry("H", hold)
    Bs = [_st_entry("B$i", gws[i]) for i in 1:3]
    sched = _sr_scheduler(H, Bs...)
    try
        @test _st_infer(sched, "H") == _ST_OUT
        hold.armed = true
        tH = Threads.@spawn _sr_infer(sched, "H", 1.0f0)
        wait(hold.running)
        tB = Threads.@spawn _sr_infer(sched, "B1", 1.0f0)
        @test _sr_until(() -> _sr_stage(sched, Bs[1]) === :staged)
        held = lock(() -> Bs[1].sched.staged, sched.cond)
        # A reload is the watcher's single control op: unload the old version, admit the new one.
        v2 = _st_entry("B1", CountingWeights())
        op = how === :unload ? _sr_unload_op("B1") : _sr_reload_op(v2)
        tC = Threads.@spawn ReactantServer._run_control(sched, op)
        @test _sr_until(() -> lock(() -> !isempty(sched.control), sched.cond))
        hold.armed = false
        notify(hold.release)                                    # the loop runs the control op first
        wait(tC)
        wait(tH)
        @test fetch(tB) isa ReactantServer.ModelNotFoundError
        @test held.released && Bs[1].sched.staged === nothing && Bs[1].sched.host_stage === :none
        how === :reload && @test _st_infer(sched, "B1") == _ST_OUT
        _sr_both_slots_free(sched, Bs[2:3], gws[2:3])
    finally
        hold.armed = false
        notify(hold.release)
        foreach(g -> notify(g.gate), gws)
        ReactantServer.shutdown!(sched)
    end
end
