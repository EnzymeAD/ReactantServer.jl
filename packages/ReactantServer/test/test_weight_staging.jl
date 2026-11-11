# Loading weights and running long programs without stalling the worker. A model off the device gets
# its weights read into host memory by a background task while the dispatch loop keeps serving the
# models already on the device, and the slow calls the loop does make (loads, long executions) run
# off the interactive thread, which is also the thread Julia services its I/O on. Against the MockBackend; uses the helpers of
# test_weight_cache.jl (_W, _WBYTES, _batched_manifest) and test_host_cache.jl (CountingWeights).

# A weights double whose reads wait at a gate once armed, so a test can hold a model's background
# read in progress. `fail_next` makes the next read throw instead.
mutable struct GatedWeights
    tensors::Dict{String, Array}
    armed::Bool
    gate::Base.Event          # notify to let held reads through
    started::Base.Event       # notified when a held read begins
    reads::Threads.Atomic{Int}
    fail_next::Bool
end
GatedWeights() = GatedWeights(
    Dict{String, Array}("w" => copy(_W)), false, Base.Event(), Base.Event(), Threads.Atomic{Int}(0), false
)
Base.haskey(g::GatedWeights, k) = haskey(g.tensors, k)
function Base.getindex(g::GatedWeights, k)
    if g.armed
        notify(g.started)
        wait(g.gate)
    end
    Threads.atomic_add!(g.reads, 1)
    if g.fail_next
        g.fail_next = false
        error("simulated read failure")
    end
    return g.tensors[k]
end
ReactantServer.weight_spec(g::GatedWeights, k) = (eltype(g.tensors[k]), size(g.tensors[k]))

function _st_entry(name, weights)
    sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    exec = ReactantServer.MockExecutable(args -> [args[1] .* args[2]], 1)
    lm = ReactantServer.LoadedModel(sig, Dict{Int, Any}(1 => exec), nothing, false, _WBYTES, nothing)
    return ReactantServer.ModelEntry(
        name, _batched_manifest(name), Dict{Int, Vector{UInt8}}(), "", weights, lm, nothing,
        identity, identity
    )
end

# A running scheduler with room on the device for `device_models` models, and `B` (gated) off the
# device while `A` is on it, so the next request for B needs a load from its weights file.
function _st_running(gw::GatedWeights; device_models::Int = 1)
    A = _st_entry("A", CountingWeights())
    B = _st_entry("B", gw)
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    reg.by_name["A"] = A
    reg.by_name["B"] = B
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    for e in (A, B)
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, device_models * _WBYTES)
    ReactantServer.start!(sched)
    @test _st_infer(sched, "A") == _ST_OUT          # A on the device, B off it
    @test B.executable.weights === nothing
    gw.reads[] = 0
    gw.armed = true
    return sched, A, B
end

const _ST_OUT = reshape(Float32[10, 100], 2, 1)
_st_infer(sched, name) = ReactantServer.infer(
    sched, ReactantServer.InferRequest(name, ["y"], [ReactantServer.NamedTensor("x", reshape(Float32[1, 1], 2, 1))])
)[1].data

@testset "staging: a model's weights are read in the background while resident models keep serving" begin
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        tB = Threads.@spawn _st_infer(sched, "B")
        wait(gw.started)                                # B's read has begun, off the dispatch loop
        @test B.sched.host_stage === :staging
        for _ in 1:3
            @test _st_infer(sched, "A") == _ST_OUT      # served while B's read is held
        end
        @test !istaskdone(tB)
        @test A.executable.weights !== nothing          # nothing was evicted for B ahead of its turn
        notify(gw.gate)
        @test fetch(tB) == _ST_OUT
        @test gw.reads[] == 1                           # read once, in the background; the dispatch only uploaded
        @test B.sched.host_stage === :none && B.sched.staged === nothing
        @test B.executable.weights !== nothing
    finally
        notify(gw.gate)
        ReactantServer.shutdown!(sched)
    end
end

@testset "staging: a failed background read fails the waiting requests; the dispatch loop never reads" begin
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        gw.armed = false
        gw.fail_next = true
        err = try
            _st_infer(sched, "B")
        catch e
            e
        end
        @test err isa ErrorException && occursin("reading the weights of model 'B' failed", err.msg)
        @test gw.reads[] == 1                           # no second read on the dispatch loop
        @test B.sched.host_stage === :none
        @test _st_infer(sched, "B") == _ST_OUT          # the next request starts a fresh read
        @test gw.reads[] == 2
        @test _st_infer(sched, "A") == _ST_OUT
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "staging: past the staging cap a model waits for a slot instead of reading on the dispatch loop" begin
    # Three models off the device, each read held at its own gate: two read in the background, and
    # the third waits (`:pending`) rather than being read on the loop, which keeps serving A.
    gws = [GatedWeights() for _ in 1:3]
    A = _st_entry("A", CountingWeights())
    Bs = [_st_entry("B$i", gws[i]) for i in 1:3]
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in (A, Bs...)
        reg.by_name[e.name] = e
    end
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    for e in (A, Bs...)
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, 1 * _WBYTES)
    ReactantServer.start!(sched)
    try
        @test _st_infer(sched, "A") == _ST_OUT
        for g in gws
            g.reads[] = 0
            g.armed = true
        end
        tasks = [Threads.@spawn(_st_infer(sched, "B$i")) for i in 1:3]
        t_end = time() + 10
        while count(e -> e.sched.host_stage === :pending, Bs) < 1 && time() < t_end
            sleep(0.01)
        end
        @test count(e -> e.sched.host_stage === :staging, Bs) == 2
        @test count(e -> e.sched.host_stage === :pending, Bs) == 1
        @test _st_infer(sched, "A") == _ST_OUT          # the loop is free meanwhile
        for g in gws
            g.armed = false
            notify(g.gate)
        end
        @test all(t -> fetch(t) == _ST_OUT, tasks)
        @test all(e -> e.sched.host_stage === :none, Bs)
        @test all(g -> g.reads[] == 1, gws)            # each model read once, all in the background
    finally
        for g in gws
            notify(g.gate)
        end
        ReactantServer.shutdown!(sched)
    end
end

@testset "staging: unloading a model does not wait for its background read, which drops what it read" begin
    # The unload runs on the dispatch loop, and a read on storage that stopped answering never
    # returns, so the unload does not wait for it. The read keeps the weights file open (see
    # test_stalled_reads.jl) and drops what it read when it returns.
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        tB = Threads.@spawn try
            _st_infer(sched, "B")
        catch err
            err
        end
        wait(gw.started)
        tE = Threads.@spawn ReactantServer.evict!(sched, "B")
        @test timedwait(() -> istaskdone(tE), 10.0) === :ok      # with B's read still held
        @test fetch(tE) === B
        @test fetch(tB) isa ReactantServer.ModelNotFoundError   # the queued request was rejected as unloaded
        @test _st_infer(sched, "A") == _ST_OUT          # the worker carries on
        @test lock(() -> length(sched.reads), sched.cond) == 1
        notify(gw.gate)
        t_end = time() + 10
        while lock(() -> !isempty(sched.reads), sched.cond) && time() < t_end
            sleep(0.01)
        end
        @test lock(() -> isempty(sched.reads), sched.cond)
        @test B.sched.host_stage === :none && B.sched.staged === nothing
    finally
        notify(gw.gate)
        ReactantServer.shutdown!(sched)
    end
end

@testset "staging: with the private host cache, a staged read is cached once uploaded" begin
    gw = GatedWeights()
    A = _st_entry("A", CountingWeights())
    B = _st_entry("B", gw)
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    reg.by_name["A"] = A
    reg.by_name["B"] = B
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    for e in (A, B)
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    wc = ReactantServer.WeightCache(backend, pool, reg, _WBYTES; host_budget = 2 * sizeof(_W))
    sched.weight_cache = wc
    ReactantServer.start!(sched)
    try
        _st_infer(sched, "A")
        _st_infer(sched, "B")                           # B evicts A
        n = gw.reads[]
        hits = wc.host_hits
        _st_infer(sched, "A")                           # A from its host copy, staged in the background
        _st_infer(sched, "B")                           # and B from its
        @test gw.reads[] == n                           # no further file read
        @test wc.host_hits == hits + 2
        @test haskey(wc.host_cached, "A") && haskey(wc.host_cached, "B")
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "off_interactive: a long blocking call made from the interactive thread leaves it free" begin
    if Threads.nthreads(:interactive) == 0
        @test_skip "needs an interactive thread (run with --threads=auto,1)"
    else
        # A call that blocks its thread without yielding, as a weight read or upload does.
        block() = (
            t_end = time() + 1.5; while time() < t_end
                @ccall gc_safe = true usleep(50_000::Cuint)::Cint
            end
        )
        # The worst lateness of a 10 ms timer on the interactive pool while `f` runs there. The timer
        # shares the interactive thread with `f`, so it measures exactly what must stay free; on the
        # default pool it would share a thread with the offloaded work itself whenever that pool has
        # one thread (Julia 1.12's default without --threads), and be starved by design.
        function worst_lag(f)
            done = Threads.Atomic{Bool}(false)
            worst = Threads.Atomic{Float64}(0.0)
            hb = Threads.@spawn :interactive while !done[]
                t = time()
                sleep(0.01)
                Threads.atomic_max!(worst, time() - t - 0.01)
            end
            fetch(Threads.@spawn :interactive f())
            done[] = true
            wait(hb)
            return worst[]
        end
        # The hazard itself: blocked in place, the interactive thread stops timers (and request
        # handling) for the whole call. If this stops holding, the case below proves nothing.
        @test worst_lag(block) > 1.0
        @test worst_lag(() -> ReactantServer.off_interactive(block)) < 0.5
        # An error comes back as itself, not wrapped in a TaskFailedException.
        err = fetch(
            Threads.@spawn :interactive try
                ReactantServer.off_interactive(() -> throw(ArgumentError("boom")))
            catch e
                e
            end
        )
        @test err isa ArgumentError
    end
end

@testset "a long execution runs off the interactive thread and leaves it free" begin
    if Threads.nthreads(:interactive) == 0
        @test_skip "needs an interactive thread (run with --threads=auto,1)"
    else
        # A program that holds its thread for 1.5 s without yielding, as the wait for a GPU program's
        # results does.
        slow = function (args)
            t_end = time() + 1.5
            while time() < t_end
                @ccall gc_safe = true usleep(50_000::Cuint)::Cint
            end
            return [args[1] .* args[2]]
        end
        sig = ReactantServer.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
        lm = ReactantServer.LoadedModel(
            sig, Dict{Int, Any}(1 => ReactantServer.MockExecutable(slow, 1)),
            Any[ReactantServer.MockBuffer(copy(_W))], true, _WBYTES, nothing
        )
        S = ReactantServer.ModelEntry(
            "S", _batched_manifest("S"), Dict{Int, Vector{UInt8}}(), "",
            FakeWeights(Dict{String, Array}("w" => copy(_W))), lm, nothing, identity, identity
        )
        backend = ReactantServer.MockBackend()
        pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
        reg = ReactantServer.ModelRegistry()
        reg.by_name["S"] = S
        sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
        S.sched = ReactantServer.ModelSchedState("S", ReactantServer.ModelSchedConfig(1.0), 0.0)
        ReactantServer.start!(sched)
        try
            # The timer runs on the interactive thread, beside the dispatch loop, which is what must
            # stay free; the offloaded run may well occupy the only default-pool thread.
            done = Threads.Atomic{Bool}(false)
            worst = Threads.Atomic{Float64}(0.0)
            hb = Threads.@spawn :interactive while !done[]
                t = time()
                sleep(0.01)
                Threads.atomic_max!(worst, time() - t - 0.01)
            end
            @test _st_infer(sched, "S") == _ST_OUT
            done[] = true
            wait(hb)
            @test worst[] < 0.5                         # timers kept firing through the 1.5 s run
            # The cost is measured where the program ran, so it reflects the run, not the hand-off.
            @test 1.4 < ReactantServer._cost_seconds(S.sched, ReactantServer.VariantKey(), 1) < 3.0
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "stage_weights: weights read ahead of a placement change are uploaded at the first dispatch, not read again" begin
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        gw.armed = false
        rows = ReactantServer.stage_weights!(sched, ["B", "A"]; hold_seconds = 30)
        byname = Dict(r.name => r for r in rows)
        @test byname["A"].state == "resident"            # on the device: nothing to read
        @test byname["B"].state == "staged" && byname["B"].read
        @test byname["B"].nbytes == _WBYTES
        @test gw.reads[] == 1
        @test B.executable.weights === nothing           # read into host memory only
        @test haskey(sched.prefetched, "B")
        # Asking again while the hold stands reads nothing.
        @test only(ReactantServer.stage_weights!(sched, ["B"])).state == "staged"
        @test gw.reads[] == 1
        @test _st_infer(sched, "B") == _ST_OUT
        @test gw.reads[] == 1                            # the dispatch uploaded what was staged
        @test isempty(sched.prefetched)
        @test_throws ReactantServer.ModelNotFoundError ReactantServer.stage_weights!(sched, ["nope"])
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "stage_weights: an unclaimed hold is released when it expires" begin
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        gw.armed = false
        @test only(ReactantServer.stage_weights!(sched, ["B"]; hold_seconds = 0.5)).state == "staged"
        ReactantServer._sweep_prefetched!(sched)
        @test haskey(sched.prefetched, "B")
        ReactantServer._sweep_prefetched!(sched; now = monotonic_time() + 1.0)
        @test isempty(sched.prefetched)
        @test _st_infer(sched, "B") == _ST_OUT           # a request then reads it the usual way
        @test gw.reads[] == 2
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "stage_weights: past the held-bytes cap a model is deferred" begin
    es = [_st_entry(n, CountingWeights()) for n in ("A", "B", "C")]
    backend = ReactantServer.MockBackend()
    pool = ReactantServer.MemoryPool(backend, ReactantServer.MockClient(), ReactantServer.MockDevice(0), "mock", nothing)
    reg = ReactantServer.ModelRegistry()
    for e in es
        reg.by_name[e.name] = e
    end
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 1024, 30.0))
    for e in es
        e.sched = ReactantServer.ModelSchedState(e.name, ReactantServer.ModelSchedConfig(1.0), 0.0)
    end
    # A host budget of three models' weights: the cap on held bytes is half of it, one model. Room on
    # the device for one, which startup fills.
    sched.weight_cache = ReactantServer.WeightCache(backend, pool, reg, _WBYTES; host_budget = 3 * sizeof(_W))
    ReactantServer.start!(sched)
    try
        off = [e.name for e in es if e.executable.weights === nothing]
        @test length(off) == 2
        rows = Dict(r.name => r.state for r in ReactantServer.stage_weights!(sched, off))
        @test sort(collect(values(rows))) == ["deferred", "staged"]
        @test length(sched.prefetched) == 1
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "load_costs: upload and read times, the host copy, and loads per model" begin
    gw = GatedWeights()
    sched, A, B = _st_running(gw)
    try
        gw.armed = false
        c = ReactantServer.load_costs(sched, ["A", "B"])
        @test c["A"].upload_seconds > 0 && c["A"].read_seconds > 0
        @test c["A"].loads >= 1
        @test !c["B"].host_copy && B.executable.weights === nothing
        snap = ReactantServer.control_status(sched)
        @test snap.host_weight_budget_bytes == 0
        resp = ReactantServer._handle_model_control_status(ReactantServer.InferContext(sched, sched.registry, ReactantServer.SharedMemoryRegistry(), "mock"))
        ms = Dict(m.name => m for m in resp.models)
        @test ms["A"].upload_seconds ≈ c["A"].upload_seconds && ms["A"].loads == c["A"].loads
        staged = ReactantServer._handle_stage_weights(
            ReactantServer.InferContext(sched, sched.registry, ReactantServer.SharedMemoryRegistry(), "mock"), ReactantServer.control.StageWeightsRequest(; names = ["B"], hold_seconds = 10.0)
        )
        @test only(staged.models).state == "staged" && only(staged.models).weight_nbytes == _WBYTES
    finally
        ReactantServer.shutdown!(sched)
    end
end
