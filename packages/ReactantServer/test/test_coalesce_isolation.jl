# A request whose batched inputs disagree on their batch extent must never share a dispatch or be
# split (SCH-1). `_executable_rows` counts a request's first batched input alone and
# `_coalesce_inputs` padded every input by that count, so such a request coalesced with others
# could line the merged shapes up: the program ran and handed an innocent caller a row of its own
# input paired with a neighbor's.

const _CI = ReactantServer

# A two-input model, y = a .+ b', over a (2, n) and b (n): the batch axes sit at different positions
# (axis 2 of a, axis 1 of b), so each extent is read along its own axis.
function _ci_manifest(name)
    ina = _CI.TensorSpec("a", _CI.F32, _CI.Dim[_CI.Dim(_CI.FIXED, 2), _CI.Dim(_CI.BATCH)], 2)
    inb = _CI.TensorSpec("b", _CI.F32, _CI.Dim[_CI.Dim(_CI.BATCH)], 1)
    outy = _CI.TensorSpec("y", _CI.F32, _CI.Dim[_CI.Dim(_CI.FIXED, 2), _CI.Dim(_CI.BATCH)], 2)
    return _CI.Manifest(
        "2.0", name, "", _CI.TensorSpec[ina, inb], _CI.TensorSpec[outy], nothing, nothing,
        _CI.BatchingSpec(Int[]), _CI.Provenance(Dict{String, Any}()), 1
    )
end

# One program per compiled size n that, like a static-shape XLA program, refuses inputs of any other
# extent. Every call is recorded as (rows of a, rows of b), so a test can see what each dispatch ran.
function _ci_model(sizes::Vector{Int}, calls::Vector{Tuple{Int, Int}})
    prog(n) = _CI.MockExecutable(1) do args
        a, b = args[1], args[2]
        push!(calls, (size(a, 2), length(b)))
        (size(a, 2) == n && length(b) == n) ||
            error("program for batch $n got inputs of $(size(a, 2)) and $(length(b)) rows")
        return [a .+ reshape(b, 1, :)]
    end
    sig = _CI.ModelSignature(["a", "b"], DataType[Float32, Float32], String[], 1, ["y"], DataType[Float32], 1)
    return _CI.LoadedModel(sig, Dict{Int, Any}(n => prog(n) for n in sizes), Any[])
end

function _ci_scheduler(sizes::Vector{Int}, calls::Vector{Tuple{Int, Int}})
    backend = _CI.MockBackend()
    pool = _CI.MemoryPool(backend, _CI.MockClient(), _CI.MockDevice(0), "mock", nothing)
    reg = _CI.ModelRegistry()
    reg.by_name["pair"] = _CI.ModelEntry(
        "pair", _ci_manifest("pair"), Dict{Int, Vector{UInt8}}(), "", nothing,
        _ci_model(sizes, calls), nothing, identity, identity
    )
    return _CI.Scheduler(reg, backend, pool, _CI.SchedulerConfig(30.0, 1024, 30.0))
end

_ci_req(a, b) = _CI.InferRequest(
    "pair", ["y"], [_CI.NamedTensor("a", a), _CI.NamedTensor("b", b)], Int64(0)
)
_ci_qr(a, b) = (req = _ci_req(a, b); _CI.QueuedRequest(req, req.inputs, 0.0, Channel{Any}(1)))

# Drive select_dispatch! and execute_and_record! directly (no spawned loop) so the dispatches are
# deterministic.
@testset "a request whose batch extents disagree never shares a dispatch" begin
    calls = Tuple{Int, Int}[]
    s = _ci_scheduler([8], calls)
    entry = s.registry.by_name["pair"]
    st = entry.sched = _CI.ModelSchedState("pair", _CI.ModelSchedConfig(1.0), 0.0)

    X = _ci_qr(fill(100.0f0, 2, 3), Float32[4, 5])     # a: 3 rows, b: 2
    Y = _ci_qr(fill(10.0f0, 2, 1), Float32[1])         # well formed, 1 row
    Z = _ci_qr(fill(20.0f0, 2, 1), Float32[2, 3])      # a: 1 row, b: 2
    W1 = _ci_qr(fill(30.0f0, 2, 1), Float32[6])
    W2 = _ci_qr(fill(40.0f0, 2, 2), Float32[7, 8])
    @test _CI._batch_extents_agree(entry, Y.prepared)
    @test !_CI._batch_extents_agree(entry, X.prepared)
    @test !_CI._batch_extents_agree(entry, [_CI.NamedTensor("a", zeros(Float32, 2, 1))])   # b missing
    append!(st.queue, [X, Y, Z, W1, W2])

    # X leads and goes alone; it used to take Y and Z with it, both merged inputs coming to 5 rows
    # plus 3 of padding, so the program ran and Y's row of a met Z's first row of b.
    d = _CI.select_dispatch!(s, 0.0)
    @test d.taken == [X]
    _CI.execute_and_record!(s, d)
    err = take!(X.reply)
    @test err isa ErrorException && occursin("got inputs of 3 and 2 rows", err.msg)   # unpadded

    # Y goes next, padded to the compiled size; Z ends its window.
    d = _CI.select_dispatch!(s, 0.0)
    @test d.taken == [Y]
    @test d.size == 8
    _CI.execute_and_record!(s, d)
    y = take!(Y.reply)

    d = _CI.select_dispatch!(s, 0.0)
    @test d.taken == [Z]
    _CI.execute_and_record!(s, d)
    @test take!(Z.reply) isa ErrorException

    # The well-formed requests behind Z coalesce as before.
    d = _CI.select_dispatch!(s, 0.0)
    @test d.taken == [W1, W2]
    @test d.size == 8
    _CI.execute_and_record!(s, d)
    @test only(take!(W1.reply)).data == fill(36.0f0, 2, 1)
    @test only(take!(W2.reply)).data == Float32[47 48; 47 48]
    @test _CI.select_dispatch!(s, 0.0) === nothing

    # Y's answer is exactly what it gets alone.
    solo = _ci_qr(fill(10.0f0, 2, 1), Float32[1])
    push!(st.queue, solo)
    _CI.execute_and_record!(s, _CI.select_dispatch!(s, 0.0))
    @test only(y).data == only(take!(solo.reply)).data == fill(11.0f0, 2, 1)

    # X and Z each reached the program on their own, unpadded; only the good dispatches counted.
    @test calls == [(3, 2), (8, 8), (1, 2), (8, 8), (8, 8)]
    @test st.dispatch_count == 3
    @test st.requests_served == 4
end

@testset "a lone request whose batch extents disagree fails alone and is never split" begin
    calls = Tuple{Int, Int}[]
    s = _ci_scheduler([1, 4], calls)
    entry = s.registry.by_name["pair"]
    _CI.start!(s)
    empty!(calls)    # start! warms each compiled size; count only the requests
    try
        # Alone: it runs at its own extent with its inputs as sent, so the run fails for this caller.
        bad = _ci_req(fill(1.0f0, 2, 3), Float32[1, 2])
        err = try
            _CI.infer(s, bad)
        catch e
            e
        end
        @test err isa ErrorException && occursin("batch size 3", err.msg)
        @test isempty(calls)

        # Over the largest compiled size (4): it used to be cut into 4 + 4 + 2 rows of each input,
        # which ran, dropped b's eleventh row and answered. Now it runs whole and fails.
        big = _ci_req(fill(1.0f0, 2, 10), Float32.(1:11))
        @test _CI._max_compiled_rows(entry, big.inputs) === nothing
        err = try
            _CI.infer(s, big)
        catch e
            e
        end
        @test err isa ErrorException && occursin("batch size 10", err.msg)
        @test isempty(calls)

        # A well-formed oversize request is still split and joined back.
        good = _ci_req(fill(1.0f0, 2, 10), Float32.(1:10))
        @test _CI._max_compiled_rows(entry, good.inputs) == 4
        @test only(_CI.infer(s, good)).data == fill(1.0f0, 2, 10) .+ reshape(Float32.(1:10), 1, :)
        @test calls == [(4, 4), (4, 4), (4, 4)]
    finally
        _CI.shutdown!(s)
    end
end

@testset "_coalesce_inputs refuses inputs whose rows disagree" begin
    # plan_batch never merges such a request; this guards the merge itself, which would otherwise
    # pad every input by the first input's shortfall and leave them out of step.
    entry = _ci_scheduler([8], Tuple{Int, Int}[]).registry.by_name["pair"]
    ok = [_CI.NamedTensor("a", zeros(Float32, 2, 2)), _CI.NamedTensor("b", zeros(Float32, 2))]
    bad = [_CI.NamedTensor("a", zeros(Float32, 2, 3)), _CI.NamedTensor("b", zeros(Float32, 2))]
    merged = _CI._coalesce_inputs(entry, [ok, ok], 4, 8)
    @test size(_CI._named(merged, "a").data) == (2, 8)
    @test size(_CI._named(merged, "b").data) == (8,)
    @test_throws ErrorException _CI._coalesce_inputs(entry, [bad], 3, 8)
    @test_throws ErrorException _CI._coalesce_inputs(entry, [bad, ok], 5, 8)
end
