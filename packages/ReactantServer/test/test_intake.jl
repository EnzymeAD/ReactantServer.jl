# Request intake: one client's mistake is refused, or fails, for that client alone, and never costs
# the worker or the other clients. Through the worker's gRPC handlers, against the mock runtime; uses
# the helpers of test_status_cancel.jl (_SC, _sc_entry, _sc_sched, _sc_body, _sc_call, _sc_code).

import InterProcessCommunication as _IK_IPC
using Random: randstring

# How many times the mock program has run. Counted at file scope: the dispatch loop calls the program
# in the world it was started in, so the function must exist before any loop starts.
const _IK_RUNS = Threads.Atomic{Int}(0)
_ik_program(a) = (Threads.atomic_add!(_IK_RUNS, 1); [a[1] .* a[2]])

# A program compiled for x of (2, 4) that, like a static-shape XLA program, refuses anything else.
# Each run records the shape it was given.
const _IK_SHAPES = Tuple{Vararg{Int}}[]
function _ik_program4(a)
    push!(_IK_SHAPES, size(a[1]))
    size(a[1]) == (2, 4) || error("the program takes x of (2, 4), got $(size(a[1]))")
    return [a[1] .* a[2]]
end

# A preprocess with a bug for some inputs: a negative first value gets a third feature row (a wrong
# fixed axis), a zero one comes out as Float64. Both pass validation, which sees the client's x.
function _ik_buggy_pre(ins)
    x = ins[1].data
    x[1] < 0 && return [_SC.NamedTensor("x", zeros(Float32, 3, size(x, 2)))]
    x[1] == 0 && return [_SC.NamedTensor("x", Float64.(x))]
    return ins
end

# Model "pre": x and y f32 (2, n), batch axis last, compiled for a batch of 4 only, so every dispatch
# is padded to 4 and takes as many queued requests as it can.
function _ik_entry4(name)
    spec(n) = _SC.TensorSpec(n, _SC.F32, _SC.Dim[_SC.Dim(_SC.FIXED, 2), _SC.Dim(_SC.BATCH)], 2)
    man = _SC.Manifest(
        "2.0", name, "", _SC.TensorSpec[spec("x")], _SC.TensorSpec[spec("y")], nothing, nothing,
        _SC.BatchingSpec(Int[]), _SC.Provenance(Dict{String, Any}()), 1
    )
    sig = _SC.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    model = _SC.LoadedModel(
        sig, Dict{Int, Any}(4 => _SC.MockExecutable(_ik_program4, 1)), Any[_SC.MockBuffer(Float32[2, 3])]
    )
    return _SC.ModelEntry(name, man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, _ik_buggy_pre, identity)
end

# y from a ModelInfer response's bytes.
function _ik_y(bytes::Vector{UInt8})
    resp = _SC.ProtoBuf.decode(_SC.ProtoBuf.ProtoDecoder(IOBuffer(bytes)), _SC.inference.ModelInferResponse)
    return collect(reinterpret(Float32, resp.raw_output_contents[1]))
end

_ik_bytes(x::Array{Float32}) = collect(reinterpret(UInt8, vec(x)))
_ik_tensor(name, x::Matrix{Float32}) = _SC.inference.var"ModelInferRequest.InferInputTensor"(;
    name, datatype = "FP32", shape = Int64[reverse(size(x))...]
)

@testset "a request naming an input or a requested output twice is refused (WRK-11)" begin
    sched, ctx = _sc_sched(_sc_entry("m"; fn = _ik_program))
    _SC.start!(sched)
    _IK_RUNS[] = 0
    try
        # The input x twice, with different values. A coalesced dispatch used to run the first copy
        # and a request dispatched alone the last, so the answer depended on the load.
        a, b = reshape(Float32[1, 2], 2, 1), reshape(Float32[5, 7], 2, 1)
        twice = _SC._encode_message(
            _SC.inference.ModelInferRequest(;
                model_name = "m", inputs = [_ik_tensor("x", a), _ik_tensor("x", b)],
                raw_input_contents = [_ik_bytes(a), _ik_bytes(b)]
            )
        )
        r = _sc_call(ctx, twice)
        @test _sc_code(r) == _SCStatus.INVALID_ARGUMENT
        @test r isa _SC.gRPCServer.GRPCError && occursin("input 'x' appears more than once", r.message)
        r = _sc_call(ctx, _sc_body("m"; outputs = ["y", "y"]))
        @test _sc_code(r) == _SCStatus.INVALID_ARGUMENT
        @test r isa _SC.gRPCServer.GRPCError && occursin("requested output 'y' appears more than once", r.message)
        @test _IK_RUNS[] == 0                                 # neither reached the model

        # Validation refuses a duplicate as well, for a request that did not come through the codec.
        dup = _SC.InferRequest("m", String[], [_SC.NamedTensor("x", a), _SC.NamedTensor("x", b)], Int64(0))
        err = try
            _SC._validate_inputs(sched.registry.by_name["m"], dup)
        catch e
            e
        end
        @test _sc_code(err) == _SCStatus.INVALID_ARGUMENT

        # A well-formed request still runs.
        @test _sc_call(ctx, _sc_body("m"; outputs = ["y"])) isa Vector{UInt8}
        @test _IK_RUNS[] == 1
    finally
        _SC.shutdown!(sched)
    end
end

@testset "a request whose preprocessed input has the wrong fixed axis or element type fails alone (STG-1)" begin
    sched, ctx = _sc_sched(_ik_entry4("pre"))
    entry = sched.registry.by_name["pre"]
    entry.sched = _SC.ModelSchedState("pre", _SC.ModelSchedConfig(1.0), 0.0)
    sched.running = true                      # accepts work; the dispatches below are run by hand
    empty!(_IK_SHAPES)
    # Queued in this order: two well-formed requests around one whose preprocess adds a feature row,
    # one whose preprocess emits Float64, and two more well-formed ones.
    xs = [Float32[k, 10k] for k in (1, -1, 2, 0, 3, 4)]
    calls = Task[]
    for (i, x) in enumerate(xs)
        push!(calls, Threads.@spawn _sc_call(ctx, _sc_body("pre", reshape(x, 2, 1); outputs = ["y"])))
        @test _sc_until(() -> lock(() -> length(entry.sched.queue), sched.cond) == i)
    end
    # Every request passed intake; the queue holds them in order.
    dispatched = Vector{Int}[]
    while (d = _SC.select_dispatch!(sched, 0.0)) !== nothing
        push!(dispatched, [findfirst(==(qr.req.inputs[1].data[1]), first.(xs)) for qr in d.taken])
        _SC.execute_and_record!(sched, d)
    end
    replies = fetch.(calls)

    # The odd requests ended the coalescing window each time: the one before each ran alone, the odd
    # one ran alone too (it led its own dispatch), and the last two well-formed requests ran together.
    # Before, the first four were merged, the concatenation failed, and all four got INTERNAL.
    @test dispatched == [[1], [2], [3], [4], [5, 6]]
    for i in (1, 3, 5, 6)
        @test replies[i] isa Vector{UInt8}
        @test _ik_y(replies[i]) == xs[i] .* Float32[2, 3]       # its own rows, not a neighbor's
    end
    @test _sc_code(replies[2]) == _SCStatus.INTERNAL            # the program refused 3 feature rows
    @test _sc_code(replies[4]) == _SCStatus.INTERNAL            # and Float64 data
    @test _IK_SHAPES == [(2, 4), (3, 4), (2, 4), (2, 4)]        # the Float64 run stopped before it
    @test entry.sched.requests_served == 4
end

@testset "registering a region of the shared host-weight store is INVALID_ARGUMENT (WRK-9)" begin
    # A stand-in for a weight region, `/rsw-<model>-<digest>`, against the registry a worker serves
    # with (`serve` builds it with the defaults).
    key = "/rsw-wrk9_" * string(getpid(); base = 36) * "_" * string(rand(UInt32); base = 36) * "-0123abcd"
    w = _IK_IPC.SharedMemory(key, 64)
    shm = _SC.SharedMemoryRegistry(; reap_interval = 0)
    try
        req = _SC.inference.SystemSharedMemoryRegisterRequest(; name = "w", key = key, offset = 0, byte_size = 64)
        err = try
            _SC._handle_shm_register(shm, req)
        catch e
            e
        end
        @test _sc_code(err) == _SCStatus.INVALID_ARGUMENT
        @test err isa _SC.gRPCServer.GRPCError && occursin("shared host-weight store", err.message)
        @test isempty(_SC.shm_regions(shm))
    finally
        _SC.shm_teardown!(shm)
        finalize(w)
    end
end

# The `model` label value of every request-metric sample in a worker's exposition.
_ik_request_labels(text) =
    Set(m.captures[1] for m in eachmatch(r"^worker_request\w*\{[^}]*model=\"([^\"]*)\""m, text))

@testset "request metrics label only the models the worker serves (WRK-7)" begin
    sched, ctx = _sc_sched(_sc_entry("m"); metrics = true)
    _SC.start!(sched)
    try
        @test _sc_call(ctx, _sc_body("m")) isa Vector{UInt8}
        # Ten thousand names nobody serves, and an empty one. Each used to add a counter and an
        # 18-bucket histogram that lived as long as the worker.
        codes = [_sc_code(_sc_call(ctx, _sc_body(randstring(16)))) for _ in 1:10_000]
        @test all(==(_SCStatus.NOT_FOUND), codes)
        @test _sc_code(_sc_call(ctx, _sc_body(""))) == _SCStatus.INVALID_ARGUMENT

        io = IOBuffer()
        _SC.Prometheus.expose(io, ctx.metrics.registry)
        text = String(take!(io))
        @test _ik_request_labels(text) == Set(["m", _SC.UNKNOWN_MODEL_LABEL])
        @test _SC.UNKNOWN_MODEL_LABEL == "_unknown"
        @test count(l -> startswith(l, "worker_requests_total{"), split(text, '\n')) == 3
        counted(model, status) = (
            m = match(Regex("^worker_requests_total\\{model=\"$model\",status=\"$status\"\\} (\\S+)", "m"), text);
            m === nothing ? nothing : parse(Float64, m.captures[1])
        )
        @test counted("_unknown", "NOT_FOUND") == 10_000
        @test counted("_unknown", "INVALID_ARGUMENT") == 1
        @test counted("m", "OK") == 1

        # The label follows what the worker serves now: a meta model counts under its own name, and
        # a model no longer registered under the fixed label.
        reg = sched.registry
        @test _SC.request_model_label(reg, "m") == "m"
        reg.meta["pipe"] = _SC.MetaEntry("pipe", reg.by_name["m"].manifest, ["m"], identity)
        @test _SC.request_model_label(reg, "pipe") == "pipe"
        delete!(reg.meta, "pipe")
        @test _SC.request_model_label(reg, "pipe") == "_unknown"
        @test _SC.request_model_label(reg, "") == "_unknown"
    finally
        _SC.shutdown!(sched)
    end
end
