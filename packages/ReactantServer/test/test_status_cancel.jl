# The worker's status contract (every failure gets exactly one gRPC code, and the metrics label is
# that code), the dispatch timeout and cancellation (a caller that stops waiting withdraws its
# request, which then never runs), and a dispatch loop that dies (the worker says so and rejects
# everything with UNAVAILABLE instead of hanging while it reports ready). Mock backend only.

const _SC = ReactantServer
const _SCStatus = ReactantServer.gRPCServer.StatusCode

# One mock model "m": input x and output y, both f32 (2, n) with the batch axis last, compiled at
# batch sizes 1 and 4. `fn` runs the program (default y = x .* w); `pre`/`post` are its hooks.
function _sc_entry(name = "m"; fn = a -> [a[1] .* a[2]], pre = identity, post = identity, variant_spec = Tuple{String, Int}[])
    spec(n) = _SC.TensorSpec(n, _SC.F32, _SC.Dim[_SC.Dim(_SC.FIXED, 2), _SC.Dim(_SC.BATCH)], 2)
    man = _SC.Manifest(
        "2.0", name, "", _SC.TensorSpec[spec("x")], _SC.TensorSpec[spec("y")], nothing, nothing,
        _SC.BatchingSpec(Int[]), _SC.Provenance(Dict{String, Any}()), 1
    )
    sig = _SC.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1, variant_spec)
    exec = _SC.MockExecutable(fn, 1)
    model = _SC.LoadedModel(sig, Dict{Int, Any}(1 => exec, 4 => exec), Any[_SC.MockBuffer(Float32[2, 3])])
    return _SC.ModelEntry(name, man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, pre, post)
end

# A scheduler over `entries` (not started), its InferContext, and the metrics the context counts into.
function _sc_sched(entries...; depth = 64, timeout = 30.0, metrics = false)
    backend = _SC.MockBackend()
    pool = _SC.MemoryPool(backend, _SC.MockClient(), _SC.MockDevice(0), "mock", nothing)
    reg = _SC.ModelRegistry()
    for e in entries
        reg.by_name[e.name] = e
    end
    # FIFO: no cost warmup at start!, which would run a gated program before a test opens its gate.
    sched = _SC.Scheduler(reg, backend, pool, _SC.SchedulerConfig(30.0, depth, timeout; discipline = _SC.FIFO))
    wm = nothing
    if metrics
        cfg = _SC.ServerConfig(
            ["."], "", _SC.RuntimeConfig(_SC.CPU_BACKEND, 0, 0.9, true, true), sched.cfg,
            _SC.EndpointsConfig("127.0.0.1", 0)
        )
        wm = _SC.WorkerMetrics(sched, backend, pool, cfg; worker_name = "w")
    end
    ctx = _SC.InferContext(sched, reg, _SC.SharedMemoryRegistry(), "mock", wm)
    return sched, ctx
end

# A serialized ModelInferRequest for model `model` with x = `x` (Julia shape (2, n)).
function _sc_body(model, x::Matrix{Float32} = Float32[1 2; 3 4][:, 1:1]; outputs = String[])
    req = _SC.inference.ModelInferRequest(;
        model_name = model,
        inputs = [_SC.inference.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[reverse(size(x))...])],
        outputs = [_SC.inference.var"ModelInferRequest.InferRequestedOutputTensor"(; name = o) for o in outputs],
        raw_input_contents = [collect(reinterpret(UInt8, vec(x)))]
    )
    return _SC._encode_message(req)
end

# Run the raw ModelInfer handler and return the GRPCError it throws (or the response bytes).
function _sc_call(ctx, body; cancelled = nothing)
    return try
        _SC._handle_infer_raw(ctx, body, 0; cancelled = cancelled)
    catch e
        e
    end
end

_sc_code(r) = r isa _SC.gRPCServer.GRPCError ? r.code : r
_sc_until(f; timeout = 5.0) = timedwait(f, timeout; pollint = 0.01) === :ok

@testset "status contract: one code per failure, by type" begin
    code = _SC._grpc_status
    @test code(_SC.QueueFullError("m", 4)) == _SCStatus.RESOURCE_EXHAUSTED
    @test code(_SC.WorkerUnavailableError("draining")) == _SCStatus.UNAVAILABLE
    # Not resident under externally-managed residency: another replica may hold it. Never
    # FAILED_PRECONDITION, which a client reads as a stale shared-memory registration.
    @test code(_SC.NotResidentError("m")) == _SCStatus.UNAVAILABLE
    @test code(_SC.ModelNotFoundError("m")) == _SCStatus.NOT_FOUND
    @test code(_SC.ModelNotFoundError("m", "model 'm' was unloaded")) == _SCStatus.NOT_FOUND
    # The reload path's rejection of a request whose model was replaced in preprocess, still a
    # plain ErrorException there, is retryable NOT_FOUND too; any other ErrorException is INTERNAL.
    @test code(
        ErrorException("model 'm' was replaced during the request (the version that preprocessed it was unloaded); retry")
    ) == _SCStatus.NOT_FOUND
    @test code(_SC.DeadlineExceeded("m")) == _SCStatus.DEADLINE_EXCEEDED
    @test code(_SC.DispatchTimeoutError("m", 1.0)) == _SCStatus.DEADLINE_EXCEEDED
    @test code(_SC.RequestCancelledError("m")) == _SCStatus.CANCELLED
    @test code(_SC.InvalidInputError("bad")) == _SCStatus.INVALID_ARGUMENT
    @test code(_SC.BertText.TextInputError("bad")) == _SCStatus.INVALID_ARGUMENT
    @test code(_SC.UnregisteredRegionError("r")) == _SCStatus.FAILED_PRECONDITION
    # Server-side exceptions stay INTERNAL. gRPCServer's own mapping would have made these
    # NOT_FOUND, INVALID_ARGUMENT and OUT_OF_RANGE, telling the client the request was at fault.
    for e in (KeyError(:k), ArgumentError("a"), BoundsError([1], 2), ErrorException("e"), OutOfMemoryError())
        @test code(e) == _SCStatus.INTERNAL
        # Logged on the worker with its backtrace, since the client sees only the message.
        err = @test_logs (:error, "status test failed") _SC._as_grpc_error(e, backtrace(), "status test")
        @test err isa _SC.gRPCServer.GRPCError && err.code == _SCStatus.INTERNAL
    end
    # The metrics label is the wire code's name, for raw exceptions and GRPCErrors alike.
    @test _SC._status_label(KeyError(:k)) == "INTERNAL"
    @test _SC._status_label(_SC.QueueFullError("m", 4)) == "RESOURCE_EXHAUSTED"
    @test _SC._status_label(_SC.gRPCServer.GRPCError(_SCStatus.NOT_FOUND, "x")) == "NOT_FOUND"
    # A long message is cut to fit a header.
    @test ncodeunits(_SC._status_message(ErrorException("x"^5000))) < 1100

    # The response encode: the codec's client errors (ErrorException, ArgumentError) are
    # INVALID_ARGUMENT, a stale region FAILED_PRECONDITION, and anything else is rethrown as is (to
    # become INTERNAL), running out of memory included.
    enc(e) = try
        _SC._as_encode_error(() -> throw(e))
    catch x
        x
    end
    @test _sc_code(enc(ErrorException("requested output 'z' is not produced by the model"))) == _SCStatus.INVALID_ARGUMENT
    @test _sc_code(enc(ArgumentError("shared memory access is out of bounds"))) == _SCStatus.INVALID_ARGUMENT
    @test _sc_code(enc(_SC.UnregisteredRegionError("r"))) == _SCStatus.FAILED_PRECONDITION
    @test enc(OutOfMemoryError()) isa OutOfMemoryError
    @test enc(KeyError(:k)) isa KeyError
    # The decode keeps mapping client data to INVALID_ARGUMENT, except running out of memory.
    dec(e) = try
        _SC._as_invalid(() -> throw(e))
    catch x
        x
    end
    @test _sc_code(dec(ErrorException("malformed"))) == _SCStatus.INVALID_ARGUMENT
    @test dec(OutOfMemoryError()) isa OutOfMemoryError
end

@testset "status contract through the ModelInfer handler, with the label equal to the wire code" begin
    # Hooks that fail in the ways a bundle's can: a client mistake (InvalidInputError, bad UTF-8),
    # and server-side bugs (a KeyError in preprocess, an ArgumentError in postprocess).
    pre_invalid(inputs) = throw(_SC.InvalidInputError("x must be non-negative"))
    pre_keyerror(inputs) = Dict{String, Any}()["missing"]
    post_argerror(outputs) = throw(ArgumentError("postprocess bug"))
    pre_text(inputs) = (_SC.BertText.wire_text(UInt8[0x61, 0xff]); inputs)
    sched, ctx = _sc_sched(
        _sc_entry("ok"), _sc_entry("invalid"; pre = pre_invalid), _sc_entry("keyerr"; pre = pre_keyerror),
        _sc_entry("argerr"; post = post_argerror), _sc_entry("text"; pre = pre_text);
        metrics = true
    )
    _SC.start!(sched)
    try
        @test _sc_call(ctx, _sc_body("ok")) isa Vector{UInt8}
        @test _sc_code(_sc_call(ctx, _sc_body("invalid"))) == _SCStatus.INVALID_ARGUMENT
        @test _sc_code(_sc_call(ctx, _sc_body("text"))) == _SCStatus.INVALID_ARGUMENT
        # Before, gRPCServer turned this KeyError into NOT_FOUND (the gateway then failed over and
        # refreshed its routes) while the metric counted INTERNAL.
        @test _sc_code(_sc_call(ctx, _sc_body("keyerr"))) == _SCStatus.INTERNAL
        @test _sc_code(_sc_call(ctx, _sc_body("argerr"))) == _SCStatus.INTERNAL
        # A requested output the model does not produce is still the client's mistake.
        @test _sc_code(_sc_call(ctx, _sc_body("ok"; outputs = ["nope"]))) == _SCStatus.INVALID_ARGUMENT
        @test _sc_code(_sc_call(ctx, _sc_body("unknown"))) == _SCStatus.NOT_FOUND

        io = IOBuffer()
        _SC.Prometheus.expose(io, ctx.metrics.registry)
        text = String(take!(io))
        counted(model, status) = occursin("worker_requests_total{model=\"$model\",status=\"$status\"}", text)
        @test counted("keyerr", "INTERNAL")
        @test counted("invalid", "INVALID_ARGUMENT")
        @test counted("text", "INVALID_ARGUMENT")
        @test counted("ok", "INVALID_ARGUMENT")
    finally
        _SC.shutdown!(sched)
    end

    # The scheduler's own rejections, by the handler: a full queue is RESOURCE_EXHAUSTED and a
    # stopped scheduler UNAVAILABLE (both were INTERNAL), a model gone from the scheduler NOT_FOUND.
    sched, ctx = _sc_sched(_sc_entry("m"); depth = 1)
    sched.running = true                              # accepts work, but no loop dispatches it
    entry = sched.registry.by_name["m"]
    entry.sched = _SC.ModelSchedState("m", _SC.ModelSchedConfig(1.0), 0.0)
    first = Threads.@spawn _sc_call(ctx, _sc_body("m"))
    @test _sc_until(() -> length(entry.sched.queue) == 1)
    @test _sc_code(_sc_call(ctx, _sc_body("m"))) == _SCStatus.RESOURCE_EXHAUSTED
    # Shutting down answers the queued request UNAVAILABLE, and every later one the same way.
    lock(sched.cond) do
        sched.running = false
        _SC._reject_pending_locked!(sched)
    end
    @test _sc_code(fetch(first)) == _SCStatus.UNAVAILABLE
    @test _sc_code(_sc_call(ctx, _sc_body("m"))) == _SCStatus.UNAVAILABLE
    sched.running = true
    entry.sched = nothing
    @test _sc_code(_sc_call(ctx, _sc_body("m"))) == _SCStatus.NOT_FOUND
end

# A model whose program blocks on `gate`, counting its runs, so a test can hold the GPU busy.
function _sc_gated(gate, ran)
    return _sc_entry(
        "g"; fn = a -> begin
            Threads.atomic_add!(ran, 1)
            wait(gate)
            [a[1] .* a[2]]
        end
    )
end

_sc_req(model = "g") = _SC.InferRequest(model, ["y"], [_SC.NamedTensor("x", reshape(Float32[1, 2], 2, 1))])

@testset "dispatch timeout: a caller waiting too long is answered DEADLINE_EXCEEDED; its request never runs" begin
    gate = Base.Event()
    ran = Threads.Atomic{Int}(0)
    sched, _ = _sc_sched(_sc_gated(gate, ran); timeout = 0.5)
    _SC.start!(sched)
    try
        # A occupies the (mock) GPU; B queues behind it. Both waits pass the 0.5 s timeout.
        a = Threads.@spawn try
            _SC.infer(sched, _sc_req())
        catch e
            e
        end
        @test _sc_until(() -> ran[] == 1)
        t0 = time()
        b = try
            _SC.infer(sched, _sc_req())
        catch e
            e
        end
        @test b isa _SC.DispatchTimeoutError
        @test 0.4 <= time() - t0 < 3.0
        # B is still queued, now withdrawn, so the loop will drop it rather than run it.
        @test only(sched.registry.by_name["g"].sched.queue).withdrawn[]
        @test fetch(a) isa _SC.DispatchTimeoutError     # executing: its result will be discarded
        notify(gate)
        # The loop comes to B, finds it withdrawn and drops it: the program ran for A only.
        @test _sc_until(() -> isempty(sched.registry.by_name["g"].sched.queue))
        sleep(0.2)
        @test ran[] == 1
        # And the worker serves on.
        @test _SC.infer(sched, _sc_req())[1].data == reshape(Float32[2, 6], 2, 1)
    finally
        notify(gate)
        _SC.shutdown!(sched)
    end

    # 0 disables the timeout: a slow request simply completes.
    slow = _sc_entry("s"; fn = a -> (sleep(0.3); [a[1] .* a[2]]))
    sched, _ = _sc_sched(slow; timeout = 0.0)
    _SC.start!(sched)
    try
        @test _SC.infer(sched, _sc_req("s"))[1].data == reshape(Float32[2, 6], 2, 1)
    finally
        _SC.shutdown!(sched)
    end
end

@testset "cancellation: a cancelled call withdraws its request and is answered CANCELLED" begin
    gate = Base.Event()
    ran = Threads.Atomic{Int}(0)
    sched, ctx = _sc_sched(_sc_gated(gate, ran))
    _SC.start!(sched)
    try
        a = Threads.@spawn _SC.infer(sched, _sc_req())
        @test _sc_until(() -> ran[] == 1)
        flag = Threads.Atomic{Bool}(false)
        b = Threads.@spawn try
            _SC.infer(sched, _sc_req(); cancelled = () -> flag[])
        catch e
            e
        end
        @test _sc_until(() -> length(sched.registry.by_name["g"].sched.queue) == 1)
        sleep(0.2)
        @test !istaskdone(b)                       # waits while its client is still there
        flag[] = true
        @test _sc_until(() -> istaskdone(b); timeout = 2.0)
        @test fetch(b) isa _SC.RequestCancelledError
        notify(gate)
        @test fetch(a)[1].data == reshape(Float32[2, 6], 2, 1)   # A, not cancelled, completes
        @test _sc_until(() -> isempty(sched.registry.by_name["g"].sched.queue))
        sleep(0.2)
        @test ran[] == 1                           # B never ran

        # A call already cancelled is neither preprocessed nor queued.
        @test_throws _SC.RequestCancelledError _SC.infer(sched, _sc_req(); cancelled = () -> true)
        # A meta's sub-calls re-enter `infer` with the call's probe in scope, and inherit it.
        @test_throws _SC.RequestCancelledError Base.ScopedValues.with(
            () -> _SC.infer(sched, _sc_req()), _SC._CANCEL_PROBE => () -> true
        )
        @test ran[] == 1

        # Through the registered ModelInfer RPC, with a ServerContext whose cancelled flag is set
        # (gRPCServer 0.1.2 never sets it; this is the plumbing that takes effect once it does).
        sctx = _SC.gRPCServer.ServerContext(; payload = ctx)
        sctx.cancelled = true
        r = try
            _SC._model_infer_rpc(sctx, _sc_body("g"))
        catch e
            e
        end
        @test _sc_code(r) == _SCStatus.CANCELLED
        sctx.cancelled = false
        @test _SC._model_infer_rpc(sctx, _sc_body("g")) isa Vector{UInt8}
    finally
        notify(gate)
        _SC.shutdown!(sched)
    end
end

@testset "a dispatch loop that dies is fatal: logged, not live or ready, everything UNAVAILABLE" begin
    # A shape-variant model whose variant is read from input "v". A request without it fails the
    # variant lookup, which `infer` now catches for its own request...
    e = _sc_entry("v"; variant_spec = [("v", 1)])
    sched, ctx = _sc_sched(e, _sc_entry("m"))
    failures = Any[]
    _SC.on_dispatch_failure!(sched, err -> push!(failures, err))
    _SC.start!(sched)
    try
        @test_throws ErrorException _SC.infer(sched, _sc_req("v"))
        @test _SC.dispatch_alive(sched)
        @test isempty(failures)

        # ...but one that reaches the queue anyway (here by `submit!` directly) kills the loop in
        # selection, where no single request can be held to it. Before, the loop died silently and
        # every later request blocked forever while the worker reported ready.
        qr = _SC.QueuedRequest(_sc_req("v"))
        _SC.submit!(sched, qr)
        r = take!(qr.reply)
        @test r isa _SC.WorkerUnavailableError
        @test _sc_code(_SC._as_grpc_error(r, nothing, "t")) == _SCStatus.UNAVAILABLE
        @test !_SC.dispatch_alive(sched)
        @test length(failures) == 1 && only(failures) isa KeyError
        @test sched.failure isa KeyError

        # Every later request and control command is rejected UNAVAILABLE rather than queued.
        @test _sc_code(_sc_call(ctx, _sc_body("m"))) == _SCStatus.UNAVAILABLE
        @test_throws _SC.WorkerUnavailableError _SC.evict!(sched, "m")

        # The worker reports itself not live and not ready, everywhere it is asked.
        @test !_SC._handle_server_live(ctx).live
        @test !_SC._handle_server_ready(ctx).ready
        @test !_SC._model_ready(ctx, "m")
        @test !_SC._RegistryReady(sched.registry, sched)()
        @test !_SC._DispatchLive(sched)()

        # A hook installed after the death runs at once (a blocking serve installs the exit hook
        # right after bring-up, so a loop that dies in between still exits the process).
        late = Any[]
        _SC.on_dispatch_failure!(sched, err -> push!(late, err))
        @test length(late) == 1
    finally
        _SC.shutdown!(sched)
    end

    # A healthy worker says live and ready.
    sched, ctx = _sc_sched(_sc_entry("m"))
    _SC.start!(sched)
    try
        @test _SC._handle_server_live(ctx).live
        @test _SC._handle_server_ready(ctx).ready
        @test _SC._RegistryReady(sched.registry, sched)()
    finally
        _SC.shutdown!(sched)
    end
end

@testset "a failure outside the execution fails only the requests still unanswered" begin
    answered = _SC.QueuedRequest(_sc_req("m"))
    put!(answered.reply, _SC.DeadlineExceeded("m"))
    open_qr = _SC.QueuedRequest(_sc_req("m"))
    _SC._fail_unanswered!([answered, open_qr], ErrorException("boom"))
    @test take!(answered.reply) isa _SC.DeadlineExceeded
    @test !isready(answered.reply)                 # not answered a second time
    @test take!(open_qr.reply) isa ErrorException
end
