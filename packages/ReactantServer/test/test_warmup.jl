# The request-path warm-up (warmup.jl): before a model serves, its model.jl hooks and the worker code
# specialized on its tensor types run once on a synthesized request, so its first real request does
# not compile them. The warm-up must run no program and leave every scheduling and request counter
# as it was, and a hook that fails or hangs on the synthesized request must not stop the model.

using ReactantServer: warm_request_path!, warm_request_paths!, load_bundle_entry, load_model!,
    MockBackend, MockClient, MockDevice, MockBuffer, MockExecutable, MemoryPool, ModelEntry,
    ModelRegistry, ModelSignature, LoadedModel, Scheduler, SchedulerConfig, InferRequest,
    NamedTensor, TensorSpec, Dim, FIXED, BATCH, F32, Manifest, BatchingSpec, Provenance, infer
using Logging: with_logger
using Test: TestLogger

# A batched mock model: x (2, n) f32 -> y (2, n) f32, y = x .* w, compiled at sizes 1 and 4. The
# program counts its executions, and the hooks count their calls.
function _wu_mock(name; pre = identity, post = identity)
    spec(n) = TensorSpec(n, F32, Dim[Dim(FIXED, 2), Dim(BATCH)], 2)
    man = Manifest(
        "2.0", name, "", TensorSpec[spec("x")], TensorSpec[spec("y")], nothing, nothing,
        BatchingSpec(Int[]), Provenance(Dict{String, Any}()), 1
    )
    runs = Threads.Atomic{Int}(0)
    exec = MockExecutable(args -> (Threads.atomic_add!(runs, 1); [args[1] .* args[2]]), 1)
    sig = ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1)
    model = LoadedModel(sig, Dict{Int, Any}(1 => exec, 4 => exec), Any[MockBuffer(Float32[10, 100])])
    entry = ModelEntry(name, man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, pre, post)
    return entry, runs
end

function _wu_mock_sched(entry)
    backend = MockBackend()
    pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
    reg = ModelRegistry()
    reg.by_name[entry.name] = entry
    return Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
end

# Everything the scheduler keeps about a model that scheduling reads or the metrics report.
_wu_sched_state(st) = (
    st.vtime, st.recent_compute_ema, st.ema_last_update, st.dispatch_count, st.requests_served,
    st.rows_served, st.total_compute, copy(st.wait_samples), copy(st.batch_size_hist),
    length(st.queue), st.host_stage,
    sort!([(k, c.seconds, c.runs, c.refreshed, c.remeasured) for (k, c) in st.cost_estimate]),
)

@testset "warm-up: runs each hook once, runs no program, and changes no scheduling state" begin
    pre_calls = Threads.Atomic{Int}(0)
    post_calls = Threads.Atomic{Int}(0)
    entry, runs = _wu_mock(
        "wu";
        pre = ts -> (Threads.atomic_add!(pre_calls, 1); ts),
        post = ts -> (Threads.atomic_add!(post_calls, 1); ts)
    )
    sched = _wu_mock_sched(entry)
    ReactantServer.start!(sched)       # fair: the cost warmup runs each program twice
    try
        cfg = ReactantServer.ServerConfig(
            ["."], "", ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
            sched.cfg, ReactantServer.EndpointsConfig("127.0.0.1", 0)
        )
        metrics = ReactantServer.WorkerMetrics(sched, sched.backend, sched.pool, cfg; worker_name = "w")
        exposed() = filter(
            l -> startswith(l, "worker_"), split(sprint(io -> ReactantServer.Prometheus.expose(io, metrics.registry)), '\n')
        )
        # A scrape decays the EMA to the time of the scrape, so the state is read after it.
        lines = exposed()
        runs_before, state, vfloor = runs[], _wu_sched_state(entry.sched), sched.vfloor
        r = warm_request_path!(entry)
        @test r.complete
        @test (pre_calls[], post_calls[]) == (1, 1)
        @test runs[] == runs_before                  # no program ran
        @test _wu_sched_state(entry.sched) == state
        @test sched.vfloor == vfloor
        @test exposed() == lines                     # no request, dispatch or compute counter moved
        # The model serves as before.
        out = infer(sched, InferRequest("wu", ["y"], [NamedTensor("x", Float32[1 2; 3 4])]))
        @test out[1].data == Float32[10 20; 300 400]
        @test (pre_calls[], post_calls[]) == (2, 2)
    finally
        ReactantServer.shutdown!(sched)
    end
end

@testset "warm-up: a hook that fails or hangs on the synthesized request does not stop the model" begin
    # preprocess rejects the zero request: postprocess is not run without it (the request path never
    # does that), the stages that do not depend on it still run, and the failure is logged, not thrown.
    post_calls = Threads.Atomic{Int}(0)
    entry, runs = _wu_mock(
        "wu_reject";
        pre = ts -> (all(iszero, ts[1].data) && throw(ReactantServer.InvalidInputError("all zeros")); ts),
        post = ts -> (Threads.atomic_add!(post_calls, 1); ts)
    )
    logger = TestLogger()
    r = with_logger(() -> warm_request_path!(entry), logger)
    @test !r.complete
    @test post_calls[] == 0 && runs[] == 0
    @test any(l -> l.level == Base.CoreLogging.Info && occursin("preprocess failed", l.message), logger.logs)
    sched = _wu_mock_sched(entry)
    ReactantServer.start!(sched)
    try
        out = infer(sched, InferRequest("wu_reject", ["y"], [NamedTensor("x", reshape(Float32[1, 2], 2, 1))]))
        @test out[1].data == reshape(Float32[10, 200], 2, 1)
    finally
        ReactantServer.shutdown!(sched)
    end

    # A hook that does not return in time: the caller waits only `timeout`, and the warm-up finishes
    # in the background (waited for here, so it cannot compile during a later measurement).
    post_calls = Threads.Atomic{Int}(0)
    entry, _ = _wu_mock(
        "wu_hang"; pre = ts -> (sleep(1.5); ts), post = ts -> (Threads.atomic_add!(post_calls, 1); ts)
    )
    logger = TestLogger()
    t = @elapsed r = with_logger(() -> warm_request_path!(entry; timeout = 0.2), logger)
    @test !r.complete
    @test t < 1.2
    @test any(l -> l.level == Base.CoreLogging.Warn && occursin("not warmed within", l.message), logger.logs)
    @test timedwait(() -> post_calls[] == 1, 30.0) === :ok
end

# A word tagger with host code on both sides, the shape of a text model: preprocess splits each text
# into words and lays out word ids per row; postprocess turns each word's scores into a tag and its
# probability, as CSR over the batch. The program scores each id against three tags (f32 rank 3),
# compiled at batch size 2 only, so a one-text request is padded on the dispatch loop. Its client
# tensors are u8 rank 2 and i32/f32/i64 rank 1, types the package image has no codec code for.
const _WU_TEXT_MANIFEST = """
format_version: "2.0"
name: wu_text
client_inputs:
  - {name: texts, dtype: u8, shape: cn, dims: {c: -1}}
  - {name: text_lens, dtype: i32, shape: n, dims: {}}
executable_inputs:
  - {name: ids, dtype: i64, shape: sn, dims: {s: 8}}
executable_outputs:
  - {name: scores, dtype: f32, shape: ksn, dims: {k: 3, s: 8}}
client_outputs:
  - {name: starts, dtype: i32, shape: m, dims: {m: -1}}
  - {name: labels, dtype: i32, shape: m, dims: {m: -1}}
  - {name: probs, dtype: f32, shape: m, dims: {m: -1}}
  - {name: row_offsets, dtype: i64, shape: r, dims: {r: -1}}
batching: {compiled_batch_sizes: [2]}
"""

# scores[n, s, k] = ids[n, s] * w[k], row-major.
const _WU_TEXT_MLIR = """
module {
  func.func @main(%x: tensor<2x8xi64>, %w: tensor<3xf32>) -> tensor<2x8x3xf32> {
    %0 = stablehlo.convert %x : (tensor<2x8xi64>) -> tensor<2x8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<2x8xf32>) -> tensor<2x8x3xf32>
    %2 = stablehlo.broadcast_in_dim %w, dims = [2] : (tensor<3xf32>) -> tensor<2x8x3xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<2x8x3xf32>
    return %3 : tensor<2x8x3xf32>
  }
}
"""

const _WU_TEXT_JL = raw"""
const S = 8
const VOCAB = Dict{String, Int64}(w => i for (i, w) in enumerate(split("the a of and to in is was for on with as by at from", ' ')))
const WORD = r"\p{L}[\p{L}\p{N}'-]*|\p{N}+"
const CTX = :wu_text_request
const CALLS = Threads.Atomic{Int}(0)    # preprocess calls, read by the test

word_id(w::AbstractString) = get(VOCAB, lowercase(w), Int64(100 + hash(w) % 900))

function words(text::String)
    spans = UnitRange{Int}[]
    for m in eachmatch(WORD, text)
        push!(spans, m.offset:(m.offset + ncodeunits(m.match) - 1))
    end
    return spans
end

function preprocess(inputs)
    Threads.atomic_add!(CALLS, 1)
    by = Dict(t.name => t.data for t in inputs)
    raw = by["texts"]::Matrix{UInt8}
    lens = by["text_lens"]::Vector{Int32}
    texts = String[String(raw[1:lens[b], b]) for b in axes(raw, 2)]
    spans = Vector{UnitRange{Int}}[words(t) for t in texts]
    ids = zeros(Int64, S, length(texts))
    for b in eachindex(texts), (j, r) in enumerate(Iterators.take(spans[b], S))
        ids[j, b] = word_id(SubString(texts[b], first(r), last(r)))
    end
    task_local_storage(CTX, (texts, spans))
    return [ReactantServer.NamedTensor("ids", ids)]
end

function postprocess(outputs)
    texts, spans = pop!(task_local_storage(), CTX)::Tuple{Vector{String}, Vector{Vector{UnitRange{Int}}}}
    scores = only(outputs).data::Array{Float32, 3}      # (tags, S, rows)
    starts, labels, probs = Int32[], Int32[], Float32[]
    offsets = Int64[0]
    for (b, sp) in enumerate(spans)
        for j in 1:min(S, length(sp))
            col = view(scores, :, j, b)
            m = maximum(col)
            z = sum(x -> exp(x - m), col)
            p, k = findmax(col)
            push!(starts, Int32(length(SubString(texts[b], 1, prevind(texts[b], first(sp[j]))))))
            push!(labels, Int32(k))
            push!(probs, exp(p - m) / z)
        end
        push!(offsets, length(starts))
    end
    NT = ReactantServer.NamedTensor
    return [NT("starts", starts), NT("labels", labels), NT("probs", probs), NT("row_offsets", offsets)]
end

register_model("wu_text"; preprocess, postprocess)
"""

function _write_wu_text_bundle(root)
    dir = joinpath(root, "wu_text")
    mkpath(dir)
    write(joinpath(dir, "manifest.yaml"), _WU_TEXT_MANIFEST)
    write(joinpath(dir, "model.b2.mlir"), stablehlo_artifact(_WU_TEXT_MLIR))
    SafeTensors.serialize(
        joinpath(dir, "weights.safetensors"), Dict("w" => Float32[1, -1, 0.5]),
        Dict("argument_order" => JSON3.write(["w"]))
    )
    write(joinpath(dir, "model.jl"), _WU_TEXT_JL)
    return dir
end

const _WU_TEXT = "The patient Maria Garcia, 52, was seen at 10 Elm Street on 3 May"

function _wu_text_inputs(text)
    bytes = Vector{UInt8}(codeunits(text))
    return [NamedTensor("texts", reshape(bytes, :, 1)), NamedTensor("text_lens", Int32[length(bytes)])]
end

# One ModelInfer, raw bytes in and out, through the handler the gRPC server registers for it, off the
# interactive thread as the server runs it.
_wu_request(ctx, body) = ReactantServer.off_interactive(() -> ReactantServer._handle_infer_raw(ctx, body, 0))
_wu_request_body(model) = ReactantServer._encode_message(
    ReactantServer.encode_infer_request(ReactantServer.inference, model, _wu_text_inputs(_WU_TEXT))
)

# Run `f` and return the seconds Julia spent compiling meanwhile, in any task.
function _wu_compile_seconds(f)
    c0 = first(Base.cumulative_compile_time_ns())
    f()
    return (first(Base.cumulative_compile_time_ns()) - c0) / 1.0e9
end

# Wait until nothing has compiled for `quiet` seconds (at most `limit`), so that a measurement right
# after sees its own compiles rather than another task's: the interactive monitor's first ticks after
# `serve`, or anything an earlier test left running.
function _wu_await_quiet(; quiet = 0.5, limit = 20.0)
    t_end = monotonic_time() + limit
    last = first(Base.cumulative_compile_time_ns())
    since = monotonic_time()
    while monotonic_time() < t_end
        sleep(0.05)
        c = first(Base.cumulative_compile_time_ns())
        c == last || ((last, since) = (c, monotonic_time()))
        monotonic_time() - since >= quiet && return true
    end
    return false
end

# The hook code the first request would compile without the warm-up, measured on a fresh copy of the
# bundle (its model.jl in a new module, so its own methods are uncompiled): preprocess and postprocess
# on the same text, on a default-pool task as a request runs them.
function _wu_cold_hook_seconds(dir, text)
    fresh = load_bundle_entry(dir)
    try
        return _wu_compile_seconds() do
            fetch(
                Threads.@spawn :default begin
                    prepared = Base.invokelatest(fresh.preprocess, _wu_text_inputs(text))
                    rows = size(prepared[1].data, 2)
                    Base.invokelatest(fresh.postprocess, [NamedTensor("scores", zeros(Float32, 3, 8, rows))])
                end
            )
        end
    finally
        ReactantServer.close_weights!(fresh.weights)
    end
end

_wu_calls(entry) = getfield(parentmodule(entry.preprocess), :CALLS)[]

# Robustness: the warm-up's effect is compared with what the same hooks compile on a fresh copy of the
# bundle, measured in the same process right after, so the bound scales with the machine and its load
# instead of being an absolute time. The fresh copy's own methods are new to Julia but the Base code
# they call is already compiled by then, so it is a lower bound on what an unwarmed first request
# compiles; the warmed request must compile under a quarter of it. The compile counter is global, so
# only the request is inside the measurement: the test's calling code is compiled beforehand, and the
# request waits until nothing else has compiled for half a second (the interactive monitor compiles
# its first tick just after `serve` returns). Which hook calls happened is checked exactly, with a
# counter in the bundle.
@testset "warm-up: the first request to a model with host code compiles little (CPU)" begin
    Base.cumulative_compile_timing(true)
    mktempdir() do root
        dir = _write_wu_text_bundle(root)
        cfg = ReactantServer.ServerConfig(
            [root], "",
            ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", grpc_free_port())
        )
        srv = ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend(), blocking = false)
        try
            entry = srv.registry.by_name["wu_text"]
            @test _wu_calls(entry) == 1                       # warmed at startup, once
            @test entry.sched.dispatch_count == 0 && entry.sched.requests_served == 0
            ctx = srv.server.context
            # Compile the test's own calling code first, with a request for a model the worker does
            # not serve: it fails before any model's code runs.
            try
                _wu_request(ctx, _wu_request_body("wu_absent"))
                @test false
            catch err
                @test err isa ReactantServer.gRPCServer.GRPCError
            end
            body = _wu_request_body("wu_text")
            _wu_await_quiet()
            c0 = first(Base.cumulative_compile_time_ns())
            resp = _wu_request(ctx, body)
            warm = (first(Base.cumulative_compile_time_ns()) - c0) / 1.0e9
            msg = ReactantServer.ProtoBuf.decode(
                ReactantServer.ProtoBuf.ProtoDecoder(IOBuffer(resp)), ReactantServer.inference.ModelInferResponse
            )
            labels = collect(reinterpret(Int32, msg.raw_output_contents[2]))
            @test length(labels) == 8                         # a tag for each of the first S = 8 words
            @test collect(reinterpret(Int64, msg.raw_output_contents[4])) == [0, 8]
            @test _wu_calls(entry) == 2
            cold = _wu_cold_hook_seconds(dir, _WU_TEXT)
            @info "warm-up test: compile on the first request" warm cold
            @test cold > 0.05                                 # the fixture compiles enough to compare
            @test warm < cold / 4
        finally
            ReactantServer.stop!(srv)
        end
    end
end

@testset "warm-up: a hot load warms the new version before it is admitted (CPU)" begin
    mktempdir() do root
        dir = _write_wu_text_bundle(root)
        backend = ReactantServer.ReactantBackend()
        runtime = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        pool = ReactantServer.resolve_client(backend, runtime)
        sched = Scheduler(ModelRegistry(), backend, pool, SchedulerConfig(30.0, 64, 30.0))
        ReactantServer.start!(sched)
        try
            entry = load_bundle_entry(dir)
            @test _wu_calls(entry) == 0
            @test load_model!(
                sched, backend, pool, entry; state = ReactantServer.UNPINNED, on_demand = false
            ) == "wu_text"
            @test _wu_calls(entry) == 1                       # warmed by the load, before any request
            @test entry.sched.dispatch_count == 0 && entry.sched.requests_served == 0
            out = infer(sched, InferRequest("wu_text", String[], _wu_text_inputs(_WU_TEXT)))
            @test [t.name for t in out] == ["starts", "labels", "probs", "row_offsets"]
            @test _wu_calls(entry) == 2
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end
