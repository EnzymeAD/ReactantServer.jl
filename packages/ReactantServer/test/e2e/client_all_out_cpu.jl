# Shared-memory client for the CPU supervisor e2e (run_e2e_cpu.sh, step 11), through the embedded
# gateway of a node on the lpt_packing scheduler: every worker goes out at once, and the node
# recovers.
#
# Each phase starts from a baseline: every model serves through shared memory, and both workers hold
# the client's region. The script then takes every worker out and brings them back (phase A: stopped
# with SIGSTOP, then resumed together; phase B: stopped, then worker0 resumed alone and the rest
# later; phase C: killed with SIGKILL and restarted by the supervisor), and signals each step
# through marker files. While every worker is marked down, the gateway answers a request
# UNAVAILABLE itself, well under the request's deadline (phases A and C). While worker0 alone is
# back, every model serves from it promptly (phase B). Once the workers are back, every request
# succeeds, and the client is on shared memory again under the same region, which both workers
# hold: a restarted worker lost its registration, so it holds the region only because the client
# registered it again after that worker answered FAILED_PRECONDITION. If the client latched to
# inline transport meanwhile, its background re-probe (every second here) has to bring it back.
#
#   julia --project=packages/ReactantServerClient client_all_out_cpu.jl <gateway_port> <worker0_port> <worker1_port> <marker_dir>

using ReactantServerClient
using ReactantServerClient: inference
import gRPCClient

const GATEWAY_PORT = parse(Int, ARGS[1])
const WORKER_PORTS = (parse(Int, ARGS[2]), parse(Int, ARGS[3]))
const MARKERS = ARGS[4]
const MODELS = ("scale4", "slow4")   # the node's models, each placed on both workers

# A request that is answered at once (served, or UNAVAILABLE from the gateway) takes milliseconds;
# its deadline is 30 s. One that waits for a stopped worker takes the whole deadline.
const DEADLINE = 30.0
const PROMPT_SECONDS = 5.0
# After the workers are back: at least this many rounds of one request per model, and at most this
# long to see a round go through shared memory with both workers holding the region.
const MIN_ROUNDS = 5
const RECOVERY_SECONDS = 120.0

# Every line is flushed at once: the script prints a phase's lines as soon as the phase ends.
say(xs...) = (println(xs...); flush(stdout))
note(label, msg) = say("INFO  ", label, ": ", msg)
const FAILURES = String[]
check(name, cond; detail = "") = cond ? say("PASS  ", name) : (say("FAIL  ", name, "  ", detail); push!(FAILURES, name))

struct OneIO <: AbstractInferenceIO
    x::Vector{Float32}
    y::Vector{Float32}
end
OneIO(x) = OneIO(x, zeros(Float32, 4))
Base.length(::OneIO) = 1
ReactantServerClient.item_input_bytes(::OneIO) = 4 * sizeof(Float32)
ReactantServerClient.output_specs(::OneIO) = [OutputSpec("y", Float32, [4])]
function ReactantServerClient.infer_encode_chunk!(io::OneIO, r, slot)
    x = scratch(slot, "x", (4,), Float32)       # no batch axis: one item, shape [4]
    pool_view(x) .= io.x
    return x
end
ReactantServerClient.infer_decode_chunk!(io::OneIO, r, response) = (io.y .= vec(InferOutput("y", response, Float32)); nothing)

model(name; retry = true) = KServeModel(
    "127.0.0.1", GATEWAY_PORT, name; shared_memory = :on, max_batch_size = 1, deadline = DEADLINE,
    retry = retry ? ReactantServerClient.RetryPolicy() : ReactantServerClient.RetryPolicy(enabled = false)
)

# The (key, byte_size) a worker has registered under `name`, or nothing (also when it cannot say).
function held_on(port, name; label = "")
    return try
        resp = inference.GRPCInferenceService.SystemSharedMemoryStatus(
            "127.0.0.1", port, inference.SystemSharedMemoryStatusRequest(; name = name); deadline = 10
        )
        r = get(resp.regions, name, nothing)
        r === nothing ? nothing : (r.key, r.byte_size)
    catch e
        note(label, "SystemSharedMemoryStatus on port $port failed: $(sprint(showerror, e))")
        nothing
    end
end

mark(f) = (flush(stdout); touch(joinpath(MARKERS, f)))
function wait_marker(f)
    t0 = time()
    while !isfile(joinpath(MARKERS, f))
        time() - t0 > 1800 && (say("FAIL  marker $f never appeared"); exit(1))
        sleep(0.05)
    end
    return nothing
end

# The gRPC status a failed request carries, wherever the driver wrapped it; nothing if none.
function grpc_status(e)
    e isa gRPCClient.gRPCServiceCallException && return e.grpc_status
    e isa TaskFailedException && return grpc_status(e.task.exception)
    e isa CapturedException && return grpc_status(e.ex)
    e isa CompositeException && return isempty(e.exceptions) ? nothing : grpc_status(first(e.exceptions))
    return nothing
end

# One request; whether it succeeded with its own outputs, and how long it took. x[1] stays 0: slow4
# sleeps x[1] seconds in its preprocess, and scale4 fails on x[1] == -1 (run_e2e_cpu.sh).
function served(m, k; label)
    io = OneIO(Float32[0, k, -k, 0.5])
    t = time()
    ok = try
        infer_sync(m, io)
        io.y == 2 .* io.x
    catch e
        note(label, "request $k for $(m.model_name) failed: $(sprint(showerror, e))")
        false
    end
    return ok, time() - t
end

on_shm(m) = ReactantServerClient.is_shm_backed(ReactantServerClient.get_or_create_pool!(m))
region() = ReactantServerClient.pool_name(ReactantServerClient.get_or_create_pool!(SERVING[1]))

kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 1.0)
const SERVING = [model(n) for n in MODELS]
const NO_RETRY = [model(n; retry = false) for n in MODELS]

# Every model serves through shared memory and both workers hold the region (the first call also
# registers it). Returns the region name.
function baseline(label)
    for (name, m) in zip(MODELS, SERVING)
        ok, _ = served(m, 1; label = "$label baseline")
        check("$label baseline: $name serves", ok)
        check("$label baseline: $name is on shared memory", on_shm(m))
    end
    name = region()
    holders = [held_on(p, name; label = "$label baseline") !== nothing for p in WORKER_PORTS]
    check("$label baseline: both workers hold the region", all(holders); detail = "$holders")
    return name
end

# Every worker is marked down: the gateway answers each model's request UNAVAILABLE without calling
# a worker. No retry, so the client reports the first answer.
function held(label)
    for (name, m) in zip(MODELS, NO_RETRY)
        io = OneIO(Float32[0, 1, -1, 0.5])
        t = time()
        status = try
            infer_sync(m, io)
            :served
        catch e
            note(label, "$name was answered: $(sprint(showerror, e))")
            grpc_status(e)
        end
        dt = time() - t
        check(
            "$label: $name is answered UNAVAILABLE within $(PROMPT_SECONDS) s (its deadline is $(DEADLINE) s)",
            status == gRPCClient.GRPC_UNAVAILABLE && dt <= PROMPT_SECONDS;
            detail = "status $status after $(round(dt; digits = 2)) s"
        )
    end
    return nothing
end

# worker0 alone is back (the others are stopped): every model serves, each request promptly, so none
# waited on a stopped worker.
function alone(label, n = 4)
    for (name, m) in zip(MODELS, SERVING)
        runs = [served(m, k; label) for k in 1:n]
        check(
            "$label: $name serves from worker0 alone, every request within $(PROMPT_SECONDS) s",
            all(first, runs) && maximum(last, runs) <= PROMPT_SECONDS;
            detail = "$(count(first, runs))/$n served, slowest $(round(maximum(last, runs); digits = 2)) s"
        )
    end
    check("$label: on shared memory", all(on_shm, SERVING))
    return nothing
end

# The workers are back: send one request per model per round until a round has gone through shared
# memory with both workers holding the region, at least MIN_ROUNDS rounds and for at most
# RECOVERY_SECONDS. Every request must succeed. A route that was on shared memory before a request
# and inline after it was latched by that request (and is unlatched by the background re-probe).
function recovered(label, name)
    results = Bool[]
    latches = 0
    rounds = 0
    holders = Bool[]
    t0 = time()
    while true
        rounds += 1
        round_shm = true
        for m in SERVING
            before = on_shm(m)
            ok, _ = served(m, rounds; label)
            push!(results, ok)
            after = on_shm(m)
            (before && !after) && (latches += 1)
            round_shm &= before && after
        end
        holders = [held_on(p, name; label) !== nothing for p in WORKER_PORTS]
        rounds >= MIN_ROUNDS && round_shm && all(holders) && break
        time() - t0 > RECOVERY_SECONDS && break
        sleep(0.2)
    end
    secs = round(time() - t0; digits = 1)
    note(label, "$(length(results)) requests in $rounds rounds over $secs s; latched to inline $latches time(s)")
    check("$label: every request after the workers returned succeeds", all(results); detail = "$(count(results))/$(length(results))")
    check("$label: every model is on shared memory again, same region", all(on_shm, SERVING) && region() == name)
    check("$label: both workers hold the region", all(holders); detail = "$holders")
    return nothing
end

const REGION = baseline("phase A")
mark("baseA")
wait_marker("downA")
held("phase A (every worker stopped)")
mark("checkedA")
wait_marker("backA")
recovered("phase A (every worker resumed together)", REGION)
mark("doneA")

check("phase B baseline: same region", baseline("phase B") == REGION)
mark("baseB")
wait_marker("oneB")
alone("phase B (worker0 resumed, worker1 still stopped)")
mark("servedB")
wait_marker("backB")
recovered("phase B (worker1 resumed too)", REGION)
mark("doneB")

check("phase C baseline: same region", baseline("phase C") == REGION)
mark("baseC")
wait_marker("downC")
held("phase C (every worker killed)")
mark("checkedC")
wait_marker("backC")
recovered("phase C (every worker restarted)", REGION)
mark("doneC")

kserve_shutdown()
if isempty(FAILURES)
    say("CPU e2e all-workers-out client: all checks passed")
else
    say("CPU e2e all-workers-out client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
