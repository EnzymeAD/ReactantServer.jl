# Shared-memory client for the CPU supervisor e2e (run_e2e_cpu.sh), through the embedded gateway.
#
# Phase A (CORE-3): many concurrent single-item requests with shared-memory outputs and deadlines
# short enough that clients give up while their requests are still queued or running, so pool
# slots are reused under late writers. Every request that succeeds must carry exactly 2x its own
# input (scale4 multiplies by weights of 2).
#
# Phase B (CORE-2): after the script restarts one worker (it waits for <marker dir>/restarted), the
# same client process, with the same region name and key, keeps working: the restarted worker
# answers FAILED_PRECONDITION, the client re-registers through the gateway, and the workers that
# kept the registration keep it. Every request also answers promptly, although phase A gave up on
# about a thousand requests over this client's connection to the gateway (GW-18).
#
#   julia --project=packages/ReactantServerClient client_shm_cpu.jl <gateway_port> <worker0_port> <marker_dir>

using ReactantServerClient
using ReactantServerClient: inference
import ReactantServerCore

const GATEWAY_PORT = parse(Int, ARGS[1])
const WORKER0_PORT = parse(Int, ARGS[2])
const MARKERS = ARGS[3]

const FAILURES = String[]
check(name, cond; detail = "") = cond ? println("PASS  ", name) : (println("FAIL  ", name, "  ", detail); push!(FAILURES, name))

struct OneIO <: AbstractInferenceIO
    x::Vector{Float32}
    y::Vector{Float32}
end
OneIO(x) = OneIO(x, zeros(Float32, 4))
Base.length(::OneIO) = 1
ReactantServerClient.item_input_bytes(::OneIO) = 4 * sizeof(Float32)
ReactantServerClient.output_specs(::OneIO) = [OutputSpec("y", Float32, [4])]   # 16 bytes per item
function ReactantServerClient.infer_encode_chunk!(io::OneIO, r, slot)
    x = scratch(slot, "x", (4,), Float32)       # scale4 has no batch axis: one item, shape [4]
    pool_view(x) .= io.x
    return x
end
ReactantServerClient.infer_decode_chunk!(io::OneIO, r, response) = (io.y .= vec(InferOutput("y", response, Float32)); nothing)

model(deadline; retry = false) = KServeModel(
    "127.0.0.1", GATEWAY_PORT, "scale4"; shared_memory = :on, max_batch_size = 1, deadline = deadline,
    retry = retry ? ReactantServerClient.RetryPolicy() : ReactantServerClient.RetryPolicy(enabled = false)
)

function status_on_worker0(name)
    resp = inference.GRPCInferenceService.SystemSharedMemoryStatus(
        "127.0.0.1", WORKER0_PORT, inference.SystemSharedMemoryStatusRequest(; name = name); deadline = 10
    )
    r = get(resp.regions, name, nothing)
    return r === nothing ? nothing : (r.key, r.byte_size)
end

kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
slow = model(60.0; retry = true)
warm = OneIO(Float32[1, 2, 3, 4])
infer_sync(slow, warm)
check("warm-up through shared memory", warm.y == Float32[2, 4, 6, 8]; detail = "got $(warm.y)")
pool = ReactantServerClient.get_or_create_pool!(slow)
check("the gateway route uses the shared-memory pool", ReactantServerClient.is_shm_backed(pool))
name = ReactantServerClient.pool_name(pool)

# ---- phase A: give-ups and slot reuse ----
ok = Threads.Atomic{Int}(0)
gaveup = Threads.Atomic{Int}(0)
wrong = Threads.Atomic{Int}(0)
models = [model(d) for d in (0.001, 0.002, 0.004, 0.008, 0.016, 0.5)]
tasks = map(1:8) do task
    Threads.@spawn for k in 1:150
        x = Float32[task, k, task * k, rand(Float32)]
        io = OneIO(x)
        try
            infer_sync(models[1 + (task + k) % length(models)], io)
            Threads.atomic_add!(io.y == 2 .* x ? ok : wrong, 1)
        catch
            Threads.atomic_add!(gaveup, 1)
        end
    end
end
foreach(wait, tasks)
println("phase A: ok=$(ok[]) gave_up=$(gaveup[]) wrong=$(wrong[]) inline_fallbacks=$(ReactantServerClient.inline_fallback_count())")
check("phase A: no request read another request's outputs", wrong[] == 0; detail = "$(wrong[]) wrong")
check("phase A: requests succeeded", ok[] > 0)
check("phase A: some requests were given up", gaveup[] > 0)
check("phase A: the client still uses shared memory", ReactantServerClient.is_shm_backed(ReactantServerClient.get_or_create_pool!(slow)))
before = status_on_worker0(name)
check("worker0 holds the registration", before !== nothing; detail = "status $(before)")

# ---- phase B: one worker restarts under the same client ----
touch(joinpath(MARKERS, "phase_a_done"))
t0 = time()
while !isfile(joinpath(MARKERS, "restarted"))
    time() - t0 > 1800 && (println("FAIL  restart marker never appeared"); exit(1))
    sleep(1)
end
# Each request must answer within PROMPT_SECONDS, not merely within its 60 s deadline. The slowest
# one normally takes about 1.3 s, even with two of these tests running at once: the first to reach
# the restarted worker, which answers FAILED_PRECONDITION, so the client re-registers the region
# through the gateway and retries. 5 s leaves room for a slower or busier host. A request that
# instead waits out its deadline is GW-18: the gateway cut this client's connection off for phase
# A's give-ups, and libcurl parked the next requests behind the replacement connection.
const PROMPT_SECONDS = 5.0
seconds = Float64[]
results = map(1:20) do k
    io = OneIO(Float32[k, -k, 2k, 0.5])
    t = time()
    ok = try
        infer_sync(slow, io)
        io.y == 2 .* io.x
    catch e
        println("phase B request $k failed: ", sprint(showerror, e))
        false
    end
    push!(seconds, time() - t)
    println("phase B request $k: ", ok ? "ok" : "failed", " in $(round(seconds[end]; digits = 3)) s")
    ok
end
check("phase B: every request after the restart succeeds with its own outputs", all(results); detail = "$(count(results))/20")
check(
    "phase B: every request after the restart answers within $(PROMPT_SECONDS) s", maximum(seconds) <= PROMPT_SECONDS;
    detail = "slowest $(round(maximum(seconds); digits = 1)) s"
)
check("phase B: still on shared memory, same region", ReactantServerClient.pool_name(ReactantServerClient.get_or_create_pool!(slow)) == name)
after = status_on_worker0(name)
check("phase B: worker0 kept the same registration", after == before; detail = "before $(before), after $(after)")

kserve_shutdown()
if isempty(FAILURES)
    println("CPU e2e shared-memory client: all checks passed")
else
    println("CPU e2e shared-memory client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
