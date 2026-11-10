# Shared-memory client for the CPU supervisor e2e (run_e2e_cpu.sh, step 8), through the embedded
# gateway: a client that registers its region while a worker is out.
#
# The gateway registers a region on the workers it considers healthy only. A worker that was out
# during the register does not know the region when it returns, so the first request it gets that
# uses the region fails there with FAILED_PRECONDITION; the client registers the same region again
# through the gateway (a no-op on the worker that holds it) and retries once. The caller sees no
# error, and the client stays on shared memory under the same region name.
#
# Phase 1: worker1 is stopped (SIGSTOP), so the gateway marks it down, but it stays alive; the
# client starts a fresh pool and registers it then; worker1 resumes (SIGCONT) and is ready again.
# Phase 2: the same with worker1 killed (SIGKILL) and restarted by the supervisor, which loses
# every registration. The script drives worker1 and signals each step through marker files.
#
#   julia --project=packages/ReactantServerClient client_shm_outage_cpu.jl <gateway_port> <worker0_port> <worker1_port> <marker_dir>

using ReactantServerClient
using ReactantServerClient: inference

const GATEWAY_PORT = parse(Int, ARGS[1])
const WORKER0_PORT = parse(Int, ARGS[2])
const WORKER1_PORT = parse(Int, ARGS[3])
const MARKERS = ARGS[4]

const FAILURES = String[]
check(name, cond; detail = "") = cond ? println("PASS  ", name) : (println("FAIL  ", name, "  ", detail); push!(FAILURES, name))

struct OneIO <: AbstractInferenceIO
    x::Vector{Float32}
    y::Vector{Float32}
end
OneIO(x) = OneIO(x, zeros(Float32, 4))
Base.length(::OneIO) = 1
ReactantServerClient.item_input_bytes(::OneIO) = 4 * sizeof(Float32)
ReactantServerClient.output_specs(::OneIO) = [OutputSpec("y", Float32, [4])]
function ReactantServerClient.infer_encode_chunk!(io::OneIO, r, slot)
    x = scratch(slot, "x", (4,), Float32)       # scale4 has no batch axis: one item, shape [4]
    pool_view(x) .= io.x
    return x
end
ReactantServerClient.infer_decode_chunk!(io::OneIO, r, response) = (io.y .= vec(InferOutput("y", response, Float32)); nothing)

model() = KServeModel(
    "127.0.0.1", GATEWAY_PORT, "scale4"; shared_memory = :on, max_batch_size = 1, deadline = 30.0,
    retry = ReactantServerClient.RetryPolicy()
)

# The (key, byte_size) a worker has registered under `name`, or nothing.
function held_on(port, name)
    resp = inference.GRPCInferenceService.SystemSharedMemoryStatus(
        "127.0.0.1", port, inference.SystemSharedMemoryStatusRequest(; name = name); deadline = 10
    )
    r = get(resp.regions, name, nothing)
    return r === nothing ? nothing : (r.key, r.byte_size)
end

mark(f) = touch(joinpath(MARKERS, f))
function wait_marker(f)
    t0 = time()
    while !isfile(joinpath(MARKERS, f))
        time() - t0 > 1800 && (println("FAIL  marker $f never appeared"); exit(1))
        sleep(0.05)
    end
    return nothing
end

# One request; whether it succeeded with its own outputs.
function served(m, k)
    io = OneIO(Float32[k, -k, 2k, 0.5])
    return try
        infer_sync(m, io)
        io.y == 2 .* io.x
    catch e
        println("request $k failed: ", sprint(showerror, e))
        false
    end
end

# Warm up the whole path (namespace probe, register, infer, shutdown, a second init) with both
# workers up, so the registration during the outage is quick.
for _ in 1:2
    kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
    check("warm-up through shared memory", served(model(), 1))
    kserve_shutdown()
end
mark("warm")

function phase(label, n)
    wait_marker("down$n")
    kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
    m = model()
    check("$label: registered and served while worker1 is out", served(m, 1))
    pool = ReactantServerClient.get_or_create_pool!(m)
    check("$label: on shared memory", ReactantServerClient.is_shm_backed(pool))
    name = ReactantServerClient.pool_name(pool)
    w0 = held_on(WORKER0_PORT, name)
    check("$label: worker0 holds the region", w0 !== nothing)
    mark("registered$n")
    wait_marker("back$n")
    # Round robin gives worker1 every other request once the gateway routes to it again, which can
    # be a probe round after it reports ready (it first drops the regions it missed while out). The
    # first request it gets meets FAILED_PRECONDITION there, and the client registers again and
    # retries without the caller seeing it. Send requests until worker1 holds the region, at least
    # ten and for at most a minute; every one must succeed.
    results = Bool[]
    reached = false
    t0 = time()
    while length(results) < 10 || (!reached && time() - t0 < 60)
        push!(results, served(m, length(results) + 1))
        reached = reached || held_on(WORKER1_PORT, name) !== nothing
        reached || sleep(0.2)
    end
    check("$label: every request after worker1 returns succeeds", all(results); detail = "$(count(results))/$(length(results))")
    p = ReactantServerClient.get_or_create_pool!(m)
    check(
        "$label: still on shared memory, same region",
        ReactantServerClient.is_shm_backed(p) && ReactantServerClient.pool_name(p) == name
    )
    check("$label: worker1 now holds the region, so a request reached it", reached; detail = "$(length(results)) requests over $(round(time() - t0; digits = 1)) s")
    check("$label: worker0 kept the same registration", held_on(WORKER0_PORT, name) == w0)
    kserve_shutdown()
    return mark("done$n")
end

phase("phase 1 (worker1 stopped, then resumed)", 1)
phase("phase 2 (worker1 killed, then restarted)", 2)

if isempty(FAILURES)
    println("CPU e2e shared-memory outage client: all checks passed")
else
    println("CPU e2e shared-memory outage client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
