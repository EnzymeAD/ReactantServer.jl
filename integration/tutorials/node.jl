# # Run a Node
#
# A *node* is one machine running ReactantServer. The node supervisor, `ReactantServerNode`, runs a
# whole node from one node file: it starts one worker process per device, restarts workers that
# die, and when there are two or more workers it puts an embedded gateway in front of them, so
# clients always talk to a single endpoint. The container image runs exactly this supervisor.
#
# This page runs a node with two CPU workers, which takes the same shape as a two-GPU machine: two
# workers behind the embedded gateway.
#
# ## Install
#
# Add the supervisor together with the worker and gateway packages, plus the client and (for the
# setup step) the export tooling and Lux:
#
# ```julia
# using Pkg
# Pkg.add(["ReactantServerNode", "ReactantServer", "ReactantServerGateway",
#          "ReactantServerClient", "ReactantServerExport", "Lux"])
# ```
#
# The supervisor itself never loads Reactant. Each worker is a separate `julia` process that loads
# `ReactantServer`, and the gateway one that loads `ReactantServerGateway`; both run in the
# supervisor's own environment, which is why all three packages go in the same one. Pkg keeps
# their versions compatible with each other.
#
# ## Setup: a bundle
#
# Export the MLP from [Export a Model to a Bundle](export.md) into `models/mlp`:

using Lux, Random, ReactantServerExport
using Test #src

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
ps, st = Lux.setup(Random.Xoshiro(0), model)
export_bundle(
    :lux, model, ps, st, randn(Float32, 4, 1);
    dir = joinpath("models", "mlp"), name = "mlp", batch_sizes = [1, 8]
)

# ## Describe the node
#
# Without a `workers:` list, the supervisor creates one worker per detected GPU. Worker `i` serves
# gRPC on `base_port + i` and metrics on `metrics_base_port + i`; the gateway takes the public
# ports, 8001 for gRPC and 8002 for health and metrics.

write(
    "node.yaml", """
    model_repo: $(abspath("models"))
    base_port: 8080
    metrics_base_port: 9100
    global:
      runtime:
        backend: cpu           # cuda on a GPU host
      endpoints:
        host: 127.0.0.1
    """
)

# On a GPU host, that file already scales: the supervisor runs one worker per visible GPU. Without
# GPUs, two environment variables ask for a CPU node with two workers: `REACTANT_GPUS=0` (a CPU
# node) and `REACTANT_CPU_WORKERS=2`.
#
# ## Start the node
#
# The supervisor is a long-running process of its own, like the container it runs in. From a shell:
#
# ```sh
# REACTANT_GPUS=0 REACTANT_CPU_WORKERS=2 \
#   julia --project --handle-signals=no -e 'using ReactantServerNode; exit(supervise("node.yaml"))'
# ```
#
# `--handle-signals=no` lets the supervisor handle SIGTERM and Ctrl-C itself, so it stops its
# workers before exiting. To keep this page one runnable script, it starts the same command from
# Julia, in this environment, with its output shown in this session (`run` without `wait` would
# otherwise discard it):

supervise_cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --handle-signals=no
    -e 'using ReactantServerNode; exit(supervise("node.yaml"))'`
supervise_cmd = addenv(supervise_cmd, "REACTANT_GPUS" => "0", "REACTANT_CPU_WORKERS" => "2")
node = run(pipeline(supervise_cmd; stdout, stderr); wait = false)
atexit(() -> process_running(node) && kill(node)) #src

# Every line the workers and the gateway print comes out on the supervisor's output with a `[worker0]`,
# `[worker1]`, or `[gateway]` prefix. Each worker compiles its models before it reports ready, so
# wait for `/readyz` to answer 200 on the gateway and on each worker:

using Downloads

function wait_ready(url; timeout = 1800)
    deadline = time() + timeout
    while time() < deadline
        process_exited(node) && error("the node exited before $url was ready")
        r = Downloads.request(url; throw = false)
        r isa Downloads.Response && r.status == 200 && return true
        sleep(2)
    end
    return error("$url was not ready after $timeout seconds")
end

wait_ready("http://127.0.0.1:9100/readyz")      # worker0
wait_ready("http://127.0.0.1:9101/readyz")      # worker1
wait_ready("http://127.0.0.1:8002/readyz")      # the gateway

# ## Query it
#
# Clients use the gateway's port, 8001, whether the node has one worker or eight. Both workers
# serve `mlp`, so the gateway spreads its requests across them. The first request the gateway and
# a worker handle also compiles their request paths, which can take longer than the client's default
# 10 second deadline, so this allows a minute:

using ReactantServerClient

kserve_init()
mlp = KServeModel("grpc://127.0.0.1:8001", "mlp"; max_batch_size = 8, deadline = 60)
x = randn(Random.Xoshiro(1), Float32, 4, 8)
y = InferOutput("output", infer_sync(mlp, [InferInput("input", x)]), Float32)
maximum(abs, y .- first(model(x, ps, st)))      # ≈ 0

@test y ≈ first(model(x, ps, st)) #src

# One scrape of the gateway's `/metrics` covers the whole node: the gateway's own series plus every
# worker's, each tagged with a `worker` label.

metrics = String(take!(Downloads.download("http://127.0.0.1:8002/metrics", IOBuffer())))
occursin("worker=\"worker1\"", metrics)          # true

@test occursin("worker=\"worker0\"", metrics) #src
@test occursin("worker=\"worker1\"", metrics) #src

# ## Stop
#
# SIGTERM, which is what `kill` sends and what a container runtime sends on stop, makes the
# supervisor drain its workers and the gateway, then exit with status 0:

kserve_shutdown()
kill(node)
wait(node)
node.exitcode                                    # 0

@test node.exitcode == 0 #src

# [Deployment](../../deployment.md) describes how the supervisor detects devices, sizes each worker's
# thread pool, and where its children load from; [Node Configuration](../../node_config.md) covers
# every key of the node file; and [Multi-GPU Gateway](../../gateway.md) covers the gateway's scheduling
# modes.
