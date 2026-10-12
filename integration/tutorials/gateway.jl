# # Front Workers with a Gateway
#
# `ReactantServerGateway` is a gRPC reverse proxy that puts several workers behind one KServe V2
# endpoint. The node supervisor already embeds one whenever a node has two or more workers
# ([Run a Node](node.md)); this page runs the gateway on its own, over workers you start yourself.
# That is the shape for workers the supervisor does not manage, such as workers on other machines.
#
# The gateway also decides where each model's traffic goes. This page uses `lpt_packing`, the mode
# that places every model on a chosen number of workers, so a model's requests concentrate on few
# devices and coalesce into full batches.
#
# ## Install
#
# ```julia
# using Pkg
# Pkg.add(["ReactantServerGateway", "ReactantServer", "ReactantServerClient",
#          "ReactantServerExport", "Lux"])
# ```
#
# The gateway itself does not load Reactant; only the workers do.
#
# ## Setup: two models
#
# Export two small MLPs from [Export a Model to a Bundle](../basics/export.md), differing only in their
# random initialization, into one model repository:

using Lux, Random, ReactantServerExport
using Test #src

model = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 3))
params = Dict{String, Any}()
for (name, seed) in ("mlp_a" => 1, "mlp_b" => 2)
    ps, st = Lux.setup(Random.Xoshiro(seed), model)
    params[name] = (ps, st)
    export_bundle(
        :lux, model, ps, st, randn(Float32, 4, 1);
        dir = joinpath("models", name), name = name, batch_sizes = [1, 8]
    )
end

# ## Start two workers
#
# One node file describes both workers. Worker `i` listens on `base_port + i`. `lpt_packing` moves
# scheduling decisions to the gateway, so the workers run the first-in, first-out `fifo` discipline
# instead of reordering requests themselves:

write(
    "node.yaml", """
    model_repo: $(abspath("models"))
    base_port: 8080
    metrics_base_port: 9100
    global:
      runtime: { backend: cpu }
      scheduler: { discipline: fifo }
      endpoints: { host: 127.0.0.1 }
    workers:
      - { name: worker0 }
      - { name: worker1 }
    """
)

# Each worker is its own process, started with `serve` and the worker's name. From two shells:
#
# ```sh
# julia --project --threads=auto,1 -e 'using ReactantServer; serve("node.yaml"; worker = "worker0")'
# julia --project --threads=auto,1 -e 'using ReactantServer; serve("node.yaml"; worker = "worker1")'
# ```
#
# To keep this page one runnable script, it starts the same commands from Julia, with their output
# shown in this session (`run` without `wait` would otherwise discard it):

function start(code; threads = "auto,1")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads -e $code`
    return run(pipeline(cmd; stdout, stderr); wait = false)
end

processes = [
    start("using ReactantServer; serve(\"node.yaml\"; worker = \"worker0\")"),
    start("using ReactantServer; serve(\"node.yaml\"; worker = \"worker1\")"),
]
atexit(() -> foreach(kill, processes)) #src

# Wait until both workers have compiled their models and answer `/readyz` on their metrics ports:

using Downloads

function wait_ready(url; timeout = 1800)
    deadline = time() + timeout
    while time() < deadline
        any(process_exited, processes) && error("a process exited before $url was ready")
        r = Downloads.request(url; throw = false)
        r isa Downloads.Response && r.status == 200 && return true
        sleep(2)
    end
    return error("$url was not ready after $timeout seconds")
end

wait_ready("http://127.0.0.1:9100/readyz")
wait_ready("http://127.0.0.1:9101/readyz")

# ## Configure the gateway
#
# A `gateway.yml` lists the gateway's own listen addresses and the worker endpoints. The gateway
# learns which models each worker serves by asking the workers, so it needs no model list.
# `metrics_endpoints` and `worker_names` are optional: with them, the gateway's `/metrics` also
# carries every worker's series, labeled by worker name.

write(
    "gateway.yml", """
    listen:
      grpc: 127.0.0.1:8001
      metrics: 127.0.0.1:8002
    endpoints: ["127.0.0.1:8080", "127.0.0.1:8081"]
    metrics_endpoints: ["127.0.0.1:9100", "127.0.0.1:9101"]
    worker_names: [worker0, worker1]
    scheduling:
      mode: lpt_packing
      default_replicas: 1      # place each model on one worker
    """
)

# `lpt_packing` requires every worker to serve the same models with a `fifo` or `edf` discipline,
# and checks both at startup. [Multi-GPU Gateway](../../gateway.md) documents the other modes,
# `round_robin` and `least_outstanding`, which have no such preconditions, and every scheduling
# setting.
#
# ## Start the gateway
#
# The gateway is a process of its own too. [`serve_gateway`](@ref) loads `gateway.yml` and serves
# until stopped; from a shell:
#
# ```sh
# julia --project --threads=auto -e 'using ReactantServerGateway; serve_gateway("gateway.yml")'
# ```

push!(processes, start("using ReactantServerGateway; serve_gateway(\"gateway.yml\")"; threads = "auto"))
wait_ready("http://127.0.0.1:8002/readyz")

# ## Query through it
#
# Clients address the gateway's port, and it routes each request by model name. The first request
# a freshly started gateway or worker handles also compiles its request path, which can take longer
# than the client's default 10 second deadline, so this allows a minute:

using ReactantServerClient

kserve_init()
models = Dict(name => KServeModel("grpc://127.0.0.1:8001", name; deadline = 60) for name in ("mlp_a", "mlp_b"))
x = randn(Random.Xoshiro(3), Float32, 4, 8)
y_a = InferOutput("output", infer_sync(models["mlp_a"], [InferInput("input", x)]), Float32)
y_b = InferOutput("output", infer_sync(models["mlp_b"], [InferInput("input", x)]), Float32)
y_a ≈ first(model(x, params["mlp_a"]...))                    # true

@test y_a ≈ first(model(x, params["mlp_a"]...)) #src
@test y_b ≈ first(model(x, params["mlp_b"]...)) #src

# ## See the placement
#
# The gateway reports its placement on `/metrics`: `gateway_model_replicas` counts the workers
# each model is placed on, and `gateway_placement_weight` gives its share of the model's traffic
# per worker.

metrics = String(take!(Downloads.download("http://127.0.0.1:8002/metrics", IOBuffer())))
foreach(println, filter(startswith("gateway_model_replicas"), split(metrics, "\n")))

@test occursin("gateway_model_replicas{model=\"mlp_a\"} 1", metrics) #src
@test occursin("gateway_model_replicas{model=\"mlp_b\"} 1", metrics) #src

# Placement follows measured load: the gateway accumulates each model's compute time and repacks
# periodically, moving models between workers to even out the load. A model with more traffic
# than one device can serve gets more replicas through `scheduling.models.<name>.replicas`, and
# every setting can be changed at runtime over the gateway's control plane.
#
# ## Stop

kserve_shutdown()
foreach(kill, processes)
foreach(wait, processes)

# [Multi-GPU Gateway](../../gateway.md) is the reference for the gateway's configuration, its
# routing and scheduling modes, its runtime control RPCs, and its metrics.
