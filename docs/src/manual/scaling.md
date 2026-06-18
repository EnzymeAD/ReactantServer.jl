# Scaling to Multiple GPUs

[Getting Started](getting_started.md) ran one model on one GPU. This page adds GPUs. The key
point: **nothing about your model, client, or external interface changes.** You run the same
container (or the same `supervise` call), the supervisor puts one worker on each GPU, and it
fronts them with an embedded gateway so clients still send one request to one endpoint and it is
routed to a worker that serves the model.

For help deciding between the distributed (memory-constrained) and replicated (compute-constrained)
multi-GPU shapes, with an example configuration for each, see
[Common Use Cases](common_use_cases.md).

## The node file is unchanged

`gpus: auto` already means "one worker per visible GPU", so the single-GPU node file from
[Getting Started](getting_started.md) scales as-is: give the host more GPUs and the supervisor
runs more workers. You only edit the node file to do something non-default:

```yaml
model_repo: /var/lib/reactantserver/models
base_port: 8080           # worker i listens on base_port + i (8080, 8081, ...)
metrics_base_port: 9100   # and metrics on metrics_base_port + i (9100, 9101, ...)
gpus: auto                # or an integer count, or an explicit device list

global:
  runtime:
    backend: cuda
  endpoints:
    host: 0.0.0.0

# Optional: replicate or pin specific models to device memory on specific workers.
# models:
#   mlp: [worker0, worker1]   # served (and kept device-resident) on both GPUs
```

A model listed on more than one worker is replicated, and the gateway load-balances requests for
it across those workers. See [Node Configuration](node_config.md) for the `models:` map and
[On-demand Weights](on_demand_weights.md) for fitting more models than GPU memory holds.

## Run it with docker compose

Same container as before; just grant it all the GPUs (the shipped `docker-compose.yml` already
reserves every GPU):

```
REACTANTSERVER_MODELS=$PWD/models docker compose up
```

The supervisor detects N GPUs, runs `worker0..workerN-1` (each pinned to one device), and runs
the embedded gateway on `localhost:8001` (gRPC) and `localhost:8002` (metrics). Clients are
unchanged from [Getting Started](getting_started.md): they still call `8001` with the model name,
and the gateway routes each request to a worker serving that model.

## Run it from pure Julia

One `supervise` call on the multi-GPU host does the whole fan-out, just as it does for one GPU:

```julia
using ReactantServerNode
ReactantServerNode.supervise("node.yaml")   # one worker per GPU + the embedded gateway
```

This single parent process spawns one [`ReactantServer.serve`](@ref) worker subprocess per GPU
and the gateway as another subprocess, multiplexes their logs onto its stdout with `[worker0]` /
`[gateway]` prefixes, and restarts any child that dies. (You can still run each worker and the
gateway by hand, as separate `serve` / `serve_gateway` processes, but the supervisor is the
intended path and the only one the container uses.)

## How the supervisor decides what to start

The supervisor's job, on startup, is to turn "this host + this node file" into a concrete set of
child processes. It does so in three steps.

**1. Detect the devices.** The first of these that yields a non-empty answer wins
(`ReactantServerNode.detect_gpus`):

1. `REACTANT_GPUS` environment variable: a count (`2`) or an explicit list (`0,2` or GPU UUIDs);
   `0` means a CPU node.
2. the node file's `gpus:` key (`auto`, a count, or a list).
3. a `CUDA_VISIBLE_DEVICES` already set on the container.
4. `nvidia-smi` enumeration.
5. `/dev/nvidiaN` device nodes.

For a CUDA node that finds no devices, startup fails with guidance (run with `--gpus all`, set
`REACTANT_GPUS`, or set `backend: cpu`).

**2. Materialize the workers** (`ReactantServerCore.materialize_node!`). With no `workers:` list,
one worker is synthesized per detected device (`worker0..workerN-1`). With an explicit list, each
worker is assigned a device positionally (or by its `gpu:` key). Either way each worker is pinned
to exactly one device through its own `CUDA_VISIBLE_DEVICES`, so inside the worker the device is
always ordinal 0 and the single-GPU worker code runs unchanged. The materialized node file is
written to `/run/reactantserver/node.yaml` for inspection.

**3. Decide on the gateway** by worker count, in the default `all` role:

- **One worker → no gateway.** A lone worker already serves the full KServe V2 API, so the
  supervisor binds it directly to the public ports (8001/8002) and starts no gateway. No extra
  process, no extra hop.
- **Two or more workers → workers plus the embedded gateway.** Each worker binds `base_port + i`
  (and `metrics_base_port + i`), and the gateway binds the public 8001/8002. The supervisor
  synthesizes the gateway's worker list (and worker metrics list) from the node file, so the
  gateway needs no config of its own.

The external interface (8001 for gRPC, 8002 for metrics/health) is therefore identical whether
you run 1 GPU or 8, which is why your client and compose ports never change as you scale.

The `REACTANT_ROLE` environment variable can override the default `all` role (`workers` runs only
the workers, `gateway` runs only the gateway) for splitting a deployment across machines; that
multi-node topology is beyond the scope of these guides.

### Watch it without a GPU

To see the decision logic and the prefixed logs on a machine with no GPU, run the supervisor as a
CPU node with two synthetic workers:

```
REACTANT_GPUS=0 REACTANT_CPU_WORKERS=2 \
  julia --project=packages/ReactantServerNode -e 'using ReactantServerNode; ReactantServerNode.supervise("node.yaml")'
```

with `backend: cpu` in the node file. You will see `worker0`, `worker1`, and `gateway` start,
their logs interleaved with `[name]` prefixes, and the gateway serving on 8001/8002, exactly the
multi-worker shape it takes on a multi-GPU host.

## Metrics

One scrape on `8002` still covers the whole node: the embedded gateway serves its own
`gateway_*` series and merges in every worker's `/metrics`. Each worker tags its series with
`worker` and `gpu` labels itself (the `gpu` value is the physical device behind its
`CUDA_VISIBLE_DEVICES`), so per-GPU and per-worker breakdowns need no Prometheus relabeling, e.g.
`sum by (gpu) (rate(worker_dispatch_total[1m]))`. See [Docker Deployment](docker.md) for the
scrape config.
