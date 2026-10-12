# Scale to Many GPUs

The container from [Serve on One GPU](single_gpu.md) scales to a multi-GPU host
unchanged: grant it more GPUs and the supervisor inside runs one worker per GPU, puts the embedded
gateway in front of them, and keeps the same public ports. Clients do not change. What you choose
is how models are spread across the GPUs, and that depends on whether your constraint is compute
or memory.

| Situation | Shape | Optimizes for |
|---|---|---|
| Models fit on every GPU, and you need throughput | Replicated | Spreading each model's load across GPUs |
| Models do not all fit on every GPU | Distributed | Serving more models than any one GPU holds |

Both shapes are the same image and the same node file; they differ in the gateway's scheduling
mode and the workers' scheduling discipline.

## Grant every GPU

Run the image as before, granting all GPUs:

```sh
podman run -d --name reactantserver \
  --device nvidia.com/gpu=all \
  --ipc=host --pids-limit=-1 \
  -p 8001:8001 -p 8002:8002 \
  -v $PWD/models:/var/lib/reactantserver/models \
  -v reactant-compile-cache:/var/cache/reactant-compile \
  --stop-timeout 30 \
  ghcr.io/enzymead/reactantserver:0.1
```

With Docker, use `--gpus all`. The logs show one `[workerN]` per GPU and a `[gateway]`; the
workers listen on internal ports (8080 and up), and the gateway takes 8001 and 8002. Wait for
`curl -sf http://127.0.0.1:8002/readyz` as before: the gateway reports ready once a worker does.

## Replicated: every model on every GPU

By default every worker serves every model in the repository, and the gateway spreads each
model's requests across all of them (`round_robin`). That is the replicated shape, and it needs no
configuration. Each GPU holds a full copy of every model's weights, so it fits only when the
whole repository fits on one GPU.

Several workers on one host each hold a host-RAM copy of every model's weights as well, unless
they share one. Turn that on in the node file:

```yaml
global:
  runtime:
    backend: cuda
    shared_host_weights: true
    shared_host_weights_mode: "660"
```

`shared_host_weights` keeps one copy of each model's weights in `/dev/shm` for all of the node's
workers, which is another reason for `--ipc=host` (or a `--shm-size` larger than the weights). See
[On-demand Weights](../../on_demand_weights.md).

## Distributed: each model on a few GPUs

When the models do not all fit on every GPU, or when you want each model's requests to
concentrate on few GPUs so they coalesce into full batches, use the gateway's `lpt_packing` mode.
It places each model on a fixed number of GPUs, balancing measured compute against each GPU's
weight memory, and repacks as the load changes.

`lpt_packing` needs the workers on the `fifo` or `edf` discipline, since placement decisions move
to the gateway; this example uses `fifo`. Both settings can be passed as environment variables:

```sh
podman run -d --name reactantserver \
  --device nvidia.com/gpu=all \
  --ipc=host --pids-limit=-1 \
  -p 8001:8001 -p 8002:8002 \
  -v $PWD/models:/var/lib/reactantserver/models \
  -v reactant-compile-cache:/var/cache/reactant-compile \
  -e INFERENCE_SERVER_SCHEDULER_DISCIPLINE=fifo \
  -e REACTANT_GATEWAY_SCHEDULING_MODE=lpt_packing \
  -e REACTANT_GATEWAY_SCHEDULING_DEFAULT_REPLICAS=1 \
  --stop-timeout 30 \
  ghcr.io/enzymead/reactantserver:0.1
```

`INFERENCE_SERVER_*` variables override node file keys for the workers, and `REACTANT_GATEWAY_*`
variables override the embedded gateway's settings. `default_replicas: 1` places each model on one
GPU, the shape that coalesces best; a number greater than one, or `all`, replicates every model
that many times.

The gateway waits for every worker to finish compiling before it starts, then logs each placement
as an `lpt_packing: repack` line. The `gateway_model_replicas` and `gateway_placement_weight`
series on `:8002/metrics` show where each model landed:

```sh
curl -s http://127.0.0.1:8002/metrics | grep -E '^gateway_(model_replicas|placement_weight)'
```

### Give a busy model more GPUs

Per-model settings need a `gateway.yml`, mounted at `/etc/reactantserver/gateway.yml`. Once the
file exists, it owns the gateway's worker list, so it lists the workers' endpoints too:

```yaml
# gateway.yml for a 4-GPU node
endpoints: ["127.0.0.1:8080", "127.0.0.1:8081", "127.0.0.1:8082", "127.0.0.1:8083"]
metrics_endpoints: ["127.0.0.1:9100", "127.0.0.1:9101", "127.0.0.1:9102", "127.0.0.1:9103"]
worker_names: [worker0, worker1, worker2, worker3]
scheduling:
  mode: lpt_packing
  default_replicas: 1
  models:
    big-model:
      replicas: 2          # this model on two GPUs
```

Each replica puts the model's full weights on its GPU, so size replica counts against GPU memory;
the gateway warns at each repack when a placement does not fit.

A running gateway can also be retuned without a restart, including a model's replica count, over
its control plane. [Multi-GPU Gateway](../../gateway.md) documents every scheduling setting, the
routing policies, and the runtime control RPCs.

## Beyond one host

Each node exposes the interfaces a multi-node control plane needs: the KServe V2 data plane, the
worker control RPCs, and health and metrics over HTTP. The project does not ship a multi-node
control plane. The simplest multi-node setup is a standalone gateway pointed at every node's
workers, as in [Front Workers with a Gateway](../julia/gateway.md); [Deployment](../../deployment.md)
describes the endpoint contract.
