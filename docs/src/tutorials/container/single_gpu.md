# Serve on One GPU

This page runs the ReactantServer container image on a GPU host, serves a model from a directory
of bundles, and queries it. The image is the supported way to deploy: it carries Julia, every
package with its precompiled caches, and the CUDA libraries Reactant needs, so the host needs only
an NVIDIA driver and a container runtime.

## What you need

- An NVIDIA GPU with its driver installed.
- podman or Docker, with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  so containers can see the GPU. Podman reads GPUs through CDI: generate the spec once with
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
- A model repository: a directory whose subdirectories are bundles.
  [Export a Model](../basics/export.md) writes one, `models/mlp`, in a few lines of Julia on any
  machine; copy `models/` to the GPU host.

## Run the container

The image is published as `ghcr.io/enzymead/reactantserver`, tagged with each release. Mount the
model repository and publish the two ports:

```sh
podman run -d --name reactantserver \
  --device nvidia.com/gpu=0 \
  --ipc=host --pids-limit=-1 \
  -p 8001:8001 -p 8002:8002 \
  -v $PWD/models:/var/lib/reactantserver/models \
  -v reactant-compile-cache:/var/cache/reactant-compile \
  --stop-timeout 30 \
  ghcr.io/enzymead/reactantserver:0.1
```

With Docker, replace `--device nvidia.com/gpu=0` with `--gpus device=0`; the other flags are the
same. In order:

- `--device nvidia.com/gpu=0` gives the container one GPU. The supervisor inside runs one worker
  per GPU it can see, so the GPUs you grant are the node's topology.
- `--ipc=host` shares the host's `/dev/shm`, so a client on the host can send tensors through
  shared memory instead of the network. A container's own `/dev/shm` is 64 MiB, too small for it.
- `--pids-limit=-1` lifts podman's default 2048-thread limit, which a worker's thread pools can
  cross while loading models.
- The model repository is mounted writable: each bundle caches its compiled programs in its own
  `.cache/` directory, so later starts load them in milliseconds instead of compiling again.
- The `reactant-compile-cache` volume keeps XLA's autotuning results across container
  recreation.
- `--stop-timeout 30` gives the supervisor time to drain its workers on stop.

[Deployment](../../deployment.md) explains each setting in more depth, and lists the alternatives
when `--ipc=host` is not an option.

## Wait for it to be ready

On its first start the worker compiles every model before it accepts traffic, which takes from
seconds for a small model to much longer for a large set. Follow the logs, and poll `/readyz`
until it answers 200:

```sh
podman logs -f reactantserver
curl -sf http://127.0.0.1:8002/readyz && echo ready
```

With one GPU the node runs a single worker on the public ports and no gateway: KServe V2 gRPC on
8001, and `/readyz`, `/healthz`, and Prometheus `/metrics` on 8002.

## Query it

Any KServe V2 or Triton gRPC client works. From Julia, with `ReactantServerClient` installed on any
machine that can reach the host:

```julia
using ReactantServerClient

kserve_init()
mlp = KServeModel("grpc://127.0.0.1:8001", "mlp")
x = reshape(Float32[1, 2, 3, 4], 4, 1)               # (features, batch)
y = InferOutput("output", infer_sync(mlp, [InferInput("input", x)]), Float32)
kserve_shutdown()
```

[Run Inference](../basics/inference.md) covers batched inference over a dataset and the shared-memory
transport.

## Configure the node

The image runs `/etc/reactantserver/node.yaml`, a node file that serves everything in the model
repository with one worker per visible GPU. To change it, mount your own file over that path:

```yaml
# node.yaml
model_repo: /var/lib/reactantserver/models
base_port: 8080
metrics_base_port: 9100
gpus: auto

global:
  runtime:
    backend: cuda
  endpoints:
    host: 0.0.0.0
```

```sh
podman run ... -v $PWD/node.yaml:/etc/reactantserver/node.yaml:ro ghcr.io/enzymead/reactantserver:0.1
```

Any key can also be overridden from the environment with `-e`, for example
`-e INFERENCE_SERVER_SCHEDULER_DISCIPLINE=fifo`. [Node Configuration](../../node_config.md) covers
every key and its environment variable.

## Stop

```sh
podman stop reactantserver
```

The supervisor drains its workers on SIGTERM before exiting.

## Next

- [Scale to Many GPUs](multi_gpu.md): the same image and node file on a multi-GPU host.
- [Deployment](../../deployment.md): the image's tags and provenance, every container setting,
  health checks, and building the image yourself.
- [Monitoring](../../monitoring.md): scraping the node's metrics.
- [Security](../../security.md): what to know before exposing an endpoint.
