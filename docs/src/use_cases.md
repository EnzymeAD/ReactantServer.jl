# Use Cases

Most deployments match one of the three cases below. Find the closest one and set only the
settings it names; everything else can stay at its default. The defaults assume nothing about
your models:

- each GPU worker loads every model it can fit onto its GPU at startup, and serves the rest on
  demand
- no copy of a model's weights is kept in host RAM (`runtime.host_weight_budget: 0`), so host
  memory use stays small, but a model that has to be reloaded onto the GPU reads its weights file
  from disk

The settings are described in full on [Node Configuration](node_config.md) and
[On-demand Weights](on_demand_weights.md).

## Static: every model fits on the GPU

Every model stays on the GPU from startup onwards. Nothing is swapped, nothing is kept in host
RAM, and no weights file is read after startup.

```yaml
model_repo: /var/lib/reactantserver/models
global:
  runtime:
    backend: cuda
```

That is the whole configuration: the defaults already do this. With several GPUs, every worker
serves every model, and the [gateway](gateway.md) spreads requests across them. If the models fit
on the node only when split between GPUs, give each worker its share with `models_include`.

**Check it:** the worker logs no `model weights reloaded from disk` warnings, and
`worker_weight_evicts_total` stays at 0. If either appears, the models do not all fit: use the
hot swapping case.

## Hot swapping: the models do not all fit

The models move on and off the GPU as requests arrive. A reload is fast when the model's weights
are in host RAM and much slower when they come from disk, so give host RAM a budget:

```yaml
model_repo: /var/lib/reactantserver/models
global:
  runtime:
    backend: cuda
    host_weight_budget: 64g          # host RAM for weight copies; size it to the swapped models
    shared_host_weights: true        # with several GPUs: one copy per model for the whole node
    shared_host_weights_mode: "660"
```

- `host_weight_budget` is the setting that matters. Size it to hold the weights of the models
  that get swapped (roughly the sum of their `weights.safetensors` files), within the RAM the
  machine can spare. When it is too small, the least recently loaded copies are dropped and those
  models are read from disk on their next reload.
- `shared_host_weights: true` makes the budget one pool for the whole node, holding each model
  once however many GPUs load it. Without it, each worker keeps its own copies and gets an equal
  share of the budget.
- `host_weight_budget: unlimited` keeps every model in host RAM with no cap. Use it only when RAM
  comfortably holds every model.

**Check it:** after warm-up `worker_host_weight_hits_total` keeps growing and
`worker_host_weight_misses_total` barely moves. A `model weights reloaded from disk` warning
means that model was not in host RAM when it was needed.

**Common mistakes:**

- Leaving `host_weight_budget` at 0: every swap reads the disk, and the worker warns about it.
- A budget larger than the free RAM, or than a container's memory limit (shared copies live in
  `/dev/shm`, which counts against it): the machine runs out of memory instead of evicting.

## Mixed: hot swapping with some models pinned

Some models must always answer at full speed, and the rest share what is left of the GPU. Pin the
latency-critical models to GPU memory and configure the rest as in the hot swapping case:

```yaml
model_repo: /var/lib/reactantserver/models
global:
  runtime:
    backend: cuda
    host_weight_budget: 64g
    shared_host_weights: true
    shared_host_weights_mode: "660"
models:                              # pinned to GPU memory on the listed workers
  detector: [worker0, worker1]
  classifier: [worker0]
```

- A model in the `models:` map is loaded at startup and never leaves the GPU on the listed
  workers. Every other model is swapped as in the hot swapping case.
- `residency: system` (under `scheduler.models`) is the middle ground: the model stays in host
  RAM permanently, so its reloads are always fast, but it is swapped on the GPU like any other.
  It counts against `host_weight_budget`.

**Common mistakes:**

- Pinning so much that no GPU memory is left for swapping. The worker warns at startup
  (`no room for the on-demand cache`), and every unpinned model then reloads on every request.
- `residency: system` models that add up to more than `host_weight_budget`. They still load,
  the worker warns that the budget is overcommitted, and no other model's copy stays in RAM.
