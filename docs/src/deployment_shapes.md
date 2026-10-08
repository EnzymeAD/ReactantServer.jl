# Deployment Shapes

The deployment decision comes down to how many GPUs you have and, with more than one, whether
your constraint is memory or compute:

| Situation | Shape | Optimizes for |
|---|---|---|
| One GPU, many models | Single GPU | Fitting many models on limited hardware |
| Several GPUs, models do not all fit everywhere | Multi-GPU distributed | Serving more models than any one GPU holds |
| Several GPUs, models fit everywhere, need throughput | Multi-GPU replicated | Spreading compute load across replicas |
| Many machines | Multi-node | Whatever your control plane decides |

Each shape is selected by a configuration value, not a different architecture, and each is
strictly additive: a single-GPU deployment never pays for the gateway in dependencies or runtime
cost, and a gateway deployment never requires a control plane.

## Single GPU

A single worker is the entire deployment. The worker speaks the full KServe V2 gRPC surface
itself: one Julia process, one YAML file, nothing else to operate. This is the recommended
starting point for small labs and the conceptual foundation for the distributed multi-GPU case,
which scales the same idea across cards. The [Scheduling](scheduling.md) page covers the fair
scheduler and batch coalescing; [On-demand Weights](on_demand_weights.md) covers the weight
cache that serves a catalog larger than device memory.

## Multi-GPU, distributed without replication

You have several cards, but your model library is large enough that it cannot fit on every GPU.
Your constraint is memory. The gateway uses LPT (longest-processing-time) packing to place each
model on one GPU, balancing memory footprint against compute load; spreading models across cards
by their load keeps any one card from being monopolized. The workers switch to the simpler FIFO
discipline and the placement intelligence moves upstream to the gateway. There is still no
external infrastructure to stand up: the gateway is one more Julia process and one more YAML file,
and there is no placement file to maintain.

## Multi-GPU, replicated

You have enough memory that your models fit on more than one card. Your constraint is compute:
you want to spread a model's request load across replicas for throughput. The gateway gives a
replicated model a replica count, places it on that many distinct GPUs, and routes its requests
to fill one replica's batch before moving to the next, so batch coalescing is preserved across
replicas. Size replica counts for the model's expected concurrency, and make sure the memory is
there: replicating a model puts its full weight footprint on every GPU it lands on, and an
over-subscribed GPU thrashes, loading and evicting weights on nearly every request. See
[Multi-GPU Gateway](gateway.md) for the scheduling modes and knobs.

## Multi-node (bring your own control plane)

Beyond a single host, the project deliberately does not ship a multi-node control plane: anyone
who needs multi-node coordination is usually at a scale where a generic control plane would not
fit. Instead, each node exposes the interface a control plane integrates against, so you can build
or adapt your own coordination layer on top of ReactantServer nodes. The endpoint contract is the
KServe V2 gRPC data plane (`ModelInfer`, `RepositoryIndex`, `ServerReady`) plus the worker
control RPCs (`ModelControlStatus`, `SetModelResidency`, `SetModelPolicy` for residency and
policy, `CompactMemory` to defragment device memory), and an admin HTTP port serving `/healthz`,
`/readyz`, and Prometheus `/metrics`. A node's embedded gateway additionally answers
`GatewayControlService` for its own scheduling state. Your control plane discovers which models a
node serves via `RepositoryIndex` and routes `ModelInfer` to a node that reports the model ready.
The project supplies the seam; the organization supplies the tooling that encodes needs only it
can know.

The simplest multi-node setup reuses the standalone gateway pointed at every node's worker
endpoints; a larger deployment replaces it with your own control plane speaking the same gRPC
interface. See [Multi-GPU Gateway](gateway.md) for the standalone gateway's configuration.

Who the project is built for, and how each shape serves that mission, is on the
[Philosophy](design/philosophy.md) page.
