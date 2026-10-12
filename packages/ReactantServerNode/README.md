# ReactantServerNode.jl

The **node supervisor**: one lightweight parent process that runs a whole machine from a single
node file. It detects the visible GPUs, spawns one [`ReactantServer`](../ReactantServer) worker
subprocess per device (each restricted to its own GPU), runs an embedded
[`ReactantServerGateway`](../ReactantServerGateway) when there are two or more workers so clients
keep a single endpoint, multiplexes every child's output onto one stream with `[name]` line
prefixes, and restarts children that die. It is also the entrypoint of the ReactantServer
container image.

It depends only on [`ReactantServerCore`](../ReactantServerCore), never on Reactant or the gRPC
layer: it orchestrates subprocesses and stays out of the data path. The workers and the gateway
it launches load ReactantServer and ReactantServerGateway from one of:

- **One Julia environment** (the Pkg workflow): add ReactantServerNode, ReactantServer, and
  ReactantServerGateway to the same environment and run the supervisor there. Every child runs
  in that environment, and the three packages' compat bounds keep their versions in step.
- **A ReactantServer workspace**: the container image, or a checkout of this repository run with
  `--project=packages/ReactantServerNode`. `REACTANT_WORKSPACE_ROOT` points at the workspace
  explicitly.

With neither, the supervisor stops with a `ConfigError` saying which packages to add, instead of
starting children that cannot load.

Public entry point: `supervise`.

Part of the [ReactantServer.jl](../../README.md) workspace. See
[Deployment](https://enzymead.github.io/ReactantServer.jl/dev/deployment/) for how the supervisor
turns a host and a node file into workers, and
[Node Configuration](https://enzymead.github.io/ReactantServer.jl/dev/node_config/) for the node
file itself.
