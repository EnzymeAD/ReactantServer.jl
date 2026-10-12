# Single-container node supervisor. One lightweight parent process detects the visible GPUs,
# spawns one ordinary ReactantServer worker subprocess per device (each restricted to its GPU via
# CUDA_VISIBLE_DEVICES, so the existing single-device worker runs unchanged), optionally runs the
# gateway as another child, multiplexes every child's stdout/stderr onto its own stdout with a
# `[name]` line prefix, and restarts children that die. This makes `docker run --gpus all` the
# whole multi-GPU deployment story; the per-GPU-container layout remains available for multi-node
# setups via the `workers` and `gateway` roles.
#
# This package never imports Reactant (or gRPC/HTTP): it only orchestrates subprocesses, so it
# starts in about a second and stays out of the data path.
module ReactantServerNode

using ReactantServerCore
import ReactantServerCore: ConfigError, _node_workers, _worker_name, _worker_port
import TOML
import YAML

include("gpus.jl")
include("spec.jl")
include("signals.jl")
include("supervisor.jl")
include("main.jl")

export supervise

end # module ReactantServerNode
