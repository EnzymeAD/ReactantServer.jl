#!/usr/bin/env bash
# Launch one ReactantServer worker. The worker reads the node file and serves the slice named by
# REACTANT_WORKER_NAME. When the node has a single worker, REACTANT_WORKER_NAME may be unset and
# the sole worker is selected automatically.
set -euo pipefail

NODE_FILE="${REACTANT_NODE_FILE:-/etc/reactantserver/node.yaml}"

# --threads=auto,1: a host-sized default pool runs per-request preprocess/postprocess in parallel
# while one interactive thread runs the GPU dispatch loop, so CPU hook work overlaps the
# serialized GPU execution. The supervisor (entrypoint.node.sh) sets the same on its worker
# children; this covers the direct single-worker escape hatch.
exec julia --threads=auto,1 --project=/opt/reactantserver/packages/ReactantServer -e '
    using ReactantServer
    worker = get(ENV, "REACTANT_WORKER_NAME", "")
    ReactantServer.serve(ARGS[1]; worker = isempty(worker) ? nothing : worker)
' "${NODE_FILE}"
