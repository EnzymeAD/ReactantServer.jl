#!/usr/bin/env bash
# Launch one ReactantServer worker. The worker reads the node file and serves the slice named by
# REACTANT_WORKER_NAME. When the node has a single worker, REACTANT_WORKER_NAME may be unset and
# the sole worker is selected automatically.
set -euo pipefail

NODE_FILE="${REACTANT_NODE_FILE:-/etc/reactantserver/node.yaml}"

exec julia --project=/opt/reactantserver/packages/ReactantServer -e '
    using ReactantServer
    worker = get(ENV, "REACTANT_WORKER_NAME", "")
    ReactantServer.serve(ARGS[1]; worker = isempty(worker) ? nothing : worker)
' "${NODE_FILE}"
