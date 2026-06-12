#!/usr/bin/env bash
# Launch the pure-Julia reactant-gateway. It reads its own gateway.yml (listen addresses and a
# flat list of worker endpoints across one or more nodes) and autodiscovers which models each
# endpoint serves via RepositoryIndex, then serves the KServe V2 gRPC proxy.
set -euo pipefail

GATEWAY_FILE="${REACTANT_GATEWAY_FILE:-/etc/reactantserver/gateway.yml}"

exec julia --project=/opt/reactantserver/packages/ReactantServerGateway -e '
    using ReactantServerGateway
    ReactantServerGateway.serve_gateway(ARGS[1])
' "${GATEWAY_FILE}"
