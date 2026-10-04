#!/usr/bin/env bash
# Role-aware container healthcheck for the supervised node image.
#
# all / gateway roles: the gateway's /readyz aggregates worker readiness (200 once at least one
# worker reports ServerReady), so a cheap curl on the admin port is the truth.
# workers role: no gateway in the container; fall back to the Julia probe, which reads the node
# file and reports ready when at least one worker answers ServerReady.
set -euo pipefail

ROLE="${REACTANT_ROLE:-all}"

case "${ROLE}" in
  workers)
    exec julia --project=/opt/reactantserver /usr/local/bin/healthcheck.worker.jl
    ;;
  *)
    METRICS_PORT="${REACTANT_GATEWAY_METRICS_PORT:-8002}"
    exec curl -fsS "http://127.0.0.1:${METRICS_PORT}/readyz" > /dev/null
    ;;
esac
