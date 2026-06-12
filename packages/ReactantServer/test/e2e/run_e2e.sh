#!/usr/bin/env bash
# End-to-end test of the full serving stack: ensure the gateway/worker images exist, bring up
# two GPU workers plus the gateway via podman compose, run the client over both the TCP and the
# shared-memory data paths, then tear the stack down. Run from anywhere; paths resolve relative
# to the repository root.
set -euo pipefail

# This script lives at packages/ReactantServer/test/e2e/; the repo root is four levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
E2E_DIR="packages/ReactantServer/test/e2e"

# Absolute volume sources in the compose file resolve against this.
export REACTANT_REPO="$REPO_ROOT"

ENGINE="${ENGINE:-podman}"                 # for image build / existence checks
COMPOSE_BIN="${COMPOSE_BIN:-podman-compose}"   # forwards CDI `devices:` (podman compose does not)
COMPOSE_FILE="$E2E_DIR/docker-compose.yml"
PROJECT="reactant-e2e"
COMPOSE=("$COMPOSE_BIN" -p "$PROJECT" -f "$COMPOSE_FILE")
# Host-side published ports (remapped off 8001/8002/8080 to avoid a co-resident Triton). These
# must match the host side of the port mappings in docker-compose.yml. The client reads the
# gateway/worker ports from the environment; the readiness poll uses the metrics port.
METRICS_PORT="${E2E_METRICS_PORT:-19501}"
export E2E_GATEWAY_PORT="${E2E_GATEWAY_PORT:-18501}"
export E2E_WORKER0_PORT="${E2E_WORKER0_PORT:-18080}"
READY_TIMEOUT="${E2E_READY_TIMEOUT:-600}"

cleanup() {
    echo "== tearing down stack =="
    "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== [1/5] generating scale4 bundle =="
julia --project=packages/ReactantServer "$E2E_DIR/gen_scale4.jl"

echo "== [2/5] ensuring images (build if missing) =="
"$ENGINE" image exists reactantserver-worker:latest  || make worker
"$ENGINE" image exists reactantserver-gateway:latest || make gateway

echo "== [3/5] instantiating client env =="
julia --project="$E2E_DIR" -e 'using Pkg; Pkg.instantiate()'

echo "== [4/5] bringing up stack =="
"${COMPOSE[@]}" up -d

echo "   waiting for gateway /readyz (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${METRICS_PORT}/readyz', timeout=2)" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
        echo "ERROR: gateway not ready within ${READY_TIMEOUT}s; recent logs:"
        "${COMPOSE[@]}" logs 2>&1 | tail -80 || true
        exit 1
    fi
    sleep 3
done
echo "   gateway ready."

echo "== [5/5] running e2e client =="
set +e
julia --project="$E2E_DIR" "$E2E_DIR/client.jl"
client_rc=$?
set -e

if ((client_rc != 0)); then
    echo "== client failed (rc=${client_rc}); recent stack logs: =="
    "${COMPOSE[@]}" logs 2>&1 | tail -120 || true
fi
exit "$client_rc"
