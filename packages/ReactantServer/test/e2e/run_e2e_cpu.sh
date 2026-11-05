#!/usr/bin/env bash
# CPU end-to-end test of the supervised single-container layout, with no containers: the
# ReactantServerNode supervisor spawns two CPU workers plus the embedded gateway as host
# subprocesses, the client drives scale4 through the gateway, and SIGTERM shuts the tree down.
# Verifies worker synthesis (no workers list in the node file), the env-only embedded gateway,
# prefixed log multiplexing, graceful shutdown, and what a client is told of an INTERNAL error in
# each `grpc.error_detail` mode (a second node runs with `full`). Needs no GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
E2E_DIR="packages/ReactantServer/test/e2e"

WORK="$(mktemp -d)"
SUP_PID=""
cleanup() {
    # TERM, not KILL: the supervisor forwards SIGTERM to its children and waits for them;
    # SIGKILL would orphan every worker and the gateway.
    if [[ -n "$SUP_PID" ]] && kill -0 "$SUP_PID" 2>/dev/null; then
        kill -TERM "$SUP_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SUP_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SUP_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"
# Generous default: the first run precompiles ReactantServer under the preference override
# below (a different preferences hash forces a full Reactant recompile, CPU-only and slow).
READY_TIMEOUT="${E2E_CPU_READY_TIMEOUT:-1800}"

echo "== [1/6] generating scale4 bundle =="
julia --project=packages/ReactantServer "$E2E_DIR/gen_scale4.jl"

mkdir -p "$WORK/models"
cp -r "$E2E_DIR/models/scale4" "$WORK/models/scale4"
# A postprocess that fails on a sentinel input (x[1] == -1, so y[1] == -2): the client forces an
# INTERNAL error with it and compares what it is told with the worker's log.
cat > "$WORK/models/scale4/model.jl" << 'EOF'
function postprocess(out)
    out[1].data[1] == -2 && error("e2e forced failure on the sentinel input")
    return out
end
register_model(basename(@__DIR__); postprocess = postprocess)
EOF

echo "== [2/6] writing node file and Reactant preference override =="
# No workers list: the supervisor synthesizes them (REACTANT_CPU_WORKERS below).
cat > "$WORK/node.yaml" << EOF
model_repo: $WORK/models
base_port: $BASE_PORT
metrics_base_port: $METRICS_BASE_PORT
global:
  cache_dir: $WORK/cache
  runtime:
    backend: cpu
  endpoints:
    host: 127.0.0.1
EOF

# The worker project's LocalPreferences.toml pins Reactant's persistent compile cache to
# /var/cache/reactant-compile (a Docker volume mount point), which is not writable on a dev
# host. Prepend a load-path entry that disables the persistent cache for every child.
mkdir -p "$WORK/prefs"
cat > "$WORK/prefs/Project.toml" << 'EOF'
[extras]
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
EOF
cat > "$WORK/prefs/LocalPreferences.toml" << 'EOF'
[Reactant]
persistent_cache_enabled = false
EOF
export JULIA_LOAD_PATH="$WORK/prefs:@:@v#.#:@stdlib"

echo "== [3/6] starting supervisor (2 CPU workers + embedded gateway) =="
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --project=packages/ReactantServerNode -e '
    using ReactantServerNode
    exit(ReactantServerNode.supervise(ARGS[1]; runtime_dir = ARGS[2]))
' "$WORK/node.yaml" "$WORK/run" > "$WORK/supervisor.log" 2>&1 &
SUP_PID=$!

echo "   waiting for gateway /readyz on :$GW_METRICS_PORT (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/readyz" >/dev/null 2>&1; do
    if ! kill -0 "$SUP_PID" 2>/dev/null; then
        echo "ERROR: supervisor exited early; log:"
        tail -80 "$WORK/supervisor.log"
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "ERROR: gateway not ready within ${READY_TIMEOUT}s; supervisor and gateway lines:"
        grep -E "^\[(supervisor|gateway)\]" "$WORK/supervisor.log" | tail -120
        exit 1
    fi
    sleep 2
done
echo "   gateway ready."

echo "== [4/6] running CPU e2e client =="
set +e
julia --project=packages/ReactantServerGateway "$E2E_DIR/client_cpu.jl" "$GW_PORT" "$BASE_PORT" redacted |
    tee "$WORK/client_redacted.out"
client_rc=${PIPESTATUS[0]}
set -e
if ((client_rc != 0)); then
    echo "== client failed (rc=${client_rc}); recent supervisor log: =="
    tail -120 "$WORK/supervisor.log"
    exit "$client_rc"
fi

# The gateway's /metrics aggregates every worker's export; workers self-tag their series.
metrics="$(curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/metrics")"
for tag in 'worker="worker0"' 'worker="worker1"' 'gateway_worker_metrics_up'; do
    if ! grep -qF "$tag" <<< "$metrics"; then
        echo "ERROR: aggregated /metrics is missing $tag"
        exit 1
    fi
done
echo "   aggregated /metrics carries per-worker series."

# The forced INTERNAL error is in the log of the worker the client was told about, under the same
# request id, with the exception text the client did not see.
check_forced_log() {
    local mode="$1" log="$2" worker
    worker="$(sed -n 's/^forced-internal worker=//p' "$WORK/client_$mode.out")"
    if [[ -z "$worker" ]] || ! grep -qE "^\[$worker\] .*request_id = .?e2e-internal-$mode" "$log"; then
        echo "ERROR: no [$worker] log line for request id e2e-internal-$mode ($mode error detail)"
        grep -n "e2e-internal\|e2e forced failure" "$log" | tail -20
        exit 1
    fi
    if ! grep -q "^\[$worker\] .*e2e forced failure on the sentinel input" "$log"; then
        echo "ERROR: the worker log does not carry the forced exception text ($mode error detail)"
        exit 1
    fi
    echo "   forced INTERNAL ($mode): the client message and the [$worker] log line share a request id."
}
check_forced_log redacted "$WORK/supervisor.log"

echo "== [5/6] SIGTERM shutdown =="
kill -TERM "$SUP_PID"
sup_rc=0
wait "$SUP_PID" || sup_rc=$?
if ((sup_rc != 0)); then
    echo "ERROR: supervisor exited $sup_rc on SIGTERM; recent log:"
    tail -80 "$WORK/supervisor.log"
    exit 1
fi

# Smoke-check the multiplexed log: every child line carries its prefix.
for tag in worker0 worker1 gateway supervisor; do
    if ! grep -q "^\[$tag\] " "$WORK/supervisor.log"; then
        echo "ERROR: no [$tag] lines in the supervisor log"
        tail -40 "$WORK/supervisor.log"
        exit 1
    fi
done

echo "== [6/6] full INTERNAL error detail (node global.grpc.error_detail: full) =="
# A second node, identical but for `global.grpc.error_detail: full`, which reaches the workers through
# their config and the embedded gateway through the supervisor's mirror.
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"
sed -e "s/^base_port: .*/base_port: $BASE_PORT/" -e "s/^metrics_base_port: .*/metrics_base_port: $METRICS_BASE_PORT/" \
    "$WORK/node.yaml" > "$WORK/node_full.yaml"
printf '  grpc:\n    error_detail: full\n' >> "$WORK/node_full.yaml"
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --project=packages/ReactantServerNode -e '
    using ReactantServerNode
    exit(ReactantServerNode.supervise(ARGS[1]; runtime_dir = ARGS[2]))
' "$WORK/node_full.yaml" "$WORK/run_full" > "$WORK/supervisor_full.log" 2>&1 &
SUP_PID=$!
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/readyz" >/dev/null 2>&1; do
    if ! kill -0 "$SUP_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        echo "ERROR: the full-detail node did not become ready; log:"
        tail -80 "$WORK/supervisor_full.log"
        exit 1
    fi
    sleep 2
done
set +e
julia --project=packages/ReactantServerGateway "$E2E_DIR/client_cpu.jl" "$GW_PORT" "$BASE_PORT" full |
    tee "$WORK/client_full.out"
client_rc=${PIPESTATUS[0]}
set -e
if ((client_rc != 0)); then
    echo "== client failed (rc=${client_rc}); recent supervisor log: =="
    tail -120 "$WORK/supervisor_full.log"
    exit "$client_rc"
fi
check_forced_log full "$WORK/supervisor_full.log"
kill -TERM "$SUP_PID"
sup_rc=0
wait "$SUP_PID" || sup_rc=$?
if ((sup_rc != 0)); then
    echo "ERROR: the full-detail supervisor exited $sup_rc on SIGTERM; recent log:"
    tail -80 "$WORK/supervisor_full.log"
    exit 1
fi

echo "CPU e2e: PASS (workers synthesized, gateway routed, clean shutdown, INTERNAL error detail)"
