#!/usr/bin/env bash
# CPU end-to-end test of the supervised single-container layout, with no containers: the
# ReactantServerNode supervisor spawns two CPU workers plus the embedded gateway as host
# subprocesses, the client drives scale4 through the gateway, and SIGTERM drains the tree down.
# Verifies worker synthesis (no workers list in the node file), the embedded gateway, prefixed log
# multiplexing, the graceful drain (with requests held in flight (slow4), the gateway and then the
# workers refuse new requests with UNAVAILABLE, finish the admitted ones, and every process exits
# 0), and what a client is told of an INTERNAL error in each `grpc.error_detail` mode (a second
# node runs with `full`). Needs no GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
E2E_DIR="packages/ReactantServer/test/e2e"

WORK="$(mktemp -d)"
SUP_PID=""
cleanup() {
    # TERM, not KILL: the supervisor drains its children and waits for them; SIGKILL would orphan
    # every worker and the gateway.
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

echo "== [1/7] generating scale4 bundle =="
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
# slow4: scale4 (a model's name is its directory's) with a preprocess hook that sleeps x[1]
# seconds, so the drain check can hold requests in flight.
cp -r "$E2E_DIR/models/scale4" "$WORK/models/slow4"
cat > "$WORK/models/slow4/model.jl" << 'EOF'
register_model("slow4"; preprocess = function (inputs)
    sleep(Float64(inputs[1].data[1]))
    return inputs
end)
EOF

echo "== [2/7] writing node file, gateway config and Reactant preference override =="
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

# The embedded gateway's config, pinned to this test's workers. Without REACTANT_GATEWAY_FILE the
# supervisor reads /etc/reactantserver/gateway.yml whenever that file exists (a mounted config wins
# in the image), so on a host with a deployment's file there the test gateway would route to that
# deployment's workers. This lists what the supervisor would otherwise synthesize: worker i on
# BASE_PORT + i, with metrics on METRICS_BASE_PORT + i. Every node this script starts gets one.
# A gateway file owns its gRPC settings (the supervisor mirrors the node's global.grpc only into an
# env-configured gateway), so it also states the node's `grpc.error_detail` (argument 2).
write_gateway_config() {
    cat > "$1" << EOF
listen:
  grpc: "127.0.0.1:$GW_PORT"
  metrics: "127.0.0.1:$GW_METRICS_PORT"
endpoints:
  - "127.0.0.1:$BASE_PORT"
  - "127.0.0.1:$((BASE_PORT + 1))"
metrics_endpoints:
  - "127.0.0.1:$METRICS_BASE_PORT"
  - "127.0.0.1:$((METRICS_BASE_PORT + 1))"
worker_names: [worker0, worker1]
grpc:
  error_detail: $2
EOF
}
write_gateway_config "$WORK/gateway.yml" redacted

echo "== [3/7] starting supervisor (2 CPU workers + embedded gateway) =="
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_FILE="$WORK/gateway.yml" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --threads=1,0 --project=packages/ReactantServerNode -e '
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

echo "== [4/7] running CPU e2e client =="
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

echo "== [5/7] SIGTERM under load: the node drains =="
# The drain client holds requests in flight, sends the supervisor SIGTERM itself, and checks both
# drain stages (gateway, then workers) from the outside.
set +e
julia --project=packages/ReactantServerGateway "$E2E_DIR/drain_cpu.jl" \
    "$GW_PORT" "$GW_METRICS_PORT" "$BASE_PORT" "$METRICS_BASE_PORT" "$SUP_PID"
drain_rc=$?
set -e
if ((drain_rc != 0)); then
    echo "== drain client failed (rc=${drain_rc}); recent supervisor log: =="
    tail -120 "$WORK/supervisor.log"
    exit "$drain_rc"
fi
sup_rc=0
wait "$SUP_PID" || sup_rc=$?

echo "== [6/7] exit codes =="
# The supervisor exited 0 (argument 1) and every child drained and exited 0 on its own: no SIGTERM
# or SIGKILL fallback was needed. Argument 2 is the node's log.
check_clean_exit() {
    local rc="$1" log="$2"
    if ((rc != 0)); then
        echo "ERROR: supervisor exited $rc on SIGTERM; recent log:"
        tail -80 "$log"
        exit 1
    fi
    for child in gateway worker0 worker1; do
        if ! grep -q "^\[supervisor\] $child stopped (code=0, signal=0)" "$log"; then
            echo "ERROR: $child did not exit 0 after its drain; supervisor lines:"
            grep "^\[supervisor\]" "$log" | tail -40
            exit 1
        fi
    done
    if grep -q "^\[supervisor\] .*sending SIG" "$log"; then
        echo "ERROR: the supervisor had to signal a child:"
        grep "^\[supervisor\] .*sending SIG" "$log"
        exit 1
    fi
    echo "   supervisor and every child exited 0."
}
check_clean_exit "$sup_rc" "$WORK/supervisor.log"

# Smoke-check the multiplexed log: every child line carries its prefix.
for tag in worker0 worker1 gateway supervisor; do
    if ! grep -q "^\[$tag\] " "$WORK/supervisor.log"; then
        echo "ERROR: no [$tag] lines in the supervisor log"
        tail -40 "$WORK/supervisor.log"
        exit 1
    fi
done

echo "== [7/7] full INTERNAL error detail (node global.grpc.error_detail: full) =="
# A second node, identical but for `global.grpc.error_detail: full`, which reaches the workers through
# their config and the embedded gateway through its pinned gateway config (the Node unit tests cover
# the supervisor's mirror of global.grpc into an env-configured gateway).
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"
sed -e "s/^base_port: .*/base_port: $BASE_PORT/" -e "s/^metrics_base_port: .*/metrics_base_port: $METRICS_BASE_PORT/" \
    "$WORK/node.yaml" > "$WORK/node_full.yaml"
printf '  grpc:\n    error_detail: full\n' >> "$WORK/node_full.yaml"
write_gateway_config "$WORK/gateway_full.yml" full
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_FILE="$WORK/gateway_full.yml" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --threads=1,0 --project=packages/ReactantServerNode -e '
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
check_clean_exit "$sup_rc" "$WORK/supervisor_full.log"

echo "CPU e2e: PASS (workers synthesized, gateway routed, graceful drain, clean exit, INTERNAL error detail)"
