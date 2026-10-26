#!/usr/bin/env bash
# Bring up a worker (single mode) or a supervised node with its gateway (node mode) on a generated
# model repository, run the stress driver against it, then shut everything down and collect the
# logs. See README.md beside this file.
#
#   tools/residency_stress/run.sh WORK_DIR [single|node]
#
# WORK_DIR is gen_models.jl's OUT_DIR (it holds models/ and catalog.toml); each run writes
# WORK_DIR/run-<timestamp>/. Knobs (environment):
#
#   RS_BACKEND=cuda|cpu        (cuda)  cuda runs every worker under `gpu-lease`
#   RS_GPUS=N                  (2)     node mode: workers (one per leased GPU)
#   RS_MEM_FRACTION=F          (0.10)  runtime.device_memory_fraction: the BFC arena per GPU
#   RS_HOST_BUDGET=SIZE        (2g)    runtime.host_weight_budget (0, 512m, unlimited, ...)
#   RS_SHARED=true|false       (false) runtime.shared_host_weights
#   RS_MIGRATION=off|copy|replace (off) runtime.weights_migration
#   RS_EXEC_CACHE=true|false   (true)  runtime.executable_cache
#   RS_COMPACTION_INTERVAL=N   (0)     scheduler.compaction_interval (single mode only)
#   RS_GW_COMPACTION=MODE      (off)   gateway scheduling.compaction_mode (node mode)
#   RS_POLL=SECONDS            (3)     model_poll_seconds for the directory watcher
#   RS_DEBUG=1                         JULIA_DEBUG=ReactantServer (per load/evict memory lines)
#   RS_READY_TIMEOUT=SECONDS   (1800)
#   RS_LEASE_TIMEOUT=DURATION  (5m)    how long gpu-lease waits for free GPUs
#   RS_STRESS_THREADS=N        (8)     driver threads
# plus every RS_* knob of stress.jl (RS_DURATION, RS_CONCURRENCY, RS_CHAOS, ...).
set -euo pipefail

# RS_REPO_ROOT lets a frozen copy of this script run against the tree (bash reads a script as it
# runs, so editing run.sh under a long run can derail it; launch a copy for long runs).
REPO_ROOT="${RS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK="$(cd "$1" && pwd)"
MODE="${2:-single}"
BACKEND="${RS_BACKEND:-cuda}"
RUN="$WORK/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
echo "run dir: $RUN"
# The driver renames, removes and updates bundles, so each run serves a hard-linked copy of the
# generated repository: the server and the driver only ever replace files (write a new file, rename
# it over the old), never write into one, so the pristine tree is never touched.
cp -al "$WORK/models" "$RUN/models"

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"

# A writable Reactant compile cache (the worker project's LocalPreferences pins /var/cache, which a
# dev host cannot write). Shared across runs in WORK so autotune results are reused.
mkdir -p "$RUN/prefs" "$WORK/reactant-cache"
printf '[extras]\nReactant = "3c362404-f566-11ee-1572-e11a4b42c853"\n' > "$RUN/prefs/Project.toml"
printf '[Reactant]\npersistent_cache_enabled = true\npersistent_cache_directory = "%s"\npersistent_kernel_cache_enabled = false\npersistent_autotune_cache_enabled = true\n' \
    "$WORK/reactant-cache" > "$RUN/prefs/LocalPreferences.toml"
export JULIA_LOAD_PATH="$RUN/prefs:@:@v#.#:@stdlib"

{
    echo "model_repo: $RUN/models"
    echo "base_port: $BASE_PORT"
    echo "metrics_base_port: $METRICS_BASE_PORT"
    if [[ "$MODE" == "node" && "$BACKEND" == "cuda" ]]; then echo "gpus: ${RS_GPUS:-2}"; fi
    echo "global:"
    echo "  cache_dir: $RUN/cache"
    echo "  model_poll_seconds: ${RS_POLL:-3}"
    echo "  runtime:"
    echo "    backend: $BACKEND"
    echo "    device_memory_fraction: ${RS_MEM_FRACTION:-0.10}"
    echo "    preallocate: true"
    echo "    host_weight_budget: ${RS_HOST_BUDGET:-2g}"
    echo "    shared_host_weights: ${RS_SHARED:-false}"
    echo "    shared_host_weights_mode: \"600\""
    echo "    weights_migration: ${RS_MIGRATION:-off}"
    echo "    executable_cache: ${RS_EXEC_CACHE:-true}"
    echo "  scheduler:"
    echo "    compaction_interval: ${RS_COMPACTION_INTERVAL:-0}"
    echo "  endpoints:"
    echo "    host: 127.0.0.1"
    if [[ "$MODE" == "single" ]]; then
        echo "workers:"
        echo "  - { name: worker0 }"
    fi
} > "$RUN/node.yaml"
cp "$RUN/node.yaml" "$RUN/node.yaml.used"

[[ "${RS_DEBUG:-0}" == "1" ]] && export JULIA_DEBUG=ReactantServer

SUP_PID=""
cleanup() {
    if [[ -n "$SUP_PID" ]] && kill -0 "$SUP_PID" 2>/dev/null; then
        # TERM the julia children first (gpu-lease forwards, the supervisor forwards to workers).
        pkill -TERM -P "$SUP_PID" 2>/dev/null || true
        kill -TERM "$SUP_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SUP_PID" 2>/dev/null || break
            sleep 1
        done
        pkill -KILL -P "$SUP_PID" 2>/dev/null || true
        kill -KILL "$SUP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Prefix every server log line with the Unix time, so log events line up with the driver's.
stamp() { perl -MTime::HiRes=time -ne 'BEGIN { $| = 1 } printf "%.3f %s", time, $_'; }

LEASE=()
if [[ "$BACKEND" == "cuda" ]]; then
    NEED=1
    [[ "$MODE" == "node" ]] && NEED="${RS_GPUS:-2}"
    LEASE=(gpu-lease -need "$NEED" -wait -timeout "${RS_LEASE_TIMEOUT:-5m}")
fi

if [[ "$MODE" == "single" ]]; then
    TARGET="127.0.0.1:$BASE_PORT"
    METRICS="http://127.0.0.1:$METRICS_BASE_PORT/metrics"
    READY="http://127.0.0.1:$METRICS_BASE_PORT/readyz"
    "${LEASE[@]}" julia --project="$REPO_ROOT/packages/ReactantServer" --threads=auto,1 \
        -e 'using ReactantServer; ReactantServer.serve(ARGS[1])' "$RUN/node.yaml" > >(stamp > "$RUN/worker.log") 2>&1 &
    SUP_PID=$!
else
    TARGET="127.0.0.1:$GW_PORT"
    METRICS="http://127.0.0.1:$GW_METRICS_PORT/metrics"
    READY="http://127.0.0.1:$GW_METRICS_PORT/readyz"
    export REACTANT_WORKSPACE_ROOT="$REPO_ROOT"
    export REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT"
    export REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT"
    export REACTANT_GATEWAY_SCHEDULING_COMPACTION_MODE="${RS_GW_COMPACTION:-off}"
    if [[ "$BACKEND" == "cpu" ]]; then
        export REACTANT_GPUS=0
        export REACTANT_CPU_WORKERS="${RS_GPUS:-2}"
    fi
    "${LEASE[@]}" julia --handle-signals=no --project="$REPO_ROOT/packages/ReactantServerNode" -e '
        using ReactantServerNode
        exit(ReactantServerNode.supervise(ARGS[1]; runtime_dir = ARGS[2]))
    ' "$RUN/node.yaml" "$RUN/supervisor" > >(stamp > "$RUN/worker.log") 2>&1 &
    SUP_PID=$!
fi

echo "waiting for $READY (log: $RUN/worker.log)"
deadline=$((SECONDS + ${RS_READY_TIMEOUT:-1800}))
until curl -fsS "$READY" > /dev/null 2>&1; do
    if ! kill -0 "$SUP_PID" 2>/dev/null; then
        echo "ERROR: server exited during startup"
        tail -60 "$RUN/worker.log"
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "ERROR: not ready within ${RS_READY_TIMEOUT:-1800}s"
        tail -60 "$RUN/worker.log"
        exit 1
    fi
    sleep 3
done
echo "ready after ${SECONDS}s"

# Every julia process under the launcher (the workers, and in node mode also the gateway and the
# supervisor) is sampled for RSS; the driver re-reads this file each sample.
pid_tree() {
    local p
    for p in $(pgrep -P "$1" 2>/dev/null); do
        echo "$p"
        pid_tree "$p"
    done
}
(
    while kill -0 "$SUP_PID" 2>/dev/null; do
        { echo "$SUP_PID"; pid_tree "$SUP_PID"; } | while read -r p; do
            [[ "$(cat /proc/$p/comm 2>/dev/null)" == julia ]] && echo "$p"
        done > "$RUN/pids.tmp" && mv "$RUN/pids.tmp" "$RUN/pids"
        sleep 10
    done
) &
sleep 1
if [[ "$MODE" == "node" ]]; then
    # Per-worker metrics too: the gateway aggregates, but per-worker series make drift attributable.
    METRICS="http://127.0.0.1:$METRICS_BASE_PORT/metrics"
    for i in $(seq 1 $((${RS_GPUS:-2} - 1))); do METRICS="$METRICS,http://127.0.0.1:$((METRICS_BASE_PORT + i))/metrics"; done
fi

set +e
RS_TARGET="$TARGET" RS_METRICS="$METRICS" RS_CATALOG="$WORK/catalog.toml" RS_REPO="$RUN/models" \
    RS_OUT="$RUN/stress" RS_PID_FILE="$RUN/pids" \
    julia --threads="${RS_STRESS_THREADS:-8}" --project="$REPO_ROOT/packages/ReactantServerGateway" \
    "$REPO_ROOT/tools/residency_stress/stress.jl" 2>&1 | tee "$RUN/stress.log"
rc=${PIPESTATUS[0]}
set -e

curl -fsS "${METRICS%%,*}" > "$RUN/metrics-final.txt" 2>/dev/null || true
cleanup
SUP_PID=""
# Anything alarming the server logged, for the report.
grep -nE "Error|ERROR|error:|admission|failed|SIGBUS|segfault|signal \(|corrupt|overcommitted|reloaded from disk" "$RUN/worker.log" > "$RUN/worker-alerts.txt" || true
echo "stress exit $rc; alerts: $(wc -l < "$RUN/worker-alerts.txt") lines in $RUN/worker-alerts.txt"
exit "$rc"
