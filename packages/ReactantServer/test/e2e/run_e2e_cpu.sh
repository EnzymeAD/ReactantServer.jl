#!/usr/bin/env bash
# CPU end-to-end test of the supervised single-container layout, with no containers: the
# ReactantServerNode supervisor spawns two CPU workers plus the embedded gateway as host
# subprocesses, the client drives scale4 through the gateway, and SIGTERM drains the tree down.
# Verifies worker synthesis (no workers list in the node file), the embedded gateway, prefixed log
# multiplexing, the graceful drain (with requests held in flight (slow4), the gateway and then the
# workers refuse new requests with UNAVAILABLE, finish the admitted ones, and every process exits
# 0), and what a client is told of an INTERNAL error in each `grpc.error_detail` mode (a second
# node runs with `full`), and, on that node, a client that registers shared memory while a worker is
# out (stopped and resumed, then killed and restarted), shared-memory outputs under client give-ups
# and a worker restart under one client: the worker is stopped (SIGSTOP), so the supervisor's liveness
# check has to find it and kill it; and, on a third node whose gateway runs lpt_packing, that the node
# recovers when every worker goes out at once (stopped and resumed together, stopped and resumed one
# at a time, killed and restarted). Needs no GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
E2E_DIR="packages/ReactantServer/test/e2e"

WORK="$(mktemp -d)"
SUP_PID=""
SHM_CLIENT_PID=""
OUTAGE_CLIENT_PID=""
ALL_OUT_CLIENT_PID=""
STOPPED_PIDS="" # workers step 11 holds stopped (SIGSTOP) right now
cleanup() {
    for pid in "$SHM_CLIENT_PID" "$OUTAGE_CLIENT_PID" "$ALL_OUT_CLIENT_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # A stopped worker cannot drain; resume it first, so the supervisor's drain below finishes.
    for pid in $STOPPED_PIDS; do
        kill -CONT "$pid" 2>/dev/null || true
    done
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

echo "== [1/11] generating scale4 bundle =="
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

echo "== [2/11] writing node file, gateway config and Reactant preference override =="
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
# env-configured gateway, except rapid_reset_protection, which also reaches a file that does not set
# it), so it also states the node's `grpc.error_detail` (argument 2).
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

echo "== [3/11] starting supervisor (2 CPU workers + embedded gateway) =="
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

echo "== [4/11] running CPU e2e client =="
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

# Once ready, each worker's liveness is checked on its own /healthz, and the gateway's on its admin
# listener (the default 10 s interval).
for w in worker0 worker1 gateway; do
    deadline=$((SECONDS + 60))
    until grep -q "^\[supervisor\] $w is ready; checking its liveness" "$WORK/supervisor.log"; do
        if ((SECONDS >= deadline)); then
            echo "ERROR: the supervisor never started checking $w's liveness; supervisor lines:"
            grep "^\[supervisor\]" "$WORK/supervisor.log" | tail -40
            exit 1
        fi
        sleep 1
    done
done
if grep -q "^\[supervisor\] .*failed a liveness check" "$WORK/supervisor.log"; then
    echo "ERROR: a healthy child failed a liveness check:"
    grep "^\[supervisor\] .*liveness" "$WORK/supervisor.log"
    exit 1
fi
echo "   the supervisor checks both workers' and the gateway's liveness, and all answer."

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

echo "== [5/11] SIGTERM under load: the node drains =="
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

echo "== [6/11] exit codes =="
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
    # (Step 8's liveness kill of a stopped worker is deliberate, not part of the shutdown.)
    if [[ -n "$(grep "^\[supervisor\] .*sending SIG" "$log" | grep -v "liveness checks in a row" || true)" ]]; then
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

echo "== [7/11] full INTERNAL error detail (node global.grpc.error_detail: full) =="
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
# Liveness thresholds short enough for step 9 to see a stopped worker killed within about 20 s, and
# a gateway probe every second, so step 8 sees a stopped worker marked down within about 6 s (the
# 5 s probe deadline), well before its liveness check would kill it.
REACTANT_LIVENESS_INTERVAL_SECONDS=2 \
REACTANT_LIVENESS_TIMEOUT_SECONDS=5 \
REACTANT_LIVENESS_FAILURES=3 \
REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS=1 \
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

echo "== [8/11] shared memory: a client registers while a worker is out, then the worker returns =="
# The gateway registers a region on its healthy workers only. Phase 1 stops worker1 (SIGSTOP: alive,
# answering nothing), waits for the gateway to mark it down, lets a fresh client pool register, and
# resumes it (SIGCONT) well before its liveness check would kill it. Phase 2 kills worker1 instead
# and waits for the supervisor to restart it, which loses every registration. Each time the client's
# requests that reach worker1 meet FAILED_PRECONDITION there, and the client registers again and
# retries, so it sees no error and stays on shared memory (client_shm_outage_cpu.jl).
OUTAGE="$WORK/outage"
mkdir -p "$OUTAGE"
log="$WORK/supervisor_full.log"
julia --project=packages/ReactantServerClient --threads=2 "$E2E_DIR/client_shm_outage_cpu.jl" \
    "$GW_PORT" "$BASE_PORT" "$((BASE_PORT + 1))" "$OUTAGE" > "$WORK/shm_outage_client.log" 2>&1 &
OUTAGE_CLIENT_PID=$!
await_marker() { # await_marker <name>: the outage client's marker file, failing if the client exits
    until [[ -f "$OUTAGE/$1" ]]; do
        if ! kill -0 "$OUTAGE_CLIENT_PID" 2>/dev/null; then
            echo "ERROR: the outage client exited while this step waited for '$1'; its log:"
            tail -60 "$WORK/shm_outage_client.log"
            exit 1
        fi
        sleep 0.2
    done
}
worker1_ready() { # 1 or 0, as the gateway reports worker1
    curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/metrics" 2>/dev/null |
        awk '/^gateway_worker_ready\{worker="worker1"\} / { print ($2 + 0) }'
}
await_worker1() { # await_worker1 <1|0>
    local deadline=$((SECONDS + READY_TIMEOUT))
    until [[ "$(worker1_ready)" == "$1" ]]; do
        if ((SECONDS >= deadline)); then
            echo "ERROR: the gateway never reported worker1 ready=$1"
            exit 1
        fi
        sleep 0.2
    done
}
worker1_pid() { grep -oE 'started worker1 \(pid [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$'; }
await_marker warm
w1_pid="$(worker1_pid)"
liveness_kills="$(grep -c 'worker1 failed 3 liveness checks' "$log" || true)"
echo "   phase 1: stopping worker1 (pid $w1_pid) with SIGSTOP while a client registers"
kill -STOP "$w1_pid"
t_stop=$SECONDS
await_worker1 0
touch "$OUTAGE/down1"
await_marker registered1
kill -CONT "$w1_pid"
echo "   the client registered with worker1 out; worker1 resumed after $((SECONDS - t_stop)) s"
await_worker1 1
if [[ "$(worker1_pid)" != "$w1_pid" ]] ||
    (($(grep -c 'worker1 failed 3 liveness checks' "$log" || true) != liveness_kills)); then
    echo "ERROR: worker1 was restarted during phase 1, which needs it to stay alive"
    exit 1
fi
touch "$OUTAGE/back1"
await_marker done1
echo "   phase 2: killing worker1 (pid $w1_pid) with SIGKILL while a client registers"
kill -KILL "$w1_pid"
await_worker1 0
touch "$OUTAGE/down2"
await_marker registered2
deadline=$((SECONDS + READY_TIMEOUT))
until new_pid="$(worker1_pid)" && [[ -n "$new_pid" && "$new_pid" != "$w1_pid" ]]; do
    if ((SECONDS >= deadline)); then
        echo "ERROR: worker1 was not restarted within ${READY_TIMEOUT}s"
        exit 1
    fi
    sleep 1
done
await_worker1 1
echo "   worker1 restarted (pid $new_pid) and is ready again"
touch "$OUTAGE/back2"
set +e
wait "$OUTAGE_CLIENT_PID"
outage_rc=$?
set -e
grep -E "^(PASS|FAIL|CPU e2e)" "$WORK/shm_outage_client.log" || true
if ((outage_rc != 0)); then
    echo "== the outage client failed (rc=${outage_rc}); its log and recent supervisor log: =="
    tail -60 "$WORK/shm_outage_client.log"
    tail -120 "$log"
    exit "$outage_rc"
fi

echo "== [9/11] shared memory: short deadlines, then a worker restart under the same client =="
# Phase A drives shared-memory outputs with deadlines short enough that clients give up while
# requests run (CORE-3). Then worker1 is stopped (SIGSTOP: alive, but answering nothing, as a hung
# worker); the supervisor's liveness check kills it and the supervisor restarts it. Once the
# gateway reports it ready again, phase B checks that the same client process re-registers the
# same region and keeps getting correct answers (CORE-2), each within a few seconds although phase
# A gave up on about a thousand requests over its connection to the gateway (GW-18). It runs on
# this second node so that any failure reaches the client with its full error text.
mkdir -p "$WORK/markers"
julia --project=packages/ReactantServerClient --threads=4 "$E2E_DIR/client_shm_cpu.jl" \
    "$GW_PORT" "$BASE_PORT" "$WORK/markers" > "$WORK/shm_client.log" 2>&1 &
SHM_CLIENT_PID=$!
until [[ -f "$WORK/markers/phase_a_done" ]]; do
    if ! kill -0 "$SHM_CLIENT_PID" 2>/dev/null; then
        echo "ERROR: shared-memory client exited during phase A; its log:"
        tail -60 "$WORK/shm_client.log"
        exit 1
    fi
    sleep 1
done
log="$WORK/supervisor_full.log"
old_pid="$(grep -oE 'started worker1 \(pid [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')"
seen_ready="$(grep -c "health: worker ready again" "$log" || true)"
echo "   stopping worker1 (pid $old_pid) with SIGSTOP; the supervisor's liveness check should kill it"
kill -STOP "$old_pid"
deadline=$((SECONDS + READY_TIMEOUT))
until new_pid="$(grep -oE 'started worker1 \(pid [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')" &&
    [[ -n "$new_pid" && "$new_pid" != "$old_pid" ]] &&
    (($(grep -c "health: worker ready again" "$log" || true) > seen_ready)); do
    if ((SECONDS >= deadline)); then
        echo "ERROR: worker1 did not come back within ${READY_TIMEOUT}s"
        exit 1
    fi
    sleep 2
done
if ! grep -q "^\[supervisor\] worker1 failed 3 liveness checks in a row; sending SIGKILL" "$log"; then
    echo "ERROR: worker1 came back, but not through its liveness check; supervisor lines:"
    grep "^\[supervisor\]" "$log" | tail -40
    exit 1
fi
echo "   worker1 was killed by its liveness check, restarted (pid $new_pid), and the gateway reports it ready"
touch "$WORK/markers/restarted"
set +e
wait "$SHM_CLIENT_PID"
shm_rc=$?
set -e
grep -E "^(PASS|FAIL|phase|CPU e2e)" "$WORK/shm_client.log" || true
if ((shm_rc != 0)); then
    echo "== shared-memory client failed (rc=${shm_rc}); its log and recent supervisor log: =="
    tail -60 "$WORK/shm_client.log"
    tail -120 "$log"
    exit "$shm_rc"
fi

echo "== [10/11] a stopped gateway is killed by its liveness check and restarted =="
# A hung gateway takes the node's public endpoint down with it. Stopped (SIGSTOP), it answers
# nothing; the supervisor's check (2 s interval, 5 s timeout, 3 failures on this node) must kill it,
# and the restarted gateway must serve the node again.
old_gw="$(grep -oE 'started gateway \(pid [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')"
echo "   stopping the gateway (pid $old_gw) with SIGSTOP"
kill -STOP "$old_gw"
deadline=$((SECONDS + READY_TIMEOUT))
until new_gw="$(grep -oE 'started gateway \(pid [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')" &&
    [[ -n "$new_gw" && "$new_gw" != "$old_gw" ]] &&
    curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/readyz" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
        echo "ERROR: the gateway did not come back within ${READY_TIMEOUT}s; supervisor lines:"
        grep "^\[supervisor\]" "$log" | tail -40
        exit 1
    fi
    sleep 2
done
if ! grep -q "^\[supervisor\] gateway failed 3 liveness checks in a row; sending SIGKILL" "$log"; then
    echo "ERROR: the gateway came back, but not through its liveness check; supervisor lines:"
    grep "^\[supervisor\]" "$log" | tail -40
    exit 1
fi
echo "   the gateway was killed by its liveness check, restarted (pid $new_gw), and is ready again"

kill -TERM "$SUP_PID"
sup_rc=0
wait "$SUP_PID" || sup_rc=$?
check_clean_exit "$sup_rc" "$WORK/supervisor_full.log"

echo "== [11/11] every worker out at once: the node recovers =="
# A third node, whose gateway runs lpt_packing (which needs the workers' fifo discipline) with every
# model placed on both workers (default_replicas: all), probing every second as on the second node.
# The supervisor's liveness checks are off on it (liveness.failures: 0), so a worker stopped for
# longer than a check allows is never killed, and only the gateway reacts. Its gateway file lists no
# metrics_endpoints, so a /metrics scrape never waits for a stopped worker's.
#  - Phase A: every worker stopped (SIGSTOP). Once the gateway marks them all down, every model is
#    held: the gateway answers a request UNAVAILABLE without calling a worker. Every probe round then
#    times out, and after 3 such rounds (WEDGE_EXIT_ROUNDS) the gateway concludes its own gRPC client
#    is wedged and exits, and the supervisor restarts it. The workers stay stopped until that restart
#    (or a bound), and then resume together.
#  - Phase B: every worker stopped, then worker0 alone resumed, well before the wedge exit. The
#    membership repack places every model on worker0, which serves them alone; once worker1 resumes,
#    the next one places every model on both workers again.
#  - Phase C: every worker killed (SIGKILL). A refused connection does not count toward the wedge exit,
#    so the gateway stays up; the supervisor restarts the workers, and the client registers its
#    region on them again after their FAILED_PRECONDITION.
# The client (client_all_out_cpu.jl) checks each phase's baseline and recovery from its side, and this
# script stops, resumes and kills the workers and checks the gateway's side, through marker files.
t11="$(date +%s%3N)"
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"
sed -e "s/^base_port: .*/base_port: $BASE_PORT/" -e "s/^metrics_base_port: .*/metrics_base_port: $METRICS_BASE_PORT/" \
    "$WORK/node.yaml" > "$WORK/node_out.yaml"
printf '  scheduler:\n    discipline: fifo\n  grpc:\n    error_detail: full\nliveness:\n  failures: 0\n' >> "$WORK/node_out.yaml"
cat > "$WORK/gateway_out.yml" << EOF
listen:
  grpc: "127.0.0.1:$GW_PORT"
  metrics: "127.0.0.1:$GW_METRICS_PORT"
endpoints:
  - "127.0.0.1:$BASE_PORT"
  - "127.0.0.1:$((BASE_PORT + 1))"
worker_names: [worker0, worker1]
grpc:
  error_detail: full
scheduling:
  mode: lpt_packing
  default_replicas: all
EOF
log="$WORK/supervisor_out.log"
REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS=1 \
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_FILE="$WORK/gateway_out.yml" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --threads=1,0 --project=packages/ReactantServerNode -e '
    using ReactantServerNode
    exit(ReactantServerNode.supervise(ARGS[1]; runtime_dir = ARGS[2]))
' "$WORK/node_out.yaml" "$WORK/run_out" > "$log" 2>&1 &
SUP_PID=$!
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/readyz" >/dev/null 2>&1; do
    if ! kill -0 "$SUP_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        echo "ERROR: the lpt_packing node did not become ready; log:"
        tail -80 "$log"
        exit 1
    fi
    sleep 2
done

OUT="$WORK/all_out"
mkdir -p "$OUT"
CLIENT_LOG="$WORK/all_out_client.log"
FULL_PLACEMENT="scale4:worker0 scale4:worker1 slow4:worker0 slow4:worker1"
WORKER0_PLACEMENT="scale4:worker0 slow4:worker0"
now_ms() { date +%s%3N; }
since() { awk -v a="$1" -v b="$(now_ms)" 'BEGIN { printf "%.1f", (b - a) / 1000 }'; }
gw_metrics() { curl -fsS --max-time 5 "http://127.0.0.1:${GW_METRICS_PORT}/metrics" 2>/dev/null || true; }
readyz_code() { curl -s -o /dev/null --max-time 5 -w '%{http_code}' "http://127.0.0.1:${GW_METRICS_PORT}/readyz" 2>/dev/null || true; }
workers_ready() { # "<worker0> <worker1>": 1 or 0 each, as the gateway last probed them
    gw_metrics | awk '/^gateway_worker_ready\{worker="worker0"\} / { a = $2 + 0 }
        /^gateway_worker_ready\{worker="worker1"\} / { b = $2 + 0 } END { print a " " b }'
}
placement() { # the model:worker pairs of the gateway's placement (weight above 0), sorted
    gw_metrics | sed -nE 's/^gateway_placement_weight\{model="([^"]+)",worker="([^"]+)"\} (.+)$/\1:\2 \3/p' |
        awk '$2 + 0 > 0 { print $1 }' | sort | paste -sd ' ' -
}
membership_repacks() { gw_metrics | awk '/^gateway_repacks_total\{trigger="membership"\} / { n = $2 + 0 } END { print n + 0 }'; }
routed_to() { # routed_to <model> <worker>: requests the gateway routed to that replica since it was placed
    gw_metrics | awk -v k="gateway_replica_routed_total{model=\"$1\",worker=\"$2\"}" '$1 == k { n = $2 + 0 } END { print n + 0 }'
}
pid_of() { grep -oE "started $1 \(pid [0-9]+" "$log" | tail -1 | grep -oE '[0-9]+$' || true; }
gateway_starts() { grep -c "^\[supervisor\] started gateway (pid" "$log" || true; }
log_lines() { wc -l < "$log"; }
# Predicates for `await`.
all_down() { [[ "$(readyz_code)" == 503 && "$(workers_ready)" == "0 0" ]]; }
all_up() { [[ "$(readyz_code)" == 200 && "$(workers_ready)" == "1 1" ]]; }
workers_are() { [[ "$(workers_ready)" == "$1" ]]; }
placed() { [[ "$(placement)" == "$1" ]]; }
restarted() { # restarted <worker> <old pid>
    local p
    p="$(pid_of "$1")"
    [[ -n "$p" && "$p" != "$2" ]]
}
marker() { # the client's marker file; failing the step if the client exits first
    [[ -f "$OUT/$1" ]] && return 0
    kill -0 "$ALL_OUT_CLIENT_PID" 2>/dev/null || out_fail "the client exited while the step waited for '$1'"
    return 1
}
out_fail() { # out_fail <why>: fail the step with the supervisor's, the gateway's and the client's view
    echo "FAIL  step 11: $1"
    echo "-- supervisor and gateway lines:"
    grep -E "^\[(supervisor|gateway)\]" "$log" | tail -80
    echo "-- client log:"
    tail -40 "$CLIENT_LOG"
    exit 1
}
await() { # await <seconds> <what> <command...>: until the command succeeds, failing after <seconds>
    local limit="$1" what="$2"
    shift 2
    local deadline=$((SECONDS + limit))
    until "$@"; do
        ((SECONDS < deadline)) || out_fail "$what: not within ${limit}s"
        sleep 0.2
    done
}
client_lines() { grep -E "^(PASS|FAIL|INFO)  " "$CLIENT_LOG" | grep -F "  $1" || true; }
# A check the client fails fails its phase but not at once: the gateway has recovered by then (this
# script checked it), so the later phases still run, and the step fails once its node has stopped.
FAILED_PHASES=""
phase_end() { # phase_end <A|B|C> <title> <start, ms> <what passed>: the phase's PASS or FAIL line
    if grep -qE "^FAIL  phase $1[ (]" "$CLIENT_LOG"; then
        echo "FAIL  phase $1 ($2) in $(since "$3") s: the client failed the check(s) above"
        FAILED_PHASES="$FAILED_PHASES $1"
    else
        echo "PASS  phase $1 ($2) in $(since "$3") s: $4"
    fi
}

await 60 "both workers ready on the lpt_packing node" all_up
await 60 "the startup placement of every model on both workers" placed "$FULL_PLACEMENT"
julia --project=packages/ReactantServerClient --threads=2 "$E2E_DIR/client_all_out_cpu.jl" \
    "$GW_PORT" "$BASE_PORT" "$((BASE_PORT + 1))" "$OUT" > "$CLIENT_LOG" 2>&1 &
ALL_OUT_CLIENT_PID=$!
w0="$(pid_of worker0)"
w1="$(pid_of worker1)"

# Phase A: every worker stopped until the gateway restarts, then resumed together.
await "$READY_TIMEOUT" "the client's baseline before phase A" marker baseA
client_lines "phase A baseline"
tA=$(now_ms)
gw="$(pid_of gateway)"
starts="$(gateway_starts)"
from=$(($(log_lines) + 1))
echo "   phase A: stopping every worker (pids $w0 $w1) with SIGSTOP; the gateway is pid $gw"
STOPPED_PIDS="$w0 $w1"
kill -STOP "$w0" "$w1"
await 60 "the gateway reporting every worker down" all_down
echo "   the gateway reports every worker down after $(since "$tA") s"
touch "$OUT/downA"
await 60 "the client's requests while every worker is down" marker checkedA
held_lines="$(tail -n +"$from" "$log" | grep -c "no ready worker serves the model" || true)"
((held_lines >= 2)) || out_fail "the gateway logged no held-model answer for each model (found $held_lines)"
client_lines "phase A (every worker stopped)"
# Hold the stop past the wedge exit: about 3 rounds of a 5 s probe deadline, a 5 s discovery deadline
# and the 1 s interval, so about 35 s. The bound is far past that.
until (($(gateway_starts) > starts)) || (($(since "$tA" | cut -d. -f1) >= 90)); do
    sleep 0.2
done
stall="$(since "$tA")"
if (($(gateway_starts) > starts)); then
    echo "   the gateway restarted $stall s after the workers stopped (pid $gw -> $(pid_of gateway)); its last lines:"
    tail -n +"$from" "$log" | grep -E "^\[supervisor\] .*gateway|wedged" | head -5 | sed 's/^/     /'
else
    echo "   the gateway did not restart within $stall s of the workers stopping"
fi
kill -CONT "$w0" "$w1"
STOPPED_PIDS=""
echo "   every worker resumed"
await 300 "the gateway ready with both workers ready after phase A" all_up
await 60 "the placement of every model on both workers after phase A" placed "$FULL_PLACEMENT"
[[ "$(pid_of worker0)" == "$w0" && "$(pid_of worker1)" == "$w1" ]] ||
    out_fail "a worker was restarted in phase A (pids $w0 $w1 -> $(pid_of worker0) $(pid_of worker1))"
touch "$OUT/backA"
await 300 "the client's recovery after phase A" marker doneA
client_lines "phase A (every worker resumed together)"
gw_restarts=$(($(gateway_starts) - starts))
((gw_restarts <= 1)) || out_fail "the gateway restarted $gw_restarts times in phase A"
phase_end A "every worker stopped, then resumed together" "$tA" \
    "the gateway restarted $gw_restarts time(s), the workers kept their pids, every model serves on both workers through shared memory"

# Phase B: every worker stopped, then worker0 alone resumed, then worker1.
await 60 "the client's baseline before phase B" marker baseB
client_lines "phase B baseline"
tB=$(now_ms)
gw="$(pid_of gateway)"
starts="$(gateway_starts)"
membership="$(membership_repacks)"
echo "   phase B: stopping every worker with SIGSTOP; the gateway is pid $gw"
STOPPED_PIDS="$w0 $w1"
kill -STOP "$w0" "$w1"
await 60 "the gateway reporting every worker down" all_down
kill -CONT "$w0"
STOPPED_PIDS="$w1"
echo "   worker0 resumed $(since "$tB") s after the workers stopped"
await 60 "the gateway reporting worker0 ready and worker1 down" workers_are "1 0"
await 60 "a membership repack placing every model on worker0 alone" placed "$WORKER0_PLACEMENT"
(($(membership_repacks) > membership)) || out_fail "the placement changed without a membership repack"
touch "$OUT/oneB"
await 60 "the client's requests with worker0 alone" marker servedB
client_lines "phase B (worker0 resumed, worker1 still stopped)"
kill -CONT "$w1"
STOPPED_PIDS=""
echo "   worker1 resumed $(since "$tB") s after the workers stopped"
await 60 "the gateway ready with both workers ready after phase B" all_up
await 60 "a membership repack placing every model on both workers again" placed "$FULL_PLACEMENT"
(($(membership_repacks) >= membership + 2)) ||
    out_fail "expected two membership repacks in phase B, saw $(($(membership_repacks) - membership))"
touch "$OUT/backB"
await 300 "the client's recovery after phase B" marker doneB
client_lines "phase B (worker1 resumed too)"
for m in scale4 slow4; do
    (($(routed_to "$m" worker1) > 0)) || out_fail "no $m request was routed to worker1 after it returned"
done
[[ "$(pid_of gateway)" == "$gw" && "$(gateway_starts)" == "$starts" ]] ||
    out_fail "the gateway restarted in phase B (pid $gw -> $(pid_of gateway))"
phase_end B "every worker stopped, then resumed one at a time" "$tB" \
    "worker0 served every model alone, then both workers again, through $(($(membership_repacks) - membership)) membership repacks; the gateway kept pid $gw"

# Phase C: every worker killed and restarted by the supervisor.
await 60 "the client's baseline before phase C" marker baseC
client_lines "phase C baseline"
tC=$(now_ms)
gw="$(pid_of gateway)"
starts="$(gateway_starts)"
from=$(($(log_lines) + 1))
echo "   phase C: killing every worker (pids $w0 $w1) with SIGKILL; the gateway is pid $gw"
kill -KILL "$w0" "$w1"
await 60 "the gateway reporting every worker down" all_down
echo "   the gateway reports every worker down after $(since "$tC") s"
touch "$OUT/downC"
await 60 "the client's requests while every worker is down" marker checkedC
client_lines "phase C (every worker killed)"
await "$READY_TIMEOUT" "worker0 restarted" restarted worker0 "$w0"
await "$READY_TIMEOUT" "worker1 restarted" restarted worker1 "$w1"
await "$READY_TIMEOUT" "the gateway ready with both restarted workers ready" all_up
await 60 "the placement of every model on both workers after phase C" placed "$FULL_PLACEMENT"
echo "   both workers restarted (pids $(pid_of worker0) $(pid_of worker1)) and are ready after $(since "$tC") s"
if [[ "$(pid_of gateway)" != "$gw" || "$(gateway_starts)" != "$starts" ]]; then
    echo "   the gateway's lines in phase C:"
    tail -n +"$from" "$log" | grep -E "^\[(supervisor|gateway)\]" | tail -60
    out_fail "the gateway restarted in phase C (pid $gw -> $(pid_of gateway))"
fi
touch "$OUT/backC"
set +e
wait "$ALL_OUT_CLIENT_PID"
all_out_rc=$?
set -e
client_lines "phase C (every worker restarted)"
grep -E "^CPU e2e" "$CLIENT_LOG" || true
phase_end C "every worker killed and restarted" "$tC" \
    "the gateway kept pid $gw, and the client registered again and stays on shared memory"

kill -TERM "$SUP_PID"
sup_rc=0
wait "$SUP_PID" || sup_rc=$?
check_clean_exit "$sup_rc" "$log"
echo "   step 11 took $(since "$t11") s, starting its node included"
if [[ -n "$FAILED_PHASES" ]]; then
    echo "FAIL  step 11: phase(s)$FAILED_PHASES failed"
    exit 1
fi
if ((all_out_rc != 0)); then
    echo "ERROR: the client exited $all_out_rc with no failed check; its log:"
    tail -40 "$CLIENT_LOG"
    exit 1
fi

echo "CPU e2e: PASS (workers synthesized, gateway routed, graceful drain, clean exit, INTERNAL error detail, shared memory, a register during a worker outage, liveness restarts, every worker out at once)"
