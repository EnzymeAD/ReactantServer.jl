#!/usr/bin/env bash
# Run one worked example the way its tutorial page tells a reader to, on the CPU:
#
#     bash integration/run_example.sh object_detection
#     bash integration/run_example.sh transformers
#
# Steps, matching docs/src/tutorials/<example>.md: instantiate the example's three environments,
# export its bundles, serve them with --cpu in the background, run its client, and stop the server.
# The export downloads pretrained weights and provisions torch through CondaPkg, so this needs
# network and several GB of disk; CI runs it only under the `ci:tutorials-full` label.
set -euo pipefail

EXAMPLE="${1:?usage: run_example.sh object_detection|transformers}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DIR="examples/$EXAMPLE"

case "$EXAMPLE" in
    object_detection)
        EXPORT="$DIR/export/export.jl"; SERVE="$DIR/server/serve.jl"; CLIENT="$DIR/client/detect.jl"
        PORT_VAR=OD_PORT ;;
    transformers)
        EXPORT="$DIR/export/export_stablehlo.jl"; SERVE="$DIR/server/serve.jl"; CLIENT="$DIR/client/query.jl"
        PORT_VAR=TX_PORT ;;
    *)
        echo "unknown example: $EXAMPLE" >&2
        exit 2 ;;
esac

PORT="${!PORT_VAR:-8080}"
export "$PORT_VAR=$PORT"
READY_TIMEOUT="${EXAMPLE_READY_TIMEOUT:-3600}"
LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$LOG"
}
trap cleanup EXIT

echo "== [1/4] instantiating the export, server, and client environments =="
for env in export server client; do
    julia --project="$DIR/$env" -e 'using Pkg; Pkg.instantiate()'
done

echo "== [2/4] exporting =="
julia --project="$DIR/export" "$EXPORT"

echo "== [3/4] serving on the CPU (port $PORT) =="
julia --project="$DIR/server" "$SERVE" --cpu > "$LOG" 2>&1 &
SERVER_PID=$!
# The server opens its gRPC port only after every model has compiled, so an accepted connection
# means it is ready.
deadline=$((SECONDS + READY_TIMEOUT))
until (echo > "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null; do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
        echo "ERROR: the server exited before it was ready; log:"
        tail -80 "$LOG"
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "ERROR: the server was not ready within ${READY_TIMEOUT}s; log:"
        tail -80 "$LOG"
        exit 1
    fi
    sleep 5
done
echo "   ready after ${SECONDS}s."

echo "== [4/4] running the client =="
if ! julia --project="$DIR/client" "$CLIENT"; then
    echo "ERROR: the client failed; server log:"
    tail -80 "$LOG"
    exit 1
fi

echo "$EXAMPLE: PASS"
