#!/usr/bin/env bash
# Host-side resource monitor for the single-GPU soak. The worker exposes no HTTP metrics, so GPU
# memory and container RSS are observed from the host. Appends a timestamped CSV row per interval
# combining nvidia-smi (GPU 2) with docker stats for the worker container. Over a flat request mix,
# a steadily rising gpu_mem_used_mib or worker_mem indicates a leak.
#
# Usage:
#   docker/monitor_gpu2.sh [CSV_OUT] [INTERVAL_SECONDS] [GPU_INDEX]
# Defaults: CSV_OUT=soak_monitor.csv  INTERVAL_SECONDS=15  GPU_INDEX=2
set -euo pipefail

CSV_OUT="${1:-soak_monitor.csv}"
INTERVAL="${2:-15}"
GPU_INDEX="${3:-2}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.gpu2.yml}"

# Resolve the worker container id from the compose project, falling back to a name filter.
worker_cid() {
    docker compose -f "$COMPOSE_FILE" ps -q worker0 2>/dev/null \
        || docker ps --filter "name=worker0" --format '{{.ID}}' | head -n1
}

if [ ! -f "$CSV_OUT" ]; then
    echo "epoch,iso_time,gpu_mem_used_mib,gpu_util_pct,worker_mem,worker_cpu_pct,shm_regions" > "$CSV_OUT"
fi

echo "monitoring GPU $GPU_INDEX + worker0 every ${INTERVAL}s -> $CSV_OUT (Ctrl-C to stop)"
while true; do
    epoch="$(date +%s)"
    iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null | tr -d ' ' || echo 'NA,NA')"
    gpu_mem="${gpu%%,*}"
    gpu_util="${gpu##*,}"

    cid="$(worker_cid || true)"
    if [ -n "${cid:-}" ]; then
        stats="$(docker stats --no-stream --format '{{.MemUsage}};{{.CPUPerc}}' "$cid" 2>/dev/null || echo 'NA;NA')"
        wmem="$(echo "$stats" | cut -d';' -f1 | cut -d'/' -f1 | tr -d ' ')"
        wcpu="$(echo "$stats" | cut -d';' -f2 | tr -d ' %')"
    else
        wmem="NA"; wcpu="NA"
    fi

    # Count POSIX shm regions created by the SHM transport (should stay bounded; a climb is a leak).
    shm="$(ls /dev/shm 2>/dev/null | grep -c -E 'reactant|kserve|InferenceBufferPool' || true)"

    echo "$epoch,$iso,$gpu_mem,$gpu_util,$wmem,$wcpu,$shm" >> "$CSV_OUT"
    sleep "$INTERVAL"
done
