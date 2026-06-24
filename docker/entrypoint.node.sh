#!/usr/bin/env bash
# Launch the node supervisor: detect the visible GPUs, spawn one ReactantServer worker
# subprocess per device plus the embedded gateway (role `all`, the default), multiplex their
# output onto this container's stdout with [name] line prefixes, and restart children that die.
# Roles (REACTANT_ROLE): all | workers | gateway. The node file (REACTANT_NODE_FILE, default
# /etc/reactantserver/node.yaml) needs no workers list; one is synthesized per detected GPU.
set -euo pipefail

# On a container recreate the previous node's CUDA allocations are freed asynchronously by the
# driver, so a fast restart can init CUDA and grab its (preallocated) BFC arena while several GB are
# still held by the dying process. That squeezes the out-of-arena headroom CUDA command buffers need
# and surfaces as an intermittent startup OOM. Poll until every visible GPU has drained below a
# used-memory threshold before launching the workers. A clean boot passes immediately; a timeout
# proceeds anyway (with a warning) so a genuine co-tenant never deadlocks startup. Tunables:
#   REACTANT_GPU_RECLAIM_TIMEOUT_SECONDS  (default 30; 0 disables the wait)
#   REACTANT_GPU_RECLAIM_MAX_USED_PERCENT (default 15; per-GPU used-memory ceiling treated as clear)
wait_for_gpu_reclaim() {
    local timeout="${REACTANT_GPU_RECLAIM_TIMEOUT_SECONDS:-30}"
    local max_used_pct="${REACTANT_GPU_RECLAIM_MAX_USED_PERCENT:-15}"
    [[ "$timeout" =~ ^[0-9]+$ ]] && [ "$timeout" -gt 0 ] || return 0
    # Without nvidia-smi the gate cannot observe memory. Warn only when GPUs are actually present
    # (device nodes exist) but the tool is missing , the real misconfig, usually a missing "utility"
    # driver capability. Stay silent on a genuinely CPU-only node.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if compgen -G "/dev/nvidia[0-9]*" >/dev/null 2>&1; then
            echo "[entrypoint] WARNING: GPUs present but nvidia-smi not found; reclaim wait skipped and out-of-pool metrics unavailable. Set NVIDIA_DRIVER_CAPABILITIES=compute,utility."
        fi
        return 0
    fi

    local deadline=$(( $(date +%s) + timeout ))
    while :; do
        local busy=0 idx used total pct
        while IFS=',' read -r idx used total; do
            idx="${idx// /}"; used="${used// /}"; total="${total// /}"
            [[ "$total" =~ ^[0-9]+$ ]] && [ "$total" -gt 0 ] || continue
            pct=$(( used * 100 / total ))
            if [ "$pct" -gt "$max_used_pct" ]; then
                busy=1
                echo "[entrypoint] GPU ${idx}: ${used}/${total} MiB used (${pct}%); waiting for prior-process reclamation"
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)

        [ "$busy" -eq 0 ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[entrypoint] WARNING: GPU memory still occupied after ${timeout}s; starting anyway (co-tenant, or slow reclamation)"
            return 0
        fi
        sleep 2
    done
}

wait_for_gpu_reclaim

# --handle-signals=no lets the supervisor's own SIGTERM/SIGINT handler run (Julia's default
# runtime handling would otherwise consume SIGTERM and die without shutting children down).
exec julia --handle-signals=no --project=/opt/reactantserver/packages/ReactantServerNode -e '
    using ReactantServerNode
    ReactantServerNode.main()
'
