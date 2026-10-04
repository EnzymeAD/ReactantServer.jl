#!/usr/bin/env bash
# Check a loaded node image (bazel run //deploy:image_load first):
#
#   1. every entry project loads its packages from the baked caches, with nothing precompiled and
#      no cache rejected, which is what makes a container's first start fast;
#   2. the programs the entrypoint and healthcheck call are present and runnable.
#
# Runs without a GPU: the CUDA base's driver stub stands in for libcuda.so.1, which the CUDA build
# of libReactantExtra.so needs merely to load.
#
#   bazel run //deploy:image_check
#   bazel run //deploy:image_check -- localhost/other:tag
set -euo pipefail

image="${1:-$RS_IMAGE}"
fail=0
echo "==> checking $image"

podman image exists "$image" || { echo "FAILED: $image is not loaded (bazel run //deploy:image_load)" >&2; exit 1; }

# What each entry project loads. The root project serves the worker-role healthcheck, which
# imports only these two (deploy/runtime/healthcheck.worker.jl).
modules_for() {
    case "$1" in
        .) echo "gRPCClient YAML" ;;
        *) basename "$1" ;;
    esac
}

for p in $RS_PROJECTS; do
    mods="$(modules_for "$p")"
    log="$(podman run --rm --entrypoint bash -e JULIA_DEBUG=loading "$image" -c '
        set -e
        mkdir -p /tmp/stub && ln -s "$1" /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH="/tmp/stub:$LD_LIBRARY_PATH"
        cd "$2/$3"
        mods="$(echo $4 | tr " " ",")"
        julia --project=. -e "t = @elapsed(@eval using $mods); println(\"loaded in \", round(t; digits = 1), \" s\")"
    ' bash "$RS_STUB" "$RS_APP" "$p" "$mods" 2>&1)" || { echo "FAILED: $p: loading $mods failed"; echo "$log" | tail -20; fail=1; continue; }
    # A rejection "since the flags are mismatched" is Julia passing over a cache built for other
    # compiler settings: its bundled stdlib caches ship a debug-build variant beside the one that
    # loads. Any other rejection means a cache was stale, and any precompile means one was missing.
    bad="$(printf '%s\n' "$log" | grep -E 'Rejecting cache file|Precompiling|Being precompiled' |
        grep -v 'since the flags are mismatched' || true)"
    if [ -n "$bad" ]; then
        echo "FAILED: $p: loading $mods compiled or rejected caches:"
        printf '%s\n' "$bad" | sed 's/^/    /' | cut -c1-240 | head -20
        fail=1
    else
        echo "ok: $p: $mods from baked caches, $(printf '%s\n' "$log" | grep '^loaded in' | tail -1)"
    fi
done

tools="$(podman run --rm --entrypoint bash "$image" -c '
    set -e
    /usr/bin/tini --version
    curl --version | head -1
    for f in entrypoint.node.sh entrypoint.worker.sh healthcheck.node.sh healthcheck.worker.jl; do
        test -e "/usr/local/bin/$f" || { echo "missing /usr/local/bin/$f"; exit 1; }
    done
    test -f /etc/reactantserver/node.yaml || { echo "missing /etc/reactantserver/node.yaml"; exit 1; }
    echo "entrypoints, healthchecks and /etc/reactantserver/node.yaml present"
' 2>&1)" && printf '%s\n' "$tools" | sed 's/^/ok: /' || { echo "FAILED: tools: $tools"; fail=1; }

[ "$fail" -eq 0 ] && echo "==> $image passed" || { echo "==> $image FAILED" >&2; exit 1; }
