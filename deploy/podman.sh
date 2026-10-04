#!/usr/bin/env bash
# The loader for //deploy:image_load. rules_oci probes `command -v docker` before podman, and on a
# host where a docker CLI is installed but its socket is unreachable that probe wins and the load
# fails, so the loader is named rather than discovered.
exec podman "$@"
