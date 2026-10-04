# Docker (container) deployment

An alternative to the native launcher: one container runs the whole node. The image's entrypoint is
the supervisor (`ReactantServerNode`), which detects every GPU granted to the container, spawns one
single-GPU worker per device, runs the embedded gateway with two or more workers (or binds a lone
worker to the public ports directly), and multiplexes all logs onto stdout with `[worker0]` /
`[gateway]` prefixes. The external interface is the same as the native path: KServe V2 gRPC on
`:8001`, health/metrics on `:8002`.

The image is built with Bazel from committed locks; see [`deploy/`](../deploy/README.md) for the
build, the base image, and how to run it. This directory holds the runtime scripts that image runs,
and the repository root's `docker-compose.yml` runs it:

```
bazel run //deploy:image_load    # loads localhost/reactantserver:bazel
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

## Files

- `entrypoint.node.sh` — the supervisor entrypoint (GPU-reclaim gate + `ReactantServerNode.main`).
- `entrypoint.worker.sh` — single-worker escape hatch (`REACTANT_WORKER_NAME`).
- `healthcheck.node.sh` / `healthcheck.worker.jl` — role-aware container healthcheck.
- The node config baked at `/etc/reactantserver/node.yaml` comes from `config/node.default.yaml`;
  the commented template is `config/node.yaml`. Mount your own file over that path to override.

## Autotune knobs (env)

Settable in the compose `environment:` (or `docker run -e`), via the `INFERENCE_SERVER_*` config
overrides:

- `INFERENCE_SERVER_RUNTIME_AUTOTUNE` (default `true`) — `false` compiles with
  `xla_gpu_autotune_level=0` (deterministic gemm/conv selection, no timing trials, cleaner startup
  memory probe).
- `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE` (default inherits `LocalPreferences.toml`, i.e. enabled)
  — toggle the persistent per-fusion autotune cache.
- `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE_DIR` (default `/var/cache/reactant-compile`) — where the
  autotune cache lives; the compose file mounts a named volume there so it persists.

## Notes

- Requires the NVIDIA Container Toolkit; `docker compose up` grants all GPUs. `ipc: host` makes the
  KServe system-shared-memory regions visible to the workers.
- First startup is slow: every model compiles to a device executable before the gRPC plane serves.
  The compose healthcheck's `start_period` covers this; raise it for large model sets.
- The image has no baked `HEALTHCHECK` (an OCI image configuration has no such field), so the
  compose file sets one; pass `--health-cmd /usr/local/bin/healthcheck.node.sh` to `docker run`.
- Mount the model repository writable: compiled executables are cached under each bundle's
  `.cache/`, so a read-only mount recompiles every program on every start.
