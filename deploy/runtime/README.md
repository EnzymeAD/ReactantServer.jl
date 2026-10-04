# Container runtime scripts

The entrypoints and healthchecks the node image runs. `//deploy:app_layer` installs them in
`/usr/local/bin`; the image's entrypoint is `tini` running `entrypoint.node.sh`. For the build itself
see [`deploy/README.md`](../README.md), and for running the image see the repository root's
`docker-compose.yml`.

- `entrypoint.node.sh`: the supervisor entrypoint (GPU-reclaim gate, then `ReactantServerNode.main`).
  The supervisor detects every GPU granted to the container, spawns one single-GPU worker per
  device, runs the embedded gateway with two or more workers (or binds a lone worker to the public
  ports directly), and multiplexes all logs onto stdout with `[worker0]` / `[gateway]` prefixes.
- `entrypoint.worker.sh`: single-worker escape hatch with no supervisor (`--entrypoint`), serving
  the worker named by `REACTANT_WORKER_NAME`.
- `healthcheck.node.sh`: role-aware healthcheck. With a gateway it curls `/readyz` on the admin port;
  with `REACTANT_ROLE=workers` it runs `healthcheck.worker.jl`, which reads the node file and reports
  ready when at least one worker answers ServerReady. The image has no baked `HEALTHCHECK` (an OCI
  image configuration has no such field), so the compose file sets it.

The node config baked at `/etc/reactantserver/node.yaml` comes from `config/node.default.yaml`; the
commented template is `config/node.yaml`. Mount your own file over that path to override.

## Autotune knobs (env)

Settable in the compose `environment:` (or `docker run -e`), via the `INFERENCE_SERVER_*` config
overrides:

- `INFERENCE_SERVER_RUNTIME_AUTOTUNE` (default `true`): `false` compiles with
  `xla_gpu_autotune_level=0` (deterministic gemm/conv selection, no timing trials, cleaner startup
  memory probe).
- `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE` (default inherits `LocalPreferences.toml`, i.e. enabled):
  toggle the persistent per-fusion autotune cache.
- `INFERENCE_SERVER_RUNTIME_AUTOTUNE_CACHE_DIR` (default `/var/cache/reactant-compile`): where the
  autotune cache lives; the compose file mounts a named volume there so it persists.
