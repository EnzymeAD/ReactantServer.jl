# Docker (container) deployment

An alternative to the native launcher: one container runs the whole node. The image's entrypoint is
the supervisor (`ReactantServerNode`), which detects every GPU granted to the container, spawns one
single-GPU worker per device, runs the embedded gateway with two or more workers (or binds a lone
worker to the public ports directly), and multiplexes all logs onto stdout with `[worker0]` /
`[gateway]` prefixes. The external interface is the same as the native path: KServe V2 gRPC on
`:8001`, health/metrics on `:8002`.

```
git submodule update --init lib/gRPCServer.jl
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 julia --project=. -e 'using Pkg; Pkg.instantiate()'  # generates Manifest.toml (gitignored)
make image        # or: docker build -f docker/Dockerfile -t reactantserver .
REACTANTSERVER_MODELS=/path/to/bundles docker compose up
```

## Why this image works (the nvJitLink fix)

The earlier image aborted at first GPU compile with `Invalid handle. Cannot load symbol
cublasLtCreate`. That was misleading: Reactant statically links cuBLASLt (the symbol is present).
The real gap was `libnvJitLink.so.13`, which CUDA-13 cuBLASLt `dlopen`s to JIT its GEMM kernels. It
is not in the Reactant artifact and not injected by the NVIDIA container runtime (which adds only
`libcuda.so.1`), so a bare Julia image had no nvJitLink and cuBLASLt's JIT bring-up failed.

This image is a multi-stage build on `julia:1.12.6-trixie` (Debian 13) that copies only
`libnvJitLink.so.13` (and a build-time `libcuda.so` stub) from an official CUDA 13.1 image and
registers it with the loader (`/opt/cuda-userspace/lib` via `ldconfig`). It deliberately does not
bring the rest of the CUDA userspace: Reactant resolves its own bundled cuDNN 9 / NCCL 2 via
`$ORIGIN`, and a system copy on `LD_LIBRARY_PATH` would shadow them with a mismatched version. The
CUDA base tag is the `CUDA_BASE_IMAGE` build arg (default `nvidia/cuda:13.1.0-devel-ubuntu24.04`).

## Files

- `Dockerfile` — the multi-stage node image (above).
- `entrypoint.node.sh` — the supervisor entrypoint (GPU-reclaim gate + `ReactantServerNode.main`).
- `entrypoint.worker.sh` — single-worker escape hatch (`REACTANT_WORKER_NAME`).
- `healthcheck.node.sh` / `healthcheck.worker.jl` — role-aware container healthcheck.
- `certs/` — drop a corporate CA here for TLS-inspecting proxies (see `certs/README.md`).
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
  The image `HEALTHCHECK` / compose `start_period` cover this; raise it for large model sets.
- `Manifest.toml` is gitignored, so generate it locally (command above) before `make image`.
