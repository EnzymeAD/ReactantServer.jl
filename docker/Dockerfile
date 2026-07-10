# ============================================================================================
# DEPRECATED — do not use for deployment. The supported path is the NATIVE launcher + systemd:
# see private/deploy/INSTALL.native.md. This image aborts at GPU-compile time inside the
# container with `Invalid handle. Cannot load symbol cublasLtCreate` + SIGABRT (a container
# CDI / cuBLASLt issue); running the same supervisor natively uses the host driver directly and
# does not hit it. Kept only for reference / a future container fix. NOTE: Manifest.toml is
# gitignored, so `COPY ... Manifest.toml` below needs a locally-generated Manifest in the build
# context (run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` first); a fresh clone has none.
# ============================================================================================
#
# ReactantServer.jl node image. The default entrypoint is the supervisor (ReactantServerNode):
# it detects the visible GPUs, runs one worker subprocess per device, and multiplexes their
# logs. With two or more workers it also runs an embedded gateway; with a single worker it binds
# that worker to the public ports directly. A multi-GPU deployment is just
#
#   docker run --gpus all --ipc=host -p 8001:8001 -p 8002:8002 \
#     -v /path/to/bundles:/var/lib/reactantserver/models:ro reactantserver
#
# The single-worker entrypoint.worker.sh escape hatch and the REACTANT_ROLE roles are documented
# in docker/README.md.
#
# Build context is the repository root. After the migration to a workspace monorepo, the only
# remaining git submodule is lib/gRPCServer.jl (Reactant, HTTP, Prometheus, gRPCClient are now
# registry deps); run `git submodule update --init lib/gRPCServer.jl` before building.
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t reactantserver .   # or: make image

FROM docker.io/library/julia:1.12.6-trixie

ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PROJECT=/opt/reactantserver \
    REACTANT_WORKSPACE_ROOT=/opt/reactantserver \
    JULIA_SSL_CA_ROOTS_PATH=/etc/ssl/certs/ca-certificates.crt

# git/CA certs for the General registry and artifact downloads during instantiate;
# build-essential (make + C toolchain) so packages with a native build step, e.g.
# InterProcessCommunication.jl, can compile their helpers; tini as PID 1 (zombie reaping and
# conventional signal semantics under the supervisor); curl for the gateway-backed healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates build-essential tini curl \
 && rm -rf /var/lib/apt/lists/*

# Corporate TLS-inspection (MITM) trust. On networks where an inspecting proxy re-signs HTTPS with
# an internal root CA, drop that root (PEM, .crt extension) into docker/certs/ and it is added to
# the system trust store here. JULIA_SSL_CA_ROOTS_PATH (below) then points Julia's NetworkOptions
# at the resulting system bundle, so both Downloads.jl/libcurl (registry + artifact downloads) and
# LibGit2 (the registry git clone fallback) accept the proxy's certificates. update-ca-certificates
# only processes *.crt, so the directory's README is ignored; with no cert present this is a no-op
# and the build behaves exactly as before.
COPY docker/certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

WORKDIR /opt/reactantserver

# Workspace root environment, the member packages, and the vendored gRPCServer submodule (the
# remaining forks resolve from the registries via the shared Manifest.toml). The worker is
# built on the workspace root; the entrypoint then runs on the ReactantServer member project.
# .dockerignore drops the package test trees (e.g. the large e2e weights) from the packages/ copy.
COPY Project.toml Manifest.toml ./
COPY packages/ ./packages/
COPY lib/gRPCServer.jl/ ./lib/gRPCServer.jl/

# CUDA artifact selection for Reactant_jll. Its platform augmentation honors the REACTANT_GPU
# and REACTANT_GPU_VERSION environment variables to choose the prebuilt artifact. Without them a
# GPU-less build selects the CPU (gpu=none) artifact and `client("cuda")` then fails at runtime
# with "No functional client was found". These are set as ENV (not ARG) so the same selection
# applies at build time (artifact download + precompile of the CUDA libReactantExtra) and at
# runtime (load-time wrapper selection in the worker). The value must match the host CUDA major
# version: Reactant ships 12.9 and 13.1 builds, and the bundled cuBLASLt/cuDNN must match the
# host driver's CUDA runtime or symbol loading (e.g. cublasLtCreate) aborts during GPU compile.
# This box has the CUDA 13.1 toolkit (driver 580 reports max CUDA 13.0), so 13.1 is the correct
# choice; use 12.9 on CUDA 12 hosts.
ARG XLA_CUDA_GPU_VERSION=13.1
ENV REACTANT_GPU=cuda \
    REACTANT_GPU_VERSION=${XLA_CUDA_GPU_VERSION}

# Install the General registry (the custom JULIA_DEPOT_PATH starts empty, and resolve does not
# auto-install it), then resolve, instantiate, and precompile the whole workspace so the worker
# starts fast (no first-request precompile). The workspace [sources] (in the copied root
# Project.toml) already point the vendored deps at lib/ and ReactantServerCore at packages/.
#
# Precompiling the CUDA build loads libReactantExtra, whose DT_NEEDED list includes
# libcuda.so.1 (the NVIDIA *driver*, injected by the container runtime at run time and absent in
# `podman build`, since --device does not accept CDI names). The bundled cuBLAS/cuDNN/cudart come
# from the artifact; only libcuda is missing. Precompilation never calls into the driver (the GPU
# client is created lazily at serve time), so a do-nothing stub libcuda.so.1 satisfies the
# loader's NEEDED dependency and lets precompilation complete. The stub is exposed only via
# LD_LIBRARY_PATH for the precompile command and removed afterwards, so at run time the real
# CDI-injected libcuda.so.1 is used, not the stub.
RUN mkdir -p /tmp/cudastub && : > /tmp/empty.c \
 && gcc -shared -fPIC -Wl,-soname,libcuda.so.1 -o /tmp/cudastub/libcuda.so.1 /tmp/empty.c \
 && LD_LIBRARY_PATH=/tmp/cudastub julia --project=/opt/reactantserver -e 'using Pkg; Pkg.Registry.add("General"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' \
 && rm -rf /tmp/cudastub /tmp/empty.c

# entrypoint.node.sh is the default (supervisor: workers + embedded gateway);
# entrypoint.worker.sh remains as the single-worker escape hatch for per-GPU-container layouts.
COPY docker/entrypoint.node.sh   /usr/local/bin/entrypoint.node.sh
COPY docker/entrypoint.worker.sh /usr/local/bin/entrypoint.worker.sh
RUN chmod +x /usr/local/bin/entrypoint.node.sh /usr/local/bin/entrypoint.worker.sh

# Healthchecks: healthcheck.node.sh dispatches on REACTANT_ROLE (curl on the gateway's /readyz
# for all/gateway; the pure-Julia ServerReady probe, which imports only gRPCClient and YAML, for
# workers).
COPY docker/healthcheck.node.sh   /usr/local/bin/healthcheck.node.sh
COPY docker/healthcheck.worker.jl /usr/local/bin/healthcheck.worker.jl
RUN chmod +x /usr/local/bin/healthcheck.node.sh

# Baked default node config: gpus auto, no workers list (synthesized per detected GPU). Mount
# your own file over this path to customize.
COPY docker/node.default.yaml /etc/reactantserver/node.yaml

# 8001/8002: the embedded gateway's client gRPC and metrics/health admin ports (the default
# all-in-one role). In the workers role the per-worker ports (base_port+i, default 8080..) and
# metrics ports (9100..) are bound instead; publish the range you configured. No system CUDA
# toolkit is installed: Reactant supplies the PJRT plugin via JLLs and only needs the host
# driver, exposed by the NVIDIA container runtime.
EXPOSE 8001 8002

# start_period is generous because every model compiles (Reactant -> device executable) before
# the gRPC plane accepts traffic; raise it for large model sets.
HEALTHCHECK --interval=30s --timeout=20s --retries=3 --start-period=300s \
  CMD ["/usr/local/bin/healthcheck.node.sh"]

# tini reaps zombies and keeps signal semantics conventional; the supervisor itself installs
# SIGTERM/SIGINT handlers and shuts its children down gracefully.
ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint.node.sh"]
