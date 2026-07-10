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
# Why this build works where the previous one aborted: Reactant statically links cuBLASLt into
# libReactantExtra.so, but CUDA-13 cuBLASLt dlopens libnvJitLink.so.13 to JIT its GEMM kernels.
# That library is NOT in the Reactant artifact and is NOT injected by the NVIDIA container runtime
# (which adds only libcuda.so.1), so a bare Julia image aborted at first GPU compile with cuBLASLt's
# "Invalid handle was passed to cublasLtCreate". We fix it by copying just libnvJitLink.so.13 from a
# matching CUDA 13.1 image. We deliberately do NOT bring the whole CUDA userspace: Reactant resolves
# its own bundled cuDNN 9 / NCCL 2 via $ORIGIN, and a system copy on LD_LIBRARY_PATH would shadow
# them with a mismatched version.
#
# Build context is the repository root. The only git submodule is lib/gRPCServer.jl (Reactant, HTTP,
# Prometheus, gRPCClient are registry deps); run `git submodule update --init lib/gRPCServer.jl`
# before building. NOTE: Manifest.toml is gitignored, so `COPY Manifest.toml` needs a locally
# generated Manifest in the build context (run `REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1
# julia --project=. -e 'using Pkg; Pkg.instantiate()'` first); a fresh clone has none.
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t reactantserver .   # or: make image

# --- CUDA userspace source stage: we pull exactly two files out of here (nvJitLink + the libcuda
# stub) and discard the rest, so this stage's size only affects build/pull, not the final image.
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE} AS cudalibs

# --- Final image: Debian 13 (Trixie) with Julia preinstalled.
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

# The one CUDA userspace library Reactant needs but does not bundle: libnvJitLink.so.13 (CUDA-13
# cuBLASLt dlopens it by soname to JIT GEMM kernels). Install it into a dir that holds ONLY this
# library and register that dir with the loader cache, so cuBLASLt finds it at run time WITHOUT any
# cuDNN/NCCL from the base image shadowing Reactant's own bundled copies (which resolve via $ORIGIN).
COPY --from=cudalibs /usr/local/cuda/lib64/libnvJitLink.so.13* /opt/cuda-userspace/lib/
RUN echo /opt/cuda-userspace/lib > /etc/ld.so.conf.d/reactant-cuda.conf && ldconfig

# Build-only libcuda stub, taken from the CUDA image's stubs dir (replaces the old hand-compiled gcc
# stub). libReactantExtra.so lists libcuda.so.1 in DT_NEEDED (the NVIDIA driver, injected by the
# container runtime at run time, absent during `docker build`). Precompilation never calls the
# driver, so this do-nothing stub satisfies the loader's NEEDED entry during precompile only. It is
# exposed solely via LD_LIBRARY_PATH for the precompile RUN below; at run time the real
# CDI-injected libcuda.so.1 is used, not this stub (this dir is never on the runtime path).
COPY --from=cudalibs /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stub/libcuda.so.1

WORKDIR /opt/reactantserver

# Workspace root environment, the member packages, and the vendored gRPCServer submodule (the
# remaining forks resolve from the registries via the shared Manifest.toml). The worker is
# built on the workspace root; the entrypoint then runs on the ReactantServer member project.
# .dockerignore drops the package test trees (e.g. the large e2e weights) from the packages/ copy.
COPY Project.toml Manifest.toml ./
COPY packages/ ./packages/
COPY lib/gRPCServer.jl/ ./lib/gRPCServer.jl/

# CUDA artifact selection for Reactant_jll. Its platform augmentation honors REACTANT_GPU and
# REACTANT_GPU_VERSION to choose the prebuilt artifact. Set as ENV (not ARG) so the same selection
# applies at build time (artifact download + precompile of the CUDA libReactantExtra) and at run
# time (load-time wrapper selection in the worker). Reactant ships 12.9 and 13.1 builds; 13.1 here
# matches the libnvJitLink.so.13 copied above.
ARG XLA_CUDA_GPU_VERSION=13.1
ENV REACTANT_GPU=cuda \
    REACTANT_GPU_VERSION=${XLA_CUDA_GPU_VERSION}

# Install the General registry (the custom JULIA_DEPOT_PATH starts empty, and resolve does not
# auto-install it), then resolve, instantiate, and precompile the whole workspace so the worker
# starts fast (no first-request precompile). LD_LIBRARY_PATH points at the build-only libcuda stub
# so libReactantExtra's NEEDED libcuda.so.1 resolves during precompile; nvJitLink is not needed here
# (it is dlopened lazily at the first serve-time GPU compile), so it is not on this path.
RUN LD_LIBRARY_PATH=/opt/cuda-stub julia --project=/opt/reactantserver -e 'using Pkg; Pkg.Registry.add("General"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'

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
COPY config/node.default.yaml /etc/reactantserver/node.yaml

# 8001/8002: the embedded gateway's client gRPC and metrics/health admin ports (the default
# all-in-one role). In the workers role the per-worker ports (base_port+i, default 8080..) and
# metrics ports (9100..) are bound instead; publish the range you configured.
EXPOSE 8001 8002

# start_period is generous because every model compiles (Reactant -> device executable) before
# the gRPC plane accepts traffic; raise it for large model sets.
HEALTHCHECK --interval=30s --timeout=20s --retries=3 --start-period=300s \
  CMD ["/usr/local/bin/healthcheck.node.sh"]

# tini reaps zombies and keeps signal semantics conventional; the supervisor itself installs
# SIGTERM/SIGINT handlers and shuts its children down gracefully.
ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint.node.sh"]
