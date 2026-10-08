# Reproducible node image (Bazel)

`//deploy:image` builds the node image (the `ReactantServerNode` supervisor as entrypoint under
`tini`, KServe V2 gRPC on `:8001`, health and metrics on `:8002`) from committed locks, with no
resolve at build time. The same commit always yields the same packages, artifacts, and system
libraries, which is what a validated production deployment needs. This image is the supported way
to deploy ReactantServer; it is published as `ghcr.io/enzymead/reactantserver:latest`, and the
docs' Deployment page covers running it. Nothing else in the repository depends on Bazel: tests,
docs, and running from a source checkout stay plain Julia.

```
bazel build //deploy:image
bazel test  //deploy:image_precompile_test  # the image starts without precompiling (no podman)
bazel run   //deploy:image_load          # loads it into podman as localhost/reactantserver:bazel
bazel run   //deploy:image_check         # the loaded image: caches, tini, curl, entrypoints
bazel run   //deploy:image_push -- --repository ghcr.io/<owner>/reactantserver --tag <tag>
bazel test  //deploy:manifest_current    # the Julia lock still matches the workspace Project.toml files
bazel run   //deploy:relock              # re-resolve the Julia lock after a [deps] or [compat] change
bazel run   //deploy:relock_debs         # re-resolve the Ubuntu package lock (curl, tini)
```

The Julia-specific parts (the distribution, depot and precompile-cache layers, the image
environment, and the precompile test) are the image rules of
[julia_depot](https://github.com/csvance/julia_depot) (`julia/image.bzl`); this
package adds the Ubuntu packages, the CUDA driver stub the build loads Reactant against, the
application layer, and the image itself, assembled with rules_oci.

Julia, the CUDA base image, and every package are fetched and pinned by Bazel. The build host needs
Bazel (the version in `.bazelversion`; Bazelisk picks it up), GNU tar, and a C compiler with `make`,
because InterProcessCommunication.jl generates its constants file by compiling a small C program when
the depot is instantiated. Podman is needed only to load, check, and run the result, and to
re-resolve the Ubuntu lock.

## The Julia lock: `deploy/Manifest.toml`

The workspace root `Manifest.toml` stays gitignored so development and CI resolve fresh and pick up
compat bumps. The image must not, so `deploy/Manifest.toml` is the production lock for the root
`Project.toml`: every build action stages the root project, the member `Project.toml` files, and
this manifest in a temporary tree and instantiates from it, with no `Pkg.resolve` anywhere in the
build. It lives under `deploy/` only so the root gitignore can stay as it is; a symlinked
`deploy/Project.toml` does not work, because Pkg resolves the real path of the project and would
read the root manifest instead.

`//deploy:manifest_current` fails when a `[deps]` or `[compat]` change leaves the lock stale, or
when the lock was resolved under a Julia other than the pinned distribution. Move it deliberately
and review the diff like code:

```
bazel run //deploy:relock                           # keep every version that still resolves
bazel run //deploy:relock -- Reactant Reactant_jll  # also move the named packages forward
```

## The Ubuntu lock: `deploy/debs.lock.json`

`curl` (the healthcheck) and `tini` (PID 1) are not in the CUDA base, so the image adds them and the
libraries `curl` needs as pinned `.deb` files. `//deploy:relock_debs` resolves them by running apt
inside the pinned base image, so the lock holds exactly what the base lacks: a layer that re-shipped
a package the base already has (`libc6`, `libssl3t64`) would overwrite the base's copy. Each package
is pinned by a `snapshot.ubuntu.com` URL, which serves the archive as it was at the lock's timestamp
indefinitely, and by sha256. The build downloads them through `deploy/debs.bzl` and unpacks them
with the `bsdtar` toolchain, so neither `apt` nor `dpkg` runs during the build. Each package's
control stanza is written to `/var/lib/dpkg/status.d/`, so image scanners (Trivy, Grype, Syft) still
see what was added.

```
bazel run //deploy:relock_debs                      # resolve against today's snapshot
bazel run //deploy:relock_debs -- 20260928T000000Z  # resolve against a given snapshot
```

Moving the base image digest in `MODULE.bazel` calls for a relock, since the delta is computed
against the base.

## Layers

In image order, least often changed first, so a push after a code change moves only the last
layers:

- `debs_layer`: `curl`, `tini`, and their libraries, from the Ubuntu lock.
- `julia_layer`: Julia at `/opt/julia`, its relative symlinks kept.
- `depot_layer`: the Julia depot at `/opt/julia-depot`, instantiated from the Julia lock into an
  empty depot (the General registry is fetched fresh; the lock pins every package by tree hash, so
  the registry state cannot change what is installed). Set `JULIA_PKG_SERVER` through
  `--action_env` in an untracked `user.bazelrc` to go through a mirror.
- `compiled_layer`: the precompile caches for every project the image starts Julia in (the
  supervisor, a worker, the gateway, and the worker-role healthcheck), so a container serves without
  first compiling about 150 packages.
- `app_layer`: the workspace at `/opt/reactantserver` (root project plus the Julia lock, member
  packages without their tests, `config/`), the entrypoints and healthchecks from
  [`runtime/`](runtime/README.md) in `/usr/local/bin`, and the default node file at
  `/etc/reactantserver/node.yaml`.

`image_env` writes the image's environment: the depot path below, `JULIA_PROJECT`, the portable
`JULIA_CPU_TARGET`, `JULIA_PKG_OFFLINE=true`, Julia's `bin/` ahead of the base's `PATH`, and the
CUDA and NVIDIA variables.

The base is `nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04`, pinned by digest. It is deliberately
fat: Reactant_jll's CUDA 13.1 artifact links cuDNN and cuBLASLt statically but `dlopen`s
`libnvJitLink.so.13`, `libnvrtc.so.13`, and `libcupti.so` by soname, and the NVIDIA container
runtime injects none of them (only the driver's `libcuda.so.1`). `REACTANT_GPU=cuda` and
`REACTANT_GPU_VERSION=13.1` are set both when the depot is instantiated (they select the Reactant
artifact) and in the image environment (they select it again at load time); they must agree with
the base image's CUDA version.

### How the caches are built outside the image

A Julia precompile cache records the source files it was built from, and Julia rejects it when those
files are not where it expects. Files inside a depot are recorded relative to it (`@depot/...`),
which is what lets registry packages move; files outside every depot are recorded by absolute path.
The workspace's own packages live outside the package depot, so the image lists the workspace root
as a depot too:

```
JULIA_DEPOT_PATH=/opt/julia-depot:/opt/reactantserver:/opt/julia/local/share/julia:/opt/julia/share/julia
```

`compiled_layer` unpacks the depot and application layers into one temporary tree with the image's
layout and precompiles against the same list with that tree in place of `/opt`, so its caches
load unchanged in the image. The loader needs `libcuda.so.1` to load Reactant; the build takes the
driver stub from the CUDA base as a build-time layer (`cuda_stub_layer`, unpacked beside the others
but never part of the image), puts it on `LD_LIBRARY_PATH`, and hides the host's GPUs. The caches are compiled for
the official Julia build's x86_64 CPU targets, so they load on any x86_64 host, not only CPUs like
the build machine's. `//deploy:image_precompile_test` verifies, on the layers and without a
container, that each entry project loads with no cache rejected and nothing precompiled;
`//deploy:image_check` repeats that inside a loaded container and also checks `tini`, `curl`, and
the entrypoints.

## Running it

The docs' Deployment page ("Running the container") is the guide to running the image, including
the shared-memory, process-limit and thread settings a container needs. What follows are the
details specific to how this image is built.

The image expects the host driver to be provided at run time, like any CUDA image. With the NVIDIA
Container Toolkit and a CDI spec podman can read, `--device nvidia.com/gpu=<n>` is enough. Where CDI
is unavailable, bind the device nodes (`/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia<minor>`)
and the driver libraries onto their sonames (`libcuda.so.1`, `libnvidia-ml.so.1`,
`libnvidia-nvvm.so.4`, `libnvidia-ptxjitcompiler.so.1`, `libnvidia-gpucomp.so.<version>`). Note that
`/dev/nvidia<minor>` is the device-node minor number from `nvidia-smi -q`, which need not equal
`nvidia-smi`'s index.

Mount the model repository at `/var/lib/reactantserver/models` read-write: the serialized
executable cache lives inside each bundle (`<bundle>/.cache/`), so a read-only mount recompiles
every program on every start. Mount your own node file over `/etc/reactantserver/node.yaml` to
change the configuration. Do not mount a volume over `/opt/julia-depot/compiled`: it would hide the
baked caches.

The healthcheck is `/usr/local/bin/healthcheck.node.sh`, but it is not part of the image: an OCI
image configuration has no healthcheck field (only Docker's image format does), so rules_oci cannot
set one. The repository's `docker-compose.yml` sets it; with `podman run`, pass it at run time:

```
podman run ... --health-cmd /usr/local/bin/healthcheck.node.sh --health-interval 30s \
    --health-timeout 20s --health-retries 3 --health-start-period 300s localhost/reactantserver:bazel
```

## Publishing

`.github/workflows/image.yml` builds the image, runs `//deploy:manifest_current` and
`//deploy:image_precompile_test`, pushes it to the GitHub Container Registry as
`ghcr.io/enzymead/reactantserver`, and attests its build provenance. It needs no secrets: the
run's own `GITHUB_TOKEN` pushes the image, and the owner in the path is the repository's,
lowercased, so a fork publishes to its own namespace. The first push creates the package, which an
organization owner then makes public once in the package's settings.

Images are versioned by ReactantServer releases. Each round of releases registers
ReactantServer last, so its version names the whole image: worker, gateway, and node, built from
the tagged tree and its `deploy/Manifest.toml`. When the General registry merges a ReactantServer
version, TagBot pushes `ReactantServer-vX.Y.Z` on the registered commit, and that push runs the
workflow, which publishes `:X.Y.Z`, plus `:X.Y` and `:latest` when no higher release exists in
those lines (a patch to an older line moves neither). Run by hand (Actions, Image, Run workflow) on
a branch, it publishes `:sha-<commit>` only, so `:latest` always means the newest release; run by
hand on a `ReactantServer-vX.Y.Z` tag, it publishes that release. The run summary records the
tags, the pushed digest, and the commit.

The attestation is [SLSA build provenance](https://slsa.dev/spec/v1.0/provenance), produced by
`actions/attest`: a statement that this repository's `image.yml`, at a given commit and run, built
the image with a given digest, signed through Sigstore with the run's OIDC identity (no signing
key is kept anywhere). It is stored with GitHub and pushed next to the image. This matters more
for this image than for most because the build is not bit-for-bit reproducible (the precompile
caches differ between builds), so rebuilding from source does not reproduce the published digest;
the attestation is what ties a digest to its source. Verify an image before deploying it with:

```
gh attestation verify oci://ghcr.io/enzymead/reactantserver:latest --repo EnzymeAD/ReactantServer.jl
```

`gh` needs to be logged in (`gh auth login`), and verifies the digest the tag currently resolves
to; pass `@sha256:<digest>` instead of a tag to check a pinned image.

## Consuming from another module

The image and its layers are public. A deployment repository can take a `bazel_dep` on
`reactant_server` (with a `git_override` on a commit) and `oci_load` or extend
`@reactant_server//deploy:image` instead of rebuilding it. The repository names this module
declares (`reactantserver_julia`, `reactantserver_cuda_base`, `reactantserver_debs`) are prefixed so
they cannot collide with a consumer's own.
