# Container Releases

Each published image `ghcr.io/enzymead/reactantserver:X.Y.Z` is built from the tagged
`ReactantServer-vX.Y.Z` tree, its committed `deploy/Manifest.toml`, and the base image and Julia
distribution pinned in `MODULE.bazel`. Each release has a page listing what it ships, what it
requires of the host, its digests, and what changed from the release before it.

{{releases}}

Julia, the base image, and the compiler and runtime packages move only through
`bazel run //deploy:relock` or a new base image digest, never as a side effect of a release, so
consecutive releases often share them. **Changes from previous** names what moved. A release that
changes only ReactantServer's own packages keeps the numerics stack; any other change is a change to
revalidate (see [Reproducibility & Regulated Use](../reproducibility.md)).

## Accelerators

Each release page has a section for every accelerator its image supports, listing what the image
bundles for it and what it requires of the host. Released images support CUDA only.

### CUDA

The CUDA that runs a model comes from Reactant_jll, not from the base image. The image ships the
Reactant_jll artifact selected for `REACTANT_GPU=cuda` and the release's `REACTANT_GPU_VERSION`,
which carries `ptxas`, `fatbinary`, `libdevice` and `libnccl.so.2`, and links cuDNN, cuBLAS and
cuBLASLt, and NVSHMEM into `libReactantExtra.so`. A release page's CUDA rows are the versions that
artifact was built against. Reactant loads `libnvJitLink.so.13`, `libnvrtc.so.13` and `libcupti.so`
at run time, and those come from the base image's CUDA. The base also carries its own cuDNN and
cuBLAS packages, which Reactant does not load.

**NVIDIA driver.** The base image sets `NVIDIA_REQUIRE_CUDA`, which the NVIDIA Container Toolkit
checks when the container starts, so the host driver must support the base image's CUDA release.
The toolkit itself is required as described in [Deployment](../deployment.md).

**GPU.** The minimum compute capability is the oldest target the Reactant_jll artifact is built
for.
