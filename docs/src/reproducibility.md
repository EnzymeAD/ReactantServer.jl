# Reproducibility & Regulated Use

A validated deployment needs more than correct outputs: it needs the same outputs for the same
input, today and after a restart, a redeploy, or a scale-out. This page lists what can change a
model's output on ReactantServer, what controls each source, and how the `regulated` runtime
profile and the release images fit together into a deployment you can validate once and rely on.

The short version: the same bundle, served from the same image digest, on the same GPU class, with
the same node file, gives the same outputs. Each section below covers one thing that can
break that, and how to hold it fixed.

## Numerics

`runtime.numerics` sets the precision of f32 matrix multiplications and convolutions, and it is the
largest source of difference between deployments.

- **`tf32`:** TF32 tensor cores, which keep 10 mantissa bits of each operand instead of f32's 23.
  Faster on NVIDIA Ampere and newer, with results that differ from full f32 in the low digits.
  Startup fails unless TF32 is confirmed active, so a worker cannot silently serve in f32.
- **`f32`:** full f32 everywhere, on every GPU generation, attested bit-exact by the startup probe.
  The choice when a hardware refresh must not change outputs, at the cost of tensor-core
  throughput.
- **`auto`** (the default): TF32 where the GPU supports it, full f32 where it does not. One bundle
  runs everywhere, but its numerics follow the hardware, so a fleet that mixes GPU generations
  serves different results for the same request. Not suitable for a validated deployment.

Pin the mode the deployment was validated with. Even within one mode, XLA and cuBLAS choose kernels
per GPU model, so results match a validation only on the same GPU class; validate again when the
class changes. [Node Configuration](node_config.md) describes how each mode is enforced.

## Batch size

A bundle may carry several compiled batch sizes (`model.b1.mlir`, `model.b4.mlir`, ...). Each is a
separate program, and XLA is free to choose different kernels, tilings, and reduction orders for
each. A request served by the batch-1 program can therefore differ in the last bits from the same
request served by the batch-8 program.

Under `runtime.batch_sizes: all` (the default) the scheduler picks the program from how many
requests are queued at the moment of dispatch, so a request's output can depend on unrelated
traffic: the same input returns slightly different values at night and at peak load.

`runtime.batch_sizes: largest` removes that dependence: every dispatch runs the largest compiled
program, padded to a full batch. The cost is the full batch's compute on every dispatch, including
a lone single-row request.

One program removes the batch-size dependence, but whether a row's output is also independent of
its position in the batch, and of the other rows beside it, depends on the model. Models whose
rows are computed independently behave this way in practice; an operation that mixes values across
the batch axis does not. Turning coalescing off, described next, removes that dependence on other
traffic.

## Coalescing

By default the scheduler coalesces: requests to the same model that are queued together are packed
into one dispatch, so where a request's rows land in the batch, and which other requests' rows sit
beside them, depends on traffic. Even with one compiled program, a kernel is not guaranteed to give
bit-identical results for a row in every slot.

`scheduler.coalescing: false` serves exactly one request per dispatch. With `batch_sizes: largest`
as well, a request always runs the same program with its rows in the same slots, followed by the
same zero padding, so its result no longer depends on anything another client sends. The cost is
throughput: each request pays for a full padded batch on its own.

What remains is set by the request itself. A client's pre-batched request runs as one dispatch, so
whether one of its rows is affected by the other rows the client put beside it still depends on the
model; the same request always gives the same result. A request larger than the largest compiled
size is split into the same pieces every time, and each piece runs alone.

The gateway does not yet know whether its workers coalesce: under `lpt_packing` its default
`routing_fill_mode: run` still concentrates a model's requests on one replica at a time. That
changes no outputs, but it can queue requests on one GPU while another sits idle; prefer
`routing_fill_mode: spread` for a fleet with coalescing off.

## Nondeterministic kernels

Some GPU kernels are not deterministic from run to run: a reduction or scatter that accumulates
with atomic additions sums in whatever order the threads finish, so the same program on the same
input can return different low bits. The XLA option `xla_gpu_exclude_nondeterministic_ops` asks
XLA to leave such implementations out of the compiled program. Set it through `runtime.xla_flags`;
the `regulated` profile sets it for you.

## Compilation and autotuning

When XLA compiles a program for a GPU it autotunes by default (`runtime.autotune: true`): it times
several candidate kernels for each matrix multiplication and convolution and keeps the fastest.
Autotuning improves performance greatly, but it is also a source of non-determinism. Timing is
noisy, so even with everything else held constant (the same image, bundle, GPU, driver, and
settings), two compiles of the same model can choose different algorithms, and different algorithms
can give different low bits.

A choice, once made, is held by two caches:

- **The executable cache** (`runtime.executable_cache`, on by default) stores each compiled program,
  kernel choices included, in the bundle's `.cache/` directory and loads it on later starts, so a
  restart does not autotune again. A program is compiled fresh whenever its entry is missing: on the
  first start, for a new bundle directory (each model version has its own), for a bundle copied
  without its `.cache/`, on a new host or volume, and after an image upgrade or a change to the
  compile settings.
- **The autotune cache** (`runtime.autotune_cache`, `runtime.autotune_cache_dir`;
  `/var/cache/reactant-compile` in the image) stores the timing decisions themselves, so a fresh
  compile reuses earlier choices instead of measuring again.

With autotuning on, make both caches persistent and back them up. Neither failure is visible: a
lost cache does not stop a worker, it only autotunes again and may choose differently, which is a
change to revalidate. In practice:

- Mount the autotune cache directory on a persistent volume, and keep each bundle directory
  writable and persistent (see [Container settings](@ref) for the mounts).
- Back up the autotune cache and the bundles' `.cache/` directories as part of the validation
  record, and restore them, rather than letting a new or rebuilt host autotune from scratch.
- After a restart, check that every program loaded from the executable cache: the worker logs
  `executable cache: hit` for each, and `worker_executable_cache_misses_total` stays at zero.

**`runtime.autotune: false`** removes the dependence on timing altogether: XLA uses its default
kernels, the same on every compile, at some cost in speed.

## The software stack

Every component between the bundle and the GPU can change numerics: Reactant and its XLA build, the
CUDA libraries, and the driver. The release image pins all of them except the host's driver:

- Each image is built from a tagged release and its locked `deploy/Manifest.toml`, which fixes
  Reactant and Reactant_jll by version and content hash, together with the CUDA userspace.
- Deploy by digest (`ghcr.io/enzymead/reactantserver@sha256:...`) or by release tag (`:0.1.0`),
  never by `:latest`, and verify the image's build attestation before deploying it. See
  [The container image](@ref).
- Keep the host's NVIDIA driver fixed across the validated fleet, and treat a driver upgrade as a
  change to validate.

## The regulated profile

`runtime.profile: regulated` applies the settings above that have a safe default:

- `runtime.batch_sizes: largest`
- `runtime.xla_flags: {xla_gpu_exclude_nondeterministic_ops: true}`
- `scheduler.coalescing: false`

Anything set explicitly wins, so either piece can be turned back off. The profile does not set
`runtime.numerics`, because the right mode is the one the deployment was validated with. Set it as
well:

```yaml
global:
  runtime:
    profile: regulated
    numerics: tf32   # or f32: whichever the deployment was validated with
```

or with environment variables:

```text
INFERENCE_SERVER_RUNTIME_PROFILE=regulated
INFERENCE_SERVER_RUNTIME_NUMERICS=tf32
```

## Startup evidence

Each worker records its effective settings at startup. Keep these log lines with the validation
record:

- **`Effective configuration`:** every resolved setting, including `numerics`, `profile`,
  `batch_sizes`, `coalescing`, `xla_flags`, and `autotune`.
- **`TF32 probe`:** whether TF32 arithmetic is in use on the device, and under `f32`, that the pin
  is bit-exact.
- **The `REGULATED PROFILE` banner:** the precision in effect under the regulated profile, as an
  info line under `f32` or `tf32` and as a warning under `auto`, together with `batch_sizes`,
  `coalescing`, and the XLA flags.
- **`model loaded`:** per model, the compiled batch sizes actually loaded and the numerics outcome
  (operations pinned, algorithms rewritten or stripped).

## Checklist

- Serve from a release image pinned by digest or release tag, with its build attestation verified.
- Set `runtime.profile: regulated` and pin `runtime.numerics` to the validated mode.
- Run every worker on the validated GPU class and driver version.
- With autotuning on, keep the executable cache and the autotune cache on persistent storage,
  back both up with the validation record, and confirm a restart loads every program from the cache.
  Otherwise set `runtime.autotune: false`.
- Leave coalescing off, as the profile sets it. If it is turned back on, confirm for each model
  that a request's output is bit-identical alone and inside a full batch.
- Archive each worker's startup evidence lines.
- Revalidate when the image, the GPU class, the driver, the numerics mode, or a bundle changes.
