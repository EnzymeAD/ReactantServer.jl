# Reproducibility & Regulated Use

A validated deployment needs more than correct outputs: it needs the same outputs for the same
input, today and after a restart, a redeploy, or a scale-out. This page lists what can change a
model's output on ReactantServer, what controls each source, and how the `regulated` runtime
profile and the release images fit together into a deployment you can validate once and rely on.

The short version: the same bundle, served from the same image digest, on the same GPU class, with
the same node configuration, gives the same outputs. Each section below covers one thing that can
break that, and how to hold it fixed.

## Numerics

`runtime.numerics` sets the precision of f32 matrix multiplications and convolutions, and it is the
largest source of difference between deployments.

- **`tf32`:** TF32 tensor cores, which keep 10 mantissa bits of each operand instead of f32's 23.
  Faster on NVIDIA Ampere and newer, with results that differ from full f32 in the low digits.
  Startup fails unless TF32 is confirmed active: on a GPU without TF32, with
  `NVIDIA_TF32_OVERRIDE=0` set, or if the startup probe matmul does not show TF32 arithmetic.
- **`f32`:** full f32 everywhere, on every GPU generation, attested bit-exact by the startup probe.
  The choice when a hardware refresh must not change outputs, at the cost of tensor-core
  throughput.
- **`auto`** (the default): TF32 where the GPU supports it, full f32 where it does not. One bundle
  runs everywhere, but its numerics follow the hardware, so a fleet that mixes GPU generations
  serves different results for the same request. Not suitable for a validated deployment.

Pin the mode the deployment was validated with. Even within one mode, XLA and cuBLAS choose kernels
per GPU model, so results match a validation only on the same GPU class; validate again when the
class changes. [Node Configuration](node_config.md) has the full reference.

## Batch size

A bundle may carry several compiled batch sizes (`model.b1.mlir`, `model.b4.mlir`, ...). Each is a
separate program, and XLA is free to choose different kernels, tilings, and reduction orders for
each. A request served by the batch-1 program can therefore differ in the last bits from the same
request served by the batch-8 program.

Under `runtime.batch_sizes: all` (the default) the scheduler picks the program from how many
requests are queued at the moment of dispatch, so a request's output can depend on unrelated
traffic: the same input returns slightly different values at night and at peak load.

`runtime.batch_sizes: largest` removes that dependence. Only the largest compiled size is loaded,
and every dispatch runs that one program: a partial batch is padded with zero rows and each
request's rows are sliced back out. The cost is the full batch's compute on every dispatch,
including a lone single-row request.

One program removes the batch-size dependence, but whether a row's output is also independent of
its position in the batch, and of the other rows beside it, depends on the model. Models whose
rows are computed independently behave this way in practice; an operation that mixes values across
the batch axis does not. Check it for each model as part of validation: send the same input alone,
and again alongside other requests, and compare the outputs bit for bit.

## Nondeterministic kernels

Some GPU kernels are not deterministic from run to run: a reduction or scatter that accumulates
with atomic additions sums in whatever order the threads finish, so the same program on the same
input can return different low bits. The XLA option `xla_gpu_exclude_nondeterministic_ops` asks
XLA to leave such implementations out of the compiled program. Set it through `runtime.xla_flags`;
the `regulated` profile sets it for you.

## Compilation and autotuning

When XLA compiles a program for a GPU it can autotune: time several candidate kernels and keep the
fastest. Timing is noisy, so two compiles of the same model on the same GPU class can choose
differently, and different kernels can give different low bits.

Three settings hold the choice fixed:

- **The executable cache** (`runtime.executable_cache`, on by default) stores each compiled program
  in the bundle's `.cache/` directory and loads it on later starts. A loaded program is the exact
  program that was compiled, with its kernel choices included, so a restart cannot change them.
  Entries are keyed by the Reactant_jll build, the device kind and compute capability, the
  numerics mode, the autotune setting, and the XLA flags; changing any of those compiles fresh.
- **Cached autotune results** (`runtime.autotune_cache`, `runtime.autotune_cache_dir`) make a
  fresh compile reuse earlier timing decisions instead of measuring again.
- **`runtime.autotune: false`** skips autotuning and uses XLA's default kernels. Selection no
  longer depends on timing at all, at some cost in speed.

## The software stack

Every component between the bundle and the GPU can change numerics: Reactant and its XLA build, the
CUDA libraries, and the driver. The release image pins all of them except the host's driver:

- Each image is built from a tagged release and its locked `deploy/Manifest.toml`, which fixes
  Reactant and Reactant_jll by version and content hash, together with the CUDA userspace.
- Deploy by digest (`ghcr.io/enzymead/reactantserver@sha256:...`) or by release tag (`:0.1.0`),
  never by `:latest`, and verify the image's build attestation before deploying it. See
  [Running the container](@ref).
- Keep the host's NVIDIA driver fixed across the validated fleet, and treat a driver upgrade as a
  change to validate.

## The regulated profile

`runtime.profile: regulated` applies the settings above that have a safe default:

- `runtime.batch_sizes: largest`
- `runtime.xla_flags: {xla_gpu_exclude_nondeterministic_ops: true}`

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
  `batch_sizes`, `xla_flags`, and `autotune`.
- **`TF32 probe`:** whether TF32 arithmetic is in use on the device, and under `f32`, that the pin
  is bit-exact.
- **The `REGULATED PROFILE` banner:** the precision in effect under the regulated profile, as an
  info line under `f32` or `tf32` and as a warning under `auto`.
- **`model loaded`:** per model, the compiled batch sizes actually loaded and the numerics outcome
  (operations pinned, algorithms rewritten or stripped).

## Checklist

- Serve from a release image pinned by digest or release tag, with its build attestation verified.
- Set `runtime.profile: regulated` and pin `runtime.numerics` to the validated mode.
- Run every worker on the validated GPU class and driver version.
- Keep the executable cache on, and cache autotune results or turn autotuning off.
- For each model, confirm a request's output is bit-identical alone and inside a full batch.
- Archive each worker's startup evidence lines.
- Revalidate when the image, the GPU class, the driver, the numerics mode, or a bundle changes.
