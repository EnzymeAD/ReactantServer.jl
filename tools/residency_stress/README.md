# residency_stress

A stress and fuzz suite for weight residency: the on-demand device weight cache and its admission,
the host RAM budget (private and node-shared `/dev/shm` stores), memory compaction, eviction, the
weights layout migration, and the directory watcher's add, remove, rename and update paths. Every
answer is checked against the exact weights the model should be serving, so the suite catches wrong,
stale and corrupted weights, not only errors.

## Pieces

- `gen_models.jl`: writes a model repository of "heavy weights, trivial compute" bundles and a
  `catalog.toml` describing them. Each synthetic model returns, per weight tensor, an exact u64
  position-sensitive checksum of the tensor's bits plus a per-row nonce from the request, so the
  driver can tell a correct answer from another model's weights, a stale version, a tensor read in
  the wrong axis order, a single corrupted element, or a row from another request. Profiles:
  `tiny` (a few MB, for CPU runs), `small` (about 1.2 GB) and `standard` (about 6 GB). The models
  mix f32, bf16 and both in one file; few large and many small tensors; odd 3 to 5 axis shapes;
  legacy-order and program-order (`axis_order: xla`) weights; NaN payloads, signed zeros,
  infinities and subnormals; a multi-shape input model; an unbatched model; two versions of one
  model; two models with identical weights; and alternate weight files for weights-only updates.
- `export_bit.jl`: exports N ResNetV2-50 BiT models from Luximm.jl (random weights, one seed
  each) with a host reference output, and adds them to a catalog (checked to 2 percent relative
  error, since TF32 may be in effect). Needs no GPU. Run it into the OUT_DIR first, then
  `gen_models.jl` into the same OUT_DIR (it keeps the BiT entries).
- `stress.jl`: the driver. Concurrent workload (round robin over every model for maximal churn,
  same-model bursts for coalescing, random picks), periodic stampedes of concurrent requests at the
  least recently used model, and chaos actions: `CompactMemory` with and without a reload list
  (including an unknown name), rename, a registry promotion chain (X to X-old and Y to X in one
  poll), remove and re-add, weights-only update (atomic rename of an alternate file over
  `weights.safetensors`), and copy under a new name. During a change's settle window the previous
  weights and names stay acceptable; after it they are findings. A leak probe periodically stops
  traffic, compacts with nothing reloaded and records device memory, RSS and `/dev/shm`. Writes
  `events.log`, `samples.csv`, `report.txt` and `findings.toml`.
- `run.sh`: brings up one worker (`single`) or the node supervisor with its gateway (`node`) on a
  hard-linked copy of the repository, under `gpu-lease` for CUDA, runs the driver, shuts down, and
  extracts alarming worker log lines into `worker-alerts.txt`.
- `shm_fuzz.jl`: a multi-process fuzz of the node-shared weight store alone (no GPU, no Reactant):
  processes race to load through the budgeted cache, pin and release floors, trim, and load two
  versions of each model, verifying every byte and the budget.
- `alloc_probe.jl`: builds known free-block layouts in a small BFC arena and uploads tensors that
  admission would consider placeable, reporting which fail (GPU, under `gpu-lease`). Its last case
  replays a model that admission placed in a free block of exactly its size and that BFC then
  refused after a 10 s wait.
- `grpc_reset_repro.jl`: shows that a gRPCClient call in flight when its handle is reset (what the
  gateway does to a worker whose probe timed out) returns an empty response instead of raising.

## Running

```
# Generate (once). OUT_DIR on a large local filesystem, not a network share.
julia tools/residency_stress/export_bit.jl OUT_DIR 4                     # optional BiT models
julia --project=tools/residency_stress tools/residency_stress/gen_models.jl OUT_DIR standard

# One worker. Keep the arena and the host budget well below the total weights.
RS_MEM_FRACTION=0.06 RS_HOST_BUDGET=2g RS_DURATION=1800 tools/residency_stress/run.sh OUT_DIR single

# Several workers behind the gateway, sharing host weights through /dev/shm.
RS_GPUS=2 RS_SHARED=true RS_MEM_FRACTION=0.06 RS_HOST_BUDGET=2g RS_DURATION=1800 \
    tools/residency_stress/run.sh OUT_DIR node

# No GPU: the store fuzz and a CPU dry run of the driver.
julia --project=packages/ReactantServerCore tools/residency_stress/shm_fuzz.jl 8 120 4
RS_BACKEND=cpu RS_DURATION=120 tools/residency_stress/run.sh TINY_OUT_DIR single
```

`run.sh` documents its knobs in its header (arena fraction, host budget, shared store, weights
migration, compaction interval, gateway compaction mode, watcher poll interval, debug logging); the
driver's are the `Opts` fields at the top of `stress.jl` (duration, concurrency, chaos actions and
cadence, settle window, leak-probe period). The generator uses the same Reactant version as the
worker; if Luximm resolves a newer one, set `RS_STABLEHLO_TARGET` (see `export_bit.jl`). bash reads
a script as it runs, so for long runs launch a copy of `run.sh` with `RS_REPO_ROOT` pointing at this
tree rather than editing it underneath a run.

A run passes when `report.txt` lists no findings and the quiesce probes show device memory returning
to the same floor. Also read `worker-alerts.txt`: admission fallbacks, disk reloads and failed
watcher changes are logged there without failing a request.
