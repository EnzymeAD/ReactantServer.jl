# loadgen

Standalone soak / benchmark tools for a running ReactantServer node. Reactant-free: they need only
`ReactantServerClient` (which pulls in `ReactantServerCore` and `gRPCClient`), so run them under that
package's project. They connect to a node's gateway over KServe V2 gRPC and learn what it serves from
its `DescribeModels` RPC, so they need no access to the model repository; they do not build or manage
the server (they were relocated here from the removed container tooling).

- `loadgen.jl` — drives sustained, concurrent, zero-input inference across every model to surface
  memory leaks, races, and instability. Synthesizes inputs from each model's description (every
  compiled input shape of a multi-shape model included), stages them in the client's buffer pool
  and sends raw bytes on every transport, runs to a fixed duration, and prints a rolling summary
  and a per-model table. Configured entirely through
  `LOADGEN_*` environment variables (documented in the file header), including the `tcp` / `shm` /
  `mixed` transports, per-model coverage floors, `rows` / `burst` batch modes, deadline pressure,
  and a shed-ceiling concurrency ramp. Exit codes: 1 request errors, 3 coverage shortfall,
  4 wedged transport (stall watchdog).
- `probe_large_variants.jl`: sends one correctly-sized zero input per compiled input shape of a
  multi-shape model (e.g. `text_fuse_net_gpu`), largest first, so every compiled variant actually
  executes and allocates its full scratch. Use it to confirm the largest programs fit on the card.

## Run

Point the tools at a running node (the native launcher serves gRPC on `:8001`, metrics on `:8002`):

```bash
LOADGEN_GATEWAY=grpc://127.0.0.1:8001 \
LOADGEN_METRICS=http://127.0.0.1:8002/metrics \
LOADGEN_DURATION_SECONDS=300 LOADGEN_CONCURRENCY=32 LOADGEN_TRANSPORT=mixed \
  julia --project=packages/ReactantServerClient tools/loadgen/loadgen.jl

PROBE_GATEWAY=grpc://127.0.0.1:8001 PROBE_MODEL=text_fuse_net_gpu \
  julia --project=packages/ReactantServerClient tools/loadgen/probe_large_variants.jl
```

The `LOADGEN_*` / `PROBE_*` default host names (`gateway:8001`) are legacy container-network defaults;
set them explicitly for a native run as shown.
