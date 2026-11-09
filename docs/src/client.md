# Client Usage

`ReactantServerClient` is the inference client for a `ReactantServer` worker or the
[Gateway](gateway.md). It builds KServe V2 requests against `ReactantServerCore`'s protobuf
messages and forwards them over gRPC. It depends only on `ReactantServerCore` and the gRPC
layer, so it carries **no Reactant/XLA dependency** and installs quickly on a plain client
machine. (The same KServe V2 wire protocol means any Triton or KServe gRPC client also works;
this package is the convenient Julia option.)

[Run Inference](tutorials/inference.md) walks through using it: one-shot requests, batched
inference over a dataset, and the shared-memory transport. This page is the reference for how the
client behaves.

## One-shot requests

[`KServeModel`](@ref) takes either `(host, port, model_name)` or a `url` of the form
`grpc://host:port` (or bare `host:port`); `grpcs`/`https` selects a secure channel.
`max_batch_size` caps how many items the batched drivers coalesce per request and must be at least
1; anything smaller is refused with an `ArgumentError` when the model is constructed.
[`describe_model`](@ref) reports a `max_batch_size` of 0 for a meta model and when the server does
not know the limit, so do not pass that value through: pass the largest of the description's
`batch_sizes`, or 1 when it lists none. `deadline` is the per-request timeout in seconds (10 by
default). [`infer_sync`](@ref)`(model, inputs)` sends
one request built from [`InferInput`](@ref) tensors, and [`InferOutput`](@ref) reads a named output
back. Tensors use Julia's column-major layout with the batch axis last on both ends; the wire codec
converts to KServe's row-major shapes.

The one-shot path encodes tensor data element by element as typed contents, so it emits a
one-time performance warning (once per session) and suits small, occasional requests. For
datasets, use the IO-driven drivers below: they stage through the buffer pool and ship raw
bytes on every transport.

## Batched inference over a dataset

For many items, implement [`AbstractInferenceIO`](@ref) and use [`infer_async`](@ref)
(concurrent) or [`infer_sync`](@ref)`(model, io)` (serial). The driver stages each chunk
through a shared staging pool, an `InferenceBufferPool` wrapping
`ReactantServerCore.BufferPool`. The first call to each server probes it with the
`IsSameIPCNamespace` RPC: if the server confirms it shares the client's IPC namespace, inputs
travel over the system shared-memory data plane with zero extra copies on the wire; otherwise,
or if the server does not implement the probe, the driver uses inline transport. Inline sends
one raw little-endian byte blob per input (`raw_input_contents`) that aliases the staging
pool directly, so no typed per-element encoding happens on the client either; what shared
memory still buys is skipping the transfer itself, since the server reads the registered
region instead of receiving the payload.

The `shared_memory` keyword on [`KServeModel`](@ref) controls this choice:

- `:auto` (default) probes and uses shared memory when available.
- `:on` forces it, and errors loudly if the server is in a different namespace.
- `:off` always sends inline; the probe is not sent.

Shared memory is not supported on macOS yet: the server cannot reopen a region the client
creates. There `:auto` sends inline without probing and warns once per process, and `:on` throws
an `ArgumentError` when the model is constructed.

There is no silent runtime fallback once shared memory is chosen. A stale registration (the
server restarted and no longer knows the region) is re-registered and retried once in-band; if
that fails, the client latches that endpoint to inline transport and a background task
re-probes it every `shm_reprobe_interval` seconds, restoring shared memory when the server
recovers.

The shared-memory pool is created on the first call that chooses shared memory, and every page of
it is allocated then, so `/dev/shm` must have room for the whole pool (`pool_bytes`, plus a page of
slot control words per 32 slots). When it has not, `:auto` latches the endpoint to inline
transport with a warning, and the background re-probe creates the pool once there is room; `:on`
fails with an error that says so. Allocating up front is what keeps a full `/dev/shm` from
killing the client with SIGBUS partway through staging a request.

Your IO implements `length`, `item_input_bytes`, `infer_encode_chunk!` (request the chunk's
input buffers with `scratch`, write into them, and return them tagged with their tensor names),
and `infer_decode_chunk!` (consume the response). The pool's fixed-slot allocator hands every
chunk a disjoint slot, so concurrent `infer_async` calls are safe. `kserve_init(; pool_bytes,
n_slots, shm_reprobe_interval)` sizes the staging pool (256 MiB by default) and its number of
slots (8 by default), which bounds how many chunks can be in flight against one pool;
`kserve_shutdown()` tears the pool and the gRPC subsystem down.

[Run Inference](tutorials/inference.md) builds a complete IO for a small model.

`scratch` is the same buffer-request interface meta models use through `call.scratch`; see the
[Meta Models](meta_models.md) page. The lower-level path is still supported: carve a
`subslot`, `pool_view` it, and return an `InferInput(name, sub, shape, T)` descriptor; both
forms may be mixed in the returned vector. A descriptor whose shape needs more bytes than its
subslot holds is refused with an `ArgumentError` on either transport.

Slot boundaries are aligned (to a page for slots of at least a page, otherwise to 64 bytes), and
`scratch` places every buffer at an offset aligned for its element type: within one call it lays
the buffers out by alignment, largest first, which leaves no padding, so the buffers fill exactly
`item_input_bytes` per item whatever the dtypes and chunk size. A `subslot` is carved at the cursor
as is; pass `align = sizeof(T)` before viewing it as `T` (`pool_view` refuses a misaligned offset
with an error that says so).

Outputs can also be read back through shared memory: return a non-empty
[`output_specs`](@ref) vector of [`OutputSpec`](@ref)`(name, dtype, per_item_dims)` entries and
the driver asks the server to write each declared output into the staging slot, then copies the
bytes into the response before `infer_decode_chunk!` sees it, so [`InferOutput`](@ref) works
the same on either transport. The default (empty) keeps every output inline. A non-empty
vector also opts the IO into explicit-output mode: the request asks for exactly those outputs,
so declare every output the IO consumes. Outputs with data-dependent shapes cannot be sized
ahead of time and must stay inline; an `OutputSpec` with a negative dimension (such as the `-1`
that [`describe_model`](@ref) reports for a variable axis) is refused with an `ArgumentError`.

### When a chunk fails

[`infer_sync`](@ref)`(model, io)` stops at the first chunk that fails and throws its exception.
[`infer_async`](@ref) may have other chunks in flight when one fails:

- Once a chunk has failed, no further chunk starts. A chunk already in flight finishes the attempt
  it has sent (a shared-memory attempt cannot be abandoned safely) and then makes no further
  attempt: no retry or backoff after a shed, and no re-registration of a stale shared-memory
  registration. A result that still arrives is delivered to `infer_decode_chunk!`.
- The call waits for those chunks before it throws, so once it returns or throws, nothing writes
  into the IO or holds a pool slot for it.
- It throws the original exception, not a `TaskFailedException`, so a
  `gRPCClient.gRPCServiceCallException`, or an exception from your own `infer_encode_chunk!` or
  `infer_decode_chunk!`, is caught by type exactly as from `infer_sync`. The chunk's backtrace stays
  on the exception stack (`current_exceptions()`), where Julia prints it as "caused by".

An unrecoverable shared-memory failure is not thrown: on either driver the endpoint latches to
inline transport as described above and the whole batch runs again inline, so a chunk decoded
before the failure is decoded again. When several chunks of an `infer_async` call fail, the first
failure is thrown, except that an unrecoverable shared-memory failure takes precedence over the
others, so that fallback happens even when another chunk failed first.

### Slot ownership

A chunk that gives up (its deadline passed, the call failed, or the connection dropped) returns
its slot at once, while the server may still be running the request. So that a late server can
never write that request's outputs into a slot another chunk is now using, every slot of the
shared-memory pool has a control word in a control area after the data slots (one 8-byte word
per slot, padded to 128 bytes), and every request staged in the pool carries three request
parameters naming it: `reactant_shm_control_region`, `reactant_shm_control_offset` and
`reactant_shm_control_token`. The word holds a state and a token that is new for every attempt,
retries included:

- the client sets it to `OWNED(t)` before sending attempt `t`;
- before writing shared-memory outputs, the worker takes an open-file-description lock on the
  word, moves it from `OWNED(t)` to `WRITING(t)`, checks before every 1 MiB of the copy that it
  still reads `WRITING(t)`, then stores `DONE(t)` and unlocks; if the word no longer reads
  `OWNED(t)` it writes nothing and the request fails with `CANCELLED`;
- a client that gives up moves `OWNED(t)` to `CANCELLED(t)` and reuses the slot immediately; if
  the word already reads `WRITING(t)` the slot is set aside (quarantined) until it reads
  `DONE(t)`;
- the worker also drops, at dispatch and with `CANCELLED`, a queued request whose word no longer
  reads `OWNED(t)`.

Nothing waits on the other side. A slot whose writer still holds `WRITING` after 30 seconds is
checked through the writer's lock: if the lock is free the writer died (the kernel releases it)
and the slot is reclaimed; if it is held the writer is alive but stuck, and the slot stays retired,
never reused, until that writer finishes. Requests from clients that send no control parameters
(third-party KServe clients) behave as before, except that a worker no longer writes shared-memory
outputs once the request's deadline has passed. The protocol needs Linux; where it is unavailable
the client sends no control parameters (on macOS shared memory is off anyway) and a worker refuses
them with `INVALID_ARGUMENT`. The registered window is the whole object, control area included.

A chunk never waits for a slot. When none is free (all in flight, quarantined or retired) it is
sent inline at once through a private buffer, and `ReactantServerClient.inline_fallback_count()`
counts these; a count that keeps growing means the pool has too few slots for the concurrency.
`infer_async` keeps at most as many chunks in flight as the pool has slots, so one call alone does
not fall back, and an inference started from inside `infer_decode_chunk!` of another (which used to
deadlock when the outer call held every slot) runs inline. Only a chunk too large to send in one
gRPC message (by the model's `max_send_message_length` or `max_receive_message_length`) waits for
a shared-memory slot, for at most the model's `deadline`, and then fails with an error that says
so.

## Validating an IO against a model

Because an IO hard-codes a model's tensor names, dtypes, and shapes, it can silently drift
from the served model. [`validate_io`](@ref) dry-runs an IO against the model's true I/O spec
without sending an inference request: it runs `infer_encode_chunk!` and `infer_decode_chunk!`
once over synthetic, spec-shaped data and surfaces name, dtype, shape, and indexing mismatches
as errors. Get the spec from [`model_io_spec`](@ref)`(model)` against a running worker or gateway
(the KServe ModelMetadata RPC; it throws if the server is unreachable or does not implement it),
or from [`manifest_io_spec`](@ref)`(path)` offline against a model's `manifest.yaml`, suitable for
a build-time check. Reported shapes are Julia column-major (batch axis last), matching the
axis order you build arrays in.

```julia
validate_io(model, io)                              # online: fetches the spec from the server
validate_io(manifest_io_spec("manifest.yaml"), io)  # offline: no server needed
```

The harness runs your real methods, so call it on a representative or dummy IO (it may write
into the IO's buffers at positions `1:items`). For example, validating an IO that writes a length-4
output against a model whose output is actually length 3 reports the shape mismatch, and an
`infer_decode_chunk!` that indexed `out[:, j]` past the real batch size surfaces as an indexing
error, both without sending a request.

### Bounds checks in hot paths

`validate_io` can only catch an out-of-range index if the access is bounds-checked. Julia's
built-in `@inbounds` elides the check unconditionally, so an `@inbounds` access stays unchecked
even inside the dry run. If you want `@inbounds` performance in production but a robust check
during validation, write the access with [`@infer_inbounds`](@ref) instead. It elides the check in normal use, but because `validate_io` runs your methods
inside [`with_bounds_checks`](@ref), an `@infer_inbounds` access is bounds-checked during the
dry run and an out-of-range index is caught rather than silently read. This is opt-in: using
`@infer_inbounds` is the user's responsibility, and bare `@inbounds` remains unchecked unless
Julia is started with `--check-bounds=yes` (which `Pkg.test` does).

```julia
# Bounds-checked under validate_io, elided in production:
@infer_inbounds buf[(4k - 3):(4k)] .= io.inputs[i]

# Force the checked behavior anywhere, not just in validate_io:
with_bounds_checks() do
    run_my_io_step()
end
```

## Describing the served models

KServe's metadata writes `-1` for a batch axis and a variable axis alike and says nothing about
batch sizes. [`describe_models`](@ref)`(url)` asks a worker or the gateway for every model it
routes requests to, and [`describe_model`](@ref)`(model)` for one; each [`ModelDescription`](@ref)
carries the tensors in Julia column-major order with the kind of every axis (`:fixed`, `:batch`,
`:variable`), the compiled batch sizes and the worker's coalescing limit, the compiled input
shapes of a model with variable axes, and the sub-models a meta model calls. That is enough to
build a valid request without the bundle; the load generator in `tools/loadgen` works this way.

```julia
for d in describe_models("grpc://127.0.0.1:8001")
    println(d.name, " ", [(t.name, t.shape, t.axes) for t in d.inputs], " batches ", d.batch_sizes)
end
```

## Retries

A worker or gateway at its concurrency cap (`endpoints.max_concurrent_requests`, see
[Node Configuration](node_config.md)) sheds requests with `RESOURCE_EXHAUSTED`. By default the
client retries: `ReactantServerClient.RetryPolicy` waits `initial_backoff` seconds and grows the
interval by
`factor` each attempt (1s, 2s, 4s, ...) up to `max_backoff`, until the model's `deadline`
budget is spent. The per-attempt deadline sent to the server shrinks to the remaining budget,
so retries never push total wall time past the original `deadline`. `jitter` (full jitter:
wait uniform in `[0, backoff]`) keeps concurrent chunk requests from synchronizing into a
retry storm, and `min_budget` (seconds) is the smallest remaining budget worth another
attempt. `retry_unavailable` also retries `UNAVAILABLE` (no replica reachable). Deadline,
`NOT_FOUND`, and `INVALID_ARGUMENT` errors always fail fast. Pass a `RetryPolicy(...)` as the
`retry` keyword to [`KServeModel`](@ref); `ReactantServerClient.RetryPolicy(enabled = false)`
turns retries off, so every failure is returned to the caller.

Every documented name in `ReactantServerClient`, including the signatures covered here, is
collected on the [API](api.md) page.
