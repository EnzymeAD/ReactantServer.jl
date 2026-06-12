# ReactantServerClient.jl

A **Reactant-free** inference client for a [`ReactantServer`](../ReactantServer) worker or the
[gateway](../ReactantServerGateway), speaking KServe V2 over gRPC. Because it depends only on
[`ReactantServerCore`](../ReactantServerCore) and the gRPC layer, it carries no Reactant/XLA
stack and installs quickly on a plain client machine.

Public surface: `KServeModel`, `infer_sync`, `infer_async`, `InferInput`, `InferOutput`,
`AbstractInferenceIO`, `kserve_init`, `kserve_shutdown`. Batched inference stages
tensors through the shared-memory `BufferPool` and falls back to inline transport when the
server cannot map the client's shared memory.

```julia
using ReactantServerClient
kserve_init()
model = KServeModel("grpc://127.0.0.1:8080", "scale4"; max_batch_size = 1)
resp  = infer_sync(model, [InferInput("INPUT__0", Float32[1, 2, 3, 4])])
y     = InferOutput("OUTPUT__0", resp, Float32)
kserve_shutdown()
```

Part of the [ReactantServer.jl](../../README.md) workspace. See
[Client Usage](../../docs/src/manual/client_usage.md).
