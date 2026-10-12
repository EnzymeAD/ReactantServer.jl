# ReactantServerClient.jl

A **Reactant-free** inference client for a [`ReactantServer`](../ReactantServer) worker or the
[gateway](../ReactantServerGateway), speaking KServe V2 over gRPC. Because it depends only on
[`ReactantServerCore`](../ReactantServerCore) and the gRPC layer, it carries no Reactant/XLA
stack and installs quickly on a plain client machine.

Public surface: `KServeModel`, `infer_sync`, `infer_async`, `InferInput`, `InferOutput`,
`AbstractInferenceIO`, `kserve_init`, `kserve_shutdown`. Batched inference stages
tensors through the shared-memory `BufferPool`. Each server is probed once with the
`IsSameIPCNamespace` RPC to decide whether shared memory can work; servers in a different
namespace (or that lack the probe) use inline transport instead. The `shared_memory` keyword on
`KServeModel` (`:auto`/`:on`/`:off`) overrides this per model.

Part of the [ReactantServer.jl](../../README.md) workspace. The
[Query a Server](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/basics/client/) tutorial walks through one-shot and
batched inference; [Client Usage](https://enzymead.github.io/ReactantServer.jl/dev/client/) is the reference for IO validation, retries, and the
shared-memory data path.
