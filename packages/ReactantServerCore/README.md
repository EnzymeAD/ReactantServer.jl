# ReactantServerCore.jl

The shared, **Reactant-free** substrate for the ReactantServer monorepo. The worker, gateway,
and client all build on it; nothing here depends on Reactant.

It provides the canonical dtype vocabulary, the KServe V2 protobuf message types, the
transport-agnostic boundary types (`NamedTensor`, `InferRequest`), the `manifest.yaml` parser,
the typed server/cluster configuration, the wire ↔ boundary codec, the server-side
shared-memory registry, and the concurrency-safe staging `BufferPool` (a fixed-slot allocator
shared by the client and server data planes).

The gRPC service stubs are kept out of this package: the generated protobuf is split into
messages (compiled here) plus `grpc_client_stubs.jl` / `grpc_server_stubs.jl`, which Core ships
but does not compile. Consumers include the one they need via `inference_client_stubs_path()` /
`inference_server_stubs_path()`, so Core never pulls `gRPCClient`/`gRPCServer`.

Part of the [ReactantServer.jl](../../README.md) workspace.
