```@meta
CurrentModule = ReactantServer
```

# Transport

The wire layer beneath the boundary types: the KServe V2 codec that translates protobuf
messages to and from [`InferRequest`](@ref) / [`NamedTensor`](@ref), and the system
shared-memory registry that backs the Triton-compatible zero-copy data plane. These live in
`ReactantServerCore` (shared by the worker and the gateway) and are documented here for
contributors.

## Codec

```@autodocs
Modules = [ReactantServerCore]
Pages = ["codec.jl"]
```

## Shared memory

```@autodocs
Modules = [ReactantServerCore]
Pages = ["shared_memory.jl"]
```

## Staging buffer pool

The concurrency-safe staging pool the client drives inference through (and the precompile
workloads exercise). A fixed-slot allocator over one contiguous backing region; a request may
span several physically contiguous slots.

```@autodocs
Modules = [ReactantServerCore]
Pages = ["buffer_pool.jl"]
```
