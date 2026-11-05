# Core API

Every exported name in `ReactantServerCore` that carries a docstring: the shared substrate the other packages build on: dtypes, protobuf messages, boundary types, the manifest, configuration, the codec, shared memory, the weight store and the staging `BufferPool`. `ReactantServer` re-exports all of it. See the [API overview](api.md) for
the other packages and the unexported internals the guides refer to.

```@autodocs
Modules = [ReactantServerCore]
Public = true
Private = false
```
