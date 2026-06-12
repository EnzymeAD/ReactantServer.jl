```@meta
CurrentModule = ReactantServerClient
```

# Client

`ReactantServerClient` is the inference client for talking to a `ReactantServer` worker or the
[gateway](gateway.md) over KServe V2 gRPC. It depends only on `ReactantServerCore` and the gRPC
layer, so it carries **no Reactant dependency** and installs on a plain client machine. See
[Client Usage](../manual/client_usage.md) for a worked example.

```@autodocs
Modules = [ReactantServerClient]
```
