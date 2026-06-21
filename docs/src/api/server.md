```@meta
CurrentModule = ReactantServer
```

# Server & Lifecycle

The top-level entry points for starting and stopping a worker, plus the bundle registration
hook called from a model's `model.jl`.

```@docs
serve
serve_worker
stop!
register_model
register_meta_model
RunningServer
```

## Node supervisor

`ReactantServerNode.supervise` is the container entry point: it detects the visible GPUs, spawns
one [`serve`](@ref) worker subprocess per device, multiplexes the children's output with
`[name]` line prefixes, and restarts them on failure. With two or more workers it also runs the
embedded gateway; with a single worker it binds that worker to the public ports directly and
runs no gateway. See [Docker Deployment](@ref) for the operational surface.

```@meta
CurrentModule = ReactantServerNode
```

```@docs
supervise
```
