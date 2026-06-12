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
RunningServer
```
