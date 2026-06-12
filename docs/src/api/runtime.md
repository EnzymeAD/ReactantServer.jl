```@meta
CurrentModule = ReactantServer
```

# Runtime & Weights

The device-side runtime: weight loading and materialization, the on-demand GPU weight cache,
and the shared device memory pool. These are internal interfaces, documented here for operators
and contributors. See [On-demand Weights](../manual/on_demand_weights.md) for the operational
view.

## Weights

```@autodocs
Modules = [ReactantServer]
Pages = ["runtime/weights.jl"]
```

## Weight stores (host RAM)

```@autodocs
Modules = [ReactantServerCore]
Pages = ["weight_store.jl"]
```

## Weight cache

```@autodocs
Modules = [ReactantServer]
Pages = ["runtime/weight_cache.jl"]
```

## Memory pool

```@autodocs
Modules = [ReactantServer]
Pages = ["runtime/memory_pool.jl"]
```
