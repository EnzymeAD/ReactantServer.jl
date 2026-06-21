```@meta
CurrentModule = ReactantServer
```

# Manifest & Boundary

The bundle metadata types parsed from `manifest.yaml`, the transport-agnostic request/response
boundary types, and the canonical dtype enumeration. These live in `ReactantServerCore` (the
shared substrate). See [Bundles & model.jl](../manual/bundles.md) for the manifest format.

## Manifest

```@docs
Manifest
TensorSpec
Dim
BatchingSpec
load_manifest
is_meta
```

## Boundary

```@docs
NamedTensor
InferRequest
DeadlineExceeded
```

## Datatypes

```@docs
DType
```
