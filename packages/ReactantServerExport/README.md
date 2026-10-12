# ReactantServerExport.jl

Offline tooling that produces ReactantServer model **bundles**: a directory holding a
`manifest.yaml`, a compiled StableHLO program per batch size, and the shared
`weights.safetensors`. `export_bundle` traces a Reactant-traceable model (for example a Lux model)
into a bundle; with `PythonCall` loaded, a package extension adds the PyTorch frontends
(`export_bundle(:pytorch, ...)` and `export_bundle(:torchscript, ...)`). `write_bundle` and
`IOSpec` are the low-level writer underneath.

This package is the bundle-format authority for producers. The server never loads it: export
happens offline in its own environment, and the worker reads only the bundle it writes. It is
kept in sync with the server by round-trip tests that export a model, run it through the
runtime, and compare against the native forward pass.

Part of the [ReactantServer.jl](../../README.md) repository, though not a member of the serving
workspace. See [Bundles](https://enzymead.github.io/ReactantServer.jl/dev/bundles/) for the bundle
format and the export frontends, and the
[Export a Model](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/export/) tutorial for exporting a model end to end.
