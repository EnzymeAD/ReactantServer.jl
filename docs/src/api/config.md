```@meta
CurrentModule = ReactantServer
```

# Configuration

The typed configuration a worker resolves from its cluster file. These types live in
`ReactantServerCore` (the shared substrate) and are re-exported by `ReactantServer`. See
[Cluster Configuration](../manual/cluster_config.md) for how these map onto the YAML.

```@docs
ServerConfig
RuntimeConfig
SchedulerConfig
ModelSchedConfig
EndpointsConfig
ModelControlMode
SchedulingDiscipline
ResidencyMode
ResidencyState
```

## Node files

Parsing and validation of the node (cluster) file, and the resolution of one worker's
[`ServerConfig`](@ref) from it:

```@autodocs
Modules = [ReactantServerCore]
Pages = ["node.jl"]
```
