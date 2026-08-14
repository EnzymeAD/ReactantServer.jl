# Contributing

Developer notes for working on ReactantServer.jl. For using the server, start with
[Getting Started](docs/src/manual/getting_started.md).

After cloning, instantiate the workspace (gRPCServer resolves from the `scelles-merge` tag of
github.com/csvance/gRPCServer.jl via the workspace `[sources]`):

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Testing

Each package is tested in its own environment; all tests run on CPU and need no GPU:

```
julia --project=packages/ReactantServerCore    -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer         -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway  -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerNode     -e 'using Pkg; Pkg.test()'
```

The worker's pre/post-processing overlaps GPU execution only with more than one thread, so run its
suite multithreaded to exercise that path: `Pkg.test(; julia_args=["--threads=auto,1"])`.

`ReactantServerExport` is not a workspace member; its export round-trip tests (export a model, run
it through the runtime, and compare to a native Lux/PyTorch forward pass) run under their own env:

```
julia --project=packages/ReactantServerExport/test packages/ReactantServerExport/test/runtests.jl
```

The PyTorch portion skips gracefully when `torch`/`torchax` are unavailable.
`packages/ReactantServer/test/spike_reactant.jl` (and the `spike_*.jl` siblings) are standalone
scripts that exercise the Reactant runtime and export paths in isolation.

## Formatting

[Runic.jl](https://github.com/fredrikekre/Runic.jl) is the formatter. Runic has **no
configuration** — the rules are fixed, so output is deterministic.

Install the `runic` CLI (Julia ≥ 1.12 provides it as a Pkg app):

```
julia -e 'using Pkg; Pkg.Apps.add("Runic")'
```

Format in place (repo, directory, or single file):

```
runic --inplace .
runic --inplace packages/ReactantServerCore/src
runic --inplace path/to/file.jl
```

Check that files are formatted without modifying them (exits non-zero when changes are needed —
useful for CI):

```
runic --check .
```

A pre-commit hook runs `runic --inplace packages docs examples tools docker` and re-stages the formatted
files, so commits never ship unformatted code. `.git/hooks/` is machine-local and not versioned;
(re)install the hook in a fresh clone with:

```
cp scripts/pre-commit-runic .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

The generated protobuf bindings under `packages/ReactantServerCore/src/proto/` are formatted like
any other source; after regenerating them (see below), run `runic --inplace` over the generated
tree before committing so the diff stays clean.

## Regenerating the protobuf bindings

The KServe V2 messages and gRPC service stubs in
`packages/ReactantServerCore/src/proto/inference/` are generated from
`proto_src/grpc_predict_v2.proto` with ProtoBuf.jl. Load gRPCServer and gRPCClient alongside
ProtoBuf so both the server method builders / `register_GRPCInferenceService!` and the client
constructors are emitted, and keep `add_kwarg_constructors=true` (the handlers and codec build
messages with keyword arguments):

```julia
using ProtoBuf, gRPCServer, gRPCClient
ProtoBuf.protojl("grpc_predict_v2.proto", "proto_src", "packages/ReactantServerCore/src/proto";
    always_use_modules=true, add_kwarg_constructors=true)
```

The generated file is then split so `ReactantServerCore` compiles only the messages (no gRPC
dependency): the messages stay in `grpc_predict_v2_pb.jl`, while the gRPCClient and gRPCServer
service stubs are extracted into `grpc_client_stubs.jl` and `grpc_server_stubs.jl`, which
`ReactantServerCore` ships but does not compile. Each consumer includes the stub file it needs
(client stubs in the client and gateway; server stubs in the worker and gateway) via
`ReactantServerCore.inference_client_stubs_path()` / `inference_server_stubs_path()`.

The control-plane bindings in `packages/ReactantServerCore/src/proto/control/` are generated the same
way from `proto_src/reactant_control_v1.proto`:

```julia
using ProtoBuf, gRPCServer, gRPCClient
ProtoBuf.protojl("reactant_control_v1.proto", "proto_src", "<scratch dir>";
    always_use_modules=true, add_kwarg_constructors=true)
```

Generate into a scratch directory rather than over the real one, because the split here is manual and
the two stub files are hand-maintained:

1. The messages-only body (everything before the first `# gRPCClient.jl BEGIN` marker) becomes
   `reactant_control_v1_pb.jl`, **minus** the `import gRPCClient` / `import gRPCServer` lines the
   generator adds when a file declares services. `ReactantServerCore` must not gain a gRPC dependency.
   Message order changes when you add a message (the generator emits them in topological order so a
   nested-message field is defined before its use), so expect a large, mostly-cosmetic diff.
2. This proto declares two services, `ControlService` (answered by workers) and
   `GatewayControlService` (answered by the gateway), so the generator emits **two** BEGIN/END blocks
   per generator, in an order that does not follow the proto. Match them by content, not position.
   Move the `<Service>_<Rpc>_Method` builders and `register_<Service>!` into
   `control_server_stubs.jl` and the `<Service>_<Rpc>_Client` constructors into
   `control_client_stubs.jl`, preserving the hand-tuned deadlines already in those files.

## Documentation

The docs are built with Documenter on its default HTML theme, with two plugins:
DocumenterLandingPage renders the VitePress-style landing page (hero + emoji
feature tiles) from the YAML frontmatter block in `docs/src/index.md`, and
DocumenterCodeBlocks enhances the code blocks. They are published to GitHub
Pages by the `.github/workflows/docs.yml` workflow: every push and PR builds,
and pushes to `main` (or a tag) deploy the live site — PRs get a preview URL
(`deploydocs(push_preview = true)` in `docs/make.jl`). To build locally:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Output lands in `docs/build/`.

The landing page's `@raw html` frontmatter in `docs/src/index.md` is the single
source of truth for the hero and tile copy; it is the same VitePress `home`
layout the docs used under DocumenterVitepress. The plugin that renders it
lives at `github.com/csvance/DocumenterLandingPage.jl`, and `docs/Project.toml`
sources it from that git URL so the workflow's `Pkg.instantiate()` can resolve
it. For local work on the plugin, override the entry with a path to a local
checkout, e.g. `[sources] DocumenterLandingPage = {path =
"../../DocumenterLandingPage.jl"}`, then `Pkg.instantiate()` again. No Node
or npm is needed anywhere in the build.
