# Documenter build for ReactantServer.jl.
#
# Local build (no GPU required): from the repository root, instantiate the docs environment
# (docs/Project.toml resolves ReactantServer via [sources], which transitively pulls in the
# vendored lib/ submodules) and run this script:
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl
#
# The output is written to docs/build/. The build only reads docstrings, so loading ReactantServer
# on CPU is sufficient; nothing here starts a server or touches a GPU.

using Documenter
using DocumenterMermaid
using ReactantServerCore
using ReactantServer
using ReactantServerGateway
using ReactantServerClient
using ReactantServerNode

for m in (ReactantServerCore, ReactantServer, ReactantServerGateway, ReactantServerClient, ReactantServerNode)
    DocMeta.setdocmeta!(m, :DocTestSetup, :(using $(Symbol(m))); recursive=true)
end

makedocs(;
    modules  = [ReactantServerCore, ReactantServer, ReactantServerGateway, ReactantServerClient, ReactantServerNode],
    authors  = "Carroll Vance <cs.vance@icloud.com>",
    sitename = "ReactantServer.jl",
    # The repository is hosted on Azure DevOps, which Documenter's automatic remote
    # detection does not recognize. Disable remote "source"/"edit" links for the local
    # build. Set a `Remotes.URL` here if Azure DevOps source links are wanted later.
    remotes = nothing,
    format = Documenter.HTML(;
        size_threshold = 400_000,
        # The workspace root is not a package, so Documenter cannot infer a version for the
        # search inventory; set it explicitly to match the member packages.
        inventory_version = "0.1.0",
        # Navbar link to the repository; set explicitly because remote detection is off.
        repolink  = "https://github.com/EnzymeAD/ReactantServer.jl",
        edit_link = "main",
        # Pretty (directory) URLs only in CI; plain .html files build locally so the
        # site is browseable straight from docs/build/ over file://.
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting Started"          => "manual/getting_started.md",
            "Common Use Cases"         => "manual/common_use_cases.md",
            "Scaling to Multiple GPUs" => "manual/scaling.md",
            "Client Usage"             => "manual/client_usage.md",
            "Node Configuration"       => "manual/node_config.md",
            "Bundles & model.jl"       => "manual/bundles.md",
            "Meta Models"              => "manual/meta_models.md",
            "Object Detection"         => "manual/object_detection.md",
            "On-demand Weights"        => "manual/on_demand_weights.md",
            "Multi-GPU Gateway"        => "manual/multi_gpu_gateway.md",
            "Docker Deployment"        => "manual/docker.md",
        ],
        "Design" => [
            "Philosophy"   => "design/philosophy.md",
            "Architecture" => "design/architecture.md",
        ],
        "API Reference" => [
            "Server & Lifecycle"  => "api/server.md",
            "Gateway"             => "api/gateway.md",
            "Client"              => "api/client.md",
            "Configuration"       => "api/config.md",
            "Scheduling"          => "api/scheduling.md",
            "Runtime & Weights"   => "api/runtime.md",
            "Manifest & Boundary" => "api/manifest_boundary.md",
            "Transport"           => "api/transport.md",
        ],
    ],
    checkdocs = :exports,
    doctest   = false,
    # Start lenient: warn (do not fail) on internals that lack docstrings. Tighten to `[]`
    # once docstring coverage is filled out.
    warnonly = [:missing_docs],
)

deploydocs(; repo = "github.com/EnzymeAD/ReactantServer.jl")