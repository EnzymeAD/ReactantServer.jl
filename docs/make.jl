# The Documenter build, on the DocumenterVitepress theme (the VitePress look
# Lux.jl's docs use). `doctest = false` on purpose: the guide pages carry
# illustrative Julia rather than executable examples, because executing any of
# them would compile XLA programs and turn the docs build into a GPU-hours
# exercise. The API page reads docstrings only, which is the point of @autodocs.
#
# The landing page (docs/src/index.md) carries a VitePress home layout: the hero
# and the emoji capability boxes are YAML frontmatter inside a @raw html block,
# exactly as Lux.jl's index.md does.
using Documenter
using DocumenterCodeBlocks
using DocumenterVitepress
using ReactantServer
using ReactantServerClient
using ReactantServerCore
using ReactantServerExport
using ReactantServerGateway
using ReactantServerNode

makedocs(;
    sitename = "ReactantServer.jl",
    modules = [
        ReactantServer,
        ReactantServerClient,
        ReactantServerCore,
        ReactantServerExport,
        ReactantServerGateway,
        ReactantServerNode,
    ],
    authors = "Carroll Vance <cs.vance@icloud.com>",
    doctest = false,
    # DocumenterCodeBlocks enhances code blocks (reference links, hover tooltips,
    # line numbers, JuliaSyntax highlighting), but its post-processing targets the
    # standard Documenter HTML writer only. Under MarkdownVitepress the plugin runs
    # and no-ops, so these features activate if the docs ever move to Documenter's
    # own HTML backend.
    plugins = [CodeBlocks()],
    # The API page deliberately documents the exported surface of the five
    # documented packages plus the internal helpers their docstrings reference;
    # the rest of each module's internal docstrings are not in the manual, which
    # would otherwise fail the build. Unresolved @refs are errors by default in
    # this Documenter, and there are none left by the time this is committed;
    # the category below is the safety valve while a docstring is being edited.
    warnonly = [:missing_docs],
    # Only the pages listed above are built: the legacy manual/api/design pages
    # from the pre-Vitepress docs remain in the tree but are not processed.
    pagesonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/EnzymeAD/ReactantServer.jl",
        devbranch = "main",
        deploy_url = "https://enzymead.github.io/ReactantServer.jl/",
        # The workspace root (../Project.toml) is not a package, so Documenter
        # cannot infer a version for the search inventory; set it explicitly to
        # match the member packages.
        inventory_version = "0.1.0",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Bundles" => "bundles.md",
        "Node Configuration" => "node_config.md",
        "Scheduling" => "scheduling.md",
        "On-demand Weights" => "on_demand_weights.md",
        "Multi-GPU Gateway" => "gateway.md",
        "Client Usage" => "client.md",
        "Meta Models" => "meta_models.md",
        "Object Detection" => "object_detection.md",
        "Transformer Text Models" => "transformers.md",
        "Deployment" => "deployment.md",
        "API" => "api.md",
    ],
)

# Documenter.deploydocs is NOT compatible with DocumenterVitepress; this is the
# theme's own deployer. `push_preview = true` lands PR builds on a preview URL.
DocumenterVitepress.deploydocs(;
    repo = "github.com/EnzymeAD/ReactantServer.jl.git",
    push_preview = true,
    devbranch = "main",
)
