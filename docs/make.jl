# The Documenter build, on the default Documenter HTML theme with two plugins:
# DocumenterLandingPage renders the VitePress-style landing page (hero + emoji
# capability tiles) from the YAML frontmatter in docs/src/index.md, and
# DocumenterCodeBlocks enhances the code blocks (line numbers, reference
# links, hover tooltips, JuliaSyntax highlighting). `doctest = false` on
# purpose: the guide pages carry illustrative Julia rather than executable
# examples, because executing any of them would compile XLA programs and turn
# the docs build into a GPU-hours exercise. The API page reads docstrings only,
# which is the point of @autodocs.
#
# The landing page frontmatter (docs/src/index.md) is exactly the VitePress
# home layout the docs previously carried under DocumenterVitepress; the
# plugin replaces the block with rendered HTML at build time.
using Documenter
using DocumenterCodeBlocks
using DocumenterLandingPage
using Dates
using Literate
using ReactantServer
using ReactantServerClient
using ReactantServerCore
using ReactantServerExport
using ReactantServerGateway
using ReactantServerNode

# The native tutorials are Literate scripts under integration/tutorials/, which the integration
# suite (integration/runtests.jl) runs in CI, so every page shows code that is tested as written.
# They are rendered to markdown here without being executed (running them starts servers and
# compiles XLA programs). `#src` lines (the scripts' test assertions and process cleanup) are
# dropped, and the fences become plain `julia` blocks so Documenter does not try to run them. The
# output directory is gitignored; EditURL points the "Edit on GitHub" link at the script.
const NATIVE_TUTORIALS = [
    "Export a Model" => "export",
    "Serve a Model" => "serve",
    "Query a Server" => "client",
    "Run a Node" => "node",
    "Front Workers with a Gateway" => "gateway",
]
let src = joinpath(@__DIR__, "..", "integration", "tutorials"),
        out = joinpath(@__DIR__, "src", "tutorials", "native")
    rm(out; force = true, recursive = true)
    for (_, name) in NATIVE_TUTORIALS
        Literate.markdown(
            joinpath(src, "$name.jl"), out;
            flavor = Literate.DocumenterFlavor(), execute = false, credit = false,
            codefence = "```julia" => "```"
        )
    end
end

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
    # The API page deliberately documents the exported surface of the five
    # documented packages plus the internal helpers their docstrings reference;
    # the rest of each module's internal docstrings are not in the manual, which
    # would otherwise fail the build. Unresolved @refs are errors by default in
    # this Documenter, and there are none left by the time this is committed;
    # the category below is the safety valve while a docstring is being edited.
    warnonly = [:missing_docs],
    # Only the pages listed below are built; nothing else under docs/src is processed.
    # The design pages (architecture, philosophy) are the pre-Vitepress design docs, kept
    # alive and linked from the README; the legacy manual/ and api/ trees were superseded
    # by the flat guide pages and deleted.
    pagesonly = true,
    repo = Documenter.Remotes.GitHub("EnzymeAD", "ReactantServer.jl"),
    format = Documenter.HTML(
        edit_link = "main",
        canonical = "https://enzymead.github.io/ReactantServer.jl/",
        # The workspace root (../Project.toml) is not a package, so Documenter
        # cannot infer a version for the search inventory; set it explicitly to
        # match the member packages.
        inventory_version = "0.1.0",
        footer = "Made with [Documenter.jl](https://documenter.juliadocs.org/stable/), [DocumenterLandingPage.jl](https://github.com/csvance/DocumenterLandingPage.jl), and [DocumenterCodeBlocks.jl](https://github.com/fredrikekre/DocumenterCodeBlocks.jl)<br>© Copyright $(Dates.year(Dates.today())).",
    ),
    plugins = [
        LandingPage(),
        CodeBlocks(),
    ],
    # Sections nest in the sidebar; the files stay flat under docs/src so page URLs do not move.
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Native Julia" => [
                [title => "tutorials/native/$name.md" for (title, name) in NATIVE_TUTORIALS]...,
                "Serve an Object Detector" => "tutorials/object_detection.md",
                "Serve Transformer Text Models" => "tutorials/transformers.md",
            ],
            "Container" => [
                "Serve on One GPU" => "tutorials/container_single_gpu.md",
                "Scale to Many GPUs" => "tutorials/container_multi_gpu.md",
            ],
        ],
        "Building Models" => [
            "Bundles" => "bundles.md",
            "Meta Models" => "meta_models.md",
            "Worked Examples" => [
                "Object Detection" => "object_detection.md",
                "Transformer Text Models" => "transformers.md",
            ],
        ],
        "Running in Production" => [
            "Deployment" => "deployment.md",
            "Node Configuration" => "node_config.md",
            "Multi-GPU Gateway" => "gateway.md",
            "Reproducibility & Regulated Use" => "reproducibility.md",
            "Monitoring" => "monitoring.md",
            "Security" => "security.md",
        ],
        "Clients" => [
            "Client Usage" => "client.md",
        ],
        "How It Works" => [
            "Architecture" => "design/architecture.md",
            "Scheduling" => "scheduling.md",
            "On-demand Weights" => "on_demand_weights.md",
        ],
        "Project" => [
            "Philosophy" => "design/philosophy.md",
        ],
        "API Reference" => "api.md",
    ],
)

# PRs get a preview URL via `push_preview = true`; pushes to main deploy the
# live site (tag pushes deliberately do not build, see .github/workflows/docs.yml).
# This is Documenter's own deployer, unlike the
# VitePress-specific DocumenterVitepress.deploydocs the docs previously used.
Documenter.deploydocs(;
    repo = "github.com/EnzymeAD/ReactantServer.jl.git",
    push_preview = true,
    devbranch = "main",
)
