# Container release records: what each published image ships and what it requires of the host.
#
# The data is committed under docs/releases/:
#
#   <version>.toml    one per published image, written by `record` from the release's git tag and
#                     the registry. It names the components it was built from (the Julia version,
#                     the base image digest, the Reactant_jll version and CUDA artifact) and records
#                     the package versions from that tag's deploy/Manifest.toml.
#   components.toml   what those components are: Julia's LLVM, the base image's tag and driver
#                     requirement, the CUDA a Reactant_jll artifact bundles. Written by hand, once
#                     per new component, since none of it is in this repository.
#   overview.md       the prose of the overview page; `{{releases}}` marks where its table goes.
#
# docs/make.jl calls `render` to write docs/src/releases/ (gitignored) before makedocs: the
# overview, and one page per release. Rendering fails, naming the missing entry, when a release uses
# a component components.toml does not describe yet.
#
# After a release's image is published, from the repository root:
#
#   julia docs/releases.jl record 0.1.2                          # digest from the registry
#   julia docs/releases.jl record 0.1.2 --digest sha256:... --published 2026-11-02
module ContainerReleases

    using Dates, Downloads, TOML

    const DATA = joinpath(@__DIR__, "releases")
    const OUT = joinpath(@__DIR__, "src", "releases")
    const ROOT = dirname(@__DIR__)
    const IMAGE = "ghcr.io/enzymead/reactantserver"
    const REPO_URL = "https://github.com/EnzymeAD/ReactantServer.jl"

    # The compiler and runtime packages a release records, in the order its page lists them. A change
    # to any of these between releases is a change to the numerics stack.
    const STACK_PACKAGES = [
        "Reactant", "ReactantCore", "Reactant_jll", "Enzyme", "EnzymeCore", "Enzyme_jll", "LLVM",
        "LLVMExtra_jll", "GPUCompiler",
    ]

    # ReactantServer's own packages. A release that changes only these keeps the numerics stack.
    const SERVER_PACKAGES = [
        "ReactantServer", "ReactantServerCore", "ReactantServerGateway", "ReactantServerNode",
    ]

    const SERVER_SECTION = "ReactantServer packages"

    load_components() = TOML.parsefile(joinpath(DATA, "components.toml"))

    function load_releases()
        releases = Dict{String, Any}[]
        for file in readdir(DATA; join = true)
            endswith(file, ".toml") && basename(file) != "components.toml" || continue
            r = TOML.parsefile(file)
            basename(file) == r["version"] * ".toml" ||
                error("$file records version $(r["version"]); a record is named after its version")
            push!(releases, r)
        end
        return sort!(releases; by = r -> VersionNumber(r["version"]), rev = true)
    end

    function lookup(c, kind, key, version)
        haskey(c[kind], key) && return c[kind][key]
        error("release $version uses $kind $key, which docs/releases/components.toml does not describe; add it")
    end

    # What a release builds on, from components.toml.
    function components(r, c)
        jll = lookup(c, "reactant_jll", r["packages"]["Reactant_jll"], r["version"])
        return (
            julia = lookup(c, "julia", r["julia"], r["version"]),
            base = lookup(c, "base_image", r["base_image"], r["version"]),
            jll = get(jll["cuda"], r["cuda"]) do
                error("release $(r["version"]) uses Reactant_jll $(r["packages"]["Reactant_jll"]) for CUDA $(r["cuda"]), which docs/releases/components.toml does not describe; add it")
            end,
        )
    end

    code(s) = "`$s`"

    # What a release ships, as (section, label, value) in page order. The release pages, the overview
    # table and the change summary all read these rows, so a row added here appears in each.
    function rows(r, c)
        (; julia, base, jll) = components(r, c)
        out = Tuple{String, String, String}[
            ("Requirements", "NVIDIA driver", base["min_driver"] * " or newer"),
            ("Requirements", "GPU compute capability", jll["min_compute_capability"] * " or newer"),
            ("Julia and the image", "Julia", r["julia"]),
            ("Julia and the image", "LLVM (Julia's)", julia["llvm"]),
            ("Julia and the image", "Base image", code(base["tag"])),
            ("Julia and the image", "Base image digest", code(r["base_image"])),
            ("Julia and the image", "Operating system", base["os"]),
            ("Julia and the image", "Platform", code(r["platform"])),
            ("CUDA (bundled by Reactant_jll)", "CUDA toolkit", jll["cuda"]),
            ("CUDA (bundled by Reactant_jll)", "cuDNN", jll["cudnn"]),
            ("CUDA (bundled by Reactant_jll)", "NCCL", jll["nccl"]),
            ("CUDA (bundled by Reactant_jll)", "NVSHMEM", jll["nvshmem"]),
            ("CUDA (bundled by Reactant_jll)", "Compute capabilities built", join(code.(jll["compute_capabilities"]), ", ")),
        ]
        for p in STACK_PACKAGES
            push!(out, ("Compiler and runtime packages", p, r["packages"][p]))
        end
        for p in SERVER_PACKAGES
            push!(out, (SERVER_SECTION, p, r["server_packages"][p]))
        end
        return out
    end

    # The rows that differ from the previous release, as (label, before, after).
    function changes(r, prev, c)
        before = Dict((s, l) => v for (s, l, v) in rows(prev, c))
        return [(s, l, before[(s, l)], v) for (s, l, v) in rows(r, c) if get(before, (s, l), nothing) != v]
    end

    function change_summary(r, prev, c)
        prev === nothing && return "First release"
        changed = changes(r, prev, c)
        stack = unique(l for (s, l, _, _) in changed if s != SERVER_SECTION)
        isempty(stack) && return "ReactantServer packages only"
        return join(stack, ", ")
    end

    function table(io, header, body)
        println(io, "| ", join(header, " | "), " |")
        println(io, "|", join(fill(":--", length(header)), "|"), "|")
        for row in body
            println(io, "| ", join(row, " | "), " |")
        end
        return println(io)
    end

    function release_page(io, r, prev, c)
        v = r["version"]
        tag = "ReactantServer-v$v"
        println(io, "```@meta\nEditURL = \"../../releases/$v.toml\"\n```\n")
        println(io, "# Release $v\n")
        println(
            io, code("$IMAGE:$v"), ", published ", r["published"], " from [`", tag, "`](", REPO_URL,
            "/tree/", tag, ") (commit ", code(r["commit"]), ").\n"
        )
        if prev === nothing
            println(io, "This is the first container release.\n")
        else
            changed = changes(r, prev, c)
            pv = prev["version"]
            if all(s == SERVER_SECTION for (s, _, _, _) in changed)
                println(io, "Compared with [$pv]($pv.md), only ReactantServer's own packages changed; the numerics stack is the same.\n")
            else
                println(io, "Compared with [$pv]($pv.md), this release changes the numerics stack; revalidate before moving to it (see [Reproducibility & Regulated Use](../reproducibility.md)).\n")
            end
            table(io, ["Changed", pv, v], [[l, b, a] for (_, l, b, a) in changed])
        end
        section = ""
        body = Vector{String}[]
        for (s, l, val) in rows(r, c)
            if s != section
                isempty(body) || table(io, ["Component", "Version"], body)
                println(io, "## ", s, "\n")
                section, body = s, Vector{String}[]
            end
            push!(body, [l, val])
        end
        table(io, ["Component", "Version"], body)
        (; base, jll) = components(r, c)
        println(io, "## Digests\n")
        table(
            io, ["", "Digest"], [
                ["Image", code(r["image_digest"])],
                ["Base image, `linux/amd64`", code(base["amd64"])],
                ["Reactant_jll artifact (tree hash)", code(jll["artifact"])],
                ["Reactant_jll built from Reactant.jl", code(jll["source_commit"])],
            ]
        )
        print(io, "Pull by digest with ", code("$IMAGE@$(r["image_digest"])"), ".")
        if r["labels"]
            println(io, " The image also records its ReactantServer package versions and the commit it was built from in its labels (`podman inspect`).")
        else
            println(io, " This image predates the version labels later images carry.")
        end
        return println(io)
    end

    function overview_page(io, releases, c)
        template = read(joinpath(DATA, "overview.md"), String)
        occursin("{{releases}}", template) || error("docs/releases/overview.md has no {{releases}} marker")
        buf = IOBuffer()
        body = map(enumerate(releases)) do (i, r)
            prev = i < length(releases) ? releases[i + 1] : nothing
            get_row(label) = only(v for (_, l, v) in rows(r, c) if l == label)
            v = r["version"]
            ["[$v]($v.md)", string(r["published"]), r["julia"], r["packages"]["Reactant"], get_row("CUDA toolkit"), get_row("NVIDIA driver"), change_summary(r, prev, c)]
        end
        table(buf, ["Release", "Published", "Julia", "Reactant", "CUDA", "Driver", "Changes from previous"], body)
        println(io, "```@meta\nEditURL = \"../../releases/overview.md\"\n```\n")
        return print(io, replace(template, "{{releases}}" => rstrip(String(take!(buf)))))
    end

    """
        render() -> pages

    Write the overview and one page per release into docs/src/releases/, and return the Documenter
    `pages` entries for them, newest release first.
    """
    function render()
        c = load_components()
        releases = load_releases()
        rm(OUT; recursive = true, force = true)
        mkpath(OUT)
        open(io -> overview_page(io, releases, c), joinpath(OUT, "index.md"), "w")
        for (i, r) in enumerate(releases)
            prev = i < length(releases) ? releases[i + 1] : nothing
            open(io -> release_page(io, r, prev, c), joinpath(OUT, r["version"] * ".md"), "w")
        end
        return ["Overview" => "releases/index.md", [r["version"] => "releases/$(r["version"]).md" for r in releases]...]
    end

    # The digest the registry serves for a release tag. The image is public, so an anonymous pull token
    # is enough.
    function registry_digest(version)
        repo = replace(IMAGE, "ghcr.io/" => "")
        auth = String(take!(Downloads.download("https://ghcr.io/token?scope=repository:$repo:pull", IOBuffer())))
        token = match(r"\"token\"\s*:\s*\"([^\"]+)\"", auth)[1]
        resp = Downloads.request(
            "https://ghcr.io/v2/$repo/manifests/$version"; method = "HEAD", output = devnull,
            headers = [
                "Authorization" => "Bearer $token",
                "Accept" => "application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json",
            ]
        )
        resp.status == 200 || error("$IMAGE:$version: the registry answered $(resp.status); is the image published?")
        for (k, v) in resp.headers
            lowercase(k) == "docker-content-digest" && return v
        end
        return error("$IMAGE:$version: the registry sent no digest")
    end

    function capture(m, what)
        m === nothing && error("could not find $what")
        return m[1]
    end

    """
        record(version; digest = nothing, published = today())

    Write docs/releases/<version>.toml from the `ReactantServer-v<version>` tag: the Julia version, base
    image digest, platform and CUDA selection pinned in that tree, and the package versions from its
    deploy/Manifest.toml. The image digest comes from the registry unless given.
    """
    function record(version; digest = nothing, published = today())
        tag = "ReactantServer-v$version"
        show(path) = read(`git -C $ROOT show $tag:$path`, String)
        manifest = TOML.parse(show("deploy/Manifest.toml"))
        mod = show("MODULE.bazel")
        build = show("deploy/BUILD.bazel")
        julia = capture(match(r"name = \"reactantserver_julia\",\s*version = \"([^\"]+)\"", mod), "the Julia version in MODULE.bazel")
        julia == manifest["julia_version"] || error("MODULE.bazel pins Julia $julia but deploy/Manifest.toml is for $(manifest["julia_version"])")
        base = capture(match(r"name = \"reactantserver_cuda_base\",.*?digest = \"(sha256:[0-9a-f]{64})\""s, mod), "the base image digest in MODULE.bazel")
        platform = capture(match(r"name = \"reactantserver_cuda_base\",.*?platforms = \[\"([^\"]+)\"\]"s, mod), "the base image platform in MODULE.bazel")
        cuda = capture(match(r"\"REACTANT_GPU_VERSION\": \"([^\"]+)\"", build), "REACTANT_GPU_VERSION in deploy/BUILD.bazel")
        pkgversion(p) = manifest["deps"][p][1]["version"]
        commit = readchomp(`git -C $ROOT rev-list -n 1 $tag`)
        digest = something(digest, registry_digest(version))
        file = joinpath(DATA, "$version.toml")
        open(file, "w") do io
            println(io, "# Container release $version, written by `julia docs/releases.jl record $version`.")
            println(io, "version = \"$version\"")
            println(io, "published = $published")
            println(io, "commit = \"$commit\"")
            println(io, "image_digest = \"$digest\"")
            println(io, "labels = $(occursin("name = \"image_labels\"", build))")
            println(io, "platform = \"$platform\"")
            println(io, "julia = \"$julia\"")
            println(io, "base_image = \"$base\"")
            println(io, "cuda = \"$cuda\"")
            println(io, "\n[packages]")
            foreach(p -> println(io, p, " = \"", pkgversion(p), "\""), STACK_PACKAGES)
            println(io, "\n[server_packages]")
            foreach(p -> println(io, p, " = \"", pkgversion(p), "\""), SERVER_PACKAGES)
        end
        r = TOML.parsefile(file)
        try
            components(r, load_components())
        catch err
            @warn "wrote $file, but the docs will not build until components.toml describes it" exception = err
        end
        return file
    end

    function main(args)
        (length(args) >= 2 && args[1] == "record") ||
            error("usage: julia docs/releases.jl record <version> [--digest sha256:...] [--published YYYY-MM-DD]")
        opts = Dict(args[i] => args[i + 1] for i in 3:2:(length(args) - 1))
        file = record(
            args[2]; digest = get(opts, "--digest", nothing),
            published = haskey(opts, "--published") ? Date(opts["--published"]) : today()
        )
        return println("wrote ", relpath(file, ROOT))
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ContainerReleases.main(ARGS)
end
