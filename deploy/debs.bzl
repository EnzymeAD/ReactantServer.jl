"""The Ubuntu packages locked in deploy/debs.lock.json, as pinned downloads.

Each locked package becomes an http_file repository fetched by its snapshot URL and checked
against its sha256, so the build needs no apt, no dpkg and no view of what the archive holds today.
One hub repository gathers them into a single filegroup for //deploy:debs_layer.

    debs = use_extension("//deploy:debs.bzl", "debs")
    debs.lock(name = "reactantserver_debs", lock = "//deploy:debs.lock.json")
    use_repo(debs, "reactantserver_debs")

The lock is written by //deploy:relock_debs; see relock_debs.sh for why it is resolved inside the
base image and why the URLs are snapshot URLs.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

def _repo_name(hub, package):
    # Package names may hold '.', '+' and '-', none of which are safe in every repository name.
    return hub + "_" + package.replace(".", "_").replace("+", "_").replace("-", "_")

def _hub_impl(rctx):
    rctx.file("BUILD.bazel", """\
# Generated from {lock} by //deploy:debs.bzl.
filegroup(
    name = "debs",
    srcs = {srcs},
    visibility = ["//visibility:public"],
)
""".format(
        lock = rctx.attr.lock,
        srcs = json.encode_indent(rctx.attr.debs, indent = "    ").replace("\n", "\n    "),
    ))

_hub = repository_rule(
    implementation = _hub_impl,
    attrs = {
        "debs": attr.string_list(),
        "lock": attr.string(),
    },
)

def _debs_impl(mctx):
    for mod in mctx.modules:
        for tag in mod.tags.lock:
            lock = json.decode(mctx.read(tag.lock))
            labels = []
            for p in lock["packages"]:
                repo = _repo_name(tag.name, p["name"])
                http_file(
                    name = repo,
                    urls = [p["url"]],
                    sha256 = p["sha256"],
                    # The .deb basename, so a failed extraction names the package.
                    downloaded_file_path = p["url"].rsplit("/", 1)[1].replace("%2b", "+"),
                )
                labels.append("@{}//file".format(repo))
            _hub(name = tag.name, debs = labels, lock = str(tag.lock))
    return mctx.extension_metadata(reproducible = True)

debs = module_extension(
    implementation = _debs_impl,
    tag_classes = {
        "lock": tag_class(attrs = {
            "name": attr.string(mandatory = True),
            "lock": attr.label(mandatory = True, allow_single_file = [".json"]),
        }),
    },
)
