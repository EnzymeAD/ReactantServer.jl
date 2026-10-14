"""The Ubuntu packages locked in deploy/debs.lock.json, as checked-in files.

Each locked package is a .deb under deploy/debs/, stored with Git LFS, and becomes a repository
holding that file once it has been checked against the lock's sha256. The build needs no apt, no
dpkg, and no network: snapshot.ubuntu.com is consulted only by //deploy:relock_debs, which writes
both the lock and the files. A lock entry's `url` records where its file came from. One hub
repository gathers the packages into a single filegroup for //deploy:debs_layer.

    debs = use_extension("//deploy:debs.bzl", "debs")
    debs.lock(name = "reactantserver_debs", lock = "//deploy:debs.lock.json")
    use_repo(debs, "reactantserver_debs")

The lock is written by //deploy:relock_debs; see relock_debs.sh for why it is resolved inside the
base image and why the URLs are snapshot URLs.
"""

# The checked-in file for a lock entry: its snapshot URL's basename, with apt's escaped '+'.
def deb_filename(url):
    return url.rsplit("/", 1)[1].replace("%2b", "+")

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

# The start of a Git LFS pointer file, which is what deploy/debs/ holds when the checkout did not
# fetch the LFS objects.
_LFS_POINTER = "version https://git-lfs"

def _deb_impl(rctx):
    src = rctx.path(rctx.attr.src)
    if rctx.read(src).startswith(_LFS_POINTER):
        fail(("{} is a Git LFS pointer, not the package: fetch the LFS objects with `git lfs pull` " +
              "(in GitHub Actions, check out with `lfs: true`)").format(rctx.attr.src))
    name = rctx.attr.src.name.rsplit("/", 1)[-1]
    # A file:// download, so the lock's sha256 is checked exactly as it was for the snapshot URL.
    rctx.download(url = "file://" + str(src), output = "file/" + name, sha256 = rctx.attr.sha256)
    rctx.file("file/BUILD.bazel", """\
filegroup(
    name = "file",
    srcs = ["{}"],
    visibility = ["//visibility:public"],
)
""".format(name))

_deb = repository_rule(
    implementation = _deb_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".deb"]),
        "sha256": attr.string(mandatory = True),
    },
)

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
                _deb(
                    name = repo,
                    src = Label("//deploy:debs/" + deb_filename(p["url"])),
                    sha256 = p["sha256"],
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
