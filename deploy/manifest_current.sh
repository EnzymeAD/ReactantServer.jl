#!/usr/bin/env bash
# Fail if deploy/Manifest.toml is not current for the workspace's Project.toml files, or was
# resolved under a different Julia than the one //deploy builds with.
set -euo pipefail

julia="$(readlink -f "$RS_JULIA")"
stage="$TEST_TMPDIR/workspace"
mkdir -p "$stage"
for f in $RS_PROJECTS; do
    mkdir -p "$stage/$(dirname "$f")"
    cp -L "$f" "$stage/$f"
done
cp -L "$RS_MANIFEST" "$stage/Manifest.toml"

# The trailing separator keeps Julia's bundled depots (the stdlib caches) on the path.
JULIA_DEPOT_PATH="$TEST_TMPDIR/depot:" "$julia" --startup-file=no --project="$stage" -e '
using Pkg, TOML
m = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
ok = true
if m["julia_version"] != string(VERSION)
    println(stderr, "FAIL: deploy/Manifest.toml was resolved under Julia ", m["julia_version"],
        " but the pinned distribution is ", VERSION, ".")
    ok = false
end
if !Pkg.is_manifest_current(Pkg.Types.Context())
    println(stderr, "FAIL: deploy/Manifest.toml is not current for the workspace Project.toml files.")
    ok = false
end
ok || (println(stderr, "\nRe-resolve it with: bazel run //deploy:relock"); exit(1))
println("ok: deploy/Manifest.toml is current (Julia ", VERSION, ")")
'
