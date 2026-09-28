#!/usr/bin/env bash
# Re-resolve deploy/Manifest.toml for the current workspace and write it back into the source tree.
#
#   bazel run //deploy:relock                           # keep every version that still resolves
#   bazel run //deploy:relock -- Reactant Reactant_jll  # also move the named packages forward
#
# Resolution needs only the registry; no artifacts are downloaded. Review the diff like code: it is
# the production lock.
set -euo pipefail

: "${BUILD_WORKSPACE_DIRECTORY:?run this with bazel run //deploy:relock}"
julia="$(readlink -f "$RS_JULIA")"
root="$BUILD_WORKSPACE_DIRECTORY"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$root/Project.toml" "$stage/"
# The same member set as //:workspace_projects: every package except the offline export tooling.
for p in "$root"/packages/*/Project.toml; do
    case "$p" in */ReactantServerExport/*) continue ;; esac
    rel="${p#"$root"/}"
    mkdir -p "$stage/$(dirname "$rel")"
    cp "$p" "$stage/$rel"
done
if [ -f "$root/deploy/Manifest.toml" ]; then
    cp "$root/deploy/Manifest.toml" "$stage/Manifest.toml"
fi

"$julia" --startup-file=no --project="$stage" -e '
using Pkg
isempty(ARGS) ? Pkg.resolve() : Pkg.update(ARGS)
Pkg.status(["Reactant", "Reactant_jll", "gRPCServer", "gRPCClient", "HTTP"]; mode = Pkg.PKGMODE_MANIFEST)
' "$@"

cp "$stage/Manifest.toml" "$root/deploy/Manifest.toml"
echo "wrote $root/deploy/Manifest.toml"
