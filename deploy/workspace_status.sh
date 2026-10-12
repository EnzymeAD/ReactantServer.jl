#!/bin/sh
# Bazel's workspace status command (.bazelrc). A --stamp build of //deploy:image_labels reads
# STABLE_REACTANTSERVER_COMMIT into the image's org.opencontainers.image.revision label.
#
# The commit is printed only from a checkout whose tracked files match it. Outside git, or with
# uncommitted changes, nothing is printed and the image carries no revision, rather than one naming
# a commit the image was not built from. Untracked files (user.bazelrc, scratch) do not count.
#
# The key is prefixed because a module that builds the image through a bazel_dep runs its own
# status command, and a generic STABLE_GIT_COMMIT there would name the consumer's commit.
commit="$(git rev-parse --verify HEAD 2>/dev/null)" || exit 0
[ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || exit 0
echo "STABLE_REACTANTSERVER_COMMIT $commit"
