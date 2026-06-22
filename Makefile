# Container-based image builds for ReactantServer. Targets shell out to a container engine
# (podman by default; rootless is fine). The build context for the images is the repository
# root.
#
#   make            # build the node image (default)
#   make image      # build the reactantserver node image (supervisor: workers + embedded gateway)
#   make loadgen    # build the dummy-data load generator image (light; no Reactant)
#   make e2e        # full-stack end-to-end test (supervised multi-GPU container; TCP and SHM)
#   make docs       # build the Documenter site into docs/build/ (CPU only; no GPU needed)
#   make clean      # remove the images this Makefile builds

SHELL := /bin/bash

ENGINE        ?= podman
NODE_IMAGE    ?= reactantserver:latest
LOADGEN_IMAGE ?= reactantserver-loadgen:latest
JULIA         ?= julia

.PHONY: all image loadgen e2e docs clean help

all: image

## image: build the reactantserver node image (large; needs the lib/ submodules checked out)
image:
	$(ENGINE) build -f docker/Dockerfile -t $(NODE_IMAGE) .

## loadgen: build the dummy-data load generator image (light; no Reactant)
loadgen:
	$(ENGINE) build -f docker/Dockerfile.loadgen -t $(LOADGEN_IMAGE) .

## e2e: full-stack end-to-end test (supervised multi-GPU container via podman; TCP and SHM paths)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e.sh

## docs: build the Documenter site into docs/build/ (instantiates docs/ first; CPU only)
docs:
	$(JULIA) --project=docs -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

## clean: remove the images built by this Makefile (ignores ones that are absent)
clean:
	-$(ENGINE) rmi $(NODE_IMAGE) $(LOADGEN_IMAGE)

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
