# Developer tasks for ReactantServer. The supported deployment is running the node supervisor
# natively (see the Deployment page in the docs); the container image is an alternative (docker/).
#
#   make image      # build the reactantserver node image (docker/Dockerfile)
#   make e2e        # native CPU end-to-end test (host processes; no containers)
#   make docs       # build the Documenter site into docs/build/ (CPU only; no GPU needed)
#   make clean      # remove the image this Makefile builds
#   make help       # list the available targets

SHELL := /bin/bash

ENGINE     ?= podman
NODE_IMAGE ?= reactantserver:latest
JULIA      ?= julia

.PHONY: all image e2e docs clean help

all: help

## image: build the reactantserver node image (needs lib/gRPCServer.jl + a local Manifest.toml; see docker/README.md)
image:
	$(ENGINE) build -f docker/Dockerfile -t $(NODE_IMAGE) .

## e2e: native CPU end-to-end test (supervisor + embedded gateway as host processes; no containers)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e_cpu.sh

## docs: build the Documenter site into docs/build/ (instantiates docs/ first; CPU only)
docs:
	$(JULIA) --project=docs -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

## clean: remove the image built by this Makefile (ignores it if absent)
clean:
	-$(ENGINE) rmi $(NODE_IMAGE)

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
