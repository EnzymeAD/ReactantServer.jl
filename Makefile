# Developer tasks for ReactantServer. The supported deployment is running the node supervisor
# natively (see the Deployment page in the docs); the container image, built with Bazel, is an
# alternative (deploy/).
#
#   make image      # build the locked node image with Bazel and load it into podman (deploy/)
#   make e2e        # native CPU end-to-end test (host processes; no containers)
#   make docs       # build the Documenter site into docs/build/ (CPU only; no GPU needed)
#   make clean      # remove the image this Makefile builds
#   make help       # list the available targets

SHELL := /bin/bash

ENGINE     ?= podman
NODE_IMAGE ?= localhost/reactantserver:bazel
JULIA      ?= julia

.PHONY: all image e2e docs clean help

all: help

## image: build the locked node image (deploy/Manifest.toml) with Bazel and load it into podman as localhost/reactantserver:bazel
image:
	bazel run //deploy:image_load

## e2e: native CPU end-to-end test (supervisor + embedded gateway as host processes; no containers)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e_cpu.sh

## docs: build the Documenter site into docs/build/ (instantiates docs/ first; CPU only)
docs:
	$(JULIA) --project=docs -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

## clean: remove the image loaded by this Makefile (ignores it if absent)
clean:
	-$(ENGINE) rmi $(NODE_IMAGE)

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
