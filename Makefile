# Developer tasks for ReactantServer. The container image builds were removed with the Docker
# deployment tooling; the supported deployment is the native launcher + systemd (see
# private/deploy/INSTALL.native.md).
#
#   make e2e        # native CPU end-to-end test (host processes; no containers)
#   make docs       # build the Documenter site into docs/build/ (CPU only; no GPU needed)
#   make help       # list the available targets

SHELL := /bin/bash

JULIA ?= julia

.PHONY: all e2e docs help

all: help

## e2e: native CPU end-to-end test (supervisor + embedded gateway as host processes; no containers)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e_cpu.sh

## docs: build the Documenter site into docs/build/ (instantiates docs/ first; CPU only)
docs:
	$(JULIA) --project=docs -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
