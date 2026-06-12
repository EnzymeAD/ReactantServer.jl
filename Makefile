# Container-based image builds for ReactantServer. Targets shell out to a container engine
# (podman by default; rootless is fine). The build context for both images is the repository
# root. The gateway is now pure Julia (no Go toolchain or protoc required).
#
#   make            # build the gateway image (default)
#   make gateway    # build the reactant-gateway image
#   make worker     # build the ReactantServer worker image (large: pulls Reactant/CUDA artifacts)
#   make e2e        # full-stack end-to-end test (2 GPU workers + gateway; TCP and SHM)
#   make clean      # remove the images this Makefile builds

SHELL := /bin/bash

ENGINE        ?= podman
GATEWAY_IMAGE ?= reactantserver-gateway:latest
WORKER_IMAGE  ?= reactantserver-worker:latest
LOADGEN_IMAGE ?= reactantserver-loadgen:latest

.PHONY: all gateway worker loadgen e2e clean help

all: gateway

## gateway: build the pure-Julia reactant-gateway image
gateway:
	$(ENGINE) build -f docker/Dockerfile.gateway -t $(GATEWAY_IMAGE) .

## worker: build the ReactantServer worker image (large; needs the lib/ submodules checked out)
worker:
	$(ENGINE) build -f docker/Dockerfile.worker -t $(WORKER_IMAGE) .

## loadgen: build the dummy-data load generator image (light; no Reactant)
loadgen:
	$(ENGINE) build -f docker/Dockerfile.loadgen -t $(LOADGEN_IMAGE) .

## e2e: full-stack end-to-end test (gateway + two GPU workers via podman; TCP and SHM paths)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e.sh

## clean: remove the images built by this Makefile (ignores ones that are absent)
clean:
	-$(ENGINE) rmi $(GATEWAY_IMAGE) $(WORKER_IMAGE)

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
