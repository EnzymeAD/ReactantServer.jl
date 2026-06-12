#!/usr/bin/env bash
# Entrypoint for the load generator container: run the soak driver against the gateway. Load
# parameters come from the LOADGEN_* environment variables (see loadgen.jl); JULIA_NUM_THREADS is
# read by Julia directly. Run via docker-compose.gpu2.yml.
set -euo pipefail

exec julia --project=/opt/reactantserver/packages/ReactantServerClient /usr/local/bin/loadgen.jl
