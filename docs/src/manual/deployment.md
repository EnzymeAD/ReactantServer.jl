# Deployment

A node runs as a single supervisor process (`ReactantServerNode`): it detects every visible GPU,
spawns one single-GPU worker subprocess per device, multiplexes their logs, and restarts children
that die. With two or more workers it also runs an embedded gateway on the public ports; with a
single worker it binds that worker to the public ports directly. The external interface is the same
either way: KServe V2 gRPC on `:8001`, health and metrics on `:8002` (`/readyz`, `/healthz`,
`/metrics`), matching Triton's ports.

The supported deployment is **native** (no containers): the supervisor uses the host's NVIDIA
driver directly. The former Docker image is not used; it aborted at GPU-compile time inside the
container (`Cannot load symbol cublasLtCreate`), which a native run does not hit.

## Running natively

Instantiate the workspace once, then run the node supervisor. `ReactantServerNode.main()` reads
`REACTANT_NODE_FILE` for the node config and honors the standard environment overrides:

```bash
# once, to resolve and precompile the workspace (selects the CUDA build via REACTANT_GPU_*):
REACTANT_GPU=cuda REACTANT_GPU_VERSION=13.1 \
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# run the supervisor across four GPUs, serving a bundle directory:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles \
REACTANT_NODE_FILE=config/node.gpu0123.yaml \
  julia --handle-signals=no --project=packages/ReactantServerNode \
    -e 'using ReactantServerNode; ReactantServerNode.main()'
```

`--handle-signals=no` lets the supervisor's own handler run so it shuts its worker children down on
SIGTERM. `REACTANT_GPU_VERSION` selects the Reactant CUDA build (`12.9` or `13.1`) and must be set
before `instantiate`. `INFERENCE_SERVER_MODEL_DIRS` overrides the node file's model repository
(colon-separated); the runtime tunables under [Node Configuration](@ref) take
`INFERENCE_SERVER_*` overrides the same way. For an always-on service, run this command under a
process manager such as systemd, with `Restart=on-failure` and a `SIGTERM`-based graceful stop
(`KillMode=mixed` pairs with `--handle-signals=no`).

Every model compiles to a device executable on every worker before the gRPC plane accepts traffic,
so first startup is slow (minutes to hours for a large model set). Watch readiness with
`curl -sf http://127.0.0.1:8002/readyz`, not the process state.

## Running under systemd

For an always-on node, run the supervisor from a system service. Put the tunables in an
`EnvironmentFile` and let the unit run the same command as above. Adjust the user, the checkout
path, and the GPU list for your host.

`/etc/reactantserver/reactantserver.env`:

```ini
CUDA_VISIBLE_DEVICES=0,1,2,3
INFERENCE_SERVER_MODEL_DIRS=/path/to/bundles
REACTANT_NODE_FILE=config/node.gpu0123.yaml
REACTANT_GPU=cuda
REACTANT_GPU_VERSION=13.1
```

`/etc/systemd/system/reactantserver.service`:

```ini
[Unit]
Description=ReactantServer node supervisor
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=YOUR_DEPLOY_USER
WorkingDirectory=/path/to/ReactantServer.jl
EnvironmentFile=/etc/reactantserver/reactantserver.env
# Absolute path to julia: systemd does not source your shell rc, so a juliaup install under the
# user's home is not on PATH. `julia --version` in a login shell shows the binary to use here.
ExecStart=/home/YOUR_DEPLOY_USER/.juliaup/bin/julia --handle-signals=no --project=packages/ReactantServerNode -e 'using ReactantServerNode; ReactantServerNode.main()'
Restart=on-failure
RestartSec=10
# First boot compiles every model on every worker (minutes to hours) AFTER the unit is already
# active; systemd cannot gate that (the supervisor sends no sd_notify). Check readiness with
# `curl -sf http://127.0.0.1:8002/readyz`, not `systemctl is-active`.
TimeoutStartSec=infinity
# Graceful stop: SIGTERM to the supervisor only, which drains its workers; pairs with
# --handle-signals=no. Anything still alive after TimeoutStopSec is SIGKILLed.
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
```

Enable and watch it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now reactantserver.service
journalctl -u reactantserver -f
until curl -sf http://127.0.0.1:8002/readyz; do sleep 15; done; echo READY
```

`sudo systemctl stop reactantserver` sends SIGTERM to the supervisor, which drains its workers
within `TimeoutStopSec` before exiting. Run the workspace `Pkg.instantiate()` once (as in the
previous section) before enabling the unit, so the first start is not also resolving dependencies.

## Configuring

The node is described by one YAML node file (see [Node Configuration](@ref)); the supervisor
synthesizes one worker per visible GPU when no `workers:` list is given. Gateway scheduling
(`round_robin` or `lpt_packing`) is covered in [Multi-GPU Gateway](@ref). The commented templates
under `config/` (`node.default.yaml`, `node.yaml`, `node.gpu0123.yaml`, `gateway.yml`) remain as
reference configs.

## Roles

`REACTANT_ROLE` selects what the supervisor runs. The default is `all` (workers plus the embedded
gateway on one host), which is the documented deployment. The `workers` and `gateway` roles exist in
the code to split a deployment across machines, but multi-node is not a shipped example.

## Metrics

One scrape on `:8002` covers everything: with multiple workers the embedded gateway serves its own
`gateway_*` series and fans out to each worker's metrics endpoint, merging them; with a single
worker `:8002` is that worker's own `/metrics`. Each worker tags its series with `worker` and `gpu`
labels. A ready-to-run Prometheus + Grafana stack lives under `config/monitoring/`; because the node
runs natively (not in a container), that stack's Prometheus scrapes the host at
`host.docker.internal:8002` rather than over a Docker network (see `config/monitoring/README.md`).

## Security

ReactantServer is designed to run on a trusted network behind your own perimeter. Be aware of the
following before exposing any endpoint:

- All gRPC traffic (worker and gateway) is cleartext h2c. TLS settings are parsed by the gateway
  config but not yet enforced; a configured cert triggers a startup warning.
- There is no authentication or authorization on the KServe data plane, the worker control-plane
  RPCs (residency and policy), or the Prometheus metrics listener (which binds `0.0.0.0:8002` by
  default).
- Model bundles are trusted input: a bundle's optional `model.jl` executes arbitrary Julia in the
  server process. Only serve bundles you built or audited.
- POSIX shared memory is a local trust boundary. Client-registered regions and the optional
  node-shared host-weight store live in `/dev/shm`; the shared weight regions default to mode `666`
  (world-writable) for friction-free sharing. Set `runtime.shared_host_weights_mode: "660"` on
  production or multi-user systems.
