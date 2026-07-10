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

The launcher and systemd unit, with full site-specific install/enable/journal steps, are documented
in `private/deploy/INSTALL.native.md`. In short:

```bash
# manual (from the checkout):
MODELS=/path/to/bundles GPUS=0,1,2,3 CUDA_MAJOR_MINOR=13.1 \
  private/deploy/serve_native.sh
```

`serve_native.sh` instantiates the workspace, generates the node config, exports the runtime
environment, and execs the supervisor. `CUDA_MAJOR_MINOR` selects the Reactant CUDA build (`12.9`
or `13.1`). For an always-on service, install the `reactantserver.service` systemd unit (see the
install notes): it runs the launcher as the deploy user with `Restart=on-failure` and a graceful
SIGTERM stop that drains the workers.

Every model compiles to a device executable on every worker before the gRPC plane accepts traffic,
so first startup is slow (minutes to hours for a large model set). Watch readiness with
`curl -sf http://127.0.0.1:8002/readyz`, not the process state.

## Configuring

The node is described by one YAML node file (see [Node Configuration](@ref)); the supervisor
synthesizes one worker per visible GPU when no `workers:` list is given. Gateway scheduling
(`round_robin` or `lpt_packing`) is covered in [Multi-GPU Gateway](@ref). The commented templates
under `docker/` (`node.default.yaml`, `node.yaml`, `node.gpu0123.yaml`, `gateway.yml`) remain as
reference configs.

## Roles

`REACTANT_ROLE` selects what the supervisor runs. The default is `all` (workers plus the embedded
gateway on one host), which is the documented deployment. The `workers` and `gateway` roles exist in
the code to split a deployment across machines, but multi-node is not a shipped example.

## Metrics

One scrape on `:8002` covers everything: with multiple workers the embedded gateway serves its own
`gateway_*` series and fans out to each worker's metrics endpoint, merging them; with a single
worker `:8002` is that worker's own `/metrics`. Each worker tags its series with `worker` and `gpu`
labels. A ready-to-run Prometheus + Grafana stack lives under `docker/monitoring/`.

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
