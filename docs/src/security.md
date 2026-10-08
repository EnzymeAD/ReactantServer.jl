# Security

ReactantServer is designed to run on a trusted network behind your own perimeter. Be aware of the
following before exposing any endpoint:

- **Cleartext h2c.** All gRPC traffic (worker and gateway) is cleartext h2c. TLS settings are
  parsed by the gateway config but not yet enforced; a configured cert triggers a startup warning.
- **No authentication or authorization** on the KServe data plane, the worker control-plane RPCs
  (residency and policy), the gateway's own scheduling control plane (`GatewayControlService`,
  which can repack the fleet and change a model's replica count and routing at runtime), or the
  Prometheus metrics listener (which binds `0.0.0.0:8002` by default). The control plane is
  mutating, so treat access to the gRPC ports as administrative access.
- **Trusted bundles.** A bundle's optional `model.jl` executes arbitrary Julia in the server
  process, so bundles are trusted input. Only serve bundles you built or audited; see
  [Bundles](bundles.md).
- **Shared memory is a local trust boundary.** Client-registered regions and the optional
  node-shared host-weight store live in `/dev/shm`; the shared weight regions default to mode
  `666` (world-writable) for friction-free sharing. Set `runtime.shared_host_weights_mode: "660"`
  on production or multi-user systems.
