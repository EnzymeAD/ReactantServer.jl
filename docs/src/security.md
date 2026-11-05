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
- **Internal errors are redacted by default.** With `grpc.error_detail: redacted` (the default,
  set node-wide under `global.grpc`), a client is told of a failure inside the server only by a
  request id and the worker's (or gateway's) name, and the gateway never names a worker by host
  and port; the exception and backtrace go to that server's log under the same id. Keep
  `full`, which sends the exception text to clients, to development and testing. See
  [Node Configuration](node_config.md).
- **The receive size limit is a memory commitment.** The worker and gateway gRPC servers allocate a
  buffer of a request message's full declared length as soon as its 5-byte length prefix arrives,
  before the payload. A client that sends a length prefix and nothing more therefore makes the
  process hold a buffer of up to `grpc.max_recv_msg_bytes` (512 MiB by default) for as long as the
  request stays open, and can do so on as many requests at once as the server admits. Where
  untrusted or unauthenticated clients can reach a worker or the gateway, lower
  `grpc.max_recv_msg_bytes` on both to the largest inline message the deployment needs (see
  [Node Configuration](node_config.md); tensors sent through shared memory do not count). On a node
  set it in `global.grpc`, which the supervisor mirrors into the embedded gateway. An environment
  override wins over the YAML value, so also set or clear `INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES`
  (workers) and `REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES` (gateway).
