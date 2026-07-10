# Corporate TLS-inspection certs

Drop a corporate root CA here (PEM, `.crt` extension) if your network runs an HTTPS-inspecting
proxy that re-signs traffic with an internal root. The Dockerfile `COPY docker/certs/` +
`update-ca-certificates` step adds it to the image's system trust store, and
`JULIA_SSL_CA_ROOTS_PATH` points Julia at that bundle, so registry and artifact downloads during the
build succeed through the proxy.

`update-ca-certificates` only processes `*.crt`, so this README is ignored. With no cert present the
step is a no-op. Cert files are gitignored (`docker/certs/*.crt`); never commit them.
