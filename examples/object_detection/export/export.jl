# Step 1 of 3: export the object detection bundles.
#
#   julia --project=examples/object_detection/export examples/object_detection/export/export.jl
#
# Writes examples/object_detection/bundles/{object_detector, _stage1, _stage2}. Skips if present
# (delete the bundles dir to re-export). Drives the shared converter in-process so torch imports
# before Reactant (the converter's required order).

const CONFIG    = normpath(joinpath(@__DIR__, "..", "detector.convert.yaml"))
const CONVERTER = normpath(joinpath(@__DIR__, "..", "..", "..", "tools", "convert_to_stablehlo.jl"))

# Corporate SSL: torchvision downloads pretrained weights with Python's urllib, which honors
# SSL_CERT_FILE (not CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE). Point it at the OS trust store that holds
# the corporate CA (the same bundle the repo's Docker images use via update-ca-certificates).
if !haskey(ENV, "SSL_CERT_FILE")
    for candidate in (get(ENV, "REQUESTS_CA_BUNDLE", ""), get(ENV, "CURL_CA_BUNDLE", ""),
                      get(ENV, "JULIA_SSL_CA_ROOTS_PATH", ""),
                      "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
        if !isempty(candidate) && isfile(candidate)
            ENV["SSL_CERT_FILE"] = candidate
            @info "Using system CA bundle for Python TLS" SSL_CERT_FILE = candidate
            break
        end
    end
end

# Hand the converter its CLI args, then run it (its own top-level loads PythonCall before Reactant).
empty!(ARGS); append!(ARGS, [CONFIG, "--only", "object_detector"])
include(CONVERTER)
