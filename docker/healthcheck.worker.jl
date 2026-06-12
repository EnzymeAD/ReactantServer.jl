# Container healthcheck for a ReactantServer worker. Calls the worker's KServe ServerReady over
# loopback and exits 0 (ready) or 1 (not ready). Deliberately lightweight: it imports only
# gRPCClient and YAML, never ReactantServer, so it does not pay the Reactant load on every probe.
# It uses gRPCClient's raw Vector{UInt8} support to issue ServerReady with an empty request and
# reads the single `ready` bool (field 1) from the response.

import gRPCClient
import YAML

const NODE = get(ENV, "REACTANT_NODE_FILE", "/etc/reactantserver/node.yaml")
const WORKER = get(ENV, "REACTANT_WORKER_NAME", "")

# Resolve a worker's listen port the same way the package does: an explicit `port`, else
# `base_port` + the worker's index in declaration order.
function worker_port(node::AbstractDict, name::AbstractString)
    workers = node["workers"]
    base = get(node, "base_port", nothing)
    for (i, w) in enumerate(workers)
        wname = String(w["name"])
        (isempty(name) || wname == name) || continue
        haskey(w, "port") && return Int(w["port"])
        base === nothing && error("worker '$wname' has no port and base_port is unset")
        return Int(base) + (i - 1)
    end
    error("worker '$name' not found in node file")
end

# ServerReadyResponse has a single `ready` bool at field 1; absent means false (proto3 default).
function ready_from_response(body::AbstractVector{UInt8})
    i = firstindex(body)
    while i <= lastindex(body)
        tag = body[i]; i += 1
        field = tag >> 3
        wiretype = tag & 0x07
        if field == 1 && wiretype == 0
            v = 0; shift = 0
            while i <= lastindex(body)
                b = body[i]; i += 1
                v |= Int(b & 0x7f) << shift
                (b & 0x80) == 0 && break
                shift += 7
            end
            return v != 0
        end
        return false  # ServerReadyResponse carries no other fields
    end
    return false
end

function main()
    node = YAML.load_file(NODE; dicttype = Dict{String,Any})
    port = worker_port(node, WORKER)
    client = gRPCClient.gRPCServiceClient{Vector{UInt8},false,Vector{UInt8},false}(
        "127.0.0.1", port, "/inference.GRPCInferenceService/ServerReady"; deadline = 5)
    ready = try
        ready_from_response(gRPCClient.grpc_sync_request(client, UInt8[]))
    catch
        false
    end
    exit(ready ? 0 : 1)
end

main()
