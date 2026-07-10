# Container healthcheck for ReactantServer workers. Calls KServe ServerReady over loopback and
# exits 0 (ready) or 1 (not ready). With REACTANT_WORKER_NAME set it probes that worker alone
# (the per-GPU container layout); with it unset it probes every worker in the node file and
# reports ready when at least one is, the right semantic for a supervised multi-GPU container
# where one failed GPU must not get the whole node killed. Deliberately lightweight: it imports
# only gRPCClient and YAML, never ReactantServer, so it does not pay the Reactant load on every
# probe. It uses gRPCClient's raw Vector{UInt8} support to issue ServerReady with an empty
# request and reads the single `ready` bool (field 1) from the response.

import gRPCClient
import YAML

const NODE = get(ENV, "REACTANT_NODE_FILE", "/etc/reactantserver/node.yaml")
const WORKER = get(ENV, "REACTANT_WORKER_NAME", "")

# Resolve worker listen ports the same way the package does: an explicit `port`, else
# `base_port` + the worker's index in declaration order. An empty `name` selects every worker.
function worker_ports(node::AbstractDict, name::AbstractString)
    workers = node["workers"]
    base = get(node, "base_port", nothing)
    ports = Int[]
    for (i, w) in enumerate(workers)
        wname = String(w["name"])
        (isempty(name) || wname == name) || continue
        if haskey(w, "port")
            push!(ports, Int(w["port"]))
        else
            base === nothing && error("worker '$wname' has no port and base_port is unset")
            push!(ports, Int(base) + (i - 1))
        end
    end
    isempty(ports) && error("worker '$name' not found in node file")
    return ports
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

function probe(port::Int)
    client = gRPCClient.gRPCServiceClient{Vector{UInt8},false,Vector{UInt8},false}(
        "127.0.0.1", port, "/inference.GRPCInferenceService/ServerReady"; deadline = 5)
    return try
        ready_from_response(gRPCClient.grpc_sync_request(client, UInt8[]))
    catch
        false
    end
end

function main()
    # The supervisor leaves its materialized node file (synthesized workers list) at the
    # conventional runtime path; prefer it so auto-detected workers are probed too.
    node_file = isfile("/run/reactantserver/node.yaml") && isempty(WORKER) ?
                "/run/reactantserver/node.yaml" : NODE
    node = YAML.load_file(node_file; dicttype = Dict{String,Any})
    exit(any(probe, worker_ports(node, WORKER)) ? 0 : 1)
end

main()
