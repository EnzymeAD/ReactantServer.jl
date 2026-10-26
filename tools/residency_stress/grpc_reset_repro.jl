# Repro for a gRPCClient behavior the stress suite hit through the gateway: an in-flight unary call
# whose gRPCCURL handle is shut down and re-initialized (the gateway's reset_clients!, run when a
# worker probe hits its deadline) returns normally with an empty response instead of raising. The
# server is a TCP black hole that accepts and never answers, like a stalled worker.
#
#   julia -t 4 --project=packages/ReactantServerGateway tools/residency_stress/grpc_reset_repro.jl
#
# Prints one line per call; "RETURNED normally ... length 0" is the bug.
import gRPCClient
using Sockets

srv = listen(ip"127.0.0.1", 0)
port = Int(getsockname(srv)[2])
Threads.@spawn while true
    c = accept(srv)
    Threads.@spawn (
        try
            while isopen(c)
                readavailable(c)
            end
        catch
        end
    )
end

grpc = gRPCClient.gRPCCURL()
gRPCClient.grpc_init(grpc)
client = gRPCClient.gRPCServiceClient{Vector{UInt8}, false, Vector{UInt8}, false}(
    "127.0.0.1", port, "/inference.GRPCInferenceService/ModelInfer"; grpc = grpc, deadline = 20
)
println("firing 4 calls at a server that never answers"); flush(stdout)
tasks = [
    Threads.@spawn try
        r = gRPCClient.grpc_sync_request(client, UInt8[0x0a, 0x01, 0x61])
        "RETURNED normally: $(typeof(r)) of length $(length(r))"
    catch e
        "threw: " * first(sprint(showerror, e), 120)
    end
        for _ in 1:4
]
sleep(2)
println("reset: grpc_shutdown + grpc_init with the calls in flight"); flush(stdout)
gRPCClient.grpc_shutdown(grpc)
gRPCClient.grpc_init(grpc)
for t in tasks
    println(fetch(t)); flush(stdout)
end
exit(0)
