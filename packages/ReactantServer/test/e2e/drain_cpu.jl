# Drain client for the CPU supervisor e2e (run_e2e_cpu.sh): holds slow4 requests in flight through
# the gateway and directly on worker0, sends the supervisor SIGTERM, and checks that the node drains
# from the outside in. The gateway refuses new requests with UNAVAILABLE and reports 503 while the
# workers still serve its in-flight request; once it has exited, worker0 does the same while its
# own in-flight request finishes. Every admitted request completes with its result. slow4 is
# scale4 (y = 2x) with a preprocess hook that sleeps x[1] seconds.
#
#   julia --project=packages/ReactantServerGateway drain_cpu.jl \
#       <gateway_port> <gateway_metrics_port> <worker0_port> <worker0_metrics_port> <supervisor_pid>

import gRPCClient
import HTTP
using Sockets
using ReactantServerGateway
const Inf = ReactantServerGateway.inference

const GATEWAY_PORT, GATEWAY_METRICS_PORT, WORKER0_PORT, WORKER0_METRICS_PORT, SUPERVISOR_PID =
    parse.(Int, ARGS[1:5])

const FAILURES = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    return if cond
        println("PASS  ", name)
    else
        println("FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
        push!(FAILURES, name)
    end
end

function infer(port::Integer, secs::Real)
    x = Float32[secs, 1, 2, 3]
    inp = Inf.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[4])
    req = Inf.ModelInferRequest(;
        model_name = "slow4", inputs = [inp], raw_input_contents = [collect(reinterpret(UInt8, x))]
    )
    resp = Inf.GRPCInferenceService.ModelInfer("127.0.0.1", port, req; deadline = 60)
    return collect(reinterpret(Float32, resp.raw_output_contents[1]))
end

# The gRPC status a call fails with, 0 when it succeeds, -1 for a non-gRPC failure.
function status(f)
    try
        f()
        return 0
    catch e
        return e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : -1
    end
end

function readyz(port::Integer)
    try
        return HTTP.get("http://127.0.0.1:$port/readyz"; retry = false, status_exception = false, readtimeout = 10).status
    catch
        return 0
    end
end

function port_open(port::Integer)
    try
        close(Sockets.connect(Sockets.localhost, port))
        return true
    catch
        return false
    end
end

function wait_until(pred, timeout::Real)
    deadline = time() + timeout
    while time() < deadline
        pred() && return true
        sleep(0.1)
    end
    return pred()
end

const UNAVAILABLE = 14

# Warm slow4's request path on both workers (the gateway round-robins) and directly on worker0.
check("slow4 through the gateway", infer(GATEWAY_PORT, 0) == Float32[0, 2, 4, 6])
check("slow4 through the gateway, other worker", infer(GATEWAY_PORT, 0) == Float32[0, 2, 4, 6])
check("slow4 directly on worker0", infer(WORKER0_PORT, 0) == Float32[0, 2, 4, 6])

# Held in flight: 6 s through the gateway, 14 s directly on worker0. Then stop the node.
via_gateway = Threads.@spawn infer(GATEWAY_PORT, 6)
direct = Threads.@spawn infer(WORKER0_PORT, 14)
sleep(1.5)
println("sending SIGTERM to the supervisor (pid $SUPERVISOR_PID)")
run(`kill -TERM $SUPERVISOR_PID`)

# Stage 1: the gateway drains while the workers keep serving its in-flight request.
check("gateway /readyz reports 503 while draining", wait_until(() -> readyz(GATEWAY_METRICS_PORT) == 503, 10))
s = status(() -> infer(GATEWAY_PORT, 0))
check("gateway refuses a new request with UNAVAILABLE", s == UNAVAILABLE; detail = "status $s")
check("worker0 still ready while the gateway drains", readyz(WORKER0_METRICS_PORT) == 200)
y = try
    fetch(via_gateway)
catch e
    sprint(showerror, e)
end
check("in-flight request through the gateway finished", y == Float32[12, 2, 4, 6]; detail = "got $y")

# Stage 2: the gateway has exited, and the workers drain.
check("gateway exits after its drain", wait_until(() -> !port_open(GATEWAY_PORT), 30))
check("worker0 /readyz reports 503 while draining", wait_until(() -> readyz(WORKER0_METRICS_PORT) == 503, 15))
s = status(() -> infer(WORKER0_PORT, 0))
check("worker0 refuses a new request with UNAVAILABLE", s == UNAVAILABLE; detail = "status $s")
ready = try
    Inf.GRPCInferenceService.ServerReady("127.0.0.1", WORKER0_PORT, Inf.ServerReadyRequest(); deadline = 10).ready
catch e
    sprint(showerror, e)
end
check("worker0 ServerReady is false while draining", ready === false; detail = "got $ready")
y = try
    fetch(direct)
catch e
    sprint(showerror, e)
end
check("in-flight request on worker0 finished", y == Float32[28, 2, 4, 6]; detail = "got $y")
check("worker0 exits after its drain", wait_until(() -> !port_open(WORKER0_PORT), 30))

if isempty(FAILURES)
    println("CPU e2e drain client: all checks passed")
else
    println("CPU e2e drain client: $(length(FAILURES)) check(s) failed")
    exit(1)
end
