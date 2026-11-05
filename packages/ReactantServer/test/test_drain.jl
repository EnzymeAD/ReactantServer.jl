# Graceful drain of a worker. Once `stop!(srv; drain_seconds)` starts, a new ModelInfer gets
# UNAVAILABLE and every readiness signal (ServerReady, ModelReady, RepositoryIndex, /readyz)
# reports not ready, while the calls already admitted finish. A blocking `serve` drains and tears
# everything down when a drain is requested, and announces `:up` only once its port is bound.

import HTTP
using Sockets

const _DrInf = ReactantServer.inference
const _DrCore = ReactantServer.ReactantServerCore

const _DRAIN_MANIFEST = """
format_version: "2.0"
name: slow4
executable_inputs:
  - {name: x, dtype: f32, shape: c, dims: {c: 4}}
executable_outputs:
  - {name: y, dtype: f32, shape: c, dims: {c: 4}}
batching: {compiled_batch_sizes: [1]}
"""

const _DRAIN_MLIR = """
module {
  func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.multiply %x, %w : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

# y = 2x, with a preprocess hook that sleeps x[1] seconds first, so a request can be held in flight.
function _write_slow_bundle(root)
    dir = write_bundle(
        root, "slow4"; manifest_yaml = _DRAIN_MANIFEST, mlir_text = _DRAIN_MLIR,
        weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"]
    )
    write(
        joinpath(dir, "model.jl"), """
        register_model("slow4"; preprocess = function (inputs)
            sleep(Float64(inputs[1].data[1]))
            return inputs
        end)
        """
    )
    return dir
end

_drain_cfg(root, port, mport) = ReactantServer.ServerConfig(
    [root], "",
    ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
    ReactantServer.SchedulerConfig(30.0, 64, 30.0),
    ReactantServer.EndpointsConfig("127.0.0.1", port, mport)
)

function _slow_infer(port, secs)
    x = Float32[secs, 1, 2, 3]
    inp = _DrInf.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[4])
    req = _DrInf.ModelInferRequest(;
        model_name = "slow4", inputs = [inp], raw_input_contents = [collect(reinterpret(UInt8, x))]
    )
    resp = WorkerRPC.ModelInfer("127.0.0.1", port, req; deadline = 60)
    return collect(reinterpret(Float32, resp.raw_output_contents[1]))
end

_grpc_status(f) = try
    f()
    nothing
catch e
    e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : e
end

_readyz(mport) = HTTP.get("http://127.0.0.1:$mport/readyz"; retry = false, status_exception = false, readtimeout = 30).status

_port_open(port) = try
    close(Sockets.connect(Sockets.localhost, port))
    true
catch
    false
end

@testset "graceful drain (CPU)" begin
    mktempdir() do root
        _write_slow_bundle(root)
        port, mport = grpc_free_port(), grpc_free_port()
        srv = ReactantServer.serve(_drain_cfg(root, port, mport); backend = ReactantServer.ReactantBackend(), blocking = false)
        stopper = nothing
        try
            @test _slow_infer(port, 0) == Float32[0, 2, 4, 6]       # warm the request path
            @test _readyz(mport) == 200

            # One request held in flight for 4 s, then the drain starts.
            inflight = Threads.@spawn _slow_infer(port, 4)
            @test timedwait(() -> srv.drain.inflight[] == 1, 30) === :ok
            stopper = Threads.@spawn ReactantServer.stop!(srv; drain_seconds = 30)
            @test timedwait(() -> _DrCore.is_draining(srv.drain), 10) === :ok

            # Draining: new inference is refused with UNAVAILABLE, and the worker reports not ready.
            @test _grpc_status(() -> _slow_infer(port, 0)) == Int(ReactantServer._G.StatusCode.UNAVAILABLE)
            @test !grpc_call(WorkerRPC.ServerReady, port, _DrInf.ServerReadyRequest()).ready
            @test !grpc_call(WorkerRPC.ModelReady, port, _DrInf.ModelReadyRequest(; name = "slow4")).ready
            @test isempty(grpc_call(WorkerRPC.RepositoryIndex, port, _DrInf.RepositoryIndexRequest(; ready = true)).models)
            all_models = grpc_call(WorkerRPC.RepositoryIndex, port, _DrInf.RepositoryIndexRequest(; ready = false)).models
            @test [(m.name, m.state) for m in all_models] == [("slow4", "UNAVAILABLE")]
            @test grpc_call(WorkerRPC.ServerLive, port, _DrInf.ServerLiveRequest()).live
            @test _readyz(mport) == 503
            @test !istaskdone(stopper)                                # still waiting for the request

            # The admitted request finishes with its result, and only then does the server stop.
            @test fetch(inflight) == Float32[8, 2, 4, 6]
            @test timedwait(() -> istaskdone(stopper), 60) === :ok
            fetch(stopper)
            stopper = nothing
            @test !_port_open(port)
            @test !_port_open(mport)
            @test istaskdone(srv.scheduler.task)                      # the dispatch loop was joined
        finally
            stopper === nothing ? ReactantServer.stop!(srv) : wait(stopper)
        end
    end
end

@testset "blocking serve: :up once bound, drains and returns on request" begin
    mktempdir() do root
        _write_slow_bundle(root)
        port, mport = grpc_free_port(), grpc_free_port()
        up_connects = Bool[]
        events = Symbol[]
        beat = ReactantServer.register_heartbeat!(
            (ev, _) -> begin
                ev in (:up, :down) || return
                push!(events, ev)
                ev === :up && push!(up_connects, _port_open(port))
            end
        )
        _DrCore._reset_drain!()
        server = Threads.@spawn ReactantServer.serve(_drain_cfg(root, port, mport); backend = ReactantServer.ReactantBackend())
        try
            @test timedwait(() -> :up in events || istaskdone(server), 600) === :ok
            @test up_connects == [true]                               # the port accepted at :up
            @test _slow_infer(port, 0) == Float32[0, 2, 4, 6]
            inflight = Threads.@spawn _slow_infer(port, 2)
            @test timedwait(() -> _readyz(mport) == 200 && !istaskdone(inflight), 30) === :ok
            sleep(0.5)                                               # let the request be admitted
            @test request_drain!(30)                                 # a serving server acts on it
            @test fetch(inflight) == Float32[4, 2, 4, 6]             # admitted before the drain
            @test timedwait(() -> istaskdone(server), 60) === :ok
            @test fetch(server) === nothing
            @test events == [:up, :down]
            @test !_port_open(port)
            @test !_port_open(mport)
        finally
            istaskdone(server) || (request_drain!(0); wait(server))
            _DrCore._reset_drain!()
            lock(() -> filter!(f -> f !== beat, ReactantServer._HEARTBEATS), ReactantServer._HEARTBEAT_LOCK)
        end
    end
end
