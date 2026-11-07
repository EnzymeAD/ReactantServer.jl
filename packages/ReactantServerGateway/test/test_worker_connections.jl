# The gateway's connections to a worker (GW-1). The health probes used to share one handle, and with
# it `max_concurrent_streams` slots, with inference. A probe's deadline counts the wait for a slot, so
# with every slot held by inference queued at a busy worker the probe timed out, the handle was reset,
# every call in flight on it was aborted (inline ones ran twice, shared-memory ones failed), and three
# such rounds made the gateway exit. Now the probes have a handle of their own, and the inference
# handle is reset only when the inference calls themselves show that its connection is parked behind
# one that never answered (gRPCClient 1.2.1 with CURLOPT_PIPEWAIT; GW-18). Reuses `_slow_worker`,
# `_gw_slow`, `_gw_status`, `_gw_readyz` and `_gateway_file` from test_drain.jl, and `MockWorker`,
# `_start_mock` and `_infer` from test_gateway.jl.

import gRPCClient
using Sockets

const _CGW = ReactantServerGateway

# Serve a gateway over the workers in `path` with a fast prober, and with the wedge exit far out of
# reach, so that a regression shows up as a test failure rather than as the test process exiting.
function _probed_gateway(path)
    env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "0.5", "REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS" => "1000000"]
    return withenv(() -> _CGW.serve_gateway(path; blocking = false), env...)
end

@testset "a worker holding every inference slot neither loses its calls nor fails its probes (GW-1)" begin
    worker, wport = _slow_worker()
    path = _gateway_file(wport)
    write(path, replace(read(path, String), "endpoints:" => "grpc:\n  max_concurrent_requests_per_worker: 0\nendpoints:"))
    gw = _probed_gateway(path)
    # The test's own client handle, with room for every call at once (gRPCClient's default is 16).
    tgrpc = gRPCClient.gRPCCURL(; max_streams = 64)
    try
        gport, aport = _CGW.bound_ports(gw)
        @test timedwait(() -> _gw_status(() -> _gw_slow(gport, 0)) === nothing, 30) === :ok
        @test timedwait(() -> _gw_readyz(aport) == 200, 30) === :ok
        wc = only(_CGW.all_clients(gw.pool))
        slots = gw.cfg.max_concurrent_streams_per_worker
        @test slots == 32

        # Every slot to the worker held, for 16 s, by clients that each send a 6 s call (longer than
        # the 5 s probe deadline) as soon as their previous one ends.
        tchan = gRPCClient.gRPCChannel("127.0.0.1", gport; grpc = tgrpc)
        slow(secs) = _DGInf.GRPCInferenceService.ModelInfer(
            tchan, _DGInf.ModelInferRequest(; model_name = "slow", id = string(secs)); deadline = 30
        )
        # One answered call first: libcurl holds back the calls on a new connection until it has
        # answered one (CURLOPT_PIPEWAIT), which would send the calls below one at a time.
        @test slow(0).id == "0"
        ok = Threads.Atomic{Int}(0)
        failed = Threads.Atomic{Int}(0)
        stop_at = time() + 16
        clients = map(1:slots) do _
            Threads.@spawn while time() < stop_at
                answered = try
                    slow(6).id == "6"
                catch
                    false
                end
                Threads.atomic_add!(answered ? ok : failed, 1)
            end
        end
        @test timedwait(() -> gw.drain.inflight[] == slots, 30) === :ok
        ready = Bool[]
        wedged = Int[]
        while !all(istaskdone, clients)        # some 40 probe rounds while the slots are full
            push!(ready, _gw_readyz(aport) == 200)
            push!(wedged, gw.prober.wedged_rounds)
            sleep(0.5)
        end
        # Every call finished with its own answer: none was aborted (and, with another replica,
        # run twice).
        @test failed[] == 0
        @test ok[] >= 2 * slots
        # The probes kept answering, so the gateway stayed ready and never counted a round toward
        # its wedge exit, and the inference connection was never reset.
        @test all(ready)
        @test maximum(wedged) == 0
        @test wc.resets[] == 0 && wc.unanswered[] == 0
    finally
        gRPCClient.grpc_shutdown(tgrpc)
        _CGW.stop!(gw)
        gRPCServer.stop!(worker; force = true)
    end
end

@testset "invoke_infer counts calls that reach their deadline unanswered; any answer clears the count" begin
    body(model) = (io = IOBuffer(); ReactantServerGateway.PB.encode(ReactantServerGateway.PB.ProtoEncoder(io), GWInf.ModelInferRequest(; model_name = model)); take!(io))
    # A worker that accepts and never answers.
    srv = Sockets.listen(Sockets.localhost, 0)
    hport = Int(Sockets.getsockname(srv)[2])
    held = Any[]
    acceptor = @async try
        while isopen(srv)
            push!(held, Sockets.accept(srv))
        end
    catch
    end
    grpc = gRPCClient.gRPCCURL(; sticky = true, max_streams = 1)
    chan = gRPCClient.gRPCChannel("127.0.0.1", hport; grpc = grpc, deadline = 5)
    wc = _CGW.WorkerClients("127.0.0.1:$hport", grpc, chan, 5.0, 5.0)
    status(f) = try
        f()
        nothing
    catch e
        e.grpc_status
    end
    try
        # The first call takes the handle's only slot and reaches its deadline unanswered; the second
        # expires waiting for the slot, which says nothing about the connection and is not counted.
        first_call = @async status(() -> _CGW.invoke_infer(wc, body("m"); deadline = 2))
        sleep(0.2)
        @test status(() -> _CGW.invoke_infer(wc, body("m"); deadline = 1)) == gRPCClient.GRPC_DEADLINE_EXCEEDED
        @test fetch(first_call) == gRPCClient.GRPC_DEADLINE_EXCEEDED
        @test wc.unanswered[] == 1
        # Below the threshold nothing is reset; at it, the handle is reset once and the count cleared.
        @test !_CGW.reset_if_parked!(wc)
        wc.unanswered[] = _CGW.PARKED_AFTER_UNANSWERED
        @test _CGW.reset_if_parked!(wc)
        @test wc.unanswered[] == 0 && wc.resets[] == 1
        @test !_CGW.reset_if_parked!(wc)
    finally
        gRPCClient.grpc_shutdown(grpc)
        close(srv)
        foreach(close, held)
    end

    # Any answer from a worker clears the count, a status it sends (here NOT_FOUND) as much as a success.
    w = MockWorker("answers", ["m"], false, false, false)
    s, p = _start_mock(w)
    grpc = gRPCClient.gRPCCURL(; sticky = true)
    chan = gRPCClient.gRPCChannel("127.0.0.1", p; grpc = grpc, deadline = 5)
    wc = _CGW.WorkerClients("127.0.0.1:$p", grpc, chan, 5.0, 5.0)
    try
        wc.unanswered[] = 3
        @test status(() -> _CGW.invoke_infer(wc, body("nope"))) == gRPCClient.GRPC_NOT_FOUND
        @test wc.unanswered[] == 0
        wc.unanswered[] = 3
        @test status(() -> _CGW.invoke_infer(wc, body("m"))) === nothing
        @test wc.unanswered[] == 0
    finally
        gRPCClient.grpc_shutdown(grpc)
        close(s)
    end
end

# A TCP proxy in front of a worker. While `hold` is set, a connection it accepts is read and never
# answered (the worker's silent window while it starts, or a frozen worker); otherwise the connection
# is relayed to the worker. A connection keeps the treatment it got when it was accepted.
mutable struct _HoldProxy
    server::Sockets.TCPServer
    port::Int
    hold::Threads.Atomic{Bool}
    socks::Vector{Any}
end

function _relay(from, to)
    return @async try
        while !eof(from)
            write(to, readavailable(from))
        end
    catch
    finally
        try
            close(to)
        catch
        end
    end
end

function _hold_proxy(upstream::Integer)
    srv = Sockets.listen(Sockets.localhost, 0)
    p = _HoldProxy(srv, Int(Sockets.getsockname(srv)[2]), Threads.Atomic{Bool}(false), Any[])
    @async while isopen(srv)
        c = try
            Sockets.accept(srv)
        catch
            break
        end
        push!(p.socks, c)
        if p.hold[]
            @async try
                while !eof(c)
                    readavailable(c)
                end
            catch
            end
        else
            u = Sockets.connect(Sockets.localhost, upstream)
            push!(p.socks, u)
            _relay(c, u)
            _relay(u, c)
        end
    end
    return p
end

function _close_proxy(p::_HoldProxy)
    close(p.server)
    for s in p.socks
        try
            close(s)
        catch
        end
    end
    return nothing
end

@testset "an inference connection that never answered is reset, and the worker is served again (GW-1, GW-18)" begin
    w = MockWorker("worker0", ["m"], false, false, false)
    s, p = _start_mock(w)
    proxy = _hold_proxy(p)
    gw = _probed_gateway(_gateway_file(proxy.port))
    try
        port, aport = _CGW.bound_ports(gw)
        # The probes connect through the proxy and are answered; no inference has been sent yet, so the
        # inference handle has no connection.
        @test timedwait(() -> _CGW.has_model(gw.routes, "m") && _gw_readyz(aport) == 200, 30) === :ok
        wc = only(_CGW.all_clients(gw.pool))
        status(f) = try
            f()
            nothing
        catch e
            e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : e
        end
        # The test's own connection to the gateway answers a call first (the gateway knows no model
        # "nope"), so that the calls below that it gives up on cannot park it on the client side.
        @test status(() -> _infer(port, "nope")) == gRPCClient.GRPC_NOT_FOUND

        # The first inference connection is accepted and never answered, and every call on it is given
        # up at its deadline. libcurl keeps it pooled and, because it never answered, makes every later
        # call wait behind it (CURLOPT_PIPEWAIT) rather than open another.
        proxy.hold[] = true
        quick(model) = grpc_call(
            GWInf.GRPCInferenceService.ModelInfer, port, GWInf.ModelInferRequest(; model_name = model); deadline = 1
        )
        given_up = asyncmap(_ -> status(() -> quick("m")), 1:(_CGW.PARKED_AFTER_UNANSWERED + 1))
        @test all(==(gRPCClient.GRPC_DEADLINE_EXCEEDED), given_up)
        @test timedwait(() -> wc.unanswered[] >= _CGW.PARKED_AFTER_UNANSWERED || wc.resets[] > 0, 5) === :ok
        @test w.infer_calls == 0                # none of them reached the worker
        # The worker is reachable again, but the calls would keep waiting on the dead connection until
        # libcurl dropped it after 118 s idle. The probes answer on their own connection, so the
        # gateway resets the inference handle, and the worker is served again within a few rounds.
        proxy.hold[] = false
        @test timedwait(() -> status(() -> _infer(port, "m")) === nothing, 20) === :ok
        @test wc.resets[] >= 1
        @test gw.prober.wedged_rounds == 0
    finally
        _CGW.stop!(gw)
        _close_proxy(proxy)
        close(s)
    end
end
