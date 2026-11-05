# The SHM register fan-out behind a real gateway, against mock workers that hold a real Core
# SharedMemoryRegistry (the worker's registration rules, CORE-2) and real POSIX objects: a client
# that re-registers after one worker restarted must not lose its registration on the healthy
# workers, a partial failure rolls back only what the call created, and a name held by another
# client's live object comes back as ALREADY_EXISTS. Run with 2 and with 4 workers.

const _GSI = ReactantServerGateway.inference
const _GSC = ReactantServerCore

mutable struct ShmMockWorker
    reg::_GSC.SharedMemoryRegistry
    fail_register::Bool
end

function _shm_mock_server(w::ShmMockWorker)
    server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", 0; context = w)
    ReactantServerGateway.register_GRPCInferenceService!(
        server;
        ServerReady = (ctx, req) -> _GSI.ServerReadyResponse(; ready = true),
        RepositoryIndex = (ctx, req) -> _GSI.RepositoryIndexResponse(),
        # The worker's own mapping of registration errors (transport/grpc.jl `_as_invalid`).
        SystemSharedMemoryRegister = (ctx, req) -> begin
            m = ctx.payload
            m.fail_register && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, "mock register failure"))
            try
                _GSC.shm_register!(m.reg, req.name, req.key, req.offset, req.byte_size)
            catch e
                e isa _GSC.ShmRegionExistsError &&
                    throw(gRPCServer.GRPCError(gRPCServer.StatusCode.ALREADY_EXISTS, sprint(showerror, e)))
                throw(gRPCServer.GRPCError(gRPCServer.StatusCode.INVALID_ARGUMENT, sprint(showerror, e)))
            end
            _GSI.SystemSharedMemoryRegisterResponse()
        end,
        SystemSharedMemoryUnregister = (ctx, req) -> begin
            _GSC.shm_unregister!(ctx.payload.reg, req.name)
            _GSI.SystemSharedMemoryUnregisterResponse()
        end,
        SystemSharedMemoryStatus = (ctx, req) -> _GSC.encode_shm_status(_GSI, ctx.payload.reg, req.name),
    )
    return server
end

_gw_register(port, name, key, bytes) = grpc_call(
    _GSI.GRPCInferenceService.SystemSharedMemoryRegister, port,
    _GSI.SystemSharedMemoryRegisterRequest(; name = name, key = key, offset = 0, byte_size = bytes)
)
_gw_unregister(port, name) = grpc_call(
    _GSI.GRPCInferenceService.SystemSharedMemoryUnregister, port,
    _GSI.SystemSharedMemoryUnregisterRequest(; name = name)
)
_gw_status_code(f) =
try
    f()
    nothing
catch e
    e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : rethrow()
end

@testset "SHM register fan-out with $n workers (CORE-2)" for n in (2, 4)
    workers = [ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false) for _ in 1:n]
    servers = [_shm_mock_server(w) for w in workers]
    ports = [grpc_start_ephemeral!(s) for s in servers]
    gatewayfile = tempname() * ".yaml"
    write(
        gatewayfile, """
        listen:
          grpc: "127.0.0.1:0"
          metrics: "127.0.0.1:0"
        endpoints:
        $(join(["  - \"127.0.0.1:$p\"" for p in ports], "\n"))
        """
    )
    gw = ReactantServerGateway.serve_gateway(gatewayfile; blocking = false)
    gw_port, _ = ReactantServerGateway.bound_ports(gw)
    # Two clients' pools: real objects with the names and keys ReactantServerClient registers.
    p1 = _GSC.BufferPool(1 << 16; n_slots = 2)
    p2 = _GSC.BufferPool(1 << 16; n_slots = 2)
    name, key = _GSC.pool_region_name(p1), "/" * _GSC.pool_region_name(p1)
    bytes = sizeof(p1)
    held(w) = get(_GSC.shm_regions(w.reg), name, nothing)
    try
        # Every worker registers it.
        @test _gw_register(gw_port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        before = [held(w) for w in workers]
        @test all(r -> r !== nothing && r.attached, before)

        # Worker 2 restarts (its registry is empty). The client re-registers the same name and key
        # through the gateway: the restarted worker registers it again, and every healthy worker
        # keeps its existing registration (the same mapping, never unregistered in between).
        workers[2].reg = _GSC.SharedMemoryRegistry(; reap_interval = 0)
        @test _gw_register(gw_port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        for (i, w) in enumerate(workers)
            r = held(w)
            @test r !== nothing && r.attached
            i == 2 || @test r === before[i]
        end
        before = [held(w) for w in workers]

        # A partial failure: worker 1 restarted (so this call creates its registration) and the
        # last worker fails. The call fails, worker 1's new registration is rolled back, and the
        # registrations that existed before the call are left exactly as they were.
        workers[1].reg = _GSC.SharedMemoryRegistry(; reap_interval = 0)
        workers[n].fail_register = true
        @test _gw_status_code(() -> _gw_register(gw_port, name, key, bytes)) == gRPCClient.GRPC_FAILED_PRECONDITION
        @test held(workers[1]) === nothing
        for i in 2:n
            @test held(workers[i]) === before[i] && before[i].attached
        end
        workers[n].fail_register = false
        @test _gw_register(gw_port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        before = [held(w) for w in workers]
        @test all(r -> r !== nothing && r.attached, before)

        # Another client's live object under the same name: ALREADY_EXISTS reaches the client and
        # every worker keeps the first client's registration.
        other = "/" * _GSC.pool_region_name(p2)
        @test _gw_status_code(() -> _gw_register(gw_port, name, other, sizeof(p2))) == gRPCClient.GRPC_ALREADY_EXISTS
        @test all(i -> held(workers[i]) === before[i] && before[i].attached, 1:n)

        # Even when the conflict is on some workers only, nothing that existed is rolled back.
        workers[1].reg = _GSC.SharedMemoryRegistry(; reap_interval = 0)
        @test _gw_status_code(() -> _gw_register(gw_port, name, other, sizeof(p2))) == gRPCClient.GRPC_ALREADY_EXISTS
        @test held(workers[1]) === nothing       # created by this call on worker 1, rolled back
        @test all(i -> held(workers[i]) === before[i] && before[i].attached, 2:n)

        # Unregister fans out and is idempotent.
        @test _gw_unregister(gw_port, name) isa _GSI.SystemSharedMemoryUnregisterResponse
        @test all(w -> held(w) === nothing, workers)
        @test _gw_unregister(gw_port, name) isa _GSI.SystemSharedMemoryUnregisterResponse
    finally
        ReactantServerGateway.stop!(gw)
        foreach(close, servers)
        foreach(w -> _GSC.shm_teardown!(w.reg), workers)
        close(p1)
        close(p2)
        rm(gatewayfile; force = true)
    end
end
