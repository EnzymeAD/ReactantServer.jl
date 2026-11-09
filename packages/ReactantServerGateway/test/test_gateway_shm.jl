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
    ready::Bool              # what ServerReady reports
    hang::Bool               # while set, ServerReady and the shared-memory RPCs do not answer
    fail_unregister::Bool
    calls::Int               # shared-memory and namespace RPCs received
end
ShmMockWorker(reg, fail_register) = ShmMockWorker(reg, fail_register, true, false, false, 0)

# Count a shared-memory RPC, then hold it while the worker is frozen (at most a minute).
function _shm_mock_call(m::ShmMockWorker)
    m.calls += 1
    t = time() + 60
    while m.hang && time() < t
        sleep(0.05)
    end
    return nothing
end

function _shm_mock_server(w::ShmMockWorker; port::Integer = 0)
    server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", Int(port); context = w)
    ReactantServerGateway.register_GRPCInferenceService!(
        server;
        ServerReady = (ctx, req) -> begin
            m = ctx.payload
            t = time() + 60
            while m.hang && time() < t
                sleep(0.05)
            end
            _GSI.ServerReadyResponse(; ready = m.ready)
        end,
        RepositoryIndex = (ctx, req) -> _GSI.RepositoryIndexResponse(),
        IsSameIPCNamespace = (ctx, req) -> (_shm_mock_call(ctx.payload); _GSI.IsSameIPCNamespaceResponse(; same = true)),
        # The worker's own mapping of registration errors (transport/grpc.jl `_as_invalid`).
        SystemSharedMemoryRegister = (ctx, req) -> begin
            m = ctx.payload
            _shm_mock_call(m)
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
            m = ctx.payload
            _shm_mock_call(m)
            m.fail_unregister && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, "mock unregister failure"))
            _GSC.shm_unregister!(m.reg, req.name)
            _GSI.SystemSharedMemoryUnregisterResponse()
        end,
        SystemSharedMemoryStatus = (ctx, req) -> (_shm_mock_call(ctx.payload); _GSC.encode_shm_status(_GSI, ctx.payload.reg, req.name)),
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

# --- A worker that is down: the fan-outs skip it, and it drops what it missed before it returns ---
#
# Register, unregister and the namespace probe go to the workers the health record considers healthy.
# A worker marked down is not asked: a register succeeds on the others (before, the whole register
# failed with FAILED_PRECONDITION, so a client could not register during any worker's outage), and
# a hung one cannot hold the call. Before a worker that stayed alive takes requests again, it drops
# every name it missed, so a name a client reused for another object cannot reach the old one.

# A gateway over `ports` whose prober runs one round at startup and then only when a test calls
# `_check_once`.
function _shm_gateway(ports)
    path = tempname() * ".yaml"
    write(
        path, """
        listen:
          grpc: "127.0.0.1:0"
          metrics: "127.0.0.1:0"
        endpoints:
        $(join(["  - \"127.0.0.1:$p\"" for p in ports], "\n"))
        """
    )
    env = ["REACTANT_GATEWAY_HEALTH_INTERVAL_SECONDS" => "3600", "REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS" => "0"]
    try
        return withenv(() -> ReactantServerGateway.serve_gateway(path; blocking = false), env...)
    finally
        rm(path; force = true)
    end
end

_shm_down(gw, url) = lock(() -> haskey(gw.routes.down_since, url), gw.routes.lock)
_gw_same_ns(port) = grpc_call(
    _GSI.GRPCInferenceService.IsSameIPCNamespace, port, _GSI.IsSameIPCNamespaceRequest(; name = "probe")
).same

@testset "SHM register and unregister skip a worker that is down, and need one that is up" begin
    w1 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    w2 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    s1 = _shm_mock_server(w1); p1 = grpc_start_ephemeral!(s1)
    s2 = _shm_mock_server(w2); p2 = grpc_start_ephemeral!(s2)
    gw = _shm_gateway([p1, p2])
    port, _ = ReactantServerGateway.bound_ports(gw)
    pool = _GSC.BufferPool(1 << 16; n_slots = 2)
    name, key, bytes = _GSC.pool_region_name(pool), "/" * _GSC.pool_region_name(pool), sizeof(pool)
    held(w) = get(_GSC.shm_regions(w.reg), name, nothing)
    try
        # Worker 2 reports not ready (alive, and still answering the shared-memory RPCs).
        w2.ready = false
        ReactantServerGateway._check_once(gw.prober)
        @test _shm_down(gw, "127.0.0.1:$p2")
        w2.calls = 0
        @test _gw_same_ns(port)
        @test _gw_register(port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w1) !== nothing && held(w2) === nothing
        @test _gw_unregister(port, name) isa _GSI.SystemSharedMemoryUnregisterResponse
        @test held(w1) === nothing
        @test w2.calls == 0                  # not asked anything: no probe, pre-check, or register

        # A worker that is down and hung holds nothing up either.
        w2.hang = true
        t = time()
        @test _gw_register(port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        @test time() - t < 3.0
        w2.hang = false

        # No worker up: UNAVAILABLE, for the register and the unregister, and the probe says no.
        w1.ready = false
        ReactantServerGateway._check_once(gw.prober)
        @test _gw_status_code(() -> _gw_register(port, name, key, bytes)) == gRPCClient.GRPC_UNAVAILABLE
        @test _gw_status_code(() -> _gw_unregister(port, name)) == gRPCClient.GRPC_UNAVAILABLE
        @test !_gw_same_ns(port)
    finally
        w2.hang = false
        ReactantServerGateway.stop!(gw)
        close(s1)
        close(s2)
        foreach(w -> _GSC.shm_teardown!(w.reg), (w1, w2))
        close(pool)
    end
end

@testset "SHM register succeeds while a worker is killed, and reaches it once it is back" begin
    w1 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    w2 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    s1 = _shm_mock_server(w1); p1 = grpc_start_ephemeral!(s1)
    s2 = _shm_mock_server(w2); p2 = grpc_start_ephemeral!(s2)
    gw = _shm_gateway([p1, p2])
    port, _ = ReactantServerGateway.bound_ports(gw)
    pool = _GSC.BufferPool(1 << 16; n_slots = 2)
    name, key, bytes = _GSC.pool_region_name(pool), "/" * _GSC.pool_region_name(pool), sizeof(pool)
    held(w) = get(_GSC.shm_regions(w.reg), name, nothing)
    try
        gRPCServer.stop!(s2; force = true)          # connections to worker 2 are refused
        ReactantServerGateway._check_once(gw.prober)
        @test _gw_same_ns(port)
        @test _gw_register(port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w1) !== nothing
        # Worker 2 restarts with an empty registry; once the prober finds it ready, the client's
        # register (what it sends after the worker answers FAILED_PRECONDITION) reaches it, and
        # worker 1 keeps the registration it had.
        before = held(w1)
        w2.reg = _GSC.SharedMemoryRegistry(; reap_interval = 0)
        s2 = _shm_mock_server(w2; port = p2)
        gRPCServer.start!(s2)
        ReactantServerGateway._check_once(gw.prober)
        @test !_shm_down(gw, "127.0.0.1:$p2")
        @test _gw_register(port, name, key, bytes) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w2) !== nothing && held(w1) === before
    finally
        ReactantServerGateway.stop!(gw)
        close(s1)
        gRPCServer.stop!(s2; force = true)
        foreach(w -> _GSC.shm_teardown!(w.reg), (w1, w2))
        close(pool)
    end
end

@testset "a worker that missed a register or unregister while down drops the name before it returns" begin
    w1 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    w2 = ShmMockWorker(_GSC.SharedMemoryRegistry(; reap_interval = 0), false)
    s1 = _shm_mock_server(w1); p1 = grpc_start_ephemeral!(s1)
    s2 = _shm_mock_server(w2); p2 = grpc_start_ephemeral!(s2)
    u2 = "127.0.0.1:$p2"
    gw = _shm_gateway([p1, p2])
    port, _ = ReactantServerGateway.bound_ports(gw)
    # A client that reuses one region name for another object, as a stock Triton client may: the
    # name stays `name`, the objects behind it change.
    name = "input_data"
    x = _GSC.BufferPool(1 << 16; n_slots = 2)
    keyof(p) = "/" * _GSC.pool_region_name(p)
    held(w) = get(_GSC.shm_regions(w.reg), name, nothing)
    try
        @test _gw_register(port, name, keyof(x), sizeof(x)) isa _GSI.SystemSharedMemoryRegisterResponse
        old = held(w2)
        @test old !== nothing

        # Worker 2 is marked down but stays alive. Meanwhile the client unregisters the name,
        # unlinks the object and registers the name for a new one; worker 2 hears none of it.
        w2.ready = false
        ReactantServerGateway._check_once(gw.prober)
        @test _gw_unregister(port, name) isa _GSI.SystemSharedMemoryUnregisterResponse
        close(x)
        y = _GSC.BufferPool(1 << 16; n_slots = 2)
        @test _gw_register(port, name, keyof(y), sizeof(y)) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w1).identity != old.identity
        @test held(w2) === old                       # still the unlinked object

        # It answers ready again, but cannot yet be told to drop the name: it stays down.
        w2.ready = true
        w2.fail_unregister = true
        ReactantServerGateway._check_once(gw.prober)
        @test _shm_down(gw, u2)
        @test held(w2) === old
        # Once it can, it drops the name and only then is it up: a request under the name there
        # fails as unregistered (FAILED_PRECONDITION) instead of using the old object, and the
        # client's register then reaches it.
        w2.fail_unregister = false
        ReactantServerGateway._check_once(gw.prober)
        @test !_shm_down(gw, u2)
        @test held(w2) === nothing
        @test _gw_register(port, name, keyof(y), sizeof(y)) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w2).identity == held(w1).identity
        # Nothing is left to drop: the next outage and return of worker 2 drops nothing.
        @test isempty(ReactantServerGateway.missed_names(gw.server.context.gate, u2))

        # A name reused without an unregister in between: the register alone is missed.
        w2.ready = false
        ReactantServerGateway._check_once(gw.prober)
        close(y)
        z = _GSC.BufferPool(1 << 16; n_slots = 2)
        @test _gw_register(port, name, keyof(z), sizeof(z)) isa _GSI.SystemSharedMemoryRegisterResponse
        @test held(w2) !== nothing && held(w2).identity != held(w1).identity
        w2.ready = true
        ReactantServerGateway._check_once(gw.prober)
        @test !_shm_down(gw, u2) && held(w2) === nothing
        close(z)
    finally
        ReactantServerGateway.stop!(gw)
        close(s1)
        close(s2)
        foreach(w -> _GSC.shm_teardown!(w.reg), (w1, w2))
    end
end
