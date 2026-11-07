# Client-driver shared-memory recovery against a mock KServe gRPC server.
#
# Exercises the full stack the real client uses (register / unregister / IsSameIPCNamespace /
# ModelInfer over gRPC) without pulling in the worker or Reactant. The mock lets a test flip the
# server's "registry" out from under the client (simulating a restart) and count register calls, so
# we can assert: transparent re-registration keeps SHM (Tier 1), the re-registration is coalesced to
# a single call under concurrency, an unrecoverable SHM failure latches to inline (Tier 2), and a
# background re-probe restores SHM once the server recovers (Tier 3).

module ShmRecoveryTests

using Test
using ReactantServerClient
import ReactantServerCore
using ReactantServerClient: inference
using ReactantServerClient.inference   # KServe message types in scope
import gRPCServer
using Sockets

const RSC = ReactantServerClient
const inf = inference

free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

# Mutable mock-server state a test can poke between requests.
mutable struct MockState
    regions::Set{String}                 # region names the server currently "knows"
    same_ns::Bool                        # IsSameIPCNamespace answer
    reject_register::Bool                # when true, register throws (SHM unrecoverable)
    probe_sleep::Float64                 # seconds the IsSameIPCNamespace handler stalls before replying
    register_calls::Threads.Atomic{Int}  # total SystemSharedMemoryRegister invocations
    infer_calls::Threads.Atomic{Int}
    lock::ReentrantLock
    unregister_calls::Threads.Atomic{Int}
    registered_names::Vector{String}     # every name a register call carried, in order
    fail_infer::Bool                     # when true, ModelInfer fails with INVALID_ARGUMENT
end
MockState() = MockState(
    Set{String}(), true, false, 0.0, Threads.Atomic{Int}(0),
    Threads.Atomic{Int}(0), ReentrantLock(), Threads.Atomic{Int}(0), String[], false
)

# Extract a tensor's shared_memory_region string parameter, or nothing when the tensor is inline.
function _region_of(t)
    p = get(t.parameters, "shared_memory_region", nothing)
    (p === nothing || p.parameter_choice === nothing) && return nothing
    return p.parameter_choice.name === :string_param ? String(p.parameter_choice[]) : nothing
end

function _mk_server(st::MockState, port::Integer)
    server = gRPCServer.GRPCServerHTTPJl(
        "127.0.0.1", Int(port); context = st,
        max_message_size = 64 * 1024 * 1024
    )
    # The client's generated module carries no server registration (it never serves gRPC), so the
    # mock registers its four unary RPCs through the runtime primitive directly. Mirror the
    # generated `register_GRPCInferenceService_<Rpc>!` call shape (MethodDescriptor + register_method!).
    _reg = (name, Req, Resp, handler) -> gRPCServer.register_method!(
        server.dispatcher, "inference.GRPCInferenceService",
        gRPCServer.MethodDescriptor(name, gRPCServer.MethodType.UNARY, Req, Resp, handler)
    )
    _reg(
        "IsSameIPCNamespace", inf.IsSameIPCNamespaceRequest, inf.IsSameIPCNamespaceResponse, (ctx, req) -> begin
            st.probe_sleep > 0 && sleep(st.probe_sleep)   # simulate a restarted-but-unresponsive server
            inf.IsSameIPCNamespaceResponse(; same = st.same_ns)
        end
    )
    _reg(
        "SystemSharedMemoryRegister", inf.SystemSharedMemoryRegisterRequest, inf.SystemSharedMemoryRegisterResponse, (ctx, req) -> begin
            Threads.atomic_add!(st.register_calls, 1)
            @lock st.lock push!(st.registered_names, req.name)
            st.reject_register &&
                throw(gRPCServer.GRPCError(gRPCServer.StatusCode.INTERNAL, "register rejected"))
            @lock st.lock push!(st.regions, req.name)
            inf.SystemSharedMemoryRegisterResponse()
        end
    )
    _reg(
        "SystemSharedMemoryUnregister", inf.SystemSharedMemoryUnregisterRequest, inf.SystemSharedMemoryUnregisterResponse, (ctx, req) -> begin
            Threads.atomic_add!(st.unregister_calls, 1)
            @lock st.lock (isempty(req.name) ? empty!(st.regions) : delete!(st.regions, req.name))
            inf.SystemSharedMemoryUnregisterResponse()
        end
    )
    _reg(
        "ModelInfer", inf.ModelInferRequest, inf.ModelInferResponse, (ctx, req) -> begin
            Threads.atomic_add!(st.infer_calls, 1)
            st.fail_infer &&
                throw(gRPCServer.GRPCError(gRPCServer.StatusCode.INVALID_ARGUMENT, "rejected by the mock"))
            for t in req.inputs
                name = _region_of(t)
                name === nothing && continue           # inline input: nothing to check
                @lock st.lock (name in st.regions) || throw(
                    gRPCServer.GRPCError(
                        gRPCServer.StatusCode.FAILED_PRECONDITION, "unregistered shared memory region: $name"
                    )
                )
            end
            inf.ModelInferResponse(;
                model_name = req.model_name,
                outputs = inf.var"ModelInferResponse.InferOutputTensor"[],
                raw_output_contents = Vector{UInt8}[]
            )
        end
    )
    return server
end

# Minimal IO: one 4-element Float32 input per item, no declared outputs (inline). Decode just counts.
mutable struct RecoveryIO <: AbstractInferenceIO
    n::Int
    decoded::Threads.Atomic{Int}
end
RecoveryIO(n) = RecoveryIO(n, Threads.Atomic{Int}(0))
Base.length(io::RecoveryIO) = io.n
ReactantServerClient.item_input_bytes(::RecoveryIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(io::RecoveryIO, r, slot) =
    scratch(slot, "x", (4, length(r)), Float32)
ReactantServerClient.infer_decode_chunk!(io::RecoveryIO, r, response) =
    (Threads.atomic_add!(io.decoded, 1); nothing)

_noretry() = RSC.RetryPolicy(enabled = false)

@testset "stale-registration classification" begin
    stale_fp = RSC.gRPCClient.gRPCServiceCallException(
        RSC.gRPCClient.GRPC_FAILED_PRECONDITION,
        "unregistered shared memory region: pool"
    )
    # Message-only signal (status rewritten to UNAVAILABLE in transit) is still recognized.
    stale_msg = RSC.gRPCClient.gRPCServiceCallException(
        RSC.gRPCClient.GRPC_UNAVAILABLE,
        "worker x: unregistered shared memory region: pool"
    )
    other = RSC.gRPCClient.gRPCServiceCallException(RSC.gRPCClient.GRPC_INVALID_ARGUMENT, "bad shape")
    @test RSC._is_stale_registration(stale_fp)
    @test RSC._is_stale_registration(stale_msg)
    @test !RSC._is_stale_registration(other)
    @test !RSC._is_stale_registration(ErrorException("nope"))
    # A stale registration is never treated as a plain retryable shed (it is recovered explicitly).
    @test !RSC._retryable_shed(stale_fp, RSC.RetryPolicy())
    @test !RSC._retryable_shed(stale_msg, RSC.RetryPolicy())
end

@testset "SHM recovery: transparent re-register, coalescing, latch, unlatch" begin
    st = MockState()
    st.reject_register = false
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 8, shm_reprobe_interval = 0.0)
        m = KServeModel(
            "127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 1,
            # Generous: the first call to a fresh mock server compiles its whole request path, and
            # under coverage on a slow runner that alone has taken more than 10 s.
            deadline = 60.0, retry = _noretry()
        )
        key = (m.host, m.port, m.shared_memory)

        # Tier 1a: initial registration + a working SHM inference.
        io1 = RecoveryIO(1)
        infer_sync(m, io1)
        @test io1.decoded[] == 1
        @test st.register_calls[] == 1
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        # Tier 1b: server "restarts" (forgets the region); the next call transparently re-registers
        # and retries on SHM, with exactly one re-registration.
        @lock st.lock empty!(st.regions)
        reg0 = st.register_calls[]
        io2 = RecoveryIO(1)
        infer_sync(m, io2)
        @test io2.decoded[] == 1
        @test st.register_calls[] == reg0 + 1
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        # Coalescing: clear the registry, then fire a concurrent batch. Every in-flight chunk fails
        # stale at once but they must coalesce to a single re-registration.
        @lock st.lock empty!(st.regions)
        reg1 = st.register_calls[]
        io3 = RecoveryIO(8)
        infer_async(m, io3)
        @test io3.decoded[] == 8
        @test st.register_calls[] == reg1 + 1

        # Tier 2: re-registration now fails -> latch this endpoint to inline and complete inline.
        st.reject_register = true
        @lock st.lock empty!(st.regions)
        io4 = RecoveryIO(1)
        infer_sync(m, io4)
        @test io4.decoded[] == 1
        @test haskey(RSC._latched, key)
        @test !RSC.is_shm_backed(RSC.get_or_create_pool!(m))
        # While latched, the hot path stays inline: no further register attempts.
        reg2 = st.register_calls[]
        io5 = RecoveryIO(3)
        infer_async(m, io5)
        @test io5.decoded[] == 3
        @test st.register_calls[] == reg2

        # Tier 3: the server recovers; a background re-probe pass restores SHM and unlatches.
        st.reject_register = false
        RSC._reprobe_once()
        @test !haskey(RSC._latched, key)
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))
        io6 = RecoveryIO(1)
        infer_sync(m, io6)
        @test io6.decoded[] == 1
        # No (re)registration above sent an Unregister first: re-registering the same object is
        # idempotent on the worker, and a pre-emptive unregister failed requests in flight (CLI-8).
        @test st.unregister_calls[] == 0
        # Every registration carried the one pool's name.
        @test length(unique(st.registered_names)) == 1
    finally
        kserve_shutdown()
        close(server)
    end
end

# Whether every slot of `pool` can be taken at once right now (taken and given back).
function _all_slots_free(pool)
    s = ReactantServerCore.try_acquire_slot!(pool, pool.n_slots)
    s === nothing && return false
    ReactantServerCore.release_slot!(s)
    return true
end

# A chunk failure timed to land while other chunks are still decoding (CLI-3). The chunk holding
# item `fail_at` fails in its decode once two other chunks are inside theirs; every other chunk
# spends `hold` seconds in its decode, then writes its items. Counts the items encoded, the decodes
# running, and the items written.
struct ChunkFailure <: Exception
    item::Int
end
mutable struct SlowFailIO <: AbstractInferenceIO
    n::Int
    fail_at::Int
    hold::Float64
    out::Vector{Int}
    encoded::Threads.Atomic{Int}
    active::Threads.Atomic{Int}
    writes::Threads.Atomic{Int}
end
SlowFailIO(n, fail_at, hold) = SlowFailIO(
    n, fail_at, hold, zeros(Int, n), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
)
Base.length(io::SlowFailIO) = io.n
ReactantServerClient.item_input_bytes(::SlowFailIO) = 4 * sizeof(Float32)
function ReactantServerClient.infer_encode_chunk!(io::SlowFailIO, r, slot)
    Threads.atomic_add!(io.encoded, length(r))
    return scratch(slot, "x", (4, length(r)), Float32)
end
function ReactantServerClient.infer_decode_chunk!(io::SlowFailIO, r, response)
    if io.fail_at in r
        timedwait(() -> io.active[] >= 2, 30.0)
        throw(ChunkFailure(io.fail_at))
    end
    Threads.atomic_add!(io.active, 1)
    try
        sleep(io.hold)
        io.out[r] .= r
        Threads.atomic_add!(io.writes, length(r))
    finally
        Threads.atomic_sub!(io.active, 1)
    end
    return nothing
end

@testset "infer_async finishes every chunk it started, then throws the chunk's own exception (CLI-3)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        for mode in (:on, :off)
            kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
            m = KServeModel("127.0.0.1", port, "m"; shared_memory = mode, max_batch_size = 1, deadline = 60.0, retry = _noretry())
            infer_sync(m, RecoveryIO(1))                    # warm up: the first call compiles the request path
            pool = RSC.get_or_create_pool!(m).pool
            # Four slots and one item per chunk: four chunks in flight. The first fails while the
            # other three are inside their decode.
            io = SlowFailIO(12, 1, 0.5)
            err, stack = try
                infer_async(m, io)
                nothing, nothing
            catch e
                e, current_exceptions()
            end
            # The state the moment the call has thrown.
            active, writes, slots_free = io.active[], io.writes[], _all_slots_free(pool)
            sleep(1.0)                                      # longer than a decode: a chunk still running writes meanwhile
            @test err isa ChunkFailure && err.item == 1      # the chunk's exception itself, not a TaskFailedException
            @test active == 0                                # no chunk was still decoding into the io
            @test slots_free                                 # every chunk had given its slot back
            @test io.writes[] == writes                      # nothing wrote into the io after the call
            @test writes == 3 && io.out[2:4] == 2:4          # the chunks already in flight finished
            @test io.encoded[] == 4                          # and none started after the failure
            # The chunk's backtrace is kept, below the error on the exception stack.
            @test stack !== nothing && length(stack) >= 2
            @test occursin("infer_decode_chunk!", sprint(show, MIME"text/plain"(), stack))
        end
    finally
        kserve_shutdown()
        close(server)
    end
end

@testset "infer_async throws the server's gRPC error unwrapped, as infer_sync does (CLI-3)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
        m = KServeModel("127.0.0.1", port, "m"; shared_memory = :off, max_batch_size = 1, deadline = 60.0, retry = _noretry())
        infer_sync(m, RecoveryIO(1))                        # warm up
        st.fail_infer = true
        for f in (infer_sync, infer_async)
            err = try
                f(m, RecoveryIO(8))
                nothing
            catch e
                e
            end
            @test err isa RSC.gRPCClient.gRPCServiceCallException
            @test err isa RSC.gRPCClient.gRPCServiceCallException &&
                err.grpc_status == RSC.gRPCClient.GRPC_INVALID_ARGUMENT
        end
    finally
        kserve_shutdown()
        close(server)
    end
end

@testset "an unrecoverable shared-memory failure in infer_async latches to inline, as in infer_sync (CLI-3)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 8, shm_reprobe_interval = 0.0)
        m = KServeModel("127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 1, deadline = 60.0, retry = _noretry())
        key = (m.host, m.port, m.shared_memory)
        io1 = RecoveryIO(4)
        infer_async(m, io1)
        @test io1.decoded[] == 4
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        # The server forgets the region and refuses to register it again: every chunk fails with
        # ShmUnrecoverableError, which latches the endpoint to inline, and the batch completes inline.
        st.reject_register = true
        @lock st.lock empty!(st.regions)
        io2 = RecoveryIO(8)
        infer_async(m, io2)
        @test io2.decoded[] == 8
        @test haskey(RSC._latched, key)
        @test !RSC.is_shm_backed(RSC.get_or_create_pool!(m))
    finally
        kserve_shutdown()
        close(server)
    end
end

# The pool's pages are allocated when it is created (CORE-4), so a full /dev/shm fails its creation
# instead of killing the client with SIGBUS while it stages. Simulated here; the real kernel case is
# in ReactantServerCore's test_shm_backing.jl.
@testset "a pool /dev/shm has no room for: :auto goes inline, :on fails loudly (CORE-4)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    full = nothing
    try
        full = ReactantServerCore._SHM_FALLOCATE[]
        kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
        ReactantServerCore._SHM_FALLOCATE[] = (fd, off, len) -> Libc.ENOSPC
        mk(mode) = KServeModel(
            "127.0.0.1", port, "m"; shared_memory = mode, max_batch_size = 1, deadline = 60.0,
            retry = _noretry()
        )

        # :on: an error that says what to do; nothing is cached, so the next call tries again.
        on = mk(:on)
        err = try
            infer_sync(on, RecoveryIO(1))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("shared_memory=:on", err.msg) && occursin("/dev/shm", err.msg)
        @test occursin("No space left", err.msg)
        @test RSC._shm_pool[] === nothing
        @test !haskey(RSC._pool_routes, (on.host, on.port, :on))

        # :auto: sent inline, and latched for the background re-probe.
        auto = mk(:auto)
        key = (auto.host, auto.port, :auto)
        io = RecoveryIO(3)
        @test_logs (:warn, r"pool of .* could not be created.*inline transport") infer_sync(auto, io)
        @test io.decoded[] == 3
        @test haskey(RSC._latched, key)
        @test !RSC.is_shm_backed(RSC.get_or_create_pool!(auto))
        @test st.register_calls[] == 0
        # A re-probe pass while /dev/shm is still full leaves it latched.
        RSC._reprobe_once()
        @test haskey(RSC._latched, key)
        @test RSC._shm_pool[] === nothing

        # Once there is room, the re-probe creates the pool and restores shared memory.
        ReactantServerCore._SHM_FALLOCATE[] = full
        RSC._reprobe_once()
        @test !haskey(RSC._latched, key)
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(auto))
        io2 = RecoveryIO(2)
        infer_sync(auto, io2)
        @test io2.decoded[] == 2
        io3 = RecoveryIO(1)
        infer_sync(on, io3)
        @test io3.decoded[] == 1
    finally
        full === nothing || (ReactantServerCore._SHM_FALLOCATE[] = full)
        kserve_shutdown()
        close(server)
    end
end

@testset "kserve_init again gives the pool a new name (CLI-2)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        names = String[]
        keys = String[]
        for _ in 1:3
            kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
            m = KServeModel("127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 1, deadline = 60.0, retry = _noretry())
            io = RecoveryIO(2)
            infer_sync(m, io)
            @test io.decoded[] == 2
            pool = RSC._shm_pool[]
            push!(names, RSC.pool_name(pool))
            push!(keys, RSC.shmid(pool.pool.backing))
        end
        # Each generation has its own name and key, so a chunk still in flight from an earlier
        # generation can never re-register its name onto a later generation's memory, and an older
        # pool's unlink can never remove a newer pool's name.
        @test allunique(names) && allunique(keys)
        @test names == [k[2:end] for k in keys]
        # The earlier generations' objects are gone; only the current one exists.
        @test all(k -> ReactantServerCore.shm_key_identity(k) === :gone, keys[1:(end - 1)])
        @test ReactantServerCore.shm_key_identity(keys[end]) isa ReactantServerCore.ShmIdentity
    finally
        kserve_shutdown()
        close(server)
    end
end

# Mixed dtypes with an odd per-item size (37 bytes), staged through typed views: before CORE-15 a
# Float32 buffer after the UInt8 one was misaligned for odd chunk sizes, and every slot but the
# first of a 10^6-byte, 3-slot pool started at an odd offset.
mutable struct MixedIO <: AbstractInferenceIO
    n::Int
    decoded::Threads.Atomic{Int}
end
Base.length(io::MixedIO) = io.n
ReactantServerClient.item_input_bytes(::MixedIO) = 3 + 4 * 4 + 8 + 5 * 2
function ReactantServerClient.infer_encode_chunk!(io::MixedIO, r, slot)
    k = length(r)
    ins = scratch(slot, ["m" => ((3, k), UInt8), "x" => ((4, k), Float32), "d" => ((k,), Float64), "h" => ((5, k), Float16)])
    m, x, d, h = pool_view(ins...)
    m .= 0x01
    x .= 2.0f0
    d .= 3.0
    h .= Float16(4)
    return ins
end
ReactantServerClient.infer_decode_chunk!(io::MixedIO, r, response) = (Threads.atomic_add!(io.decoded, length(r)); nothing)

@testset "mixed-dtype IO on an odd pool geometry (CORE-15)" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        for mode in (:on, :off), mbs in (1, 3, 1000)
            kserve_init(; pool_bytes = 10^6, n_slots = 3, shm_reprobe_interval = 0.0)
            m = KServeModel("127.0.0.1", port, "m"; shared_memory = mode, max_batch_size = mbs, deadline = 60.0, retry = _noretry())
            n = mbs == 1000 ? 20_000 : 10                # 1000: chunks of slot_bytes ÷ 37, then a remainder
            io = MixedIO(n, Threads.Atomic{Int}(0))
            infer_sync(m, io)
            @test io.decoded[] == n
            io2 = MixedIO(n, Threads.Atomic{Int}(0))
            infer_async(m, io2)
            @test io2.decoded[] == n
            @test RSC.is_shm_backed(RSC.get_or_create_pool!(m)) == (mode === :on)
        end
    finally
        kserve_shutdown()
        close(server)
    end
end

@testset "re-probe poller lifecycle" begin
    # A positive interval starts a single background task; kserve_shutdown retires it.
    kserve_init(; shm_reprobe_interval = 0.1)
    RSC._ensure_reprobe_running!()
    t = RSC._reprobe_task[]
    @test t !== nothing && !istaskdone(t)
    RSC._ensure_reprobe_running!()
    @test RSC._reprobe_task[] === t          # idempotent: no second task
    kserve_shutdown()
    @test RSC._reprobe_task[] === nothing
    # A non-positive interval leaves the poller disabled.
    kserve_init(; shm_reprobe_interval = 0.0)
    RSC._ensure_reprobe_running!()
    @test RSC._reprobe_task[] === nothing
    kserve_shutdown()
end

@testset "shutdown does not block on an in-flight re-probe" begin
    # Regression: kserve_shutdown must return promptly even while the poller is parked in a probe
    # against a restarted-but-unresponsive server (the old code did wait(t) and hung).
    st = MockState()
    st.same_ns = true
    st.probe_sleep = 2.0                         # the IsSameIPCNamespace handler stalls 2s
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.02)
        m = KServeModel("127.0.0.1", port, "m"; shared_memory = :on)
        RSC.latch_inline!(m)                     # populates _latched and starts the poller
        sleep(0.4)                               # let the poller enter (and stall inside) a probe
        t0 = time()
        kserve_shutdown()
        dt = time() - t0
        @test dt < 1.0                           # must not wait out the 2s probe
        @test RSC._reprobe_task[] === nothing
    finally
        close(server)
    end
end

end # module
