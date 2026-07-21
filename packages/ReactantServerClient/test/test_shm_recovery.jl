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
using ReactantServerCore: inference
using ReactantServerCore.inference   # bring the KServe message types into scope for the server stubs
import gRPCServer
using Sockets

const RSC = ReactantServerClient
const inf = inference

# Server-side service stubs (register_GRPCInferenceService! + method descriptors). Included into this
# module, which has `inference` message types in scope and `import gRPCServer`, exactly as the worker
# and gateway include them.
include(ReactantServerCore.inference_server_stubs_path())

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
end
MockState() = MockState(Set{String}(), true, false, 0.0, Threads.Atomic{Int}(0),
                        Threads.Atomic{Int}(0), ReentrantLock())

# Extract a tensor's shared_memory_region string parameter, or nothing when the tensor is inline.
function _region_of(t)
    p = get(t.parameters, "shared_memory_region", nothing)
    (p === nothing || p.parameter_choice === nothing) && return nothing
    p.parameter_choice.name === :string_param ? String(p.parameter_choice[]) : nothing
end

function _mk_router(st::MockState)
    router = gRPCServer.gRPCRouter(; max_receive_message_length = 64 * 1024 * 1024,
                                   max_send_message_length = 64 * 1024 * 1024)
    register_GRPCInferenceService!(router;
        IsSameIPCNamespace = (req, ctx) -> begin
            st.probe_sleep > 0 && sleep(st.probe_sleep)   # simulate a restarted-but-unresponsive server
            inf.IsSameIPCNamespaceResponse(; same = st.same_ns)
        end,
        SystemSharedMemoryRegister = (req, ctx) -> begin
            Threads.atomic_add!(st.register_calls, 1)
            st.reject_register &&
                throw(gRPCServer.gRPCServiceCallException(gRPCServer.GRPC_INTERNAL, "register rejected"))
            @lock st.lock push!(st.regions, req.name)
            inf.SystemSharedMemoryRegisterResponse()
        end,
        SystemSharedMemoryUnregister = (req, ctx) -> begin
            @lock st.lock (isempty(req.name) ? empty!(st.regions) : delete!(st.regions, req.name))
            inf.SystemSharedMemoryUnregisterResponse()
        end,
        ModelInfer = (req, ctx) -> begin
            Threads.atomic_add!(st.infer_calls, 1)
            for t in req.inputs
                name = _region_of(t)
                name === nothing && continue           # inline input: nothing to check
                @lock st.lock (name in st.regions) || throw(gRPCServer.gRPCServiceCallException(
                    gRPCServer.GRPC_FAILED_PRECONDITION, "unregistered shared memory region: $name"))
            end
            inf.ModelInferResponse(; model_name = req.model_name,
                outputs = inf.var"ModelInferResponse.InferOutputTensor"[],
                raw_output_contents = Vector{UInt8}[])
        end,
    )
    return router
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
    stale_fp = RSC.gRPCClient.gRPCServiceCallException(RSC.gRPCClient.GRPC_FAILED_PRECONDITION,
                                                       "unregistered shared memory region: pool")
    # Message-only signal (status rewritten to UNAVAILABLE in transit) is still recognized.
    stale_msg = RSC.gRPCClient.gRPCServiceCallException(RSC.gRPCClient.GRPC_UNAVAILABLE,
                                                        "worker x: unregistered shared memory region: pool")
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
    server = gRPCServer.serve!(_mk_router(st), "127.0.0.1", port)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 8, shm_reprobe_interval = 0.0)
        m = KServeModel("127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 1,
                        deadline = 10.0, retry = _noretry())
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
    server = gRPCServer.serve!(_mk_router(st), "127.0.0.1", port)
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
