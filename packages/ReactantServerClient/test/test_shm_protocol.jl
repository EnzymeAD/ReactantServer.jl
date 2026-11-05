# The client's side of the slot control protocol (CORE-3) against a mini worker built from Core's
# real registry and codec: it maps the client's pool through its own descriptor, drops given-up
# requests before "running" them, and writes shared-memory outputs under the protocol. Deadlines
# shorter than the worker's latency make clients give up while their requests are still running,
# so slots are reused under late writers; every response that does succeed must carry exactly its
# own outputs. Also: chunks never wait for a slot (CLI-10).

module ShmProtocolTests

using Test
using ReactantServerClient
import ReactantServerCore
using ReactantServerClient: inference
using ReactantServerClient.inference
import gRPCServer
using Sockets

const RSC = ReactantServerClient
const C = ReactantServerCore
const inf = inference

free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

mutable struct MiniWorker
    reg::C.SharedMemoryRegistry
    latency::Function                 # seconds to "compute" one request
    infer_calls::Threads.Atomic{Int}
    controlled::Threads.Atomic{Int}   # requests that carried slot control
    dropped::Threads.Atomic{Int}      # dropped at "dispatch": the client had given up
    refused::Threads.Atomic{Int}      # output write refused: the client gave up meanwhile
end
MiniWorker(latency) = MiniWorker(
    C.SharedMemoryRegistry(; reap_interval = 0), latency, Threads.Atomic{Int}(0),
    Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
)

_status(e) = e isa C.RequestCancelled ? gRPCServer.StatusCode.CANCELLED :
    e isa C.DeadlineExceeded ? gRPCServer.StatusCode.DEADLINE_EXCEEDED :
    e isa C.UnregisteredRegionError ? gRPCServer.StatusCode.FAILED_PRECONDITION :
    gRPCServer.StatusCode.INVALID_ARGUMENT

function _mini_infer(w::MiniWorker, req)
    Threads.atomic_add!(w.infer_calls, 1)
    haskey(req.parameters, C.SLOT_CONTROL_TOKEN_PARAM) && Threads.atomic_add!(w.controlled, 1)
    try
        d = C.decode_infer_request(inf, req, w.reg)
        sleep(w.latency())                          # queued
        if C.request_cancelled(d.request)           # the scheduler's dispatch-time check
            Threads.atomic_add!(w.dropped, 1)
            throw(C.RequestCancelled("m", "dropped at dispatch"))
        end
        sleep(w.latency())                          # executing
        x = d.request.inputs[1].data
        y = C.NamedTensor("y", 2 .* x)
        return try
            C.encode_infer_response(inf, "m", d, [y], w.reg)
        catch e
            e isa C.RequestCancelled && Threads.atomic_add!(w.refused, 1)
            rethrow()
        end
    catch e
        throw(gRPCServer.GRPCError(_status(e), sprint(showerror, e)))
    end
end

function _mini_server(w::MiniWorker, port)
    server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", Int(port); context = w, max_message_size = 64 * 1024 * 1024)
    reg = (name, Req, Resp, handler) -> gRPCServer.register_method!(
        server.dispatcher, "inference.GRPCInferenceService",
        gRPCServer.MethodDescriptor(name, gRPCServer.MethodType.UNARY, Req, Resp, handler)
    )
    reg("IsSameIPCNamespace", inf.IsSameIPCNamespaceRequest, inf.IsSameIPCNamespaceResponse, (ctx, req) -> inf.IsSameIPCNamespaceResponse(; same = true))
    reg(
        "SystemSharedMemoryRegister", inf.SystemSharedMemoryRegisterRequest, inf.SystemSharedMemoryRegisterResponse, (ctx, req) -> begin
            C.shm_register!(w.reg, req.name, req.key, req.offset, req.byte_size)
            inf.SystemSharedMemoryRegisterResponse()
        end
    )
    reg(
        "SystemSharedMemoryUnregister", inf.SystemSharedMemoryUnregisterRequest, inf.SystemSharedMemoryUnregisterResponse, (ctx, req) -> begin
            C.shm_unregister!(w.reg, req.name)
            inf.SystemSharedMemoryUnregisterResponse()
        end
    )
    reg("ModelInfer", inf.ModelInferRequest, inf.ModelInferResponse, (ctx, req) -> _mini_infer(w, req))
    return server
end

# Items with distinct values: item i is Float32[i, i + 0.25, i + 0.5, i + 0.75]; the model doubles
# them. The output is declared, so it comes back through shared memory on a SHM pool.
struct ScaleIO <: AbstractInferenceIO
    first::Int
    n::Int
    got::Vector{Vector{Float32}}
    nested::Union{Nothing, Function}   # called from inside decode (nested inference)
end
ScaleIO(first, n; nested = nothing) = ScaleIO(first, n, [Float32[] for _ in 1:n], nested)
Base.length(io::ScaleIO) = io.n
_item(i) = Float32[i, i + 0.25, i + 0.5, i + 0.75]
ReactantServerClient.item_input_bytes(::ScaleIO) = 4 * sizeof(Float32)
ReactantServerClient.output_specs(::ScaleIO) = [OutputSpec("y", Float32, [4])]
function ReactantServerClient.infer_encode_chunk!(io::ScaleIO, r, slot)
    x = scratch(slot, "x", (4, length(r)), Float32)
    v = pool_view(x)
    for (j, i) in enumerate(r)
        v[:, j] .= _item(io.first + i - 1)
    end
    return x
end
function ReactantServerClient.infer_decode_chunk!(io::ScaleIO, r, response)
    io.nested === nothing || io.nested()
    y = InferOutput("y", response, Float32)
    for (j, i) in enumerate(r)
        io.got[i] = collect(y[:, j])
    end
    return nothing
end
_correct(io::ScaleIO) = all(i -> io.got[i] == 2 .* _item(io.first + i - 1), 1:io.n)

# Whether every slot of `pool` can be taken at once right now (taken and given back).
function _all_slots_free(pool)
    s = ReactantServerCore.try_acquire_slot!(pool, pool.n_slots)
    s === nothing && return false
    ReactantServerCore.release_slot!(s)
    return true
end

@testset "every successful chunk carries its own outputs under give-ups and slot reuse (CORE-3)" begin
    @test ReactantServerCore.slot_control_supported() == Sys.islinux()
    w = MiniWorker(() -> rand() * 0.02)                    # 0 to 20 ms queued, and again executing
    port = free_port()
    server = _mini_server(w, port); gRPCServer.start!(server)
    try
        kserve_init(; pool_bytes = 1 << 16, n_slots = 2, shm_reprobe_interval = 0.0)
        m = KServeModel(
            "127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 8,
            deadline = 0.02, retry = RSC.RetryPolicy(enabled = false)
        )
        # Warm up with a generous deadline (the first call compiles the request path).
        warm = KServeModel("127.0.0.1", port, "m"; shared_memory = :on, max_batch_size = 8, deadline = 60.0, retry = RSC.RetryPolicy(enabled = false))
        io0 = ScaleIO(1, 8)
        infer_sync(warm, io0)
        @test _correct(io0)
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        ok = Threads.Atomic{Int}(0)
        failed = Threads.Atomic{Int}(0)
        wrong = Threads.Atomic{Int}(0)
        tasks = map(1:4) do task
            Threads.@spawn for k in 1:60
                io = ScaleIO(1000 * task + 10 * k, 8)
                try
                    infer_sync(m, io)
                    Threads.atomic_add!(_correct(io) ? ok : wrong, 1)
                catch
                    Threads.atomic_add!(failed, 1)
                end
            end
        end
        foreach(wait, tasks)
        @info "slot control under give-ups" ok = ok[] failed = failed[] wrong = wrong[] dropped = w.dropped[] refused = w.refused[] fallbacks = RSC.inline_fallback_count()
        @test wrong[] == 0                                  # never another request's outputs
        @test ok[] > 0 && failed[] > 0                      # both outcomes happened
        # Requests staged in the pool carry slot control; those that found no free slot went inline.
        @test w.controlled[] > 0
        @test RSC.inline_fallback_count() > 0
        # Every slot is back: none is stuck quarantined once the writers are done.
        pool = RSC.get_or_create_pool!(m).pool
        @test timedwait(() -> _all_slots_free(pool), 10.0) === :ok
    finally
        kserve_shutdown()
        close(server)
        C.shm_teardown!(w.reg)
    end
end

@testset "a chunk never waits for a slot: nested inference and inline fallback (CLI-10)" begin
    w = MiniWorker(() -> 0.0)
    port = free_port()
    server = _mini_server(w, port); gRPCServer.start!(server)
    try
        for mode in (:on, :off)
            kserve_init(; pool_bytes = 1 << 16, n_slots = 2, shm_reprobe_interval = 0.0)
            m = KServeModel("127.0.0.1", port, "m"; shared_memory = mode, max_batch_size = 1, deadline = 60.0, retry = RSC.RetryPolicy(enabled = false))
            infer_sync(m, ScaleIO(1, 1))                        # warm up
            before = RSC.inline_fallback_count()
            # Every outer chunk decodes while holding its slot and runs a whole inference inside its
            # decode: with both slots held by the outer call, the inner chunks find none free. They
            # used to wait for one forever.
            inner = ScaleIO[]
            lk = ReentrantLock()
            nested = () -> begin
                io = ScaleIO(500, 3)
                infer_sync(m, io)
                @lock lk push!(inner, io)
            end
            outer = ScaleIO(1, 4; nested = nested)
            t = Threads.@spawn infer_async(m, outer)
            @test timedwait(() -> istaskdone(t), 60.0) === :ok
            istaskdone(t) && fetch(t)
            @test _correct(outer)
            @test length(inner) == 4 && all(_correct, inner)
            if mode === :on
                @test RSC.inline_fallback_count() > before     # the inner chunks went inline
            end
        end
    finally
        kserve_shutdown()
        close(server)
        C.shm_teardown!(w.reg)
    end
end

end # module
