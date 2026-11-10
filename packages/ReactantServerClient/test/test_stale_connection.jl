# A call that meets a stale pooled connection, one whose server end went away without an HTTP/2
# GOAWAY (a process that exits or is killed sends none), is sent once more on a fresh connection.
# Nothing else is sent again, and nothing more than once.
#
# Before, such a call failed with INTERNAL, "Send failure: Broken pipe", which no retry policy retries:
# after the gateway restarted, the first request of a client failed (CPU e2e step 11, phase A).
#
# The rules are checked on `_call_unary` with stand-in calls; the failure itself is reproduced with a
# TCP relay in front of the mock server of test_shm_recovery.jl, which drops its connections the way a
# dying process does: a close with no GOAWAY.

module StaleConnectionTests

using Test
using ReactantServerClient
import gRPCClient
import gRPCServer
using Sockets
using ..ShmRecoveryTests: MockState, _mk_server, RecoveryIO, free_port

const RSC = ReactantServerClient
const G = gRPCClient

# A finished call as `_call_unary` reads it: the request fields it looks at, and what `fetch` gives.
struct FakeReq
    code::Int
    ex::Union{Nothing, Exception}
    grpc_status::Int
    response::IOBuffer
    response_read_header::Bool
    url::String
end
FakeReq(code; status = G.GRPC_OK, ex = nothing, answered = false) =
    FakeReq(code, ex, status, answered ? IOBuffer(UInt8[0x00, 0x00]) : IOBuffer(), false, "127.0.0.1:1")
struct FakeCall
    req::FakeReq
    result::Any                      # the response, or the exception `fetch` throws
end
Base.fetch(c::FakeCall) = c.result isa Exception ? throw(c.result) : c.result

internal(msg) = G.gRPCServiceCallException(G.GRPC_INTERNAL, msg)
# libcurl's codes and gRPCClient's report for each way a call can fail. The message texts are what
# libcurl writes; `_call_unary` reads the code, so a refusal carrying a stale send's text is still a
# refusal.
stale(code = G.CURLE_HTTP2) = FakeCall(FakeReq(Int(code)), internal("Send failure: Broken pipe"))
server_status() = FakeCall(FakeReq(0; status = G.GRPC_INTERNAL), internal("the worker failed"))
timed_out() = FakeCall(
    FakeReq(Int(G.CURLE_OPERATION_TIMEDOUT)), G.gRPCServiceCallException(G.GRPC_DEADLINE_EXCEEDED, "Deadline exceeded.")
)
refused() = FakeCall(FakeReq(Int(G.CURLE_COULDNT_CONNECT)), internal("Send failure: Broken pipe"))
unresolved() = FakeCall(FakeReq(Int(G.CURLE_COULDNT_RESOLVE_HOST)), internal("Could not resolve host: x"))
recv_error() = FakeCall(FakeReq(Int(G.CURLE_RECV_ERROR)), internal("Recv failure: Connection reset by peer"))
got_nothing() = FakeCall(FakeReq(Int(G.CURLE_GOT_NOTHING)), internal("Empty reply from server"))
answered() = FakeCall(FakeReq(Int(G.CURLE_HTTP2); answered = true), internal("Send failure: Broken pipe"))
local_failure() = FakeCall(
    FakeReq(Int(G.CURLE_HTTP2); ex = G.gRPCServiceCallException(G.GRPC_CANCELLED, "cancelled")),
    G.gRPCServiceCallException(G.GRPC_CANCELLED, "cancelled")
)
unavailable() = FakeCall(FakeReq(0; status = G.GRPC_UNAVAILABLE), G.gRPCServiceCallException(G.GRPC_UNAVAILABLE, "no replica"))
done(x) = FakeCall(FakeReq(0), x)

# Run `_call_unary` over `calls`, one per send. Returns what it returned or threw, the deadline each
# send was given, and when each was sent. `hold` is how long the first send takes to fail.
function drive(calls...; budget = 10.0, cancelled = () -> false, hold = 0.0)
    budgets = Float64[]
    times = Float64[]
    out = try
        RSC._call_unary(budget; cancelled = cancelled) do b
            push!(budgets, b)
            push!(times, time())
            length(times) == 1 && hold > 0 && sleep(hold)
            calls[length(times)]
        end
    catch e
        e
    end
    return out, budgets, times
end

@testset "a stale pooled connection: sent once more, and it succeeds" begin
    for code in (G.CURLE_HTTP2, G.CURLE_SEND_ERROR)
        out, budgets, times = drive(stale(code), done(:answer))
        @test out === :answer
        @test length(budgets) == 2
        # Sent again within the delay bound (plus the scheduler's slack), with what is left of the
        # deadline: the first send's, less the time since it went out.
        gap = times[2] - times[1]
        @test 0 <= gap <= RSC.STALE_RESEND_MAX_DELAY + 0.1
        @test budgets[1] == 10.0
        @test budgets[2] < budgets[1]
        @test abs((budgets[1] - budgets[2]) - gap) < 0.02
    end
    # No deadline stays no deadline.
    out, budgets, _ = drive(stale(), done(:answer); budget = Inf)
    @test out === :answer && budgets == [Inf, Inf]
end

@testset "a stale pooled connection twice: two sends, and the second failure as gRPCClient reported it" begin
    first_failure, second_failure = stale(), stale()
    out, budgets, _ = drive(first_failure, second_failure, done(:never))
    @test length(budgets) == 2
    @test out === second_failure.result               # not relabeled: still INTERNAL, not UNAVAILABLE
    @test out.grpc_status == G.GRPC_INTERNAL
end

@testset "no second send for anything but a stale pooled connection" begin
    for (what, call) in (
            "a status the server sent" => server_status(),
            "a deadline" => timed_out(),
            "a refusal (whatever its text)" => refused(),
            "a name that does not resolve" => unresolved(),
            "a receive failure" => recv_error(),
            "an empty reply" => got_nothing(),
            "a call whose answer had begun" => answered(),
            "a call gRPCClient failed itself" => local_failure(),
            "UNAVAILABLE from the server" => unavailable(),
        )
        out, budgets, _ = drive(call, done(:never))
        @testset "$what" begin
            @test out === call.result
            @test length(budgets) == 1
        end
    end
end

@testset "no second send past the deadline, or once the call has failed elsewhere" begin
    # The first send used up the deadline before it failed: no time is left for another.
    out, budgets, _ = drive(stale(), done(:never); budget = 0.2, hold = 0.3)
    @test out isa G.gRPCServiceCallException && out.grpc_status == G.GRPC_INTERNAL
    @test length(budgets) == 1
    # The call this send belongs to has already failed (another chunk of an infer_async call).
    out, budgets, _ = drive(stale(), done(:never); cancelled = () -> true)
    @test out isa G.gRPCServiceCallException && length(budgets) == 1
end

@testset "the delay before the second send stays within its bound" begin
    @test 0 < RSC.STALE_RESEND_MAX_DELAY <= 0.1
    delays = [RSC._stale_resend_delay() for _ in 1:10_000]
    @test all(d -> 0 <= d <= RSC.STALE_RESEND_MAX_DELAY, delays)
    @test maximum(delays) > 0.9 * RSC.STALE_RESEND_MAX_DELAY   # uniform over the whole range
    @test minimum(delays) < 0.1 * RSC.STALE_RESEND_MAX_DELAY
end

@testset "with RetryPolicy: neither adds sends to the other" begin
    policy = RSC.RetryPolicy(; initial_backoff = 0.01, jitter = false, min_budget = 0.01)
    m = KServeModel("grpc://h:1", "x"; deadline = 5.0, retry = policy)
    function run(calls...)
        sent = Ref(0)
        out = try
            RSC._infer_with_retry(m) do budget_s
                RSC._call_unary(budget_s) do _
                    calls[sent[] += 1]
                end
            end
        catch e
            e
        end
        return out, sent[]
    end
    # A stale connection twice ends the call: the policy does not retry INTERNAL.
    out, sent = run(stale(), stale(), done(:never))
    @test out isa G.gRPCServiceCallException && out.grpc_status == G.GRPC_INTERNAL && sent == 2
    # The policy retries UNAVAILABLE; that attempt meets a stale connection and is sent once more.
    out, sent = run(unavailable(), stale(), done(:answer))
    @test out === :answer && sent == 3
    # Each attempt the policy makes gets at most one more send.
    out, sent = run(stale(), unavailable(), stale(), stale(), done(:never))
    @test out isa G.gRPCServiceCallException && out.grpc_status == G.GRPC_INTERNAL && sent == 4
end

# A TCP relay to `target` that can drop every connection it carries at once, as a dying process
# does: closed, with no HTTP/2 GOAWAY. It keeps accepting, so the next connection goes through.
mutable struct Relay
    server::Sockets.TCPServer
    conns::Vector{Tuple{TCPSocket, TCPSocket}}
end
function _pump(from, to)
    try
        while true
            data = readavailable(from)
            isempty(data) && eof(from) && break
            isempty(data) || write(to, data)
        end
    catch
    end
    try
        close(to)
    catch
    end
    return nothing
end
function Relay(target::Integer)
    server = Sockets.listen(Sockets.localhost, 0)
    r = Relay(server, Tuple{TCPSocket, TCPSocket}[])
    errormonitor(
        @async while isopen(server)
            client = try
                Sockets.accept(server)
            catch
                break
            end
            upstream = Sockets.connect(Sockets.localhost, target)
            push!(r.conns, (client, upstream))
            @async _pump(client, upstream)
            @async _pump(upstream, client)
        end
    )
    return r
end
relay_port(r::Relay) = Int(Sockets.getsockname(r.server)[2])
function drop!(r::Relay)
    for (a, b) in r.conns
        close(a)
        close(b)
    end
    return empty!(r.conns)
end
Base.close(r::Relay) = (close(r.server); drop!(r))

@testset "infer_sync over a connection whose server end went away: sent once more, on shared memory" begin
    st = MockState()
    port = free_port()
    server = _mk_server(st, port); gRPCServer.start!(server)
    relay = Relay(port)
    try
        kserve_init(; pool_bytes = 1 << 20, n_slots = 4, shm_reprobe_interval = 0.0)
        m = KServeModel(
            "127.0.0.1", relay_port(relay), "m"; shared_memory = :on, max_batch_size = 1,
            deadline = 60.0, retry = RSC.RetryPolicy(enabled = false)
        )
        io = RecoveryIO(1)
        infer_sync(m, io)                              # registers, then infers, over one connection
        @test io.decoded[] == 1 && RSC.is_shm_backed(RSC.get_or_create_pool!(m))
        infers, registers = st.infer_calls[], st.register_calls[]

        # The connection's server end goes away. The next request fails to send on it, and is sent once
        # more on a fresh connection; the server sees it once.
        drop!(relay)
        io = RecoveryIO(1)
        infer_sync(m, io)
        @test io.decoded[] == 1
        @test st.infer_calls[] == infers + 1
        @test st.register_calls[] == registers         # no shared-memory recovery: nothing was stale there
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        # Both recoveries at once: the server end went away and the server forgot the region (a
        # restart). The request is sent once more, meets FAILED_PRECONDITION, the region is registered
        # again and the request sent again: two requests and one register reach the server.
        drop!(relay)
        @lock st.lock empty!(st.regions)
        infers, registers = st.infer_calls[], st.register_calls[]
        io = RecoveryIO(1)
        infer_sync(m, io)
        @test io.decoded[] == 1
        @test st.infer_calls[] == infers + 2
        @test st.register_calls[] == registers + 1
        @test RSC.is_shm_backed(RSC.get_or_create_pool!(m))

        # A status the server sends is not sent again.
        st.fail_infer = true
        infers = st.infer_calls[]
        err = try
            infer_sync(m, RecoveryIO(1))
            nothing
        catch e
            e
        end
        @test err isa G.gRPCServiceCallException && err.grpc_status == G.GRPC_INVALID_ARGUMENT
        @test st.infer_calls[] == infers + 1
    finally
        kserve_shutdown()
        close(relay)
        close(server)
    end
end

end # module
