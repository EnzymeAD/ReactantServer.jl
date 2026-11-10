# One more send for a call that met a stale pooled connection.
#
# gRPCClient keeps a pool of connections per handle (by default the process-wide handle every
# KServeModel shares), and libcurl sends the next call to the same host and port over a pooled
# HTTP/2 connection. When the server process at the other end goes away without closing the
# connection the HTTP/2 way (it exited or was killed, so no GOAWAY was sent), libcurl still takes the
# idle connection for alive, and the next call on it fails to send: libcurl ends the call with
# CURLE_HTTP2 (16) on an HTTP/2 connection, or CURLE_SEND_ERROR (55) on a plain one, with "Send
# failure: Broken pipe" as its text, and gRPCClient raises INTERNAL, which `RetryPolicy` does not
# retry. libcurl drops that connection with the failure, so the same call sent again goes out on a
# fresh one. After the gateway restarted, the first request of a client failed this way and the
# next one worked (CPU e2e step 11).
#
# `_call_unary` sends such a call once more. Once: a second failure is thrown as gRPCClient reports
# it, still INTERNAL, so no other retry multiplies it. It is told apart by libcurl's result code on
# the request, never by the message text.

# How long, at most, a call waits before it is sent again: a uniform random delay in
# [0, STALE_RESEND_MAX_DELAY] seconds. Nothing has to settle first (libcurl has already dropped the
# dead connection); the delay spreads out the reconnects of callers whose connection died at the same
# moment (every chunk in flight in an `infer_async` call, every client of a restarted gateway), so
# they do not all open one in the same instant. A connection over loopback or a LAN opens in well
# under the 50 ms this spreads them over, and 50 ms is half a percent of the default 10 s deadline.
const STALE_RESEND_MAX_DELAY = 0.05

_stale_resend_delay() = rand() * STALE_RESEND_MAX_DELAY

# libcurl's result codes for a call that could not be sent on a pooled connection whose server end is
# gone. CURLE_HTTP2 is what libcurl 8 reports for the failed write on an HTTP/2 connection, which is
# every connection gRPCClient makes: a server dropped without a GOAWAY ended the next call with it in
# every reproduction (test_stale_connection.jl), its text the socket's "Send failure: Broken pipe".
# CURLE_SEND_ERROR is the same failure on a connection without the HTTP/2 layer. Not
# CURLE_OPERATION_TIMEDOUT (28, a deadline), CURLE_COULDNT_CONNECT (7, a refusal: there was no
# connection to reuse), CURLE_COULDNT_RESOLVE_HOST (6), and not the receive-side CURLE_RECV_ERROR
# (56) and CURLE_GOT_NOTHING (52): no reproduction of this failure ended with them, and a connection
# that breaks while a call waits for its answer may have carried the call to the server already.
const _STALE_CONNECTION_CODES = (gRPCClient.CURLE_HTTP2, gRPCClient.CURLE_SEND_ERROR)

"""
    _stale_connection(req, e) -> Bool

Whether the call `req`, which failed with `e`, met a stale pooled connection: gRPCClient reported a
libcurl failure (INTERNAL with no exception of its own, so not a deadline, a cancellation, a size limit
or a shut-down handle), the server sent no status, libcurl's result code (`req.code`, which gRPCClient
1.2 keeps on the request) is one of `_STALE_CONNECTION_CODES`, and no byte of an answer arrived.
"""
function _stale_connection(req, e)
    e isa gRPCClient.gRPCServiceCallException || return false
    e.grpc_status == gRPCClient.GRPC_INTERNAL || return false
    req.ex === nothing || return false
    req.grpc_status == gRPCClient.GRPC_OK || return false
    req.code in _STALE_CONNECTION_CODES || return false
    return !req.response_read_header && req.response.size == 0
end

"""
    _call_unary(start, budget; cancelled = () -> false)

Run one unary RPC and return its response. `start(b)` sends it with a deadline of `b` seconds and
returns the call: the RPC invoked with `gRPCClient.gRPCAsync()`, so its request (and libcurl's result
code on it) is at hand. A call that met a stale pooled connection (`_stale_connection`) is sent once
more through `start`, after a random delay of at most `STALE_RESEND_MAX_DELAY`, with the deadline left
of `budget` (an infinite `budget` stays infinite). It is not sent again when that deadline would not
outlast the delay, or once `cancelled()` holds (the call it belongs to has already failed elsewhere).
Every failure but that one, and the second send's, is thrown as gRPCClient reported it, so what
callers do with it (`RetryPolicy`, which retries UNAVAILABLE and RESOURCE_EXHAUSTED, and the
shared-memory recovery, which handles FAILED_PRECONDITION) is unchanged.

`start` builds the request each time it is called, so a shared-memory request is armed afresh for the
second send, as a retry's is (CORE-3).
"""
function _call_unary(start, budget::Real; cancelled = () -> false)
    t0 = time()
    call = start(budget)
    try
        return fetch(call)
    catch e
        _stale_connection(call.req, e) || rethrow()
        delay = _stale_resend_delay()
        left = isfinite(budget) ? budget - (time() - t0) - delay : budget
        (left > 0 && !cancelled()) || rethrow()
        @debug "a call met a stale pooled connection; sending it once more on a fresh one" url = call.req.url code = Int(call.req.code) error = e.message delay_s = delay
        sleep(delay)
        return fetch(start(left))
    end
end
