# Cooperative drain: how a process running a blocking worker or gateway is asked to stop taking new
# requests, finish the ones it has, and exit.
#
# SIGTERM cannot do this. The worker and the gateway run with more than one thread, so they keep
# Julia's default signal handling (`--handle-signals=no` is unsafe there: GC safepoints then have no
# fault handler). Under the default handling SIGTERM runs the `atexit` hooks on the task that the
# interactive thread happened to be running, then kills the process with the signal. In a worker
# that task is usually the GPU dispatch loop, so a drain started from an `atexit` hook would wait for
# queued work that can no longer be dispatched. Instead the node supervisor (single threaded, and
# the only process that handles SIGTERM itself) asks each child explicitly: it spawns the child with
# a pipe on stdin and `STDIN_DRAIN_ENV` in its environment, and closes the pipe to request a drain.
# The child watches stdin on an ordinary task (`watch_stdin_drain`), answers with
# `DRAIN_ACK_LINE` on stdout once a server has accepted the request, and exits when the drain ends.
# A child that is still starting (nothing serving yet, so nothing to drain) exits at once instead.
#
# The request itself is process-wide: `request_drain!` records it and wakes every blocking server
# waiting in `wait_for_drain_request`. It stays set; a process is drained once.

"""
    STDIN_DRAIN_ENV

`REACTANT_STDIN_DRAIN_SECONDS`: when set to a positive number, end of file on stdin asks this
process to drain gracefully within that many seconds and exit. The node supervisor sets it on the
children it spawns with a pipe on stdin; a process whose stdin is a terminal or `/dev/null` must
not have it set.
"""
const STDIN_DRAIN_ENV = "REACTANT_STDIN_DRAIN_SECONDS"

"""
    DRAIN_ACK_LINE

The line a child prints on stdout once a serving worker or gateway has accepted a drain request,
so the supervisor knows the drain is under way and waits for it instead of sending SIGTERM.
"""
const DRAIN_ACK_LINE = "reactantserver: drain accepted"

mutable struct _DrainState
    const lock::ReentrantLock
    const event::Base.Event     # notified once a drain is requested; never reset outside tests
    requested::Bool
    seconds::Float64            # the drain budget of the first request
    serving::Int                # blocking servers that act on a request (see `begin_serving!`)
    watching::Bool              # the stdin watcher has been started
end

const _DRAIN = _DrainState(ReentrantLock(), Base.Event(), false, 0.0, 0, false)

"""
    request_drain!(seconds) -> Bool

Ask every blocking `serve` / `serve_gateway` in this process to drain: stop taking new requests,
let the admitted ones finish for at most `seconds`, then shut down and return. The first request
wins and later ones are ignored. Returns whether a blocking server is serving and will act on it;
a request made while none is (for instance during startup) is still recorded and is acted on as
soon as one begins serving.
"""
function request_drain!(seconds::Real)
    return lock(_DRAIN.lock) do
        if !_DRAIN.requested
            _DRAIN.requested = true
            _DRAIN.seconds = max(Float64(seconds), 0.0)
            notify(_DRAIN.event)
        end
        _DRAIN.serving > 0
    end
end

"Whether a drain has been requested in this process."
drain_requested() = lock(() -> _DRAIN.requested, _DRAIN.lock)

"""
    begin_serving!() / end_serving!()

Bracket the time a blocking server will act on a drain request. Between the two a request is left
to the server (`wait_for_drain_request`); outside them the stdin watcher treats a request as "stop
now", since nothing is serving that could drain. Call `begin_serving!` before the listener binds, so
no request can arrive after the port is open but before the server waits for one.
"""
begin_serving!() = lock(() -> (_DRAIN.serving += 1; nothing), _DRAIN.lock)
end_serving!() = lock(() -> (_DRAIN.serving = max(_DRAIN.serving - 1, 0); nothing), _DRAIN.lock)

"""
    wait_for_drain_request() -> Float64

Block the calling task until a drain is requested (returning at once if one already was) and return
its budget in seconds. Interruptible like any `wait`.
"""
function wait_for_drain_request()
    wait(_DRAIN.event)
    return lock(() -> _DRAIN.seconds, _DRAIN.lock)
end

"""
    watch_stdin_drain(; env=ENV, io=stdin) -> Union{Task, Nothing}

When `STDIN_DRAIN_ENV` is set to a positive number, start (once per process) a task that waits for
end of file on `io` and then requests a drain with that budget. If a blocking server is serving, it
prints `DRAIN_ACK_LINE` on stdout and leaves the drain to the server; otherwise the process is
still starting and has nothing to drain, so it exits with status 0 at once. Returns the task, or
`nothing` when the variable is unset or the watcher is already running.
"""
function watch_stdin_drain(; env::AbstractDict = ENV, io::IO = stdin)
    secs = tryparse(Float64, strip(get(env, STDIN_DRAIN_ENV, "")))
    (secs === nothing || !(secs > 0)) && return nothing
    return lock(_DRAIN.lock) do
        _DRAIN.watching && return nothing
        _DRAIN.watching = true
        return errormonitor(@async _await_stdin_drain(io, secs))
    end
end

function _await_stdin_drain(io::IO, secs::Float64)
    try
        # The supervisor writes nothing: end of file (its end of the pipe closed) is the request.
        while !eof(io)
            readavailable(io)
        end
    catch err
        @warn "drain: reading stdin failed; treating it as a drain request" exception = err
    end
    if lock(() -> _DRAIN.serving > 0, _DRAIN.lock)
        # Acknowledge in one write, and before the request wakes the server, so nothing the drain
        # itself prints can land inside the line.
        write(stdout, DRAIN_ACK_LINE * "\n")
        flush(stdout)
        request_drain!(secs)
        @info "drain requested: no longer accepting new requests; draining" budget_seconds = secs
    else
        request_drain!(secs)
        @info "drain requested before serving began; exiting"
        flush(stdout)
        flush(stderr)
        exit(0)
    end
    return nothing
end

"""
    DrainGate()

One server's admission gate for draining: counts the requests between admission and response, and
once `drain_inflight!` has started refuses new ones. The worker and the gateway each wrap
one in a gRPC interceptor that answers a refused request with UNAVAILABLE.
"""
struct DrainGate
    draining::Threads.Atomic{Bool}
    inflight::Threads.Atomic{Int}
end
DrainGate() = DrainGate(Threads.Atomic{Bool}(false), Threads.Atomic{Int}(0))

"Whether the gate has started draining (and refuses new requests)."
is_draining(g::DrainGate) = g.draining[]

"""
    enter!(g::DrainGate) -> Bool

Admit one request: `true` when admitted (pair it with `leave!`), `false` when the gate is
draining (refuse the request; there is nothing to undo).
"""
function enter!(g::DrainGate)
    # Count first, then check. `drain_inflight!` sets the flag before it reads the count, so a
    # request is either refused here or counted by the drain, never neither.
    Threads.atomic_add!(g.inflight, 1)
    g.draining[] || return true
    Threads.atomic_sub!(g.inflight, 1)
    return false
end

"Release a request admitted by `enter!`."
leave!(g::DrainGate) = (Threads.atomic_sub!(g.inflight, 1); nothing)

"""
    drain_inflight!(g::DrainGate, seconds; what="server", poll=nothing) -> Int

Start refusing new requests, then wait at most `seconds` for the admitted ones to finish (with
`seconds <= 0`, not at all). `poll`, when given, is called on every check (a server uses it to keep
its readiness reported as down). Returns how many requests were still in flight when the wait ended
(0 when it drained).
"""
function drain_inflight!(g::DrainGate, seconds::Real; what::AbstractString = "server", poll = nothing)
    g.draining[] = true
    seconds > 0 || return max(g.inflight[], 0)
    t0 = monotonic_time()
    deadline = t0 + Float64(seconds)
    @info "drain: $what refuses new requests (UNAVAILABLE) and reports not ready; waiting for in-flight requests" in_flight = g.inflight[] budget_seconds = seconds
    while true
        poll === nothing || poll()
        (g.inflight[] <= 0 || monotonic_time() >= deadline) && break
        sleep(0.05)
    end
    left = max(g.inflight[], 0)
    if left == 0
        @info "drain: $what drained" seconds = round(monotonic_time() - t0; digits = 2)
    else
        @warn "drain: $what still has requests in flight after the drain budget; stopping anyway" in_flight = left budget_seconds = seconds
    end
    return left
end

# Clear the process-wide request (and allow another stdin watcher) so a test can drain more than
# one server. Not for production use: a process is drained once.
function _reset_drain!()
    lock(_DRAIN.lock) do
        _DRAIN.requested = false
        _DRAIN.seconds = 0.0
        _DRAIN.watching = false
        Base.reset(_DRAIN.event)
    end
    return nothing
end
