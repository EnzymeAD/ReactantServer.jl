# The supervision core: spawn each child with piped stdout/stderr, pump both streams line by
# line into a bounded log queue that one writer task drains onto the shared sink (so lines never
# interleave mid-line, the docker-compose look, and a sink that stops taking lines never stops a
# child), restart children that die with capped exponential backoff, kill (and so restart) a worker
# that has been ready but stops answering its liveness check, and on shutdown stop the
# children stage by stage: the gateway first, then the workers. A child with a drain budget is asked
# to drain by closing its stdin (see ReactantServerCore's drain.jl) and given that long to finish
# its requests; anything else is sent SIGTERM. Each stage's children are waited for in parallel, and
# stragglers are sent SIGKILL.

# One run of a child: its process and our ends of the pipes we gave it, with the tasks pumping them.
struct _Run
    proc::Base.Process
    out::Pipe
    err::Pipe
    pumps::Vector{Task}
end

mutable struct Child
    spec::ChildSpec
    proc::Union{Base.Process, Nothing}
    restarts::Int          # consecutive restarts without a stable run
    started_at::Float64
    stdin::Union{Pipe, Nothing}   # our end of a draining child's stdin; closing it requests the drain
    drain_acked::Bool             # the child printed DRAIN_ACK_LINE since it last started
    run::Union{_Run, Nothing}     # the current (or last) run
    pid::Union{Int, Nothing}      # the current (or last) run's pid, when it could be read
end
Child(spec::ChildSpec) = Child(spec, nothing, 0, 0.0, nothing, false, nothing, nothing)

# Lines the log queue gave up on while it was full, at the place in the stream where they were lost.
mutable struct _Dropped
    n::Int
    by::Dict{String, Int}   # per stream name ("worker0", "supervisor", ...)
end

# Every line bound for the sink, children's and the supervisor's own, in the order they arrived.
# Pumps and the supervision loops only ever add to it, which never blocks; one writer task takes
# from it and does the (possibly blocking) writes. When the queue holds `max_lines` lines or
# `max_bytes` bytes, a new line is dropped and counted in a `_Dropped` entry at the tail, which the
# writer turns into one line saying how many were lost there once the sink takes lines again.
mutable struct _LogQueue
    const cond::Threads.Condition
    const items::Vector{Union{String, _Dropped}}
    bytes::Int
    const max_lines::Int
    const max_bytes::Int
    writer::Union{Task, Nothing}   # nothing until `run_supervisor!`: lines are written directly before
    closed::Bool                   # set once the node has stopped; later lines are dropped
    failed::Int                    # lines the sink refused (a write error) since it last took one
    failure::String                # the first of those errors
end
_LogQueue(max_lines::Integer, max_bytes::Integer) = _LogQueue(
    Threads.Condition(), Union{String, _Dropped}[], 0, Int(max_lines), Int(max_bytes), nothing,
    false, 0, ""
)

# How many lines, and bytes, the log queue holds before it drops new ones: enough for a burst (a
# worker's startup output, a crash backtrace) while the sink is briefly slow, and a bound on what a
# stuck sink can cost in memory.
const _LOG_QUEUE_LINES = 10_000
const _LOG_QUEUE_BYTES = 8 << 20
# How long the node, once stopped, gives the writer to put the last queued lines on the sink.
const _LOG_FLUSH_SECONDS = 10.0

struct Supervisor
    children::Vector{Child}
    sink::IO
    log::_LogQueue                     # see `_LogQueue`; the only path onto `sink`
    shutting_down::Threads.Atomic{Bool}
    exit_signal::Channel{Int}          # first value wins: the supervisor's exit code
    max_restarts::Int                  # consecutive failed restarts before giving up; 0 = never
    backoff_cap::Float64
    stable_seconds::Float64            # uptime that resets the restart counter
end

function Supervisor(
        specs::Vector{ChildSpec}; sink::IO = stdout, max_restarts::Int = 0,
        backoff_cap::Real = 30.0, stable_seconds::Real = 60.0,
        log_lines::Integer = _LOG_QUEUE_LINES, log_bytes::Integer = _LOG_QUEUE_BYTES
    )
    return Supervisor(
        Child[Child(s) for s in specs], sink, _LogQueue(log_lines, log_bytes),
        Threads.Atomic{Bool}(false), Channel{Int}(8), max_restarts,
        Float64(backoff_cap), Float64(stable_seconds)
    )
end

# Queue one line from the stream `name`; never blocks. Before the supervisor runs (the notes
# `build_supervisor` logs) nothing else writes, so the line goes straight to the sink.
function _emit(sup::Supervisor, name::AbstractString, line::AbstractString)
    q = sup.log
    text = string("[", name, "] ", line)
    @lock q.cond begin
        q.closed && return nothing
        if q.writer === nothing
            println(sup.sink, text)
            flush(sup.sink)
        elseif length(q.items) >= q.max_lines || q.bytes + sizeof(text) > q.max_bytes
            gap = isempty(q.items) ? nothing : q.items[end]
            if !(gap isa _Dropped)
                gap = _Dropped(0, Dict{String, Int}())
                push!(q.items, gap)
            end
            gap.n += 1
            gap.by[name] = get(gap.by, name, 0) + 1
        else
            push!(q.items, text)
            q.bytes += sizeof(text)
            notify(q.cond)
        end
    end
    return nothing
end

# The supervisor's own messages, one prefixed line per line of `msg`.
function _slog(sup::Supervisor, msg::AbstractString)
    for line in eachsplit(msg, '\n')
        _emit(sup, "supervisor", line)
    end
    return nothing
end

_dropped_text(d::_Dropped) = string(
    "[supervisor] the log sink fell behind and ", d.n, " line(s) were dropped here (",
    join(("$k: $v" for (k, v) in sort!(collect(d.by))), ", "), ")"
)

# The writer task: take queued lines in order and write each to the sink, which may block for as
# long as the sink's reader stalls; only this task waits for it.
function _write_log(sup::Supervisor)
    q = sup.log
    while true
        item = @lock q.cond begin
            while isempty(q.items) && !q.closed
                wait(q.cond)
            end
            isempty(q.items) && return nothing   # closed, and everything queued is written
            it = popfirst!(q.items)
            it isa String && (q.bytes -= sizeof(it))
            it
        end
        _write_line(sup, item isa String ? item : _dropped_text(item))
    end
    return
end

# Write one line to the sink. A write that fails (a closed stdout, say) loses that line: the first
# failure is reported on stderr, and the next line the sink does take is followed by a count.
function _write_line(sup::Supervisor, text::AbstractString)
    q = sup.log
    try
        println(sup.sink, text)
        flush(sup.sink)
    catch err
        q.failed += 1
        if q.failed == 1
            q.failure = sprint(showerror, err)
            try
                println(stderr, "[supervisor] writing to the log failed ($(q.failure)); lines are lost until it works again")
            catch
            end
        end
        return nothing
    end
    if q.failed > 0
        n, why = q.failed, q.failure
        q.failed = 0
        _write_line(sup, "[supervisor] $n log line(s) were lost: writing them failed ($why)")
    end
    return nothing
end

function _start_log_writer!(sup::Supervisor)
    q = sup.log
    @lock q.cond begin
        q.writer === nothing && (q.writer = errormonitor(@async _write_log(sup)))
    end
    return nothing
end

# Stop taking lines and give the writer up to `seconds` to put the queued ones on the sink. A sink
# that is still stuck then keeps them: the node exits without waiting on its log reader.
function _close_log!(sup::Supervisor; seconds::Real = _LOG_FLUSH_SECONDS)
    q = sup.log
    w = @lock q.cond begin
        q.closed = true
        notify(q.cond)
        q.writer
    end
    w === nothing && return nothing
    if timedwait(() -> istaskdone(w), Float64(seconds); pollint = 0.02) !== :ok
        try
            println(stderr, "[supervisor] the log sink did not take the last lines within $(_secs(seconds))s; stopping without them")
        catch
        end
    end
    return nothing
end

# Drain a child stream into the log queue. eachline also yields a trailing partial line at EOF, so
# the last words of a crashing child are never lost; the queue never blocks, so the child can never
# block on a full pipe, whatever the sink does. The drain acknowledgement is recorded as it passes;
# the child writes it in one piece, but other output can still precede it on the same line.
function _pump(sup::Supervisor, c::Child, io::IO)
    for line in eachline(io)
        occursin(ReactantServerCore.DRAIN_ACK_LINE, line) && (c.drain_acked = true)
        _emit(sup, c.spec.name, line)
    end
    return nothing
end

function _start!(sup::Supervisor, c::Child)
    out = Pipe()
    err = Pipe()
    # A draining child reads its stdin from us, so closing our end is the drain request; any other
    # child inherits the supervisor's stdin, as before.
    inp = c.spec.drain_seconds > 0 ? Pipe() : nothing
    cmd = inp === nothing ? pipeline(c.spec.cmd; stdout = out, stderr = err) :
        pipeline(c.spec.cmd; stdin = inp, stdout = out, stderr = err)
    proc = run(cmd; wait = false)
    close(out.in)
    close(err.in)
    inp === nothing || close(inp.out)
    c.proc = proc
    c.stdin = inp
    c.drain_acked = false
    c.started_at = monotonic_time()
    r = _Run(proc, out, err, Task[@async(_pump(sup, c, out)), @async(_pump(sup, c, err))])
    c.run = r
    return r
end

# How the supervisor reads a child's pid: `getpid`, behind a Ref only so the tests can make it fail.
const _GETPID = Ref{Any}(getpid)

# The child's pid, or `nothing` when it has none to report: a child that exited at once is already
# reaped, and `getpid` fails with ESRCH. Any other failure is unexpected and propagates.
function _child_pid(p::Base.Process)
    return try
        _GETPID[](p)
    catch err
        (err isa Base.IOError && err.code == Base.UV_ESRCH) || rethrow()
        nothing
    end
end

_close_quietly(io) = try
    close(io)
catch
end

# How long, once a child has exited, its output pipes may stay open. They normally reach end of file
# at once; one held open longer belongs to a process the child started that inherited it (an
# `nvidia-smi` the worker ran, say), and must not hold up the restart or the node's shutdown.
const _PIPE_EOF_SECONDS = 5.0

# After the run has ended: read what is left in its pipes, for up to `_PIPE_EOF_SECONDS`, then stop
# reading them, and release our end of its stdin.
function _finish_run!(sup::Supervisor, c::Child, r::_Run)
    if timedwait(() -> all(istaskdone, r.pumps), _PIPE_EOF_SECONDS; pollint = 0.05) !== :ok
        _slog(sup, "$(c.spec.name) has exited, but a process it started still holds its output open; no longer reading it")
        _close_quietly(r.out)
        _close_quietly(r.err)
        timedwait(() -> all(istaskdone, r.pumps), 1.0; pollint = 0.05)
    end
    c.stdin === nothing || _close_quietly(c.stdin)
    return nothing
end

# Run the child once: spawn it, wait for it to exit, and log the exit. Returns its uptime in
# seconds, or `nothing` if it could not be spawned.
function _run_once!(sup::Supervisor, c::Child)
    r = try
        _start!(sup, c)
    catch e
        _slog(sup, "$(c.spec.name) failed to spawn: $(sprint(showerror, e))")
        return nothing
    end
    pid = _child_pid(r.proc)
    c.pid = pid === nothing ? nothing : Int(pid)
    _slog(sup, "started $(c.spec.name) (pid $(pid === nothing ? "unknown: it has already exited" : pid))")
    chk = c.spec.liveness
    chk === nothing || errormonitor(@async _watch_liveness(sup, c, r.proc, chk))
    wait(r.proc)
    _finish_run!(sup, c, r)
    uptime = monotonic_time() - c.started_at
    sup.shutting_down[] ||
        _slog(sup, "$(c.spec.name) exited (code=$(r.proc.exitcode), signal=$(r.proc.termsignal), uptime=$(round(uptime; digits = 1))s)")
    return uptime
end

# Sleep up to `secs` while `going()` holds. Returns whether it still holds.
function _pause(going, secs::Real)
    deadline = monotonic_time() + secs
    while going()
        now = monotonic_time()
        now >= deadline && return true
        sleep(min(0.2, deadline - now))
    end
    return false
end

# Check one run of a worker with `chk` (see `LivenessCheck`) until it exits or the node stops:
# wait for its first ready answer, then ask `/healthz` every interval, and send SIGKILL after
# `chk.failures` failures in a row. The exit then takes the ordinary restart path.
function _watch_liveness(sup::Supervisor, c::Child, proc::Base.Process, chk::LivenessCheck)
    name = c.spec.name
    going() = process_running(proc) && !sup.shutting_down[]
    try
        armed = false
        failed = 0
        while _pause(going, chk.interval)
            if !armed
                armed = _probe(chk, "/readyz") === nothing
                (armed && going()) &&
                    _slog(sup, "$name is ready; checking its liveness every $(_secs(chk.interval))s (SIGKILL after $(chk.failures) failures in a row, $(_secs(chk.timeout))s to answer each)")
                continue
            end
            why = _probe(chk, "/healthz")
            going() || break
            if why === nothing
                failed > 0 && _slog(sup, "$name answered its liveness check again after $failed failure(s)")
                failed = 0
                continue
            end
            failed += 1
            _slog(sup, "$name failed a liveness check ($failed of $(chk.failures)): $why")
            failed < chk.failures && continue
            _slog(sup, "$name failed $(chk.failures) liveness checks in a row; sending SIGKILL so it is restarted")
            kill(proc, Base.SIGKILL)
            _pause(() -> process_running(proc), 30.0) &&
                _slog(sup, "$name is still running 30s after SIGKILL (blocked in the kernel?); it is restarted once it exits")
            break
        end
    catch err
        _slog(sup, "liveness checks of $name stopped after an unexpected error; it is still restarted when it exits:\n$(_describe(err, catch_backtrace()))")
    end
    return nothing
end

_hostport(chk::LivenessCheck) = chk.ip isa Sockets.IPv6 ? "[$(chk.ip)]:$(chk.port)" : "$(chk.ip):$(chk.port)"

# One HTTP/1.1 GET of `path` on the child's listener: `nothing` when it answers 200 within the
# timeout, otherwise a short reason. Only the status line is read. Closing the socket on a timeout
# also cancels a connect still in progress, so nothing is left waiting behind a hung child.
function _probe(chk::LivenessCheck, path::AbstractString)
    sock = Sockets.TCPSocket()
    t = @async begin
        Sockets.connect(sock, chk.ip, chk.port)
        write(sock, "GET $path HTTP/1.1\r\nHost: $(_hostport(chk))\r\nConnection: close\r\n\r\n")
        readline(sock)
    end
    done = timedwait(() -> istaskdone(t), chk.timeout; pollint = 0.05) === :ok
    _close_quietly(sock)
    done || return "no answer to $path within $(_secs(chk.timeout))s"
    if istaskfailed(t)
        err = t.result
        return "$path: $(err isa Exception ? sprint(showerror, err) : string(err))"
    end
    status = fetch(t)
    m = match(r"^HTTP/\d(?:\.\d)? (\d{3})", status)
    m === nothing && return isempty(status) ? "$path: the connection closed without an answer" :
        "$path: unexpected answer $(repr(first(status, 80)))"
    code = parse(Int, m[1])
    return code == 200 ? nothing : "$path answered HTTP $code"
end

# One look at a worker's GPU through nvidia-smi: its used and total memory (MiB) and whether `pid`
# (the worker's previous run, if known) still holds memory on it. Returns a NamedTuple, or a Symbol
# when it cannot tell: `:timeout` (nvidia-smi did not answer in time), `:failed`, or `:absent` (no
# GPU with that index or UUID).
function _gpu_memory(r::GpuReclaim, pid)
    out, status = _read_bounded(
        `$(r.smi) --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits`, r.smi_timeout
    )
    status === :ok || return status
    row = nothing
    for line in eachsplit(out, '\n')
        f = strip.(split(line, ','))
        length(f) >= 4 && (f[1] == r.device || f[2] == r.device) && (row = f; break)
    end
    row === nothing && return :absent
    used, total = tryparse(Int, row[3]), tryparse(Int, row[4])
    (used === nothing || total === nothing || total <= 0) && return :failed
    held = false
    if pid !== nothing
        apps, status = _read_bounded(`$(r.smi) --query-compute-apps=pid,gpu_uuid --format=csv,noheader`, r.smi_timeout)
        status === :timeout && return :timeout
        status === :ok && (
            held = any(eachsplit(apps, '\n')) do line
                f = strip.(split(line, ','))
                length(f) >= 2 && f[1] == string(pid) && f[2] == row[2]
            end
        )
    end
    return (used = used, total = total, held = held)
end

# Before a worker is restarted: wait, for up to `r.timeout` seconds, until its GPU has released the
# memory its previous run held (see `GpuReclaim`), polling once a second, and log what happened. A
# GPU that cannot be observed (nvidia-smi does not answer or fails, or does not list it) is not
# waited for. Returns early once the node shuts down.
function _await_gpu_reclaim(sup::Supervisor, c::Child, r::GpuReclaim)
    r.timeout > 0 || return nothing
    name, gpu = c.spec.name, r.device
    t0 = monotonic_time()
    waiting = false
    while !sup.shutting_down[]
        st = _gpu_memory(r, c.pid)
        if st isa Symbol
            why = st === :timeout ? "nvidia-smi did not answer within $(_secs(r.smi_timeout))s" :
                st === :absent ? "nvidia-smi does not list GPU $gpu" : "nvidia-smi failed"
            _slog(sup, "$why; restarting $name without waiting for its GPU memory")
            return nothing
        end
        pct = 100 * st.used / st.total
        busy = pct > r.max_used_percent || st.held
        usage = "$(st.used)/$(st.total) MiB used ($(round(Int, pct))%)" *
            (st.held ? ", pid $(c.pid) still holds memory on it" : "")
        elapsed = monotonic_time() - t0
        if !busy
            waiting && _slog(sup, "GPU $gpu released $name's memory after $(round(elapsed; digits = 1))s: $usage")
            return nothing
        end
        if !waiting
            _slog(sup, "waiting up to $(_secs(r.timeout))s for GPU $gpu to release $name's memory before restarting it: $usage")
            waiting = true
        end
        if elapsed >= r.timeout
            _slog(sup, "GPU $gpu still has $usage after $(_secs(r.timeout))s; restarting $name anyway")
            return nothing
        end
        _sleep_interruptible(sup, min(1.0, r.timeout - elapsed))
    end
    return nothing
end

# Bring a child whose supervision hit an unexpected error back to a known state before it is
# restarted: kill it if it is still running (a second copy would fight it for its ports and its
# GPU), then stop reading its pipes.
function _put_down!(sup::Supervisor, c::Child)
    r = c.run
    r === nothing && return nothing
    if process_running(r.proc)
        _slog(sup, "sending SIGKILL to $(c.spec.name), so it can be restarted")
        kill(r.proc, Base.SIGKILL)
        timedwait(() -> !process_running(r.proc), _KILL_WAIT_SECONDS; pollint = 0.05) === :ok ||
            _slog(sup, "$(c.spec.name) is still running $(_secs(_KILL_WAIT_SECONDS))s after SIGKILL; restarting it anyway")
    end
    _close_quietly(r.out)
    _close_quietly(r.err)
    c.stdin === nothing || _close_quietly(c.stdin)
    return nothing
end

# An exception with the first lines of its backtrace, for the log; never throws itself.
function _describe(err, bt; lines::Integer = 12)
    text = try
        sprint(showerror, err, bt)
    catch
        "$(typeof(err)) (its message could not be shown)"
    end
    ls = split(text, '\n')
    length(ls) > lines && (ls = [ls[1:lines]; "  ..."])
    return join(ls, '\n')
end

"""
    request_shutdown!(sup, code=0)

Ask the supervisor to shut the node down (idempotent; the first requested exit code wins). The
signal bridge calls this on SIGTERM/SIGINT; tests and the crash-loop guard call it directly.
"""
function request_shutdown!(sup::Supervisor, code::Integer = 0)
    isready(sup.exit_signal) && return nothing
    try
        put!(sup.exit_signal, Int(code))
    catch
    end
    return nothing
end

# Backoff sleep that returns early once shutdown begins, so a crash-looping child never delays
# node shutdown.
function _sleep_interruptible(sup::Supervisor, secs::Real)
    deadline = monotonic_time() + secs
    while !sup.shutting_down[] && monotonic_time() < deadline
        sleep(min(0.2, max(deadline - monotonic_time(), 0.01)))
    end
    return nothing
end

# One child's supervision loop: run, drain, log the exit, back off, restart. Runs until
# shutdown, or until the child exceeds max_restarts without a stable run (which takes the whole
# node down so the container restart policy can take over). An unexpected error in one run is
# logged, the child is put down if it still runs, and it is restarted like a child that exited.
function _supervise!(sup::Supervisor, c::Child)
    while !sup.shutting_down[]
        uptime = nothing
        try
            uptime = _run_once!(sup, c)
        catch err
            sup.shutting_down[] && break
            _slog(sup, "supervising $(c.spec.name) failed unexpectedly; restarting it:\n$(_describe(err, catch_backtrace()))")
            _put_down!(sup, c)
        end
        sup.shutting_down[] && break
        uptime !== nothing && uptime >= sup.stable_seconds && (c.restarts = 0)
        c.restarts += 1
        _check_restart_budget!(sup, c) || break
        delay = min(2.0^(c.restarts - 1), sup.backoff_cap)
        _slog(sup, "restarting $(c.spec.name) in $(round(delay; digits = 1))s")
        _sleep_interruptible(sup, delay)
        if c.spec.reclaim !== nothing
            try
                _await_gpu_reclaim(sup, c, c.spec.reclaim)
            catch err
                _slog(sup, "waiting for $(c.spec.name)'s GPU memory failed; restarting it now:\n$(_describe(err, catch_backtrace()))")
            end
        end
    end
    return nothing
end

# The task behind each child's supervision. Supervision must never end silently while the node
# runs: an error the loop itself cannot absorb, or the loop ending for any reason but a shutdown it
# asked for, is logged and takes the node down with status 1, so the container restart policy acts.
function _supervise_task(sup::Supervisor, c::Child; supervise = _supervise!)
    try
        supervise(sup, c)
    catch err
        _slog(sup, "supervision of $(c.spec.name) cannot continue; shutting the node down:\n$(_describe(err, catch_backtrace()))")
        request_shutdown!(sup, 1)
        return nothing
    end
    # Returning while the node runs means the restart budget ran out, which already asked for exit 1.
    sup.shutting_down[] || request_shutdown!(sup, 1)
    return nothing
end

function _check_restart_budget!(sup::Supervisor, c::Child)
    (sup.max_restarts > 0 && c.restarts > sup.max_restarts) || return true
    _slog(sup, "$(c.spec.name) failed $(c.restarts) consecutive times (max $(sup.max_restarts)); shutting the node down")
    request_shutdown!(sup, 1)
    return false
end

# How long a child asked to drain has to acknowledge it (DRAIN_ACK_LINE) before it is sent SIGTERM
# instead: one that is still starting, or stuck, cannot drain, and should not hold shutdown for its
# whole drain budget. The child reads the request on Julia's event loop, which its main thread
# drives, so a worker busy compiling (or running a long batch on that thread) answers late.
const _DRAIN_ACK_SECONDS = 10.0
# How long to wait for a child to disappear after SIGKILL before giving up on it.
const _KILL_WAIT_SECONDS = 5.0

# Stop every live child, one stage at a time in stage order (the gateway, then the workers), so the
# workers keep serving the requests the gateway is still draining.
function _shutdown_children!(sup::Supervisor)
    sup.shutting_down[] = true
    for stage in sort!(unique(c.spec.stage for c in sup.children))
        _stop_stage!(sup, [c for c in sup.children if c.spec.stage == stage])
    end
    return nothing
end

# One child on its way down. `phase` is :ack (drain requested, waiting for the acknowledgement),
# :drain (acknowledged, draining), :term (SIGTERM sent), :kill (SIGKILL sent) or :done; `deadline`
# is when the current phase gives up.
mutable struct _Stopping
    c::Child
    p::Base.Process
    phase::Symbol
    deadline::Float64
end

# Ask every live child of one stage to stop, then wait for all of them at once: each child's
# windows run concurrently, so the stage takes as long as its slowest child, not the sum.
function _stop_stage!(sup::Supervisor, group::Vector{Child})
    t0 = monotonic_time()
    stopping = _Stopping[]
    for c in group
        p = c.proc
        (p === nothing || !process_running(p)) && continue
        if c.stdin !== nothing
            _slog(sup, "asking $(c.spec.name) to drain (up to $(_secs(c.spec.drain_seconds))s)")
            try
                close(c.stdin.in)
            catch
            end
            push!(stopping, _Stopping(c, p, :ack, t0 + _DRAIN_ACK_SECONDS))
        else
            kill(p, Base.SIGTERM)
            push!(stopping, _Stopping(c, p, :term, t0 + c.spec.grace_seconds))
        end
    end
    while !isempty(stopping)
        now = monotonic_time()
        for s in stopping
            c, p = s.c, s.p
            if !process_running(p)
                _slog(sup, "$(c.spec.name) stopped (code=$(p.exitcode), signal=$(p.termsignal)) after $(round(now - t0; digits = 1))s")
                s.phase = :done
            elseif s.phase === :ack && c.drain_acked
                s.phase = :drain
                s.deadline = t0 + c.spec.drain_seconds + c.spec.grace_seconds
            elseif now >= s.deadline
                if s.phase === :ack
                    _slog(sup, "$(c.spec.name) did not acknowledge the drain request within $(_secs(_DRAIN_ACK_SECONDS))s; sending SIGTERM")
                    kill(p, Base.SIGTERM)
                    s.phase, s.deadline = :term, now + c.spec.grace_seconds
                elseif s.phase === :kill
                    _slog(sup, "$(c.spec.name) is still running $(_secs(_KILL_WAIT_SECONDS))s after SIGKILL; giving up on it")
                    s.phase = :done
                else
                    window = s.phase === :drain ? c.spec.drain_seconds + c.spec.grace_seconds : c.spec.grace_seconds
                    _slog(sup, "$(c.spec.name) did not exit within $(_secs(window))s; sending SIGKILL")
                    kill(p, Base.SIGKILL)
                    s.phase, s.deadline = :kill, now + _KILL_WAIT_SECONDS
                end
            end
        end
        filter!(s -> s.phase !== :done, stopping)
        isempty(stopping) || sleep(0.1)
    end
    return nothing
end

_secs(x::Real) = isinteger(x) ? Int(x) : x

"""
    run_supervisor!(sup; install_signal_handlers=true) -> Int

Run every child's supervision loop and block until shutdown is requested (SIGTERM/SIGINT, a
crash-loop budget breach, `request_shutdown!`, or exit code 1 when supervising a child cannot
continue, which is logged; it never stops silently). Children are stopped gracefully before it
returns the exit code: the gateway is asked to drain and waited for, then the workers, in
parallel; a child that does not finish within its windows is sent SIGKILL. Refuses to run
(`check_signal_threads`) with Julia's signal handling off and more than one thread.
"""
function run_supervisor!(sup::Supervisor; install_signal_handlers::Bool = true)
    check_signal_threads()
    _start_log_writer!(sup)
    cond = Base.AsyncCondition()
    if install_signal_handlers
        install_term_handlers!(cond)
    end
    @async begin
        try
            wait(cond)
            _slog(sup, "received shutdown signal")
            request_shutdown!(sup, 0)
        catch
            # cond closed below after a non-signal shutdown; nothing to do
        end
    end
    tasks = [errormonitor(@async _supervise_task(sup, c)) for c in sup.children]
    code = take!(sup.exit_signal)
    _shutdown_children!(sup)
    # Every child has now exited or been given up on after SIGKILL; each loop ends once its child's
    # pipes are done (`_finish_run!`). A child that did not die keeps its loop waiting, so bound this.
    if timedwait(() -> all(istaskdone, tasks), _PIPE_EOF_SECONDS + 5.0; pollint = 0.05) !== :ok
        stuck = [c.spec.name for (c, t) in zip(sup.children, tasks) if !istaskdone(t)]
        _slog(sup, "stopping without waiting further for $(join(stuck, ", "))")
    end
    close(cond)
    _slog(sup, "node stopped (exit $code)")
    _close_log!(sup)
    return code
end
