# The supervision core: spawn each child with piped stdout/stderr, pump both streams line by
# line onto the shared sink under a lock (so lines never interleave mid-line, the docker-compose
# look), restart children that die with capped exponential backoff, and on shutdown stop the
# children stage by stage: the gateway first, then the workers. A child with a drain budget is asked
# to drain by closing its stdin (see ReactantServerCore's drain.jl) and given that long to finish
# its requests; anything else is sent SIGTERM. Each stage's children are waited for in parallel, and
# stragglers are sent SIGKILL.

mutable struct Child
    spec::ChildSpec
    proc::Union{Base.Process, Nothing}
    restarts::Int          # consecutive restarts without a stable run
    started_at::Float64
    stdin::Union{Pipe, Nothing}   # our end of a draining child's stdin; closing it requests the drain
    drain_acked::Bool             # the child printed DRAIN_ACK_LINE since it last started
end
Child(spec::ChildSpec) = Child(spec, nothing, 0, 0.0, nothing, false)

struct Supervisor
    children::Vector{Child}
    sink::IO
    lk::ReentrantLock                  # serializes whole lines onto the sink
    shutting_down::Threads.Atomic{Bool}
    exit_signal::Channel{Int}          # first value wins: the supervisor's exit code
    max_restarts::Int                  # consecutive failed restarts before giving up; 0 = never
    backoff_cap::Float64
    stable_seconds::Float64            # uptime that resets the restart counter
end

function Supervisor(
        specs::Vector{ChildSpec}; sink::IO = stdout, max_restarts::Int = 0,
        backoff_cap::Real = 30.0, stable_seconds::Real = 60.0
    )
    return Supervisor(
        Child[Child(s) for s in specs], sink, ReentrantLock(),
        Threads.Atomic{Bool}(false), Channel{Int}(8), max_restarts,
        Float64(backoff_cap), Float64(stable_seconds)
    )
end

function _emit(sup::Supervisor, name::AbstractString, line::AbstractString)
    lock(sup.lk) do
        println(sup.sink, "[", name, "] ", line)
        flush(sup.sink)
    end
    return nothing
end

_slog(sup::Supervisor, msg::AbstractString) = _emit(sup, "supervisor", msg)

# Drain a child stream onto the sink. eachline also yields a trailing partial line at EOF, so
# the last words of a crashing child are never lost; pumping unconditionally means a chatty
# child can never block on a full pipe. The drain acknowledgement is recorded as it passes; the
# child writes it in one piece, but other output can still precede it on the same line.
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
    c.started_at = time()
    t1 = @async _pump(sup, c, out)
    t2 = @async _pump(sup, c, err)
    return proc, t1, t2
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
    deadline = time() + secs
    while !sup.shutting_down[] && time() < deadline
        sleep(min(0.2, max(deadline - time(), 0.01)))
    end
    return nothing
end

# One child's supervision loop: run, drain, log the exit, back off, restart. Runs until
# shutdown, or until the child exceeds max_restarts without a stable run (which takes the whole
# node down so the container restart policy can take over).
function _supervise!(sup::Supervisor, c::Child)
    while !sup.shutting_down[]
        local proc, t1, t2
        try
            proc, t1, t2 = _start!(sup, c)
        catch e
            _slog(sup, "$(c.spec.name) failed to spawn: $(sprint(showerror, e))")
            c.restarts += 1
            _check_restart_budget!(sup, c) || break
            _sleep_interruptible(sup, min(2.0^(c.restarts - 1), sup.backoff_cap))
            continue
        end
        _slog(sup, "started $(c.spec.name) (pid $(getpid(proc)))")
        wait(proc)
        wait(t1)
        wait(t2)
        uptime = time() - c.started_at
        sup.shutting_down[] && break
        _slog(sup, "$(c.spec.name) exited (code=$(proc.exitcode), signal=$(proc.termsignal), uptime=$(round(uptime; digits = 1))s)")
        uptime >= sup.stable_seconds && (c.restarts = 0)
        c.restarts += 1
        _check_restart_budget!(sup, c) || break
        delay = min(2.0^(c.restarts - 1), sup.backoff_cap)
        _slog(sup, "restarting $(c.spec.name) in $(round(delay; digits = 1))s")
        _sleep_interruptible(sup, delay)
    end
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
    t0 = time()
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
        now = time()
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
crash-loop budget breach, or `request_shutdown!`). Children are stopped gracefully before it
returns the exit code: the gateway is asked to drain and waited for, then the workers, in
parallel; a child that does not finish within its windows is sent SIGKILL. Refuses to run
(`check_signal_threads`) with Julia's signal handling off and more than one thread.
"""
function run_supervisor!(sup::Supervisor; install_signal_handlers::Bool = true)
    check_signal_threads()
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
    tasks = [@async _supervise!(sup, c) for c in sup.children]
    code = take!(sup.exit_signal)
    _shutdown_children!(sup)
    foreach(wait, tasks)
    close(cond)
    _slog(sup, "node stopped (exit $code)")
    return code
end
