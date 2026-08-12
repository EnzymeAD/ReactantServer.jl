# The supervision core: spawn each child with piped stdout/stderr, pump both streams line by
# line onto the shared sink under a lock (so lines never interleave mid-line, the docker-compose
# look), restart children that die with capped exponential backoff, and on shutdown SIGTERM
# everything, wait out a grace window, and SIGKILL stragglers.

mutable struct Child
    spec::ChildSpec
    proc::Union{Base.Process, Nothing}
    restarts::Int          # consecutive restarts without a stable run
    started_at::Float64
end
Child(spec::ChildSpec) = Child(spec, nothing, 0, 0.0)

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
# child can never block on a full pipe.
function _pump(sup::Supervisor, name::AbstractString, io::IO)
    for line in eachline(io)
        _emit(sup, name, line)
    end
    return nothing
end

function _start!(sup::Supervisor, c::Child)
    out = Pipe()
    err = Pipe()
    proc = run(pipeline(c.spec.cmd; stdout = out, stderr = err); wait = false)
    close(out.in)
    close(err.in)
    c.proc = proc
    c.started_at = time()
    t1 = @async _pump(sup, c.spec.name, out)
    t2 = @async _pump(sup, c.spec.name, err)
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

# SIGTERM every live child, give each its grace window, SIGKILL what remains.
function _shutdown_children!(sup::Supervisor)
    sup.shutting_down[] = true
    for c in sup.children
        p = c.proc
        p === nothing && continue
        process_running(p) && kill(p, Base.SIGTERM)
    end
    for c in sup.children
        p = c.proc
        p === nothing && continue
        deadline = time() + c.spec.grace_seconds
        while process_running(p) && time() < deadline
            sleep(0.1)
        end
        if process_running(p)
            _slog(sup, "$(c.spec.name) did not exit within $(c.spec.grace_seconds)s; sending SIGKILL")
            kill(p, Base.SIGKILL)
        end
    end
    return nothing
end

"""
    run_supervisor!(sup; install_signal_handlers=true) -> Int

Run every child's supervision loop and block until shutdown is requested (SIGTERM/SIGINT, a
crash-loop budget breach, or `request_shutdown!`). Children are terminated gracefully before it
returns the exit code.
"""
function run_supervisor!(sup::Supervisor; install_signal_handlers::Bool = true)
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
