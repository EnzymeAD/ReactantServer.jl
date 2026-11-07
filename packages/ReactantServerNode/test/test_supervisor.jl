# The supervision core, driven with real subprocesses (julia one-liners standing in for
# workers): line-prefixed multiplexing of stdout and stderr, restart with backoff, the
# crash-loop budget, partial-line flushing at EOF, and graceful vs forced shutdown.

_script_spec(name, script; grace = 5.0) =
    RSN.ChildSpec(name, `$(Base.julia_cmd()) --startup-file=no -e $script`, grace)

_sink_lines(sink) = split(String(take!(copy(sink))), '\n'; keepempty = false)

@testset "supervisor: prefixing and interleaving" begin
    sink = IOBuffer()
    specs = [
        _script_spec(
            "a", """
            for i in 1:20
                println("out line \$i")
            end
            for i in 1:5
                println(stderr, "err line \$i")
            end
            sleep(30)
            """
        ),
        _script_spec(
            "b", """
            for i in 1:20
                println("bee \$i")
            end
            sleep(30)
            """
        ),
    ]
    sup = RSN.Supervisor(specs; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(
        () -> count(l -> startswith(l, "[a] "), _sink_lines(sink)) >= 25 &&
            count(l -> startswith(l, "[b] "), _sink_lines(sink)) >= 20
    )
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0

    lines = _sink_lines(sink)
    # Every line is whole and carries exactly one prefix.
    @test all(l -> match(r"^\[(a|b|supervisor)\] ", l) !== nothing, lines)
    @test any(l -> l == "[a] err line 5", lines)        # stderr multiplexed too
    @test any(l -> l == "[b] bee 20", lines)
    @test any(l -> startswith(l, "[supervisor] started a"), lines)
end

@testset "supervisor: partial line at EOF survives" begin
    sink = IOBuffer()
    sup = RSN.Supervisor(
        [_script_spec("c", """print("dying words"); exit(7)""")]; sink = sink,
        max_restarts = 1
    )
    code = RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test code == 1                                      # budget breached → node exits 1
    lines = _sink_lines(sink)
    @test any(l -> l == "[c] dying words", lines)        # no trailing newline, still captured
    @test any(l -> occursin("exited (code=7", l), lines)
end

@testset "supervisor: restart with backoff and budget" begin
    sink = IOBuffer()
    sup = RSN.Supervisor([_script_spec("crashy", "exit(3)")]; sink = sink, max_restarts = 2)
    code = RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test code == 1
    lines = _sink_lines(sink)
    @test count(l -> startswith(l, "[supervisor] started crashy"), lines) == 3   # 1 + 2 restarts
    @test any(l -> occursin("restarting crashy in 1.0s", l), lines)
    @test any(l -> occursin("restarting crashy in 2.0s", l), lines)
    @test any(l -> occursin("failed 3 consecutive times", l), lines)
end

@testset "supervisor: graceful shutdown terminates children" begin
    sink = IOBuffer()
    sup = RSN.Supervisor([_script_spec("sleeper", "println(\"up\"); sleep(120)")]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> any(l -> l == "[sleeper] up", _sink_lines(sink)))
    t0 = time()
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    @test time() - t0 < 30                               # did not wait for the 120s sleep
    @test !process_running(sup.children[1].proc)
end

@testset "supervisor: SIGKILL fallback for a TERM-ignoring child" begin
    sink = IOBuffer()
    # A child that traps and ignores SIGTERM (a Julia child cannot: its runtime handles SIGTERM
    # on a dedicated signal thread); the supervisor must escalate to SIGKILL after the grace
    # window.
    sup = RSN.Supervisor(
        [
            RSN.ChildSpec(
                "stubborn",
                `sh -c 'trap "" TERM; echo "ignoring TERM"; sleep 120'`, 1.0
            ),
        ]; sink = sink
    )
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> any(l -> l == "[stubborn] ignoring TERM", _sink_lines(sink)))
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    @test any(l -> occursin("sending SIGKILL", l), _sink_lines(sink))
    @test !process_running(sup.children[1].proc)
end

@testset "supervisor: refuses --handle-signals=no with more than one thread" begin
    @test RSN.check_signal_threads(1, 8) === nothing     # Julia's own signal handling: any threads
    @test RSN.check_signal_threads(0, 1) === nothing     # handling off, one thread: the entrypoint
    @test_throws ReactantServerCore.ConfigError RSN.check_signal_threads(0, 2)
    err = try
        RSN.check_signal_threads(0, 2)
    catch e
        e
    end
    @test occursin("--threads=1,0", sprint(showerror, err))
    @test RSN.check_signal_threads() === nothing         # this test process keeps its handlers
end

# A child that speaks the drain protocol as the worker and the gateway do (ReactantServerCore's
# stdin watcher): it serves until the supervisor closes its stdin, acknowledges, spends `work`
# seconds "draining", and exits 0. It prints when its drain began and ended, so the test can order
# the children's drains on the host clock.
function _drain_spec(name, stage; drain = 10.0, grace = 5.0, work = 1.0)
    script = """
    using ReactantServerCore
    ReactantServerCore.watch_stdin_drain()
    ReactantServerCore.begin_serving!()
    println("up"); flush(stdout)
    ReactantServerCore.wait_for_drain_request()
    println("drain begins ", time()); flush(stdout)
    sleep($work)
    println("drain ends ", time()); flush(stdout)
    """
    cmd = addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`,
        ReactantServerCore.STDIN_DRAIN_ENV => string(drain)
    )
    return RSN.ChildSpec(name, cmd, grace, drain, stage)
end

function _stamp(lines, name, what)
    l = only(filter(l -> startswith(l, "[$name] drain $what "), lines))
    return parse(Float64, last(split(l)))
end

@testset "supervisor: drains the gateway first, then the workers in parallel" begin
    sink = IOBuffer()
    specs = [
        _drain_spec("w0", 2; work = 2.0),
        _drain_spec("gw", 1; work = 1.0),
        _drain_spec("w1", 2; work = 2.0),
    ]
    sup = RSN.Supervisor(specs; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> count(l -> endswith(l, "] up"), _sink_lines(sink)) == 3; timeout = 300)
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    lines = _sink_lines(sink)
    # The gateway finished its drain and exited before either worker was asked.
    gw_stopped = findfirst(l -> startswith(l, "[supervisor] gw stopped"), lines)
    @test gw_stopped !== nothing
    @test gw_stopped < findfirst(l -> startswith(l, "[supervisor] asking w0 to drain"), lines)
    @test gw_stopped < findfirst(l -> startswith(l, "[supervisor] asking w1 to drain"), lines)
    @test _stamp(lines, "gw", "ends") < min(_stamp(lines, "w0", "begins"), _stamp(lines, "w1", "begins"))
    # The workers drained at the same time, not one after the other.
    @test abs(_stamp(lines, "w0", "begins") - _stamp(lines, "w1", "begins")) < 1.5
    for name in ("gw", "w0", "w1")
        @test any(l -> startswith(l, "[supervisor] asking $name to drain (up to 10s)"), lines)
        @test any(l -> startswith(l, "[$name] ") && occursin(ReactantServerCore.DRAIN_ACK_LINE, l), lines)
        @test any(l -> startswith(l, "[supervisor] $name stopped (code=0, signal=0)"), lines)
    end
    @test !any(l -> occursin("SIGTERM", l) || occursin("SIGKILL", l), lines)
end

@testset "supervisor: a child that never acknowledges the drain gets SIGTERM" begin
    sink = IOBuffer()
    # Asked to drain (stdin closed), but it never reads stdin, as a worker still compiling does.
    deaf = RSN.ChildSpec(
        "deaf", `$(Base.julia_cmd()) --startup-file=no -e 'println("up"); flush(stdout); sleep(120)'`,
        5.0, 60.0, 1
    )
    sup = RSN.Supervisor([deaf]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> any(==("[deaf] up"), _sink_lines(sink)))
    t0 = time()
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    @test time() - t0 < 40                               # the ack window, not the 60 s drain
    lines = _sink_lines(sink)
    @test any(l -> occursin("deaf did not acknowledge the drain request within 10s; sending SIGTERM", l), lines)
    @test any(l -> startswith(l, "[supervisor] deaf stopped") && occursin("signal=15", l), lines)
    @test !any(l -> occursin("SIGKILL", l), lines)
end

# A sink whose writes wait until its gate opens, as a stdout whose reader has stopped reading does.
struct _GatedSink <: IO
    buf::IOBuffer
    gate::Base.Event
end
_GatedSink() = _GatedSink(IOBuffer(), Base.Event())
Base.unsafe_write(s::_GatedSink, p::Ptr{UInt8}, n::UInt) = (wait(s.gate); unsafe_write(s.buf, p, n))
Base.write(s::_GatedSink, b::UInt8) = (wait(s.gate); write(s.buf, b))
Base.flush(::_GatedSink) = nothing

_counted(lines, re) = [parse(Int, m[1]) for m in (match(re, l) for l in lines) if m !== nothing]

@testset "supervisor: a blocked log sink never blocks a child" begin
    sink = _GatedSink()
    done = joinpath(mktempdir(), "done")
    # Far more output than a pipe holds (64 KiB) or the log queue keeps (50 lines), then a marker
    # file. While the sink takes nothing, the child gets to the marker only if the supervisor keeps
    # reading its pipe. Before, the pump waited on the sink, the pipe filled, and the child blocked.
    script = """
    for i in 1:20000
        println("line \$i")
    end
    flush(stdout)
    touch($(repr(done)))
    sleep(120)
    """
    sup = RSN.Supervisor([_script_spec("chatty", script)]; sink = sink, log_lines = 50)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> isfile(done); timeout = 60)
    @test position(sink.buf) == 0                        # the sink has still taken nothing
    notify(sink.gate)
    @test wait_for(() -> any(l -> occursin("were dropped here", l), _sink_lines(sink.buf)))
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    lines = _sink_lines(sink.buf)
    got = _counted(lines, r"^\[chatty\] line (\d+)$")
    dropped = _counted(lines, r"^\[supervisor\] the log sink fell behind and \d+ line\(s\) were dropped here \(.*chatty: (\d+)")
    @test issorted(got) && allunique(got)                # one child's lines keep their order
    @test !isempty(dropped) && length(got) + sum(dropped) == 20000   # every line written or counted
    @test any(l -> startswith(l, "[supervisor] node stopped (exit 0)"), lines)
end

@testset "supervisor: getpid failing does not end supervision" begin
    # A child that exits at once is already reaped when its pid is read, and getpid throws ESRCH.
    # Before, that escaped the loop, the child's task ended, and it was never restarted.
    RSN._GETPID[] = p -> throw(Base.IOError("getpid: no such process (ESRCH)", Base.UV_ESRCH))
    try
        sink = IOBuffer()
        sup = RSN.Supervisor([_script_spec("quick", "sleep(0.5); exit(3)")]; sink = sink, max_restarts = 1)
        @test RSN.run_supervisor!(sup; install_signal_handlers = false) == 1   # budget, not a silent stop
        lines = _sink_lines(sink)
        @test count(l -> startswith(l, "[supervisor] started quick (pid unknown"), lines) == 2
        @test count(l -> occursin("quick exited (code=3", l), lines) == 2
        @test any(l -> occursin("restarting quick in 1.0s", l), lines)
    finally
        RSN._GETPID[] = getpid
    end
end

@testset "supervisor: an unexpected error in a run restarts the child" begin
    # The first pid lookup fails with an error the loop has no handler of its own for. The running
    # child is killed (a second copy would fight it for its ports) and restarted.
    calls = Ref(0)
    RSN._GETPID[] = p -> ((calls[] += 1) == 1 ? error("injected supervision fault") : getpid(p))
    try
        sink = IOBuffer()
        sup = RSN.Supervisor([_script_spec("steady", "println(\"up\"); flush(stdout); sleep(120)")]; sink = sink)
        t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
        @test wait_for(() -> count(==("[steady] up"), _sink_lines(sink)) == 2)
        lines = _sink_lines(sink)
        @test any(l -> startswith(l, "[supervisor] supervising steady failed unexpectedly; restarting it:"), lines)
        @test any(l -> occursin("injected supervision fault", l), lines)
        @test any(l -> l == "[supervisor] sending SIGKILL to steady, so it can be restarted", lines)
        # The first run failed before its "started" line; the second logs one.
        @test count(l -> startswith(l, "[supervisor] started steady (pid "), lines) == 1
        @test process_running(sup.children[1].proc)          # the second copy is supervised
        RSN.request_shutdown!(sup)
        @test fetch(t) == 0
    finally
        RSN._GETPID[] = getpid
    end
end

@testset "supervisor: supervision that cannot continue takes the node down" begin
    sink = IOBuffer()
    sup = RSN.Supervisor([_script_spec("x", "exit(0)")]; sink = sink)
    RSN._start_log_writer!(sup)
    RSN._supervise_task(sup, sup.children[1]; supervise = (s, c) -> error("loop broken"))
    @test take!(sup.exit_signal) == 1
    # A loop that returns while the node runs (rather than throwing) also takes it down.
    RSN._supervise_task(sup, sup.children[1]; supervise = (s, c) -> nothing)
    @test take!(sup.exit_signal) == 1
    RSN._close_log!(sup)
    lines = _sink_lines(sink)
    @test any(l -> l == "[supervisor] supervision of x cannot continue; shutting the node down:", lines)
    @test any(l -> occursin("loop broken", l), lines)
end

# A sink whose writes fail while `broken` is set, as a stdout whose reader has gone away.
mutable struct _BrokenSink <: IO
    const buf::IOBuffer
    broken::Bool
end
_check_sink(s::_BrokenSink) = s.broken && throw(Base.IOError("write: broken pipe (EPIPE)", Base.UV_EPIPE))
Base.unsafe_write(s::_BrokenSink, p::Ptr{UInt8}, n::UInt) = (_check_sink(s); unsafe_write(s.buf, p, n))
Base.write(s::_BrokenSink, b::UInt8) = (_check_sink(s); write(s.buf, b))
Base.flush(::_BrokenSink) = nothing

@testset "supervisor: a failing sink does not end supervision" begin
    # Before, the write error escaped the "started" line and ended the child's supervision silently.
    sink = _BrokenSink(IOBuffer(), true)
    sup = RSN.Supervisor([_script_spec("crashy", "sleep(1); exit(3)")]; sink = sink, max_restarts = 2)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    sleep(0.5)
    sink.broken = false
    @test fetch(t) == 1                                  # restarted until the budget ran out
    lines = _sink_lines(sink.buf)
    @test any(l -> occursin(r"^\[supervisor\] \d+ log line\(s\) were lost: writing them failed \(.*EPIPE", l), lines)
    @test count(l -> occursin("crashy exited (code=3", l), lines) == 3
end

# Kill the `sleep` grandchildren a test's children reported (`grandchild <pid>` lines).
function _kill_grandchildren(lines)
    for l in lines
        m = match(r"\] grandchild (\d+)$", l)
        m === nothing || run(ignorestatus(`kill $(m[1])`))
    end
    return nothing
end

@testset "supervisor: a grandchild holding a child's stderr delays neither restart nor shutdown" begin
    # The child exits, but a process it started inherited its stderr and keeps it open, as an
    # `nvidia-smi` a worker ran for a metrics scrape can. Before, the loop waited for end of file on
    # that pipe, so the restart (and the node's shutdown) waited for the grandchild.
    holder(name, rest) = RSN.ChildSpec(name, `sh -c "sleep 120 >/dev/null & echo \"grandchild \$!\"; $rest"`, 2.0)
    sink = IOBuffer()
    sup = RSN.Supervisor([holder("holder", "exit 3")]; sink = sink, max_restarts = 1)
    t0 = time()
    try
        @test RSN.run_supervisor!(sup; install_signal_handlers = false) == 1
        @test time() - t0 < 30                           # two runs and one restart, not 120 s
        lines = _sink_lines(sink)
        @test count(l -> startswith(l, "[supervisor] started holder"), lines) == 2
        @test count(l -> occursin("holder has exited, but a process it started still holds its output open", l), lines) == 2
    finally
        _kill_grandchildren(_sink_lines(sink))
    end

    sink = IOBuffer()
    sup = RSN.Supervisor([holder("runner", "echo up; exec sleep 120")]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    try
        @test wait_for(() -> any(==("[runner] up"), _sink_lines(sink)))
        t0 = time()
        RSN.request_shutdown!(sup)
        @test fetch(t) == 0
        @test time() - t0 < 15                           # the pipe's bound, not the grandchild's 120 s
        @test any(l -> startswith(l, "[supervisor] runner stopped"), _sink_lines(sink))
    finally
        _kill_grandchildren(_sink_lines(sink))
    end
end

# A stand-in worker with an HTTP listener on `port` answering /readyz and /healthz with 200. Its
# first run (no `marker` file yet) misbehaves 3 s after it starts, by `mode`: "stop" stops the whole
# process (SIGSTOP, as a deadlock or a wedged runtime would: nothing answers); "unlive" answers
# /healthz with 503, as a worker whose dispatch loop is stalled does. "never_ready" answers /readyz
# with 503 and never answers /healthz, as a worker still compiling its models does.
const _FAKE_WORKER = raw"""
using Sockets
port, mode, marker = parse(Int, ARGS[1]), ARGS[2], ARGS[3]
first_run = !isfile(marker)
touch(marker)
srv = listen(ip"127.0.0.1", port)
t0 = time()
println("up"); flush(stdout)
misbehaving() = first_run && time() - t0 > 3.0
while true
    s = accept(srv)
    @async try
        path = split(readline(s))[2]
        while !isempty(readline(s))
        end
        code = 200
        if mode == "never_ready"
            path == "/readyz" ? (code = 503) : sleep(3600)
        elseif mode == "unlive" && misbehaving() && path == "/healthz"
            code = 503
        elseif mode == "stop" && misbehaving()
            ccall(:kill, Cint, (Cint, Cint), getpid(), 19)
        end
        write(s, "HTTP/1.1 $code X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    catch
    finally
        close(s)
    end
end
"""

function _free_port()
    s = RSN.Sockets.listen(RSN.Sockets.IPv4("127.0.0.1"), 0)
    port = Int(RSN.Sockets.getsockname(s)[2])
    close(s)
    return port
end

function _fake_worker(name, mode, dir; interval = 0.5, timeout = 1.0, failures = 3)
    port = _free_port()
    cmd = `$(Base.julia_cmd()) --startup-file=no -e $_FAKE_WORKER $port $mode $(joinpath(dir, "marker"))`
    chk = RSN.LivenessCheck("127.0.0.1", port; interval_seconds = interval, timeout_seconds = timeout, failures = failures)
    return RSN.ChildSpec(name, addenv(cmd, "JULIA_LOAD_PATH" => "@stdlib"), 5.0, 0.0, 1, chk)
end

@testset "supervisor: a worker that stops answering once ready is killed and restarted ($mode)" for (mode, why) in (
        ("stop", "no answer to /healthz within 1s"), ("unlive", "/healthz answered HTTP 503"),
    )
    # Before, the supervisor reacted only to an exit, so a worker that hung (or whose dispatch loop
    # stalled) after it had become ready was never restarted.
    sink = IOBuffer()
    sup = RSN.Supervisor([_fake_worker("w", mode, mktempdir())]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    try
        @test wait_for(() -> count(==("[supervisor] w is ready; checking its liveness every 0.5s (SIGKILL after 3 failures in a row, 1s to answer each)"), _sink_lines(sink)) == 2; timeout = 90)
        lines = _sink_lines(sink)
        for k in 1:3
            @test any(==("[supervisor] w failed a liveness check ($k of 3): $why"), lines)
        end
        @test any(==("[supervisor] w failed 3 liveness checks in a row; sending SIGKILL so it is restarted"), lines)
        @test any(l -> startswith(l, "[supervisor] w exited (") && occursin("signal=9,", l), lines)
        @test count(l -> startswith(l, "[supervisor] started w (pid "), lines) == 2
        @test process_running(sup.children[1].proc)          # the restarted copy, checked again
    finally
        RSN.request_shutdown!(sup)
        @test fetch(t) == 0
    end
end

@testset "supervisor: a worker that has not been ready is never killed by its liveness check" begin
    # Startup can compile for hours. Until /readyz first answers 200, /healthz is not even asked.
    sink = IOBuffer()
    sup = RSN.Supervisor([_fake_worker("slow", "never_ready", mktempdir())]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    try
        @test wait_for(() -> any(==("[slow] up"), _sink_lines(sink)))
        sleep(6.0)                                           # a dozen check intervals
        lines = _sink_lines(sink)
        @test !any(l -> occursin("liveness", l) || occursin("SIGKILL", l), lines)
        @test count(l -> startswith(l, "[supervisor] started slow"), lines) == 1
        @test process_running(sup.children[1].proc)
    finally
        RSN.request_shutdown!(sup)
        @test fetch(t) == 0
    end
end

@testset "supervisor: grace windows run in parallel" begin
    sink = IOBuffer()
    # TERM-ignoring children (exec keeps the ignored disposition, so SIGKILL leaves no orphan).
    stubborn(name) = RSN.ChildSpec(name, `sh -c 'trap "" TERM; echo "ignoring TERM"; exec sleep 120'`, 2.0)
    sup = RSN.Supervisor([stubborn("s1"), stubborn("s2"), stubborn("s3")]; sink = sink)
    t = @async RSN.run_supervisor!(sup; install_signal_handlers = false)
    @test wait_for(() -> count(l -> endswith(l, "] ignoring TERM"), _sink_lines(sink)) == 3)
    t0 = time()
    RSN.request_shutdown!(sup)
    @test fetch(t) == 0
    @test time() - t0 < 5                                # one 2 s window, not three in a row
    lines = _sink_lines(sink)
    @test count(l -> occursin("did not exit within 2s; sending SIGKILL", l), lines) == 3
    @test all(c -> !process_running(c.proc), sup.children)
end
