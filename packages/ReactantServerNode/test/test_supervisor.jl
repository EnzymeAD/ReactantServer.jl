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
