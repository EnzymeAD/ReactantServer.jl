# The cooperative drain: the per-server DrainGate, the process-wide drain request, and the stdin
# watcher the node supervisor's stop protocol relies on (end of file on stdin is the request).

const RSC = ReactantServerCore

@testset "drain: DrainGate admits until draining, then waits for the admitted" begin
    g = RSC.DrainGate()
    @test !RSC.is_draining(g)
    @test RSC.enter!(g)
    @test RSC.enter!(g)
    @test g.inflight[] == 2
    RSC.leave!(g)
    # One request still in flight: the drain refuses new ones at once and waits for it.
    t = @async (sleep(0.5); RSC.leave!(g))
    polls = Threads.Atomic{Int}(0)
    t0 = time()
    left = RSC.drain_inflight!(g, 10; poll = () -> Threads.atomic_add!(polls, 1))
    @test left == 0
    @test 0.4 < time() - t0 < 5
    @test polls[] >= 2                      # the poll hook ran on every check
    @test RSC.is_draining(g)
    @test !RSC.enter!(g)                    # refused, and nothing left counted
    @test g.inflight[] == 0
    wait(t)

    # A request that outlives the budget: the drain gives up and reports it.
    g2 = RSC.DrainGate()
    @test RSC.enter!(g2)
    t0 = time()
    @test RSC.drain_inflight!(g2, 0.3) == 1
    @test time() - t0 < 5
    RSC.leave!(g2)
end

@testset "drain: the process-wide request" begin
    RSC._reset_drain!()
    try
        @test !RSC.drain_requested()
        # Nothing serving: the request is recorded but nobody acts on it yet.
        @test !request_drain!(7)
        @test RSC.drain_requested()
        @test RSC.wait_for_drain_request() == 7.0     # already requested: returns at once
        @test !request_drain!(99)                       # the first request wins
        @test RSC.wait_for_drain_request() == 7.0
        RSC._reset_drain!()

        # A server waiting for a request wakes with its budget.
        RSC.begin_serving!()
        try
            waiter = @async RSC.wait_for_drain_request()
            sleep(0.2)
            @test !istaskdone(waiter)
            @test request_drain!(3)                     # a serving server acts on it
            @test timedwait(() -> istaskdone(waiter), 10) === :ok
            @test fetch(waiter) == 3.0
        finally
            RSC.end_serving!()
        end
        @test !request_drain!(5)                        # nothing serving any more
    finally
        RSC._reset_drain!()
    end
end

@testset "drain: stdin watcher" begin
    RSC._reset_drain!()
    try
        # Off unless the supervisor's variable is set to a positive budget.
        @test RSC.watch_stdin_drain(; env = Dict{String, String}()) === nothing
        @test RSC.watch_stdin_drain(; env = Dict(RSC.STDIN_DRAIN_ENV => "0")) === nothing
        @test RSC.watch_stdin_drain(; env = Dict(RSC.STDIN_DRAIN_ENV => "soon")) === nothing

        # Serving: end of file on the pipe requests the drain and prints the acknowledgement.
        inp = Pipe()
        Base.link_pipe!(inp)
        out = Pipe()
        RSC.begin_serving!()
        try
            redirect_stdout(out) do
                t = RSC.watch_stdin_drain(; env = Dict(RSC.STDIN_DRAIN_ENV => "12.5"), io = inp.out)
                @test t isa Task
                # Once per process: a second call starts nothing.
                @test RSC.watch_stdin_drain(; env = Dict(RSC.STDIN_DRAIN_ENV => "1"), io = inp.out) === nothing
                write(inp.in, "ignored bytes\n")           # data is not a request, end of file is
                sleep(0.2)
                @test !RSC.drain_requested()
                close(inp.in)
                @test timedwait(() -> istaskdone(t), 10) === :ok
            end
        finally
            RSC.end_serving!()
        end
        close(out.in)
        @test RSC.drain_requested()
        @test RSC.wait_for_drain_request() == 12.5
        @test readline(out) == RSC.DRAIN_ACK_LINE
    finally
        RSC._reset_drain!()
    end
end

# A real child process, as the supervisor runs one: stdin is a pipe and the variable is set.
function _drain_child(script; budget = "5")
    inp, out = Pipe(), Pipe()
    cmd = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`, RSC.STDIN_DRAIN_ENV => budget)
    p = run(pipeline(cmd; stdin = inp, stdout = out, stderr = stderr); wait = false)
    close(inp.out)
    close(out.in)
    return p, inp, out
end

@testset "drain: a child still starting exits 0 on the request" begin
    p, inp, out = _drain_child(
        """
        using ReactantServerCore
        ReactantServerCore.watch_stdin_drain()
        println("starting"); flush(stdout)
        sleep(120)   # never begins serving
        """
    )
    @test readline(out) == "starting"
    t0 = time()
    close(inp.in)
    @test timedwait(() -> process_exited(p), 60) === :ok
    @test p.exitcode == 0 && p.termsignal == 0
    @test time() - t0 < 30
end

@testset "drain: a serving child acknowledges and drains" begin
    p, inp, out = _drain_child(
        """
        using ReactantServerCore
        ReactantServerCore.watch_stdin_drain()
        ReactantServerCore.begin_serving!()
        println("serving"); flush(stdout)
        secs = ReactantServerCore.wait_for_drain_request()
        println("draining ", secs); flush(stdout)
        """; budget = "42"
    )
    @test readline(out) == "serving"
    close(inp.in)
    lines = String[]
    for _ in 1:2
        push!(lines, readline(out))
    end
    @test RSC.DRAIN_ACK_LINE in lines
    @test "draining 42.0" in lines
    @test timedwait(() -> process_exited(p), 60) === :ok
    @test p.exitcode == 0
end
