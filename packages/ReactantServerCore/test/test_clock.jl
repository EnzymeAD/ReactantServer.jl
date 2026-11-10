# The monotonic clock intervals are measured with, and the guard that keeps `time()` to the sites
# that need the wall clock.

@testset "monotonic_time: seconds on the clock time_ns() reads" begin
    a = monotonic_time()
    t = time_ns() / 1.0e9
    b = monotonic_time()
    @test a isa Float64
    @test a <= t <= b
    @test all(diff([monotonic_time() for _ in 1:1000]) .>= 0)   # never goes back
    sleep(0.2)
    @test 0.15 < monotonic_time() - b < 5
end

# The guard every package's suite runs over its own sources; see wall_clock_guard.jl.
include("wall_clock_guard.jl")

@testset "wall-clock guard: finds calls, not comments or prose" begin
    root = mktempdir()
    mkdir(joinpath(root, "src"))
    write(
        joinpath(root, "src", "a.jl"), """
        t0 = time()                 # a call
        x = Base.time() - t0        # another
        y = monotonic_time()        # not time()
        # time() in a comment
        \"\"\"Prose naming `time()`.\"\"\"
        z = time_ns()
        """
    )
    write(joinpath(root, "src", "gen_pb.jl"), "t = time()\n")      # generated bindings: skipped
    @test [(f, i) for (f, i, _) in wall_clock_calls(root)] == [("src/a.jl", 1), ("src/a.jl", 2)]
end

check_wall_clock("ReactantServerCore", joinpath(@__DIR__, ".."))
