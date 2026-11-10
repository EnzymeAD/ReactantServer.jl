# The monotonic clock intervals are measured with.

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
