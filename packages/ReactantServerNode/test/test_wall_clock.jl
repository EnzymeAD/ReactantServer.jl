# This package's time() calls are the wall-clock sites listed for it in ReactantServerCore's
# test/wall_clock_guard.jl, the guard every package's suite shares; anything else measures an
# interval and belongs on monotonic_time().
const _WALL_CLOCK_GUARD = joinpath(@__DIR__, "..", "..", "ReactantServerCore", "test", "wall_clock_guard.jl")
isfile(_WALL_CLOCK_GUARD) ? include(_WALL_CLOCK_GUARD) : @info "no wall-clock guard here (not the monorepo); skipping it"
@isdefined(check_wall_clock) && check_wall_clock("ReactantServerNode", joinpath(@__DIR__, ".."))
