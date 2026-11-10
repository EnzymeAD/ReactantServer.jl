# The wall-clock guard. `time()` reads the wall clock, which the host steps whenever its time is
# corrected, so an interval measured with it (an elapsed time, a timeout, a deadline, an age) is off
# by every step it spans; intervals are measured with `monotonic_time()` (src/clock.jl). The few
# `time()` calls that remain are listed below, each with why it needs the wall clock, and a package's
# suite fails on a call in its sources (`src/`, `ext/`) that is not listed, or on a listing that no
# longer matches a call.
#
# Each package's suite includes this file and checks its own sources (`check_wall_clock`), so the
# suite CI runs for a change to a package also checks that package. A call is matched by its whole
# line, stripped of indentation: changing such a line means updating its listing too, and saying why
# it still reads the wall clock.

# (package, file under the package, the line, why it reads the wall clock)
const WALL_CLOCK_SITES = [
    (
        "ReactantServerCore", "src/weight_store.jl",
        "bump_load_rank(rank, now=time(); halflife=HOST_CACHE_HALFLIFE_SECONDS) -> Float64",
        "docstring of the line below",
    ),
    (
        "ReactantServerCore", "src/weight_store.jl",
        "function bump_load_rank(rank::Real, now::Real = time(); halflife::Real = HOST_CACHE_HALFLIFE_SECONDS)",
        "a load rank is a timestamp kept as a region lock file's mtime, which other workers compare",
    ),
    (
        "ReactantServerCore", "src/weight_store.jl", "now = time()",
        "the access time `_set_mtime` gives a region lock file, a file timestamp",
    ),
    (
        "ReactantServer", "src/runtime/executable_cache.jl",
        "JSON3.write(io, Dict(\"format\" => EXEC_CACHE_FORMAT, \"sources\" => sources, \"updated_unix\" => floor(Int, time())))",
        "a date written into the executable cache's hash record on disk",
    ),
    (
        "ReactantServer", "src/runtime/weights_migration.jl",
        "settled = time() - st.ctime >= _SHA_RECORD_MIN_AGE[]",
        "the age of a weights file from its ctime, a wall-clock file timestamp",
    ),
    (
        "ReactantServerGateway", "src/lpt_packing.jl", "now = time()",
        "the time of day a repack reports (`last_repack_unix_seconds`); its interval math uses `mono`",
    ),
    (
        "ReactantServerGateway", "src/lpt_packing.jl",
        "time(), sort(ready_urls), poll.fleet_compute, s.compute_accum,",
        "the time of day a prober tick reports (`last_tick_unix_seconds`)",
    ),
]

# Files left out of the check: (package, file under the package, why).
const WALL_CLOCK_UNCHECKED = [
    (
        "ReactantServer", "src/transport/responsiveness.jl",
        "the worker's lag monitor: its clock is left to the change that reworks both monitors",
    ),
    (
        "ReactantServerGateway", "src/responsiveness.jl",
        "the gateway's lag monitor: its clock is left to the change that reworks both monitors",
    ),
]

# Every `time()` call in a package's `src/` and `ext/` as (file, line number, stripped line), leaving
# out the generated protobuf bindings, comments, and mentions in backticks (prose that names it).
function wall_clock_calls(root::AbstractString)
    found = Tuple{String, Int, String}[]
    for sub in ("src", "ext")
        isdir(joinpath(root, sub)) || continue
        for (dir, _, files) in walkdir(joinpath(root, sub)), f in sort(files)
            (endswith(f, ".jl") && !endswith(f, "_pb.jl")) || continue
            path = joinpath(dir, f)
            rel = replace(relpath(path, root), '\\' => '/')
            for (i, line) in enumerate(eachline(path))
                code = replace(line, r"(^|\s)#.*$" => "")
                occursin(r"(?<![`\w])time\(\)", code) && push!(found, (rel, i, String(strip(line))))
            end
        end
    end
    return found
end

"""
    check_wall_clock(package, root)

Check that the `time()` calls in the sources of `package` (whose directory is `root`) are exactly
the ones `WALL_CLOCK_SITES` lists for it.
"""
function check_wall_clock(package::AbstractString, root::AbstractString)
    unchecked = [e[2] for e in WALL_CLOCK_UNCHECKED if e[1] == package]
    listed = [(e[2], e[3]) for e in WALL_CLOCK_SITES if e[1] == package]
    unlisted = String[]
    for (rel, i, line) in wall_clock_calls(root)
        rel in unchecked && continue
        k = findfirst(==((rel, line)), listed)
        k === nothing ? push!(unlisted, "$package/$rel:$i: $line") : deleteat!(listed, k)
    end
    @testset "$package: time() only where the wall clock is the point" begin
        # A call here measures an interval: use `monotonic_time()`, or, if it needs the wall clock,
        # list it in WALL_CLOCK_SITES (packages/ReactantServerCore/test/wall_clock_guard.jl).
        @test unlisted == String[]
        # A listing whose call has gone or changed: remove or update it.
        @test listed == Tuple{String, String}[]
    end
    return nothing
end
