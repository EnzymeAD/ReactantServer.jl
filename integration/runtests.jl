# Run the Julia tutorials as integration tests.
#
#     julia --project=integration integration/runtests.jl            # every tutorial
#     julia --project=integration integration/runtests.jl node       # just tutorials/node.jl
#
# Each tutorial runs in a fresh Julia process (servers bind fixed ports and nodes spawn children,
# so nothing may leak from one to the next) and a fresh working directory, exactly as a reader
# would run it top to bottom. The scripts carry their own `@test` lines, marked `#src` so the
# rendered docs omit them; a script that throws or fails a test fails its testset here.
using Test

# Run order matters only for readability of the log; every script is self-contained.
const TUTORIALS = ["export", "serve", "client", "node", "gateway"]

selected = isempty(ARGS) ? TUTORIALS : ARGS
for name in selected
    name in TUTORIALS || error("unknown tutorial $name; expected one of $(join(TUTORIALS, ", "))")
end

@testset "tutorials" begin
    for name in selected
        script = joinpath(@__DIR__, "tutorials", "$name.jl")
        @testset "$name" begin
            mktempdir() do dir
                # The child wraps the script in a testset so its #src assertions are counted, and
                # exits nonzero when any of them fails or the script throws.
                code = "using Test; @testset \"$name\" begin include(\"$(escape_string(script))\") end"
                cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) --threads=auto,1 -e $code`
                t = @elapsed ok = success(pipeline(Cmd(cmd; dir); stdout, stderr))
                @info "tutorial $name finished" ok seconds = round(t; digits = 1)
                @test ok
            end
        end
    end
end
