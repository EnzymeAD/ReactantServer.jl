# Convert bundles' legacy-order weights files to the program's axis order offline, with the same
# code (and the same proofs) the worker runs under `runtime.weights_migration`. Use it for model
# directories the workers cannot write, or to convert ahead of a deploy.
#
# Usage:
#   julia --project=packages/ReactantServer tools/migrate_weights.jl (--copy | --replace) <path>...
#
#   --copy      write a verified weights.T.safetensors beside each original and point the manifest
#               at it; the original is untouched and older servers keep loading it
#   --replace   also swap the copy in as weights.safetensors and mark the manifest
#               format_version "2.1" (older servers refuse such a bundle); use it once every server
#               reading the directory runs ReactantServer 0.2 or later
#   <path>      a bundle directory, or a model directory whose subdirectories are bundles
#
# Exits nonzero if any bundle failed; a failed bundle is left loadable as it was.

using ReactantServer

function main(args)
    modes = Dict("--copy" => ReactantServer.MIGRATE_COPY, "--replace" => ReactantServer.MIGRATE_REPLACE)
    flags = filter(a -> startswith(a, "--"), args)
    paths = filter(a -> !startswith(a, "--"), args)
    (length(flags) == 1 && haskey(modes, only(flags)) && !isempty(paths)) || begin
        println(stderr, "usage: migrate_weights.jl (--copy | --replace) <bundle or model dir>...")
        return 2
    end
    mode = modes[only(flags)]
    bundles = String[]
    for p in paths
        if isfile(joinpath(p, "manifest.yaml"))
            push!(bundles, p)
        elseif isdir(p)
            append!(bundles, filter(d -> isfile(joinpath(d, "manifest.yaml")), readdir(p; join = true)))
        else
            println(stderr, "not a bundle or model directory: $p")
            return 2
        end
    end
    failed = 0
    for b in bundles
        r = ReactantServer.migrate_bundle_weights!(b, mode)
        println(rpad(string(r), 10), b)
        failed += r === :failed
    end
    println("$(length(bundles)) bundles, $failed failed")
    return failed == 0 ? 0 : 1
end

exit(main(ARGS))
