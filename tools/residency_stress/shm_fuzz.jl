# Multi-process fuzz of the node-shared host weight store (ReactantServerCore's SharedWeightStore):
# several processes share a small node budget in /dev/shm and race to load, pin, release, evict,
# sweep stale versions of, and trim the same models' regions, checking every byte they read. No GPU
# and no Reactant: this exercises the flock protocol and the budget ledger directly, far faster than
# workers can.
#
#   julia --project=packages/ReactantServerCore tools/residency_stress/shm_fuzz.jl [PROCS] [SECONDS] [BUDGET_MB]
#
# The regions use a private prefix (`rsfz<pid>`), never the server's `rsw`, and are removed at the
# end. Exit 0 when every read verified and the ledger held, 1 otherwise.

using ReactantServerCore: ReactantServerCore
const WS = ReactantServerCore
using Random: Xoshiro

const NMODELS = 6
const NVERSIONS = 2
const PIN_P = parse(Float64, get(ENV, "FUZZ_PIN", "0.1"))   # chance an operation pins a floor

# Model m, version v: three Float32 tensors whose sizes differ per model.
specs(m) = [(Float32, (m * 100_003,)), (Float32, (17, 31 * m)), (Float32, (1 + 997 * m,))]
# Element i of tensor t of (m, v): a value no other (model, version, tensor) produces at that index.
pattern(m, v, t, i) = Float32((m * 7 + v * 3 + t) % 251) + Float32(i % 4093) / 4096.0f0
region_bytes(m) = WS._layout(specs(m))[2]

function fill_model!(arrays, m, v)
    for (t, a) in enumerate(arrays)
        for i in eachindex(a)
            a[i] = pattern(m, v, t, i)
        end
    end
    return nothing
end

function verify(arrays, m, v)
    for (t, a) in enumerate(arrays)
        for i in eachindex(a)
            a[i] == pattern(m, v, t, i) || return (t, i, a[i])
        end
    end
    return nothing
end

digest(m, v) = WS.weights_digest("m$m", specs(m); content = UInt64(1000 + v))

function child(prefix, budget, seconds, id)
    store = WS.SharedWeightStore(; mode = 0o600, budget, prefix)
    rng = Xoshiro(id)
    stats = Dict(:loads => 0, :hits => 0, :nofit => 0, :pins => 0, :trims => 0, :bad => 0, :errors => 0, :over => 0)
    pinned = Dict{Int, Int}()                  # model => version pinned by this process
    t_end = time() + seconds
    while time() < t_end
        m = rand(rng, 1:NMODELS)
        v = rand(rng, 1:NVERSIONS)
        op = rand(rng)
        try
            if op < 0.75                       # a load through the cache (an upload's lifetime)
                r = WS.with_cached_host_weights(store, "m$m", digest(m, v), specs(m), a -> fill_model!(a, m, v)) do arrays
                    sleep(rand(rng) * 0.01)    # an upload in progress: hold the region
                    verify(arrays, m, v)
                end
                if r === nothing
                    stats[:nofit] += 1
                else
                    hit, bad = r
                    stats[:loads] += 1
                    stats[:hits] += hit
                    if bad !== nothing
                        stats[:bad] += 1
                        println("BAD p$id m$m v$v tensor $(bad[1]) index $(bad[2]) value $(bad[3]) hit=$hit")
                    end
                end
            elseif op < 0.75 + PIN_P && !haskey(pinned, m)   # pin a floor (residency: system), check it, keep it
                arrays = WS.materialize_host_weights!(store, "m$m", digest(m, v), specs(m), a -> fill_model!(a, m, v))
                bad = verify(arrays, m, v)
                bad === nothing || (stats[:bad] += 1; println("BAD pinned p$id m$m v$v $bad"))
                pinned[m] = v
                stats[:pins] += 1
            elseif op < 0.95 && !isempty(pinned) && rand(rng) < 0.5     # release a floor
                k = rand(rng, collect(keys(pinned)))
                WS.release_host_weights!(store, "m$k")
                delete!(pinned, k)
            else
                stats[:trims] += WS.trim_host_cache!(store)
            end
            # The ledger: what the regions hold must fit the budget, except for pinned floors that
            # were admitted over it (they warn and load anyway).
            held = WS.host_cache_bytes(store)
            pinned_bytes = sum((region_bytes(k) for k in keys(pinned)); init = 0)
            held > budget + NMODELS * region_bytes(NMODELS) && (stats[:over] += 1)
        catch e
            stats[:errors] += 1
            println("ERROR p$id: ", sprint(showerror, e))
        end
    end
    for k in collect(keys(pinned))
        WS.release_host_weights!(store, "m$k")
    end
    println("STATS p$id ", join(["$k=$v" for (k, v) in sort!(collect(stats))], " "))
    return stats[:bad] + stats[:errors]
end

function parent(nprocs, seconds, budget_mb)
    prefix = "rsfz" * string(getpid())
    budget = budget_mb * 2^20
    total = sum(region_bytes(m) for m in 1:NMODELS)
    println("prefix=$prefix procs=$nprocs seconds=$seconds budget=$(Base.format_bytes(budget)) all models=$(Base.format_bytes(total))")
    me = abspath(@__FILE__)
    proj = Base.active_project()
    procs = [
        run(pipeline(`$(Base.julia_cmd()) --project=$proj $me --child $prefix $budget $seconds $i`; stdout = stdout, stderr = stderr); wait = false)
            for i in 1:nprocs
    ]
    # Sample the ledger while the children run.
    store = WS.SharedWeightStore(; mode = 0o600, budget, prefix)
    peak = 0
    t_end = time() + seconds + 30
    while any(process_running, procs) && time() < t_end
        peak = max(peak, WS.host_cache_bytes(store))
        sleep(0.05)
    end
    foreach(wait, procs)
    failed = count(p -> !success(p), procs)
    leftover = filter(f -> startswith(f, prefix), readdir("/dev/shm"))
    regions = filter(f -> !endswith(f, ".lock"), leftover)
    println("peak region bytes $(Base.format_bytes(peak)) (budget $(Base.format_bytes(budget)), one of each model $(Base.format_bytes(total)))")
    println("left in /dev/shm: $(length(regions)) regions, $(length(leftover) - length(regions)) lock files")
    for f in leftover
        rm(joinpath("/dev/shm", f); force = true)
    end
    println(failed == 0 ? "PASS" : "FAIL: $failed processes reported bad reads or errors")
    return failed == 0 ? 0 : 1
end

if length(ARGS) >= 1 && ARGS[1] == "--child"
    exit(child(ARGS[2], parse(Int, ARGS[3]), parse(Float64, ARGS[4]), parse(Int, ARGS[5])) == 0 ? 0 : 1)
else
    nprocs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
    seconds = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 60.0
    budget_mb = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    exit(parent(nprocs, seconds, budget_mb))
end
