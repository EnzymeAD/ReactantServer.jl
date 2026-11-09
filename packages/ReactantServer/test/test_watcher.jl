# Dynamic model-directory watching: bundle signatures, repository scanning, and the
# load/reload/unload + debounce behavior of a poll round. The poll round is driven manually
# (_watch_once!) against a real CPU backend so the swaps are deterministic with no timing.

using ReactantServer: bundle_signature, scan_repository, BundleWatcher, _watch_once!,
    get_model, infer, InferRequest, NamedTensor
using Logging: with_logger
using Test: TestLogger

# Run one poll under a captured logger and return the watcher's decided action
# (:load/:reload/:unload), or `nothing` when the poll applied no change (e.g. the
# debounce's first observation). The "watcher: change detected" log fires on this (the test) task, inside
# `_apply_change!`, so it is captured here; the load/unload summaries run on the dispatch thread.
function _poll_action(w)
    logger = TestLogger()
    with_logger(logger) do
        _watch_once!(w)
    end
    recs = filter(r -> r.message == "watcher: change detected", logger.logs)
    return isempty(recs) ? nothing : recs[1].kwargs[:action]
end

# A tiny fixed-shape (4) scale model: y = x .* w. Reused for every bundle written below.
const _W_MANIFEST = """
format_version: "2.0"
name: PLACEHOLDER
executable_inputs:
  - { name: x, dtype: f32, shape: c, dims: { c: 4 } }
executable_outputs:
  - { name: y, dtype: f32, shape: c, dims: { c: 4 } }
batching:
  compiled_batch_sizes: [1]
"""
const _W_MLIR = """
module {
  func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.multiply %x, %w : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

_w_manifest(name) = replace(_W_MANIFEST, "PLACEHOLDER" => name)

_write_scale_bundle(root, name; w = Float32[2, 2, 2, 2]) =
    write_bundle(
    root, name; manifest_yaml = _w_manifest(name), mlir_text = _W_MLIR,
    weights = Dict("w" => w), argument_order = ["w"]
)

@testset "bundle_signature and scan_repository" begin
    mktempdir() do root
        _write_scale_bundle(root, "m1")
        dir1 = joinpath(root, "m1")

        sig1 = bundle_signature(dir1)
        names = first.(sig1)
        @test "manifest.yaml" in names
        @test "model.mlir" in names
        @test "weights.safetensors" in names

        scan = scan_repository([root], nothing)
        @test haskey(scan, "m1")
        @test scan["m1"][1] == dir1
        @test scan["m1"][2] == sig1

        # The include allowlist filters by directory/model name.
        @test isempty(scan_repository([root], Set(["other"])))
        @test haskey(scan_repository([root], Set(["m1"])), "m1")

        # A missing root is skipped rather than raising (watcher must survive a transient mount).
        @test isempty(scan_repository([joinpath(root, "does-not-exist")], nothing))

        # Changing a bundle file changes its signature (size differs here, independent of mtime
        # resolution).
        write(joinpath(dir1, "manifest.yaml"), _w_manifest("m1") * "\n# touched\n")
        @test bundle_signature(dir1) != sig1
    end
end

@testset "watcher load / reload / unload with debounce (CPU)" begin
    mktempdir() do root
        backend = ReactantServer.ReactantBackend()
        runtime = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        pool = ReactantServer.resolve_client(backend, runtime)
        reg = ReactantServer.ModelRegistry()
        sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
        ReactantServer.start!(sched)
        cfg = ReactantServer.ServerConfig(
            [root], "", runtime,
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", 0)
        )
        # Construct the watcher while the repo is empty, so the startup-seeded `seen` is empty and
        # the bundle added below registers as a new model.
        w = BundleWatcher(sched, backend, pool, cfg; interval = 0.1, on_demand = false)

        try
            @testset "new bundle: debounced, then loaded" begin
                _write_scale_bundle(root, "scale4")
                # First poll only marks the change pending; it must not load yet (guards against a
                # half-written bundle).
                @test _poll_action(w) === nothing    # first poll: pending, no decision logged
                @test get_model(reg, "scale4") === nothing
                # Second poll sees the same signature (stable) and decides to load.
                @test _poll_action(w) === :load
                entry = get_model(reg, "scale4")
                @test entry !== nothing
                @test entry.executable !== nothing
                # It actually serves.
                out = infer(sched, InferRequest("scale4", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))
                @test out[1].data == Float32[2, 4, 6, 8]
            end

            @testset "changed bundle: reloaded" begin
                old = get_model(reg, "scale4")
                # New weights (4x). Size identical, so rewrite the manifest too to guarantee the
                # signature changes regardless of mtime granularity.
                _write_scale_bundle(root, "scale4"; w = Float32[4, 4, 4, 4])
                write(joinpath(root, "scale4", "manifest.yaml"), _w_manifest("scale4") * "\n# v2\n")
                @test _poll_action(w) === nothing     # pending
                @test get_model(reg, "scale4") === old   # not yet swapped
                @test _poll_action(w) === :reload     # stable -> reload
                new = get_model(reg, "scale4")
                @test new !== nothing
                @test new !== old                    # load_model! replaced the entry
                out = infer(sched, InferRequest("scale4", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))
                @test out[1].data == Float32[4, 8, 12, 16]
            end

            @testset "removed bundle: unloaded" begin
                rm(joinpath(root, "scale4"); recursive = true)
                @test _poll_action(w) === nothing     # pending removal
                @test get_model(reg, "scale4") !== nothing
                @test _poll_action(w) === :unload     # stable -> unload
                @test get_model(reg, "scale4") === nothing
                # Inference against the unloaded model now errors.
                @test_throws Exception infer(
                    sched,
                    InferRequest("scale4", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])])
                )
            end
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "a bundle migrated as it loads is loaded once (CPU)" begin
    mktempdir() do root
        backend = ReactantServer.ReactantBackend()
        rt = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        runtime = ReactantServer.RuntimeConfig(
            (f === :weights_migration ? ReactantServer.MIGRATE_REPLACE : getfield(rt, f) for f in fieldnames(typeof(rt)))...
        )
        pool = ReactantServer.resolve_client(backend, runtime)
        reg = ReactantServer.ModelRegistry()
        sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
        ReactantServer.start!(sched)
        cfg = ReactantServer.ServerConfig(
            [root], "", runtime,
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", 0)
        )
        w = BundleWatcher(sched, backend, pool, cfg; interval = 0.1, on_demand = false)
        try
            dir = _write_scale_bundle(root, "mig4")          # legacy axis order
            @test _poll_action(w) === nothing                # pending
            @test _poll_action(w) === :load
            @test occursin("format_version: \"2.1\"", read(joinpath(dir, "manifest.yaml"), String))   # migrated
            # The migration rewrote the bundle's files during the load; that is not a new change.
            @test _poll_action(w) === nothing
            @test _poll_action(w) === nothing
            out = infer(sched, InferRequest("mig4", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))
            @test out[1].data == Float32[2, 4, 6, 8]
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

# A second program for the scale bundle: y = x .* w .+ x, so a version switch shows in the output.
const _W_MLIR_PLUS_X = """
module {
  func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.multiply %x, %w : tensor<4xf32>
    %1 = stablehlo.add %0, %x : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"""

# A model.jl that, the first time the loader includes it, replaces its bundle's model.mlir with
# `mlir` and makes each `moves` rename: files changing after the loader read them, as a deploy
# landing during a load does. The loader reads model.mlir before it includes model.jl.
function _change_once_jl(name, mlir; moves = Pair{String, String}[])
    lines = String[
        "let dir = @__DIR__",
        "    if !isfile(joinpath(dir, \"changed.flag\"))",
        "        write(joinpath(dir, \"changed.flag\"), \"\")",
        "        cp($(repr(mlir)), joinpath(dir, \"model.mlir\"); force = true)",
    ]
    for (a, b) in moves
        push!(lines, "        mv($(repr(a)), $(repr(b)))")
    end
    append!(lines, ["    end", "end", "register_model($(repr(name)))"])
    return join(lines, "\n") * "\n"
end

# A CPU scheduler, started, and a dynamic-mode config over `root`.
function _cpu_watch_env(root; poll = 1.0)
    backend = ReactantServer.ReactantBackend()
    runtime = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
    pool = ReactantServer.resolve_client(backend, runtime)
    reg = ReactantServer.ModelRegistry()
    sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
    cfg = ReactantServer.ServerConfig(
        [root], "", runtime, ReactantServer.SchedulerConfig(30.0, 64, 30.0),
        ReactantServer.EndpointsConfig("127.0.0.1", 0), String[], Float64(poll), ReactantServer.DYNAMIC
    )
    return (; backend, pool, reg, sched, cfg)
end

_y(sched, name) = infer(sched, InferRequest(name, ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))[1].data

@testset "a bundle whose files change while it loads is not admitted, and loads again next poll (CPU)" begin
    mktempdir() do root
        env = _cpu_watch_env(root)
        ReactantServer.start!(env.sched)
        w = BundleWatcher(env.sched, env.backend, env.pool, env.cfg; interval = 0.1, on_demand = false)
        stage = mktempdir()
        try
            dir = _write_scale_bundle(root, "m")
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            v1 = get_model(env.reg, "m")
            @test _y(env.sched, "m") == Float32[2, 4, 6, 8]

            # A redeploy whose model.mlir is replaced again while the reload reads the bundle.
            plus_x = joinpath(stage, "model.mlir")
            write(plus_x, stablehlo_artifact(_W_MLIR_PLUS_X))
            write(joinpath(dir, "model.jl"), _change_once_jl("m", plus_x))
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :reload
            # Not admitted: the running version stays (the change was seen before its control op).
            @test get_model(env.reg, "m") === v1
            @test _y(env.sched, "m") == Float32[2, 4, 6, 8]
            # The change is already pending, at the files as they are now, so the next poll reloads.
            @test _poll_action(w) === :reload
            @test get_model(env.reg, "m") !== v1
            @test _y(env.sched, "m") == Float32[3, 6, 9, 12]     # the program that landed mid-load
            @test _poll_action(w) === nothing
            @test _poll_action(w) === nothing
        finally
            ReactantServer.shutdown!(env.sched)
        end
    end
end

@testset "load_model!: a version whose files changed during the load is never admitted (CPU)" begin
    mktempdir() do root
        env = _cpu_watch_env(root)
        ReactantServer.start!(env.sched)
        dir = _write_scale_bundle(root, "m")
        function load(unchanged)
            entry = ReactantServer.load_bundle_entry(dir)
            name = ReactantServer.load_model!(
                env.sched, env.backend, env.pool, entry;
                state = ReactantServer.UNPINNED, on_demand = false, unchanged = unchanged
            )
            return name, entry
        end
        try
            @test first(load(() -> true)) == "m"
            v1 = get_model(env.reg, "m")
            # Changed before the control op: the running version keeps serving.
            name, e = load(() -> false)
            @test name === nothing && e.executable === nothing && !isopen(e.weights.io)
            @test get_model(env.reg, "m") === v1
            @test _y(env.sched, "m") == Float32[2, 4, 6, 8]
            # Changed during the compile (the second check): compiled, then discarded unadmitted.
            calls = Ref(0)
            name, e = load(() -> (calls[] += 1) == 1)
            @test name === nothing && calls[] == 2
            @test e.executable !== nothing && e.sched === nothing && !isopen(e.weights.io)
            @test get_model(env.reg, "m") === nothing      # the old version made way for the compile
            @test_throws Exception _y(env.sched, "m")
            @test first(load(() -> true)) == "m"
            @test _y(env.sched, "m") == Float32[2, 4, 6, 8]
        finally
            ReactantServer.shutdown!(env.sched)
        end
    end
end

@testset "bundles changed or added while the worker starts are picked up by its first polls (CPU)" begin
    mktempdir() do root
        stage = mktempdir()
        plus_x = joinpath(stage, "model.mlir")
        write(plus_x, stablehlo_artifact(_W_MLIR_PLUS_X))
        _write_scale_bundle(stage, "z")
        # While the startup reads `a`, its model.mlir is replaced and bundle `z` is moved in.
        dir = _write_scale_bundle(root, "a")
        write(joinpath(dir, "model.jl"), _change_once_jl("a", plus_x; moves = [joinpath(stage, "z") => joinpath(root, "z")]))
        env = _cpu_watch_env(root; poll = 3600.0)
        _, reg, sched, w = ReactantServer._bring_up(env.cfg, env.backend)
        try
            # Drive the polls by hand: the watcher's own loop polled at most once (marks pending).
            ReactantServer.stop_watching!(w)
            wait(w.task)
            @test get_model(reg, "z") === nothing           # it arrived after the repository was read
            @test _y(sched, "a") == Float32[2, 4, 6, 8]     # the program read at startup
            _watch_once!(w)
            _watch_once!(w)
            @test get_model(reg, "z") !== nothing
            @test _y(sched, "z") == Float32[2, 4, 6, 8]
            @test _y(sched, "a") == Float32[3, 6, 9, 12]    # reloaded with the program that landed
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "a hot reload under a host weight budget serves the new version's weights (CPU)" begin
    # On-demand weights, unpinned, fair discipline (admission runs the model, so `load_model!` reads
    # the new version's weights ahead of its control op), with a finite host budget. The private
    # store still holds v1's copy under the model's name then; it must not reach v2's program.
    stores = Any[nothing]
    Sys.islinux() && isdir("/dev/shm") && push!(stores, "trl" * string(getpid()))
    for prefix in stores
        mktempdir() do root
            env = _cpu_watch_env(root)
            store, budget = prefix === nothing ? (ReactantServer.PrivateWeightStore(), 1 << 20) :
                (ReactantServer.SharedWeightStore(; budget = 1 << 20, prefix = prefix), nothing)
            env.sched.weight_cache = ReactantServer.WeightCache(
                env.backend, env.pool, env.reg, 1 << 20; store = store, host_budget = budget
            )
            ReactantServer.start!(env.sched)
            dir = _write_scale_bundle(root, "m")
            load() = ReactantServer.load_model!(
                env.sched, env.backend, env.pool, ReactantServer.load_bundle_entry(dir);
                state = ReactantServer.UNPINNED, on_demand = true, store = store
            )
            try
                @test ReactantServer.host_budgeted(env.sched.weight_cache)
                @test load() == "m"
                @test _y(env.sched, "m") == Float32[2, 4, 6, 8]
                prefix === nothing && @test haskey(env.sched.weight_cache.host_cached, "m")
                # Deploy new weights of the same shape: a new file renamed over the old one.
                staging = mktempdir(root)
                _write_scale_bundle(staging, "m"; w = Float32[5, 5, 5, 5])
                Base.Filesystem.rename(joinpath(staging, "m", "weights.safetensors"), joinpath(dir, "weights.safetensors"))
                @test load() == "m"
                @test _y(env.sched, "m") == Float32[5, 10, 15, 20]
                prefix === nothing && @test env.sched.weight_cache.host_cached["m"].entry === get_model(env.reg, "m")
            finally
                ReactantServer.shutdown!(env.sched)
                if prefix !== nothing
                    for f in readdir("/dev/shm"; join = true)
                        startswith(basename(f), prefix * "-") && rm(f; force = true)
                    end
                end
            end
        end
    end
end

@testset "model name comes from the bundle directory, not the manifest" begin
    mktempdir() do root
        # A manifest whose `name` disagrees with the directory: the directory wins.
        write_bundle(
            root, "served-name"; manifest_yaml = _w_manifest("something-else"),
            mlir_text = _W_MLIR, weights = Dict("w" => Float32[2, 2, 2, 2]),
            argument_order = ["w"]
        )
        entry = ReactantServer.load_bundle_entry(joinpath(root, "served-name"))
        @test entry.name == "served-name"
        @test entry.manifest.name == "served-name"   # rewritten so all consumers agree

        # A manifest with no `name` at all is fine; identity still comes from the directory.
        noname = replace(_w_manifest("ignored"), r"^name:.*\n"m => "")
        write_bundle(
            root, "anonymous"; manifest_yaml = noname, mlir_text = _W_MLIR,
            weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"]
        )
        @test ReactantServer.load_bundle_entry(joinpath(root, "anonymous")).name == "anonymous"
    end
end

@testset "a bundle named _unknown, the request-metrics label, is refused (CPU)" begin
    BundleError = ReactantServer.BundleError
    _reserved(e) = e isa BundleError && occursin("'_unknown' is reserved", sprint(showerror, e))
    mktempdir() do root
        # Refused before anything is read: an empty directory gives the reserved-name error, not a
        # missing manifest.
        empty = mkpath(joinpath(root, "_unknown"))
        err = try
            ReactantServer.load_bundle_entry(empty)
            nothing
        catch e
            e
        end
        @test _reserved(err)
        rm(empty; recursive = true)

        # A complete bundle under the name is refused too, at startup (load_bundles) as for any
        # invalid bundle; an allowlist that leaves it out loads the rest.
        _write_scale_bundle(root, "_unknown")
        _write_scale_bundle(root, "m")
        err = try
            ReactantServer.load_bundles([root])
            nothing
        catch e
            e
        end
        @test _reserved(err)
        reg = ReactantServer.load_bundles([root]; include = ["m"])
        @test haskey(reg.by_name, "m") && !haskey(reg.by_name, "_unknown")
        @test ReactantServer.request_model_label(reg, "_unknown") == ReactantServer.UNKNOWN_MODEL_LABEL
    end
    mktempdir() do root
        # In dynamic mode the watcher loads the other bundle and logs the refusal, registering
        # nothing under the reserved name. Both bundles appear after the watcher starts, as the
        # other watcher tests' do (one present at its creation counts as already loaded).
        env = _cpu_watch_env(root)
        ReactantServer.start!(env.sched)
        try
            w = BundleWatcher(env.sched, env.backend, env.pool, env.cfg; interval = 0.1, on_demand = false)
            _write_scale_bundle(root, "_unknown")
            _write_scale_bundle(root, "m")
            @test _poll_action(w) === nothing
            logger = TestLogger()
            with_logger(logger) do
                _watch_once!(w)
            end
            @test get_model(env.reg, "m") !== nothing
            @test get_model(env.reg, "_unknown") === nothing
            @test !haskey(env.reg.meta, "_unknown")
            @test any(logger.logs) do r
                r.message == "watcher: failed to apply model change; will retry next poll" &&
                    r.kwargs[:name] == "_unknown" && _reserved(first(r.kwargs[:exception]))
            end
            @test _y(env.sched, "m") == Float32[2, 4, 6, 8]
        finally
            ReactantServer.shutdown!(env.sched)
        end
    end
end

@testset "renamed bundle directory: an unload of the old name and a load of the new one (CPU)" begin
    mktempdir() do root
        backend = ReactantServer.ReactantBackend()
        runtime = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        pool = ReactantServer.resolve_client(backend, runtime)
        reg = ReactantServer.ModelRegistry()
        sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
        ReactantServer.start!(sched)
        cfg = ReactantServer.ServerConfig(
            [root], "", runtime,
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", 0)
        )
        w = BundleWatcher(sched, backend, pool, cfg; interval = 0.1, on_demand = false)
        x1 = [NamedTensor("x", Float32[1, 1, 1, 1])]
        # Every action one poll decided, by model name.
        function poll_actions()
            logger = TestLogger()
            with_logger(logger) do
                _watch_once!(w)
            end
            recs = filter(r -> r.message == "watcher: change detected", logger.logs)
            return Dict(String(r.kwargs[:name]) => r.kwargs[:action] for r in recs)
        end

        try
            _write_scale_bundle(root, "m-staging")
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            staging = get_model(reg, "m-staging")

            mv(joinpath(root, "m-staging"), joinpath(root, "m-production"))
            @test isempty(poll_actions())                       # both diffs debounce one poll
            @test poll_actions() == Dict("m-staging" => :unload, "m-production" => :load)
            @test get_model(reg, "m-staging") === nothing
            prod = get_model(reg, "m-production")
            @test prod !== nothing && prod !== staging          # loaded and compiled afresh
            @test infer(sched, InferRequest("m-production", ["y"], x1))[1].data == Float32[2, 2, 2, 2]
            @test_throws Exception infer(sched, InferRequest("m-staging", ["y"], x1))

            # A promotion chain in one poll: `-production` -> `-production-old` and a new `-staging`
            # -> `-production`. Each name serves the bundle now under it.
            _write_scale_bundle(root, "m-staging"; w = Float32[3, 3, 3, 3])
            write(joinpath(root, "m-staging", "manifest.yaml"), _w_manifest("m-staging") * "\n# v3\n")
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            mv(joinpath(root, "m-production"), joinpath(root, "m-production-old"))
            mv(joinpath(root, "m-staging"), joinpath(root, "m-production"))
            @test isempty(poll_actions())
            @test poll_actions() == Dict(
                "m-staging" => :unload, "m-production" => :reload, "m-production-old" => :load
            )
            @test get_model(reg, "m-staging") === nothing
            @test infer(sched, InferRequest("m-production", ["y"], x1))[1].data == Float32[3, 3, 3, 3]
            @test infer(sched, InferRequest("m-production-old", ["y"], x1))[1].data == Float32[2, 2, 2, 2]
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "watcher gated by model_control_mode (CPU)" begin
    mktempdir() do root
        _write_scale_bundle(root, "scale4")
        sched_cfg() = ReactantServer.SchedulerConfig(30.0, 64, 30.0)
        # CPU has no device arena, so the on-demand cache is off here; the watcher-gating assertion
        # below does not depend on it. Pass the residency mode through the 7-arg constructor.
        _runtime(residency) = ReactantServer.RuntimeConfig(
            ReactantServer.CPU_BACKEND, 0, 0.9, true,
            true, residency, false
        )
        # explicit ⇒ externally-managed residency, mirroring build_config's derivation.
        _cfg(mode) = ReactantServer.ServerConfig(
            [root], "",
            _runtime(mode == ReactantServer.EXPLICIT ? ReactantServer.EXTERNALLY_MANAGED : ReactantServer.SELF_MANAGED),
            sched_cfg(), ReactantServer.EndpointsConfig("127.0.0.1", 0), String[], 1.0, mode
        )

        # Only dynamic mode starts the filesystem watcher.
        for mode in (ReactantServer.STATIC, ReactantServer.EXPLICIT)
            _, _, sched_n, watcher_n = ReactantServer._bring_up(_cfg(mode), ReactantServer.ReactantBackend())
            @test watcher_n === nothing
            ReactantServer.shutdown!(sched_n)
        end

        _, _, sched_d, watcher_d = ReactantServer._bring_up(
            _cfg(ReactantServer.DYNAMIC),
            ReactantServer.ReactantBackend()
        )
        @test watcher_d !== nothing
        ReactantServer.stop_watching!(watcher_d)
        ReactantServer.shutdown!(sched_d)
    end
end
