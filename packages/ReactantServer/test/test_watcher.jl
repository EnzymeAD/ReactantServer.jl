# Dynamic model-directory watching: bundle signatures, repository scanning, and the
# load/reload/unload + debounce behavior of a poll round. The poll round is driven manually
# (_watch_once!) against a real CPU backend so the swaps are deterministic with no timing.

using ReactantServer: bundle_signature, scan_repository, BundleWatcher, _watch_once!,
    get_model, infer, InferRequest, NamedTensor
using Logging: with_logger
using Test: TestLogger

# Run one poll under a captured logger and return the watcher's decided action
# (:load/:reload/:unload/:rename), or `nothing` when the poll applied no change (e.g. the
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

_write_scale_bundle(root, name; w=Float32[2, 2, 2, 2]) =
    write_bundle(root, name; manifest_yaml=_w_manifest(name), mlir_text=_W_MLIR,
                 weights=Dict("w" => w), argument_order=["w"])

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
        cfg = ReactantServer.ServerConfig([root], "", runtime,
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", 0))
        # Construct the watcher while the repo is empty, so the startup-seeded `seen` is empty and
        # the bundle added below registers as a new model.
        w = BundleWatcher(sched, backend, pool, cfg; interval=0.1, on_demand=false)

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
                _write_scale_bundle(root, "scale4"; w=Float32[4, 4, 4, 4])
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
                rm(joinpath(root, "scale4"); recursive=true)
                @test _poll_action(w) === nothing     # pending removal
                @test get_model(reg, "scale4") !== nothing
                @test _poll_action(w) === :unload     # stable -> unload
                @test get_model(reg, "scale4") === nothing
                # Inference against the unloaded model now errors.
                @test_throws Exception infer(sched,
                    InferRequest("scale4", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))
            end
        finally
            ReactantServer.shutdown!(sched)
        end
    end
end

@testset "model name comes from the bundle directory, not the manifest" begin
    mktempdir() do root
        # A manifest whose `name` disagrees with the directory: the directory wins.
        write_bundle(root, "served-name"; manifest_yaml=_w_manifest("something-else"),
                     mlir_text=_W_MLIR, weights=Dict("w" => Float32[2, 2, 2, 2]),
                     argument_order=["w"])
        entry = ReactantServer.load_bundle_entry(joinpath(root, "served-name"))
        @test entry.name == "served-name"
        @test entry.manifest.name == "served-name"   # rewritten so all consumers agree

        # A manifest with no `name` at all is fine; identity still comes from the directory.
        noname = replace(_w_manifest("ignored"), r"^name:.*\n"m => "")
        write_bundle(root, "anonymous"; manifest_yaml=noname, mlir_text=_W_MLIR,
                     weights=Dict("w" => Float32[2, 2, 2, 2]), argument_order=["w"])
        @test ReactantServer.load_bundle_entry(joinpath(root, "anonymous")).name == "anonymous"
    end
end

@testset "renamed bundle directory: renamed in place, no recompile (CPU)" begin
    mktempdir() do root
        backend = ReactantServer.ReactantBackend()
        runtime = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
        pool = ReactantServer.resolve_client(backend, runtime)
        reg = ReactantServer.ModelRegistry()
        sched = ReactantServer.Scheduler(reg, backend, pool, ReactantServer.SchedulerConfig(30.0, 64, 30.0))
        ReactantServer.start!(sched)
        cfg = ReactantServer.ServerConfig([root], "", runtime,
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", 0))
        w = BundleWatcher(sched, backend, pool, cfg; interval=0.1, on_demand=false)

        try
            _write_scale_bundle(root, "m-staging")
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            entry = get_model(reg, "m-staging")
            lm = entry.executable
            @test lm !== nothing

            # The registry promotion: one atomic directory rename, contents untouched.
            mv(joinpath(root, "m-staging"), joinpath(root, "m-production"))
            @test _poll_action(w) === nothing        # add + remove both debounce one poll
            @test _poll_action(w) === :rename        # paired by (device, inode) + unchanged signature
            @test get_model(reg, "m-staging") === nothing
            e2 = get_model(reg, "m-production")
            @test e2 === entry                       # the same entry object: nothing was reloaded
            @test e2.executable === lm               # the same compiled runtime: nothing recompiled
            @test e2.name == "m-production" && e2.sched.name == "m-production"

            # It serves under the new name and rejects the old one.
            out = infer(sched, InferRequest("m-production", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))
            @test out[1].data == Float32[2, 4, 6, 8]
            @test_throws Exception infer(sched,
                InferRequest("m-staging", ["y"], [NamedTensor("x", Float32[1, 2, 3, 4])]))

            # The registry promotion chain: `-production` -> `-production-old` AND a new
            # `-staging` -> `-production` land between two polls. The second rename's target is
            # occupied until the first applies; both must resolve as renames, nothing recompiles.
            _write_scale_bundle(root, "m-staging"; w=Float32[3, 3, 3, 3])
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            prod = get_model(reg, "m-production"); prod_lm = prod.executable
            stag = get_model(reg, "m-staging");    stag_lm = stag.executable
            mv(joinpath(root, "m-production"), joinpath(root, "m-production-old"))
            mv(joinpath(root, "m-staging"), joinpath(root, "m-production"))
            @test _poll_action(w) === nothing                  # all three diffs debounce one poll
            logger = TestLogger()
            with_logger(logger) do
                _watch_once!(w)
            end
            recs = filter(r -> r.message == "watcher: change detected", logger.logs)
            @test length(recs) == 2
            @test all(r -> r.kwargs[:action] === :rename, recs)
            @test get_model(reg, "m-staging") === nothing
            @test get_model(reg, "m-production") === stag      # promoted in place
            @test get_model(reg, "m-production").executable === stag_lm
            @test get_model(reg, "m-production-old") === prod  # demoted in place
            @test get_model(reg, "m-production-old").executable === prod_lm
            out = infer(sched, InferRequest("m-production", ["y"], [NamedTensor("x", Float32[1, 1, 1, 1])]))
            @test out[1].data == Float32[3, 3, 3, 3]           # the promoted staging weights
            out = infer(sched, InferRequest("m-production-old", ["y"], [NamedTensor("x", Float32[1, 1, 1, 1])]))
            @test out[1].data == Float32[2, 2, 2, 2]           # the demoted production weights

            # rm + rename: DELETE the production model, then move a new staging model onto its
            # name. The name never vacates in the diff (it reads as a content change), so the
            # occupant must be unloaded first and the rename still applies: no compile.
            _write_scale_bundle(root, "m-staging"; w=Float32[5, 5, 5, 5])
            @test _poll_action(w) === nothing
            @test _poll_action(w) === :load
            stag2 = get_model(reg, "m-staging"); stag2_lm = stag2.executable
            rm(joinpath(root, "m-production"); recursive=true)
            mv(joinpath(root, "m-staging"), joinpath(root, "m-production"))
            @test _poll_action(w) === nothing                  # both diffs debounce one poll
            logger2 = TestLogger()
            with_logger(logger2) do
                _watch_once!(w)
            end
            acts = [r.kwargs[:action] for r in filter(r -> r.message == "watcher: change detected", logger2.logs)]
            @test acts == [:unload, :rename]                   # evict the occupant, rename in place
            @test get_model(reg, "m-staging") === nothing
            @test get_model(reg, "m-production") === stag2     # the staging entry, not a recompile
            @test get_model(reg, "m-production").executable === stag2_lm
            out = infer(sched, InferRequest("m-production", ["y"], [NamedTensor("x", Float32[1, 1, 1, 1])]))
            @test out[1].data == Float32[5, 5, 5, 5]
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
        _runtime(residency) = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true,
            true, residency, false)
        # explicit ⇒ externally-managed residency, mirroring build_config's derivation.
        _cfg(mode) = ReactantServer.ServerConfig([root], "",
            _runtime(mode == ReactantServer.EXPLICIT ? ReactantServer.EXTERNALLY_MANAGED : ReactantServer.SELF_MANAGED),
            sched_cfg(), ReactantServer.EndpointsConfig("127.0.0.1", 0), String[], 1.0, mode)

        # Only dynamic mode starts the filesystem watcher.
        for mode in (ReactantServer.STATIC, ReactantServer.EXPLICIT)
            _, _, sched_n, watcher_n = ReactantServer._bring_up(_cfg(mode), ReactantServer.ReactantBackend())
            @test watcher_n === nothing
            ReactantServer.shutdown!(sched_n)
        end

        _, _, sched_d, watcher_d = ReactantServer._bring_up(_cfg(ReactantServer.DYNAMIC),
            ReactantServer.ReactantBackend())
        @test watcher_d !== nothing
        ReactantServer.stop_watching!(watcher_d)
        ReactantServer.shutdown!(sched_d)
    end
end
