# The KaimonGate extension: argument handling, the process-global client guard, the tool schemas
# KaimonGate reflects out of the handlers, and one real start / inspect / stop cycle on CPU.
#
# `using KaimonGate` (in runtests.jl) is what loads the extension; no gate is running here, so
# registration is a no-op and only the handlers are exercised, which is what an agent calls anyway.

const KGE = Base.get_extension(ReactantServer, :ReactantServerKaimonGateExt)

@testset "KaimonGate extension" begin
    @test KGE !== nothing

    @testset "a serve guard can refuse a start before anything is allocated" begin
        ReactantServer.clear_serve_guards!()
        asked = Any[]
        ReactantServer.register_serve_guard!(cfg -> push!(asked, cfg))
        ReactantServer.register_serve_guard!(cfg -> error("no card for you"))
        err = try
            ReactantServer._check_serve_guards(:cfg); nothing
        catch e
            e
        end
        @test err isa ErrorException && err.msg == "no card for you"
        @test asked == Any[:cfg]
        ReactantServer.clear_serve_guards!()
        @test ReactantServer._check_serve_guards(:cfg) === nothing
    end

    @testset "heartbeat registration" begin
        events = Any[]
        ReactantServer.clear_heartbeats!()
        ReactantServer.register_heartbeat!((ev, d) -> push!(events, (ev, d)))
        ReactantServer.register_heartbeat!((ev, d) -> error("a callback that throws must not take the request"))
        ReactantServer._beat(:up, nothing)
        ReactantServer._beat(:request, "m")
        ReactantServer._beat(:down, nothing)
        @test events == Any[(:up, nothing), (:request, "m"), (:down, nothing)]
        ReactantServer.clear_heartbeats!()
        ReactantServer._beat(:request, "m")
        @test length(events) == 3
    end

    @testset "argument handling" begin
        @test KGE._model_names(nothing) == String[]
        # A JSON array is the declared shape, but an agent that packs names into one element means
        # the same thing.
        @test KGE._model_names(["a", "b"]) == ["a", "b"]
        @test KGE._model_names(["a, b", " c "]) == ["a", "b", "c"]
        @test KGE._model_names([""]) == String[]

        @test KGE._normalize_accelerator("gpu") == "cuda"
        @test KGE._normalize_accelerator("CUDA") == "cuda"
        @test KGE._normalize_accelerator(" cpu ") == "cpu"
        @test_throws ErrorException KGE._normalize_accelerator("tpu")

        # A bad argument is refused by the call itself, not a poll later.
        @test_throws ErrorException KGE.rserver_start("/nonexistent"; accelerator = "rocm")
        @test_throws ErrorException KGE.rserver_start("/nonexistent"; device = -1)
        @test_throws ErrorException KGE.rserver_start(
            "/nonexistent"; accelerator = "cpu", device = 1
        )
    end

    @testset "config building" begin
        mktempdir() do root
            base = (
                model_dir = root, models = String[], accelerator = "cuda", device = 2,
                host = "0.0.0.0", port = 8080, metrics_port = 9100, cache_dir = nothing,
                mem_fraction = nothing, weight_cache_fraction = nothing, numerics = nothing,
                poll_seconds = nothing,
            )
            cfg = KGE._build_config(; base...)
            @test cfg.model_dirs == [abspath(root)]
            @test cfg.runtime.backend == ReactantServer.CUDA_BACKEND
            @test cfg.runtime.device_ordinal == 2
            @test cfg.endpoints.host == "0.0.0.0"
            @test cfg.endpoints.port == 8080
            @test cfg.endpoints.metrics_port == 9100
            @test isempty(cfg.models_include)
            # No watcher unless a poll interval is asked for, and the YAML path's defaults are
            # inherited rather than restated here.
            @test cfg.model_control_mode == ReactantServer.STATIC
            @test cfg.runtime.mem_fraction == 0.9
            @test cfg.runtime.preallocate
            @test cfg.runtime.weight_cache_fraction == 1.0

            dyn = KGE._build_config(; base..., poll_seconds = 5.0, models = ["a", "b"])
            @test dyn.model_control_mode == ReactantServer.DYNAMIC
            @test dyn.model_poll_seconds == 5.0
            @test dyn.models_include == ["a", "b"]

            tuned = KGE._build_config(;
                base..., accelerator = "cpu", device = 0, mem_fraction = 0.5,
                weight_cache_fraction = 0.0, numerics = "f32"
            )
            @test tuned.runtime.backend == ReactantServer.CPU_BACKEND
            @test tuned.runtime.mem_fraction == 0.5
            @test tuned.runtime.weight_cache_fraction == 0.0
            @test tuned.runtime.numerics == ReactantServer.NUMERICS_F32

            # Config validation is the same validation a node file gets.
            @test_throws ReactantServer.ConfigError KGE._build_config(;
                base..., port = 8080, metrics_port = 8080
            )
            @test_throws ReactantServer.ConfigError KGE._build_config(;
                base..., model_dir = joinpath(root, "missing")
            )
            @test_throws ErrorException KGE._build_config(; base..., poll_seconds = -1.0)
        end
    end

    @testset "process-global client guard" begin
        mktempdir() do root
            mk(frac) = KGE._build_config(;
                model_dir = root, models = String[], accelerator = "cuda", device = 0,
                host = "127.0.0.1", port = 8080, metrics_port = 0, cache_dir = nothing,
                mem_fraction = frac, weight_cache_fraction = nothing, numerics = nothing,
                poll_seconds = nothing
            )
            saved = KGE.CLIENT_COMMIT[]
            try
                KGE.CLIENT_COMMIT[] = nothing
                # The first CUDA start records what the session's XLA client is built with.
                KGE._check_client_commit_locked(mk(0.8))
                @test KGE.CLIENT_COMMIT[].mem_fraction == 0.8
                # A second one at the same settings is fine; one that disagrees is refused rather
                # than silently inheriting the first arena.
                KGE._check_client_commit_locked(mk(0.8))
                @test_throws ErrorException KGE._check_client_commit_locked(mk(0.5))
                # The CPU backend creates no CUDA client, so it never commits.
                KGE.CLIENT_COMMIT[] = nothing
                cpu = KGE._build_config(;
                    model_dir = root, models = String[], accelerator = "cpu", device = 0,
                    host = "127.0.0.1", port = 8080, metrics_port = 0, cache_dir = nothing,
                    mem_fraction = 0.5, weight_cache_fraction = nothing, numerics = nothing,
                    poll_seconds = nothing
                )
                KGE._check_client_commit_locked(cpu)
                @test KGE.CLIENT_COMMIT[] === nothing
            finally
                KGE.CLIENT_COMMIT[] = saved
            end
        end
    end

    @testset "tool schemas" begin
        tools = KGE._build_tools()
        @test [t.name for t in tools] ==
            ["rserver_start", "rserver_status", "rserver_models", "rserver_stop"]
        metas = Dict(t.name => KaimonGate._reflect_tool(t) for t in tools)
        for (name, meta) in metas
            # The docstring is the tool description an agent reads; an undocumented handler would
            # ship a nameless tool.
            @test !isempty(meta["description"])
        end
        start = metas["rserver_start"]
        args = Dict(a["name"] => a for a in start["arguments"])
        @test args["model_dir"]["required"]
        @test !args["models"]["required"]
        @test args["models"]["type_meta"]["kind"] == "array"
        @test args["device"]["type_meta"]["kind"] == "integer"
        @test args["accelerator"]["type_meta"]["kind"] == "string"
        for kw in ("port", "metrics_port", "host", "cache_dir", "mem_fraction", "numerics", "poll_seconds")
            @test haskey(args, kw)
            @test !args[kw]["required"]
        end
    end

    @testset "no server started" begin
        @test occursin("no servers", KGE.rserver_status())
        @test_throws ErrorException KGE.rserver_stop()
        @test_throws ErrorException KGE.rserver_models()
        @test_throws ErrorException KGE.rserver_status(id = "nosuchid")
    end

    # One real cycle on CPU: start through the tool, poll status the way an agent does, read the
    # model listing, refuse a colliding port, stop.
    @testset "start / inspect / stop (CPU)" begin
        mktempdir() do root
            manifest = """
            format_version: "2.0"
            name: scale4
            executable_inputs:
              - name: x
                dtype: f32
                shape: c
                dims:
                  c: 4
            executable_outputs:
              - name: y
                dtype: f32
                shape: c
                dims:
                  c: 4
            batching:
              compiled_batch_sizes: [1]
            """
            mlir = """
            module {
              func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.multiply %x, %w : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
            """
            write_bundle(
                root, "scale4";
                manifest_yaml = manifest, mlir_text = mlir,
                weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"]
            )

            port = grpc_free_port()
            msg = KGE.rserver_start(root; accelerator = "cpu", port = port)
            id = String(match(r"starting server (\w+)", msg).captures[1])
            @test occursin("poll", msg)

            try
                # The call returns before the bundle is compiled, which is the whole reason the
                # agent polls.
                deadline = time() + 180
                while time() < deadline
                    occursin("status=running", KGE.rserver_status(id = id)) && break
                    occursin("status=failed", KGE.rserver_status(id = id)) && break
                    sleep(0.5)
                end
                status = KGE.rserver_status(id = id)
                @test occursin("status=running", status)
                @test occursin("accelerator: cpu", status)
                @test occursin("port=$(port)", replace(status, " " => "")) ||
                    occursin(string(port), status)
                @test occursin("1/1 compiled", status)
                @test occursin("no directory watcher", status)

                models = KGE.rserver_models(id = id)
                @test occursin("scale4", models)
                @test occursin("ready", models)
                @test occursin("x: f32[4]", models)
                @test occursin("y: f32[4]", models)
                @test occursin("no model `nope`", KGE.rserver_models(id = id, name = "nope"))

                # The single live server is the obvious target, so `id` is optional.
                @test occursin("scale4", KGE.rserver_models())

                # A port this session already serves on is refused before another compile.
                @test_throws ErrorException KGE.rserver_start(
                    root; accelerator = "cpu", port = port
                )
            finally
                stopped = KGE.rserver_stop(id = id)
                @test occursin("stopped server", stopped)
            end

            @test occursin("status=stopped", KGE.rserver_status(id = id))
            @test occursin("already stopped", KGE.rserver_stop(id = id))
            # A stopped server is no longer the implicit target.
            @test_throws ErrorException KGE.rserver_models()
        end
    end
end
