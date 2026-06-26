# The node file is the only supported worker config format. These tests drive the per-worker
# resolution path used by serve(): a single-worker node wraps a `global:` block (the old
# per-worker config shape) plus a shared model_repo, and resolves to a ServerConfig through
# the same load_node -> node_server_config -> build_config -> validate_config pipeline.

# Write a single-worker cluster file whose `global:` block is `global_body`, then resolve the
# sole worker's config exactly as serve() would. Returns (cfg, applied, worker_name).
function load_single_worker(dir, global_body::AbstractString;
                            base_port::Int=8080,
                            model_repo::AbstractString=joinpath(dir, "models"))
    isdir(model_repo) || mkpath(model_repo)
    path = joinpath(dir, "cluster.yaml")
    open(path, "w") do io
        println(io, "model_repo: ", model_repo)
        println(io, "base_port: ", base_port)
        if !isempty(strip(global_body))
            println(io, "global:")
            for line in split(global_body, '\n')
                println(io, "  ", line)
            end
        end
        println(io, "workers:")
        println(io, "  - { name: solo, gpu: 0 }")
    end
    cluster = ReactantServer.load_node(path)
    return ReactantServer.node_server_config(cluster, nothing)
end

@testset "config load + env overrides" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)
        cfg, applied, wname = load_single_worker(dir, """
        runtime:
          backend: cpu
          mem_fraction: 0.8
        scheduler:
          ema_halflife_seconds: 60.0
          max_queue_depth: 256
        endpoints:
          host: 0.0.0.0
        """; base_port=9000, model_repo=modeldir)

        @test wname == "solo"
        @test isempty(applied)
        @test cfg.model_dirs == [modeldir]
        @test cfg.runtime.backend == ReactantServer.CPU_BACKEND
        @test cfg.runtime.mem_fraction == 0.8
        @test cfg.scheduler.ema_halflife_seconds == 60.0
        @test cfg.scheduler.max_queue_depth == 256
        @test cfg.endpoints.port == 9000          # derived from base_port
        @test isempty(cfg.models_include)          # single worker, no models map => load all
        @test ReactantServer.validate_config(cfg) === cfg

        # environment overrides win over the file
        withenv("INFERENCE_SERVER_SCHEDULER_EMA_HALFLIFE_SECONDS" => "30",
                "INFERENCE_SERVER_ENDPOINTS_PORT" => "12345",
                "INFERENCE_SERVER_RUNTIME_BACKEND" => "cuda") do
            cfg2, applied2, _ = load_single_worker(dir, """
            runtime:
              backend: cpu
            endpoints:
              host: 0.0.0.0
            """; base_port=9000, model_repo=modeldir)
            @test cfg2.scheduler.ema_halflife_seconds == 30.0
            @test cfg2.endpoints.port == 12345
            @test cfg2.runtime.backend == ReactantServer.CUDA_BACKEND
            @test length(applied2) == 3
        end

        # invalid: missing model_repo
        badpath = joinpath(dir, "bad.yaml")
        write(badpath, "base_port: 8080\nworkers:\n  - { name: solo, gpu: 0 }\n")
        @test_throws ReactantServer.ConfigError ReactantServer.load_node(badpath)

        # invalid: port out of range fails validation
        cfgbp, _, _ = load_single_worker(dir, ""; base_port=99999, model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cfgbp)

        # the removed batch_policy block fails loudly with a migration message
        @test_throws ReactantServer.ConfigError load_single_worker(dir, """
        batch_policy:
          max_batch_size: 8
        """; model_repo=modeldir)

        # same for a stale batch_policy environment override
        withenv("INFERENCE_SERVER_BATCH_POLICY_MAX_BATCH_SIZE" => "8") do
            @test_throws ReactantServer.ConfigError load_single_worker(dir, ""; model_repo=modeldir)
        end
    end
end

@testset "scheduler config: defaults, per-model overrides, validation" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)
        cfg, _, _ = load_single_worker(dir, """
        scheduler:
          ema_halflife_seconds: 45.0
          recency_penalty_cap: 0.3
          coalescing_discount: 0.20
          cost_ema_alpha: 0.15
          discipline: fifo
          models:
            resnet50:
              weight: 2.0
              max_batch_size: 8
            yolo:
              weight: 1.5
        """; model_repo=modeldir)
        @test ReactantServer.validate_config(cfg) === cfg
        @test cfg.scheduler.recency_penalty_cap == 0.3
        @test cfg.scheduler.coalescing_discount == 0.20
        @test cfg.scheduler.cost_ema_alpha == 0.15
        @test cfg.scheduler.discipline == ReactantServer.FIFO
        @test cfg.scheduler.models["resnet50"].weight == 2.0
        @test cfg.scheduler.models["resnet50"].max_batch_size == 8
        @test cfg.scheduler.models["yolo"].weight == 1.5
        @test cfg.scheduler.models["yolo"].max_batch_size === nothing   # default: uncapped
        # unlisted models are absent; the scheduler applies the weight 1.0 / unpinned default
        @test !haskey(cfg.scheduler.models, "absent")

        # defaults when the knobs are omitted
        cfgb, _, _ = load_single_worker(dir, ""; model_repo=modeldir)
        @test cfgb.scheduler.recency_penalty_cap == 0.25
        @test cfgb.scheduler.coalescing_discount == 0.10
        @test cfgb.scheduler.cost_ema_alpha == 0.2
        @test cfgb.scheduler.discipline == ReactantServer.FAIR
        @test isempty(cfgb.scheduler.models)

        # validation rejects out-of-range knobs and bad per-model values
        for (key, val) in (("recency_penalty_cap", "1.5"), ("coalescing_discount", "1.0"),
                           ("cost_ema_alpha", "0"))
            c, _, _ = load_single_worker(dir, "scheduler:\n  $key: $val"; model_repo=modeldir)
            @test_throws ReactantServer.ConfigError ReactantServer.validate_config(c)
        end
        cw, _, _ = load_single_worker(dir,
            "scheduler:\n  models:\n    m:\n      weight: -1.0"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cw)
        cm, _, _ = load_single_worker(dir,
            "scheduler:\n  models:\n    m:\n      max_batch_size: 0"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cm)
    end
end

@testset "residency config: budget, residency states, modes, env override, validation" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)
        cfg, _, _ = load_single_worker(dir, """
        model_control_mode: explicit
        runtime:
          weight_cache_fraction: 0.5
          shared_host_weights: true
        scheduler:
          models:
            resnet50:
              pin_to_gpu: true        # back-compat alias for residency: device
            spine:
              residency: system
            yolo:
              weight: 1.5
        """; model_repo=modeldir)
        @test ReactantServer.validate_config(cfg) === cfg
        @test cfg.runtime.weight_cache_fraction == 0.5
        # residency_mode is derived from the control mode: explicit ⇒ externally-managed.
        @test cfg.model_control_mode == ReactantServer.EXPLICIT
        @test cfg.runtime.residency_mode == ReactantServer.EXTERNALLY_MANAGED
        @test cfg.runtime.shared_host_weights == true
        @test cfg.scheduler.models["resnet50"].residency == ReactantServer.PINNED_DEVICE
        @test cfg.scheduler.models["spine"].residency == ReactantServer.PINNED_SYSTEM
        @test cfg.scheduler.models["yolo"].residency === nothing   # unspecified; resolved at startup

        # defaults: self-sizing cache (fraction 1.0, wiggle 0.1), dynamic mode, self-managed, private host weights
        cfgb, _, _ = load_single_worker(dir, ""; model_repo=modeldir)
        @test cfgb.runtime.weight_cache_fraction == 1.0
        @test cfgb.runtime.weight_cache_wiggle_fraction == 0.1
        @test cfgb.model_control_mode == ReactantServer.DYNAMIC
        @test cfgb.model_poll_seconds == 15.0
        @test cfgb.runtime.residency_mode == ReactantServer.SELF_MANAGED
        @test cfgb.runtime.shared_host_weights == false
        @test cfgb.runtime.shared_host_weights_mode == 0o666
    end
end

@testset "weight cache fraction + wiggle: parse, env, validation, disable" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)

        cfg, _, _ = load_single_worker(dir, """
        runtime:
          weight_cache_fraction: 0.6
          weight_cache_wiggle_fraction: 0.05
        """; model_repo=modeldir)
        @test ReactantServer.validate_config(cfg) === cfg
        @test cfg.runtime.weight_cache_fraction == 0.6
        @test cfg.runtime.weight_cache_wiggle_fraction == 0.05

        # 0 disables the on-demand cache (all weights resident).
        cfg0, _, _ = load_single_worker(dir, "runtime:\n  weight_cache_fraction: 0.0"; model_repo=modeldir)
        @test cfg0.runtime.weight_cache_fraction == 0.0

        withenv("INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_FRACTION" => "0.75",
                "INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_WIGGLE_FRACTION" => "0.2") do
            cfge, applied, _ = load_single_worker(dir, ""; model_repo=modeldir)
            @test cfge.runtime.weight_cache_fraction == 0.75
            @test cfge.runtime.weight_cache_wiggle_fraction == 0.2
            @test ("INFERENCE_SERVER_RUNTIME_WEIGHT_CACHE_FRACTION", "0.75") in applied
        end

        # validation: fraction in [0,1], wiggle in [0,1)
        cf, _, _ = load_single_worker(dir, "runtime:\n  weight_cache_fraction: 1.5"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cf)
        cw, _, _ = load_single_worker(dir, "runtime:\n  weight_cache_wiggle_fraction: 1.0"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cw)
    end
end

@testset "grpc config: defaults, node-global parse, env override, validation" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)

        # Default: 512 MiB in each direction when no grpc block is given.
        cfg0, _, _ = load_single_worker(dir, "runtime:\n  backend: cpu"; model_repo=modeldir)
        @test cfg0.grpc.max_recv_msg_bytes == 512 * 1024 * 1024
        @test cfg0.grpc.max_send_msg_bytes == 512 * 1024 * 1024
        @test ReactantServer.validate_config(cfg0) === cfg0

        # A node-level global.grpc block resolves into the worker's ServerConfig.grpc.
        cfg, _, _ = load_single_worker(dir, """
        grpc:
          max_recv_msg_bytes: 1048576
          max_send_msg_bytes: 2097152
        """; model_repo=modeldir)
        @test cfg.grpc.max_recv_msg_bytes == 1048576
        @test cfg.grpc.max_send_msg_bytes == 2097152

        # Environment overrides win over the file.
        withenv("INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES" => "4096",
                "INFERENCE_SERVER_GRPC_MAX_SEND_MSG_BYTES" => "8192") do
            cfge, applied, _ = load_single_worker(dir, "grpc:\n  max_recv_msg_bytes: 1048576"; model_repo=modeldir)
            @test cfge.grpc.max_recv_msg_bytes == 4096
            @test cfge.grpc.max_send_msg_bytes == 8192
            @test ("INFERENCE_SERVER_GRPC_MAX_RECV_MSG_BYTES", "4096") in applied
        end

        # Non-positive is rejected.
        cbad, _, _ = load_single_worker(dir, "grpc:\n  max_recv_msg_bytes: 0"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cbad)
    end
end

@testset "model_control_mode: parse, default, residency, validation, migration" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)

        # static and dynamic are self-managed; static needs no poll interval.
        cstatic, _, _ = load_single_worker(dir, "model_control_mode: static"; model_repo=modeldir)
        @test cstatic.model_control_mode == ReactantServer.STATIC
        @test cstatic.runtime.residency_mode == ReactantServer.SELF_MANAGED
        @test ReactantServer.validate_config(cstatic) === cstatic

        cdyn, _, _ = load_single_worker(dir, "model_control_mode: dynamic\nmodel_poll_seconds: 5.0";
                                        model_repo=modeldir)
        @test cdyn.model_control_mode == ReactantServer.DYNAMIC
        @test cdyn.model_poll_seconds == 5.0
        @test cdyn.runtime.residency_mode == ReactantServer.SELF_MANAGED

        # an invalid mode string is rejected
        @test_throws ReactantServer.ConfigError load_single_worker(dir,
            "model_control_mode: bogus"; model_repo=modeldir)

        # dynamic with a non-positive poll interval fails validation
        cbadpoll, _, _ = load_single_worker(dir, "model_control_mode: dynamic\nmodel_poll_seconds: 0";
                                            model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cbadpoll)

        # the retired runtime.residency_mode key throws a migration error (YAML and env)
        @test_throws ReactantServer.ConfigError load_single_worker(dir,
            "runtime:\n  residency_mode: externally_managed"; model_repo=modeldir)
        withenv("INFERENCE_SERVER_RUNTIME_RESIDENCY_MODE" => "externally_managed") do
            @test_throws ReactantServer.ConfigError load_single_worker(dir, ""; model_repo=modeldir)
        end

        # env override selects the mode
        withenv("INFERENCE_SERVER_MODEL_CONTROL_MODE" => "explicit") do
            cenv, applied, _ = load_single_worker(dir, ""; model_repo=modeldir)
            @test cenv.model_control_mode == ReactantServer.EXPLICIT
            @test cenv.runtime.residency_mode == ReactantServer.EXTERNALLY_MANAGED
            @test ("INFERENCE_SERVER_MODEL_CONTROL_MODE", "explicit") in applied
        end
    end
end

@testset "node config: multi-worker resolution, model assignment, validation" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)
        path = joinpath(dir, "cluster.yaml")
        write(path, """
        model_repo: $modeldir
        base_port: 8080
        global:
          runtime: { backend: cpu, mem_fraction: 0.9 }
          endpoints: { host: 0.0.0.0 }
        workers:
          - { name: worker0, gpu: 0, runtime: { mem_fraction: 0.5 } }
          - { name: worker1, gpu: 1 }
        models:
          resnet50: [worker0, worker1]
          vsq_coral: [worker0]
          spine: [worker1]
        """)
        cluster = ReactantServer.load_node(path)

        # worker0: port = base_port + 0, gpu 0, per-worker mem_fraction override, its slice.
        cfg0, _, n0 = ReactantServer.node_server_config(cluster, "worker0")
        @test n0 == "worker0"
        @test cfg0.endpoints.port == 8080
        @test cfg0.runtime.device_ordinal == 0
        @test cfg0.runtime.mem_fraction == 0.5
        # The map no longer restricts loading: every worker loads all bundles (empty allowlist).
        @test isempty(cfg0.models_include)
        # Its assigned models are pinned to device on this worker; unassigned ones are not listed.
        @test cfg0.scheduler.models["resnet50"].residency == ReactantServer.PINNED_DEVICE
        @test cfg0.scheduler.models["vsq_coral"].residency == ReactantServer.PINNED_DEVICE
        @test !haskey(cfg0.scheduler.models, "spine")
        @test ReactantServer.validate_config(cfg0) === cfg0

        # worker1: port = base_port + 1, gpu 1, inherits global mem_fraction; loads all, pins its slice.
        cfg1, _, _ = ReactantServer.node_server_config(cluster, "worker1")
        @test cfg1.endpoints.port == 8081
        @test cfg1.runtime.device_ordinal == 1
        @test cfg1.runtime.mem_fraction == 0.9
        @test isempty(cfg1.models_include)
        @test cfg1.scheduler.models["resnet50"].residency == ReactantServer.PINNED_DEVICE   # pinned on both
        @test cfg1.scheduler.models["spine"].residency == ReactantServer.PINNED_DEVICE
        @test !haskey(cfg1.scheduler.models, "vsq_coral")

        # ambiguous: multi-worker cluster requires naming which worker to serve
        @test_throws ReactantServer.ConfigError ReactantServer.node_server_config(cluster, nothing)
        # unknown worker name
        @test_throws ReactantServer.ConfigError ReactantServer.node_server_config(cluster, "ghost")

        # `gpu` is optional and defaults to device ordinal 0 (the single-visible-GPU model;
        # the physical GPU is chosen out of band, e.g. via CUDA_VISIBLE_DEVICES).
        nogpu = joinpath(dir, "nogpu.yaml")
        write(nogpu, """
        model_repo: $modeldir
        base_port: 8080
        workers:
          - { name: solo }
        """)
        cfgn, _, _ = ReactantServer.node_server_config(ReactantServer.load_node(nogpu), nothing)
        @test cfgn.runtime.device_ordinal == 0

        # A multi-worker node without a models map is now valid: every worker loads all bundles
        # (empty allowlist) and pins nothing to device.
        nomodels = joinpath(dir, "nomodels.yaml")
        write(nomodels, """
        model_repo: $modeldir
        base_port: 8080
        workers:
          - { name: a, gpu: 0 }
          - { name: b, gpu: 1 }
        """)
        nm = ReactantServer.load_node(nomodels)
        cfga, _, _ = ReactantServer.node_server_config(nm, "a")
        @test isempty(cfga.models_include)
        @test isempty(cfga.scheduler.models)

        # The map merges into scheduler.models: an explicit residency wins over the map's device
        # pin, and other per-model fields (weight) are preserved.
        merged = joinpath(dir, "merged.yaml")
        write(merged, """
        model_repo: $modeldir
        base_port: 8080
        global:
          scheduler:
            models:
              resnet50: { weight: 2.0, residency: system }
        workers:
          - { name: solo }
        models:
          resnet50: [solo]
        """)
        cfgm, _, _ = ReactantServer.node_server_config(ReactantServer.load_node(merged), "solo")
        @test cfgm.scheduler.models["resnet50"].residency == ReactantServer.PINNED_SYSTEM  # explicit wins
        @test cfgm.scheduler.models["resnet50"].weight == 2.0                              # preserved

        # validation: assignment to an undefined worker is rejected
        ghosttarget = joinpath(dir, "ghost.yaml")
        write(ghosttarget, """
        model_repo: $modeldir
        base_port: 8080
        workers:
          - { name: a, gpu: 0 }
        models:
          m: [nope]
        """)
        @test_throws ReactantServer.ConfigError ReactantServer.load_node(ghosttarget)

        # validation: colliding ports are rejected
        collide = joinpath(dir, "collide.yaml")
        write(collide, """
        model_repo: $modeldir
        base_port: 8080
        workers:
          - { name: a, gpu: 0, port: 8080 }
          - { name: b, gpu: 1, port: 8080 }
        models:
          m: [a]
        """)
        @test_throws ReactantServer.ConfigError ReactantServer.load_node(collide)
    end
end

@testset "metrics endpoint: port config, env, validation, node derivation" begin
    mktempdir() do dir
        modeldir = joinpath(dir, "models"); mkpath(modeldir)

        # Default: metrics disabled.
        cfg0, _, _ = load_single_worker(dir, ""; model_repo=modeldir)
        @test cfg0.endpoints.metrics_port == 0

        # Explicit port parses and validates.
        cfg, _, _ = load_single_worker(dir, "endpoints:\n  metrics_port: 9100"; model_repo=modeldir)
        @test cfg.endpoints.metrics_port == 9100
        @test ReactantServer.validate_config(cfg) === cfg

        # Env override.
        withenv("INFERENCE_SERVER_ENDPOINTS_METRICS_PORT" => "9300") do
            cfge, applied, _ = load_single_worker(dir, ""; model_repo=modeldir)
            @test cfge.endpoints.metrics_port == 9300
            @test ("INFERENCE_SERVER_ENDPOINTS_METRICS_PORT", "9300") in applied
        end

        # metrics_port must differ from the gRPC port.
        cbad, _, _ = load_single_worker(dir, "endpoints:\n  port: 8080\n  metrics_port: 8080"; model_repo=modeldir)
        @test_throws ReactantServer.ConfigError ReactantServer.validate_config(cbad)

        # node metrics_base_port derives a per-worker metrics port (base + index).
        path = joinpath(dir, "metrics_node.yaml")
        write(path, """
        model_repo: $modeldir
        base_port: 8080
        metrics_base_port: 9100
        workers:
          - { name: worker0 }
          - { name: worker1 }
        """)
        node = ReactantServer.load_node(path)
        c0, _, _ = ReactantServer.node_server_config(node, "worker0")
        c1, _, _ = ReactantServer.node_server_config(node, "worker1")
        @test c0.endpoints.metrics_port == 9100
        @test c1.endpoints.metrics_port == 9101

        # A metrics port colliding with another worker's gRPC port is rejected.
        clash = joinpath(dir, "metrics_clash.yaml")
        write(clash, """
        model_repo: $modeldir
        base_port: 8080
        metrics_base_port: 8081
        workers:
          - { name: worker0 }
          - { name: worker1 }
        """)
        @test_throws ReactantServer.ConfigError ReactantServer.load_node(clash)
    end
end
