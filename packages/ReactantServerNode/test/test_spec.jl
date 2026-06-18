# Child spec construction and build_supervisor assembly (no processes spawned here).

# The value of KEY in a Cmd's captured environment, or nothing.
function _envval(cmd::Cmd, key::String)
    cmd.env === nothing && return nothing
    for kv in cmd.env
        startswith(kv, key * "=") && return kv[(length(key) + 2):end]
    end
    return nothing
end

_node_yaml(dir; extra="") = begin
    path = joinpath(dir, "node.yaml")
    write(path, """
    model_repo: /repo
    base_port: 8080
    metrics_base_port: 9100
    global:
      runtime:
        backend: cuda
    $extra
    """)
    path
end

@testset "workspace root and child specs" begin
    @test RSN.default_workspace_root(Dict("REACTANT_WORKSPACE_ROOT" => "/opt/rs")) == "/opt/rs"

    spec = RSN.worker_spec("worker1", "/run/node.yaml", "1", "/opt/rs"; compute_threads=8)
    @test spec.name == "worker1"
    @test _envval(spec.cmd, "REACTANT_WORKER_NAME") == "worker1"
    @test _envval(spec.cmd, "CUDA_VISIBLE_DEVICES") == "1"
    @test any(a -> a == "/run/node.yaml", spec.cmd.exec)
    # Sized to the worker's share of the host, plus one interactive thread for the GPU dispatch loop.
    @test any(a -> a == "--threads=8,1", spec.cmd.exec)
    @test any(a -> occursin("packages/ReactantServer", a), spec.cmd.exec)

    # Per-worker thread share: cores ÷ workers, floored at 1 and capped (default 16).
    @test RSN._worker_thread_count(32, 2) == 16
    @test RSN._worker_thread_count(64, 2) == 16    # capped
    @test RSN._worker_thread_count(16, 2) == 8
    @test RSN._worker_thread_count(8, 1) == 8
    @test RSN._worker_thread_count(100, 1) == 16   # capped
    @test RSN._worker_thread_count(1, 4) == 1      # floored

    # A CPU worker gets explicit empty GPU visibility (never the container's
    # CUDA_VISIBLE_DEVICES). Bind hosts stay the node file's concern.
    spec = RSN.worker_spec("worker0", "/run/node.yaml", nothing, "/opt/rs")
    @test _envval(spec.cmd, "CUDA_VISIBLE_DEVICES") == ""

    gw = RSN.gateway_spec("/opt/rs"; endpoints=["127.0.0.1:8080", "127.0.0.1:8081"])
    @test gw.name == "gateway"
    @test _envval(gw.cmd, "REACTANT_GATEWAY_WORKERS") == "127.0.0.1:8080,127.0.0.1:8081"

    # A mounted gateway.yml wins: passed as an argument, no endpoint synthesis.
    gwf = RSN.gateway_spec("/opt/rs"; gateway_path="/etc/reactantserver/gateway.yml",
                           endpoints=["127.0.0.1:8080"])
    @test any(a -> a == "/etc/reactantserver/gateway.yml", gwf.cmd.exec)
    @test _envval(gwf.cmd, "REACTANT_GATEWAY_WORKERS") === nothing
end

@testset "worker endpoints and gateway listen ports" begin
    node = Dict{String,Any}("model_repo" => "/repo", "base_port" => 8080,
                            "workers" => Any[Dict{String,Any}("name" => "a"),
                                             Dict{String,Any}("name" => "b")])
    @test RSN.worker_endpoints(node) == ["127.0.0.1:8080", "127.0.0.1:8081"]

    @test RSN._gateway_listen_ports(nothing, Dict{String,String}()) == Set([8001, 8002])
    @test RSN._gateway_listen_ports(nothing,
        Dict("REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:7001")) == Set([7001, 8002])
    mktempdir() do dir
        gwy = joinpath(dir, "gateway.yml")
        write(gwy, "listen:\n  grpc: \"0.0.0.0:6001\"\n  metrics: \"0.0.0.0:6002\"\n")
        @test RSN._gateway_listen_ports(gwy, Dict{String,String}()) == Set([6001, 6002])
    end
end

@testset "build_supervisor: all-in-one" begin
    mktempdir() do dir
        path = _node_yaml(dir)
        sink = IOBuffer()
        sup = RSN.build_supervisor(path; env=Dict("REACTANT_GPUS" => "2"), sink=sink,
                                   workspace_root="/opt/rs", runtime_dir=joinpath(dir, "run"))
        names = [c.spec.name for c in sup.children]
        @test names == ["worker0", "worker1", "gateway"]

        # The materialized node file has the synthesized workers and is what children read.
        mat = joinpath(dir, "run", "node.yaml")
        @test isfile(mat)
        node = YAML.load_file(mat; dicttype=Dict{String,Any})
        @test [w["name"] for w in node["workers"]] == ["worker0", "worker1"]
        for c in sup.children[1:2]
            @test any(a -> a == mat, c.spec.cmd.exec)
        end
        @test _envval(sup.children[1].spec.cmd, "CUDA_VISIBLE_DEVICES") == "0"
        @test _envval(sup.children[2].spec.cmd, "CUDA_VISIBLE_DEVICES") == "1"
        # Multi-worker: the gateway owns the public ports, so workers keep their node-file ports
        # (no public-port override) and the gateway fronts them.
        @test _envval(sup.children[1].spec.cmd, "INFERENCE_SERVER_ENDPOINTS_PORT") === nothing
        @test _envval(sup.children[3].spec.cmd, "REACTANT_GATEWAY_WORKERS") ==
              "127.0.0.1:8080,127.0.0.1:8081"
        # Worker metrics endpoints feed the gateway's aggregated /metrics.
        @test _envval(sup.children[3].spec.cmd, "REACTANT_GATEWAY_WORKER_METRICS") ==
              "127.0.0.1:9100,127.0.0.1:9101"
        @test occursin("role=all", String(take!(sink)))
    end
end

@testset "build_supervisor: single worker has no gateway" begin
    mktempdir() do dir
        path = _node_yaml(dir)
        sink = IOBuffer()
        sup = RSN.build_supervisor(path; env=Dict("REACTANT_GPUS" => "1"), sink=sink,
                                   workspace_root="/opt/rs", runtime_dir=joinpath(dir, "run"))
        # No gateway child: the sole worker is the node's public endpoint.
        @test [c.spec.name for c in sup.children] == ["worker0"]
        w = sup.children[1].spec.cmd
        @test _envval(w, "INFERENCE_SERVER_ENDPOINTS_PORT") == "8001"
        @test _envval(w, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT") == "8002"
        @test _envval(w, "CUDA_VISIBLE_DEVICES") == "0"
        @test occursin("no gateway", String(take!(sink)))

        # The public ports honor the gateway listen overrides.
        sup = RSN.build_supervisor(path;
            env=Dict("REACTANT_GPUS" => "1", "REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:7001",
                     "REACTANT_GATEWAY_LISTEN_METRICS" => "0.0.0.0:7002"),
            sink=IOBuffer(), workspace_root="/opt/rs", runtime_dir=joinpath(dir, "run2"))
        @test _envval(sup.children[1].spec.cmd, "INFERENCE_SERVER_ENDPOINTS_PORT") == "7001"
        @test _envval(sup.children[1].spec.cmd, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT") == "7002"
    end
end

@testset "build_supervisor: roles and errors" begin
    mktempdir() do dir
        path = _node_yaml(dir)

        # workers role (via env): no gateway child.
        sup = RSN.build_supervisor(path;
            env=Dict("REACTANT_GPUS" => "1", "REACTANT_ROLE" => "workers"),
            sink=IOBuffer(), workspace_root="/opt/rs", runtime_dir=joinpath(dir, "runw"))
        @test [c.spec.name for c in sup.children] == ["worker0"]

        # gateway role: just the gateway, no node materialization (works without GPUs).
        sup = RSN.build_supervisor(path; role=:gateway, env=Dict{String,String}(),
                                   sink=IOBuffer(), workspace_root="/opt/rs")
        @test [c.spec.name for c in sup.children] == ["gateway"]

        # CUDA node with zero devices is a hard error with guidance.
        @test_throws ReactantServerCore.ConfigError RSN.build_supervisor(path;
            env=Dict("REACTANT_GPUS" => "0"), sink=IOBuffer(), workspace_root="/opt/rs",
            runtime_dir=joinpath(dir, "rune"))

        # Bad role rejected.
        @test_throws ReactantServerCore.ConfigError RSN.build_supervisor(path;
            role=:everything, env=Dict{String,String}(), sink=IOBuffer(),
            workspace_root="/opt/rs")

        # max_restarts from the environment.
        sup = RSN.build_supervisor(path;
            env=Dict("REACTANT_GPUS" => "1", "REACTANT_SUPERVISOR_MAX_RESTARTS" => "4"),
            sink=IOBuffer(), workspace_root="/opt/rs", runtime_dir=joinpath(dir, "runm"))
        @test sup.max_restarts == 4
    end
end

@testset "build_supervisor: port collision warning" begin
    mktempdir() do dir
        path = joinpath(dir, "node.yaml")
        write(path, """
        model_repo: /repo
        base_port: 8001
        global:
          runtime:
            backend: cuda
        """)
        sink = IOBuffer()
        RSN.build_supervisor(path; env=Dict("REACTANT_GPUS" => "2"), sink=sink,
                             workspace_root="/opt/rs", runtime_dir=joinpath(dir, "run"))
        out = String(take!(sink))
        @test occursin("WARNING", out)
        @test occursin("8001", out)
    end
end

@testset "build_supervisor: cpu node" begin
    mktempdir() do dir
        path = joinpath(dir, "node.yaml")
        write(path, """
        model_repo: /repo
        base_port: 8080
        global:
          runtime:
            backend: cpu
        """)
        sup = RSN.build_supervisor(path;
            env=Dict("REACTANT_GPUS" => "0", "REACTANT_CPU_WORKERS" => "2"),
            sink=IOBuffer(), workspace_root="/opt/rs", runtime_dir=joinpath(dir, "run"))
        @test [c.spec.name for c in sup.children] == ["worker0", "worker1", "gateway"]
        @test _envval(sup.children[1].spec.cmd, "CUDA_VISIBLE_DEVICES") == ""
    end
end
