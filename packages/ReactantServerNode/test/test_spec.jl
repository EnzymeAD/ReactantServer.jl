# Child spec construction and build_supervisor assembly (no processes spawned here).

# The value of KEY in a Cmd's captured environment, or nothing.
function _envval(cmd::Cmd, key::String)
    cmd.env === nothing && return nothing
    for kv in cmd.env
        startswith(kv, key * "=") && return kv[(length(key) + 2):end]
    end
    return nothing
end

_node_yaml(dir; extra = "") = begin
    path = joinpath(dir, "node.yaml")
    write(
        path, """
        model_repo: /repo
        base_port: 8080
        metrics_base_port: 9100
        global:
          runtime:
            backend: cuda
        $extra
        """
    )
    path
end

# A workspace root with just the member Project.toml files the supervisor launches children from,
# which is all child_environment looks at; no child is ever spawned from it here.
const STUB_ROOT = let root = mktempdir()
    for pkg in ("ReactantServer", "ReactantServerGateway")
        mkpath(joinpath(root, "packages", pkg))
        touch(joinpath(root, "packages", pkg, "Project.toml"))
    end
    root
end

# A Pkg-style environment: a Project.toml declaring ReactantServerNode plus `packages`.
function _pkg_env(packages)
    dir = mktempdir()
    deps = ["ReactantServerNode = \"49f09047-4fb9-4260-828b-8494ed8e9881\""]
    for p in packages
        push!(deps, "$p = \"$(RSN._CHILD_PACKAGE_UUIDS[p])\"")
    end
    file = joinpath(dir, "Project.toml")
    write(file, "[deps]\n" * join(deps, "\n") * "\n")
    return file
end

const BOTH = ["ReactantServer", "ReactantServerGateway"]

@testset "child environment" begin
    nowhere = joinpath(mktempdir(), "Project.toml")   # an active project with nothing around it

    # Workspace shape: an explicit root, else REACTANT_WORKSPACE_ROOT, runs each child from its
    # member project and leaves the load path alone (the container image).
    ce = RSN.child_environment(BOTH; env = Dict("REACTANT_WORKSPACE_ROOT" => STUB_ROOT), active_project = nowhere)
    @test ce.worker_project == joinpath(STUB_ROOT, "packages", "ReactantServer")
    @test ce.gateway_project == joinpath(STUB_ROOT, "packages", "ReactantServerGateway")
    @test ce.load_path === nothing
    @test RSN.child_environment(BOTH; env = Dict{String, String}(), workspace_root = STUB_ROOT) ==
        RSN.workspace_environment(STUB_ROOT)
    # A root that is not a workspace is an error naming the variable, not a restart loop.
    err = try
        RSN.child_environment(BOTH; env = Dict("REACTANT_WORKSPACE_ROOT" => dirname(nowhere)))
    catch e
        e
    end
    @test err isa ReactantServerCore.ConfigError
    @test occursin("REACTANT_WORKSPACE_ROOT", sprint(showerror, err))

    # Environment shape: the supervisor's own environment declares the packages (the Pkg
    # workflow), so every child runs in it, pinned to it plus the standard library.
    env_file = _pkg_env(BOTH)
    ce = RSN.child_environment(BOTH; env = Dict{String, String}(), active_project = env_file)
    @test ce.worker_project == ce.gateway_project == dirname(env_file)
    @test ce.load_path == (Sys.iswindows() ? "@;@stdlib" : "@:@stdlib")
    # Only what the role needs is required.
    server_only = _pkg_env(["ReactantServer"])
    @test RSN.child_environment(["ReactantServer"]; env = Dict{String, String}(), active_project = server_only).load_path !== nothing
    err = try
        RSN.child_environment(BOTH; env = Dict{String, String}(), active_project = server_only)
    catch e
        e
    end
    @test err isa ReactantServerCore.ConfigError
    @test occursin("add ReactantServer and ReactantServerGateway", sprint(showerror, err))
    # A dependency is matched by UUID, not just by name.
    wrong = joinpath(mktempdir(), "Project.toml")
    write(wrong, "[deps]\nReactantServer = \"00000000-0000-0000-0000-000000000000\"\n")
    @test_throws ReactantServerCore.ConfigError RSN.child_environment(
        ["ReactantServer"]; env = Dict{String, String}(), active_project = wrong
    )
    # An explicit workspace still wins over a declaring environment.
    @test RSN.child_environment(
        BOTH; env = Dict("REACTANT_WORKSPACE_ROOT" => STUB_ROOT), active_project = env_file
    ).load_path === nothing

    # Checkout shape: the active project is the member project or the workspace root itself.
    for proj in (joinpath(STUB_ROOT, "packages", "ReactantServerNode", "Project.toml"), joinpath(STUB_ROOT, "Project.toml"))
        @test RSN.child_environment(BOTH; env = Dict{String, String}(), active_project = proj) ==
            RSN.workspace_environment(STUB_ROOT)
    end

    # Nothing provides the packages: one error naming all three fixes.
    err = try
        RSN.child_environment(BOTH; env = Dict{String, String}(), active_project = nowhere)
    catch e
        e
    end
    @test err isa ReactantServerCore.ConfigError
    msg = sprint(showerror, err)
    @test occursin("add ReactantServer and ReactantServerGateway", msg)
    @test occursin("checkout", msg)
    @test occursin("REACTANT_WORKSPACE_ROOT", msg)

    # The specs carry the environment shape: its project and the pinned load path.
    ce = RSN.child_environment(BOTH; env = Dict{String, String}(), active_project = env_file)
    w = RSN.worker_spec("worker0", "/run/node.yaml", nothing, ce)
    @test any(==("--project=$(dirname(env_file))"), w.cmd.exec)
    @test _envval(w.cmd, "JULIA_LOAD_PATH") == ce.load_path
    g = RSN.gateway_spec(ce; endpoints = ["127.0.0.1:8080"])
    @test any(==("--project=$(dirname(env_file))"), g.cmd.exec)
    @test _envval(g.cmd, "JULIA_LOAD_PATH") == ce.load_path
    # The workspace shape inherits the load path (addenv carries the parent's ENV, so it is whatever
    # this process has, which under Pkg.test is set).
    @test _envval(RSN.worker_spec("worker0", "/run/node.yaml", nothing, STUB_ROOT).cmd, "JULIA_LOAD_PATH") ==
        get(ENV, "JULIA_LOAD_PATH", nothing)

    # build_supervisor resolves by role and hands the result to every child.
    mktempdir() do dir
        path = _node_yaml(dir)
        sup = RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "2"), sink = IOBuffer(),
            active_project = env_file, runtime_dir = joinpath(dir, "run")
        )
        @test [c.spec.name for c in sup.children] == ["worker0", "worker1", "gateway"]
        @test all(c -> _envval(c.spec.cmd, "JULIA_LOAD_PATH") == ce.load_path, sup.children)
        @test all(c -> any(==("--project=$(dirname(env_file))"), c.spec.cmd.exec), sup.children)
        # The workers role needs no gateway package; all does.
        sup = RSN.build_supervisor(
            path; role = :workers, env = Dict("REACTANT_GPUS" => "1"), sink = IOBuffer(),
            active_project = server_only, runtime_dir = joinpath(dir, "runw")
        )
        @test [c.spec.name for c in sup.children] == ["worker0"]
        @test_throws ReactantServerCore.ConfigError RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "1"), sink = IOBuffer(),
            active_project = server_only, runtime_dir = joinpath(dir, "runa")
        )
    end
end

@testset "workspace root and child specs" begin
    spec = RSN.worker_spec("worker1", "/run/node.yaml", "1", "/opt/rs"; compute_threads = 8)
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

    gw = RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080", "127.0.0.1:8081"])
    @test gw.name == "gateway"
    @test _envval(gw.cmd, "REACTANT_GATEWAY_WORKERS") == "127.0.0.1:8080,127.0.0.1:8081"

    # A mounted gateway.yml wins: passed as an argument, no endpoint synthesis.
    gwf = RSN.gateway_spec(
        "/opt/rs"; gateway_path = "/etc/reactantserver/gateway.yml",
        endpoints = ["127.0.0.1:8080"]
    )
    @test any(a -> a == "/etc/reactantserver/gateway.yml", gwf.cmd.exec)
    @test _envval(gwf.cmd, "REACTANT_GATEWAY_WORKERS") === nothing

    # ...but the startup wait is still set, because it describes the supervisor's own co-launched
    # workers (which compile before answering) rather than gateway configuration, and a file that
    # owns the endpoint list cannot express it. Without this a mounted-file node crash-loops the
    # gateway for the entire compile window under lpt_packing.
    @test _envval(gw.cmd, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS") == "inf"
    @test _envval(gwf.cmd, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS") == "inf"
    # An explicit value still wins, in both shapes: the default is never forced over it. (What
    # `_envval` sees differs between them and neither is a bug: `addenv` materializes the whole
    # environment, so the endpoint shape shows the inherited "120", while the mounted-file shape adds
    # no pairs at all and leaves `cmd.env` unset, which means the child inherits ENV as it stands.)
    withenv("REACTANT_GATEWAY_STARTUP_WAIT_SECONDS" => "120") do
        for c in (
                RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"]).cmd,
                RSN.gateway_spec("/opt/rs"; gateway_path = "/etc/x/gateway.yml").cmd,
            )
            @test _envval(c, "REACTANT_GATEWAY_STARTUP_WAIT_SECONDS") != "inf"
        end
    end
end

@testset "worker endpoints and gateway listen ports" begin
    node = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[
            Dict{String, Any}("name" => "a"),
            Dict{String, Any}("name" => "b"),
        ]
    )
    @test RSN.worker_endpoints(node) == ["127.0.0.1:8080", "127.0.0.1:8081"]

    @test RSN._gateway_listen_ports(nothing, Dict{String, String}()) == Set([8001, 8002])
    @test RSN._gateway_listen_ports(
        nothing,
        Dict("REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:7001")
    ) == Set([7001, 8002])
    mktempdir() do dir
        gwy = joinpath(dir, "gateway.yml")
        write(gwy, "listen:\n  grpc: \"0.0.0.0:6001\"\n  metrics: \"0.0.0.0:6002\"\n")
        @test RSN._gateway_listen_ports(gwy, Dict{String, String}()) == Set([6001, 6002])
    end
end

@testset "build_supervisor: all-in-one" begin
    mktempdir() do dir
        path = _node_yaml(dir)
        sink = IOBuffer()
        sup = RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "2"), sink = sink,
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        names = [c.spec.name for c in sup.children]
        @test names == ["worker0", "worker1", "gateway"]

        # The materialized node file has the synthesized workers and is what children read.
        mat = joinpath(dir, "run", "node.yaml")
        @test isfile(mat)
        node = YAML.load_file(mat; dicttype = Dict{String, Any})
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
        sup = RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "1"), sink = sink,
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        # No gateway child: the sole worker is the node's public endpoint.
        @test [c.spec.name for c in sup.children] == ["worker0"]
        w = sup.children[1].spec.cmd
        @test _envval(w, "INFERENCE_SERVER_ENDPOINTS_PORT") == "8001"
        @test _envval(w, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT") == "8002"
        @test _envval(w, "CUDA_VISIBLE_DEVICES") == "0"
        @test occursin("no gateway", String(take!(sink)))

        # The public ports honor the gateway listen overrides.
        sup = RSN.build_supervisor(
            path;
            env = Dict(
                "REACTANT_GPUS" => "1", "REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:7001",
                "REACTANT_GATEWAY_LISTEN_METRICS" => "0.0.0.0:7002"
            ),
            sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run2")
        )
        @test _envval(sup.children[1].spec.cmd, "INFERENCE_SERVER_ENDPOINTS_PORT") == "7001"
        @test _envval(sup.children[1].spec.cmd, "INFERENCE_SERVER_ENDPOINTS_METRICS_PORT") == "7002"
    end
end

@testset "build_supervisor: roles and errors" begin
    mktempdir() do dir
        path = _node_yaml(dir)

        # workers role (via env): no gateway child.
        sup = RSN.build_supervisor(
            path;
            env = Dict("REACTANT_GPUS" => "1", "REACTANT_ROLE" => "workers"),
            sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "runw")
        )
        @test [c.spec.name for c in sup.children] == ["worker0"]

        # gateway role: just the gateway, no node materialization (works without GPUs).
        sup = RSN.build_supervisor(
            path; role = :gateway, env = Dict{String, String}(),
            sink = IOBuffer(), workspace_root = STUB_ROOT
        )
        @test [c.spec.name for c in sup.children] == ["gateway"]

        # CUDA node with zero devices is a hard error with guidance.
        @test_throws ReactantServerCore.ConfigError RSN.build_supervisor(
            path;
            env = Dict("REACTANT_GPUS" => "0"), sink = IOBuffer(), workspace_root = STUB_ROOT,
            runtime_dir = joinpath(dir, "rune")
        )

        # Bad role rejected.
        @test_throws ReactantServerCore.ConfigError RSN.build_supervisor(
            path;
            role = :everything, env = Dict{String, String}(), sink = IOBuffer(),
            workspace_root = STUB_ROOT
        )

        # max_restarts from the environment.
        sup = RSN.build_supervisor(
            path;
            env = Dict("REACTANT_GPUS" => "1", "REACTANT_SUPERVISOR_MAX_RESTARTS" => "4"),
            sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "runm")
        )
        @test sup.max_restarts == 4
    end
end

@testset "build_supervisor: port collision warning" begin
    mktempdir() do dir
        path = joinpath(dir, "node.yaml")
        write(
            path, """
            model_repo: /repo
            base_port: 8001
            global:
              runtime:
                backend: cuda
            """
        )
        sink = IOBuffer()
        RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "2"), sink = sink,
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        out = String(take!(sink))
        @test occursin("WARNING", out)
        @test occursin("8001", out)
    end
end

@testset "build_supervisor: shared_host_weights warning" begin
    # Multiple workers + on-demand cache + private host floor: warn about the Nx host-RAM copies.
    ondemand_node(dir; shared) = begin
        path = joinpath(dir, "node.yaml")
        write(
            path, """
            model_repo: /repo
            base_port: 8080
            metrics_base_port: 9100
            global:
              runtime:
                backend: cuda
                weight_cache_fraction: 0.6
                shared_host_weights: $shared
            """
        )
        path
    end

    # Two workers, on-demand cache, shared store off: the warning fires.
    mktempdir() do dir
        sink = IOBuffer()
        RSN.build_supervisor(
            ondemand_node(dir; shared = false); env = Dict("REACTANT_GPUS" => "2"),
            sink = sink, workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        out = String(take!(sink))
        @test occursin("shared_host_weights", out)
        @test occursin("private host copy", out)
    end

    # Same, but the shared store is on: no warning.
    mktempdir() do dir
        sink = IOBuffer()
        RSN.build_supervisor(
            ondemand_node(dir; shared = true); env = Dict("REACTANT_GPUS" => "2"),
            sink = sink, workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        @test !occursin("private host copy", String(take!(sink)))
    end

    # A single worker materializes one floor regardless, so the warning is gated on >1 worker.
    mktempdir() do dir
        sink = IOBuffer()
        RSN.build_supervisor(
            ondemand_node(dir; shared = false); env = Dict("REACTANT_GPUS" => "1"),
            sink = sink, workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        @test !occursin("private host copy", String(take!(sink)))
    end
end

@testset "build_supervisor: cpu node" begin
    mktempdir() do dir
        path = joinpath(dir, "node.yaml")
        write(
            path, """
            model_repo: /repo
            base_port: 8080
            global:
              runtime:
                backend: cpu
            """
        )
        sup = RSN.build_supervisor(
            path;
            env = Dict("REACTANT_GPUS" => "0", "REACTANT_CPU_WORKERS" => "2"),
            sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        @test [c.spec.name for c in sup.children] == ["worker0", "worker1", "gateway"]
        @test _envval(sup.children[1].spec.cmd, "CUDA_VISIBLE_DEVICES") == ""
    end
end

@testset "gateway_spec: grpc message-size env synthesis" begin
    withenv(
        "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => nothing,
        "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES" => nothing
    ) do
        # Provided sizes are synthesized into the gateway's env.
        gw = RSN.gateway_spec(
            "/opt/rs"; endpoints = ["127.0.0.1:8080"],
            grpc_max_recv = 111, grpc_max_send = 222
        )
        @test _envval(gw.cmd, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") == "111"
        @test _envval(gw.cmd, "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES") == "222"

        # Omitted -> not synthesized (the gateway uses its own default).
        gw0 = RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"])
        @test _envval(gw0.cmd, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") === nothing

        # A mounted gateway.yml suppresses synthesis (the gateway reads its own file).
        gwf = RSN.gateway_spec("/opt/rs"; gateway_path = "/etc/x/gateway.yml", grpc_max_recv = 111)
        @test _envval(gwf.cmd, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") === nothing
    end

    # An explicit process env wins over the synthesized value.
    withenv("REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => "999") do
        gwe = RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"], grpc_max_recv = 111)
        @test _envval(gwe.cmd, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") == "999"
    end
end

@testset "build_supervisor: node global.grpc reaches the embedded gateway" begin
    mktempdir() do dir
        path = joinpath(dir, "node.yaml")
        write(
            path, """
            model_repo: /repo
            base_port: 8080
            metrics_base_port: 9100
            global:
              runtime:
                backend: cuda
              grpc:
                max_recv_msg_bytes: 111
                max_send_msg_bytes: 222
            """
        )
        withenv(
            "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => nothing,
            "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES" => nothing
        ) do
            sup = RSN.build_supervisor(
                path; env = Dict("REACTANT_GPUS" => "2"),
                sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
            )
            gw = sup.children[end].spec.cmd   # the gateway is the last child
            @test _envval(gw, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") == "111"
            @test _envval(gw, "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES") == "222"
        end
    end
end
