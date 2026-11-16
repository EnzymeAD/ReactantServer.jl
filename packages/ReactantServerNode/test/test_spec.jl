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
    # The host weight budget's per-worker split: the node's worker count, 1 by default.
    @test _envval(spec.cmd, "INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET_WORKERS") == "1"
    spec4 = RSN.worker_spec("worker1", "/run/node.yaml", "1", "/opt/rs"; node_workers = 4)
    @test _envval(spec4.cmd, "INFERENCE_SERVER_RUNTIME_HOST_WEIGHT_BUDGET_WORKERS") == "4"

    # Per-worker thread share: cores ÷ workers, floored at 1 and capped (default 16).
    @test RSN._worker_thread_count(32, 2) == 16
    @test RSN._worker_thread_count(64, 2) == 16    # capped
    @test RSN._worker_thread_count(16, 2) == 8
    @test RSN._worker_thread_count(8, 1) == 8
    @test RSN._worker_thread_count(100, 1) == 16   # capped
    @test RSN._worker_thread_count(1, 4) == 1      # floored

    # A drain budget of 0 opts a child out of the stdin drain protocol (SIGTERM at once).
    nodrain = RSN.worker_spec("worker1", "/run/node.yaml", "1", "/opt/rs"; drain_seconds = 0)
    @test nodrain.drain_seconds == 0
    @test _envval(nodrain.cmd, ReactantServerCore.STDIN_DRAIN_ENV) === nothing
    # The three-argument form (a plain child) neither drains nor is staged after anything.
    plain = RSN.ChildSpec("plain", `true`, 3)
    @test (plain.grace_seconds, plain.drain_seconds, plain.stage) == (3.0, 0.0, 1)

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
        # Shutdown: the gateway (stage 1) drains for up to a minute before the workers (stage 2)
        # are asked; each child reads the drain budget from the variable its stdin protocol uses.
        gw, w0 = sup.children[3].spec, sup.children[1].spec
        @test (gw.stage, gw.drain_seconds) == (1, RSN.PUBLIC_DRAIN_SECONDS)
        @test (w0.stage, w0.drain_seconds) == (2, RSN.WORKER_DRAIN_SECONDS)
        @test RSN.PUBLIC_DRAIN_SECONDS == 60
        @test _envval(gw.cmd, ReactantServerCore.STDIN_DRAIN_ENV) == "60.0"
        @test _envval(w0.cmd, ReactantServerCore.STDIN_DRAIN_ENV) == string(RSN.WORKER_DRAIN_SECONDS)
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
        # As the public endpoint, the sole worker drains as long as a gateway would.
        @test sup.children[1].spec.drain_seconds == RSN.PUBLIC_DRAIN_SECONDS
        @test _envval(w, ReactantServerCore.STDIN_DRAIN_ENV) == "60.0"
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

@testset "build_supervisor: no host-RAM advisory for private host copies" begin
    # Without shared host weights each worker gets an equal share of runtime.host_weight_budget, so
    # private copies never multiply host RAM by the worker count and the supervisor has nothing to
    # warn about.
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
                host_weight_budget: 8g
                shared_host_weights: false
            """
        )
        sink = IOBuffer()
        RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "2"),
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

@testset "gateway_spec: GC threads" begin
    gcflags(cmd) = filter(startswith("--gcthreads"), cmd.exec)
    marks = min(Sys.CPU_THREADS, 4)
    flag = "--gcthreads=$marks,1"
    withenv(RSN._GC_THREADS_ENV => nothing) do
        for gw in (
                RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"]),
                RSN.gateway_spec("/opt/rs"; gateway_path = "/etc/x/gateway.yml"),
            )
            # Exactly one, so Base.julia_cmd() brings none to conflict with, and placed as a Julia
            # option, before the program, not among the program's arguments.
            @test gcflags(gw.cmd) == [flag]
            @test findfirst(==(flag), gw.cmd.exec) < findfirst(==("-e"), gw.cmd.exec)
        end
        # An empty variable counts as unset.
        withenv(RSN._GC_THREADS_ENV => "") do
            @test gcflags(RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"]).cmd) == [flag]
        end
        # This Julia takes the flag as meant: the mark threads (the collecting thread counts as one)
        # plus one sweep thread.
        julia = `$(Base.julia_cmd()) --startup-file=no $flag -e 'print(Threads.ngcthreads())'`
        @test parse(Int, readchomp(julia)) == marks + 1
    end
    # The mark threads are capped at the host's CPUs.
    unset = Dict{String, String}()
    @test RSN._gateway_gc_flags(unset; cpu_threads = 64) == ["--gcthreads=4,1"]
    @test RSN._gateway_gc_flags(unset; cpu_threads = 4) == ["--gcthreads=4,1"]
    @test RSN._gateway_gc_flags(unset; cpu_threads = 2) == ["--gcthreads=2,1"]
    @test RSN._gateway_gc_flags(unset; cpu_threads = 1) == ["--gcthreads=1,1"]
    # JULIA_NUM_GC_THREADS set for the supervisor applies instead: the child inherits it, and no flag
    # is added, since Julia would let the flag win over it.
    @test RSN._gateway_gc_flags(Dict(RSN._GC_THREADS_ENV => "2,1"); cpu_threads = 64) == String[]
    withenv(RSN._GC_THREADS_ENV => "2,1") do
        gw = RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"])
        @test isempty(gcflags(gw.cmd))
        @test _envval(gw.cmd, RSN._GC_THREADS_ENV) == "2,1"
    end
    # The workers keep Julia's default.
    @test isempty(gcflags(RSN.worker_spec("worker0", "/run/node.yaml", nothing, "/opt/rs").cmd))
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

        # The INTERNAL error detail is mirrored the same way.
        withenv("REACTANT_GATEWAY_GRPC_ERROR_DETAIL" => nothing) do
            gwd = RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"], grpc_error_detail = "full")
            @test _envval(gwd.cmd, "REACTANT_GATEWAY_GRPC_ERROR_DETAIL") == "full"
            @test _envval(gw0.cmd, "REACTANT_GATEWAY_GRPC_ERROR_DETAIL") === nothing
        end
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
                error_detail: full
            """
        )
        withenv(
            "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => nothing,
            "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES" => nothing,
            "REACTANT_GATEWAY_GRPC_ERROR_DETAIL" => nothing
        ) do
            sup = RSN.build_supervisor(
                path; env = Dict("REACTANT_GPUS" => "2"),
                sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
            )
            gw = sup.children[end].spec.cmd   # the gateway is the last child
            @test _envval(gw, "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES") == "111"
            @test _envval(gw, "REACTANT_GATEWAY_GRPC_MAX_SEND_MSG_BYTES") == "222"
            # One node-level error_detail reaches the gateway (the workers read global.grpc directly).
            @test _envval(gw, "REACTANT_GATEWAY_GRPC_ERROR_DETAIL") == "full"
        end
        # A value that is neither mode is refused at startup rather than silently ignored.
        write(path, replace(read(path, String), "error_detail: full" => "error_detail: verbose"))
        @test_throws ArgumentError RSN.build_supervisor(
            path; env = Dict("REACTANT_GPUS" => "2"),
            sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
    end
end

const _RRP_GW = "REACTANT_GATEWAY_GRPC_RAPID_RESET_PROTECTION"

@testset "gateway_spec: the node-wide Rapid-Reset setting" begin
    mktempdir() do dir
        silent = joinpath(dir, "silent.yml")      # a mounted gateway.yml that does not set the key
        write(silent, "grpc:\n  error_detail: full\nendpoints:\n  - \"127.0.0.1:8080\"\n")
        own = joinpath(dir, "own.yml")            # one that does
        write(own, "grpc:\n  rapid_reset_protection: false\nendpoints:\n  - \"127.0.0.1:8080\"\n")
        withenv(_RRP_GW => nothing) do
            ep = ["127.0.0.1:8080"]
            @test _envval(RSN.gateway_spec("/opt/rs"; endpoints = ep, rapid_reset_protection = true).cmd, _RRP_GW) == "true"
            @test _envval(RSN.gateway_spec("/opt/rs"; endpoints = ep, rapid_reset_protection = false).cmd, _RRP_GW) == "false"
            # Unset at node level: nothing synthesized, so the gateway keeps its own default (off).
            @test _envval(RSN.gateway_spec("/opt/rs"; endpoints = ep).cmd, _RRP_GW) === nothing
            # Unlike the message sizes, it reaches a gateway run from a mounted file that is silent on
            # it, but a value written in the file wins (the variable would override it).
            @test _envval(RSN.gateway_spec("/opt/rs"; gateway_path = silent, rapid_reset_protection = true).cmd, _RRP_GW) == "true"
            @test _envval(RSN.gateway_spec("/opt/rs"; gateway_path = own, rapid_reset_protection = true).cmd, _RRP_GW) === nothing
            @test RSN._gateway_file_sets(own, "grpc", "rapid_reset_protection")
            @test !RSN._gateway_file_sets(silent, "grpc", "rapid_reset_protection")
            @test !RSN._gateway_file_sets(nothing, "grpc", "rapid_reset_protection")
        end
        # The gateway's own variable wins: it is never overwritten.
        withenv(_RRP_GW => "false") do
            @test _envval(RSN.gateway_spec("/opt/rs"; endpoints = ["127.0.0.1:8080"], rapid_reset_protection = true).cmd, _RRP_GW) == "false"
        end
    end
end

@testset "build_supervisor: one Rapid-Reset setting reaches the gateway and every worker" begin
    mktempdir() do dir
        # Each worker's resolved setting, from the node file the supervisor materialized for them
        # (what each worker child reads), and the gateway child's synthesized variable.
        function launched(node_body; env = Dict{String, String}(), gateway_path = nothing, role = nothing, tag = "run")
            path = joinpath(dir, "node_$tag.yaml")
            write(path, "model_repo: /repo\nbase_port: 8080\n" * node_body)
            run_dir = joinpath(dir, tag)
            sup = RSN.build_supervisor(
                path; env = merge(Dict("REACTANT_GPUS" => "0", "REACTANT_CPU_WORKERS" => "2"), env),
                role = role, gateway_path = gateway_path, sink = IOBuffer(), workspace_root = STUB_ROOT,
                runtime_dir = run_dir
            )
            gw = [c.spec for c in sup.children if c.spec.name == "gateway"]
            gwval = isempty(gw) ? :none : _envval(only(gw).cmd, _RRP_GW)
            mat = joinpath(run_dir, "node.yaml")
            isfile(mat) || return Dict{String, Bool}(), gwval
            node = ReactantServerCore.load_node(mat)
            workers = Dict(
                n => ReactantServerCore.node_server_config(node, n)[1].grpc.rapid_reset_protection
                    for n in ReactantServerCore.worker_names(node)
            )
            # Every worker child runs from that file.
            @test all(c -> c.spec.name == "gateway" || any(==(mat), c.spec.cmd.exec), sup.children)
            return workers, gwval
        end
        cpu = "global:\n  runtime:\n    backend: cpu\n"
        on = "global:\n  runtime:\n    backend: cpu\n  grpc:\n    rapid_reset_protection: true\n"
        two = "workers:\n  - { name: worker0 }\n  - { name: worker1, grpc: { rapid_reset_protection: false } }\n"
        withenv(_RRP_GW => nothing, "INFERENCE_SERVER_GRPC_RAPID_RESET_PROTECTION" => nothing) do
            # Default: off everywhere, and nothing is written or synthesized.
            @test launched(cpu; tag = "default") == (Dict("worker0" => false, "worker1" => false), nothing)
            # The node file's one key turns it on for the gateway and every worker.
            @test launched(on; tag = "file") == (Dict("worker0" => true, "worker1" => true), "true")
            # So does one variable on the node container, without the key; and it wins over the key.
            @test launched(cpu; env = Dict(RSN.NODE_RAPID_RESET_ENV => "true"), tag = "env") ==
                (Dict("worker0" => true, "worker1" => true), "true")
            @test launched(on; env = Dict(RSN.NODE_RAPID_RESET_ENV => "off"), tag = "envoff") ==
                (Dict("worker0" => false, "worker1" => false), "false")
            @test launched(on; env = Dict(RSN.NODE_RAPID_RESET_ENV => ""), tag = "envempty") ==
                (Dict("worker0" => true, "worker1" => true), "true")
            # A worker's own value wins over the node-wide one, from the file or the environment.
            @test launched(on * two; tag = "perworker") == (Dict("worker0" => true, "worker1" => false), "true")
            @test launched(cpu * two; env = Dict(RSN.NODE_RAPID_RESET_ENV => "true"), tag = "perworker_env") ==
                (Dict("worker0" => true, "worker1" => false), "true")
            # A mounted gateway.yml that sets the key keeps its value; one that is silent gets the node's.
            own = joinpath(dir, "own.yml")
            write(own, "grpc:\n  rapid_reset_protection: false\nendpoints:\n  - \"127.0.0.1:8080\"\n")
            silent = joinpath(dir, "silent.yml")
            write(silent, "endpoints:\n  - \"127.0.0.1:8080\"\n")
            @test launched(on; gateway_path = own, tag = "gwown") == (Dict("worker0" => true, "worker1" => true), nothing)
            @test launched(on; gateway_path = silent, tag = "gwsilent") == (Dict("worker0" => true, "worker1" => true), "true")
            # The gateway role (a node split across machines) takes it too.
            @test launched(cpu; env = Dict(RSN.NODE_RAPID_RESET_ENV => "1"), role = :gateway, tag = "gwrole") ==
                (Dict{String, Bool}(), "true")
            # A value that is not a boolean is refused at startup, from either source.
            @test_throws ReactantServerCore.ConfigError launched(cpu; env = Dict(RSN.NODE_RAPID_RESET_ENV => "ture"), tag = "bad1")
            @test_throws ReactantServerCore.ConfigError launched(
                "global:\n  runtime:\n    backend: cpu\n  grpc:\n    rapid_reset_protection: yes please\n"; tag = "bad2"
            )
        end
    end
end

@testset "gateway file: explicit, REACTANT_GATEWAY_FILE, then the mount point" begin
    mktempdir() do dir
        mounted = joinpath(dir, "mounted.yml")
        envfile = joinpath(dir, "env.yml")
        explicit = joinpath(dir, "explicit.yml")
        foreach(f -> write(f, "endpoints: []\n"), (mounted, envfile))
        saved = RSN._DEFAULT_GATEWAY_FILE[]
        RSN._DEFAULT_GATEWAY_FILE[] = mounted
        try
            @test RSN._resolve_gateway_path(nothing, Dict{String, String}()) == mounted
            @test RSN._resolve_gateway_path(nothing, Dict("REACTANT_GATEWAY_FILE" => envfile)) == envfile
            @test RSN._resolve_gateway_path(explicit, Dict("REACTANT_GATEWAY_FILE" => envfile)) == explicit
            @test_throws ReactantServerCore.ConfigError RSN._resolve_gateway_path(
                nothing, Dict("REACTANT_GATEWAY_FILE" => joinpath(dir, "missing.yml"))
            )
            rm(mounted)
            @test RSN._resolve_gateway_path(nothing, Dict{String, String}()) === nothing
        finally
            RSN._DEFAULT_GATEWAY_FILE[] = saved
        end
    end
end

@testset "build_supervisor: worker liveness settings" begin
    mktempdir() do dir
        build(path, env; tag = "run") = RSN.build_supervisor(
            path; env = env, sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, tag)
        )
        IPv4, IPv6 = RSN.Sockets.IPv4, RSN.Sockets.IPv6
        stall(c) = _envval(c.spec.cmd, RSN.DISPATCH_STALL_ENV)

        # The defaults: each worker is checked on its own metrics port, on loopback for its wildcard
        # or loopback bind; the gateway is not checked; every worker gets the stall limit.
        sink = IOBuffer()
        sup = RSN.build_supervisor(
            _node_yaml(dir); env = Dict("REACTANT_GPUS" => "2"), sink = sink,
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        w0, w1, gw = (c.spec for c in sup.children)
        @test (w0.liveness.ip, w0.liveness.port, w1.liveness.port) == (IPv4("127.0.0.1"), 9100, 9101)
        @test (w0.liveness.interval, w0.liveness.timeout, w0.liveness.failures) == (10.0, 10.0, 6)
        @test gw.liveness.port == 8002                   # the gateway's admin listener (its own testset)
        @test stall(sup.children[1]) == stall(sup.children[2]) == "600.0"
        @test occursin(
            "worker liveness checks: /healthz every 10s once ready, 10s to answer, SIGKILL after 6 failures in a row; worker dispatch stall limit: 600s",
            String(take!(sink))
        )

        # The node file's `liveness:` block, with each REACTANT_LIVENESS_* variable over it. A sole
        # worker is checked on the public metrics port it is moved to.
        path = _node_yaml(dir; extra = "liveness: {interval_seconds: 5, timeout_seconds: 2.5, failures: 3, dispatch_stall_seconds: 120}")
        sup = build(path, Dict("REACTANT_GPUS" => "1", "REACTANT_LIVENESS_FAILURES" => "4"))
        chk = only(sup.children).spec.liveness
        @test (chk.interval, chk.timeout, chk.failures, chk.port) == (5.0, 2.5, 4, 8002)
        @test stall(only(sup.children)) == "120.0"
        sup = build(path, Dict("REACTANT_GPUS" => "1", RSN.DISPATCH_STALL_ENV => "0"))
        @test stall(only(sup.children)) == "0.0"

        # `failures: 0` turns the supervisor's check off; the workers still get their stall limit.
        sink = IOBuffer()
        sup = RSN.build_supervisor(
            _node_yaml(dir; extra = "liveness: {failures: 0}"); env = Dict("REACTANT_GPUS" => "2"), sink = sink,
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "off")
        )
        @test all(c -> c.spec.liveness === nothing, sup.children)
        @test stall(sup.children[1]) == "600.0"
        @test occursin("worker liveness checks: off (liveness.failures is 0)", String(take!(sink)))

        # A worker with no HTTP listener has nothing to check.
        bare = joinpath(dir, "bare.yaml")
        write(bare, "model_repo: /repo\nbase_port: 8080\nglobal:\n  runtime:\n    backend: cuda\n")
        sink = IOBuffer()
        sup = RSN.build_supervisor(
            bare; env = Dict("REACTANT_GPUS" => "2"), sink = sink, workspace_root = STUB_ROOT,
            runtime_dir = joinpath(dir, "bare")
        )
        @test all(c -> c.spec.liveness === nothing, filter(c -> c.spec.name != "gateway", sup.children))
        @test occursin("worker liveness checks: off, no worker has an HTTP listener to check (set metrics_base_port)", String(take!(sink)))

        # The address follows the worker's bind: a specific address is checked there, `::` on ::1, and
        # the inherited INFERENCE_SERVER_ENDPOINTS_* variables win, as they do in the worker.
        host(h; env = Dict{String, String}()) =
            build(_node_yaml(dir; extra = "liveness: {}"), merge(Dict("REACTANT_GPUS" => "2", "INFERENCE_SERVER_ENDPOINTS_HOST" => h), env); tag = "h").children[1].spec.liveness
        @test host("10.1.2.3").ip == IPv4("10.1.2.3")
        @test host("::").ip == IPv6("::1")
        @test host("0.0.0.0"; env = Dict("INFERENCE_SERVER_ENDPOINTS_METRICS_PORT" => "9500")).port == 9500

        # Settings that make no sense fail at startup rather than at the first check.
        for bad in (
                "liveness: {interval_seconds: 0}", "liveness: {timeout_seconds: -1}", "liveness: {failures: 1.5}",
                "liveness: {failures: -1}", "liveness: {dispatch_stall_seconds: -5}", "liveness: {interval: 5}",
                "liveness: {failures: true}", "liveness: often",
            )
            @test_throws ReactantServerCore.ConfigError build(_node_yaml(dir; extra = bad), Dict("REACTANT_GPUS" => "1"); tag = "bad")
        end
        @test_throws ReactantServerCore.ConfigError build(
            _node_yaml(dir), Dict("REACTANT_GPUS" => "1", "REACTANT_LIVENESS_TIMEOUT_SECONDS" => "soon"); tag = "bad"
        )
    end
end

@testset "build_supervisor: GPU-reclaim wait before worker restarts" begin
    mktempdir() do dir
        bin = mkpath(joinpath(dir, "bin"))
        smi = joinpath(bin, "nvidia-smi")
        write(smi, "#!/bin/sh\n")
        chmod(smi, 0o755)
        built(env; yaml = _node_yaml(dir)) = RSN.build_supervisor(
            yaml; env = env, sink = IOBuffer(), workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        # Each GPU worker waits on its own GPU, with the image's startup-gate variables and defaults;
        # nvidia-smi is looked up on the PATH the supervisor was given. The gateway has no GPU.
        sup = built(Dict("REACTANT_GPUS" => "2", "PATH" => bin))
        w0, w1, gw = (c.spec for c in sup.children)
        @test (w0.reclaim.device, w1.reclaim.device) == ("0", "1")
        @test (w0.reclaim.smi, w0.reclaim.timeout, w0.reclaim.max_used_percent, w0.reclaim.smi_timeout) == (smi, 30.0, 15.0, 30.0)
        @test gw.reclaim === nothing
        env = Dict(
            "REACTANT_GPUS" => "GPU-aaaa", "PATH" => bin, RSN.GPU_RECLAIM_TIMEOUT_ENV => "45",
            RSN.GPU_RECLAIM_MAX_USED_ENV => "20", RSN.NVIDIA_SMI_TIMEOUT_ENV => "5"
        )
        r = only(built(env).children).spec.reclaim
        @test (r.device, r.timeout, r.max_used_percent, r.smi_timeout) == ("GPU-aaaa", 45.0, 20.0, 5.0)
        # No wait: the timeout set to 0, no nvidia-smi on the PATH, or a CPU node.
        @test only(built(Dict("REACTANT_GPUS" => "1", "PATH" => bin, RSN.GPU_RECLAIM_TIMEOUT_ENV => "0")).children).spec.reclaim === nothing
        @test only(built(Dict("REACTANT_GPUS" => "1")).children).spec.reclaim === nothing
        cpu = joinpath(dir, "cpu.yaml")
        write(cpu, "model_repo: /repo\nbase_port: 8080\nglobal:\n  runtime:\n    backend: cpu\n")
        @test all(c -> c.spec.reclaim === nothing, built(Dict("REACTANT_GPUS" => "0", "PATH" => bin); yaml = cpu).children)
        # Values that make no sense fail at startup.
        for (var, v) in ((RSN.GPU_RECLAIM_TIMEOUT_ENV, "-1"), (RSN.GPU_RECLAIM_TIMEOUT_ENV, "soon"), (RSN.GPU_RECLAIM_MAX_USED_ENV, "150"))
            @test_throws ReactantServerCore.ConfigError built(Dict("REACTANT_GPUS" => "1", "PATH" => bin, var => v))
        end
    end
end

@testset "build_supervisor: the embedded gateway's liveness check" begin
    mktempdir() do dir
        IPv4 = RSN.Sockets.IPv4
        gw(sup) = only(c for c in sup.children if c.spec.name == "gateway").spec
        built(env; kw...) = RSN.build_supervisor(
            _node_yaml(dir; extra = "liveness: {interval_seconds: 4, failures: 2}"); env = env, sink = IOBuffer(),
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run"), kw...
        )
        # Before, the gateway had no check at all, so a hung gateway was never restarted. It is
        # checked on its admin listener (default 0.0.0.0:8002, so loopback), with the workers' settings.
        sink = IOBuffer()
        sup = RSN.build_supervisor(
            _node_yaml(dir; extra = "liveness: {interval_seconds: 4, failures: 2}");
            env = Dict("REACTANT_GPUS" => "2"), sink = sink, workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "run")
        )
        chk = gw(sup).liveness
        @test (chk.ip, chk.port, chk.interval, chk.timeout, chk.failures) == (IPv4("127.0.0.1"), 8002, 4.0, 10.0, 2)
        @test occursin("gateway liveness check: /healthz on 127.0.0.1:8002 once its /readyz has answered", String(take!(sink)))
        # Its listen address as the gateway resolves it: the environment over a mounted gateway.yml.
        @test gw(built(Dict("REACTANT_GPUS" => "2", "REACTANT_GATEWAY_LISTEN_METRICS" => "10.0.0.5:9002"))).liveness.ip == IPv4("10.0.0.5")
        mounted = joinpath(dir, "gateway.yml")
        write(mounted, "listen:\n  metrics: \"127.0.0.1:7002\"\nendpoints: []\n")
        @test gw(built(Dict("REACTANT_GPUS" => "2"); gateway_path = mounted)).liveness.port == 7002
        @test gw(built(Dict("REACTANT_GPUS" => "2", "REACTANT_GATEWAY_LISTEN_METRICS" => "[::]:9003"))).liveness.port == 9003
        # The gateway role checks it too; `failures: 0` turns it off with the workers' checks.
        sup = RSN.build_supervisor(
            _node_yaml(dir); role = :gateway, env = Dict{String, String}(), sink = IOBuffer(),
            workspace_root = STUB_ROOT, runtime_dir = joinpath(dir, "gwrole")
        )
        @test only(sup.children).spec.liveness.port == 8002
        @test gw(built(Dict("REACTANT_GPUS" => "2", "REACTANT_LIVENESS_FAILURES" => "0"))).liveness === nothing
        @test gw(built(Dict("REACTANT_GPUS" => "2", "REACTANT_GATEWAY_LISTEN_METRICS" => "nonsense"))).liveness === nothing
    end
end
