# Supervisor-side node preparation: the `gpus:` key and materialize_node!, which synthesizes the
# workers list (or assigns devices to an explicit one) for single-container multi-GPU deployment.

@testset "node gpus key" begin
    @test ReactantServer.node_gpus(Dict{String, Any}()) === :auto
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => "auto")) === :auto
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => "AUTO")) === :auto
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => 3)) == 3
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => 0)) == 0
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => [0, 2])) == ["0", "2"]
    @test ReactantServer.node_gpus(Dict{String, Any}("gpus" => ["GPU-aaaa", "GPU-bbbb"])) ==
        ["GPU-aaaa", "GPU-bbbb"]
    @test_throws ReactantServer.ConfigError ReactantServer.node_gpus(Dict{String, Any}("gpus" => -1))
    @test_throws ReactantServer.ConfigError ReactantServer.node_gpus(Dict{String, Any}("gpus" => Any[]))
    @test_throws ReactantServer.ConfigError ReactantServer.node_gpus(Dict{String, Any}("gpus" => "two"))
    @test_throws ReactantServer.ConfigError ReactantServer.node_gpus(Dict{String, Any}("gpus" => Dict()))
end

@testset "materialize_node!: synthesized workers" begin
    node = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "metrics_base_port" => 9100
    )
    sel = ReactantServer.materialize_node!(node, ["0", "1", "2"])
    @test sel == ["0", "1", "2"]
    @test ReactantServer.worker_names(node) == ["worker0", "worker1", "worker2"]
    ReactantServer.validate_node(node)
    raw1 = ReactantServer.worker_raw_config(node, "worker1")
    @test raw1["endpoints"]["port"] == 8081
    @test raw1["endpoints"]["metrics_port"] == 9101
    @test raw1["runtime"]["device_ordinal"] == 0

    # CPU node: no devices, worker count from cpu_workers.
    cpu = Dict{String, Any}("model_repo" => "/repo", "base_port" => 8080)
    sel = ReactantServer.materialize_node!(cpu, String[]; cpu_workers = 2)
    @test sel == [nothing, nothing]
    @test ReactantServer.worker_names(cpu) == ["worker0", "worker1"]

    # UUID selectors pass through verbatim.
    uuid = Dict{String, Any}("model_repo" => "/repo", "base_port" => 8080)
    sel = ReactantServer.materialize_node!(uuid, ["GPU-aaaa"])
    @test sel == ["GPU-aaaa"]
    @test ReactantServer.worker_names(uuid) == ["worker0"]
end

@testset "materialize_node!: explicit workers win" begin
    node = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[
            Dict{String, Any}("name" => "a"),
            Dict{String, Any}("name" => "b"),
        ]
    )
    sel = ReactantServer.materialize_node!(node, ["0", "1", "2"])
    @test sel == ["0", "1"]                       # positional; the extra device is unused
    @test ReactantServer.worker_names(node) == ["a", "b"]

    # A gpu: key selects among the visible devices and is consumed so the child resolves
    # device ordinal 0 behind its single-device CUDA_VISIBLE_DEVICES.
    pinned = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[
            Dict{String, Any}("name" => "a", "gpu" => 1),
            Dict{String, Any}("name" => "b", "gpu" => 0),
        ]
    )
    sel = ReactantServer.materialize_node!(pinned, ["0", "1"])
    @test sel == ["1", "0"]
    @test all(w -> !haskey(w, "gpu"), pinned["workers"])
    raw = ReactantServer.worker_raw_config(pinned, "a")
    @test raw["runtime"]["device_ordinal"] == 0

    # More workers than devices.
    over = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[
            Dict{String, Any}("name" => "a"),
            Dict{String, Any}("name" => "b"),
        ]
    )
    @test_throws ReactantServer.ConfigError ReactantServer.materialize_node!(over, ["0"])

    # gpu: key out of range.
    oob = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[Dict{String, Any}("name" => "a", "gpu" => 2)]
    )
    @test_throws ReactantServer.ConfigError ReactantServer.materialize_node!(oob, ["0", "1"])

    # Double assignment: a pinned worker claims the device another gets positionally.
    dup = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[
            Dict{String, Any}("name" => "a", "gpu" => 1),
            Dict{String, Any}("name" => "b"),
        ]
    )
    @test_throws ReactantServer.ConfigError ReactantServer.materialize_node!(dup, ["0", "1"])

    # Explicit workers on a CPU node: selectors are all nothing.
    cpu = Dict{String, Any}(
        "model_repo" => "/repo", "base_port" => 8080,
        "workers" => Any[Dict{String, Any}("name" => "a")]
    )
    @test ReactantServer.materialize_node!(cpu, String[]) == [nothing]
end
