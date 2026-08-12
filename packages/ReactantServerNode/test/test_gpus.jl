# Device detection: pure parsers plus the precedence chain, all with injected env dicts and
# fixture directories (no real nvidia-smi dependence).

@testset "gpu parsers" begin
    @test RSN._parse_smi_indices("0\n1\n2\n") == ["0", "1", "2"]
    @test RSN._parse_smi_indices(" 0 \n\n 1 ") == ["0", "1"]
    @test RSN._parse_smi_indices("") == String[]

    @test RSN._parse_reactant_gpus("3") == ["0", "1", "2"]
    @test RSN._parse_reactant_gpus("0") == String[]
    @test RSN._parse_reactant_gpus("0,2") == ["0", "2"]
    @test RSN._parse_reactant_gpus("GPU-aaaa, GPU-bbbb") == ["GPU-aaaa", "GPU-bbbb"]
    @test_throws ReactantServerCore.ConfigError RSN._parse_reactant_gpus("  ")

    @test RSN._parse_selector_list("0,1") == ["0", "1"]
    @test RSN._parse_selector_list("") == String[]
end

@testset "devfs fallback" begin
    mktempdir() do dir
        for f in ("nvidia0", "nvidia1", "nvidia10", "nvidiactl", "nvidia-uvm", "nvidia-modeset")
            touch(joinpath(dir, f))
        end
        @test RSN._devfs_gpus(dir) == ["0", "1", "10"]
    end
    mktempdir() do dir
        touch(joinpath(dir, "nvidiactl"))
        @test RSN._devfs_gpus(dir) === nothing      # control nodes alone are not GPUs
    end
    @test RSN._devfs_gpus(joinpath(tempdir(), "no-such-dir")) === nothing
end

@testset "detect_gpus precedence" begin
    node_list = Dict{String, Any}("gpus" => [0, 1])

    # REACTANT_GPUS beats everything, including the node file and CUDA_VISIBLE_DEVICES.
    env = Dict("REACTANT_GPUS" => "1", "CUDA_VISIBLE_DEVICES" => "0,1,2")
    @test RSN.detect_gpus(env; node = node_list) == ["0"]
    @test RSN.detect_gpus(Dict("REACTANT_GPUS" => "0"); node = node_list) == String[]

    # The node file's gpus key beats CUDA_VISIBLE_DEVICES.
    env = Dict("CUDA_VISIBLE_DEVICES" => "5")
    @test RSN.detect_gpus(env; node = node_list) == ["0", "1"]
    @test RSN.detect_gpus(env; node = Dict{String, Any}("gpus" => 3)) == ["0", "1", "2"]

    # gpus: auto falls through to CUDA_VISIBLE_DEVICES.
    @test RSN.detect_gpus(env; node = Dict{String, Any}("gpus" => "auto")) == ["5"]
    @test RSN.detect_gpus(Dict("CUDA_VISIBLE_DEVICES" => "GPU-aaaa")) == ["GPU-aaaa"]
    @test RSN.detect_gpus(Dict("CUDA_VISIBLE_DEVICES" => "")) == String[]
end
