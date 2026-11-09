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

# A fake nvidia-smi script in `dir` (named `name`, so "nvidia-smi" is found on a PATH with `dir`).
function _fake_smi(dir, name, body)
    p = joinpath(dir, name)
    write(p, "#!/bin/sh\n" * body)
    chmod(p, 0o755)
    return p
end

@testset "nvidia-smi at startup is bounded" begin
    dir = mktempdir()
    @test RSN._smi_gpus(; smi = _fake_smi(dir, "two", "echo 0\necho 1\n")) == ["0", "1"]
    @test RSN._smi_gpus(; smi = _fake_smi(dir, "failing", "exit 9\n")) === nothing
    # A wedged driver hangs nvidia-smi. Before, GPU detection (and so the whole node) waited for it
    # as long as it hung; now it is given up on after the timeout and detection falls through to
    # /dev (here a directory with no devices).
    _fake_smi(dir, "nvidia-smi", "echo wedged >&2\nexec sleep 60\n")
    t0 = time()
    found = withenv("PATH" => dir * ":" * ENV["PATH"]) do
        @test_logs (:warn, r"nvidia-smi did not answer within 1.0s") RSN.detect_gpus(
            Dict(RSN.NVIDIA_SMI_TIMEOUT_ENV => "1"); devdir = joinpath(dir, "no-dev")
        )
    end
    @test found == String[]
    @test time() - t0 < 15
    @test RSN.nvidia_smi_timeout(Dict{String, String}()) == RSN.NVIDIA_SMI_TIMEOUT_SECONDS == 30.0
    @test RSN.nvidia_smi_timeout(Dict(RSN.NVIDIA_SMI_TIMEOUT_ENV => "2.5")) == 2.5
    for bad in ("0", "-1", "soon")
        @test_throws ReactantServerCore.ConfigError RSN.nvidia_smi_timeout(Dict(RSN.NVIDIA_SMI_TIMEOUT_ENV => bad))
    end
end

# The container entrypoint's GPU-reclaim gate (deploy/runtime/entrypoint.node.sh), run on its own.
const _ENTRYPOINT = joinpath(@__DIR__, "..", "..", "..", "deploy", "runtime", "entrypoint.node.sh")

function _run_reclaim_gate(path_dir, env...)
    fn = match(r"(?ms)^wait_for_gpu_reclaim\(\) \{.*?^\}", read(_ENTRYPOINT, String)).match
    out = IOBuffer()
    t0 = time()
    cmd = addenv(`bash -c $(fn * "\nwait_for_gpu_reclaim\n")`, "PATH" => path_dir * ":" * ENV["PATH"], env...)
    run(pipeline(cmd; stdout = out, stderr = devnull))
    return String(take!(out)), time() - t0
end

@testset "entrypoint: the GPU-reclaim gate's nvidia-smi is bounded" begin
    if !isfile(_ENTRYPOINT) || Sys.which("bash") === nothing || Sys.which("timeout") === nothing
        @info "skipping the entrypoint test: no entrypoint script, bash, or timeout here"
    else
        # Wedged: before, the gate waited for nvidia-smi as long as it hung; now it gives up on it.
        wedged = mktempdir()
        _fake_smi(wedged, "nvidia-smi", "echo wedged >&2\nexec sleep 60\n")
        out, elapsed = _run_reclaim_gate(wedged, "REACTANT_NVIDIA_SMI_TIMEOUT_SECONDS" => "1")
        @test elapsed < 15
        @test occursin("nvidia-smi did not answer within 1s; skipping the GPU-reclaim wait", out)
        # Busy on the first look, clear on the next: the gate still reads the memory and waits.
        busy = mktempdir()
        _fake_smi(
            busy, "nvidia-smi", """
            if [ -f $busy/seen ]; then echo '0, 100, 1000'; else touch $busy/seen; echo '0, 900, 1000'; fi
            """
        )
        out, elapsed = _run_reclaim_gate(busy, "REACTANT_GPU_RECLAIM_TIMEOUT_SECONDS" => "30")
        @test occursin("GPU 0: 900/1000 MiB used (90%); waiting for prior-process reclamation", out)
        @test !occursin("WARNING", out)
        @test 1 < elapsed < 15
    end
end
