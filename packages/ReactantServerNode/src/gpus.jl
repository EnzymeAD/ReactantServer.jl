# GPU enumeration for the supervisor, without loading CUDA or Reactant. Inside a container the
# NVIDIA toolkit injects nvidia-smi showing exactly the granted devices (renumbered 0..N-1), so
# that is the primary auto path; /dev/nvidiaN is the fallback when nvidia-smi is unavailable.
# Selectors are strings end to end so GPU UUIDs pass through to CUDA_VISIBLE_DEVICES verbatim.

const _DEVFS_GPU_RE = r"^nvidia([0-9]+)$"

function _expand_count(n::Integer)
    n >= 0 || throw(ConfigError("GPU count must be >= 0, got $n"))
    return String[string(i) for i in 0:(n - 1)]
end

_parse_selector_list(s::AbstractString) =
    String[strip(String(t)) for t in split(s, ',') if !isempty(strip(t))]

# REACTANT_GPUS: a bare integer is a device count; anything else is a comma-separated selector
# list (ordinals or GPU UUIDs).
function _parse_reactant_gpus(s::AbstractString)
    v = strip(s)
    occursin(r"^[0-9]+$", v) && return _expand_count(parse(Int, v))
    sels = _parse_selector_list(v)
    isempty(sels) && throw(ConfigError("REACTANT_GPUS is set but empty; use 0 for a CPU node"))
    return sels
end

_parse_smi_indices(out::AbstractString) =
    String[strip(l) for l in split(out, '\n') if !isempty(strip(l))]

# How long any `nvidia-smi` call the supervisor makes may take: REACTANT_NVIDIA_SMI_TIMEOUT_SECONDS,
# default 30. It answers in well under a second normally and in a few seconds on a cold driver with
# many GPUs; a wedged driver can keep it waiting indefinitely, and it must not hold up the node.
const NVIDIA_SMI_TIMEOUT_ENV = "REACTANT_NVIDIA_SMI_TIMEOUT_SECONDS"
const NVIDIA_SMI_TIMEOUT_SECONDS = 30.0

function nvidia_smi_timeout(env::AbstractDict = ENV)
    v = strip(get(env, NVIDIA_SMI_TIMEOUT_ENV, ""))
    isempty(v) && return NVIDIA_SMI_TIMEOUT_SECONDS
    x = tryparse(Float64, v)
    (x !== nothing && x > 0 && isfinite(x)) ||
        throw(ConfigError("$NVIDIA_SMI_TIMEOUT_ENV must be a positive number of seconds, got $(repr(env[NVIDIA_SMI_TIMEOUT_ENV]))"))
    return x
end

# Run `cmd` with stdin and stderr on /dev/null (so nothing it leaves running can hold a pipe of
# ours) and give up on it after `timeout` seconds, sending it SIGKILL. Returns `(stdout, :ok)`, or
# `(nothing, :failed)` for a nonzero exit or a failure to start, or `(nothing, :timeout)`.
function _read_bounded(cmd::Cmd, timeout::Real)
    out = Pipe()
    p = try
        run(pipeline(cmd; stdin = devnull, stdout = out, stderr = devnull); wait = false)
    catch
        return (nothing, :failed)
    end
    close(out.in)
    reader = @async read(out, String)
    if timedwait(() -> istaskdone(reader) && !process_running(p), Float64(timeout); pollint = 0.02) !== :ok
        try
            kill(p, Base.SIGKILL)
        catch
        end
        close(out)
        return (nothing, :timeout)
    end
    (istaskfailed(reader) || !success(p)) && return (nothing, :failed)
    return (fetch(reader), :ok)
end

function _smi_gpus(; smi = Sys.which("nvidia-smi"), timeout::Real = NVIDIA_SMI_TIMEOUT_SECONDS)
    smi === nothing && return nothing
    out, status = _read_bounded(`$smi --query-gpu=index --format=csv,noheader`, timeout)
    if status === :timeout
        @warn "nvidia-smi did not answer within $(timeout)s; detecting GPUs from /dev instead" timeout_env = NVIDIA_SMI_TIMEOUT_ENV
        return nothing
    end
    out === nothing && return nothing
    idx = _parse_smi_indices(out)
    return isempty(idx) ? nothing : idx
end

function _devfs_gpus(dir::AbstractString)
    isdir(dir) || return nothing
    devs = String[]
    for f in readdir(dir)
        m = match(_DEVFS_GPU_RE, f)   # plain devices only, not nvidiactl/nvidia-uvm/nvidia-modeset
        m === nothing || push!(devs, String(m.captures[1]))
    end
    isempty(devs) && return nothing
    return sort!(devs; by = x -> parse(Int, x))
end

"""
    detect_gpus(env=ENV; node=Dict(), devdir="/dev") -> Vector{String}

Resolve the device selectors to spawn workers for, in precedence order: the `REACTANT_GPUS`
environment variable (count or comma list; `0` means a CPU node), an explicit `gpus:` list or
count in the node file, a `CUDA_VISIBLE_DEVICES` already set on the container (split into one
worker per token), `nvidia-smi` enumeration, and finally `/dev/nvidiaN`. An empty result means
no GPUs; the caller decides whether that is a CPU node or an error.
"""
function detect_gpus(
        env::AbstractDict = ENV; node::AbstractDict = Dict{String, Any}(),
        devdir::AbstractString = "/dev"
    )
    haskey(env, "REACTANT_GPUS") && return _parse_reactant_gpus(env["REACTANT_GPUS"])
    g = node_gpus(node)
    g isa Int && return _expand_count(g)
    g isa Vector{String} && return g
    haskey(env, "CUDA_VISIBLE_DEVICES") && return _parse_selector_list(env["CUDA_VISIBLE_DEVICES"])
    smi = _smi_gpus(; timeout = nvidia_smi_timeout(env))
    smi === nothing || return smi
    dev = _devfs_gpus(devdir)
    dev === nothing || return dev
    return String[]
end

# `name` on the PATH of `env` (the environment the supervisor was given, which may not be ENV), or
# `nothing`.
function _which(name::AbstractString, env::AbstractDict = ENV)
    for dir in split(get(env, "PATH", ""), ':'; keepempty = false)
        p = joinpath(dir, name)
        isfile(p) && Sys.isexecutable(p) && return p
    end
    return nothing
end
