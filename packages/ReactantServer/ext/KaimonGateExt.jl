# KaimonGateExt.jl
#
# The KaimonGate extension: registers `rserver_*` GateTools with the running Kaimon gate so an
# agent can bring a worker up in the very session where the server code is loaded, ask what it is
# serving, and shut it down again. It is a dev tool, not a deployment surface: nothing in `src/`
# knows it exists, the tools drive the same `serve` / `stop!` entry points a REPL session drives,
# and the deployment story is unchanged (ReactantServerNode, one subprocess per GPU).
#
# ── Why the server starts on a background task ────────────────────────────────────────
#
# Kaimon's agent-side tool calls carry a hard deadline (its gate client defaults to five minutes
# and `tool_progress` does not extend it). Bringing a worker up loads every selected bundle and
# compiles it, which is minutes on a real model repository, so `rserver_start` validates its
# arguments synchronously (a bad argument is an immediate error, not a poll away), then starts the
# server on a background task and returns a handle. The agent polls `rserver_status`, which is a
# fast call.
#
# ── One XLA client per process ────────────────────────────────────────────────────────
#
# Reactant's XLA client is a process-global singleton, built on the first `client("cuda")` call and
# cached from then on (`Reactant.XLA.global_backend_state`). The BFC knobs the worker sets right
# before that first call (`runtime.mem_fraction`, `runtime.preallocate`) therefore bite exactly
# once per session: a second GPU server in the same session inherits the arena the first one
# created, whatever it asks for. Rather than let that pass silently, the first CUDA start records
# what the session committed to and a later start that disagrees is refused with the reason. Two
# servers on two devices at the same settings are fine, and so is stop-then-start on one device.
#
# Which physical card a device index refers to is NOT this extension's business. `device` is the
# ordinal among the devices visible to the process, the same numbering CUDA uses, so the process
# owns `CUDA_VISIBLE_DEVICES` (or whatever convention it prefers) and this tool selects within it.
#
# ── Registration ──────────────────────────────────────────────────────────────────────
#
# KaimonGate exposes one tool-registration verb, `serve(tools = [...])`. On a running gate it
# merges by name (additive since KaimonGate 1.3), so registering here cannot drop another
# registrant's tools. What it cannot do is register before the gate binds, and the gate normally
# binds after the session's `using` lines run, so the extension registers on load and then keeps
# retrying for a short window. `KaimonGateExt.reinstall_kaimon_tools()` is the manual hammer: it
# registers unconditionally, starting a gate if none is running, exactly as a bare `serve()` would.

module KaimonGateExt

import ReactantServer
import KaimonGate

# ── The server registry ───────────────────────────────────────────────────────────────
#
# One entry per tool-started server. Process-local and lock-guarded: the bring-up task writes it
# from one thread while the status tools read it from another. Finished entries (failed or stopped)
# are trimmed oldest-first past a small cap so a long session cannot accumulate them; a starting or
# running server is never trimmed.

mutable struct ServerState
    id::String
    status::Symbol                      # :starting, :running, :failed, :stopped
    accelerator::String                 # "cuda" or "cpu"
    device::Int
    model_dir::String
    cfg::Any                            # ServerConfig
    handle::Any                         # RunningServer once up, nothing before and after
    error::Union{String, Nothing}
    started_at::Float64
    ready_at::Union{Float64, Nothing}
    stopped_at::Union{Float64, Nothing}
end

const SERVERS = Dict{String, ServerState}()
const SERVERS_LOCK = ReentrantLock()
const MAX_FINISHED_SERVERS = 10

# What the session's XLA client was built with, recorded on the first CUDA start. See the
# one-client-per-process note at the top of the file.
const CLIENT_COMMIT = Ref{Any}(nothing)

_next_id() = bytes2hex(rand(UInt8, 4))

_is_live(s::ServerState) = s.status === :starting || s.status === :running

function _trim_registry_locked!()
    finished = [s for s in values(SERVERS) if !_is_live(s)]
    n = length(finished) - MAX_FINISHED_SERVERS
    n <= 0 && return nothing
    sort!(finished; by = s -> s.started_at)
    for s in finished[1:n]
        delete!(SERVERS, s.id)
    end
    return nothing
end

# Resolve the server a tool call refers to. An explicit id always wins; with none, a single live
# server is the obvious target and anything else is an ambiguity the agent has to resolve.
function _resolve(id::Union{AbstractString, Nothing})
    return lock(SERVERS_LOCK) do
        if id !== nothing
            s = get(SERVERS, String(id), nothing)
            s === nothing && error(
                "ReactantServer KaimonGateExt: no server with id `$(id)` in this session. \
                 `rserver_status()` lists what there is."
            )
            return s
        end
        live = [s for s in values(SERVERS) if _is_live(s)]
        isempty(live) && error(
            "ReactantServer KaimonGateExt: no server is running in this session. Start one with \
             `rserver_start(model_dir=\"...\")`."
        )
        length(live) == 1 || error(
            "ReactantServer KaimonGateExt: $(length(live)) servers are running \
             ($(join(sort([s.id for s in live]), ", "))); pass `id`."
        )
        return only(live)
    end
end

# ── Argument handling ─────────────────────────────────────────────────────────────────

# The tool speaks "gpu" because that is what an agent writes; the config speaks "cuda". CPU is
# accepted so the same tool covers a machine with no card (and the test suite).
function _normalize_accelerator(s::AbstractString)
    a = lowercase(strip(s))
    (a == "gpu" || a == "cuda") && return "cuda"
    a == "cpu" && return "cpu"
    return error(
        "ReactantServer KaimonGateExt: `accelerator` must be \"gpu\" (equivalently \"cuda\") or \
         \"cpu\", got \"$(s)\"."
    )
end

# A model list arrives as a JSON array, but an agent that packs several names into one element
# ("a,b") means the same thing, so split on commas and drop the empties either way.
function _model_names(models::Union{Vector{String}, Nothing})
    models === nothing && return String[]
    out = String[]
    for m in models, part in split(m, ',')
        p = strip(part)
        isempty(p) || push!(out, String(p))
    end
    return out
end

# Build the worker config through the same `build_config` the YAML path uses, so a tool-started
# server takes exactly the file-started defaults (preallocation, the on-demand weight cache, the
# gRPC message caps) and the same validation rejects a bad argument here.
function _build_config(;
        model_dir::AbstractString, models::Vector{String}, accelerator::AbstractString,
        device::Int, host::AbstractString, port::Int, metrics_port::Int,
        cache_dir::Union{AbstractString, Nothing}, mem_fraction::Union{Float64, Nothing},
        weight_cache_fraction::Union{Float64, Nothing}, numerics::Union{AbstractString, Nothing},
        poll_seconds::Union{Float64, Nothing}
    )
    dir = abspath(expanduser(String(model_dir)))
    poll = something(poll_seconds, 0.0)
    poll >= 0 || error("ReactantServer KaimonGateExt: `poll_seconds` must be non-negative.")

    runtime = Dict{String, Any}("backend" => accelerator, "device_ordinal" => device)
    mem_fraction === nothing || (runtime["mem_fraction"] = mem_fraction)
    weight_cache_fraction === nothing ||
        (runtime["weight_cache_fraction"] = weight_cache_fraction)
    numerics === nothing || (runtime["numerics"] = lowercase(strip(String(numerics))))

    raw = Dict{String, Any}(
        "model_dirs" => [dir],
        # Static unless a poll interval is asked for: a background watcher is a surprise an agent
        # should opt into, and `dynamic` is invalid without a positive interval anyway.
        "model_control_mode" => poll > 0 ? "dynamic" : "static",
        "model_poll_seconds" => poll,
        "runtime" => runtime,
        "endpoints" => Dict{String, Any}(
            "host" => String(host), "port" => port, "metrics_port" => metrics_port,
        ),
    )
    isempty(models) || (raw["models_include"] = models)
    cache_dir === nothing || (raw["cache_dir"] = abspath(expanduser(String(cache_dir))))

    return ReactantServer.validate_config(ReactantServer.build_config(raw))
end

# Ports are validated by the config, but a collision with a server this session already owns is
# worth catching before a two-minute compile rather than as a bind error at the end of one.
function _check_ports_locked(cfg)
    for s in values(SERVERS)
        _is_live(s) || continue
        used = [s.cfg.endpoints.port]
        s.cfg.endpoints.metrics_port > 0 && push!(used, s.cfg.endpoints.metrics_port)
        for p in (cfg.endpoints.port, cfg.endpoints.metrics_port)
            p == 0 && continue
            p in used && error(
                "ReactantServer KaimonGateExt: port $(p) is already used by server $(s.id) \
                 (gRPC $(s.cfg.endpoints.port), metrics $(s.cfg.endpoints.metrics_port)). \
                 Pick another port."
            )
        end
    end
    return nothing
end

# The one-client-per-process check. Only the knobs that are read at client-creation time matter
# here; everything else in the config is per-server and can differ freely.
function _check_client_commit_locked(cfg)
    cfg.runtime.backend == ReactantServer.CUDA_BACKEND || return nothing
    want = (mem_fraction = cfg.runtime.mem_fraction, preallocate = cfg.runtime.preallocate)
    have = CLIENT_COMMIT[]
    if have !== nothing && have != want
        error(
            "ReactantServer KaimonGateExt: this session's XLA client was already created with \
             mem_fraction=$(have.mem_fraction), preallocate=$(have.preallocate), and the client \
             is process-global: a second server cannot change the BFC arena, it can only inherit \
             it. Either start this server with those settings, or restart the session. \
             (Requested mem_fraction=$(want.mem_fraction), preallocate=$(want.preallocate).)"
        )
    end
    CLIENT_COMMIT[] = want
    return nothing
end

# ── Bring-up ──────────────────────────────────────────────────────────────────────────

# `Threads.@spawn` rather than `@async`: an XLA compile is a long foreign call that does not
# yield, so running bring-up on the gate's own thread would freeze the session for the duration.
# The scheduler picks its own pool for the dispatch loop (`:interactive` when the process has one),
# so which thread starts the server does not change where it runs.
function _launch!(state::ServerState)
    Threads.@spawn begin
        try
            handle = ReactantServer.serve(state.cfg; blocking = false)
            lock(SERVERS_LOCK) do
                state.handle = handle
                state.status = :running
                state.ready_at = time()
            end
        catch e
            lock(SERVERS_LOCK) do
                state.status = :failed
                state.error = sprint(showerror, e, catch_backtrace())
                state.stopped_at = time()
                _trim_registry_locked!()
            end
        end
    end
    return state.id
end

# ── Reporting ─────────────────────────────────────────────────────────────────────────

_dur(from::Float64, to::Union{Float64, Nothing}) =
    string(round(Int, something(to, time()) - from), "s")

function _model_summary(state::ServerState)
    state.handle === nothing && return (loaded = 0, ready = 0, metas = 0, names = String[])
    reg = state.handle.registry
    sched = state.handle.scheduler
    # The dispatch loop guards registry mutation with this condition, and the watcher loads models
    # through it, so read the maps under it and format outside.
    return lock(sched.cond) do
        entries = collect(values(reg.by_name))
        return (
            loaded = length(entries),
            ready = count(e -> e.executable !== nothing, entries),
            metas = length(reg.meta),
            names = ReactantServer.model_names(reg),
        )
    end
end

function _one_line(state::ServerState)
    dev = state.accelerator == "cuda" ? "gpu:$(state.device)" : "cpu"
    m = _model_summary(state)
    age = _dur(state.started_at, state.status === :running ? nothing : state.stopped_at)
    parts = [
        "$(state.id)  $(state.status)  $(dev)  port=$(state.cfg.endpoints.port)",
        "models=$(m.ready)/$(m.loaded)$(m.metas > 0 ? " (+$(m.metas) meta)" : "")",
        "up=$(age)",
    ]
    state.status === :failed && push!(parts, "error=" * first(split(something(state.error, ""), '\n')))
    return join(parts, "  ")
end

function _detail(state::ServerState)
    cfg = state.cfg
    io = IOBuffer()
    println(io, "server $(state.id)  status=$(state.status)")
    println(
        io, "  accelerator: $(state.accelerator)",
        state.accelerator == "cuda" ? " (device $(state.device) of the visible devices)" : ""
    )
    println(io, "  endpoints:   grpc $(cfg.endpoints.host):$(cfg.endpoints.port)", cfg.endpoints.metrics_port > 0 ? ", metrics $(cfg.endpoints.host):$(cfg.endpoints.metrics_port)" : ", metrics off")
    println(io, "  model dir:   $(state.model_dir)")
    println(
        io, "  models:      ",
        isempty(cfg.models_include) ? "all bundles in the directory" : join(cfg.models_include, ", ")
    )
    println(
        io, "  control:     $(cfg.model_control_mode)",
        cfg.model_control_mode == ReactantServer.DYNAMIC ?
            " (re-scanning every $(cfg.model_poll_seconds)s)" : " (no directory watcher)"
    )
    if state.status === :starting
        println(io, "  loading and compiling bundles ($(_dur(state.started_at, nothing)) so far)")
    end
    if state.status === :running
        m = _model_summary(state)
        println(io, "  serving:     $(m.ready)/$(m.loaded) compiled$(m.metas > 0 ? ", $(m.metas) meta" : "")")
        isempty(m.names) || println(io, "  loaded:      $(join(m.names, ", "))")
        println(io, "  uptime:      $(_dur(something(state.ready_at, state.started_at), nothing))")
        sched = state.handle.scheduler
        println(
            io, "  memory:      ", ReactantServer.memory_report(
                sched.backend, sched.pool;
                registry = state.handle.registry, weight_cache = sched.weight_cache
            )
        )
    end
    state.status === :stopped &&
        println(io, "  stopped after $(_dur(state.started_at, state.stopped_at))")
    state.status === :failed && println(io, "  error: $(state.error)")
    return String(take!(io))
end

# One model's line for `rserver_models`. The client specs are the agent-facing view (what an infer
# request carries); the executable specs differ whenever a bundle preprocesses on the worker.
function _model_line(state::ServerState, name::AbstractString)
    reg = state.handle.registry
    meta = ReactantServer.get_meta(reg, name)
    if meta !== nothing
        io = IOBuffer()
        println(io, "$(name)  [meta]")
        println(io, "  calls:   $(isempty(meta.calls) ? "(none, pure Julia)" : join(meta.calls, ", "))")
        println(io, "  inputs:  $(ReactantServer._format_specs(ReactantServer.client_input_spec(meta.manifest)))")
        println(io, "  outputs: $(ReactantServer._format_specs(ReactantServer.client_output_spec(meta.manifest)))")
        return String(take!(io))
    end
    entry = ReactantServer.get_model(reg, name)
    entry === nothing && return "$(name)  [not registered]"
    io = IOBuffer()
    model = entry.executable
    println(io, "$(name)  [$(model === nothing ? "not compiled" : "ready")]")
    if model !== nothing
        println(io, "  batches: $(ReactantServer._compiled_sizes(model))")
        println(
            io, "  weights: $(model.state), $(Base.format_bytes(model.nbytes)) device-resident"
        )
    end
    println(io, "  inputs:  $(ReactantServer._format_specs(ReactantServer.client_input_spec(entry.manifest)))")
    println(io, "  outputs: $(ReactantServer._format_specs(ReactantServer.client_output_spec(entry.manifest)))")
    return String(take!(io))
end

# ── The tools ─────────────────────────────────────────────────────────────────────────
#
# Each handler is a module-level named function with a docstring; KaimonGate reflects the signature
# into the MCP schema and the docstring into the tool description, so the agent sees exactly the
# contract below.

"""
    rserver_start(model_dir; models, accelerator, device, port, metrics_port, host, cache_dir,
                  mem_fraction, weight_cache_fraction, numerics, poll_seconds) -> String

Start a ReactantServer worker in this session and return its id immediately.

`model_dir` is the directory the bundles are served out of (one bundle per subdirectory, the
directory name being the model name). `models` is an optional allowlist of bundle names, e.g.
`models=["detector", "classifier"]`; omit it to serve every bundle in the directory.

`accelerator` is `"gpu"` (the default, equivalently `"cuda"`) or `"cpu"`, and `device` is the
GPU index **among the devices visible to this process**, the same numbering CUDA uses, so
`CUDA_VISIBLE_DEVICES` (or whatever convention the process uses) decides which physical card
index 0 is. Set it before the session starts; this tool selects within what is already visible.

The call returns before the server is up: bundles are loaded and compiled on a background task,
which takes minutes on a real repository. Poll `rserver_status(id=...)` until the status is
`running`, then `rserver_models(id=...)` for what it serves and `rserver_stop(id=...)` to shut
it down. A configuration error is raised by this call, synchronously; anything that goes wrong
during bring-up lands in `rserver_status`.

`poll_seconds` turns on the directory watcher, so a bundle dropped into `model_dir` later is
picked up (and a removed one is unloaded). Omit it and the model set is fixed at startup.

The remaining keywords take the same defaults a node file takes: `mem_fraction` (0.9 of the
card for the BFC arena), `weight_cache_fraction` (1.0, the self-sizing on-demand weight cache),
`numerics` (`"auto"`, or `"f32"` to pin full f32 precision), `cache_dir`, `host`
(127.0.0.1), `port` (8080) and `metrics_port` (0, meaning no Prometheus endpoint).

One caveat worth knowing before starting a second GPU server in one session: Reactant's XLA
client is process-global and is created by the first server, so `mem_fraction` and the
preallocated arena are fixed for the whole session. A later start that disagrees is refused
rather than silently ignored; two servers on two devices at the same settings are fine.

```julia
rserver_start(model_dir="/var/lib/reactantserver/models", device=1, port=8080)
rserver_start(model_dir="./bundles", models=["scale4"], accelerator="cpu", port=9000)
```
"""
function rserver_start(
        model_dir::String;
        models::Union{Vector{String}, Nothing} = nothing,
        accelerator::String = "gpu",
        device::Int = 0,
        port::Int = 8080,
        metrics_port::Int = 0,
        host::String = "127.0.0.1",
        cache_dir::Union{String, Nothing} = nothing,
        mem_fraction::Union{Float64, Nothing} = nothing,
        weight_cache_fraction::Union{Float64, Nothing} = nothing,
        numerics::Union{String, Nothing} = nothing,
        poll_seconds::Union{Float64, Nothing} = nothing,
    )
    accel = _normalize_accelerator(accelerator)
    device >= 0 || error("ReactantServer KaimonGateExt: `device` must be non-negative.")
    accel == "cpu" && device != 0 && error(
        "ReactantServer KaimonGateExt: `device` is a GPU index; the CPU backend has one device, \
         so leave it at 0."
    )
    cfg = _build_config(;
        model_dir, models = _model_names(models), accelerator = accel, device, host, port,
        metrics_port, cache_dir, mem_fraction, weight_cache_fraction, numerics, poll_seconds,
    )

    state = lock(SERVERS_LOCK) do
        _check_ports_locked(cfg)
        _check_client_commit_locked(cfg)
        s = ServerState(
            _next_id(), :starting, accel, device, only(cfg.model_dirs), cfg, nothing,
            nothing, time(), nothing, nothing
        )
        SERVERS[s.id] = s
        _trim_registry_locked!()
        return s
    end
    _launch!(state)

    return "starting server $(state.id) on $(accel == "cuda" ? "gpu:$(device)" : "cpu"), \
            gRPC $(cfg.endpoints.host):$(cfg.endpoints.port), models from \
            $(state.model_dir). Loading and compiling bundles now; poll \
            `rserver_status(id=\"$(state.id)\")` until it reports running."
end

"""
    rserver_status(; id) -> String

Report a server started in this session, or list all of them when `id` is omitted and more than
one exists.

The detail view names the accelerator and device, the endpoints, the model directory and
allowlist, whether a directory watcher is running, how many bundles are compiled, and a memory
snapshot (device allocator, resident weight bytes, on-demand weight-cache budget). A server that
failed to come up reports its error and backtrace here, which is where a bad bundle or an
occupied port surfaces.
"""
function rserver_status(; id::Union{String, Nothing} = nothing)
    if id === nothing
        all = lock(SERVERS_LOCK) do
            sort(collect(values(SERVERS)); by = s -> s.started_at)
        end
        isempty(all) && return "no servers in this session; start one with \
            `rserver_start(model_dir=\"...\")`."
        live = [s for s in all if _is_live(s)]
        length(live) == 1 && return _detail(only(live))
        return join(map(_one_line, all), "\n")
    end
    return _detail(_resolve(id))
end

"""
    rserver_models(; id, name) -> String

List what a running server serves: each model's readiness, its compiled batch sizes (and
input-shape variants), its weight residency and device footprint, and the input and output
tensors an infer request carries. `name` restricts the listing to one model. `id` names the
server when this session has more than one.

The tensor specs shown are the CLIENT view, which is what a request sends and receives; a bundle
that preprocesses on the worker has different executable specs.
"""
function rserver_models(; id::Union{String, Nothing} = nothing, name::Union{String, Nothing} = nothing)
    state = _resolve(id)
    state.status === :running || return "server $(state.id) is $(state.status); \
        `rserver_status(id=\"$(state.id)\")` has the detail."
    sched = state.handle.scheduler
    names = lock(sched.cond) do
        ReactantServer.model_names(state.handle.registry)
    end
    if name !== nothing
        name in names || return "server $(state.id) has no model `$(name)`. It serves: \
            $(isempty(names) ? "(nothing)" : join(names, ", "))."
        names = [name]
    end
    isempty(names) && return "server $(state.id) has no models loaded from $(state.model_dir)."
    return join((_model_line(state, n) for n in names), "\n")
end

"""
    rserver_stop(; id) -> String

Shut down a server started in this session: the directory watcher, the metrics endpoint, the
gRPC server, the dispatch loop, and any shared-memory regions it registered. `id` names the
server when this session has more than one.

A server that is still starting cannot be stopped: bring-up is inside a compile that does not
yield. Wait for `rserver_status` to report running, then stop it.

Stopping does not release the session's XLA client, so a later start on the same device reuses
the arena the first server created, which is why it is also the fast way to reload a changed
bundle set.
"""
function rserver_stop(; id::Union{String, Nothing} = nothing)
    state = _resolve(id)
    state.status === :starting && return "server $(state.id) is still starting (bundles are \
        compiling); wait for `rserver_status(id=\"$(state.id)\")` to report running, then stop it."
    state.status === :running || return "server $(state.id) is already $(state.status)."
    handle = state.handle
    try
        ReactantServer.stop!(handle)
    catch e
        lock(SERVERS_LOCK) do
            state.status = :failed
            state.error = sprint(showerror, e, catch_backtrace())
            state.stopped_at = time()
        end
        return "stopping server $(state.id) failed: $(sprint(showerror, e))"
    end
    lock(SERVERS_LOCK) do
        state.status = :stopped
        state.handle = nothing
        state.stopped_at = time()
        _trim_registry_locked!()
    end
    return "stopped server $(state.id) (port $(state.cfg.endpoints.port)) after \
            $(_dur(state.started_at, state.stopped_at))."
end

# ── Registration ──────────────────────────────────────────────────────────────────────

function _build_tools()
    return KaimonGate.GateTool[
        KaimonGate.GateTool("rserver_start", rserver_start),
        KaimonGate.GateTool("rserver_status", rserver_status),
        KaimonGate.GateTool("rserver_models", rserver_models),
        KaimonGate.GateTool("rserver_stop", rserver_stop),
    ]
end

# `_RUNNING` is KaimonGate internal state, probed under `isdefined` so a gate that renames it
# degrades to "not running" rather than throwing at extension load.
_gate_running() = try
    isdefined(KaimonGate, :_RUNNING) && KaimonGate._RUNNING[]
catch
    false
end

function _install_tools()
    try
        # `force = true` keeps registration working in non-interactive processes (a gate host is
        # launched with `-e`, and the test suite runs headless), where `serve` returns early on the
        # interactivity check before it reaches the tool-merge branch.
        KaimonGate.serve(force = true, tools = _build_tools())
        return true
    catch e
        @warn "ReactantServer KaimonGateExt: registering the rserver_* tools failed" exception = e
        return false
    end
end

function _install_with_retry()
    _gate_running() && _install_tools() && return true
    # No gate yet, which in a real session means one is about to arrive: the host's preamble runs
    # `using KaimonGate`, then the project's own `using` lines (firing this `__init__`), and calls
    # `KaimonGate.serve(...)` last. So the extension normally loads BEFORE the gate binds, and this
    # loop is the path that actually registers. No `isinteractive()` guard: a gate host is not
    # interactive, which is exactly the case the loop exists for. A process that loads KaimonGate
    # and never serves a gate pays one background task that wakes once a second for 30 seconds.
    Threads.@spawn begin
        for _ in 1:30
            sleep(1.0)
            _gate_running() || continue
            _install_tools() && break
        end
    end
    return false
end

"""
    reinstall_kaimon_tools() -> Bool

Register the `rserver_*` GateTools with the Kaimon gate. Called automatically when the extension
loads; call it again after a manual `KaimonGate.stop()`/`serve()` cycle. Unlike the automatic path
this registers unconditionally: if no gate is running it starts one, exactly as `KaimonGate.serve()`
would.

Reach it from a session as
`Base.get_extension(ReactantServer, :KaimonGateExt).reinstall_kaimon_tools()`. It lives in the
extension module rather than in ReactantServer because a precompiled module cannot create a new
binding in another module, only add methods to existing ones.
"""
reinstall_kaimon_tools() = _install_tools()

# The auto-registration lives in `__init__`, not at module top level: the extension body runs once,
# at precompile time (where no gate exists and the call is a no-op), while `__init__` runs on every
# runtime load, which is when the session's gate is (or is about to be) there.
function __init__()
    _install_with_retry()
    return nothing
end

end # module KaimonGateExt
