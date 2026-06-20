# Dummy-data inference load generator for the single-GPU soak stack.
#
# Connects to the gateway over KServe V2 gRPC and drives sustained, concurrent inference with
# zero-filled inputs across every model, to surface memory leaks, races, and instability. Input
# shapes and dtypes come from each bundle's manifest (manifest_io_spec), read from the mounted
# model repository, so no dependency on the gateway's ModelMetadata RPC (the gateway does not serve
# it). Runs to a fixed duration, then prints a summary and exits nonzero if any request errored.
#
# Env knobs (all optional):
#   LOADGEN_GATEWAY           grpc URL of the gateway          (default grpc://gateway:8001)
#   LOADGEN_METRICS           gateway /metrics URL to scrape   (default http://gateway:8002/metrics)
#   LOADGEN_MODEL_REPO        path to mounted bundles          (default /var/lib/reactantserver/models)
#   LOADGEN_CONCURRENCY       number of concurrent requesters  (default 32)
#   LOADGEN_DURATION_SECONDS  soak duration                    (default 3600)
#   LOADGEN_TRANSPORT         tcp | shm | mixed                (default tcp)
#   LOADGEN_SHM_OUTPUTS       read outputs back through shm    (default true; shm path only)
#   LOADGEN_REPORT_SECONDS    rolling-summary interval         (default 30)
#   LOADGEN_MODELS            comma list to restrict the set   (default: all bundles)

using ReactantServerClient
using ReactantServerCore: load_manifest   # to read meta.calls and skip internal sub-bundles
import gRPCClient                          # for a dedicated gRPCCURL sized to CONCURRENCY
using Base.Threads

# ---- config ----------------------------------------------------------------------------------

env(k, d) = get(ENV, k, d)
const GATEWAY     = env("LOADGEN_GATEWAY", "grpc://gateway:8001")
const METRICS_URL = env("LOADGEN_METRICS", "http://gateway:8002/metrics")
const MODEL_REPO  = env("LOADGEN_MODEL_REPO", "/var/lib/reactantserver/models")
const CONCURRENCY = parse(Int, env("LOADGEN_CONCURRENCY", "32"))
const DURATION    = parse(Float64, env("LOADGEN_DURATION_SECONDS", "3600"))
const TRANSPORT   = Symbol(env("LOADGEN_TRANSPORT", "tcp"))   # :tcp | :shm | :mixed
const SHM_OUTPUTS = lowercase(env("LOADGEN_SHM_OUTPUTS", "true")) in ("1", "true", "yes", "on")
const REPORT_SEC  = parse(Float64, env("LOADGEN_REPORT_SECONDS", "30"))
# Serial warmup before the concurrent soak: one request per model with a long deadline. The first
# request to a model triggers Reactant compilation (often >10s); firing 32 concurrent cold requests
# instead stampedes the gRPCClient libcurl multi-handle into a wedge, so the soak never makes
# progress until manually restarted against warm models. A serial pass (no stampede) + a long
# deadline (tolerates compilation) compiles every model — and warming a meta fans out to compile its
# sub-models — so the soak then hits only warm models. Set LOADGEN_WARMUP=false to skip.
const WARMUP          = lowercase(env("LOADGEN_WARMUP", "true")) != "false"
const WARMUP_DEADLINE = parse(Float64, env("LOADGEN_WARMUP_DEADLINE", "300"))

# KServe wire dtype string -> Julia type. Covers the dtypes the bundles use; extend if needed.
const DTYPE = Dict(
    "BOOL" => Bool,
    "UINT8" => UInt8, "UINT16" => UInt16, "UINT32" => UInt32, "UINT64" => UInt64,
    "INT8" => Int8, "INT16" => Int16, "INT32" => Int32, "INT64" => Int64,
    "FP32" => Float32, "FP64" => Float64,
)

# ---- model discovery + dummy-input synthesis --------------------------------------------------

# One input tensor: its name, Julia element type, per-item Julia column-major dims (batch dropped),
# and whether the manifest declares a batch axis for it. A model with `has_batch=false` is unbatched
# (e.g. a meta core with mixed-rank inputs); we must NOT append a batch row to it, or it arrives one
# rank too large.
struct InputSpec
    name::String
    dtype::DataType
    per_item_dims::Vector{Int}
    has_batch::Bool
end

# Julia column-major wire dims for one input carrying `n` rows: batched inputs get the batch axis
# appended (last, matching the manifest); unbatched inputs are sent at exactly their fixed dims.
_wire_dims(inp::InputSpec, n::Int) = inp.has_batch ? Int[inp.per_item_dims..., n] : copy(inp.per_item_dims)

# One model's everything needed to fire a request: its gateway handle, ALL of its input specs, the
# prebuilt inline InferInputs for the TCP path (immutable, reused across requests), and shm output
# read-back declarations.
struct ModelSpec
    name::String
    model::KServeModel
    inputs::Vector{InputSpec}
    tcp_inputs::Vector{Any}         # one ModelInferRequest.InferInputTensor per input (inline path)
    out_specs::Vector{OutputSpec}   # shm read-back declarations; empty = outputs stay inline
end

# Julia column-major shape from manifest_io_spec has the batch axis (-1) last; drop -1 axes to get
# the per-item dims, and pin any other dynamic axis to 1 (the bundles here have none).
function per_item_dims(shape)
    dims = Int[]
    for d in shape
        d == -1 && continue
        push!(dims, d)
    end
    return dims
end

# Output read-back declarations for the shm path: every output, sized per item (Julia col-major
# shape with the trailing batch axis dropped). Returns [] when disabled or when any output has an
# unmapped dtype or a non-batch dynamic axis; an empty vector keeps that model's outputs inline.
function output_shm_specs(io)
    SHM_OUTPUTS || return OutputSpec[]
    specs = OutputSpec[]
    for oname in io.output_order
        om = io.outputs[oname]
        haskey(DTYPE, om.datatype) || return OutputSpec[]
        dims = copy(om.shape)
        !isempty(dims) && dims[end] == -1 && pop!(dims)    # batch axis is last in col-major order
        any(==(-1), dims) && return OutputSpec[]           # dynamic non-batch axis: keep inline
        push!(specs, OutputSpec(oname, DTYPE[om.datatype], dims))
    end
    return specs
end

# Models that are internal sub-bundles of a meta model (declared in some meta's meta.calls, e.g.
# <m>_stage1 / <m>_stage2 / <m>_core). Clients call the meta, which fans out to these; soaking them
# directly doubles their queue load with zero/garbage inputs and congests the queues the metas wait
# on. Skipped by default; set LOADGEN_SKIP_INTERNAL=false to soak them anyway.
function _internal_submodels(names)
    internal = Set{String}()
    for name in names
        mf = joinpath(MODEL_REPO, name, "manifest.yaml")
        isfile(mf) || continue
        try
            for c in load_manifest(mf).meta_calls
                push!(internal, String(c))
            end
        catch err
            @warn "could not read meta.calls" model = name exception = err
        end
    end
    return internal
end

function discover_models(grpc)
    names = readdir(MODEL_REPO)
    want = get(ENV, "LOADGEN_MODELS", "")
    if !isempty(want)
        sel = Set(strip.(split(want, ",")))
        names = filter(in(sel), names)
    end
    skip_internal = lowercase(env("LOADGEN_SKIP_INTERNAL", "true")) != "false"
    internal = skip_internal ? _internal_submodels(names) : Set{String}()
    isempty(internal) || @info "skipping internal meta sub-bundles" count = length(internal) models = sort(collect(internal))
    specs = ModelSpec[]
    for name in sort(names)
        name in internal && continue
        manifest = joinpath(MODEL_REPO, name, "manifest.yaml")
        isfile(manifest) || continue
        local spec
        try
            io = manifest_io_spec(manifest)
            isempty(io.input_order) && (@warn "skip: no inputs" model=name; continue)
            inputs = InputSpec[]
            unmapped = false
            for tname in io.input_order                      # ALL inputs, not just the first
                tm = io.inputs[tname]
                haskey(DTYPE, tm.datatype) || (unmapped = true; break)
                # batch axis shows up as -1 in the metadata shape; absent => unbatched input
                push!(inputs, InputSpec(tname, DTYPE[tm.datatype],
                                        per_item_dims(tm.shape), any(==(-1), tm.shape)))
            end
            unmapped && (@warn "skip: unmapped dtype" model=name; continue)
            tcp_inputs = Any[InferInput(inp.name, zeros(inp.dtype, _wire_dims(inp, 1)...))
                             for inp in inputs]
            spec = ModelSpec(name, KServeModel(GATEWAY, name; max_batch_size = 1, grpc = grpc),
                             inputs, tcp_inputs, output_shm_specs(io))
        catch err
            @warn "skip: manifest_io_spec failed" model=name exception=err
            continue
        end
        push!(specs, spec)
    end
    return specs
end

# Minimal IO for the shared-memory path: one zero-filled item of every one of a model's inputs.
struct DummyIO <: AbstractInferenceIO
    spec::ModelSpec
end
Base.length(::DummyIO) = 1
# Per-item bytes summed across all inputs (each input's per-item size; the batch row, if any, is 1).
ReactantServerClient.item_input_bytes(io::DummyIO) =
    sum(sizeof(inp.dtype) * prod(_wire_dims(inp, 1)) for inp in io.spec.inputs)
function ReactantServerClient.infer_encode_chunk!(io::DummyIO, r, slot)
    n = length(r)
    ins = InferInput[]
    for inp in io.spec.inputs
        wire = _wire_dims(inp, n)                          # Julia col-major; batch (if any) last
        nbytes = sizeof(inp.dtype) * prod(wire)
        sub = subslot(slot, nbytes)
        fill!(pool_view(sub, UInt8, nbytes), 0x00)
        push!(ins, InferInput(inp.name, sub, wire, inp.dtype))
    end
    return ins
end
ReactantServerClient.infer_decode_chunk!(::DummyIO, r, response) = nothing
# Declaring outputs opts the shm path into shared-memory read-back (explicit-output mode): the
# server writes results into the registered region instead of returning them inline, exercising
# the full shm data plane in both directions. Empty (e.g. LOADGEN_SHM_OUTPUTS=false or a dynamic
# output shape) keeps outputs inline, the safe fallback. The tcp path never consults this.
ReactantServerClient.output_specs(io::DummyIO) = io.spec.out_specs

# ---- counters ---------------------------------------------------------------------------------

const N_OK   = Atomic{Int}(0)
const N_ERR  = Atomic{Int}(0)
const LAT_NS = Atomic{Int}(0)        # cumulative successful-request latency, ns (overall summary)
const LAT_SAMPLES = Int[]            # per-window latency samples, ns; drained each report
const LAT_LOCK = ReentrantLock()     # guards LAT_SAMPLES
const ERR_SAMPLES = String[]
const ERR_LOCK = ReentrantLock()

# Linear-interpolated quantile (type 7, matching Statistics.quantile's default) over an already
# sorted vector; no external dependency. q in [0, 1].
function _quantile_sorted(sorted::Vector{Int}, q::Float64)
    n = length(sorted)
    n == 0 && return 0.0
    n == 1 && return Float64(sorted[1])
    h = (n - 1) * q + 1
    lo = clamp(floor(Int, h), 1, n)
    lo >= n && return Float64(sorted[n])
    return sorted[lo] + (h - lo) * (sorted[lo + 1] - sorted[lo])
end

function record_err(err)
    atomic_add!(N_ERR, 1)
    @lock ERR_LOCK begin
        length(ERR_SAMPLES) < 20 && push!(ERR_SAMPLES, sprint(showerror, err))
    end
end

function fire(spec::ModelSpec, use_shm::Bool)
    t0 = time_ns()
    if use_shm
        infer_async(spec.model, DummyIO(spec))
    else
        infer_sync(spec.model, spec.tcp_inputs)
    end
    dt = Int(time_ns() - t0)
    atomic_add!(N_OK, 1)
    atomic_add!(LAT_NS, dt)
    @lock LAT_LOCK push!(LAT_SAMPLES, dt)        # window sample for the distribution stats
    return nothing
end

# Serial warmup: fire one request per model with a long deadline so first-request compilation
# completes before the concurrent soak begins (see WARMUP comment). Sequential by design — a
# concurrent cold stampede wedges the gRPCClient multi-handle. Failures are logged, not fatal.
function warmup(specs::Vector{ModelSpec})
    println("warmup: priming $(length(specs)) models serially (deadline $(WARMUP_DEADLINE)s) ...")
    t0 = time(); ok = 0
    for spec in specs
        m = KServeModel(GATEWAY, spec.name; max_batch_size = 1, deadline = WARMUP_DEADLINE, grpc = spec.model.grpc)
        try
            infer_sync(m, spec.tcp_inputs)
            ok += 1
        catch err
            @warn "warmup failed (continuing)" model = spec.name exception = err
        end
    end
    println("warmup: $ok/$(length(specs)) models primed in $(round(time() - t0, digits = 1))s")
end

# ---- metrics scrape (via curl; no extra Julia deps) -------------------------------------------

# Sum a Prometheus counter across all of its label series (one per worker/gpu). The value is the
# last whitespace-separated token of each `name{labels} value` / `name value` line.
function _sum_counter(txt::AbstractString, name::AbstractString)
    total = 0.0
    for line in split(txt, '\n')
        startswith(line, name) || continue
        c = length(line) > length(name) ? line[length(name) + 1] : ' '
        (c == '{' || c == ' ') || continue          # exact metric, not a longer-named one
        v = tryparse(Float64, String(last(split(line))))
        v === nothing || (total += v)
    end
    return total
end

# Fleet weight-cache load/evict totals (summed across workers), or nothing if the scrape failed.
# Climbing evicts under steady load means the model set does not fit resident and the workers are
# thrashing weights (host->device reloads = CPU), which the loadgen surfaces inline.
function scrape_cache_counters()
    try
        txt = read(`curl -fsS --max-time 3 $METRICS_URL`, String)
        return (round(Int, _sum_counter(txt, "worker_weight_loads_total")),
                round(Int, _sum_counter(txt, "worker_weight_evicts_total")))
    catch err
        return nothing
    end
end

# ---- run --------------------------------------------------------------------------------------

function main()
    println("== loadgen: gateway=$GATEWAY transport=$TRANSPORT concurrency=$CONCURRENCY duration=$(DURATION)s ==")
    kserve_init(; n_slots = max(CONCURRENCY, ReactantServerClient.DEFAULT_POOL_SLOTS))
    # One number (CONCURRENCY) controls everything: firing tasks, the SHM staging-pool slots above,
    # and the gRPC concurrent-stream semaphore here. Without this the clients use the global handle
    # (GRPC_MAX_STREAMS=16), which caps in-flight requests at 16 no matter how high CONCURRENCY is.
    grpc = gRPCClient.gRPCCURL(; sticky = false, max_streams = CONCURRENCY)
    # Echo the single knob and everything it derived, so a glance at the logs confirms it propagated.
    println("== loadgen config: LOADGEN_CONCURRENCY=$CONCURRENCY -> firing_tasks=$CONCURRENCY, ",
            "shm_pool_slots=$(max(CONCURRENCY, ReactantServerClient.DEFAULT_POOL_SLOTS)), ",
            "grpc_max_streams=$CONCURRENCY, julia_threads=$(Threads.nthreads()) ==")
    specs = discover_models(grpc)
    isempty(specs) && (println("ERROR: no usable models discovered under $MODEL_REPO"); exit(2))
    println("discovered $(length(specs)) models; starting soak with $(nthreads()) threads")

    WARMUP && warmup(specs)

    deadline = time() + DURATION
    pick_shm(i) = TRANSPORT === :shm ? true : TRANSPORT === :mixed ? isodd(i) : false

    workers = map(1:CONCURRENCY) do _
        Threads.@spawn begin
            i = 0
            while time() < deadline
                i += 1
                spec = specs[rand(1:length(specs))]
                try
                    fire(spec, pick_shm(i))
                catch err
                    record_err(err)
                end
            end
        end
    end

    # Reporter on the INTERACTIVE pool: at concurrency >> nthreads the firing tasks saturate the
    # default (compute) pool, and a default-pool reporter wakes from sleep() with no free thread to
    # run on — so it prints once during the ramp lull then goes silent while the soak keeps running.
    # The interactive thread is reserved (run julia with --threads=N,1) so the reporter always fires.
    reporter = Threads.@spawn :interactive begin
        last_ok = 0
        last_t = time()
        last_loads = 0; last_evicts = 0
        err_shown = false
        while time() < deadline
            sleep(REPORT_SEC)
            now = time()
            ok = N_OK[]; err = N_ERR[]
            d_ok = ok - last_ok
            rps = d_ok / max(now - last_t, 1e-6)
            # Weight-cache load/evict totals across workers, with this window's delta. A rising evict
            # rate is the worker-CPU "thrash" signal (the model set does not fit resident).
            cache = scrape_cache_counters()
            cache_str = if cache === nothing
                "cache=?"
            else
                loads, evicts = cache
                s = "loads=$(loads)(+$(loads - last_loads)) evicts=$(evicts)(+$(evicts - last_evicts))"
                last_loads = loads; last_evicts = evicts
                s
            end
            # Distribution over the requests completed THIS window: drain the samples (so a one-time
            # startup-compile spike only shows in its own window, never pinning later windows) and
            # report min / median / p95 / max plus the mean. `ok`/`err` stay cumulative totals.
            window = @lock LAT_LOCK begin
                s = copy(LAT_SAMPLES); empty!(LAT_SAMPLES); s
            end
            sort!(window)
            n = length(window)
            ms(x) = round(x / 1e6, digits = 2)
            stamp = round(Int, now - (deadline - DURATION))
            if n == 0
                println("[t+$(stamp)s] ok=$ok err=$err rps=$(round(rps, digits=1)) $cache_str (no completions this window)")
            else
                mean_ms = ms(sum(window) / n)
                println("[t+$(stamp)s] ok=$ok err=$err rps=$(round(rps, digits=1)) ",
                        "mean=$(mean_ms)ms min=$(ms(window[1]))ms p50=$(ms(_quantile_sorted(window, 0.5)))ms ",
                        "p95=$(ms(_quantile_sorted(window, 0.95)))ms max=$(ms(window[n]))ms $cache_str")
            end
            # Surface the failure cause as soon as errors appear instead of only at soak end.
            if !err_shown && err > 0
                sample = @lock ERR_LOCK (isempty(ERR_SAMPLES) ? "(no sample captured)" : ERR_SAMPLES[1])
                println("    first error: ", sample)
                err_shown = true
            end
            flush(stdout)   # piped stdout (docker logs) is block-buffered; flush so reports appear live
            last_ok = ok; last_t = now
        end
    end

    foreach(wait, workers)
    wait(reporter)

    ok = N_OK[]; err = N_ERR[]
    println("\n== soak complete: ok=$ok err=$err mean=$(ok > 0 ? round((LAT_NS[]/ok)/1e6, digits=2) : 0)ms ==")
    if err > 0
        println("first error samples:")
        for s in ERR_SAMPLES
            println("  - ", s)
        end
    end
    kserve_shutdown()
    exit(err == 0 ? 0 : 1)
end

# Run only when invoked as the program (the entrypoint runs `julia loadgen.jl`); skip on `include`
# so the driver can be loaded for a syntax/symbol check without connecting to a server.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
