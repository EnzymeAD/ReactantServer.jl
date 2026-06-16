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

# KServe wire dtype string -> Julia type. Covers the dtypes the bundles use; extend if needed.
const DTYPE = Dict(
    "BOOL" => Bool,
    "UINT8" => UInt8, "UINT16" => UInt16, "UINT32" => UInt32, "UINT64" => UInt64,
    "INT8" => Int8, "INT16" => Int16, "INT32" => Int32, "INT64" => Int64,
    "FP32" => Float32, "FP64" => Float64,
)

# ---- model discovery + dummy-input synthesis --------------------------------------------------

# One model's everything needed to fire a request: its gateway handle, the input tensor name and
# Julia element type, the per-item Julia column-major dims (batch dropped), and a prebuilt inline
# InferInput for the TCP path (immutable, so safe to reuse across requests).
struct ModelSpec
    name::String
    model::KServeModel
    input_name::String
    dtype::DataType
    per_item_dims::Vector{Int}
    tcp_input::Any              # ModelInferRequest.InferInputTensor for the inline path
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

function discover_models()
    names = readdir(MODEL_REPO)
    want = get(ENV, "LOADGEN_MODELS", "")
    if !isempty(want)
        sel = Set(strip.(split(want, ",")))
        names = filter(in(sel), names)
    end
    specs = ModelSpec[]
    for name in sort(names)
        manifest = joinpath(MODEL_REPO, name, "manifest.yaml")
        isfile(manifest) || continue
        local spec
        try
            io = manifest_io_spec(manifest)
            isempty(io.input_order) && (@warn "skip: no inputs" model=name; continue)
            tname = io.input_order[1]
            tm = io.inputs[tname]
            haskey(DTYPE, tm.datatype) || (@warn "skip: unmapped dtype" model=name dtype=tm.datatype; continue)
            T = DTYPE[tm.datatype]
            dims = per_item_dims(tm.shape)
            arr = zeros(T, dims..., 1)                       # one item; Julia col-major, batch last
            spec = ModelSpec(name, KServeModel(GATEWAY, name; max_batch_size = 1),
                             tname, T, dims, InferInput(tname, arr), output_shm_specs(io))
        catch err
            @warn "skip: manifest_io_spec failed" model=name exception=err
            continue
        end
        push!(specs, spec)
    end
    return specs
end

# Minimal IO for the shared-memory path: one zero-filled item of a single model's input.
struct DummyIO <: AbstractInferenceIO
    spec::ModelSpec
end
Base.length(::DummyIO) = 1
ReactantServerClient.item_input_bytes(io::DummyIO) = sizeof(io.spec.dtype) * prod(io.spec.per_item_dims)
function ReactantServerClient.infer_encode_chunk!(io::DummyIO, r, slot)
    n = length(r)
    nbytes = n * item_input_bytes(io)
    sub = subslot(slot, nbytes)
    fill!(pool_view(sub, UInt8, nbytes), 0x00)
    # PoolInferInput shape is Julia column-major (per-item dims then batch last).
    return [InferInput(io.spec.input_name, sub, Int[io.spec.per_item_dims..., n], io.spec.dtype)]
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
const LAT_NS = Atomic{Int}(0)        # cumulative successful-request latency, nanoseconds
const LAT_MX = Atomic{Int}(0)        # max single-request latency this window, ns (reset each report)
const ERR_SAMPLES = String[]
const ERR_LOCK = ReentrantLock()

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
        infer_sync(spec.model, [spec.tcp_input])
    end
    dt = Int(time_ns() - t0)
    atomic_add!(N_OK, 1)
    atomic_add!(LAT_NS, dt)
    while true                                   # lock-free max update
        old = LAT_MX[]
        dt <= old && break
        atomic_cas!(LAT_MX, old, dt) == old && break
    end
    return nothing
end

# ---- metrics scrape (via curl; no extra Julia deps) -------------------------------------------

function scrape_metrics()
    try
        txt = read(`curl -fsS --max-time 3 $METRICS_URL`, String)
        for line in split(txt, '\n')
            (startswith(line, "gateway_requests_total") || startswith(line, "gateway_worker_ready")) &&
                println("    metric  ", line)
        end
    catch err
        println("    metrics scrape failed: ", sprint(showerror, err))
    end
end

# ---- run --------------------------------------------------------------------------------------

function main()
    println("== loadgen: gateway=$GATEWAY transport=$TRANSPORT concurrency=$CONCURRENCY duration=$(DURATION)s ==")
    kserve_init(; n_slots = max(CONCURRENCY, ReactantServerClient.DEFAULT_POOL_SLOTS))
    specs = discover_models()
    isempty(specs) && (println("ERROR: no usable models discovered under $MODEL_REPO"); exit(2))
    println("discovered $(length(specs)) models; starting soak with $(nthreads()) threads")

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

    reporter = Threads.@spawn begin
        last_ok = 0
        last_lat = 0
        last_t = time()
        err_shown = false
        while time() < deadline
            sleep(REPORT_SEC)
            now = time()
            ok = N_OK[]; err = N_ERR[]; lat = LAT_NS[]
            d_ok = ok - last_ok
            d_lat = lat - last_lat
            rps = d_ok / max(now - last_t, 1e-6)
            # Per-window mean and max so the distribution reflects this window, not the whole run.
            # In particular the one-time startup-compile spike no longer pins the max forever:
            # read-and-reset the window max with atomic_xchg!. `ok`/`err` stay cumulative totals.
            mean_ms = d_ok > 0 ? (d_lat / d_ok) / 1e6 : 0.0
            max_ms = atomic_xchg!(LAT_MX, 0) / 1e6
            println("[t+$(round(Int, now - (deadline - DURATION)))s] ok=$ok err=$err rps=$(round(rps, digits=1)) ",
                    "mean=$(round(mean_ms, digits=1))ms max=$(round(max_ms, digits=1))ms")
            # Surface the failure cause as soon as errors appear instead of only at soak end.
            if !err_shown && err > 0
                sample = @lock ERR_LOCK (isempty(ERR_SAMPLES) ? "(no sample captured)" : ERR_SAMPLES[1])
                println("    first error: ", sample)
                err_shown = true
            end
            scrape_metrics()
            last_ok = ok; last_lat = lat; last_t = now
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
