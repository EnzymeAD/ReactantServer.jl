# Dummy-data inference load generator (standalone soak / benchmark tool).
#
# Connects to a running node's gateway over KServe V2 gRPC and drives sustained, concurrent
# inference with zero-filled inputs across every model, to surface memory leaks, races, and
# instability. The gateway describes the models it serves (ControlService/DescribeModels): their
# input shapes and dtypes, which axes are batch or variable, the compiled batch sizes and input
# shapes, so the tool needs no access to the model repository. Every request stages its inputs in
# the client's buffer pool and sends raw bytes, inline or through shared memory. Runs to a fixed
# duration, then prints a per-model summary and exits nonzero if any request errored (exit 1) or
# any model fell short of the per-model coverage minimum (exit 3).
#
# Reactant-free: it needs only ReactantServerClient (which pulls in ReactantServerCore and
# gRPCClient), so run it under that package's project. Point it at any node (native launcher or
# otherwise) via the LOADGEN_* env below; run it as:
#   LOADGEN_GATEWAY=grpc://127.0.0.1:8001 LOADGEN_METRICS=http://127.0.0.1:8002/metrics \
#   LOADGEN_DURATION_SECONDS=300 \
#     julia --project=packages/ReactantServerClient tools/loadgen/loadgen.jl
# The default host names (gateway:8001/8002) are legacy container-network defaults; set the LOADGEN_*
# values above for a native run.
#
# Env knobs (all optional):
#   LOADGEN_GATEWAY            grpc URL of the gateway          (default grpc://gateway:8001)
#   LOADGEN_METRICS            gateway /metrics URL to scrape   (default http://gateway:8002/metrics)
#   LOADGEN_CONCURRENCY        number of concurrent requesters  (default 32)
#   LOADGEN_DURATION_SECONDS   soak duration                    (default 3600)
#   LOADGEN_TRANSPORT          tcp | shm | mixed                (default tcp; raw bytes either way)
#   LOADGEN_SHM_OUTPUTS        read outputs back through shm    (default true; shm path only)
#   LOADGEN_REPORT_SECONDS     rolling-summary interval         (default 30)
#   LOADGEN_MODELS             comma list to restrict the set   (default: every model the gateway routes)
#   LOADGEN_MIN_OK_PER_MODEL   coverage floor per model         (default 1; shortfall -> exit 3;
#                                                                ignored in deadline mode)
#   LOADGEN_SWEEP_EVERY        every Nth pick is round-robin    (default 4; 0 = pure random)
#   LOADGEN_BATCH_MODE         off | rows | burst | both        (default off)
#   LOADGEN_BURST_K            same-model burst run length      (default 8; burst/both modes)
#   LOADGEN_DEADLINE_SECONDS   per-request deadline; enables the deadline-pressure exit policy
#   LOADGEN_RAMP_STEP_SECONDS  enables the shed-ceiling ramp; step cadence
#   LOADGEN_RAMP_START         ramp initial concurrency         (default 4)
#   LOADGEN_RAMP_FACTOR        ramp multiplier per step         (default 2)
#   LOADGEN_RAMP_MAX           ramp ceiling; sizes handle/pool  (default 512)
#   LOADGEN_POOL_MAX_MIB       ceiling on each staging pool     (default 1024)

using ReactantServerClient
import gRPCClient                          # for a dedicated gRPCCURL sized to CONCURRENCY
using Base.Threads

# ---- config ----------------------------------------------------------------------------------

env(k, d) = get(ENV, k, d)
const GATEWAY = env("LOADGEN_GATEWAY", "grpc://gateway:8001")
const METRICS_URL = env("LOADGEN_METRICS", "http://gateway:8002/metrics")
const CONCURRENCY = parse(Int, env("LOADGEN_CONCURRENCY", "32"))
const DURATION = parse(Float64, env("LOADGEN_DURATION_SECONDS", "3600"))
const TRANSPORT = Symbol(env("LOADGEN_TRANSPORT", "tcp"))   # :tcp | :shm | :mixed
const SHM_OUTPUTS = lowercase(env("LOADGEN_SHM_OUTPUTS", "true")) in ("1", "true", "yes", "on")
const REPORT_SEC = parse(Float64, env("LOADGEN_REPORT_SECONDS", "30"))
# Ceiling on each staging pool (one inline, one shared-memory), which main() sizes to fit the
# largest request once per firing task.
const POOL_MAX_BYTES = round(Int, parse(Float64, env("LOADGEN_POOL_MAX_MIB", "1024")) * 2^20)
# Serial warmup before the concurrent soak: one request per model with a long deadline. The first
# request to a model triggers Reactant compilation (often >10s); firing 32 concurrent cold requests
# instead stampedes the gRPCClient libcurl multi-handle into a wedge, so the soak never makes
# progress until manually restarted against warm models. A serial pass (no stampede) + a long
# deadline (tolerates compilation) compiles every model — and warming a meta fans out to compile its
# sub-models — so the soak then hits only warm models. Set LOADGEN_WARMUP=false to skip.
const WARMUP = lowercase(env("LOADGEN_WARMUP", "true")) != "false"
const WARMUP_DEADLINE = parse(Float64, env("LOADGEN_WARMUP_DEADLINE", "300"))

# Coverage: every discovered model must complete at least this many soak requests, or the run
# exits 3 listing the shortfalls. The sweep component guarantees every model is picked regardless
# of duration: every SWEEP_EVERY-th request (globally) comes from a round-robin cursor instead of
# the uniform-random draw. 0 disables the sweep (bitwise-identical pick behavior to the old tool).
const MIN_OK_PER_MODEL = parse(Int, env("LOADGEN_MIN_OK_PER_MODEL", "1"))
const SWEEP_EVERY = parse(Int, env("LOADGEN_SWEEP_EVERY", "4"))

# Batch mode. :rows sends multi-row requests sized from each model's compiled batch ladder
# (client-side prebatching, exercising the multi-row execution path). :burst aims BURST_K
# consecutive global tickets at the same model so concurrent same-model requests queue together
# and the worker's plan_batch coalesces them (server-side request merging). :both does both, with
# rows constant within a burst so all requests in a run share one shape variant.
const BATCH_MODE = Symbol(env("LOADGEN_BATCH_MODE", "off"))
BATCH_MODE in (:off, :rows, :burst, :both) ||
    error("LOADGEN_BATCH_MODE must be off|rows|burst|both, got $(BATCH_MODE)")
const BATCH_ROWS = BATCH_MODE in (:rows, :both)
const BATCH_BURST = BATCH_MODE in (:burst, :both)
const BURST_K = parse(Int, env("LOADGEN_BURST_K", "8"))

# Deadline pressure: when set (>0), soak requests carry this per-request deadline (sent as the
# reactant_timeout_ns request parameter, so it survives the gateway hop and drives the worker's
# EDF admission drop). DEADLINE_EXCEEDED responses are then expected: counted separately and
# excluded from the failure exit code. Warmup always uses WARMUP_DEADLINE.
const DEADLINE_SEC = parse(Float64, env("LOADGEN_DEADLINE_SECONDS", "0"))
const DEADLINE_MODE = DEADLINE_SEC > 0

# Shed-ceiling ramp: effective concurrency starts at RAMP_START and multiplies by RAMP_FACTOR
# every RAMP_STEP seconds until sheds appear (gateway/worker shed counters or client-observed
# RESOURCE_EXHAUSTED), then records the shed point. The gRPC handle, shm
# pool, and task count are sized once at RAMP_MAX; a per-task gate enforces the current limit.
const RAMP_STEP = parse(Float64, env("LOADGEN_RAMP_STEP_SECONDS", "0"))
const RAMP_MODE = RAMP_STEP > 0
const RAMP_START = parse(Int, env("LOADGEN_RAMP_START", "4"))
const RAMP_FACTOR = parse(Int, env("LOADGEN_RAMP_FACTOR", "2"))
const RAMP_MAX = parse(Int, env("LOADGEN_RAMP_MAX", "512"))

# Stall watchdog: a wedged gRPC channel (a request whose completion is never notified, e.g. the
# multi-handle poisoned during a connection-reset storm) blocks every firing task forever while
# the reporter keeps printing "no completions this window". After this many consecutive
# zero-completion windows with requests in flight, print the summary and exit 4 instead of
# hanging until someone kills the container. 0 disables.
const STALL_WINDOWS = parse(Int, env("LOADGEN_STALL_WINDOWS", "4"))

# KServe wire dtype string -> Julia type. Covers the dtypes the bundles use; extend if needed.
const DTYPE = Dict(
    "BOOL" => Bool,
    "UINT8" => UInt8, "UINT16" => UInt16, "UINT32" => UInt32, "UINT64" => UInt64,
    "INT8" => Int8, "INT16" => Int16, "INT32" => Int32, "INT64" => Int64,
    "FP32" => Float32, "FP64" => Float64,
)

# ---- model discovery + dummy-input synthesis --------------------------------------------------

# One input tensor as sent: its name, Julia element type, Julia column-major dims with every
# variable axis filled in from a compiled shape, and the 1-based position of its batch axis (0 when
# the input has none, e.g. a cross-encoder's unbatched query). The batch axis may sit anywhere: an
# image's `whcn` puts it last, a tokenized `nc` first.
struct InputSpec
    name::String
    dtype::DataType
    dims::Vector{Int}       # the batch axis, if any, holds a placeholder 1
    batch_axis::Int
end

# Julia column-major wire dims for one input carrying `n` rows; an unbatched input is sent as is.
function _wire_dims(inp::InputSpec, n::Int)
    d = copy(inp.dims)
    inp.batch_axis > 0 && (d[inp.batch_axis] = n)
    return d
end

# Per-model result counters, all atomics so the hot path never takes a lock or a Dict lookup.
struct ModelStats
    ok::Atomic{Int}
    err::Atomic{Int}
    deadline::Atomic{Int}
    shed::Atomic{Int}
    rows::Atomic{Int}       # batch-axis rows successfully sent (rows-mode reporting)
    lat_ns::Atomic{Int}     # cumulative success latency, for the per-model mean
    warm_ok::Atomic{Bool}   # set by the serial warmup pass
end
ModelStats() = ModelStats(
    Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0),
    Atomic{Int}(0), Atomic{Int}(0), Atomic{Bool}(false)
)

# One model's everything needed to fire a request: a handle per transport (both send raw bytes
# staged through the client's buffer pool; `shm` stages into shared memory, `tcp` inline), one input
# set per compiled shape (a single entry for a model without variable axes), shm output read-back
# declarations, its batch ladder, and its counters.
struct ModelSpec
    name::String
    tcp::KServeModel                 # shared_memory = :off
    shm::KServeModel                 # shared_memory = :on
    variants::Vector{Vector{InputSpec}}
    out_specs::Vector{OutputSpec}    # shm read-back declarations; empty = outputs stay inline
    batch_sizes::Vector{Int}         # sorted ladder; [1] when ineligible or mode off
    stats::ModelStats
end

# Output read-back declarations for the shm path: every output, sized per item (its batch axis
# dropped). Returns [] when disabled or when any output has an unmapped dtype or a variable axis; an
# empty vector keeps that model's outputs inline.
function output_shm_specs(d::ModelDescription)
    SHM_OUTPUTS || return OutputSpec[]
    specs = OutputSpec[]
    for o in d.outputs
        haskey(DTYPE, o.datatype) || return OutputSpec[]
        any(==(:variable), o.axes) && return OutputSpec[]
        push!(specs, OutputSpec(o.name, DTYPE[o.datatype], [s for (s, a) in zip(o.shape, o.axes) if a !== :batch]))
    end
    return specs
end

# The input sets to send, one per compiled shape: the declared dims with every variable axis taken
# from a variant. Returns a reason string instead when the model cannot be driven blind.
function input_variants(d::ModelDescription)
    for t in d.inputs
        haskey(DTYPE, t.datatype) || return "unmapped dtype $(t.datatype)"
        count(==(:batch), t.axes) <= 1 || return "input $(t.name) has more than one batch axis"
    end
    variable = any(t -> any(==(:variable), t.axes), d.inputs)
    # A variable axis needs a size only certain compiled programs accept; without the compiled
    # shapes (a model whose preprocessing transforms its inputs reports none) there is nothing valid
    # to send.
    variable && isempty(d.variants) && return "variable input axis with no compiled shapes"
    shapes = variable ? d.variants : [Dict{String, Vector{Int}}()]
    return map(shapes) do v
        map(d.inputs) do t
            dims = haskey(v, t.name) ? copy(v[t.name]) : copy(t.shape)
            b = something(findfirst(==(:batch), t.axes), 0)
            b > 0 && (dims[b] = 1)
            InputSpec(t.name, DTYPE[t.datatype], dims, b)
        end
    end
end

# Every model the discovery pass dropped, with the reason — reported prominently at startup and in
# the final summary so a silent coverage hole is impossible.
const SKIPPED = Vector{Tuple{String, String}}()
_skip!(name, reason) = (push!(SKIPPED, (name, reason)); @warn "skip: $reason" model = name)

# Ask the gateway what it serves (ControlService/DescribeModels): every model it routes, or the
# LOADGEN_MODELS subset. A meta model's internal sub-models are not routed, so they are not listed
# and never soaked directly.
function describe_fleet(grpc)
    want = strip(get(ENV, "LOADGEN_MODELS", ""))
    names = isempty(want) ? String[] : String.(strip.(split(want, ",")))
    try
        return describe_models(GATEWAY; names = names, deadline = 30.0, grpc = grpc)
    catch ex
        println("ERROR: DescribeModels against $GATEWAY failed: ", sprint(showerror, ex))
        ex isa gRPCClient.gRPCServiceCallException && ex.grpc_status == gRPCClient.GRPC_UNIMPLEMENTED &&
            println("  The node predates ControlService/DescribeModels; update its gateway and workers.")
        exit(2)
    end
end

function build_specs(descriptions, grpc)
    specs = ModelSpec[]
    for d in descriptions
        variants = input_variants(d)
        variants isa String && (_skip!(d.name, variants); continue)
        isempty(d.inputs) && (_skip!(d.name, "no inputs"); continue)
        # Multi-row eligibility: a compiled batch size > 1 AND every input has a batch axis. Sizes
        # above the worker's coalescing cap are left out; the cap is what it executes at once.
        ladder = isempty(d.batch_sizes) ? Int[1] : copy(d.batch_sizes)
        d.max_batch_size > 0 && filter!(<=(d.max_batch_size), ladder)
        isempty(ladder) && (ladder = Int[1])
        eligible = any(>(1), ladder) && all(inp.batch_axis > 0 for inp in first(variants))
        batch_sizes = (BATCH_ROWS && eligible) ? ladder : Int[1]
        # Retries off: this is a measurement tool. The client's default retry-on-shed policy
        # (#57) would silently absorb sheds and EDF drops, hiding exactly what ramp and
        # deadline modes exist to observe.
        handle(mode) = KServeModel(
            GATEWAY, d.name; max_batch_size = maximum(batch_sizes),
            deadline = DEADLINE_MODE ? DEADLINE_SEC : 10.0, grpc = grpc, shared_memory = mode,
            retry = ReactantServerClient.RetryPolicy(enabled = false)
        )
        push!(
            specs, ModelSpec(
                d.name, handle(:off), handle(TRANSPORT === :tcp ? :off : :on), variants,
                output_shm_specs(d), batch_sizes, ModelStats()
            )
        )
    end
    return specs
end

# The largest request any model can send, inputs plus shm-read outputs, so the staging pool can be
# sized to fit one in each slot instead of spanning slots under load.
function max_request_bytes(specs)
    worst = 0
    for spec in specs, v in spec.variants
        n = maximum(spec.batch_sizes)
        ins = sum(sizeof(inp.dtype) * prod(_wire_dims(inp, n)) for inp in v)
        outs = sum((sizeof(o.dtype) * prod(o.per_item_dims) * n for o in spec.out_specs); init = 0)
        worst = max(worst, ins + outs)
    end
    return worst
end

function print_skipped()
    return if isempty(SKIPPED)
        println("== no models skipped ==")
    else
        println("== skipped models ($(length(SKIPPED))) ==")
        for (name, reason) in sort(SKIPPED)
            println("  ", rpad(name, 52), reason)
        end
    end
end

# Per-model table: warm (serial warmup succeeded), soak ok/err/dead/shed, rows, mean latency.
# Shared by the end-of-run summary and the stall watchdog's abort report.
function print_model_table(specs)
    println("\n== per-model results ==")
    println(
        rpad("model", 52), lpad("warm", 5), lpad("ok", 9), lpad("err", 6),
        lpad("dead", 7), lpad("shed", 7), lpad("rows", 10), lpad("mean_ms", 9)
    )
    for spec in specs
        s = spec.stats
        mean_ms = s.ok[] > 0 ? round(s.lat_ns[] / s.ok[] / 1.0e6, digits = 1) : 0.0
        println(
            rpad(spec.name, 52), lpad(s.warm_ok[] ? "y" : "n", 5), lpad(string(s.ok[]), 9),
            lpad(string(s.err[]), 6), lpad(string(s.deadline[]), 7), lpad(string(s.shed[]), 7),
            lpad(string(s.rows[]), 10), lpad(string(mean_ms), 9)
        )
    end
    return
end

# The IO every request goes through: `n` zero-filled items of one compiled shape of a model's inputs,
# staged in the client's buffer pool and sent as raw bytes, through shared memory or inline
# depending on the handle's transport.
struct DummyIO <: AbstractInferenceIO
    spec::ModelSpec
    inputs::Vector{InputSpec}   # one entry of spec.variants
    n::Int
end
Base.length(io::DummyIO) = io.n
# Per-item bytes summed across all inputs (each input's size at one row).
ReactantServerClient.item_input_bytes(io::DummyIO) =
    sum(sizeof(inp.dtype) * prod(_wire_dims(inp, 1)) for inp in io.inputs)
function ReactantServerClient.infer_encode_chunk!(io::DummyIO, r, slot)
    n = length(r)
    ins = PoolInferInput[]
    for inp in io.inputs
        wire = _wire_dims(inp, n)                          # Julia col-major
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
# output shape) keeps outputs inline, the safe fallback. On the inline transport the declared
# outputs are only requested by name.
ReactantServerClient.output_specs(io::DummyIO) = io.spec.out_specs

# ---- counters ---------------------------------------------------------------------------------

const N_OK = Atomic{Int}(0)
const N_ERR = Atomic{Int}(0)    # unexpected failures only (never deadline/shed)
const N_DEADLINE = Atomic{Int}(0)    # DEADLINE_EXCEEDED responses (expected under deadline mode)
const N_SHED = Atomic{Int}(0)    # RESOURCE_EXHAUSTED (expected in ramp)
const N_CONN = Atomic{Int}(0)    # subset of N_ERR: connection-teardown INTERNAL errors
const INFLIGHT = Atomic{Int}(0)    # requests currently inside fire() (stall-watchdog signal)
const LAT_NS = Atomic{Int}(0)        # cumulative successful-request latency, ns (overall summary)
const LAT_SAMPLES = Int[]            # per-window latency samples, ns; drained each report
const LAT_LOCK = ReentrantLock()     # guards LAT_SAMPLES
# First few failure messages per class (:error keeps more), to verify classification.
const SAMPLES = Dict(:error => String[], :deadline => String[], :shed => String[])
const ERR_LOCK = ReentrantLock()     # guards SAMPLES

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

# gRPC-status classification of a failed request. DEADLINE_EXCEEDED is the worker's EDF admission
# drop (or a budget expiry anywhere on the path); RESOURCE_EXHAUSTED is a shed, from the gateway
# admission cap or passed through verbatim from a worker (post-#57; pre-#57 gateways re-mapped
# worker sheds to FAILED_PRECONDITION, which this tool no longer recognizes). The string fallback
# covers client-local timeout shapes that surface as non-gRPC exceptions.
# Connection-teardown signatures (libcurl INTERNAL messages): the server slamming streams or
# connections, seen when ramp mode drives past the shed ceiling or deadline aborts churn the
# HTTP/2 connection. Counted in the err column but tolerated by the exit policy in those modes.
_is_conn_teardown(msg) =
    occursin("Broken pipe", msg) || occursin("Connection reset", msg) ||
    occursin("GOAWAY", msg) || occursin("not closed cleanly", msg) ||
    occursin("rewind was not possible", msg)

function classify(ex)
    if ex isa gRPCClient.gRPCServiceCallException
        s = ex.grpc_status
        s == gRPCClient.GRPC_DEADLINE_EXCEEDED && return :deadline
        # The worker's EDF pre-dispatch drop reaches the client as UNAVAILABLE (the gateway re-maps
        # it); the message is the tell.
        s == gRPCClient.GRPC_UNAVAILABLE &&
            occursin("deadline exceeded before dispatch", ex.message) && return :deadline
        s == gRPCClient.GRPC_RESOURCE_EXHAUSTED && return :shed
        # Empty-message INTERNAL is also teardown: libcurl aborts that populate no error buffer.
        # A genuine server-side INTERNAL failure carries its grpc-message through.
        s == gRPCClient.GRPC_INTERNAL &&
            (isempty(ex.message) || _is_conn_teardown(ex.message)) && return :conn
        return :error
    end
    msg = sprint(showerror, ex)
    occursin("DEADLINE_EXCEEDED", msg) && return :deadline
    occursin("RESOURCE_EXHAUSTED", msg) && return :shed
    return :error
end

function record_result(spec::ModelSpec, ex)
    cls = classify(ex)
    if cls === :deadline
        atomic_add!(N_DEADLINE, 1)
        atomic_add!(spec.stats.deadline, 1)
    elseif cls === :shed
        atomic_add!(N_SHED, 1)
        atomic_add!(spec.stats.shed, 1)
    else
        # :conn still lands in the err counters/column for visibility; N_CONN lets the exit
        # policy tolerate teardown-shaped errors in ramp/deadline modes without masking others.
        cls === :conn && atomic_add!(N_CONN, 1)
        atomic_add!(N_ERR, 1)
        atomic_add!(spec.stats.err, 1)
    end
    key = (cls === :deadline || cls === :shed) ? cls : :error
    return @lock ERR_LOCK begin
        buf = SAMPLES[key]
        length(buf) < (key === :error ? 20 : 5) && push!(buf, sprint(showerror, ex))
    end
end

function fire(spec::ModelSpec, use_shm::Bool, nrows::Int, inputs::Vector{InputSpec})
    t0 = time_ns()
    infer_sync(use_shm ? spec.shm : spec.tcp, DummyIO(spec, inputs, nrows))
    dt = Int(time_ns() - t0)
    atomic_add!(N_OK, 1)
    atomic_add!(spec.stats.ok, 1)
    atomic_add!(LAT_NS, dt)
    atomic_add!(spec.stats.lat_ns, dt)
    atomic_add!(spec.stats.rows, nrows)
    @lock LAT_LOCK push!(LAT_SAMPLES, dt)        # window sample for the distribution stats
    return nothing
end

# ---- pick logic --------------------------------------------------------------------------------

# Global ticket counter shared by all firing tasks: drives the coverage sweep (every SWEEP_EVERY-th
# ticket is round-robin, so every model is guaranteed traffic at any duration) and the burst runs
# (BURST_K consecutive tickets — landing on different tasks nearly simultaneously — target the same
# model, creating the same-variant queue run the worker's plan_batch coalesces).
const TICKET = Atomic{Int}(0)

function pick_spec(specs::Vector{ModelSpec}, t::Int)
    if BATCH_BURST
        return specs[1 + (t ÷ BURST_K) % length(specs)]
    elseif SWEEP_EVERY > 0 && t % SWEEP_EVERY == 0
        return specs[1 + (t ÷ SWEEP_EVERY) % length(specs)]
    else
        return specs[rand(1:length(specs))]
    end
end

# Rows and input shape for this request. In burst mode both are a function of the burst index, so
# every request within a burst shares one compiled program (mixed shapes would split the run the
# worker coalesces).
function pick_rows(spec::ModelSpec, t::Int)
    (BATCH_ROWS && length(spec.batch_sizes) > 1) || return 1
    if BATCH_BURST
        return spec.batch_sizes[1 + (t ÷ BURST_K) % length(spec.batch_sizes)]
    else
        return rand(spec.batch_sizes)
    end
end

function pick_variant(spec::ModelSpec, t::Int)
    length(spec.variants) == 1 && return spec.variants[1]
    BATCH_BURST && return spec.variants[1 + (t ÷ BURST_K) % length(spec.variants)]
    return rand(spec.variants)
end

# Serial warmup: fire one request per model with a long deadline so first-request compilation
# completes before the concurrent soak begins (see WARMUP comment). Sequential by design — a
# concurrent cold stampede wedges the gRPCClient multi-handle. In rows mode every ladder size is
# primed (each batch size is a separate compiled executable; a cold large-batch compile mid-soak
# would recreate the stampede), and so is every compiled input shape. Failures are logged, not fatal.
function warmup(specs::Vector{ModelSpec})
    nreq = sum(length(spec.batch_sizes) * length(spec.variants) for spec in specs)
    println("warmup: priming $(length(specs)) models / $nreq requests serially (deadline $(WARMUP_DEADLINE)s) ...")
    t0 = time(); nok = 0
    for spec in specs
        m = KServeModel(
            GATEWAY, spec.name; max_batch_size = maximum(spec.batch_sizes),
            deadline = WARMUP_DEADLINE, grpc = spec.tcp.grpc, shared_memory = :off,
            retry = ReactantServerClient.RetryPolicy(enabled = false)
        )
        allok = true
        for v in spec.variants, n in spec.batch_sizes
            try
                infer_sync(m, DummyIO(spec, v, n))
            catch ex
                allok = false
                shape = Dict(inp.name => _wire_dims(inp, n) for inp in v)
                @warn "warmup failed (continuing)" model = spec.name rows = n shape exception = ex
            end
        end
        if allok
            spec.stats.warm_ok[] = true
            nok += 1
        end
    end
    return println("warmup: $nok/$(length(specs)) models primed in $(round(time() - t0, digits = 1))s")
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

# Fleet counters (summed across workers), or nothing if the scrape failed. Climbing evicts under
# steady load means the model set does not fit resident and the workers are thrashing weights
# (host->device reloads = CPU). The shed counters are the ramp mode's ground truth: the gateway
# increments gateway_requests_shed_total at its admission cap, workers worker_requests_shed_total.
function scrape_counters()
    try
        txt = read(`curl -fsS --max-time 3 $METRICS_URL`, String)
        return (
            loads = round(Int, _sum_counter(txt, "worker_weight_loads_total")),
            evicts = round(Int, _sum_counter(txt, "worker_weight_evicts_total")),
            gw_shed = round(Int, _sum_counter(txt, "gateway_requests_shed_total")),
            worker_shed = round(Int, _sum_counter(txt, "worker_requests_shed_total")),
        )
    catch ex
        return nothing
    end
end

# ---- run --------------------------------------------------------------------------------------

function main()
    println(
        "== loadgen: gateway=$GATEWAY transport=$TRANSPORT concurrency=$CONCURRENCY duration=$(DURATION)s ",
        "batch_mode=$BATCH_MODE deadline=$(DEADLINE_MODE ? "$(DEADLINE_SEC)s" : "off") ",
        "ramp=$(RAMP_MODE ? "$(RAMP_START)..$(RAMP_MAX) x$(RAMP_FACTOR)/$(RAMP_STEP)s" : "off") =="
    )
    # Ramp mode sizes everything at the ceiling once (the handle cannot be resized mid-run); a
    # per-task gate enforces the current limit.
    effective_max = RAMP_MODE ? RAMP_MAX : CONCURRENCY
    if RAMP_MODE
        TRANSPORT === :tcp ||
            @warn "ramp mode with shm/mixed transport sizes the shm pool at RAMP_MAX slots; slot back-pressure can distort the ramp — tcp transport is recommended"
        steps_needed = ceil(Int, log(RAMP_FACTOR, max(RAMP_MAX / max(RAMP_START, 1), 1)))
        DURATION < steps_needed * RAMP_STEP &&
            @warn "duration too short for the ramp to reach RAMP_MAX" needed_seconds = steps_needed * RAMP_STEP duration = DURATION
    end
    # One number (effective_max) controls everything: firing tasks, the staging-pool slots, and the
    # gRPC concurrent-stream semaphore here. Without this the clients use the global handle
    # (GRPC_MAX_STREAMS=16), which caps in-flight requests at 16 no matter how high the requested
    # concurrency is. The handle opens itself, so the models can be described before kserve_init.
    grpc = gRPCClient.gRPCCURL(; sticky = false, max_streams = effective_max)
    specs = build_specs(describe_fleet(grpc), grpc)
    print_skipped()
    isempty(specs) && (println("ERROR: no usable models on $GATEWAY"); exit(2))
    # Both transports stage through the pool, so size it to hold the largest request in one slot
    # per firing task, within reason: a request bigger than a slot still works by spanning slots.
    n_slots = max(effective_max, ReactantServerClient.DEFAULT_POOL_SLOTS)
    pool_bytes = clamp(n_slots * max_request_bytes(specs), ReactantServerClient.DEFAULT_POOL_BYTES, POOL_MAX_BYTES)
    kserve_init(; n_slots = n_slots, pool_bytes = pool_bytes)
    # Echo the single knob and everything it derived, so a glance at the logs confirms it propagated.
    println(
        "== loadgen config: effective_max=$effective_max -> firing_tasks=$effective_max, ",
        "pool_slots=$n_slots, pool_bytes=$(round(pool_bytes / 2^20, digits = 1))MiB, ",
        "grpc_max_streams=$effective_max, julia_threads=$(Threads.nthreads()) =="
    )
    if BATCH_ROWS
        eligible = count(spec -> length(spec.batch_sizes) > 1, specs)
        println("batch rows mode: $eligible/$(length(specs)) models batch-eligible (ladder > 1 and all inputs batched)")
    end
    println("discovered $(length(specs)) models; starting soak with $(nthreads()) threads")
    flush(stdout)   # surface startup lines now; stdout to docker is block-buffered (see reporter)

    WARMUP && warmup(specs)
    flush(stdout)

    deadline = time() + DURATION
    pick_shm(t) = TRANSPORT === :shm ? true : TRANSPORT === :mixed ? isodd(t) : false

    # Ramp gate: tasks above cur_limit idle until the controller raises it. stop ends the run
    # early (stop-on-shed). With ramp off, cur_limit == effective_max and the gate never bites.
    cur_limit = Atomic{Int}(RAMP_MODE ? min(RAMP_START, effective_max) : effective_max)
    stop = Atomic{Bool}(false)

    workers = map(1:effective_max) do wi
        Threads.@spawn begin
            while time() < deadline && !stop[]
                if wi > cur_limit[]
                    sleep(0.05)
                    continue
                end
                t = atomic_add!(TICKET, 1)
                spec = pick_spec(specs, t)
                nrows = pick_rows(spec, t)
                inputs = pick_variant(spec, t)
                atomic_add!(INFLIGHT, 1)
                try
                    fire(spec, pick_shm(t), nrows, inputs)
                catch ex
                    # NOTE: must NOT be named `err` — `err` would collide with main()-scope locals
                    # that closures capture and SHARE (see the summary block below); a firing task
                    # writing a caught exception into a shared local races every other reader.
                    record_result(spec, ex)
                finally
                    atomic_add!(INFLIGHT, -1)
                end
            end
        end
    end

    # Ramp controller: steps the gate up every RAMP_STEP seconds until sheds appear (client-side
    # classification or the gateway/worker shed counters rising above the post-warmup baseline),
    # then records the shed point and stops the run. Runs on the interactive pool for the same
    # reason as the reporter below.
    shed_point = Ref{Union{Nothing, NamedTuple}}(nothing)
    ramp = nothing
    if RAMP_MODE
        base = scrape_counters()
        base_gw = base === nothing ? 0 : base.gw_shed
        base_wk = base === nothing ? 0 : base.worker_shed
        ramp = Threads.@spawn :interactive begin
            last_ok_r = N_OK[]
            last_t_r = time()
            while time() < deadline && !stop[]
                sleep(min(RAMP_STEP, max(deadline - time(), 0.01)))
                time() < deadline || break
                now_r = time()
                rps_r = (N_OK[] - last_ok_r) / max(now_r - last_t_r, 1.0e-6)
                last_ok_r = N_OK[]; last_t_r = now_r
                c = scrape_counters()
                gw = c === nothing ? base_gw : c.gw_shed
                wk = c === nothing ? base_wk : c.worker_shed
                if (N_SHED[] > 0 || gw > base_gw || wk > base_wk) && shed_point[] === nothing
                    shed_point[] = (concurrency = cur_limit[], rps = rps_r)
                    println(
                        ">>> ramp: first sheds at concurrency=$(cur_limit[]) rps=$(round(rps_r, digits = 1)) ",
                        "(client_shed=$(N_SHED[]) gw_shed=+$(gw - base_gw) wk_shed=+$(wk - base_wk))"
                    )
                    flush(stdout)
                    stop[] = true; break
                elseif shed_point[] === nothing && cur_limit[] < effective_max
                    new_limit = min(cur_limit[] * RAMP_FACTOR, effective_max)
                    cur_limit[] = new_limit
                    println(">>> ramp: concurrency -> $new_limit")
                    flush(stdout)
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
        last_dead = 0
        last_shed = 0
        last_t = time()
        last_loads = 0; last_evicts = 0
        err_shown = false
        last_done = 0; stalled = 0     # stall watchdog state (see STALL_WINDOWS)
        # Keep reporting (and stall-watching) past the soak deadline while requests are still in
        # flight: a wedged transport leaves the firing tasks blocked forever AFTER the deadline,
        # which is exactly when the watchdog must still be alive to abort the run.
        while (time() < deadline || INFLIGHT[] > 0) && !stop[]
            sleep(REPORT_SEC)
            try
                now = time()
                ok = N_OK[]; n_err = N_ERR[]; n_dead = N_DEADLINE[]; n_shed = N_SHED[]
                d_ok = ok - last_ok
                rps = d_ok / max(now - last_t, 1.0e-6)
                # Weight-cache load/evict totals across workers, with this window's delta. A rising evict
                # rate is the worker-CPU "thrash" signal (the model set does not fit resident).
                cache = scrape_counters()
                cache_str = if cache === nothing
                    "cache=?"
                else
                    s = "loads=$(cache.loads)(+$(cache.loads - last_loads)) evicts=$(cache.evicts)(+$(cache.evicts - last_evicts)) " *
                        "gw_shed=$(cache.gw_shed) wk_shed=$(cache.worker_shed)"
                    last_loads = cache.loads; last_evicts = cache.evicts
                    s
                end
                conc_str = RAMP_MODE ? "conc=$(cur_limit[]) " : ""
                counts = "ok=$ok err=$n_err dead=$n_dead(+$(n_dead - last_dead)) shed=$n_shed(+$(n_shed - last_shed))"
                # Distribution over the requests completed THIS window: drain the samples (so a one-time
                # startup-compile spike only shows in its own window, never pinning later windows) and
                # report min / median / p95 / max plus the mean. Totals stay cumulative.
                window = @lock LAT_LOCK begin
                    s = copy(LAT_SAMPLES); empty!(LAT_SAMPLES); s
                end
                sort!(window)
                n = length(window)
                ms(x) = round(x / 1.0e6, digits = 2)
                stamp = round(Int, now - (deadline - DURATION))
                if n == 0
                    println("[t+$(stamp)s] $conc_str$counts rps=$(round(rps, digits = 1)) $cache_str (no completions this window)")
                else
                    mean_ms = ms(sum(window) / n)
                    println(
                        "[t+$(stamp)s] $conc_str$counts rps=$(round(rps, digits = 1)) ",
                        "mean=$(mean_ms)ms min=$(ms(window[1]))ms p50=$(ms(_quantile_sorted(window, 0.5)))ms ",
                        "p95=$(ms(_quantile_sorted(window, 0.95)))ms max=$(ms(window[n]))ms $cache_str"
                    )
                end
                # Surface the failure cause as soon as errors appear instead of only at soak end.
                if !err_shown && n_err > 0
                    sample = @lock ERR_LOCK (isempty(SAMPLES[:error]) ? "(no sample captured)" : SAMPLES[:error][1])
                    println("    first error: ", sample)
                    err_shown = true
                end
                flush(stdout)   # piped stdout (docker logs) is block-buffered; flush so reports appear live
                # Stall watchdog: completions (any class) stopped while requests are in flight means
                # the transport is wedged — the blocked firing tasks will never observe the deadline,
                # so bail out with the partial summary rather than hang until the container is killed.
                total_done = ok + n_err + n_dead + n_shed
                if STALL_WINDOWS > 0 && total_done == last_done && INFLIGHT[] > 0
                    stalled += 1
                    if stalled >= STALL_WINDOWS
                        println(
                            "\nFATAL: no request completions in $stalled consecutive report ",
                            "windows with $(INFLIGHT[]) requests in flight — transport wedged; ",
                            "aborting (results below are partial)"
                        )
                        # Wedge diagnosis: where are the stuck requests? in_multi > 0 means transfers
                        # sit in libcurl without completing; sem_free == 0 with in_multi == 0 means the
                        # concurrent-stream semaphore leaked and tasks are blocked before libcurl.
                        try
                            g = specs[1].tcp.grpc
                            println(
                                "gRPC handle state: running=$(g.running) ",
                                "in_multi=$(length(g.requests)) sem_free=$(Base.n_avail(g.sem)) ",
                                "watchers=$(length(g.watchers)) timer_armed=$(g.timer !== nothing)"
                            )
                        catch diag_ex
                            println("gRPC handle state unavailable: ", sprint(showerror, diag_ex))
                        end
                        print_model_table(specs)
                        print_skipped()
                        flush(stdout)
                        exit(4)
                    end
                else
                    stalled = 0
                end
                last_done = total_done
                last_ok = ok; last_t = now; last_dead = n_dead; last_shed = n_shed
            catch e
                # A single bad iteration (e.g. a transient scrape/format error) must never kill the
                # reporter and masquerade as a silent soak — log it and keep reporting.
                println("[reporter] error this window (continuing): ", sprint(showerror, e)); flush(stdout)
            end
        end
    end

    foreach(wait, workers)
    stop[] = true                      # release the reporter/controller promptly on a full run
    ramp === nothing || wait(ramp)
    wait(reporter)

    total_ok = N_OK[]; total_err = N_ERR[]; total_dead = N_DEADLINE[]; total_shed = N_SHED[]
    total_conn = N_CONN[]
    println(
        "\n== soak complete: ok=$total_ok err=$total_err (conn=$total_conn) ",
        "dead=$total_dead shed=$total_shed ",
        "rows=$(sum(spec.stats.rows[] for spec in specs)) ",
        "mean=$(total_ok > 0 ? round((LAT_NS[] / total_ok) / 1.0e6, digits = 2) : 0)ms =="
    )

    print_model_table(specs)
    print_skipped()

    if RAMP_MODE
        sp = shed_point[]
        if sp === nothing
            println("ramp: no sheds observed up to concurrency $(cur_limit[])")
        else
            println("ramp: shed ceiling at concurrency=$(sp.concurrency) rps=$(round(sp.rps, digits = 1))")
        end
    end
    for (key, label) in ((:error, "error"), (:deadline, "deadline"), (:shed, "shed"))
        isempty(SAMPLES[key]) && continue
        println("first $label samples:")
        for s in SAMPLES[key]
            println("  - ", s)
        end
    end

    # Coverage assertion: every discovered model must have completed MIN_OK_PER_MODEL soak requests
    # (warmup does not count; the table's warm column disambiguates "compiled fine but starved"
    # from "never worked").
    # Not meaningful under deadline pressure: the deadline is deliberately below most models'
    # service time, so per-model ok floors would always fire.
    shortfalls = DEADLINE_MODE ? String[] :
        [spec.name for spec in specs if spec.stats.ok[] < MIN_OK_PER_MODEL]
    if !isempty(shortfalls)
        println("\nCOVERAGE SHORTFALL: $(length(shortfalls)) models below $(MIN_OK_PER_MODEL) ok requests:")
        for name in shortfalls
            println("  - ", name)
        end
    end

    kserve_shutdown()
    # Exit policy: real errors always fail; connection-teardown errors (the conn subset of err)
    # are tolerated in ramp/deadline modes, where slamming streams is the expected consequence of
    # driving past the shed ceiling or aborting at deadlines; deadline drops and sheds fail only
    # outside ramp AND deadline modes (ramp hunts for sheds, and past the ceiling queue wait
    # pushes requests over the client deadline; deadline pressure sheds transiently because
    # client-aborted requests still hold gateway admission slots until the worker finishes them);
    # coverage shortfalls exit 3 so CI can distinguish starvation from request failures. (Exit 4 =
    # the stall watchdog aborted a wedged run; that path exits from the reporter, never reaches
    # here.)
    tolerated = (RAMP_MODE || DEADLINE_MODE) ? total_conn : 0
    hard_fail = (total_err - tolerated) > 0 ||
        (!RAMP_MODE && !DEADLINE_MODE && (total_dead > 0 || total_shed > 0))
    return exit(hard_fail ? 1 : (isempty(shortfalls) ? 0 : 3))
end

# Run only when invoked as the program (`julia loadgen.jl`); skip on `include` so the driver can be
# loaded for a syntax/symbol check without connecting to a server.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
