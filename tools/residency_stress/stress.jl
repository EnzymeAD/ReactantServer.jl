# Residency stress driver: hammer a worker (or a gateway) with requests whose every answer is
# checked exactly against the weights the model should be serving, while forcing weight churn on
# the device and host tiers, compacting memory, and adding, removing, renaming and updating models
# through the directory watcher. See README.md beside this file.
#
#   julia --threads=8 --project=packages/ReactantServerGateway tools/residency_stress/stress.jl
#
# Configured through RS_* environment variables (see `Opts` below). Reactant-free: it speaks KServe
# V2 and the worker control service over gRPC with the gateway package's generated messages. Every
# call carries a client deadline, so a hung server cannot hang the driver.
#
# Exit status: 0 no findings, 1 findings (see OUT/report.txt), 2 usage error.

using ReactantServerGateway: ReactantServerGateway
import gRPCClient
using TOML: TOML
using Random: Xoshiro
using Printf: @sprintf
using Dates: Dates
using Base.Threads: Atomic, atomic_add!, @spawn

const Inf_ = ReactantServerGateway.inference
const Ctl = ReactantServerGateway.control

# ---- options ------------------------------------------------------------------------------------

envs(k, d) = get(ENV, k, d)
envf(k, d) = parse(Float64, envs(k, string(d)))
envi(k, d) = parse(Int, envs(k, string(d)))
envlist(k, d = "") = filter(!isempty, strip.(split(envs(k, d), ",")))

Base.@kwdef struct Opts
    target::String = envs("RS_TARGET", "127.0.0.1:8001")          # worker or gateway gRPC host:port
    metrics::Vector{String} = envlist("RS_METRICS")                # worker /metrics URLs (or the gateway's)
    catalog::String = envs("RS_CATALOG", "")                       # gen_models.jl's catalog.toml
    repo::String = envs("RS_REPO", "")                             # the served model repository (default: catalog's)
    park::String = envs("RS_PARK", "")                             # where removed bundles wait (outside the repo)
    out::String = envs("RS_OUT", "rs-out")
    duration::Float64 = envf("RS_DURATION", 300)
    concurrency::Int = envi("RS_CONCURRENCY", 8)
    seed::Int = envi("RS_SEED", 1)
    deadline::Float64 = envf("RS_DEADLINE", 120)                   # per request, seconds
    chaos_every::Float64 = envf("RS_CHAOS_SECONDS", 20)            # mean interval between chaos actions; 0 = none
    chaos::Vector{String} = envlist("RS_CHAOS", "compact,compact_reload,rename,promote,remove_add,update,copy")
    settle::Float64 = envf("RS_SETTLE_SECONDS", 120)               # time the watcher gets to apply a change
    sample_every::Float64 = envf("RS_SAMPLE_SECONDS", 10)
    quiesce_every::Float64 = envf("RS_QUIESCE_SECONDS", 300)       # leak probe period; 0 = none
    stampede::Int = envi("RS_STAMPEDE", 6)                         # concurrent requests fired at one cold model
    pids::Vector{String} = envlist("RS_PIDS")                      # worker PIDs to sample RSS for
    pid_file::String = envs("RS_PID_FILE", "")                     # or a file of PIDs (one per line), re-read each sample
    shm_prefix::String = envs("RS_SHM_PREFIX", "rsw")
    models::Vector{String} = envlist("RS_MODELS")                  # restrict to these catalog models
    max_rows::Int = envi("RS_MAX_ROWS", 4)
end

# ---- catalog and per-model state ------------------------------------------------------------------

mutable struct Model
    key::String                     # catalog name (stable identity)
    cat::Dict{String, Any}
    name::String                    # current served name
    dir::String                     # current bundle directory
    expected::Vector{UInt64}        # checksums the current weights produce
    old::Vector{Vector{UInt64}}     # still acceptable during a transition
    old_names::Vector{String}       # names that may still answer during a transition
    until::Float64                  # end of the current transition (0 = none)
    present::Bool                   # bundle is in the repository (false while parked)
    busy::Bool                      # a chaos action owns it
    ok::Int
    transitional::Int
    errors::Int
    last_sent::Float64
end

parse_cs(v) = UInt64[parse(UInt64, s; base = 16) for s in v]

const LOCK = ReentrantLock()
const MODELS = Dict{String, Model}()           # by catalog key
const FINDINGS = Vector{Dict{String, Any}}()
const KNOWN = Dict{Vector{UInt64}, String}()   # every checksum vector the generator wrote -> label
const T0 = Ref(0.0)
const PAUSE = Atomic{Bool}(false)
const INFLIGHT = Atomic{Int}(0)
const SENT = Atomic{Int}(0)
const OKS = Atomic{Int}(0)
const STOP = Atomic{Bool}(false)
const EVENTS = Ref{IO}(stdout)

now_s() = round(time() - T0[]; digits = 1)

function event(kind, msg; kw...)
    line = string(rpad(string(now_s()), 9), rpad(kind, 10), msg, isempty(kw) ? "" : "  " * join(["$k=$v" for (k, v) in kw], " "))
    lock(LOCK) do
        println(EVENTS[], line)
        flush(EVENTS[])
        kind in ("FINDING", "CHAOS", "QUIESCE", "WARN") && println(stdout, line)
    end
    return nothing
end

function finding!(kind, model, detail; kw...)
    d = Dict{String, Any}("t" => now_s(), "kind" => kind, "model" => model, "detail" => detail)
    for (k, v) in kw
        d[string(k)] = string(v)
    end
    lock(() -> push!(FINDINGS, d), LOCK)
    event("FINDING", "$kind $model: $detail"; kw...)
    return nothing
end

function load_catalog(o::Opts)
    cat = TOML.parsefile(o.catalog)
    for m in cat["model"]
        key = m["name"]
        isempty(o.models) || key in o.models || continue
        if m["kind"] == "checksum"
            cs = parse_cs(m["checksums"])
            KNOWN[cs] = key
            for a in m["alts"]
                KNOWN[parse_cs(a["checksums"])] = "$key@$(a["tag"])"
            end
        else
            cs = UInt64[]
        end
        dir = isempty(o.repo) ? m["dir"] : joinpath(o.repo, basename(m["dir"]))
        MODELS[key] = Model(key, m, key, dir, cs, Vector{UInt64}[], String[], 0.0, isdir(dir), false, 0, 0, 0, 0.0)
    end
    return cat
end

# ---- requests -------------------------------------------------------------------------------------

const _SVC = "/inference.GRPCInferenceService/ModelInfer"
const GRPC = Ref{Any}(nothing)
host_port(t) = (i = findlast(==(':'), t); (t[1:(i - 1)], parse(Int, t[(i + 1):end])))

function infer_client(o::Opts, deadline)
    h, p = host_port(o.target)
    return gRPCClient.gRPCServiceClient{Inf_.ModelInferRequest, false, Inf_.ModelInferResponse, false}(
        h, p, _SVC; grpc = GRPC[], deadline = deadline,
        max_send_message_length = 256 << 20, max_recieve_message_length = 256 << 20
    )
end

function compact!(o::Opts, reload::Vector{String})
    h, p = host_port(o.target)
    c = gRPCClient.gRPCServiceClient{Ctl.CompactMemoryRequest, false, Ctl.CompactMemoryResponse, false}(
        h, p, "/reactant_control.ControlService/CompactMemory"; grpc = GRPC[], deadline = 300
    )
    return gRPCClient.grpc_sync_request(c, Ctl.CompactMemoryRequest(; reload_models = reload))
end

errmsg(e) = e isa gRPCClient.gRPCServiceCallException ? "grpc $(e.grpc_status): $(e.message)" : sprint(showerror, e)
is_deadline(e) = e isa gRPCClient.gRPCServiceCallException && e.grpc_status == gRPCClient.GRPC_DEADLINE_EXCEEDED
# The answers a model gives while the watcher swaps it: not loaded yet, or already unloaded.
is_absent(e) = (m = errmsg(e); occursin("unknown model", m) || occursin("was unloaded", m) || occursin("not found", lowercase(m)) || occursin("no worker", lowercase(m)) || occursin("not served", lowercase(m)))

# One checksum-model request: `rows` nonces (each the sum over a variant's L values). Returns the
# request and the expected per-row nonce sums.
function checksum_request(m::Model, name::String, rows::Int, rng)
    c = m.cat
    batched = !isempty(c["batch_sizes"])
    variants = c["variants"]
    L = isempty(variants) ? 0 : rand(rng, variants)
    n = batched ? rows : 1
    x = rand(rng, UInt64, L == 0 ? n : n * L)        # row-major [n, L]: row b's L values are contiguous
    sums = L == 0 ? x : UInt64[sum(@view x[((b - 1) * L + 1):(b * L)]) for b in 1:n]
    shape = L == 0 ? Int64[n] : Int64[n, L]
    inp = Inf_.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "UINT64", shape = shape)
    req = Inf_.ModelInferRequest(; model_name = name, inputs = [inp], raw_input_contents = [collect(reinterpret(UInt8, x))])
    return req, sums
end

function bit_request(m::Model, name::String)
    c = m.cat
    x = read(c["input"])
    inp = Inf_.var"ModelInferRequest.InferInputTensor"(; name = c["input_name"], datatype = "FP32", shape = Int64.(c["input_shape"]))
    return Inf_.ModelInferRequest(; model_name = name, inputs = [inp], raw_input_contents = [x])
end

# Check a checksum-model response. Returns :ok, :old (a transitional, still-acceptable version) or
# :bad (recorded as a finding).
function check_checksum(m::Model, name, resp, sums, expected, olds, sent_at)
    K = m.cat["outputs"]
    n = length(sums)
    raw = isempty(resp.raw_output_contents) ? UInt8[] : resp.raw_output_contents[1]
    if isempty(resp.outputs) && isempty(resp.raw_output_contents)
        # A success status with nothing in it: the call was not answered at all.
        finding!("EMPTYOK", name, "OK response with no outputs (model_name=$(repr(resp.model_name)))"; sent_at)
        return :bad
    elseif length(raw) != 8 * K * n
        finding!("SHAPE", name, "response has $(length(raw)) bytes, expected $(8 * K * n)"; sent_at)
        return :bad
    end
    y = reinterpret(UInt64, raw)                 # row-major [n, K]
    got = UInt64[y[k] - sums[1] for k in 1:K]
    for b in 2:n
        rowcs = UInt64[y[(b - 1) * K + k] - sums[b] for k in 1:K]
        if rowcs != got
            finding!("ROWMIX", name, "row $b of $n does not carry its own nonce or has other weights"; sent_at)
            return :bad
        end
    end
    got == expected && return :ok
    any(==(got), olds) && return :old
    label = get(KNOWN, got, nothing)
    if label !== nothing
        # Another version of this same model, after its transition window: stale weights.
        stale = label == m.key || startswith(label, m.key * "@")
        finding!(stale ? "STALE" : "WRONGWEIGHTS", name, "served the weights of $label"; expected_key = m.key, sent_at)
    else
        same = count(k -> got[k] == expected[k], 1:K)
        finding!("CORRUPT", name, "$same of $K tensors match; mismatched=$(join(findall(k -> got[k] != expected[k], 1:K)[1:min(end, 8)], ","))"; sent_at)
    end
    return :bad
end

function check_bit(m::Model, name, resp)
    c = m.cat
    raw = isempty(resp.raw_output_contents) ? UInt8[] : resp.raw_output_contents[1]
    y = collect(reinterpret(Float32, raw))
    ref = Float32.(c["reference"])
    if length(y) != length(ref)
        finding!("SHAPE", name, "logits length $(length(y)), expected $(length(ref))")
        return :bad
    end
    rel = sqrt(sum(abs2, y .- ref)) / sqrt(sum(abs2, ref))
    rel <= c["rtol"] && return :ok
    # Which other BiT model's reference does it match (wrong weights)?
    for (k, o) in MODELS
        (o.cat["kind"] == "bit" && k != m.key) || continue
        r2 = Float32.(o.cat["reference"])
        length(r2) == length(y) || continue
        sqrt(sum(abs2, y .- r2)) / sqrt(sum(abs2, r2)) <= c["rtol"] && (finding!("WRONGWEIGHTS", name, "BiT output matches $k"); return :bad)
    end
    finding!("CORRUPT", name, "BiT logits relative error $(round(rel; sigdigits = 3)) > $(c["rtol"])")
    return :bad
end

# Fire one request at model `m` and validate it.
function fire(o::Opts, m::Model, client, rng; rows = rand(rng, 1:o.max_rows))
    # Snapshot what is acceptable when the request is sent; transitions extend acceptance.
    name, expected, olds, oldnames, until = lock(LOCK) do
        open_window = m.until > time()
        nm = (!isempty(m.old_names) && open_window && rand(rng) < 0.3) ? rand(rng, m.old_names) : m.name
        m.last_sent = time()
        # Once the window has closed, the previous weights are no longer acceptable.
        (nm, copy(m.expected), open_window ? copy(m.old) : Vector{UInt64}[], copy(m.old_names), m.until)
    end
    sent_at = now_s()
    kind = m.cat["kind"]
    req, sums = kind == "checksum" ? checksum_request(m, name, rows, rng) : (bit_request(m, name), UInt64[])
    atomic_add!(INFLIGHT, 1)
    atomic_add!(SENT, 1)
    t0 = time()
    local resp
    try
        resp = gRPCClient.grpc_sync_request(client, req)
    catch e
        atomic_add!(INFLIGHT, -1)
        lat = time() - t0
        in_transition = until > t0 || lock(() -> m.until > time(), LOCK) || !lock(() -> m.present, LOCK)
        if in_transition && (is_absent(e) || name in oldnames)
            lock(() -> m.transitional += 1, LOCK)
            event("TRANS", "$name: $(first(errmsg(e), 160))")
        elseif is_deadline(e)
            lock(() -> m.errors += 1, LOCK)
            finding!("DEADLINE", name, "no answer within $(o.deadline)s"; sent_at, latency = round(lat; digits = 1))
        else
            lock(() -> m.errors += 1, LOCK)
            finding!("ERROR", name, first(errmsg(e), 400); sent_at)
        end
        return nothing
    end
    atomic_add!(INFLIGHT, -1)
    r = kind == "checksum" ? check_checksum(m, name, resp, sums, expected, olds, sent_at) : check_bit(m, name, resp)
    lock(LOCK) do
        r === :ok ? (m.ok += 1) : r === :old ? (m.transitional += 1) : (m.errors += 1)
    end
    r === :ok && atomic_add!(OKS, 1)
    return nothing
end

# ---- workload -------------------------------------------------------------------------------------

servable(m::Model) = m.present && !m.busy
function live_models()
    return lock(LOCK) do
        [m for m in values(MODELS) if m.present]
    end
end

# Each worker task cycles through patterns: round robin over every model, a burst on one model,
# and random picks. A separate task periodically stampedes the least recently used model.
function workload(o::Opts, id::Int)
    rng = Xoshiro(o.seed * 1000 + id)
    client = infer_client(o, o.deadline)
    rr = id
    while !STOP[]
        while PAUSE[] && !STOP[]
            sleep(0.05)
        end
        ms = sort!(live_models(); by = m -> m.key)
        isempty(ms) && (sleep(0.1); continue)
        p = rand(rng)
        if p < 0.4                                   # round robin: maximal churn
            rr += 1
            fire(o, ms[mod1(rr, length(ms))], client, rng)
        elseif p < 0.6                               # burst: coalescing and resident hits
            m = rand(rng, ms)
            for _ in 1:rand(rng, 2:8)
                (STOP[] || PAUSE[]) && break
                fire(o, m, client, rng)
            end
        else
            fire(o, rand(rng, ms), client, rng)
        end
    end
    return nothing
end

function stampede(o::Opts)
    rng = Xoshiro(o.seed + 77)
    clients = [infer_client(o, o.deadline) for _ in 1:o.stampede]
    while !STOP[]
        sleep(5 + 10 * rand(rng))
        (STOP[] || PAUSE[]) && continue
        ms = live_models()
        isempty(ms) && continue
        m = argmin(x -> x.last_sent, ms)             # likely evicted from the device
        event("STAMPEDE", m.name; n = o.stampede)
        @sync for c in clients
            @spawn fire(o, m, c, Xoshiro(rand(rng, UInt64)))
        end
    end
    return nothing
end

# ---- chaos: compaction and repository changes -----------------------------------------------------

function pick_idle(rng; kind = nothing, pred = m -> true)
    return lock(LOCK) do
        c = [m for m in values(MODELS) if servable(m) && m.until < time() && pred(m) && (kind === nothing || m.cat["kind"] == kind)]
        isempty(c) ? nothing : rand(rng, c)
    end
end

# Begin a transition on `m`: the old checksums and names stay acceptable until `settle` passes.
function begin_transition!(o::Opts, m::Model; old_cs = nothing, old_name = nothing)
    m.until = time() + o.settle
    old_cs === nothing || push!(m.old, old_cs)
    old_name === nothing || push!(m.old_names, old_name)
    return nothing
end

function end_transitions!()
    lock(LOCK) do
        for m in values(MODELS)
            if m.until != 0 && time() > m.until
                m.until = 0
                empty!(m.old)
                empty!(m.old_names)
            end
        end
    end
    return nothing
end

const RENAMES = Atomic{Int}(0)

function do_rename(o::Opts, rng)
    m = pick_idle(rng)
    m === nothing && return
    lock(LOCK) do
        old = m.name
        new = m.key * "-r" * string(atomic_add!(RENAMES, 1))
        newdir = joinpath(dirname(m.dir), new)
        mv(m.dir, newdir)
        begin_transition!(o, m; old_name = old)
        m.name, m.dir = new, newdir
        event("CHAOS", "rename $old -> $new")
    end
    return
end

# Model-registry promotion chain in one poll: X -> X-old, Y -> X. Name X then serves Y's weights.
function do_promote(o::Opts, rng)
    x = pick_idle(rng; kind = "checksum")
    x === nothing && return
    # Y must answer in X's shape (as a new version of one model would), or answers X's old copy
    # gives during the window could not be told from corruption.
    same_io(m) = all(k -> m.cat[k] == x.cat[k], ("outputs", "batch_sizes", "variants"))
    y = pick_idle(rng; kind = "checksum", pred = m -> m !== x && same_io(m))
    y === nothing && return
    lock(LOCK) do
        xname, yname = x.name, y.name
        xold = xname * "-old" * string(atomic_add!(RENAMES, 1))
        xolddir = joinpath(dirname(x.dir), xold)
        mv(x.dir, xolddir)
        mv(y.dir, x.dir)
        # x keeps its weights under xold; y now answers to xname. x is not sent under its old name:
        # that name now belongs to y.
        begin_transition!(o, x)
        x.name, x.dir = xold, xolddir
        begin_transition!(o, y; old_name = yname, old_cs = nothing)
        y.name, y.dir = xname, joinpath(dirname(xolddir), xname)
        # During the window name xname may still answer with x's weights.
        push!(y.old, x.expected)
        event("CHAOS", "promote $xname -> $xold, $yname -> $xname")
    end
    return
end

function do_remove_add(o::Opts, rng)
    m = pick_idle(rng)
    m === nothing && return
    parked = joinpath(o.park, m.name)
    lock(LOCK) do
        mv(m.dir, parked)
        m.present = false
        m.busy = true
        event("CHAOS", "remove $(m.name)")
    end
    # Requests to it now must fail with "unknown model" once the watcher has unloaded it; probe that.
    @spawn begin
        sleep(o.settle)
        client = infer_client(o, 30)
        try
            req, _ = m.cat["kind"] == "checksum" ? checksum_request(m, m.name, 1, Xoshiro(1)) : (bit_request(m, m.name), nothing)
            gRPCClient.grpc_sync_request(client, req)
            finding!("NOTUNLOADED", m.name, "still answering $(o.settle)s after its bundle was removed")
        catch e
            is_absent(e) || finding!("ERROR", m.name, "after removal: " * first(errmsg(e), 300))
        end
        lock(LOCK) do
            mv(parked, m.dir)
            m.present = true
            m.busy = false
            begin_transition!(o, m)                  # absent until compiled again
            event("CHAOS", "re-add $(m.name)")
        end
    end
    return
end

# Weights-only update: atomically rename an alternate weights file over the bundle's weights file.
function do_update(o::Opts, rng)
    m = pick_idle(rng; kind = "checksum", pred = m -> !isempty(m.cat["alts"]))
    m === nothing && return
    lock(LOCK) do
        alts = [a for a in m.cat["alts"] if parse_cs(a["checksums"]) != m.expected]
        isempty(alts) && return
        a = rand(rng, alts)
        tmp = joinpath(m.dir, ".rs-update.tmp")
        cp(a["path"], tmp; force = true)
        mv(tmp, joinpath(m.dir, "weights.safetensors"); force = true)
        begin_transition!(o, m; old_cs = m.expected)
        m.expected = parse_cs(a["checksums"])
        event("CHAOS", "update $(m.name) -> $(a["tag"])")
    end
    return
end

# Copy a bundle in under a new name: identical weights under a second name.
function do_copy(o::Opts, rng)
    m = pick_idle(rng; kind = "checksum")
    m === nothing && return
    new = m.key * "-c" * string(atomic_add!(RENAMES, 1))
    staging = joinpath(o.park, "." * new)
    cp(m.dir, staging)
    rm(joinpath(staging, ".cache"); force = true, recursive = true)
    lock(LOCK) do
        dir = joinpath(dirname(m.dir), new)
        mv(staging, dir)
        cat = copy(m.cat)
        cat["alts"] = Any[]
        c = Model(new, cat, new, dir, copy(m.expected), Vector{UInt64}[], String[], 0.0, true, false, 0, 0, 0, 0.0)
        begin_transition!(o, c)
        MODELS[new] = c
        event("CHAOS", "copy $(m.name) -> $new")
    end
    return
end

function do_compact(o::Opts, rng; reload::Bool)
    names = String[]
    if reload
        ms = live_models()
        names = unique([rand(rng, ms).name for _ in 1:min(length(ms), rand(rng, 1:4))])
        rand(rng) < 0.3 && push!(names, "rs-no-such-model")
    end
    t0 = time()
    try
        r = compact!(o, names)
        event("CHAOS", "compact"; reload = join(names, "+"), reloaded = r.reloaded_models, seconds = round(time() - t0; digits = 2))
    catch e
        finding!("COMPACT", "-", "CompactMemory failed: " * first(errmsg(e), 300); reload = join(names, "+"))
    end
    return
end

function chaos(o::Opts)
    (o.chaos_every <= 0 || isempty(o.chaos)) && return
    rng = Xoshiro(o.seed + 99)
    while !STOP[]
        sleep(o.chaos_every * (0.5 + rand(rng)))
        STOP[] && break
        PAUSE[] && continue
        end_transitions!()
        a = rand(rng, o.chaos)
        try
            a == "compact" ? do_compact(o, rng; reload = false) :
                a == "compact_reload" ? do_compact(o, rng; reload = true) :
                a == "rename" ? do_rename(o, rng) :
                a == "promote" ? do_promote(o, rng) :
                a == "remove_add" ? do_remove_add(o, rng) :
                a == "update" ? do_update(o, rng) :
                a == "copy" ? do_copy(o, rng) : event("WARN", "unknown chaos action $a")
        catch e
            event("WARN", "chaos $a failed: " * sprint(showerror, e))
        end
    end
    return
end

# ---- monitoring -----------------------------------------------------------------------------------

const SAMPLE_KEYS = [
    "worker_device_memory_in_use_bytes", "worker_device_memory_peak_in_use_bytes",
    "worker_device_memory_largest_free_block_bytes", "worker_device_memory_fragmentation",
    "worker_device_memory_out_of_pool_bytes", "worker_device_memory_process_used_bytes",
    "worker_weight_cache_resident_bytes", "worker_weight_cache_max_bytes",
    "worker_host_weight_cached_bytes", "worker_host_weight_budget_bytes",
    "worker_weight_loads_total", "worker_weight_evicts_total", "worker_weight_compactions_total",
    "worker_weight_fallback_compactions_total", "worker_weight_admission_failures_total",
    "worker_host_weight_hits_total", "worker_host_weight_misses_total", "worker_models_loaded",
    "worker_models_resident",
]

function scrape(url)
    vals = Dict{String, Float64}()
    txt = try
        read(`curl -fsS --max-time 5 $url`, String)
    catch
        return vals
    end
    for line in eachline(IOBuffer(txt))
        (isempty(line) || startswith(line, "#")) && continue
        sp = split(line)
        length(sp) >= 2 || continue
        name = split(sp[1], '{')[1]
        name in SAMPLE_KEYS || continue
        v = tryparse(Float64, sp[2])
        v === nothing && continue
        vals[name] = get(vals, name, 0.0) + v        # summed over label sets (workers)
    end
    return vals
end

function rss_bytes(pid)
    try
        for l in eachline("/proc/$pid/status")
            startswith(l, "VmRSS:") && return parse(Int, split(l)[2]) * 1024
        end
    catch
    end
    return 0
end

function pids(o::Opts)
    ps = copy(o.pids)
    if !isempty(o.pid_file) && isfile(o.pid_file)
        append!(ps, filter(!isempty, strip.(readlines(o.pid_file))))
    end
    return ps
end

function shm_usage(prefix)
    bytes, regions, locks = 0, 0, 0
    for f in readdir("/dev/shm")
        startswith(f, prefix * "-") || continue
        if endswith(f, ".lock")
            locks += 1
        else
            regions += 1
            bytes += filesize(joinpath("/dev/shm", f))
        end
    end
    return bytes, regions, locks
end

function gpu_used()
    # Per-process device memory as the driver sees it (MiB), summed for our worker PIDs.
    try
        txt = read(`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`, String)
        return Dict(strip(split(l, ',')[1]) => parse(Int, strip(split(l, ',')[2])) for l in eachline(IOBuffer(txt)) if count(==(','), l) == 1)
    catch
        return Dict{String, Int}()
    end
end

const SAMPLES = Vector{Dict{String, Any}}()

function sample(o::Opts; tag = "")
    s = Dict{String, Any}("t" => now_s(), "tag" => tag, "sent" => SENT[], "ok" => OKS[])
    for u in o.metrics
        merge!(+, s, scrape(u))
    end
    ps = pids(o)
    s["rss_bytes"] = sum((rss_bytes(p) for p in ps); init = 0)
    g = gpu_used()
    s["gpu_mib"] = sum((get(g, p, 0) for p in ps); init = 0)
    b, r, l = shm_usage(o.shm_prefix)
    s["shm_bytes"], s["shm_regions"], s["shm_locks"] = b, r, l
    lock(() -> push!(SAMPLES, s), LOCK)
    return s
end

const COLUMNS = vcat(["t", "tag", "sent", "ok", "rss_bytes", "gpu_mib", "shm_bytes", "shm_regions", "shm_locks"], SAMPLE_KEYS)

function write_samples(o::Opts)
    open(joinpath(o.out, "samples.csv"), "w") do io
        println(io, join(COLUMNS, ","))
        for s in SAMPLES
            println(io, join([string(get(s, c, "")) for c in COLUMNS], ","))
        end
    end
    return
end

# How long the worker's /metrics takes to answer, once a second. A metrics endpoint that stalls is
# what an operator's dashboards and the gateway's probes see; slow answers are logged with their time
# so they can be lined up with the (timestamped) worker log.
const METRICS_LAT = Float64[]

function metrics_probe(o::Opts)
    isempty(o.metrics) && return
    url = first(o.metrics)
    while !STOP[]
        t0 = time()
        ok = try
            read(`curl -fsS -o /dev/null --max-time 60 $url`)
            true
        catch
            false
        end
        dt = time() - t0
        lock(() -> push!(METRICS_LAT, ok ? dt : Inf), LOCK)
        (dt > 2 || !ok) && event("SLOWMETR", ok ? "" : "failed"; seconds = round(dt; digits = 2), at_unix = round(t0; digits = 1))
        sleep(max(0.0, 1.0 - dt))
    end
    return
end

function monitor(o::Opts)
    while !STOP[]
        sleep(o.sample_every)
        STOP[] && break
        s = sample(o)
        event(
            "SAMPLE", "";
            sent = s["sent"], ok = s["ok"], rss = Base.format_bytes(s["rss_bytes"]), gpu_mib = s["gpu_mib"],
            dev_in_use = Base.format_bytes(round(Int, get(s, "worker_device_memory_in_use_bytes", 0))),
            loads = round(Int, get(s, "worker_weight_loads_total", 0)), evicts = round(Int, get(s, "worker_weight_evicts_total", 0)),
            host_cached = Base.format_bytes(round(Int, get(s, "worker_host_weight_cached_bytes", 0))),
            shm = Base.format_bytes(s["shm_bytes"]), shm_locks = s["shm_locks"]
        )
    end
    return
end

# Leak probe: stop traffic, wait for it to drain, compact with nothing reloaded (only pinned weights
# stay on the device), and record device memory, RSS and /dev/shm. Device memory in use must return
# to the same floor every time.
function quiesce(o::Opts)
    o.quiesce_every <= 0 && return
    n = 0
    while !STOP[]
        slept = 0.0
        while slept < o.quiesce_every && !STOP[]
            sleep(1)
            slept += 1
        end
        STOP[] && break
        PAUSE[] = true
        t = time()
        while INFLIGHT[] > 0 && time() - t < o.deadline
            sleep(0.1)
        end
        n += 1
        try
            compact!(o, String[])
            sleep(2)
            s = sample(o; tag = "quiesce$n")
            event("QUIESCE", "#$n"; dev_in_use = round(Int, get(s, "worker_device_memory_in_use_bytes", -1)), rss = s["rss_bytes"], gpu_mib = s["gpu_mib"], shm = s["shm_bytes"], shm_locks = s["shm_locks"])
        catch e
            finding!("COMPACT", "-", "quiesce compaction failed: " * first(errmsg(e), 300))
        finally
            PAUSE[] = false
        end
    end
    return
end

# ---- main -----------------------------------------------------------------------------------------

function report(o::Opts, elapsed)
    io = IOBuffer()
    println(io, "residency stress report  ", Dates.now())
    println(io, "target=$(o.target) duration=$(round(elapsed; digits = 1))s concurrency=$(o.concurrency) chaos=$(join(o.chaos, ",")) every=$(o.chaos_every)s")
    println(io, "requests sent=$(SENT[]) ok=$(OKS[]) findings=$(length(FINDINGS))")
    if !isempty(METRICS_LAT)
        lat = sort(METRICS_LAT)
        println(io, "/metrics latency: n=$(length(lat)) p50=$(round(lat[cld(end, 2)]; digits = 3))s p99=$(round(lat[cld(99 * end, 100)]; digits = 2))s max=$(round(lat[end]; digits = 2))s over_2s=$(count(>(2), lat))")
    end
    println(io)
    println(io, rpad("model", 34), lpad("ok", 8), lpad("trans", 8), lpad("bad", 6), "  current name")
    for m in sort!(collect(values(MODELS)); by = m -> m.key)
        println(io, rpad(m.key, 34), lpad(m.ok, 8), lpad(m.transitional, 8), lpad(m.errors, 6), "  ", m.name)
    end
    q = [s for s in SAMPLES if startswith(s["tag"], "quiesce")]
    if !isempty(q)
        println(io, "\nquiesce probes (after CompactMemory with no reload):")
        println(io, "  ", rpad("t", 9), lpad("dev_in_use", 14), lpad("rss", 14), lpad("gpu_mib", 9), lpad("shm", 14), lpad("shm_locks", 10))
        for s in q
            println(io, "  ", rpad(string(s["t"]), 9), lpad(string(round(Int, get(s, "worker_device_memory_in_use_bytes", -1))), 14), lpad(string(s["rss_bytes"]), 14), lpad(string(s["gpu_mib"]), 9), lpad(string(s["shm_bytes"]), 14), lpad(string(s["shm_locks"]), 10))
        end
    end
    kinds = Dict{String, Int}()
    for f in FINDINGS
        kinds[f["kind"]] = get(kinds, f["kind"], 0) + 1
    end
    println(io, "\nfindings by kind: ", isempty(kinds) ? "none" : join(["$k=$v" for (k, v) in sort!(collect(kinds))], " "))
    for f in FINDINGS[1:min(end, 200)]
        println(io, "  ", f["t"], "  ", f["kind"], "  ", f["model"], "  ", f["detail"])
    end
    txt = String(take!(io))
    write(joinpath(o.out, "report.txt"), txt)
    open(joinpath(o.out, "findings.toml"), "w") do f
        TOML.print(f, Dict("finding" => FINDINGS))
    end
    write_samples(o)
    print(txt)
    return
end

function main()
    o = Opts()
    isempty(o.catalog) && (println(stderr, "set RS_CATALOG to gen_models.jl's catalog.toml"); return 2)
    mkpath(o.out)
    o = isempty(o.park) ? Opts(; (f => getfield(o, f) for f in fieldnames(Opts) if f != :park)..., park = joinpath(o.out, "parked")) : o
    mkpath(o.park)
    EVENTS[] = open(joinpath(o.out, "events.log"), "w")
    load_catalog(o)
    isempty(MODELS) && (println(stderr, "no models in the catalog"); return 2)
    GRPC[] = gRPCClient.gRPCCURL(; max_streams = max(64, 2 * (o.concurrency + o.stampede)))
    T0[] = time()
    event("START", "models=$(length(MODELS)) target=$(o.target)"; t0_unix = round(T0[]; digits = 1))
    sample(o; tag = "start")
    # Warm pass: one request per model, serially, so each is compiled and probed once before load.
    let client = infer_client(o, max(o.deadline, 300)), rng = Xoshiro(o.seed)
        for m in sort!(collect(values(MODELS)); by = m -> m.key)
            fire(o, m, client, rng; rows = 1)
        end
    end
    tasks = Task[]
    for i in 1:o.concurrency
        push!(tasks, @spawn workload(o, i))
    end
    o.stampede > 0 && push!(tasks, @spawn stampede(o))
    push!(tasks, @spawn chaos(o))
    push!(tasks, @spawn monitor(o))
    push!(tasks, @spawn metrics_probe(o))
    push!(tasks, @spawn quiesce(o))
    t_end = T0[] + o.duration
    while time() < t_end
        sleep(min(5.0, max(0.0, t_end - time())))
    end
    STOP[] = true
    PAUSE[] = false
    for t in tasks
        try
            wait(t)
        catch e
            event("WARN", "task failed: " * sprint(showerror, e))
        end
    end
    sample(o; tag = "end")
    report(o, time() - T0[])
    close(EVENTS[])
    return isempty(FINDINGS) ? 0 : 1
end

abspath(PROGRAM_FILE) == @__FILE__() && exit(main())
