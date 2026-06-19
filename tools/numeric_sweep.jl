# Numeric sweep: for every bundle under /docker/reactantserver/models, serve it,
# generate random inputs for each compiled batch size, run a KServe inference,
# and compare every output against a direct TorchScript forward pass.
#
# Reports per-model / per-batch relative error (max|diff|/max|ref|) and a
# PASS/FAIL verdict, and writes /docker/reactantserver/numeric_sweep_report.md.
#
# Usage:
#   julia --project=examples tools/numeric_sweep.jl [model ...]   # default: all bundles

using HTTP, Sockets, Random, Printf, PythonCall

const HAS_TORCH = try
    pyimport("torch"); pyimport("torch.export"); pyimport("torchax.export")
    pyimport("torchax.ops.jaten"); pyimport("triton._C.libtriton"); pyimport("numpy"); true
catch err
    @info "skip: torch not importable" error=err; false
end
HAS_TORCH || exit(0)

using ReactantServerExport
using ReactantServer
ReactantServerExport._pyimports()

const _Inf = ReactantServer.inference
const torch = pyimport("torch")
const np = pyimport("numpy")
const pygc = pyimport("gc")

const MODELS_ROOT = "/docker/reactantserver/models"
const SOURCES = ["/docker/triton/models", "/docker/triton/dynamic"]
const REPORT = "/docker/reactantserver/numeric_sweep_report.md"
# rtol/atol compare the server against a full-f32 reference on one device. Stripping a bundle's
# baked TF32 (on a non-Ampere target) yields full f32, which matches the reference at least as well
# as TF32 would, so these need no widening. If a TF32-capable device is ever compared against a
# stripped/f32 result, widen rtol to about 2^-10 (~1e-3 already covers it), since TF32 keeps only
# ~10 mantissa bits.
const RTOL = 1.0f-3
const ATOL = 1.0f-3

# Not served on Reactant/XLA (multi-rank inputs); excluded from the sweep.
const EXCLUDE_MODELS = Set([
    "ap_lumbar_autoencoder", "autoencoder_ng_cervical", "autoencoder_ng_lumbar",
])

# --- Triton dtype <-> Julia <-> KServe ---
const TRITON_DTYPE = Dict{String,DataType}(
    "TYPE_UINT8"=>UInt8,"TYPE_UINT16"=>UInt16,"TYPE_UINT32"=>UInt32,"TYPE_UINT64"=>UInt64,
    "TYPE_INT8"=>Int8,"TYPE_INT16"=>Int16,"TYPE_INT32"=>Int32,"TYPE_INT64"=>Int64,
    "TYPE_FP16"=>Float16,"TYPE_FP32"=>Float32,"TYPE_FP64"=>Float64,"TYPE_BOOL"=>Bool)
const KSERVE_OF = Dict{DataType,String}(
    UInt8=>"UINT8",UInt16=>"UINT16",UInt32=>"UINT32",UInt64=>"UINT64",
    Int8=>"INT8",Int16=>"INT16",Int32=>"INT32",Int64=>"INT64",
    Float16=>"FP16",Float32=>"FP32",Float64=>"FP64",Bool=>"BOOL")
const JULIA_OF_KSERVE = Dict(v=>k for (k,v) in KSERVE_OF)

# --- minimal config.pbtxt parsing (matches the converter) ---
function _matched_span(s, openpos, oc, cc)
    depth=0; i=openpos; n=lastindex(s)
    while i<=n
        c=s[i]
        c==oc && (depth+=1)
        c==cc && (depth-=1; depth==0 && return (openpos,i))
        i=nextind(s,i)
    end
    error("unbalanced")
end
function extract_block(t, kw)
    m=match(Regex("\\b"*kw*"\\s*\\["), t); m===nothing && return nothing
    lb=findnext('[',t,m.offset); (o,c)=_matched_span(t,lb,'[',']'); t[nextind(t,o):prevind(t,c)]
end
function extract_objects(b)
    objs=String[]; i=firstindex(b)
    while true
        ob=findnext('{',b,i); ob===nothing && break
        (o,c)=_matched_span(b,ob,'{','}'); push!(objs,b[nextind(b,o):prevind(b,c)]); i=nextind(b,c)
    end
    objs
end
struct IOE; name::String; dtype::String; dims::Vector{Int}; end
function parse_io(t, kw)
    blk=extract_block(t,kw); blk===nothing && return IOE[]
    es=IOE[]
    for obj in extract_objects(blk)
        nm=match(r"name:\s*\"([^\"]+)\"",obj); dt=match(r"data_type:\s*(TYPE_\w+)",obj); dm=match(r"dims:\s*\[([^\]]*)\]",obj)
        nm===nothing && continue
        dims=Int[]; dm!==nothing && for tok in split(dm.captures[1],','); s=strip(tok); isempty(s)||push!(dims,parse(Int,s)); end
        push!(es, IOE(nm.captures[1], dt===nothing ? "" : dt.captures[1], dims))
    end
    es
end

# --- conversions (PyTorch row-major <-> Julia col-major) ---
function julia_to_torch(arr::AbstractArray{T}) where {T}
    bytes=Vector{UInt8}(reinterpret(UInt8, vec(collect(arr))))
    na=np.frombuffer(pybytes(bytes), dtype=ReactantServerExport._julia_to_numpy_dtype(T)).reshape(collect(reverse(size(arr))))
    torch.from_numpy(na.copy())
end
torch_to_julia(t) = (a=t.detach().cpu().contiguous().numpy();
    ReactantServerExport._numpy_to_julia(a, ReactantServerExport._numpy_dtype_to_julia(pyconvert(String, a.dtype.name))))

_free_port() = (s=Sockets.listen(Sockets.localhost,0); p=Int(Sockets.getsockname(s)[2]); close(s); p)
_encode(m)=(io=IOBuffer(); ReactantServer.ProtoBuf.encode(ReactantServer.ProtoBuf.ProtoEncoder(io),m); take!(io))
_decode(::Type{T},b) where{T}=ReactantServer.ProtoBuf.decode(ReactantServer.ProtoBuf.ProtoDecoder(IOBuffer(b)),T)

# random input array for a Triton input spec at batch N (Julia col-major)
function make_input(e::IOE, n::Int, rng)
    T = TRITON_DTYPE[e.dtype]
    sz = (reverse(e.dims)..., n)
    if T <: AbstractFloat
        return rand(rng, T, sz...)
    elseif T == Bool
        return rand(rng, Bool, sz...)
    else
        return rand(rng, T(0):T(min(typemax(T), T(64))), sz...)  # small ints, safe for UINT8/etc
    end
end

# rel error and pass/fail for one (server, ref) output pair
function compare(server, ref)
    size(server)==size(ref) || return (Inf, false, "shape $(size(server)) vs $(size(ref))")
    if eltype(ref) <: AbstractFloat
        s=Float64.(server); r=Float64.(ref)
        d=maximum(abs, s.-r); mr=maximum(abs, r)
        rel = mr>0 ? d/mr : d
        ok = isapprox(server, ref; rtol=RTOL, atol=ATOL)
        return (rel, ok, "")
    else
        mism = count(server .!= ref)
        return (mism==0 ? 0.0 : Float64(mism)/length(ref), mism==0, mism==0 ? "" : "$mism/$(length(ref)) ints differ")
    end
end

function source_for(bundle::String)
    base = endswith(bundle, "__dynamic") ? bundle[1:end-9] : bundle
    for src in SOURCES
        isfile(joinpath(src, base, "1", "model.pt")) || continue
        # the __dynamic suffix always denotes the dynamic/ copy
        endswith(bundle, "__dynamic") && !endswith(src, "dynamic") && continue
        return src, base
    end
    return nothing, base
end

struct Res; model::String; batch::Int; relerrs::Vector{Float64}; ok::Bool; note::String; end

function check_model(bundle::String)
    src, base = source_for(bundle)
    src===nothing && return [Res(bundle, 0, Float64[], false, "source .pt not found")]
    cfg = read(joinpath(src, base, "config.pbtxt"), String)
    inputs = parse_io(cfg, "input")
    bdir = joinpath(MODELS_ROOT, bundle)
    batches = sort([parse(Int, match(r"model\.b(\d+)\.mlir", f).captures[1])
                    for f in readdir(bdir) if occursin(r"^model\.b\d+\.mlir$", f)])
    isempty(batches) && return [Res(bundle, 0, Float64[], false, "no model.b*.mlir")]

    root = mktempdir(); symlink(bdir, joinpath(root, bundle))
    port = _free_port()
    cfgsrv = ReactantServer.ServerConfig([root], "",
        ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
        ReactantServer.SchedulerConfig(30.0, 64, 30.0),
        ReactantServer.EndpointsConfig("127.0.0.1", port))
    srv = nothing
    results = Res[]
    try
        srv = ReactantServer.serve(cfgsrv; backend=ReactantServer.ReactantBackend(), blocking=false)
        sleep(0.5)
        base_url = "http://127.0.0.1:$port"
        ready = HTTP.get("$base_url/v2/models/$bundle/ready"; status_exception=false)
        ready.status==200 || error("model not ready: HTTP $(ready.status)")
        ref = torch.jit.load(joinpath(src, base, "1", "model.pt"), map_location="cpu"); ref.eval()

        for n in batches
            rng = Random.Xoshiro(1234 + n)
            xs = [make_input(e, n, rng) for e in inputs]
            in_tensors = [_Inf.var"ModelInferRequest.InferInputTensor"(;
                            name=inputs[i].name, datatype=KSERVE_OF[eltype(xs[i])],
                            shape=collect(Int64, reverse(size(xs[i])))) for i in eachindex(xs)]
            req = _Inf.ModelInferRequest(; model_name=bundle, inputs=in_tensors,
                    raw_input_contents=[Vector{UInt8}(reinterpret(UInt8, vec(collect(x)))) for x in xs])
            resp = HTTP.post("$base_url/v2/models/$bundle/infer",
                ["Content-Type"=>"application/x-protobuf"], _encode(req);
                status_exception=false, readtimeout=600)
            if resp.status != 200
                body = first(String(copy(resp.body)), 160)
                push!(results, Res(bundle, n, Float64[], false, "infer HTTP $(resp.status): $body")); continue
            end
            rmsg = _decode(_Inf.ModelInferResponse, resp.body)

            py = ref([julia_to_torch(x) for x in xs]...)
            ref_outs = pyconvert(Bool, pybuiltins.isinstance(py, pybuiltins.tuple)) ||
                       pyconvert(Bool, pybuiltins.isinstance(py, pybuiltins.list)) ?
                       [torch_to_julia(o) for o in py] : [torch_to_julia(py)]

            relerrs = Float64[]; allok = true; note = ""
            length(rmsg.outputs)==length(ref_outs) || (allok=false; note="output count $(length(rmsg.outputs)) vs $(length(ref_outs))")
            for k in 1:min(length(rmsg.outputs), length(ref_outs))
                o = rmsg.outputs[k]
                T = JULIA_OF_KSERVE[pyconvert(String, string(o.datatype))]
                jshape = reverse(collect(Int, o.shape))
                sv = reshape(collect(reinterpret(T, rmsg.raw_output_contents[k])), jshape...)
                rel, ok, msg = compare(sv, ref_outs[k])
                push!(relerrs, rel); allok &= ok; isempty(msg)||(note*=" out$k:$msg")
            end
            push!(results, Res(bundle, n, relerrs, allok, strip(note)))
        end
    catch e
        push!(results, Res(bundle, 0, Float64[], false, "exception: " * join(first(split(sprint(showerror,e),'\n'),2), " | ")))
    finally
        srv === nothing || (try ReactantServer.stop!(srv) catch end)
        pygc.collect(); GC.gc()
    end
    isempty(results) && push!(results, Res(bundle, 0, Float64[], false, "no results produced"))
    return results
end

# Resumable result log: one row per (model,batch). Re-running skips models that
# already have rows, so a sweep interrupted by a timeout continues where it left off.
const RESULTS_TSV = "/docker/reactantserver/sweep_results.tsv"

function _append_results(rs::Vector{Res})
    open(RESULTS_TSV, "a") do io
        for r in rs
            errs = join([string(e) for e in r.relerrs], ",")
            println(io, join([r.model, string(r.batch), string(r.ok), errs,
                              replace(r.note, '\t'=>' ', '\n'=>' ')], '\t'))
        end
    end
end

function _load_results()
    res = Res[]
    isfile(RESULTS_TSV) || return res
    for line in eachline(RESULTS_TSV)
        isempty(strip(line)) && continue
        f = split(line, '\t')
        length(f) < 4 && continue
        relerrs = isempty(f[4]) ? Float64[] : [parse(Float64, x) for x in split(f[4], ',') if !isempty(x)]
        note = length(f) >= 5 ? f[5] : ""
        push!(res, Res(String(f[1]), parse(Int, f[2]), relerrs, parse(Bool, f[3]), String(note)))
    end
    return res
end

function main()
    bundles = isempty(ARGS) ? sort([d for d in readdir(MODELS_ROOT) if isdir(joinpath(MODELS_ROOT,d))]) : ARGS
    bundles = [b for b in bundles if !(b in EXCLUDE_MODELS)]
    prior = _load_results()
    done = Set(r.model for r in prior)
    println("Numeric sweep over $(length(bundles)) bundles ($(length(done)) already done)\n"); flush(stdout)
    for (i, b) in enumerate(bundles)
        if b in done
            println(@sprintf("[%3d/%3d] %-50s SKIP (already swept)", i, length(bundles), b)); flush(stdout)
            continue
        end
        rs = check_model(b)
        _append_results(rs)
        for r in rs
            tag = r.ok ? "PASS" : "FAIL"
            errs = isempty(r.relerrs) ? "-" : join([@sprintf("%.1e", e) for e in r.relerrs], ",")
            println(@sprintf("[%3d/%3d] %-50s b=%-2d %-4s relerr=%s %s",
                             i, length(bundles), b, r.batch, tag, errs, r.note))
        end
        flush(stdout)
    end
    write_report(_load_results())
end

function write_report(res::Vector{Res})
    bymodel = Dict{String,Vector{Res}}()
    for r in res; push!(get!(bymodel, r.model, Res[]), r); end
    models = sort(collect(keys(bymodel)))
    passed = [m for m in models if all(r->r.ok, bymodel[m])]
    failed = [m for m in models if !all(r->r.ok, bymodel[m])]
    open(REPORT, "w") do io
        println(io, "# Numeric sweep: server vs TorchScript reference\n")
        println(io, "Tolerance: rtol=$RTOL, atol=$ATOL (floats); exact match (ints/bool).\n")
        println(io, "- models checked: $(length(models))")
        println(io, "- PASS: $(length(passed))")
        println(io, "- FAIL: $(length(failed))\n")
        if !isempty(failed)
            println(io, "## FAILED\n\n| model | batch | rel err | note |")
            println(io, "|---|---|---|---|")
            for m in failed, r in bymodel[m]
                r.ok && continue
                errs = isempty(r.relerrs) ? "-" : join([@sprintf("%.2e",e) for e in r.relerrs], ", ")
                println(io, "| `$m` | $(r.batch) | $errs | $(replace(r.note,"|"=>"/")) |")
            end
            println(io)
        end
        println(io, "## PASSED (worst rel err across batches/outputs)\n")
        println(io, "| model | batches | worst rel err |")
        println(io, "|---|---|---|")
        for m in passed
            rs = bymodel[m]
            worst = maximum([isempty(r.relerrs) ? 0.0 : maximum(r.relerrs) for r in rs])
            bs = join([string(r.batch) for r in rs], ",")
            println(io, "| `$m` | $bs | $(@sprintf("%.2e", worst)) |")
        end
    end
    println("\n=== sweep summary: $(length(passed)) PASS / $(length(failed)) FAIL of $(length(models)) ===")
    isempty(failed) || println("FAILED: ", join(failed, ", "))
    println("report: $REPORT")
end

main()
