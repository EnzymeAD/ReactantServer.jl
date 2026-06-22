# Numerical parity harness for a converted bundle against its original TorchScript model.
#
# For each real sample (.npy at the model's executable input shape, see private/samples/README.md),
# this runs:
#   reference  = the original TorchScript model's full forward (incl. its data-dependent tail), and
#   candidate  = the converted bundle on the CPU backend, through the same preprocess/postprocess
#                its model.jl registers (mirrors the server's infer path),
# then compares output-by-output: integer/bool outputs must match exactly, float outputs within
# rtol/atol. Outputs whose shapes differ are reported as failures with diagnostics. A markdown
# report is written. This is the acceptance gate for each model conversion.
#
# Usage:
#   julia --project=packages/ReactantServerExport/test tools/validate_bundle.jl \
#       <model_dir> <bundle_dir> <samples_dir> [--rtol R] [--atol A] [--report PATH] [--max N]
#
#   <model_dir>    Triton source dir (config.pbtxt + 1/model.pt) -- the reference
#   <bundle_dir>   converted bundle (manifest.yaml, model.b*.mlir, weights.safetensors, model.jl)
#   <samples_dir>  dir of .npy inputs (single-input) or per-sample subdirs (multi-input)
#   --rtol/--atol  float comparison tolerances (default 1e-3 / 1e-4)
#   --max N        cap the number of samples processed

using PythonCall

# torch / torchax / triton must dlopen before Reactant's LLVM loads, or triton's statically
# linked LLVM/MLIR option registration SIGSEGVs when Reactant compiles the bundle. This mirrors
# the import order in tools/convert_to_stablehlo.jl. ReactantServerExport pulls Reactant,
# so the Python stack is imported first.
for m in ("torch", "torch.export", "torchax.export", "torchax.ops.jaten", "triton._C.libtriton", "numpy")
    try
        pyimport(m)
    catch err
        println("FATAL: cannot import '$m'."); println(sprint(showerror, err)[1:min(end, 400)]); exit(1)
    end
end
# Optional: torchvision registers torchvision::nms/roi_align so models that reference those custom ops
# load. Not needed for models that don't, so a missing torchvision is not fatal.
try
    pyimport("torchvision")
catch
end

using ReactantServerExport
using ReactantServer
const RSE = ReactantServerExport

function parse_args(args)
    rtol, atol, report, maxn, refonly, device = 1e-3, 1e-4, "", typemax(Int), false, "cpu"
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--rtol";            rtol = parse(Float64, args[i+1]); i += 2
        elseif a == "--atol";        atol = parse(Float64, args[i+1]); i += 2
        elseif a == "--report";      report = args[i+1]; i += 2
        elseif a == "--max";         maxn = parse(Int, args[i+1]); i += 2
        elseif a == "--device";      device = args[i+1] == "cuda" ? "cuda:0" : args[i+1]; i += 2
        elseif a == "--reference-only"; refonly = true; i += 1
        elseif startswith(a, "--");  error("unknown argument: $a")
        else;                        push!(pos, a); i += 1
        end
    end
    if refonly
        length(pos) == 2 || error("usage (reference-only): validate_bundle.jl <model_dir> <samples_dir> --reference-only [opts]")
        model_dir, bundle_dir, samples_dir = pos[1], "", pos[2]
    else
        length(pos) == 3 || error("usage: validate_bundle.jl <model_dir> <bundle_dir> <samples_dir> [--rtol R] [--atol A] [--report PATH] [--max N] [--reference-only]")
        model_dir, bundle_dir, samples_dir = pos[1], pos[2], pos[3]
    end
    return (; model_dir, bundle_dir, samples_dir, rtol, atol, report, maxn, refonly, device)
end

const opts = parse_args(ARGS)

# --- Python side: parse config inputs, load samples, run the reference model ---
pyexec("""
import os, re, glob
import numpy as np
import torch

def _extract_block(text, keyword):
    m = re.search(keyword + r'\\s*\\[', text)
    if not m: return None
    i, depth = m.end() - 1, 0
    for j in range(i, len(text)):
        if text[j] == '[': depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0: return text[i+1:j]
    return None

def input_specs(model_dir):
    # Returns (names, ranks) where rank is the config dims length (without the batch axis).
    cfg = os.path.join(model_dir, 'config.pbtxt')
    text = open(cfg).read() if os.path.exists(cfg) else ''
    block = _extract_block(text, 'input')
    names, ranks = [], []
    if block:
        depth, start = 0, None
        for j, c in enumerate(block):
            if c == '{':
                if depth == 0: start = j + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    obj = block[start:j]
                    nm = re.search(r'name\\s*:\\s*"([^"]+)"', obj)
                    dims = re.search(r'dims\\s*:\\s*\\[([^\\]]*)\\]', obj)
                    nd = len(re.findall(r'-?\\d+', dims.group(1))) if dims else 0
                    names.append(nm.group(1) if nm else f'INPUT__{len(names)}')
                    ranks.append(nd)
    return names, ranks

def _ensure_batch(arr, rank_no_batch):
    # The executable was compiled with a leading batch axis. Add batch=1 if the sample omits it.
    if arr.ndim == rank_no_batch:
        return arr[None, ...]
    return arr

def list_samples(samples_dir, names, ranks):
    # Returns a list of (sample_id, {name: ndarray}), batch axis ensured. Flat .npy files =>
    # single-input samples; subdirs => one sample each with per-input <name>.npy.
    rank_of = {n: r for n, r in zip(names, ranks)}
    subdirs = sorted([d for d in glob.glob(os.path.join(samples_dir, '*')) if os.path.isdir(d)])
    samples = []
    if subdirs:
        for d in subdirs:
            inp = {}
            for n in names:
                p = os.path.join(d, n + '.npy')
                if not os.path.exists(p):
                    raise FileNotFoundError(f'sample {d} missing input {n}.npy')
                inp[n] = _ensure_batch(np.load(p), rank_of[n])
            samples.append((os.path.basename(d), inp))
    else:
        files = sorted(glob.glob(os.path.join(samples_dir, '*.npy')))
        if len(names) != 1 and files:
            raise ValueError(f'model has {len(names)} inputs; use per-sample subdirs, not flat .npy')
        for p in files:
            samples.append((os.path.basename(p), {names[0]: _ensure_batch(np.load(p), rank_of[names[0]])}))
    return samples

_MODEL = {}
def load_reference(pt_path, device):
    # device is 'cpu' or 'cuda:0' (with CUDA_VISIBLE_DEVICES selecting the physical GPU). Some
    # scripted models bake device='cuda:0' into ops (e.g. an RNN's zero-initialized hidden state),
    # so they must load and run on CUDA. TF32 is disabled so the GPU reference stays full float32,
    # matching the CPU-lowered bundle.
    m = torch.jit.load(pt_path, map_location=device)
    # Some scripted models bake training=False as a constant, so .eval() raises "Can't set constant
    # training". They are already in eval mode; ignore that.
    try:
        m.eval()
    except RuntimeError:
        pass
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _MODEL['m'] = m
    _MODEL['device'] = device
    return True

def run_reference(name_arrays, names):
    m = _MODEL['m']; device = _MODEL['device']
    args = []
    for n in names:
        a = name_arrays[n]
        t = torch.from_numpy(np.ascontiguousarray(a)).to(device)
        args.append(t)
    with torch.no_grad():
        out = m(*args)
    outs = [out] if isinstance(out, torch.Tensor) else list(out)
    # Return list of (numpy_array, dtype_name) for each output tensor.
    res = []
    for o in outs:
        a = o.detach().cpu().numpy()
        res.append((a, str(a.dtype)))
    return res
""", @__MODULE__)

# numpy ndarray (Python) -> Julia Array with reversed (column-major) shape, same dtype. Standalone
# (does not use RSE's _numpy[] Ref, which only the export path initializes). _numpy_dtype_to_julia
# is a pure dtype-name lookup.
function np_to_julia(py_arr)
    np = pyimport("numpy")
    contig = np.ascontiguousarray(py_arr)
    T = RSE._numpy_dtype_to_julia(pyconvert(String, contig.dtype.name))
    shape = reverse(pyconvert(Vector{Int}, contig.shape))
    raw = pyconvert(Vector{UInt8}, contig.tobytes())
    arr = Array{T}(undef, shape...)
    copyto!(reinterpret(UInt8, vec(arr)), raw)
    return arr
end

# Compare one reference/candidate output pair. Returns (ok::Bool, detail::String).
function compare_output(ref::AbstractArray, cand::AbstractArray; rtol, atol)
    size(ref) == size(cand) || return (false, "shape mismatch ref=$(size(ref)) cand=$(size(cand))")
    isempty(ref) && return (true, "empty (n=0), shapes match")
    if eltype(ref) <: Integer || eltype(ref) === Bool
        nbad = count(ref .!= cand)
        return (nbad == 0, nbad == 0 ? "exact int match" : "$nbad/$(length(ref)) integer elements differ")
    end
    rf = Float64.(ref); cf = Float64.(cand)
    if isapprox(rf, cf; rtol=rtol, atol=atol)
        return (true, "within tol (max|Δ|=$(round(maximum(abs.(rf .- cf)); sigdigits=3)))")
    end
    return (false, "max|Δ|=$(round(maximum(abs.(rf .- cf)); sigdigits=3)) exceeds rtol=$rtol atol=$atol")
end

function main()
    pt_path = joinpath(opts.model_dir, "1", "model.pt")
    isfile(pt_path) || error("missing $pt_path")
    specs = pyeval("input_specs", @__MODULE__)(opts.model_dir)
    names = pyconvert(Vector{String}, specs[0])
    samples = pyeval("list_samples", @__MODULE__)(opts.samples_dir, specs[0], specs[1])
    nsamp = pyconvert(Int, pybuiltins.len(samples))
    nsamp == 0 && error("no samples found in $(opts.samples_dir)")
    pyeval("load_reference", @__MODULE__)(pt_path, opts.device)

    # Reference-only mode: run the original model on each sample and report the output shapes,
    # dtypes, and ranges. Used to confirm sample wiring and to learn the real (variable-count)
    # output structure before a converted bundle exists.
    if opts.refonly
        lines = String["# Reference-only run: $(basename(normpath(opts.model_dir)))",
                       "model_dir: `$(opts.model_dir)`  samples=$nsamp  inputs=$(join(names, ", "))", ""]
        nrun = min(nsamp, opts.maxn)
        for si in 0:(nrun - 1)
            sid = pyconvert(String, samples[si][0])
            ref_py = pyeval("run_reference", @__MODULE__)(samples[si][1], names)
            nref = pyconvert(Int, pybuiltins.len(ref_py))
            parts = String[]
            for k in 0:(nref - 1)
                a = np_to_julia(ref_py[k][0])
                dt = pyconvert(String, ref_py[k][1])
                rng = (eltype(a) <: AbstractFloat && !isempty(a)) ?
                    " range[$(round(minimum(a); sigdigits=4)),$(round(maximum(a); sigdigits=4))]" : ""
                push!(parts, "out$k: $dt $(size(a))$rng")
            end
            push!(lines, "- `$sid`: " * join(parts, "; "))
        end
        report_text = join(lines, "\n")
        println(report_text)
        isempty(opts.report) || (write(opts.report, report_text); println("\nreport written to: $(opts.report)"))
        exit(0)
    end

    # Candidate: load the converted bundle on the CPU backend.
    backend = ReactantServer.ReactantBackend()
    pool = ReactantServer.resolve_client(backend,
        ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true))
    parent = dirname(normpath(opts.bundle_dir))
    bname = basename(normpath(opts.bundle_dir))
    reg = ReactantServer.load_bundles([parent]; include=[bname])
    entry = ReactantServer.get_model(reg, bname)
    entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)

    lines = String["# Parity report: $(bname)", "",
                   "model_dir: `$(opts.model_dir)`  bundle: `$(opts.bundle_dir)`",
                   "rtol=$(opts.rtol) atol=$(opts.atol)  samples=$nsamp", ""]
    npass = 0
    nrun = min(nsamp, opts.maxn)
    for si in 0:(nrun - 1)
        sid = pyconvert(String, samples[si][0])
        name_arrays = samples[si][1]
        # Reference outputs (original .pt).
        ref_py = pyeval("run_reference", @__MODULE__)(name_arrays, names)
        nref = pyconvert(Int, pybuiltins.len(ref_py))
        ref_outs = [np_to_julia(ref_py[k][0]) for k in 0:(nref - 1)]

        # Candidate inputs as NamedTensors (client-facing), then preprocess -> run -> postprocess.
        cin = ReactantServer.NamedTensor[]
        for n in names
            ja = np_to_julia(name_arrays[n])
            push!(cin, ReactantServer.NamedTensor(n, ja))
        end
        local cand_outs
        ok_run = true
        detail_run = ""
        try
            prepared = Base.invokelatest(entry.preprocess, cin)
            raw = ReactantServer.run_model(backend, pool, entry.executable, prepared)
            client = Base.invokelatest(entry.postprocess, raw)
            cand_outs = [t.data for t in client]
        catch e
            ok_run = false
            detail_run = sprint(showerror, e)[1:min(end, 200)]
            cand_outs = Any[]
        end

        if !ok_run
            push!(lines, "- ❌ `$sid`: candidate run failed: $detail_run")
            continue
        end
        if length(cand_outs) != length(ref_outs)
            push!(lines, "- ❌ `$sid`: output count ref=$(length(ref_outs)) cand=$(length(cand_outs))")
            continue
        end
        sample_ok = true
        detparts = String[]
        for k in 1:length(ref_outs)
            ok, det = compare_output(ref_outs[k], cand_outs[k]; rtol=opts.rtol, atol=opts.atol)
            sample_ok &= ok
            push!(detparts, "out$(k-1): $(ok ? "✓" : "✗") $det")
        end
        npass += sample_ok ? 1 : 0
        push!(lines, "- $(sample_ok ? "✅" : "❌") `$sid`: " * join(detparts, "; "))
    end

    summary = "## Result: $npass/$nrun samples passed"
    insert!(lines, 5, summary); insert!(lines, 6, "")
    report_text = join(lines, "\n")
    println(report_text)
    if !isempty(opts.report)
        write(opts.report, report_text)
        println("\nreport written to: $(opts.report)")
    end
    exit(npass == nrun ? 0 : 1)
end

main()
