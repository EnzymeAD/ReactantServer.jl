# Batch-convert models to ReactantServer StableHLO bundles, driven by a YAML config file. Sources are
# either TorchScript models discovered in Triton model repositories (the original use) or models a
# handler builds itself (e.g. a torchvision detector from its weights, with no Triton repo at all).
#
# Usage:
#   julia --project=<env with PythonCall + ReactantServerExport> \
#       tools/convert_to_stablehlo.jl <config.yaml> [--only a,b] [--force] [--dry-run]
#
#   --only a,b   convert only the named models (resume skip still applies;
#                the per-run report is not written)
#   --force      delete the named models' bundles first (requires --only)
#   --dry-run    validate the config, print the resolved worklist with each
#                model's disposition, and exit before loading Python
#
# See tools/convert.example.yaml for the config schema: source_dirs,
# output_root, report_path, exclude (model -> reason / remove_existing), and
# handlers (special-case builder files with per-handler options). Relative
# paths in the config resolve against the config file's directory; handler
# option values whose key ends in _dir or _path are resolved the same way.
#
# For each model the batch ladder is the "floor-division halving" rule: start at
# max_batch_size, repeatedly halve (÷2) down to 1, dedupe, sort; batch size 1 is
# always present. A bundle gets one model.b{N}.mlir per ladder size that traces.
#
# Some models cannot be converted because torch.export cannot trace
# data-dependent behavior (shapes/branches driven by tensor values). Those are
# caught per-model and listed in the report; the run continues. Models that need
# bespoke export logic (e.g. wrapping the TorchScript module) are registered as
# handlers in the config instead of being hard-coded here.
#
# Handler contract: each handler file is included into its own fresh module, and
# its LAST EXPRESSION must evaluate to a function `handler(ctx) -> Vector{Int}`
# returning the batch sizes written (throw on failure). `ctx` is a
# HandlerContext (defined below); `ctx.utils` exposes shared helpers. Handler
# files are loaded after the torch/torchax/triton imports and after
# `using ReactantServerExport`, so they may freely
# `using PythonCall, ReactantServerExport` and call pyexec/pyimport.
#
# Resumable: a model whose bundle already has a manifest, weights, and at least
# one model.b*.mlir is skipped. A handler that exports but fails a later step
# (e.g. copying an extra file into the bundle) leaves a bundle that passes this
# check; recover with `--only <model> --force`.

using Printf
using YAML

# ---------------------------------------------------------------------------
# CLI and config loading. Pure Julia, runs before any Python import so config
# errors and --dry-run never pay torch startup.
# ---------------------------------------------------------------------------
function usage_error(msg)
    println(stderr, "ERROR: $msg")
    println(stderr, "Usage: julia tools/convert_to_stablehlo.jl <config.yaml> [--only a,b] [--force] [--dry-run]")
    exit(2)
end

function parse_cli(args)
    config_path = nothing
    only = String[]
    force = false
    dry_run = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--only"
            i == length(args) && usage_error("--only requires a comma-separated model list")
            append!(only, split(args[i+1], ','; keepempty=false))
            i += 2
        elseif a == "--force"
            force = true
            i += 1
        elseif a == "--dry-run"
            dry_run = true
            i += 1
        elseif startswith(a, "--")
            usage_error("unknown flag $a")
        elseif config_path === nothing
            config_path = a
            i += 1
        else
            usage_error("unexpected argument $a")
        end
    end
    config_path === nothing && usage_error("missing <config.yaml>")
    force && isempty(only) && usage_error("--force requires --only (a full forced rebuild must not happen by accident)")
    return (config_path=abspath(config_path), only=only, force=force, dry_run=dry_run)
end

config_error(msg) = (println(stderr, "config error: $msg"); exit(2))

resolve_path(p::AbstractString, base::AbstractString) =
    isabspath(p) ? String(p) : normpath(joinpath(base, p))

# Handler options are opaque to the converter except for one convention: any
# option whose key ends in _dir or _path is resolved against the config dir.
function resolve_options(opts::AbstractDict, base::AbstractString)
    out = Dict{String,Any}(String(k) => v for (k, v) in opts)
    for (k, v) in out
        if (endswith(k, "_dir") || endswith(k, "_path")) && v isa AbstractString
            out[k] = resolve_path(v, base)
        end
    end
    return out
end

function load_config(path::AbstractString)
    isfile(path) || config_error("config not found: $path")
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    raw isa AbstractDict || config_error("config root must be a mapping")
    base = dirname(abspath(path))

    # source_dirs is optional: a config may convert only handler-declared models that build themselves
    # (e.g. from torchvision weights), with no Triton repository to scan at all.
    srcs = get(raw, "source_dirs", String[])
    (srcs isa AbstractVector && all(s -> s isa AbstractString, srcs)) ||
        config_error("'source_dirs' must be a list of strings")
    out = get(raw, "output_root", nothing)
    out isa AbstractString || config_error("'output_root' must be a string")
    rep = get(raw, "report_path", nothing)
    rep isa AbstractString || config_error("'report_path' must be a string")

    raw_ex = get(raw, "exclude", Dict{String,Any}())
    raw_ex isa AbstractDict || config_error("'exclude' must be a mapping of model name to {reason, remove_existing}")
    exclude = Dict{String,NamedTuple{(:reason, :remove_existing),Tuple{String,Bool}}}()
    for (m, v) in raw_ex
        v isa AbstractDict || config_error("exclude entry '$m' must be a mapping")
        reason = get(v, "reason", nothing)
        reason isa AbstractString || config_error("exclude entry '$m' needs a string 'reason'")
        rmex = get(v, "remove_existing", false)
        rmex isa Bool || config_error("exclude entry '$m': 'remove_existing' must be a boolean")
        exclude[String(m)] = (reason=String(reason), remove_existing=rmex)
    end

    raw_h = get(raw, "handlers", Any[])
    raw_h isa AbstractVector || config_error("'handlers' must be a list")
    handlers = NamedTuple{(:file, :models, :options),Tuple{String,Vector{String},Dict{String,Any}}}[]
    handled = Set{String}()
    for (i, h) in enumerate(raw_h)
        h isa AbstractDict || config_error("handlers[$i] must be a mapping")
        f = get(h, "file", nothing)
        f isa AbstractString || config_error("handlers[$i] needs a string 'file'")
        file = resolve_path(String(f), base)
        isfile(file) || config_error("handlers[$i]: handler file not found: $file")
        ms = get(h, "models", nothing)
        (ms isa AbstractVector && !isempty(ms) && all(x -> x isa AbstractString, ms)) ||
            config_error("handlers[$i] needs a non-empty 'models' list of strings")
        opts = get(h, "options", Dict{String,Any}())
        opts isa AbstractDict || config_error("handlers[$i]: 'options' must be a mapping")
        models = String.(ms)
        for m in models
            m in handled && config_error("model '$m' appears in more than one handler entry")
            haskey(exclude, m) && config_error("model '$m' is in both 'exclude' and 'handlers'")
            push!(handled, m)
        end
        push!(handlers, (file=file, models=models, options=resolve_options(opts, base)))
    end

    (isempty(srcs) && isempty(handlers)) &&
        config_error("config has neither 'source_dirs' nor 'handlers'; nothing to convert")

    return (source_dirs=[resolve_path(String(s), base) for s in srcs],
            output_root=resolve_path(String(out), base),
            report_path=resolve_path(String(rep), base),
            exclude=exclude, handlers=handlers)
end

# ---------------------------------------------------------------------------
# Worklist construction (pure Julia, shared by --dry-run and the real run).
# ---------------------------------------------------------------------------

# A bundle counts as already present if it has a manifest, weights, and at least
# one StableHLO module (so partial bundles are not re-traced on resume).
function bundle_present(dir::AbstractString)
    isdir(dir) || return false
    isfile(joinpath(dir, "manifest.yaml")) || return false
    isfile(joinpath(dir, "weights.safetensors")) || return false
    return any(f -> startswith(f, "model.b") && endswith(f, ".mlir"), readdir(dir))
end

# Scan source dirs, disambiguating cross-directory name collisions.
function build_worklist(cfg)
    worklist = Tuple{String,String,String}[]   # (source, model, bundle_name)
    used_names = Set{String}()
    for source in cfg.source_dirs
        isdir(source) || (println("WARN: source dir missing: $source"); continue)
        for model in sort(readdir(source))
            isdir(joinpath(source, model)) || continue
            bundle_name = model in used_names ? model * "__dynamic" : model
            push!(used_names, bundle_name)
            push!(worklist, (source, model, bundle_name))
        end
    end
    # Handler-declared models with no source directory are converted too: such a handler builds the
    # model itself (e.g. from torchvision weights or a state_dict path passed in its options), so it
    # needs no Triton model dir or config.pbtxt. These carry an empty source; run_handler then builds a
    # context with empty config-derived I/O.
    seen = Set(m for (_, m, _) in worklist)
    for h in cfg.handlers, m in h.models
        m in seen && continue
        push!(seen, m)
        push!(worklist, ("", m, m))
    end
    return worklist
end

function apply_only(worklist, only::Vector{String})
    isempty(only) && return worklist
    keep = filter(t -> t[2] in only || t[3] in only, worklist)
    matched = Set{String}()
    for (_, model, bundle_name) in keep
        model in only && push!(matched, model)
        bundle_name in only && push!(matched, bundle_name)
    end
    unknown = sort!(collect(setdiff(Set(only), matched)))
    isempty(unknown) || usage_error("--only names not found in any source dir: " * join(unknown, ", "))
    return keep
end

handler_file_map(cfg) = Dict{String,String}(m => basename(h.file) for h in cfg.handlers for m in h.models)

const CLI = parse_cli(ARGS)
const CFG = load_config(CLI.config_path)

if CLI.dry_run
    worklist = apply_only(build_worklist(CFG), CLI.only)
    hfile = handler_file_map(CFG)
    println("Dry run: $(length(worklist)) models -> $(CFG.output_root)")
    for (idx, (source, model, bundle_name)) in enumerate(worklist)
        dispo = haskey(CFG.exclude, model) ? "EXCLUDED: $(CFG.exclude[model].reason)" :
                haskey(hfile, model) ? "handler $(hfile[model])" : "generic"
        present = bundle_present(joinpath(CFG.output_root, bundle_name)) ? "present" : "absent "
        @printf("[%3d/%3d] %-55s %s  %s\n", idx, length(worklist), bundle_name, present, dispo)
    end
    exit(0)
end

# ---------------------------------------------------------------------------
# Python init order (CRITICAL): torch / torchax / triton must dlopen BEFORE
# Reactant loads, or Triton's static LLVM/MLIR option registration SIGSEGVs.
# ReactantServerExport -> Reactant, so import the Python stack first. Handler
# files are only loaded after this point, so they inherit the same guarantee.
# ---------------------------------------------------------------------------
using PythonCall
try
    pyimport("torch")
    pyimport("torch.export")
    pyimport("torchax.export")
    pyimport("torchax.ops.jaten")
    pyimport("triton._C.libtriton")
    pyimport("numpy")
catch err
    println("FATAL: torch/torchax/triton not importable in this environment.")
    showerror(stdout, err)
    exit(1)
end

const torch = pyimport("torch")
const pygc = pyimport("gc")

using ReactantServerExport

# ---------------------------------------------------------------------------
# Triton dtype -> Julia type. Restricted to what ReactantServerExport can serialize.
# ---------------------------------------------------------------------------
const TRITON_DTYPE = Dict{String,DataType}(
    "TYPE_UINT8" => UInt8, "TYPE_UINT16" => UInt16, "TYPE_UINT32" => UInt32, "TYPE_UINT64" => UInt64,
    "TYPE_INT8" => Int8, "TYPE_INT16" => Int16, "TYPE_INT32" => Int32, "TYPE_INT64" => Int64,
    "TYPE_FP16" => Float16, "TYPE_FP32" => Float32, "TYPE_FP64" => Float64,
    "TYPE_BOOL" => Bool,
)

# ---------------------------------------------------------------------------
# Minimal config.pbtxt parsing via bracket matching.
# ---------------------------------------------------------------------------
function _matched_span(s::AbstractString, openpos::Int, opench::Char, closech::Char)
    depth = 0
    i = openpos
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == opench
            depth += 1
        elseif c == closech
            depth -= 1
            depth == 0 && return (openpos, i)
        end
        i = nextind(s, i)
    end
    error("unbalanced '$opench'")
end

# Inner text of `keyword [ ... ]`, or nothing if the keyword block is absent.
function extract_block(text::AbstractString, keyword::AbstractString)
    m = match(Regex("\\b" * keyword * "\\s*\\["), text)
    m === nothing && return nothing
    lb = findnext('[', text, m.offset)
    (o, c) = _matched_span(text, lb, '[', ']')
    return text[nextind(text, o):prevind(text, c)]
end

# Inner text of each top-level `{ ... }` object inside a block.
function extract_objects(block::AbstractString)
    objs = String[]
    i = firstindex(block)
    while true
        ob = findnext('{', block, i)
        ob === nothing && break
        (o, c) = _matched_span(block, ob, '{', '}')
        push!(objs, block[nextind(block, o):prevind(block, c)])
        i = nextind(block, c)
    end
    return objs
end

struct IOEntry
    name::String
    dtype_token::String
    dims::Vector{Int}
end

function parse_io_objects(text::AbstractString, keyword::AbstractString)
    block = extract_block(text, keyword)
    block === nothing && return IOEntry[]
    entries = IOEntry[]
    for obj in extract_objects(block)
        nm = match(r"name:\s*\"([^\"]+)\"", obj)
        dt = match(r"data_type:\s*(TYPE_\w+)", obj)
        dm = match(r"dims:\s*\[([^\]]*)\]", obj)
        nm === nothing && continue
        dims = Int[]
        if dm !== nothing
            for tok in split(dm.captures[1], ',')
                t = strip(tok)
                isempty(t) && continue
                push!(dims, parse(Int, t))
            end
        end
        push!(entries, IOEntry(nm.captures[1], dt === nothing ? "" : dt.captures[1], dims))
    end
    return entries
end

function parse_max_batch_size(text::AbstractString)
    m = match(r"max_batch_size:\s*(\d+)", text)
    return m === nothing ? 1 : parse(Int, m.captures[1])
end

# Floor-division halving ladder; always includes 1.
function batch_ladder(maxb::Integer)
    b = max(Int(maxb), 1)
    sizes = Int[]
    while true
        push!(sizes, b)
        b == 1 && break
        b ÷= 2
    end
    return sort!(unique(sizes))
end

# Julia col-major example array for a Triton input at batch 1: PyTorch shape is
# (1, dims...), so the Julia array is zeros(T, reverse(dims)..., 1).
function example_input(entry::IOEntry)
    T = TRITON_DTYPE[entry.dtype_token]
    return zeros(T, reverse(entry.dims)..., 1)
end

# Trim a (possibly huge Python traceback) error message to something report-sized.
function short_error(e)
    msg = sprint(showerror, e)
    lines = split(msg, '\n')
    # Prefer lines that name the exception class or the data-dependent guard.
    interesting = filter(l -> occursin(r"(Error|Exception|data.dependent|GuardOn|Unsupported|cannot|could not)"i, l), lines)
    picked = isempty(interesting) ? first(lines, 3) : first(interesting, 3)
    out = strip(join(picked, " | "))
    return length(out) > 500 ? out[1:500] * " …" : out
end

# ---------------------------------------------------------------------------
# Per-model conversion.
# ---------------------------------------------------------------------------
struct Record
    source::String
    model::String
    bundle_name::String
    outcome::Symbol            # :success :partial :failed :skipped
    sizes::Vector{Int}         # batch sizes actually written
    detail::String
end

function convert_model(source::AbstractString, model::AbstractString, bundle_name::AbstractString,
                       output_root::AbstractString)
    cfg_path = joinpath(source, model, "config.pbtxt")
    pt_path = joinpath(source, model, "1", "model.pt")
    isfile(cfg_path) || return Record(source, model, bundle_name, :failed, Int[], "config.pbtxt not found")
    isfile(pt_path) || return Record(source, model, bundle_name, :failed, Int[], "1/model.pt not found")

    text = read(cfg_path, String)
    inputs = parse_io_objects(text, "input")
    outputs = parse_io_objects(text, "output")
    isempty(inputs) && return Record(source, model, bundle_name, :failed, Int[], "no inputs parsed from config")

    # Reject unsupported / dynamic inputs up front.
    for e in inputs
        haskey(TRITON_DTYPE, e.dtype_token) ||
            return Record(source, model, bundle_name, :failed, Int[], "unsupported input dtype $(e.dtype_token)")
        any(d -> d < 0, e.dims) &&
            return Record(source, model, bundle_name, :skipped, Int[],
                          "dynamic input shape, no concrete size available (dims=$(e.dims))")
    end

    ladder = batch_ladder(parse_max_batch_size(text))
    in_names = [e.name for e in inputs]
    out_names = [e.name for e in outputs]
    ex = Tuple(example_input(e) for e in inputs)
    out_dir = joinpath(output_root, bundle_name)

    do_export(sizes) = ReactantServerExport.export_torchscript_bundle(pt_path, ex;
        dir=out_dir, name=bundle_name,
        input_names=in_names,
        output_names=isempty(out_names) ? nothing : out_names,
        batch_sizes=sizes)

    # Try the full ladder; if that throws and the ladder has more than just b1,
    # fall back to b1 alone so a usable bundle is still written.
    isdir(out_dir) && rm(out_dir; recursive=true)
    try
        do_export(ladder)
        return Record(source, model, bundle_name, :success, ladder, "")
    catch e_full
        full_err = short_error(e_full)
        if ladder == [1]
            return Record(source, model, bundle_name, :failed, Int[], full_err)
        end
        isdir(out_dir) && rm(out_dir; recursive=true)
        try
            do_export([1])
            dropped = filter(!=(1), ladder)
            return Record(source, model, bundle_name, :partial, [1],
                          "batch sizes $dropped failed to re-trace: $full_err")
        catch e1
            return Record(source, model, bundle_name, :failed, Int[], short_error(e1))
        end
    end
end

# ---------------------------------------------------------------------------
# Handlers: per-model special-case builders registered in the config.
# ---------------------------------------------------------------------------
struct HandlerContext
    source::String              # source dir containing the model
    model::String               # Triton model directory name
    bundle_name::String         # output bundle name (may carry __dynamic suffix)
    out_dir::String             # joinpath(output_root, bundle_name)
    config_text::String         # raw config.pbtxt contents
    inputs::Vector{IOEntry}
    outputs::Vector{IOEntry}
    ladder::Vector{Int}         # batch ladder from this model's max_batch_size
    options::Dict{String,Any}   # this handler's options (_dir/_path values pre-resolved)
    utils::NamedTuple           # shared helpers, see run_handler
end

# Include a handler file into its own fresh module (isolates helper names and
# pyexec state between handler files) and return its registration value.
function load_handler(path::AbstractString, idx::Int)
    isfile(path) || error("handler file not found: $path")
    mod = Module(Symbol(:Handler, idx))
    fn = Base.include(mod, path)
    fn isa Function ||
        error("handler $path: last expression must be a function handler(ctx) -> Vector{Int}, got $(typeof(fn))")
    return fn
end

function run_handler(fn::Function, options::Dict{String,Any}, hname::AbstractString,
                     source::AbstractString, model::AbstractString, bundle_name::AbstractString,
                     output_root::AbstractString)
    cfg_path = joinpath(source, model, "config.pbtxt")
    if isfile(cfg_path)
        text = read(cfg_path, String)
        inputs = parse_io_objects(text, "input")
        outputs = parse_io_objects(text, "output")
        ladder = batch_ladder(parse_max_batch_size(text))
    else
        # Source-free handler model (no Triton dir / config.pbtxt): the handler builds the model from
        # its options and declares its own I/O, so the config-derived fields are empty.
        text = ""
        inputs = IOEntry[]
        outputs = IOEntry[]
        ladder = Int[1]
    end
    ctx = HandlerContext(source, model, bundle_name, joinpath(output_root, bundle_name),
                         text, inputs, outputs, ladder, options,
                         (; batch_ladder, parse_max_batch_size, example_input, short_error))
    try
        # invokelatest: the handler function was defined by `Base.include` in a newer world age
        # than this call site; Julia 1.12's strict world-age semantics otherwise reject it.
        sizes = Base.invokelatest(fn, ctx)
        return Record(source, model, bundle_name, :success, collect(Int, sizes), "handler $hname")
    catch e
        return Record(source, model, bundle_name, :failed, Int[], "handler $hname: " * short_error(e))
    end
end

# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
function main(cfg, only::Vector{String}, force::Bool)
    mkpath(cfg.output_root)
    worklist = apply_only(build_worklist(cfg), only)

    # model -> (handler function, options, file basename)
    handler_for = Dict{String,Tuple{Function,Dict{String,Any},String}}()
    for (idx, h) in enumerate(cfg.handlers)
        fn = load_handler(h.file, idx)
        for m in h.models
            handler_for[m] = (fn, h.options, basename(h.file))
        end
    end

    records = Record[]
    total = length(worklist)
    println("Converting $total models -> $(cfg.output_root)\n")
    flush(stdout)

    for (idx, (source, model, bundle_name)) in enumerate(worklist)
        prefix = @sprintf("[%3d/%3d] %-55s", idx, total, bundle_name)
        out_dir = joinpath(cfg.output_root, bundle_name)
        if haskey(cfg.exclude, model)
            ex = cfg.exclude[model]
            ex.remove_existing && isdir(out_dir) && rm(out_dir; recursive=true)
            println("$prefix EXCLUDED")
            flush(stdout)
            push!(records, Record(source, model, bundle_name, :skipped, Int[], ex.reason))
            continue
        end
        force && isdir(out_dir) && rm(out_dir; recursive=true)
        if bundle_present(out_dir)
            println("$prefix SKIP (already present)")
            flush(stdout)
            push!(records, Record(source, model, bundle_name, :success, Int[], "already present (not re-converted)"))
            continue
        end

        special = get(handler_for, model, nothing)
        println(special === nothing ? "$prefix converting …" : "$prefix converting via $(special[3]) …")
        flush(stdout)
        rec = try
            special === nothing ?
                convert_model(source, model, bundle_name, cfg.output_root) :
                run_handler(special..., source, model, bundle_name, cfg.output_root)
        catch e
            Record(source, model, bundle_name, :failed, Int[], "driver error: " * short_error(e))
        end
        push!(records, rec)

        tag = rec.outcome == :success ? "OK     b=$(rec.sizes)" :
              rec.outcome == :partial ? "PARTIAL b=$(rec.sizes)" :
              rec.outcome == :skipped ? "SKIPPED" : "FAILED"
        println("$prefix $tag")
        rec.outcome in (:failed, :skipped, :partial) && !isempty(rec.detail) && println("          → $(rec.detail)")
        flush(stdout)

        # Keep memory bounded across the sequential model loads.
        pygc.collect()
        GC.gc()
    end

    if isempty(only)
        write_report(records, cfg.report_path, cfg.output_root)
    else
        println("\n(--only run: report not written)")
    end
    print_summary(records)
    return records
end

function _counts(records)
    c = Dict(:success => 0, :partial => 0, :failed => 0, :skipped => 0)
    for r in records
        c[r.outcome] += 1
    end
    return c
end

function write_report(records, report_path::AbstractString, output_root::AbstractString)
    c = _counts(records)
    open(report_path, "w") do io
        println(io, "# Model → StableHLO conversion report\n")
        println(io, "Output root: `$output_root`\n")
        println(io, "| outcome | count |")
        println(io, "|---------|-------|")
        println(io, "| success | $(c[:success]) |")
        println(io, "| partial | $(c[:partial]) |")
        println(io, "| failed  | $(c[:failed]) |")
        println(io, "| skipped | $(c[:skipped]) |")
        println(io, "| total   | $(length(records)) |\n")

        for (title, sym) in (("Failed", :failed), ("Skipped", :skipped), ("Partial (bundle written, some batch sizes dropped)", :partial))
            rows = filter(r -> r.outcome == sym, records)
            isempty(rows) && continue
            println(io, "## $title\n")
            println(io, "| model | source | reason |")
            println(io, "|-------|--------|--------|")
            for r in rows
                src = basename(r.source)
                detail = replace(r.detail, "|" => "\\|", "\n" => " ")
                println(io, "| `$(r.bundle_name)` | $src | $detail |")
            end
            println(io)
        end

        ok = filter(r -> r.outcome == :success, records)
        if !isempty(ok)
            println(io, "## Succeeded\n")
            for r in ok
                sizes = isempty(r.sizes) ? "(present)" : "b=$(r.sizes)"
                note = isempty(r.detail) ? "" : " ($(r.detail))"
                println(io, "- `$(r.bundle_name)` $sizes$note")
            end
        end
    end
    println("\nReport written to $report_path")
end

function print_summary(records)
    c = _counts(records)
    println("\n=== Conversion summary ===")
    println("success: $(c[:success])   partial: $(c[:partial])   failed: $(c[:failed])   skipped: $(c[:skipped])   total: $(length(records))")
    for (title, sym) in (("FAILED", :failed), ("SKIPPED", :skipped), ("PARTIAL", :partial))
        rows = filter(r -> r.outcome == sym, records)
        isempty(rows) && continue
        println("\n--- $title ---")
        for r in rows
            println("  $(r.bundle_name): $(r.detail)")
        end
    end
end

main(CFG, CLI.only, CLI.force)
