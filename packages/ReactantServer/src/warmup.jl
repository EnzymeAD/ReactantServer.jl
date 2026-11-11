# Request-path warm-up: compile, before a model serves, the Julia code its first request would
# otherwise compile on the way through the worker.
#
# The startup probe (and the fair discipline's cost warmup) runs each compiled program, but only the
# program: `run_model` on zero-filled executable inputs. A request also runs the bundle's model.jl
# hooks, which live in a module created when the bundle loads and so are never in a package image,
# and worker code specialized on the model's tensor types: the codec's decode and encode, and on the
# dispatch loop the coalescing, padding and output slicing. On a CPU worker serving the
# privacy-filter bundle, the first request compiled for 2.9 s: 2.5 s in its hooks, 0.2 s slicing its
# rank-3 logits on the dispatch loop (the interactive thread), and 0.1 s in the codec for its rank-1
# Int32 and Float32 tensors. Its first two-text request compiled the padding for 0.27 s more, also on
# the dispatch loop.
#
# The warm-up runs that path once per model on a request synthesized from the manifest: zeros at one
# row, each variable axis at the first compiled shape variant's size, or 1 where the manifest gives
# none. It calls the stages directly, not through the queue, and runs no program: the dispatch
# stages get zero outputs of the executable's declared shapes, which have the element types and
# ranks `run_model` returns. So it touches nothing scheduling reads or reports (no queue, cost or EMA
# estimate, fairness charge, residency, device memory, or request, dispatch and latency metric), and
# it needs no compiled program, so a hot load can warm a model before compiling it.
#
# Hooks are expected to be functions of their inputs: the request path already runs them
# concurrently on many tasks and threads, and the bundle guide asks for no shared mutable state.
# The warm-up's hooks run on their own task, so state a hook keeps in task-local storage between
# its preprocess and postprocess (as the privacy filter does) stays with that task. A meta model is
# not warmed: its orchestration calls other models through the scheduler.
#
# Zeros reach the hook code a first request would compile when that code is type-stable, since a
# method compiles with everything it calls statically. Code reached only through a dynamic call that
# zeros never make stays for the first request: for the privacy filter, the span decoding that runs
# only when a text has tokens, about 0.5 s of the 2.5 s. A hook may also reject zeros. The warm-up
# then logs it and goes on with the stages that do not depend on the hook (the dispatch stages on
# zero executable inputs, the encode on zero client outputs), and the model serves as usual: as
# with the probe, a failed warm-up costs only the compile it would have saved.

# How long one model's warm-up may take before startup (or a hot load) goes on without it. Model
# code runs on synthesized input here, and a hook that never returns on it must not keep the worker
# from serving; a warm-up still running then finishes in the background.
const WARMUP_TIMEOUT_SECONDS = 60.0

# The compile counter, when timing is on (`enable_compile_timing!`; `serve` turns it on).
_warm_compile_ns() = Int64(first(Base.cumulative_compile_time_ns()))

"""
    warm_request_path!(entry::ModelEntry; timeout=WARMUP_TIMEOUT_SECONDS) -> NamedTuple

Run `entry`'s request path once, on a request synthesized from its manifest, so the code a first
request would compile is compiled now: the bundle's preprocess and postprocess hooks and the
worker's codec, coalescing and output slicing for the model's tensor types. Scheduling never sees it,
nor do the request, dispatch and latency metrics, and it runs no program, so `entry` need not be
compiled yet.

It runs on a default-pool task, as [`off_interactive`](@ref) runs long work, so never on the
interactive thread, and unlike `off_interactive` the caller waits for it at most `timeout` seconds.
It never throws: a stage that fails is logged and the rest still run. Returns
`(; complete, seconds, compile_seconds)`, `complete` false when a stage failed or the time ran out.
"""
function warm_request_path!(entry::ModelEntry; timeout::Real = WARMUP_TIMEOUT_SECONDS)
    t0 = monotonic_time()
    c0 = _warm_compile_ns()
    # Woken when the stages finish or the time runs out, whichever is first; `finished` tells which.
    done = Base.Event()
    finished = Threads.Atomic{Bool}(false)
    result = Ref{Any}(nothing)
    Threads.@spawn :default begin
        try
            result[] = _warm_stages(entry)
        catch err                         # `_warm_stages` catches its own failures; this is a bug
            result[] = ("warm-up", err)
        finally
            finished[] = true
            notify(done)
        end
    end
    timer = Timer(_ -> notify(done), Float64(timeout))
    try
        wait(done)
    finally
        close(timer)
    end
    seconds = monotonic_time() - t0
    compile_seconds = (_warm_compile_ns() - c0) / 1.0e9
    if !finished[]
        @warn "warm-up: model not warmed within $(timeout) s; it serves now, and its first request compiles what the warm-up has not" model = entry.name
        return (; complete = false, seconds, compile_seconds)
    end
    failed = result[]
    if failed === nothing
        @debug "warm-up: model warmed" model = entry.name seconds = round(seconds; digits = 2) compile_seconds = round(compile_seconds; digits = 2)
    else
        stage, err = failed
        @info "warm-up: model partly warmed; its $(stage) failed on the synthesized request, and its first request compiles what was not reached" model = entry.name exception = err
    end
    return (; complete = failed === nothing, seconds, compile_seconds)
end

"""
    warm_request_paths!(registry::ModelRegistry; timeout=WARMUP_TIMEOUT_SECONDS) -> Int

Warm every regular model in `registry` ([`warm_request_path!`](@ref)), one after another, and log
the total. Returns the number of models warmed completely.
"""
function warm_request_paths!(registry::ModelRegistry; timeout::Real = WARMUP_TIMEOUT_SECONDS)
    # One at a time: over the 96 bundles of a production worker, 16 at once took 8 s instead of 15 s
    # while compiling on every thread, and the node's other workers start at the same time.
    entries = sort!(collect(values(registry.by_name)); by = e -> e.name)
    isempty(entries) && return 0
    t0 = monotonic_time()
    c0 = _warm_compile_ns()
    partial = String[]
    for entry in entries
        warm_request_path!(entry; timeout = timeout).complete || push!(partial, entry.name)
    end
    @info "warm-up: compiled each model's request path before serving" models = length(entries) seconds = round(monotonic_time() - t0; digits = 2) compile_seconds = round((_warm_compile_ns() - c0) / 1.0e9; digits = 2) partly_warmed = partial
    return length(entries) - length(partial)
end

# The stages, in request order. Returns `nothing`, or `(stage, exception)` for the first that failed.
function _warm_stages(entry::ModelEntry)
    m = entry.manifest
    variant, sizes, unbatched = _warm_programs(entry)
    failed = nothing
    # The transport: decode the client's tensors from a raw request and validate them.
    decoded = try
        client_variant = m.client_inputs === nothing ? variant : nothing
        client = something(
            _synth_zeros(client_input_spec(m), 1, client_variant),
            _synth_zeros(client_input_spec(m), 1, nothing)
        )
        d = decode_infer_request(inference, encode_infer_request(inference, entry.name, client), nothing)
        _validate_inputs(entry, d.request)
        d
    catch err
        failed = ("request decode", err)
        nothing
    end
    # The preprocess hook, as `infer` calls it.
    prepared = nothing
    if decoded !== nothing
        prepared = try
            p = Base.invokelatest(entry.preprocess, decoded.request.inputs)
            _check_variant_inputs(entry, p)
            p
        catch err
            failed = something(failed, ("preprocess", err))
            nothing
        end
    end
    # The dispatch loop's work on the model's tensors. Without preprocess output it runs on zero
    # executable inputs, the tensors preprocess has to produce.
    results = try
        exec_inputs = prepared === nothing ? _zero_inputs(entry, variant, unbatched ? 0 : 1) : prepared
        exec_inputs === nothing ? nothing : _warm_dispatch(entry, exec_inputs, sizes, unbatched)
    catch err
        failed = something(failed, ("dispatch", err))
        nothing
    end
    # The postprocess hook, only after its preprocess ran, as on the request path.
    outputs = nothing
    if prepared !== nothing && results !== nothing
        outputs = try
            Base.invokelatest(entry.postprocess, results)
        catch err
            failed = something(failed, ("postprocess", err))
            nothing
        end
    end
    # The transport again: encode the response, from zero client outputs when the hook gave none.
    if decoded !== nothing
        try
            outputs isa Vector{NamedTensor} || (outputs = _synth_zeros(client_output_spec(m), 1, nothing))
            _encode_message(encode_infer_response(inference, entry.name, decoded, outputs, nothing; deadline_ns = 0))
        catch err
            failed = something(failed, ("response encode", err))
        end
    end
    return failed
end

# The coalescing, padding and slicing a dispatch does, on `prepared` (one request's executable
# inputs) and zero outputs at the batch size it would run at. A coalescable model also pads a lone
# request up to the next compiled size and concatenates two requests (with padding when no size
# fits them exactly), and an oversize request's split and join runs on two pieces. Returns the
# request's sliced outputs: what the dispatch hands its postprocess.
function _warm_dispatch(entry::ModelEntry, prepared::Vector{NamedTensor}, sizes::Vector{Int}, unbatched::Bool)
    m = entry.manifest
    rows = _executable_rows(entry, prepared)
    B = unbatched ? 1 : rows
    coalescable = !unbatched && !isempty(m.executable_inputs) &&
        all(sp -> sp.batch_axis !== nothing, m.executable_inputs) &&
        all(o -> o.batch_axis !== nothing, m.executable_outputs)
    if coalescable && rows >= 1 && _batch_extents_agree(entry, prepared)
        B = something(_smallest_ge(sizes, rows), rows)
        B > rows && _coalesce_inputs(entry, [prepared], rows, B)
        two = _smallest_ge(sizes, 2rows)
        two === nothing || _coalesce_inputs(entry, [prepared, prepared], 2rows, two)
    end
    out = _synth_zeros(m.executable_outputs, max(B, 1), nothing)
    results = _slice_outputs(entry, out, 0, rows)
    if coalescable
        _take_rows(prepared, m.executable_inputs, 1:rows)
        _join_outputs(entry, [results, results])
    end
    return results
end

# The shape variant and batch sizes to warm (the first variant, its sizes ascending) and whether any
# program is a single unbatched module, from the bundle's programs, which a hot load has before it
# compiles them. Hand-built entries without program bytes fall back to their compiled executables,
# and then to the manifest's declared sizes.
function _warm_programs(entry::ModelEntry)
    progs = (all(isempty, values(entry.mlir_bytes)) && entry.executable !== nothing) ?
        entry.executable.execs : entry.mlir_bytes
    all(isempty, values(progs)) && return (VariantKey(), sort(entry.manifest.batching.compiled_batch_sizes), false)
    variant = minimum(k for (k, inner) in progs if !isempty(inner))
    unbatched = any(inner -> haskey(inner, 0), values(progs))
    return (variant, sort!(collect(keys(progs[variant]))), unbatched)
end
