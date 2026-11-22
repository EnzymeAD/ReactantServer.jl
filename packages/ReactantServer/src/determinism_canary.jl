# The startup determinism check (`runtime.determinism_check`, on under the regulated profile).
#
# Before serving, every compiled program of every model runs twice on the same synthesized input,
# and the outputs must agree bit for bit. A program that does not is nondeterministic on this
# device and software stack (an atomic reduction, a nondeterministic cuDNN engine, a kernel choice
# carried over from a tuned run), which a validated deployment cannot serve. One failure fails the
# whole worker closed: the models share the device and the stack, so a nondeterministic program in
# one is evidence about all of them. The worker then stays live but never reports ready, and refuses
# inference requests, so the node supervisor (which watches liveness only once a worker has been
# ready) does not restart it in a loop and the gateway, which routes only to ready workers, sends it
# nothing. The cause is in the log, `/readyz` and the metrics.
#
# What it cannot catch: a program that is deterministic but not the one that was validated (a
# different kernel choice, driver or library), which gives the same wrong bits on every run. The
# output digests it logs are for comparing against a validation record.

# Runs per program; the outputs of every run after the first are compared with the first.
const _CANARY_RUNS = 2

# A model's outcome: `:passed`, `:failed`, or `:unchecked` (no inputs can be synthesized, the model
# is not resident under externally-managed residency, or a canary run raised).
determinism_outcomes(s::Scheduler) = lock(() -> copy(s.canary.outcomes), s.canary.lock)

"""
    determinism_failed(scheduler) -> Bool

Whether the startup determinism check found a compiled program whose outputs differed between two
runs on the same input. A worker for which this holds does not report ready and refuses inference
requests with UNAVAILABLE; it stays live, so it is not restarted.
"""
determinism_failed(s::Scheduler) = s.canary.failed[]

function _record_canary!(s::Scheduler, name::AbstractString, o::CanaryOutcome)
    lock(() -> (s.canary.outcomes[String(name)] = o), s.canary.lock)
    o.status === :failed && (s.canary.failed[] = true)
    return o
end

# A canary input value: a fixed sequence in [-1, 1) for floating-point inputs, so a reduction sums
# values whose order matters (zeros, the memory probe's inputs, sum the same in any order), and zero
# for integer and boolean inputs, which are often indices or token ids that must stay in range.
function _canary_array(::Type{T}, dims) where {T}
    T <: AbstractFloat || return zeros(T, dims...)
    a = Array{T}(undef, dims...)
    for k in eachindex(a)
        u = (UInt32(k) * 0x9e3779b1) >> 16             # Fibonacci hashing: spread, reproducible
        a[k] = T(Float32(u) / 32768.0f0 - 1.0f0)
    end
    return a
end

_canary_inputs(entry::ModelEntry, variant::VariantKey, sz::Int) =
    _synth_inputs(_canary_array, entry.manifest.executable_inputs, sz == 0 ? 1 : sz, variant)

_canary_program(variant::VariantKey, sz::Int) =
    isempty(variant) ? "b$sz" : "$(join(variant, "x"))/b$sz"

# The bytes a canary run is compared on: every output's name and raw contents, in output order.
function _canary_bytes(outputs::AbstractVector{NamedTensor})
    io = IOBuffer()
    for t in outputs
        write(io, t.name)
        write(io, t.data)
    end
    return take!(io)
end

# Check one model whose weights are on the device: every compiled (variant, size) runs
# `_CANARY_RUNS` times on the same input. Records and returns the outcome.
function _check_determinism!(s::Scheduler, entry::ModelEntry)
    model = entry.executable
    model === nothing && return nothing
    isempty(entry.manifest.executable_inputs) &&
        return _record_canary!(s, entry.name, CanaryOutcome(:unchecked, "no inputs can be synthesized from the manifest", Pair{String, String}[]))
    digests = Pair{String, String}[]
    for variant in sort!(collect(keys(model.execs))), sz in sort!(collect(keys(model.execs[variant])))
        program = _canary_program(variant, sz)
        inputs = _canary_inputs(entry, variant, sz)
        inputs === nothing && continue
        reference = UInt8[]
        for run in 1:_CANARY_RUNS
            outputs = try
                first(_timed_run(s, model, inputs; offload = true))
            catch err
                @warn "determinism check: a canary run raised; the model is left unchecked" model = entry.name program = program exception = err
                return _record_canary!(s, entry.name, CanaryOutcome(:unchecked, "program $program raised on the canary input: $(sprint(showerror, err))", digests))
            end
            bytes = _canary_bytes(outputs)
            if run == 1
                reference = bytes
            elseif bytes != reference
                return _record_canary!(s, entry.name, CanaryOutcome(:failed, "program $program gave different outputs on run $run than on run 1 for the same input", digests))
            end
        end
        push!(digests, program => bytes2hex(SHA.sha256(reference)))
    end
    return _record_canary!(s, entry.name, CanaryOutcome(:passed, "", digests))
end

# Check every model the memory probe did not (the probe checks each model while it is resident, so
# none is loaded twice): load it if the weight cache has it off the device, then check it.
function _check_determinism_remaining!(s::Scheduler)
    done = determinism_outcomes(s)
    for entry in sort!(collect(values(s.registry.by_name)); by = e -> e.name)
        (entry.executable === nothing || haskey(done, entry.name)) && continue
        if s.weight_cache !== nothing
            try
                acquire!(s.weight_cache, entry)
            catch err
                err isa NotResidentError || rethrow()
                _record_canary!(s, entry.name, CanaryOutcome(:unchecked, "not resident at startup, and externally-managed residency does not load it to check", Pair{String, String}[]))
                continue
            end
        end
        _check_determinism!(s, entry)
    end
    return nothing
end

# Log the outcome: each model's output digests (for the validation record), each model left
# unchecked, and, if anything failed, a bannered error saying what the worker does about it.
function _report_determinism(s::Scheduler)
    outcomes = determinism_outcomes(s)
    names = sort!(collect(keys(outcomes)))
    for name in names
        o = outcomes[name]
        if o.status === :passed
            @info "determinism check: model passed" model = name runs = _CANARY_RUNS outputs = ["$p sha256=$d" for (p, d) in o.digests]
        elseif o.status === :unchecked
            @warn "determinism check: model not checked" model = name reason = o.detail
        end
    end
    failed = [name for name in names if outcomes[name].status === :failed]
    isempty(failed) && return nothing
    banner = "*"^100
    @error """
    $banner
    DETERMINISM CHECK FAILED: $(join(failed, ", "))
    A compiled program gave different outputs for the same input on consecutive runs, so this worker
    cannot serve reproducible results. It stays live but does not report ready, and it refuses
    inference requests; the node supervisor does not restart it. Find the cause (an XLA flag that
    readmits nondeterministic kernels, a kernel choice from the autotune cache, the driver), then
    restart the worker.
    $banner""" details = ["$name: $(outcomes[name].detail)" for name in failed]
    return nothing
end

# Why the worker is not ready, for the `/readyz` body; `nothing` when the check is not the reason.
function determinism_not_ready_reason(s::Scheduler)
    determinism_failed(s) || return nothing
    outcomes = determinism_outcomes(s)
    failed = sort!([name for (name, o) in outcomes if o.status === :failed])
    return "not ready: determinism check failed for $(join(failed, ", "))"
end
