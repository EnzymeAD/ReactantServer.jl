# SIGTERM/SIGINT bridge. The C handler is restricted to uv_async_send (async-signal-safe); the
# Julia side waits on the AsyncCondition and runs the graceful shutdown from a normal task.
#
# IMPORTANT: this only works when Julia is started with `--handle-signals=no`. With the default
# signal handling, the runtime consumes SIGTERM on its own signal thread (printing a crash
# report and exiting 143) before any user handler is consulted; signal()/sigaction dispositions
# are never reached. Disabling the runtime handlers is safe only with a single thread: they also
# serve the GC safepoints, so with a second thread running Julia code a collection segfaults.
# Julia 1.12 starts one interactive thread by default (`-t1,1`), so the supervisor must be started
# with `--threads=1,0` as well; `check_signal_threads` refuses to run otherwise. The supervisor is
# a light orchestrator (no Reactant, @async only), so one thread is all it needs; the entrypoint
# passes both flags. Children set PR_SET_PDEATHSIG as the kernel-level backstop, so even an
# ungraceful supervisor death can not orphan them.

const _SHUTDOWN_COND = Ref{Base.AsyncCondition}()

function _on_signal(::Cint)
    # Only uv_async_send here: a signal handler may run on any thread at any point, so no
    # allocation, locks, or Julia runtime entry.
    ccall(:uv_async_send, Cint, (Ptr{Cvoid},), _SHUTDOWN_COND[].handle)
    return nothing
end

"""
    install_term_handlers!(cond) -> nothing

Route SIGTERM and SIGINT to `cond` (a `Base.AsyncCondition`). Effective only under
`julia --handle-signals=no` (see the file comment); a warning is emitted otherwise. Tests drive
shutdown through `request_shutdown!` instead and never install these.
"""
function install_term_handlers!(cond::Base.AsyncCondition)
    if Base.JLOptions().handle_signals == 1
        @warn "Julia's default signal handling is active: SIGTERM will terminate the supervisor without graceful child shutdown (children still exit via PR_SET_PDEATHSIG). Start the supervisor with julia --handle-signals=no for graceful shutdown."
    end
    _SHUTDOWN_COND[] = cond
    h = @cfunction(_on_signal, Cvoid, (Cint,))
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 15, h)   # SIGTERM
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 2, h)    # SIGINT
    return nothing
end

"""
    check_signal_threads(handle_signals=Base.JLOptions().handle_signals,
                         nthreads=Threads.nthreads(:default) + Threads.nthreads(:interactive))

Refuse to run with Julia's signal handling turned off (`--handle-signals=no`) and more than one
thread: the runtime's signal handlers also serve the GC safepoints, so a collection while a second
thread runs Julia code crashes the process (a segfault, often at exit). Throws a `ConfigError`
naming the fix, `--threads=1,0`; returns `nothing` otherwise.
"""
function check_signal_threads(
        handle_signals::Integer = Base.JLOptions().handle_signals,
        nthreads::Integer = Threads.nthreads(:default) + Threads.nthreads(:interactive)
    )
    (handle_signals == 0 && nthreads > 1) || return nothing
    throw(
        ConfigError(
            "the supervisor runs with --handle-signals=no and $nthreads threads; without Julia's " *
                "signal handlers a garbage collection with more than one thread crashes the process. " *
                "Start it with --threads=1,0 (Julia 1.12 adds an interactive thread by default)."
        )
    )
end
