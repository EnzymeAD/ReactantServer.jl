# Interactive-thread responsiveness. The worker's single interactive thread runs the scheduler's
# dispatch loop and every HTTP.jl request task (HTTP.jl starts handlers on the :interactive pool), so
# anything that holds it without yielding (a blocking call, code compiling on first use, real work
# that should have been offloaded) stalls request handling, health checks and dispatch together.
# Request latency cannot isolate that; this measures it directly.
#
# The monitor is one more task on that pool. It asks to wake every `interval` seconds and records how
# late it actually woke: about a millisecond when the thread is free, and however long the thread was
# held otherwise. A lateness past `warn_after` is logged and counted. This is the worker's
# counterpart of the gateway's event-loop monitor.

const INTERACTIVE_MONITOR_INTERVAL_SECONDS = 0.1
const INTERACTIVE_MONITOR_WARN_SECONDS = 1.0

mutable struct InteractiveMonitor
    interval::Float64
    warn_after::Float64
    lag::Prometheus.Histogram
    stalls::Prometheus.Counter
    running::Threads.Atomic{Bool}
    task::Union{Task, Nothing}
end

function InteractiveMonitor(
        registry::Prometheus.CollectorRegistry; interval::Real = INTERACTIVE_MONITOR_INTERVAL_SECONDS,
        warn_after::Real = INTERACTIVE_MONITOR_WARN_SECONDS
    )
    lag = Prometheus.Histogram(
        "worker_interactive_lag_seconds",
        "How late the worker's interactive-thread monitor woke, per $(interval) s tick: the time the " *
            "thread shared by request handling and the dispatch loop was held by other work. About a " *
            "millisecond when it is free.";
        buckets = _WM_BUCKETS, registry = registry
    )
    stalls = Prometheus.Counter(
        "worker_interactive_stalls_total",
        "Monitor ticks that woke more than $(warn_after) s late (each is also logged).";
        registry = registry
    )
    return InteractiveMonitor(Float64(interval), Float64(warn_after), lag, stalls, Threads.Atomic{Bool}(true), nothing)
end

function _observe_lag!(m::InteractiveMonitor, late::Float64)
    Prometheus.observe(m.lag, late)
    if late > m.warn_after
        Prometheus.inc(m.stalls)
        ccall(:jl_generating_output, Cint, ()) == 1 && return nothing   # not while precompiling
        @warn "worker: the interactive thread was blocked; request handling and dispatch waited behind it" blocked_s = round(late; digits = 2)
    end
    return nothing
end

"""
    start_interactive_monitor!(m::InteractiveMonitor) -> InteractiveMonitor

Start the monitor on the interactive thread pool (the default pool when the process has no
interactive thread). Stop it with [`stop_interactive_monitor!`](@ref).
"""
function start_interactive_monitor!(m::InteractiveMonitor)
    tick = () -> while m.running[]
        t0 = time()
        sleep(m.interval)
        m.running[] || break                  # a tick that wakes after stop is not recorded
        _observe_lag!(m, max(0.0, time() - t0 - m.interval))
    end
    m.task = Threads.nthreads(:interactive) > 0 ? Threads.@spawn(:interactive, tick()) : Threads.@spawn(tick())
    return m
end

stop_interactive_monitor!(m::InteractiveMonitor) = (m.running[] = false; nothing)
