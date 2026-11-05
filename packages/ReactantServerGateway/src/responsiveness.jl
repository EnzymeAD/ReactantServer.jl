# Event-loop responsiveness. The gateway runs everything as tasks on one thread, by design, so any
# task that holds the thread without yielding (code compiling on first use, a long computation, a
# blocking call) delays every request, probe and scrape behind it. Nothing else measures that delay
# directly: request latency mixes it with the workers' time.
#
# The monitor is one more task on that thread. It asks to wake every `interval` seconds and records
# how late it actually woke: on a free loop that is about a millisecond, and while the thread is
# held it is however long the thread was held. A lateness past `warn_after` is logged and counted.

const LOOP_MONITOR_INTERVAL_SECONDS = 0.1
const LOOP_MONITOR_WARN_SECONDS = 1.0

mutable struct LoopMonitor
    interval::Float64
    warn_after::Float64
    lag::Prometheus.Histogram
    stalls::Prometheus.Counter
    running::Threads.Atomic{Bool}
    task::Union{Task, Nothing}
end

function LoopMonitor(
        metrics::GatewayMetrics; interval::Real = LOOP_MONITOR_INTERVAL_SECONDS,
        warn_after::Real = LOOP_MONITOR_WARN_SECONDS
    )
    lag = Prometheus.Histogram(
        "gateway_event_loop_lag_seconds",
        "How late the gateway's event-loop monitor woke, per $(interval) s tick: the time the " *
            "gateway's thread was held by other work. About a millisecond when the loop is free.";
        buckets = _LE_BUCKETS, registry = metrics.registry
    )
    stalls = Prometheus.Counter(
        "gateway_event_loop_stalls_total",
        "Monitor ticks that woke more than $(warn_after) s late (each is also logged).";
        registry = metrics.registry
    )
    return LoopMonitor(Float64(interval), Float64(warn_after), lag, stalls, Threads.Atomic{Bool}(true), nothing)
end

# Observe one tick's lateness: record it, and log and count it when it is a stall.
function _observe_lag!(m::LoopMonitor, late::Float64)
    Prometheus.observe(m.lag, late)
    if late > m.warn_after
        Prometheus.inc(m.stalls)
        # Not while precompiling: the build workload runs a gateway that compiles as it goes.
        ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
        @warn "gateway: event loop was blocked; requests, probes and scrapes waited behind it" blocked_s = round(late; digits = 2)
    end
    return nothing
end

"""
    start_loop_monitor!(m::LoopMonitor) -> LoopMonitor

Start the monitor on the calling thread (`@async`, so it shares the thread whose responsiveness it
measures). Stop it with [`stop_loop_monitor!`](@ref).
"""
function start_loop_monitor!(m::LoopMonitor)
    m.task = @async while m.running[]
        t0 = time()
        sleep(m.interval)
        m.running[] || break                  # a tick that wakes after stop is not recorded
        _observe_lag!(m, max(0.0, time() - t0 - m.interval))
    end
    return m
end

stop_loop_monitor!(m::LoopMonitor) = (m.running[] = false; nothing)
