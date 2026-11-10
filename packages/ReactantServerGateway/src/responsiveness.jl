# Event-loop responsiveness. The gateway runs everything as tasks on one thread, by design, so any
# task that holds the thread without yielding (code compiling on first use, a long computation, a
# blocking call) delays every request, probe and scrape behind it. Nothing else measures that delay
# directly: request latency mixes it with the workers' time.
#
# The monitor is one more task on that thread. It asks to wake every `interval` seconds and records
# how late it actually woke: on a free loop that is about a millisecond, and while the thread is
# held it is however long the thread was held. A lateness past `warn_after` is logged and counted.
# It also keeps the longest lateness of the last `LOOP_MONITOR_MAX_WINDOW_SECONDS` as a gauge, so the
# worst single delay in any stretch of time is visible: the histogram and the tick rate average it
# away. A tick is timed with the monotonic clock (`monotonic_time`), so a step of the system clock
# (time synchronization moving it by seconds at once) is not taken for a block.

const LOOP_MONITOR_INTERVAL_SECONDS = 0.1
const LOOP_MONITOR_WARN_SECONDS = 1.0
# Twice the 15 s Prometheus scrape interval, so every tick's lateness is in at least one scrape.
const LOOP_MONITOR_MAX_WINDOW_SECONDS = 30

mutable struct LoopMonitor
    interval::Float64
    warn_after::Float64
    lag::Prometheus.Histogram
    stalls::Prometheus.Counter
    lag_max::Prometheus.Gauge
    # The longest lateness of the ticks that ended in each second of the window, by monotonic second
    # (`window_sec`, slot `mod1(second, window)`). Only the monitor's task writes them.
    window_max::Vector{Float64}
    window_sec::Vector{Int}
    running::Threads.Atomic{Bool}
    task::Union{Task, Nothing}
end

function LoopMonitor(
        metrics::GatewayMetrics; interval::Real = LOOP_MONITOR_INTERVAL_SECONDS,
        warn_after::Real = LOOP_MONITOR_WARN_SECONDS, window::Integer = LOOP_MONITOR_MAX_WINDOW_SECONDS
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
    lag_max = Prometheus.Gauge(
        "gateway_event_loop_lag_max_seconds",
        "The longest lateness of the gateway's event-loop monitor over the last $(window) s: the " *
            "longest single time the gateway's thread was held in that window.";
        registry = metrics.registry
    )
    return LoopMonitor(
        Float64(interval), Float64(warn_after), lag, stalls, lag_max, zeros(Float64, window),
        fill(typemin(Int), window), Threads.Atomic{Bool}(true), nothing
    )
end

# The current second of the monotonic clock.
_monotonic_second() = floor(Int, monotonic_time())

# Observe one tick's lateness, ending in monotonic second `now_s`: record it, update the window's
# longest, and log and count it when it is a stall.
function _observe_lag!(m::LoopMonitor, late::Float64, now_s::Int = _monotonic_second())
    Prometheus.observe(m.lag, late)
    w = length(m.window_max)
    i = mod1(now_s, w)
    if m.window_sec[i] == now_s
        m.window_max[i] = max(m.window_max[i], late)
    else
        m.window_sec[i] = now_s
        m.window_max[i] = late
    end
    longest = 0.0
    for j in 1:w
        now_s - m.window_sec[j] < w && (longest = max(longest, m.window_max[j]))
    end
    Prometheus.set(m.lag_max, longest)
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
        t0 = monotonic_time()
        sleep(m.interval)
        m.running[] || break                  # a tick that wakes after stop is not recorded
        _observe_lag!(m, max(0.0, monotonic_time() - t0 - m.interval))
    end
    return m
end

stop_loop_monitor!(m::LoopMonitor) = (m.running[] = false; nothing)
