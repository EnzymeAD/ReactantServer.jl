# Interactive-thread responsiveness. The worker's single interactive thread runs the scheduler's
# dispatch loop and every HTTP.jl request task (HTTP.jl starts handlers on the :interactive pool), so
# anything that holds it without yielding (a blocking call, code compiling on first use, real work
# that should have been offloaded) stalls request handling, health checks and dispatch together.
# Request latency cannot isolate that; this measures it directly.
#
# The monitor is one more task on that pool. It asks to wake every `interval` seconds and records how
# late it actually woke: about a millisecond when the thread is free, and however long the thread was
# held otherwise. A lateness past `warn_after` is logged and counted. It also keeps the longest
# lateness of the last `INTERACTIVE_MONITOR_MAX_WINDOW_SECONDS` as a gauge, so the worst single delay
# in any stretch of time is visible: the histogram and the tick rate average it away. A tick is timed
# with the monotonic clock (`monotonic_time`), so a step of the system clock (time synchronization
# moving it by seconds at once) is not taken for a block. This is the worker's counterpart of the
# gateway's event-loop monitor.

const INTERACTIVE_MONITOR_INTERVAL_SECONDS = 0.1
const INTERACTIVE_MONITOR_WARN_SECONDS = 1.0
# Twice the 15 s Prometheus scrape interval, so every tick's lateness is in at least one scrape.
const INTERACTIVE_MONITOR_MAX_WINDOW_SECONDS = 30

mutable struct InteractiveMonitor
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

function InteractiveMonitor(
        registry::Prometheus.CollectorRegistry; interval::Real = INTERACTIVE_MONITOR_INTERVAL_SECONDS,
        warn_after::Real = INTERACTIVE_MONITOR_WARN_SECONDS,
        window::Integer = INTERACTIVE_MONITOR_MAX_WINDOW_SECONDS
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
    lag_max = Prometheus.Gauge(
        "worker_interactive_lag_max_seconds",
        "The longest lateness of the worker's interactive-thread monitor over the last $(window) s: the " *
            "longest single time the thread shared by request handling and the dispatch loop was held " *
            "in that window.";
        registry = registry
    )
    return InteractiveMonitor(
        Float64(interval), Float64(warn_after), lag, stalls, lag_max, zeros(Float64, window),
        fill(typemin(Int), window), Threads.Atomic{Bool}(true), nothing
    )
end

# The current second of the monotonic clock.
_monotonic_second() = floor(Int, monotonic_time())

# Observe one tick's lateness, ending in monotonic second `now_s`: record it, update the window's
# longest, and log and count it when it is a stall.
function _observe_lag!(m::InteractiveMonitor, late::Float64, now_s::Int = _monotonic_second())
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
        t0 = monotonic_time()
        sleep(m.interval)
        m.running[] || break                  # a tick that wakes after stop is not recorded
        _observe_lag!(m, max(0.0, monotonic_time() - t0 - m.interval))
    end
    m.task = Threads.nthreads(:interactive) > 0 ? Threads.@spawn(:interactive, tick()) : Threads.@spawn(tick())
    return m
end

stop_interactive_monitor!(m::InteractiveMonitor) = (m.running[] = false; nothing)
