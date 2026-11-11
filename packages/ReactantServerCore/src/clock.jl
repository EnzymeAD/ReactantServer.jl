# The clock every package measures time intervals with.
#
# `time()` reads the wall clock, which the host steps whenever its time is corrected (NTP, an
# operator, a virtual machine resuming). An interval taken across a step is off by the step: a
# forward step expires deadlines early and makes a stall detector see a stall, a backward one
# extends them. The monotonic clock only moves forward, at a steady rate, so intervals, timeouts,
# deadlines and ages are measured on it. `time()` stays only where an absolute time is the point: a
# value shown or exported as a date, compared with a file's timestamps, or persisted.

"""
    monotonic_time() -> Float64

Seconds on the host's monotonic clock: `time_ns()` (libuv's `uv_hrtime`, `CLOCK_MONOTONIC` on
Linux) as a `Float64`. Use it instead of `time()` for anything taken as a difference (an elapsed
time, a timeout, a deadline, an age), since a correction of the wall clock never moves it.

Its zero is arbitrary (on Linux, the host's boot), so a value means something only next to another
`monotonic_time()` from the same host. It is the same clock in every process and container sharing
the host's kernel, but never compare one with `time()` or a file's timestamps, show it as a date,
persist it, or send it to another host.
"""
monotonic_time() = time_ns() / 1.0e9
