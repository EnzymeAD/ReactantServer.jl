# Does a weight upload fit wherever the allocator statistics say it does? Admission (admission.jl)
# admits a model when free bytes and the largest free block cover its weights; this probe builds
# known free-block layouts in a small preallocated BFC arena and uploads tensors that should fit,
# reporting each outcome against the statistics admission reads.
#
#   gpu-lease -need 1 -wait -timeout 5m \
#     julia --project=packages/ReactantServer tools/residency_stress/alloc_probe.jl [FRACTION]
#
# FRACTION is runtime.device_memory_fraction (default 0.02, about 1 GiB on a 48 GB card).

using ReactantServer: ReactantServer
const RS = ReactantServer
const RX = RS.Reactant.XLA

frac = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.02
backend = RS.ReactantBackend()
client = RS.make_client(backend, "cuda"; device_memory_fraction = frac, preallocate = true)
device = RS.select_device(backend, client, 0)

function stats()
    s = RX.allocatorstats(device)
    limit = Int(s.bytes_limit)
    return (
        in_use = Int(s.bytes_in_use), limit = limit, free = limit - Int(s.bytes_in_use),
        largest = Int(s.largest_free_block_bytes), pool = something(s.pool_bytes, 0),
    )
end
show_stats(tag) = (s = stats(); println(rpad(tag, 44), " in_use=", s.in_use, " free=", s.free, " largest=", s.largest, " pool=", s.pool); s)

upload(nbytes) = RS.to_device(backend, client, zeros(UInt8, nbytes), device)
free!(b) = RS.free_buffer!(backend, b)

function try_upload(tag, sizes)
    bufs = Any[]
    s0 = stats()
    t0 = time()
    ok = true
    err = ""
    for n in sizes
        try
            push!(bufs, upload(n))
        catch e
            ok = false
            err = first(sprint(showerror, e), 160)
            break
        end
    end
    dt = round(time() - t0; digits = 2)
    println(
        rpad(tag, 44), ok ? " OK  " : " FAIL", " sizes=", sizes, " sum=", sum(sizes), " largest_before=", s0.largest,
        " free_before=", s0.free, " seconds=", dt, ok ? "" : "  " * err
    )
    foreach(free!, bufs)
    return ok
end

s = show_stats("empty arena")
MiB = 1 << 20

# 0. One allocation from a free chunk: BFC splits the chunk only when it is at least twice the
# request or the remainder is at least 128 MiB; otherwise the whole chunk is consumed.
big = upload(s.limit - 100MiB)
show_stats("one upload of (limit - 100 MiB)")
free!(big)

# 1. Exactly one free block of 100 MiB at the front (everything after it allocated): upload a model
# whose weights total less than the block, as admission would allow.
hole = upload(100MiB)
rest = upload(stats().free)
free!(hole)
s = show_stats("one free 100 MiB block, rest allocated")
try_upload("one tensor of 99.9 MiB", [round(Int, 99.9 * MiB)])
try_upload("tensors 33.33 + 33.33 + 33.2 MiB (sum < block)", [round(Int, 33.33 * MiB), round(Int, 33.33 * MiB), round(Int, 33.2 * MiB)])
try_upload("tensors 40 + 30 + 20 MiB (sum 90 < block)", [40MiB, 30MiB, 20MiB])
try_upload("tensors 20 + 30 + 40 MiB (sum 90 < block)", [20MiB, 30MiB, 40MiB])
try_upload("tensors 5 MiB x 18 (sum 90 < block)", fill(5MiB, 18))
free!(rest)
show_stats("all freed")

# 2. Holes: 100 MiB blocks, every other one freed, plus the tail.
blocks = Any[]
while stats().free > 101MiB
    push!(blocks, upload(100MiB))
end
for i in 1:2:length(blocks)
    free!(blocks[i])
end
s = show_stats("alternate 100 MiB blocks freed")
try_upload("one tensor = largest - 4 KiB", [s.largest - 4096])
try_upload("three tensors summing to largest - 4 KiB", fill((s.largest - 4096) ÷ 3, 3))
try_upload("three tensors of 33.3 MiB", fill(round(Int, 33.3 * MiB), 3))
try_upload("ten tensors of 9.9 MiB", fill(round(Int, 9.9 * MiB), 10))
for i in 2:2:length(blocks)
    free!(blocks[i])
end
show_stats("all freed")

# 3. Replay of the admission failure seen under stress: model sm-variant (three f32 tensors of
# 34928556, 34952500 and 34951960 bytes, 104833536 allocator-rounded) admitted into a free block of
# exactly that size, the block left by a just-evicted model of the same footprint.
rounded(n) = cld(n, 256) * 256
variant = [34928556, 34952500, 34951960]
W = sum(rounded, variant)
for (tag, settle) in (("block freed long ago", true), ("block freed just before", false))
    victim = Any[upload(n) for n in (34914048, 34914048, 34914048 + (W - 3 * 34914048))]
    rest = upload(stats().free)
    foreach(free!, victim)
    settle && sleep(1.0)
    s = show_stats("free block for the replay ($tag)")
    try_upload("sm-variant replay ($tag), W=$W", variant)
    free!(rest)
    sleep(1.0)
end
show_stats("all freed")
