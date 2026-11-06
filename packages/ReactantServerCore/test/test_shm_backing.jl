# Shared memory never raises SIGBUS (CORE-4). tmpfs allocates a page of a POSIX shared-memory object
# when it is first touched, and a touch through a mapping that tmpfs cannot back (a full /dev/shm),
# or that lies past the object's end (an object shrunk after it was mapped), kills the process with
# SIGBUS. A scenario that did so before the fix runs in a child process, so a regression fails the
# test instead of killing the suite. The full /dev/shm is a private tmpfs of a few MiB in a user and
# mount namespace (`unshare`); where unprivileged namespaces are not available those tests are
# skipped. Nothing here allocates more than about 100 KiB of the host's own /dev/shm.

const _BK = ReactantServerCore

const _UNSHARE = Sys.which("unshare")
const _PRIVATE_SHM = Ref{Union{Nothing, Bool}}(nothing)

# Whether a child can run with a private, size-limited /dev/shm: needs unprivileged user and mount
# namespaces (Linux).
function _private_shm_usable()
    if _PRIVATE_SHM[] === nothing
        _PRIVATE_SHM[] = Sys.islinux() && _UNSHARE !== nothing && success(
            pipeline(
                `$_UNSHARE --user --map-root-user --mount sh -c 'mount -t tmpfs -o size=1m tmpfs /dev/shm'`;
                stdout = devnull, stderr = devnull
            )
        )
    end
    return _PRIVATE_SHM[]
end

# Run `script` in a fresh Julia process of this test environment, with a private tmpfs of `shm_size`
# (e.g. "4m") as its /dev/shm when given, and return how it ended and what it printed. `signal` is
# the signal that killed it (7 is SIGBUS), or 0.
function _run_child(script::AbstractString; shm_size = nothing)
    jl = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    cmd = if shm_size === nothing
        jl
    else
        mount = "mount -t tmpfs -o size=$shm_size tmpfs /dev/shm && exec \"\$@\""
        `$_UNSHARE --user --map-root-user --mount sh -c $mount sh $jl`
    end
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    return (; ok = success(p), signal = Int(p.termsignal), out = String(take!(out)), err = String(take!(err)))
end

# The line a child printed starting with `tag`, or nothing.
function _child_line(r, tag)
    for l in split(r.out, '\n')
        (l == tag || startswith(l, tag * " ")) && return String(l)
    end
    return nothing
end

function _report_child(r)
    r.ok && return
    return @error "child process failed" signal = r.signal stdout = r.out stderr = last(r.err, 4000)
end

@testset "a pool's pages are allocated when it is created (CORE-4)" begin
    p = _BK.BufferPool(1 << 16; n_slots = 2)
    try
        if Sys.islinux()
            st = stat("/dev/shm/" * _BK.pool_region_name(p))
            # Every byte of the object, data slots and control area, has a page.
            @test st.blocks * 512 >= _BK.pool_region_bytes(p)
        end
    finally
        close(p)
    end

    # A full /dev/shm, simulated: creation throws ShmSpaceError and leaves no object behind.
    old = _BK._SHM_FALLOCATE[]
    asked = Tuple{Int, Int}[]
    key = string("/rsbk-", string(getpid(); base = 36), "-pool")
    try
        _BK._SHM_FALLOCATE[] = (fd, off, len) -> (push!(asked, (Int(off), Int(len))); Libc.ENOSPC)
        err = try
            _BK.BufferPool(1 << 16; n_slots = 2, key = key)
            nothing
        catch e
            e
        end
        @test err isa ShmSpaceError
        @test err.errno == Libc.ENOSPC && occursin("No space left", sprint(showerror, err))
        page = Int(ccall(:getpagesize, Cint, ()))
        @test asked == [(0, (1 << 16) + page)]          # the whole object: data slots, then a control page
        @test _BK.shm_key_identity(key) === :gone       # unlinked
        # ENOMEM (the memory limit) is a ShmSpaceError too; any other failure is a SystemError.
        _BK._SHM_FALLOCATE[] = (fd, off, len) -> Libc.ENOMEM
        @test_throws ShmSpaceError _BK.BufferPool(1 << 16; n_slots = 2, key = key)
        _BK._SHM_FALLOCATE[] = (fd, off, len) -> Libc.EBADF
        @test_throws SystemError _BK.BufferPool(1 << 16; n_slots = 2, key = key)
        @test _BK.shm_key_identity(key) === :gone
    finally
        _BK._SHM_FALLOCATE[] = old
    end
end

# Before the fix the 8 MiB pool was created on a 4 MiB /dev/shm (sizing allocates nothing), and
# staging into it killed the process with SIGBUS.
@testset "a pool a full /dev/shm cannot hold fails at creation, not with SIGBUS (CORE-4)" begin
    if !_private_shm_usable()
        @info "skipping: no unprivileged user and mount namespaces here"
        @test_skip false
    else
        r = _run_child(
            """
            using ReactantServerCore
            const RSC = ReactantServerCore
            p = try
                RSC.BufferPool(8 << 20; n_slots = 2)
            catch e
                e
            end
            if p isa RSC.BufferPool
                s = RSC.acquire_slot!(p, 2)
                fill!(RSC.pool_view(s, UInt8, sizeof(s)), 0x01)
                println("STAGED")
            else
                println(p isa RSC.ShmSpaceError ? "SPACE \$(p.errno)" : "OTHER \$(sprint(showerror, p))")
            end
            println("LEFT [", join(readdir("/dev/shm"), ","), "]")
            """;
            shm_size = "4m"
        )
        _report_child(r)
        @test r.signal != 7                               # not killed by SIGBUS
        @test r.ok
        @test _child_line(r, "SPACE") == "SPACE $(Libc.ENOSPC)"
        @test _child_line(r, "LEFT") == "LEFT []"         # nothing left in /dev/shm
    end
end

# A test object's key, short enough for macOS. A counter rather than `rand`: every testset starts
# from the same RNG state, so two testsets would draw the same key.
const _BK_SEQ = Ref(0)
_bk_key() = string("/rsbk-", string(getpid(); base = 36), "-", _BK_SEQ[] += 1)

# The child scripts below print `TAG result` for each access: `OK`, or the name of the exception.
const _BK_PROBE = """
using ReactantServerCore
const RSC = ReactantServerCore
function probe(tag, f)
    v = try
        f()
        "OK"
    catch e
        string(nameof(typeof(e)))
    end
    println(tag, " ", v)
    flush(stdout)
end
const PAGE = Int(ccall(:getpagesize, Cint, ()))
"""

@testset "a region's pages are checked and allocated before every copy (CORE-4)" begin
    page = Int(ccall(:getpagesize, Cint, ()))
    key = _bk_key()
    cfd, cptr, _ = _BK.shm_create_exclusive(key, 4 * page)
    reg = _BK.SharedMemoryRegistry(; reap_interval = 0)
    old = nothing
    try
        old = _BK._SHM_FALLOCATE[]
        # A window that starts a page into the object: the checks use absolute object offsets.
        _BK.shm_register!(reg, "w", key, page, 3 * page)
        r = _BK.shm_regions(reg)["w"]
        asked = Tuple{Int, Int}[]
        _BK._SHM_FALLOCATE[] = (fd, off, len) -> (push!(asked, (Int(off), Int(len))); Cint(0))
        _BK.shm_write!(reg, "w", 100, fill(0x07, 50))
        @test _BK.shm_read(reg, "w", 100, 50) == fill(0x07, 50)
        @test asked == [(page + 100, 50), (page + 100, 50)]
        empty!(asked)
        t = UInt64(12345)
        unsafe_store!(Ptr{UInt64}(cptr + 3 * page), _BK.slot_word(_BK.SLOT_OWNED, t))
        c = _BK.SlotControl(r, 2 * page, t)
        @test !_BK.slot_control_cancelled(c)
        @test asked == [(3 * page, 8)]
        empty!(asked)
        _BK.shm_write_controlled!(c, [(fill(0x09, 10), 0, 10), (fill(0x0a, 20), page, 20)], "m")
        @test asked == [(3 * page, 8), (page, 10), (2 * page, 20)]
        @test unsafe_load(Ptr{UInt64}(cptr + 3 * page)) == _BK.slot_word(_BK.SLOT_DONE, t)

        # No room for the pages (simulated): ShmSpaceError, and nothing is touched. The controlled
        # write fails before its claim, so the word still reads OWNED and the client can take it back.
        _BK._SHM_FALLOCATE[] = (fd, off, len) -> Libc.ENOSPC
        @test_throws ShmSpaceError _BK.shm_read(reg, "w", 0, 16)
        @test_throws ShmSpaceError _BK.shm_write!(reg, "w", 0, fill(0x01, 16))
        @test_throws ShmSpaceError _BK._shm_copy_in!(r, 0, fill(0x01, 16), 16)
        t2 = UInt64(12346)
        unsafe_store!(Ptr{UInt64}(cptr + 3 * page), _BK.slot_word(_BK.SLOT_OWNED, t2))
        c2 = _BK.SlotControl(r, 2 * page, t2)
        @test !_BK.slot_control_cancelled(c2)          # undecided: left to the claim
        @test_throws ShmSpaceError _BK.shm_write_controlled!(c2, [(fill(0x01, 16), 0, 16)], "m")
        @test unsafe_load(Ptr{UInt64}(cptr + 3 * page)) == _BK.slot_word(_BK.SLOT_OWNED, t2)
        @test unsafe_load(Ptr{UInt8}(cptr + page)) == 0x09   # the earlier output is untouched
        err = try
            _BK.shm_read(reg, "w", 0, 16)
        catch e
            e
        end
        @test err.errno == Libc.ENOSPC && err.bytes == 16 && err.name == "w"
    finally
        old === nothing || (_BK._SHM_FALLOCATE[] = old)
        _BK.shm_teardown!(reg)
        _BK._shm_munmap(cptr, 4 * page)
        close(cfd)
        ccall(:shm_unlink, Cint, (Cstring,), key)
    end
end

# A client shrinks its object after registering it. Before the fix every access below past the new
# end killed the process with SIGBUS (the first one ended the child). Objects of a few pages, in the
# host's /dev/shm.
@testset "a region shrunk after registration fails cleanly, not with SIGBUS (CORE-4)" begin
    key = _bk_key()
    r = try
        _run_child(
            _BK_PROBE * """
                key = $(repr(key))
                cfd, cptr, _ = RSC.shm_create_exclusive(key, 4 * PAGE)
                reg = RSC.SharedMemoryRegistry(; reap_interval = 0)
                RSC.shm_register!(reg, "s", key, 0, 4 * PAGE)
                region = RSC.shm_regions(reg)["s"]
                t = UInt64(77)
                unsafe_store!(Ptr{UInt64}(cptr + 64), RSC.slot_word(RSC.SLOT_OWNED, t))   # page 0
                unsafe_store!(Ptr{UInt64}(cptr + 3 * PAGE), RSC.slot_word(RSC.SLOT_OWNED, t))
                ccall(:ftruncate, Cint, (Cint, Int64), cfd.fd, PAGE)                   # the client shrinks it
                probe("READ", () -> RSC.shm_read(reg, "s", 2 * PAGE, 16))
                probe("WRITE", () -> RSC.shm_write!(reg, "s", 2 * PAGE, ones(UInt8, 16)))
                probe("COPYIN", () -> RSC._shm_copy_in!(region, 2 * PAGE, ones(UInt8, 16), 16))
                far = RSC.SlotControl(region, 3 * PAGE, t)                             # word past the end
                println("CANCELLED ", RSC.slot_control_cancelled(far))
                probe("CONTROL_WORD", () -> RSC.shm_write_controlled!(far, [(ones(UInt8, 16), 0, 16)], "m"))
                near = RSC.SlotControl(region, 64, t)                                  # output past the end
                probe("CONTROL_OUTPUT", () -> RSC.shm_write_controlled!(near, [(ones(UInt8, 16), 2 * PAGE, 16)], "m"))
                println("WORD ", unsafe_load(Ptr{UInt64}(cptr + 64)) == RSC.slot_word(RSC.SLOT_OWNED, t))
                probe("READ_INSIDE", () -> RSC.shm_read(reg, "s", 0, 16))              # what is left still reads
                probe("REREGISTER", () -> RSC.shm_register!(reg, "s", key, 0, 4 * PAGE))
                ccall(:ftruncate, Cint, (Cint, Int64), cfd.fd, 4 * PAGE)               # grown back
                probe("READ_GROWN", () -> RSC.shm_read(reg, "s", 2 * PAGE, 16))
                RSC.shm_teardown!(reg)
                """
        )
    finally
        ccall(:shm_unlink, Cint, (Cstring,), key)
    end
    _report_child(r)
    @test r.signal != 7
    @test r.ok
    for tag in ("READ", "WRITE", "COPYIN", "CONTROL_WORD", "CONTROL_OUTPUT")
        @test _child_line(r, tag) == "$tag ShmRegionShrunkError"
    end
    @test _child_line(r, "CANCELLED") == "CANCELLED false"   # undecided: left to the claim
    @test _child_line(r, "WORD") == "WORD true"               # refused before the claim
    @test _child_line(r, "READ_INSIDE") == "READ_INSIDE OK"
    # Registering the shrunk object again is refused, so a client that re-registers on
    # FAILED_PRECONDITION cannot loop.
    @test _child_line(r, "REREGISTER") == "REREGISTER ArgumentError"
    @test _child_line(r, "READ_GROWN") == "READ_GROWN OK"
end

# A client that sizes its object without allocating it (an older client, or a third-party KServe or
# Triton client) on a 2 MiB /dev/shm. Before the fix the first output write past the space left
# killed the process with SIGBUS, and so did reading a page the client never wrote.
@testset "a region a full /dev/shm cannot back fails cleanly, not with SIGBUS (CORE-4)" begin
    if !_private_shm_usable()
        @info "skipping: no unprivileged user and mount namespaces here"
        @test_skip false
    else
        r = _run_child(
            _BK_PROBE * """
                const MiB = 1 << 20
                key = "/rsbk-sparse"
                raw = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), key, 0o102, 0o600)   # O_RDWR | O_CREAT
                ccall(:ftruncate, Cint, (Cint, Int64), raw, 8MiB)                         # sized, not allocated
                cptr = ccall(:mmap, Ptr{UInt8}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64), C_NULL, 8MiB, 3, 1, raw, 0)
                reg = RSC.SharedMemoryRegistry(; reap_interval = 0)
                RSC.shm_register!(reg, "s", key, 0, 8MiB)
                region = RSC.shm_regions(reg)["s"]
                t = UInt64(91)
                unsafe_store!(Ptr{UInt64}(cptr + 8MiB - PAGE), RSC.slot_word(RSC.SLOT_OWNED, t))   # one page
                c = RSC.SlotControl(region, 8MiB - PAGE, t)
                probe("WRITE", () -> RSC.shm_write!(reg, "s", 0, ones(UInt8, 4MiB)))
                probe("COPYIN", () -> RSC._shm_copy_in!(region, 0, ones(UInt8, 4MiB), 4MiB))
                probe("CONTROLLED", () -> RSC.shm_write_controlled!(c, [(ones(UInt8, 4MiB), 0, 4MiB)], "m"))
                println("WORD ", unsafe_load(Ptr{UInt64}(cptr + 8MiB - PAGE)) == RSC.slot_word(RSC.SLOT_OWNED, t))
                probe("READ_HOLE", () -> RSC.shm_read(reg, "s", 4MiB, 4MiB))
                probe("WRITE_FITS", () -> RSC.shm_write!(reg, "s", 0, ones(UInt8, 64 * 1024)))
                # Fill /dev/shm to the last page: a control word on a page nobody wrote cannot be read.
                filler = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), "/rsbk-filler", 0o102, 0o600)
                used = 0
                for chunk in (256 * 1024, PAGE)
                    while ccall(:posix_fallocate, Cint, (Cint, Int64, Int64), filler, used, chunk) == 0
                        global used += chunk
                    end
                end
                hole = RSC.SlotControl(region, 6MiB, t)
                println("CANCELLED_HOLE ", RSC.slot_control_cancelled(hole))
                probe("CONTROLLED_HOLE", () -> RSC.shm_write_controlled!(hole, [(ones(UInt8, 16), 0, 16)], "m"))
                RSC.shm_teardown!(reg)
                """;
            shm_size = "2m"
        )
        _report_child(r)
        @test r.signal != 7
        @test r.ok
        for tag in ("WRITE", "COPYIN", "CONTROLLED", "READ_HOLE", "CONTROLLED_HOLE")
            @test _child_line(r, tag) == "$tag ShmSpaceError"
        end
        @test _child_line(r, "WORD") == "WORD true"               # refused before the claim
        @test _child_line(r, "WRITE_FITS") == "WRITE_FITS OK"
        @test _child_line(r, "CANCELLED_HOLE") == "CANCELLED_HOLE false"
    end
end

# The shared weight store reserves a region's pages before writing it, so a full /dev/shm fails a
# new region with SharedStoreFull. A region whose creator died between sizing it and reserving its
# pages has none: before the fix the next worker read its header to see whether it was READY, and on
# a full /dev/shm that read killed the worker with SIGBUS.
@testset "a shared weight region left without pages is skipped, not read (CORE-4)" begin
    if !_private_shm_usable()
        @info "skipping: no unprivileged user and mount namespaces here"
        @test_skip false
    else
        r = _run_child(
            _BK_PROBE * """
                key = "remnant"
                specs = [(Float32, (1 << 18,))]
                dg = RSC.weights_digest(key, specs)
                _, total = RSC._layout(specs)
                name = "/rsw-" * key * "-" * string(dg; base = 16)
                raw = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), name, 0o102, 0o666)   # O_RDWR | O_CREAT
                ccall(:ftruncate, Cint, (Cint, Int64), raw, total)                           # and then it died
                ccall(:close, Cint, (Cint,), raw)
                filler = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), "/rsbk-filler", 0o102, 0o600)
                used = 0
                for chunk in (256 * 1024, PAGE)
                    while ccall(:posix_fallocate, Cint, (Cint, Int64, Int64), filler, used, chunk) == 0
                        global used += chunk
                    end
                end
                probe("MATERIALIZE", () -> RSC.materialize_host_weights!(RSC.SharedWeightStore(), key, dg, specs, a -> fill!(a[1], 1.0f0)))
                println("REMNANT ", isfile("/dev/shm" * name))
                """;
            shm_size = "2m"
        )
        _report_child(r)
        @test r.signal != 7
        @test r.ok
        @test _child_line(r, "MATERIALIZE") == "MATERIALIZE SharedStoreFull"
        @test _child_line(r, "REMNANT") == "REMNANT false"        # the creator path removed it
    end
end
