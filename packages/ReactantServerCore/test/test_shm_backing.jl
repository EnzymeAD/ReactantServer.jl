# Shared memory never raises SIGBUS (CORE-4). tmpfs allocates a page of a POSIX shared-memory object
# when it is first touched, and a touch through a mapping that tmpfs cannot back (a full /dev/shm),
# or that lies past the object's end (an object shrunk after it was mapped), kills the process with
# SIGBUS. A scenario that did so before the fix runs in a child process, so a regression fails the
# test instead of killing the suite. The full /dev/shm is a private tmpfs of a few MiB in a user and
# mount namespace (`unshare`); where unprivileged namespaces are not available those tests are
# skipped. Nothing here writes more than a few KiB to the host's own /dev/shm.

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
    key = string("/rsbk-", string(getpid(); base = 36), "-", string(rand(UInt32); base = 36))
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
