# The shared-memory registry decides re-registration by object identity (CORE-2), drops
# registrations whose object is gone (CORE-11), and pools mint their names from OS entropy, retrying
# a taken name (CORE-21). Every object these tests create carries the /rsc2- prefix (or is a pool,
# which mints /shm-<pid>-<token>-...) and is unlinked before the test ends.

import InterProcessCommunication as IPC

const _RSC2 = ReactantServerCore
_r2key() = string("/rsc2-", string(getpid(); base = 36), "-", string(rand(UInt32); base = 36))
_r2bytes(shm) = unsafe_wrap(Array, convert(Ptr{UInt8}, pointer(shm)), sizeof(shm); own = false)
# Unlink a test object's name but keep our mapping, as a client that removed its object would; and
# make sure our finalizer will not unlink the name again once it may belong to another object.
_r2unlink!(shm) = (rm(shm); shm.volatile = false; nothing)

@testset "registration is decided by object identity (CORE-2)" begin
    reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0)
    ka = _r2key()
    a = IPC.SharedMemory(ka, 64)
    kb = _r2key()
    b = IPC.SharedMemory(kb, 64)
    try
        _RSC2.shm_register!(reg, "n", ka, 0, 64)
        r1 = _RSC2.shm_regions(reg)["n"]
        # The identity is the mapped object's, as a fresh open of the key reports it.
        @test r1.identity == _RSC2.shm_key_identity(ka)
        @test isopen(r1.fd)

        # The same object again: idempotent; the registration and its mapping are kept as they are,
        # so requests in flight against it are unaffected.
        _RSC2.shm_register!(reg, "n", ka, 0, 64)
        @test _RSC2.shm_regions(reg)["n"] === r1 && r1.attached

        # The same object with a different window: replaced by a fresh mapping of that object.
        _RSC2.shm_register!(reg, "n", ka, 16, 32)
        r2 = _RSC2.shm_regions(reg)["n"]
        @test r2 !== r1 && !r1.attached && !isopen(r1.fd)
        @test (r2.offset, r2.byte_size, r2.identity) == (16, 32, r1.identity)

        # Another client's live object under the same name: ALREADY_EXISTS, nothing changes.
        err = try
            _RSC2.shm_register!(reg, "n", kb, 0, 64)
            nothing
        catch e
            e
        end
        @test err isa _RSC2.ShmRegionExistsError
        @test err.name == "n" && err.key == ka
        @test _RSC2.shm_regions(reg)["n"] === r2 && r2.attached

        # Once the first client's object is unlinked its registration is stale, and the second
        # client's registration replaces it; writes then land in the second object.
        _r2unlink!(a)
        _RSC2.shm_register!(reg, "n", kb, 0, 64)
        r3 = _RSC2.shm_regions(reg)["n"]
        @test r3.key == kb && !r2.attached
        _RSC2.shm_write!(reg, "n", 0, UInt8[1, 2, 3])
        @test _r2bytes(b)[1:3] == UInt8[1, 2, 3]
        @test _r2bytes(a)[17:19] == UInt8[0, 0, 0]

        # Unregister stays idempotent.
        @test _RSC2.shm_unregister!(reg, "n") === nothing
        @test _RSC2.shm_unregister!(reg, "n") === nothing
        @test !r3.attached

        # A key that does not exist is a clean ArgumentError (INVALID_ARGUMENT), not a crash.
        @test_throws ArgumentError _RSC2.shm_register!(reg, "m", _r2key(), 0, 8)
    finally
        _RSC2.shm_teardown!(reg)
        finalize(a)
        finalize(b)
    end
end

@testset "a key recreated after unlink replaces the registration, never keeps the orphan" begin
    reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0)
    k = _r2key()
    old = IPC.SharedMemory(k, 64)
    _RSC2.shm_register!(reg, "n", k, 0, 64)
    _r2unlink!(old)                              # the client removes its object...
    new = IPC.SharedMemory(k, 64)                # ...and a new one is created under the same key
    try
        @test _RSC2.shm_key_identity(k) != _RSC2.shm_regions(reg)["n"].identity
        _RSC2.shm_register!(reg, "n", k, 0, 64)  # same name, same key string
        r = _RSC2.shm_regions(reg)["n"]
        @test r.identity == _RSC2.shm_key_identity(k)
        _RSC2.shm_write!(reg, "n", 0, UInt8[7, 7])
        @test _r2bytes(new)[1:2] == UInt8[7, 7]
        @test _r2bytes(old)[1:2] == UInt8[0, 0]   # the orphan is never written
    finally
        _RSC2.shm_teardown!(reg)
        finalize(new)
        finalize(old)
    end
end

@testset "an output goes to the registration its request was decoded against" begin
    _I = _RSC2.inference
    sp(s) = _I.InferParameter(; parameter_choice = _RSC2.ProtoBuf.OneOf(:string_param, String(s)))
    ip(i) = _I.InferParameter(; parameter_choice = _RSC2.ProtoBuf.OneOf(:int64_param, Int64(i)))
    reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0)
    k = _r2key()
    old = IPC.SharedMemory(k, 64)
    new = nothing
    try
        _RSC2.shm_register!(reg, "n", k, 0, 64)
        out = _I.var"ModelInferRequest.InferRequestedOutputTensor"(;
            name = "y", parameters = Dict(
                "shared_memory_region" => sp("n"), "shared_memory_offset" => ip(0),
                "shared_memory_byte_size" => ip(16)
            )
        )
        inp = _I.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[1])
        msg = _I.ModelInferRequest(;
            model_name = "m", inputs = [inp], outputs = [out],
            raw_input_contents = [collect(reinterpret(UInt8, Float32[1]))]
        )
        decoded = _RSC2.decode_infer_request(msg, reg)
        @test decoded.output_targets["y"].mapping === _RSC2.shm_regions(reg)["n"]
        # While the request runs, the client is replaced by another object under the same key.
        _r2unlink!(old)
        new = IPC.SharedMemory(k, 64)
        _RSC2.shm_register!(reg, "n", k, 0, 64)
        y = [_RSC2.NamedTensor("y", Float32[1, 2, 3, 4])]
        @test_throws _RSC2.UnregisteredRegionError _RSC2.encode_infer_response("m", decoded, y, reg)
        @test all(==(0x00), _r2bytes(new))       # the new object never sees the old request's output
        # An output naming an unregistered region fails at decode, before any compute.
        @test_throws _RSC2.UnregisteredRegionError _RSC2.decode_infer_request(
            _I.ModelInferRequest(;
                model_name = "m", inputs = [inp],
                outputs = [_I.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y", parameters = merge(out.parameters, Dict("shared_memory_region" => sp("nope"))))],
                raw_input_contents = [collect(reinterpret(UInt8, Float32[1]))]
            ), reg
        )
    finally
        _RSC2.shm_teardown!(reg)
        new === nothing || finalize(new)
        finalize(old)
    end
end

@testset "registrations whose object is gone are dropped (CORE-11)" begin
    reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0.2)
    objs = [IPC.SharedMemory(k, 64) for k in (_r2key(), _r2key(), _r2key(), _r2key())]
    a, b, c, d = objs
    try
        _RSC2.shm_register!(reg, "a", a.id, 0, 64)
        _RSC2.shm_register!(reg, "b", b.id, 0, 64)
        @test reg.reaper[] !== nothing           # the periodic sweep starts with the first one
        ra = _RSC2.shm_regions(reg)["a"]
        _r2unlink!(a)                            # client a exited: its object is unlinked
        _RSC2.shm_register!(reg, "c", c.id, 0, 64)   # every registration sweeps
        @test sort!(collect(keys(_RSC2.shm_regions(reg)))) == ["b", "c"]
        @test !ra.attached && !isopen(ra.fd)     # its mapping and descriptor are released
        # Unlinked and recreated under the same key: stale too.
        kd = d.id
        _RSC2.shm_register!(reg, "d", kd, 0, 64)
        _r2unlink!(d)
        d2 = IPC.SharedMemory(kd, 64)
        push!(objs, d2)
        @test _RSC2.shm_reap!(reg) == ["d"]
        # Between registrations, the timer does it.
        _r2unlink!(b)
        @test timedwait(() -> !haskey(_RSC2.shm_regions(reg), "b"), 10.0) === :ok
        # A live object stays registered (a crashed client's still-linked object looks the same).
        @test haskey(_RSC2.shm_regions(reg), "c")
        @test isempty(_RSC2.shm_reap!(reg))
    finally
        _RSC2.shm_teardown!(reg)
        @test reg.reaper[] === nothing           # teardown stops the sweep
        foreach(finalize, objs)
    end
end

@testset "pool names come from OS entropy, one per pool, and a taken name is retried (CORE-21)" begin
    Random = Base.loaded_modules[Base.PkgId(Base.UUID("9a3f8284-a2c9-5f02-9a11-845980a1fd5c"), "Random")]
    # Seeding the global RNG no longer fixes the per-process token or a pool's key.
    saved = _RSC2._naming_token[]
    try
        Random.seed!(7)
        _RSC2._init_shm_naming!()
        t1 = _RSC2._naming_token[]
        k1 = _RSC2._fresh_shm_key("x")
        Random.seed!(7)
        _RSC2._init_shm_naming!()
        @test _RSC2._naming_token[] != t1
        @test _RSC2._fresh_shm_key("x") != k1
        @test occursin(r"^/shm-\d+-[0-9a-f]{8}-x$", k1)
    finally
        _RSC2._naming_token[] = saved
    end

    # Two pools with the same name, one after the other in one process (kserve_init again), get
    # different names: an old chunk cannot re-register onto the new pool, and the old pool's unlink
    # cannot remove the new pool's name (CLI-2).
    p1 = _RSC2.BufferPool(4096; n_slots = 1, name = "same")
    n1 = _RSC2.pool_region_name(p1)
    close(p1)
    p2 = _RSC2.BufferPool(4096; n_slots = 1, name = "same")
    try
        @test _RSC2.pool_region_name(p2) != n1
        finalize(p1.backing)                     # the old pool's deferred finalizer
        @test _RSC2.shm_key_identity("/" * _RSC2.pool_region_name(p2)) isa _RSC2.ShmIdentity
        # The old name can no longer be registered: it names nothing.
        reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0)
        @test_throws ArgumentError _RSC2.shm_register!(reg, n1, "/" * n1, 0, 4096)
        _RSC2.shm_teardown!(reg)
        # The pool keeps the descriptor it created the object with.
        @test p2.fd isa _RSC2.ShmFd && isopen(p2.fd)
        @test first(_RSC2._fstat_identity(p2.fd.fd)) == _RSC2.shm_key_identity("/" * _RSC2.pool_region_name(p2))
    finally
        close(p2)
    end
    @test !isopen(p2.fd)                         # released with the mapping

    # A taken name is retried under a fresh one; an explicit key is not retried.
    taken = IPC.SharedMemory(_r2key(), 64)
    try
        fresh = _r2key()
        names = [taken.id, taken.id, fresh]
        shm, region, fd = _RSC2._create_pool_object("x", nothing, 4096; mint = _ -> popfirst!(names))
        try
            @test region == fresh[2:end] && isempty(names)
            @test sizeof(shm) == 4096 && isopen(fd)
        finally
            rm(shm)
            shm.volatile = false
            finalize(shm)
            close(fd)
        end
        @test_throws ErrorException _RSC2._create_pool_object("x", nothing, 4096; mint = _ -> taken.id)
        @test_throws SystemError _RSC2.BufferPool(4096; n_slots = 1, key = taken.id)
        @test _r2bytes(taken) == zeros(UInt8, 64)   # never shared, never truncated
    finally
        finalize(taken)
    end
end

# Two client processes with the same pid (pid 1, each in its own pid namespace, sharing /dev/shm)
# and the same seed for the global RNG, the first killed with SIGKILL while its pool is alive so its
# object stays behind. With the token drawn from the default RNG they minted the same key, and the
# second's O_EXCL creation then failed. Needs unprivileged user and pid namespaces (Linux).
@testset "same pid and RNG seed in two pid namespaces (CORE-21)" begin
    unshare = Sys.which("unshare")
    usable = Sys.islinux() && unshare !== nothing &&
        success(pipeline(`$unshare --user --map-root-user --pid --fork true`; stdout = devnull, stderr = devnull))
    if !usable
        @info "skipping: no unprivileged pid namespaces here"
        @test_skip false
    else
        script = """
        Random = Base.loaded_modules[Base.PkgId(Base.UUID("9a3f8284-a2c9-5f02-9a11-845980a1fd5c"), "Random")]
        Random.seed!(1234)
        using ReactantServerCore
        p = ReactantServerCore.BufferPool(4096; n_slots = 1)
        println(getpid(), " ", ReactantServerCore.shm_key("x"), " ", ReactantServerCore.pool_region_name(p))
        flush(stdout)
        get(ENV, "RSC2_HOLD", "") == "1" && sleep(600)
        close(p)
        """
        jl = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
        ns(env) = setenv(`$unshare --user --map-root-user --pid --fork --kill-child $jl`, env)
        held = open(ns(merge(ENV, Dict("RSC2_HOLD" => "1"))))
        line_a = readline(held)
        leftover = ""
        try
            pid_a, key_a, region_a = split(line_a)
            leftover = "/" * region_a
            kill(held, Base.SIGKILL)             # a crash: the pool's object stays in /dev/shm
            wait(held)
            @test _RSC2.shm_key_identity(leftover) isa _RSC2.ShmIdentity
            out_b = readchomp(ns(copy(ENV)))     # the restarted client, same pid and seed
            pid_b, key_b, region_b = split(out_b)
            @test pid_a == pid_b == "1"
            @test key_a != key_b
            @test region_a != region_b
            @test _RSC2.shm_key_identity("/" * region_b) === :gone   # closed cleanly
        finally
            isempty(leftover) || ccall(:shm_unlink, Cint, (Cstring,), leftover)
        end
    end
end

@testset "a client cannot register the shared host-weight store's regions (WRK-9)" begin
    # A stand-in for a weight region (`/rsw-<model>-<digest>`), unlinked before the test ends. A
    # client used to be able to register it and point an output at it, so the worker would write a
    # request's result into the weights every worker on the node serves.
    tag = string(getpid(); base = 36) * "_" * string(rand(UInt32); base = 36)
    weights_key = "/rsw-wrk9_" * tag * "-0123abcd"
    w = IPC.SharedMemory(weights_key, 64)
    own_key = _r2key()
    own = IPC.SharedMemory(own_key, 64)
    reg = _RSC2.SharedMemoryRegistry(; reap_interval = 0)
    try
        refused(r, key) = try
            _RSC2.shm_register!(r, "w", key, 0, 64)
            false
        catch e
            e isa ArgumentError && occursin("shared host-weight store", e.msg)
        end
        # Every spelling shm_open resolves to the same object is refused, before it is opened.
        for key in (weights_key, lstrip(weights_key, '/'), "//" * lstrip(weights_key, '/'))
            @test refused(reg, key)
        end
        @test refused(reg, "/rsw-budget.lock")                   # the store's lock files too
        @test isempty(_RSC2.shm_regions(reg))
        # A client's own object registers as before, including one whose name merely contains rsw.
        _RSC2.shm_register!(reg, "own", own_key, 0, 64)
        @test haskey(_RSC2.shm_regions(reg), "own")
        @test _RSC2._reserved_prefix(reg, "/rswx-1") === nothing
        @test _RSC2._reserved_prefix(reg, "/shm-rsw-1") === nothing
        # The default is the store's own default prefix; a store with another prefix is reserved by
        # passing it.
        @test _RSC2.SharedWeightStore().prefix == _RSC2.SHARED_WEIGHT_PREFIX
        custom = _RSC2.SharedMemoryRegistry(; reap_interval = 0, reserved_prefixes = ["rsw", "abc"])
        @test _RSC2._reserved_prefix(custom, "/abc-m-1") == "abc"
        @test refused(custom, weights_key)
    finally
        _RSC2.shm_teardown!(reg)
        finalize(own)
        finalize(w)
    end
end
