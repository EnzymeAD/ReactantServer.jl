# The per-slot control word protocol (CORE-3): a server never writes a slot the client has given
# up on, and the client never reuses a slot a server may still be writing. The client side is a
# SHM BufferPool; the server side is a SharedMemoryRegistry registration of the same object, which
# maps it through its own descriptor, exactly as a worker does (OFD locks conflict between the two
# descriptors even within one process). Pools mint /shm-<pid>-<token>-... names and are closed (so
# unlinked) by every test.

const _SC = ReactantServerCore
const _SCI = _SC.inference

_sc_word(pool, s) = unsafe_load(_SC._word_ptr(pool, s.index), 1, :acquire)
_sc_store!(pool, s, w) = unsafe_store!(_SC._word_ptr(pool, s.index), w, 1, :release)
_sc_W(state, t) = _SC.slot_word(state, UInt64(t))

# A worker-side view of `pool`: its object registered as "r" in a fresh registry.
function _sc_server(pool)
    reg = _SC.SharedMemoryRegistry(; reap_interval = 0)
    _SC.shm_register!(reg, "r", "/" * _SC.pool_region_name(pool), 0, _SC.pool_region_bytes(pool))
    return reg, _SC.shm_regions(reg)["r"]
end
_sc_control(r, s) = _SC.SlotControl(r, _SC.slot_control_offset(s), s.token)
_sc_bytes(pool, sub, n) = copy(_SC.pool_view(sub, UInt8, n))

if !_SC.slot_control_supported()
    @testset "slot control is off on this platform" begin
        p = _SC.BufferPool(1 << 16; n_slots = 2)
        @test !_SC.has_slot_control(p)
        close(p)
    end
else

    @testset "slot control layout and the client's transitions (CORE-3)" begin
        page = _SC._page_size()
        p = _SC.BufferPool(10^5; n_slots = 3)
        try
            @test _SC.has_slot_control(p)
            @test p.ctl_offset % page == 0 && p.ctl_offset >= sizeof(p)
            @test _SC.pool_region_bytes(p) == p.ctl_offset + page
            @test sizeof(p.backing) == _SC.pool_region_bytes(p)
            @test !_SC.has_slot_control(_SC.BufferPool(4096; n_slots = 2, use_shm = false))
            slots = [_SC.acquire_slot!(p) for _ in 1:3]
            @test [_SC.slot_control_offset(s) for s in slots] == [p.ctl_offset + 128 * (i - 1) for i in 1:3]
            @test all(s -> _sc_word(p, s) == _SC.SLOT_FREE, slots)

            s = slots[1]
            t1 = _SC.slot_arm!(s)
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_OWNED, t1) && s.token == t1
            # A retry re-arms under a fresh token, closing the previous attempt first.
            t2 = _SC.slot_arm!(s)
            @test t2 != t1 && _sc_word(p, s) == _sc_W(_SC.SLOT_OWNED, t2)
            # Tokens never repeat within the pool, across slots and attempts.
            toks = [_SC.slot_arm!(x) for x in slots for _ in 1:5]
            @test allunique([t1; t2; toks])
            # Releasing an armed slot nobody claimed closes it and frees it at once.
            t = s.token
            _SC.release_slot!(s)
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_CANCELLED, t) && s.token == 0
            @test p.free[s.index]
            # A released slot is armed again from CANCELLED.
            s = _SC.acquire_slot!(p)
            t3 = _SC.slot_arm!(s)
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_OWNED, t3)
            # The seal: an attempt nobody claimed is closed (a server without the protocol answered);
            # DONE is final; WRITING is not.
            t = s.token
            @test _SC.slot_seal!(s) && _sc_word(p, s) == _sc_W(_SC.SLOT_CANCELLED, t)
            _SC.slot_arm!(s)
            _sc_store!(p, s, _sc_W(_SC.SLOT_DONE, s.token))
            @test _SC.slot_seal!(s)
            _SC.slot_arm!(s)
            _sc_store!(p, s, _sc_W(_SC.SLOT_WRITING, s.token))
            @test !_SC.slot_seal!(s)
            # A retry cannot re-arm a slot its previous attempt's server is writing.
            @test_throws _SC.SlotWriteInProgress _SC.slot_arm!(s)
            _sc_store!(p, s, _sc_W(_SC.SLOT_DONE, s.token))
            foreach(_SC.release_slot!, [s, slots[2], slots[3]])
            @test all(p.free)
        finally
            close(p)
        end
    end

    @testset "the server claims, copies and finishes, or skips a given-up slot (CORE-3)" begin
        p = _SC.BufferPool(1 << 16; n_slots = 2)
        reg, r = _sc_server(p)
        try
            s = _SC.acquire_slot!(p)
            out = _SC.subslot(s, 4096)
            data = rand(UInt8, 4096)
            _SC.slot_arm!(s)
            c = _sc_control(r, s)
            @test !_SC.slot_control_cancelled(c)
            _SC.shm_write_controlled!(c, [(data, out.offset, 4096)], "m"; chunk_bytes = 1000)
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_DONE, s.token)
            @test _SC.slot_seal!(s) && _sc_bytes(p, out, 4096) == data
            # The claim is single use: a second writer of the same attempt (a duplicate) is refused.
            @test_throws _SC.RequestCancelled _SC.shm_write_controlled!(c, [(zeros(UInt8, 4096), out.offset, 4096)], "m")
            @test _sc_bytes(p, out, 4096) == data

            # The client gives up first: the server sees it at dispatch and writes nothing.
            _SC.slot_arm!(s)
            c = _sc_control(r, s)
            fill!(_SC.pool_view(out, UInt8, 4096), 0x00)
            _SC.release_slot!(s)                      # closes OWNED -> CANCELLED, frees at once
            @test _SC.slot_control_cancelled(c)
            @test_throws _SC.RequestCancelled _SC.shm_write_controlled!(c, [(data, out.offset, 4096)], "m")
            @test all(==(0x00), _sc_bytes(p, out, 4096))
            # A later attempt on the same slot is a different token: the old one stays refused.
            s = _SC.acquire_slot!(p)
            _SC.slot_arm!(s)
            @test _SC.slot_control_cancelled(c)
            @test !_SC.slot_control_cancelled(_sc_control(r, s))
            # The predicate never waits: with the region's lock busy it answers "not cancelled".
            held = Threads.Event()
            go = Threads.Event()
            holder = Threads.@spawn @lock r.lock (notify(held); wait(go))
            wait(held)
            @test !_SC.slot_control_cancelled(c)
            notify(go)
            wait(holder)
            # Another party holding the word's lock makes the claim fail too.
            off = _SC.slot_control_offset(s)
            @test _SC.shm_try_lock(p.fd, off, 8)
            @test_throws _SC.RequestCancelled _SC.shm_write_controlled!(_sc_control(r, s), [(data, out.offset, 4096)], "m")
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_OWNED, s.token)   # unclaimed
            _SC.shm_unlock(p.fd, off, 8)
            # Bounds are checked before the claim: an out-of-range output leaves the word alone.
            @test_throws ArgumentError _SC.shm_write_controlled!(_sc_control(r, s), [(data, 10^9, 4096)], "m")
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_OWNED, s.token)
            _SC.release_slot!(s)
        finally
            _SC.shm_teardown!(reg)
            close(p)
        end
    end

    @testset "a slot being written is quarantined, reclaimed when its writer dies, retired while it is stuck" begin
        p = _SC.BufferPool(1 << 16; n_slots = 2, retire_after = 0.3)
        reg, r = _sc_server(p)
        try
            a = _SC.acquire_slot!(p)
            b = _SC.acquire_slot!(p)
            _SC.release_slot!(b)
            # A writer is mid-copy (WRITING, holding the word's lock) when the client gives up.
            _SC.slot_arm!(a)
            ta = a.token
            off = _SC.slot_control_offset(a)
            @test _SC.shm_try_lock(r.fd, off, 8)      # the writer's lock, through the server's descriptor
            _sc_store!(p, a, _sc_W(_SC.SLOT_WRITING, ta))
            _SC.release_slot!(a)
            @test p.quarantined[a.index] && !p.free[a.index]
            @test_throws ArgumentError _SC.release_slot!(a)   # not releasable twice
            # Until the writer is done the slot is never handed out; the other one still is.
            got = _SC.try_acquire_slot!(p)
            @test got !== nothing && got.index == b.index
            @test _SC.try_acquire_slot!(p) === nothing
            # Past the retire delay the writer still holds its lock: alive but stuck, so retired.
            sleep(0.4)
            @test _SC.try_acquire_slot!(p) === nothing
            @test only(p.quarantine).retired
            # It finishes: the slot rejoins.
            _sc_store!(p, a, _sc_W(_SC.SLOT_DONE, ta))
            _SC.shm_unlock(r.fd, off, 8)
            again = _SC.try_acquire_slot!(p)
            @test again !== nothing && again.index == a.index && isempty(p.quarantine)

            # A writer that dies mid-copy leaves WRITING with no lock: reclaimed after the delay.
            _SC.slot_arm!(again)
            t = again.token
            _sc_store!(p, again, _sc_W(_SC.SLOT_WRITING, t))
            _SC.release_slot!(again)
            @test _SC.try_acquire_slot!(p) === nothing     # before the delay: still quarantined
            sleep(0.4)
            back = _SC.try_acquire_slot!(p)
            @test back !== nothing && back.index == a.index
            @test _sc_word(p, back) == _sc_W(_SC.SLOT_CANCELLED, t)
            _SC.release_slot!(back)

            # A caller parked in acquire_slot! gets a quarantined run as soon as its writer finishes.
            _SC.slot_arm!(got)
            tg = got.token
            _sc_store!(p, got, _sc_W(_SC.SLOT_WRITING, tg))
            _SC.release_slot!(got)
            hold = _SC.acquire_slot!(p)                    # the other slot
            waiter = Threads.@spawn _SC.acquire_slot!(p)
            sleep(0.2)
            @test !istaskdone(waiter)
            _sc_store!(p, got, _sc_W(_SC.SLOT_DONE, tg))
            @test timedwait(() -> istaskdone(waiter), 5.0) === :ok
            w = fetch(waiter)
            @test w.index == got.index
            _SC.release_slot!(w)
            _SC.release_slot!(hold)
            # try_acquire_slot! never passes a parked waiter.
            all2 = [_SC.acquire_slot!(p) for _ in 1:2]
            parked = Threads.@spawn _SC.acquire_slot!(p)
            @test timedwait(() -> (@lock p.alloc_lock !isempty(p.waitq)), 5.0) === :ok
            _SC.release_slot!(all2[1])
            @test _SC.try_acquire_slot!(p) === nothing || istaskdone(parked)
            _SC.release_slot!(fetch(parked))
            _SC.release_slot!(all2[2])
            # Closing a pool with a quarantined run still releases the mapping.
            q = _SC.acquire_slot!(p)
            _SC.slot_arm!(q)
            _sc_store!(p, q, _sc_W(_SC.SLOT_WRITING, q.token))
            _SC.release_slot!(q)
        finally
            _SC.shm_teardown!(reg)
            close(p)
        end
        @test p.released
    end

    @testset "slot control parameters are decoded strictly (CORE-3)" begin
        p = _SC.BufferPool(1 << 16; n_slots = 2)
        reg, r = _sc_server(p)
        other = _SC.BufferPool(1 << 16; n_slots = 1)
        _SC.shm_register!(reg, "o", "/" * _SC.pool_region_name(other), 0, _SC.pool_region_bytes(other))
        sp(x) = _SCI.InferParameter(; parameter_choice = _SC.ProtoBuf.OneOf(:string_param, String(x)))
        ip(x) = _SCI.InferParameter(; parameter_choice = _SC.ProtoBuf.OneOf(:int64_param, Int64(x)))
        try
            s = _SC.acquire_slot!(p)
            out = _SC.subslot(s, 16)
            t = _SC.slot_arm!(s)
            ctl = Dict(
                _SC.SLOT_CONTROL_REGION_PARAM => sp("r"),
                _SC.SLOT_CONTROL_OFFSET_PARAM => ip(_SC.slot_control_offset(s)),
                _SC.SLOT_CONTROL_TOKEN_PARAM => ip(t)
            )
            outp(region, off) = Dict(
                "shared_memory_region" => sp(region), "shared_memory_offset" => ip(off),
                "shared_memory_byte_size" => ip(16)
            )
            msg(params; region = "r") = _SCI.ModelInferRequest(;
                model_name = "m", parameters = params,
                inputs = [_SCI.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[4])],
                outputs = [_SCI.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y", parameters = outp(region, out.offset))],
                raw_input_contents = [collect(reinterpret(UInt8, Float32[1, 2, 3, 4]))]
            )
            d = _SC.decode_infer_request(msg(ctl), reg)
            c = d.request.slot_control
            @test c isa _SC.SlotControl && c.mapping === r && c.token == t && c.offset == _SC.slot_control_offset(s)
            @test !_SC.request_cancelled(d.request)
            @test _SC.decode_infer_request(msg(Dict{String, _SCI.InferParameter}()), reg).request.slot_control === nothing

            bad(params; kw...) = try
                _SC.decode_infer_request(msg(params; kw...), reg)
                nothing
            catch e
                e
            end
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_TOKEN_PARAM => sp("1")))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_OFFSET_PARAM => sp("1")))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_REGION_PARAM => ip(1)))) isa ErrorException
            @test bad(filter(kv -> first(kv) != _SC.SLOT_CONTROL_TOKEN_PARAM, ctl)) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_TOKEN_PARAM => ip(0)))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_OFFSET_PARAM => ip(_SC.slot_control_offset(s) + 4)))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_OFFSET_PARAM => ip(_SC.pool_region_bytes(p) - 4)))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_OFFSET_PARAM => ip(-8)))) isa ErrorException
            @test bad(merge(ctl, Dict(_SC.SLOT_CONTROL_REGION_PARAM => sp("nope")))) isa _SC.UnregisteredRegionError
            # Outputs must be in the control word's region.
            @test bad(ctl; region = "o") isa ErrorException

            # The encoder writes under the protocol: the claimed attempt's outputs land, DONE.
            y = [_SC.NamedTensor("y", Float32[5, 6, 7, 8])]
            resp = _SC.encode_infer_response("m", d, y, reg)
            @test isempty(resp.raw_output_contents[1])
            @test _sc_word(p, s) == _sc_W(_SC.SLOT_DONE, t)
            @test reinterpret(Float32, _sc_bytes(p, out, 16)) == Float32[5, 6, 7, 8]
            # A given-up attempt is not written; the request fails as cancelled.
            t2 = _SC.slot_arm!(s)
            d2 = _SC.decode_infer_request(msg(merge(ctl, Dict(_SC.SLOT_CONTROL_TOKEN_PARAM => ip(t2)))), reg)
            fill!(_SC.pool_view(out, UInt8, 16), 0x00)
            @test _SC.slot_seal!(s)                       # closes OWNED(t2), as giving up does
            @test _SC.request_cancelled(d2.request)
            @test_throws _SC.RequestCancelled _SC.encode_infer_response("m", d2, y, reg)
            @test all(==(0x00), _sc_bytes(p, out, 16))
            # Without slot control (a third-party client) nothing is written once the deadline passed.
            d3 = _SC.decode_infer_request(msg(Dict{String, _SCI.InferParameter}()), reg)
            @test_throws _SC.DeadlineExceeded _SC.encode_infer_response("m", d3, y, reg; deadline_ns = Int64(time_ns()) - 1)
            @test all(==(0x00), _sc_bytes(p, out, 16))
            _SC.encode_infer_response("m", d3, y, reg; deadline_ns = Int64(time_ns()) + 10^10)
            @test reinterpret(Float32, _sc_bytes(p, out, 16)) == Float32[5, 6, 7, 8]
            _SC.release_slot!(s)
        finally
            _SC.shm_teardown!(reg)
            close(p)
            close(other)
        end
    end

    # Interleave a client giving up with a server writing, at random points of the write (between
    # its chunks). Whatever the order: after the client reuses the slot, nothing the server does
    # reaches it; a write the client saw in progress completes before the slot is reused; and a
    # server that lost the race wrote nothing.
    @testset "randomized interleavings of give-up and write (CORE-3)" begin
        p = _SC.BufferPool(1 << 16; n_slots = 1)
        reg, r = _sc_server(p)
        n = 2048
        outcomes = Dict{Symbol, Int}()
        try
            for iter in 1:300
                s = _SC.acquire_slot!(p)
                out = _SC.subslot(s, n)
                fill!(_SC.pool_view(out, UInt8, n), 0x00)
                _SC.slot_arm!(s)
                c = _sc_control(r, s)
                pattern = fill(UInt8(1 + iter % 200), n)
                steps = rand(0:40)
                writer = @async try
                    for _ in 1:rand(0:6)
                        yield()
                    end
                    _SC.shm_write_controlled!(c, [(pattern, out.offset, n)], "m"; chunk_bytes = 128, before_chunk = yield)
                    :written
                catch e
                    e isa _SC.RequestCancelled || rethrow()
                    :refused
                end
                for _ in 1:steps
                    yield()
                end
                _SC.release_slot!(s)                       # the client gives up here
                if p.quarantined[1]
                    # The write was in progress: the slot comes back only after it completed.
                    @test _SC.try_acquire_slot!(p) === nothing || istaskdone(writer)
                    @test fetch(writer) === :written
                    reuse = _SC.try_acquire_slot!(p)
                    @test reuse !== nothing
                    @test _sc_bytes(p, out, n) == pattern
                    outcomes[:quarantined] = get(outcomes, :quarantined, 0) + 1
                else
                    reuse = _SC.try_acquire_slot!(p)
                    @test reuse !== nothing
                    view = _SC.pool_view(out, UInt8, n)
                    fill!(view, 0xee)                      # the next user's data
                    result = fetch(writer)
                    @test all(==(0xee), view)              # never overwritten after the reuse
                    outcomes[result] = get(outcomes, result, 0) + 1
                end
                _SC.release_slot!(reuse)
            end
        finally
            _SC.shm_teardown!(reg)
            close(p)
        end
        # Every interleaving class actually happened.
        @test get(outcomes, :refused, 0) > 0 && get(outcomes, :quarantined, 0) > 0 && get(outcomes, :written, 0) > 0
        @info "slot control interleavings" outcomes
    end

    # A real writer process, killed with SIGKILL or stopped with SIGSTOP in the middle of its copy.
    @testset "a writer process killed or stopped mid-copy (CORE-3)" begin
        helper = """
        using ReactantServerCore
        const C = ReactantServerCore
        key, ctl, tok, off, n = ARGS[1], parse(Int, ARGS[2]), parse(UInt64, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5])
        len = parse(Int, ARGS[6])
        reg = C.SharedMemoryRegistry(; reap_interval = 0)
        C.shm_register!(reg, "r", key, 0, len)
        c = C.SlotControl(C.shm_regions(reg)["r"], ctl, tok)
        first = Ref(true)
        tick() = (first[] && (println("WRITING"); flush(stdout); first[] = false); sleep(0.02))
        C.shm_write_controlled!(c, [(fill(0xab, n), off, n)], "m"; chunk_bytes = 256, before_chunk = tick)
        println("DONE")
        flush(stdout)
        """
        n = 1 << 16                                     # 256 chunks of 20 ms: about 5 s of copying
        for sig in (Base.SIGKILL, :SIGSTOP)
            p = _SC.BufferPool(1 << 18; n_slots = 1, retire_after = 1.0)
            proc = nothing
            try
                s = _SC.acquire_slot!(p)
                out = _SC.subslot(s, n)
                t = _SC.slot_arm!(s)
                cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $helper
                    $("/" * _SC.pool_region_name(p)) $(_SC.slot_control_offset(s)) $t $(out.offset) $n $(_SC.pool_region_bytes(p))`
                proc = open(cmd)
                @test readline(proc) == "WRITING"
                _SC.release_slot!(s)                    # the client gives up mid-copy
                @test p.quarantined[1]
                if sig === Base.SIGKILL
                    kill(proc, Base.SIGKILL)
                    wait(proc)
                    @test _SC.try_acquire_slot!(p) === nothing   # within the delay: still held
                    sleep(1.2)
                    back = _SC.try_acquire_slot!(p)               # the lock is gone: reclaimed
                    @test back !== nothing
                    @test _sc_word(p, back) == _sc_W(_SC.SLOT_CANCELLED, t)
                    _SC.release_slot!(back)
                else
                    run(`kill -STOP $(getpid(proc))`)
                    sleep(1.2)
                    @test _SC.try_acquire_slot!(p) === nothing   # alive and holding its lock: retired
                    @test only(p.quarantine).retired
                    run(`kill -CONT $(getpid(proc))`)
                    @test readline(proc) == "DONE"
                    wait(proc)
                    back = _SC.try_acquire_slot!(p)               # it finished: the slot rejoins
                    @test back !== nothing
                    @test _sc_word(p, back) == _sc_W(_SC.SLOT_DONE, t)
                    @test all(==(0xab), _sc_bytes(p, out, n))      # and its copy was complete
                    _SC.release_slot!(back)
                end
            finally
                if proc !== nothing && process_running(proc)
                    run(`kill -CONT $(getpid(proc))`)
                    kill(proc, Base.SIGKILL)
                end
                close(p)
            end
        end
    end

end
