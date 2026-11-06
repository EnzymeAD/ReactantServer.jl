# Shared-memory data plane end to end: create a POSIX region, register it over gRPC, and run
# inference referencing it for inputs (inline output) and for both inputs and outputs (with a
# nonzero output offset). Mirrors the Triton system shared-memory extension.

import InterProcessCommunication as IPC

const _ShmInf = ReactantServer.inference
const _RSCore = ReactantServer.ReactantServerCore

_sp(s) = _ShmInf.InferParameter(; parameter_choice = ReactantServer.ProtoBuf.OneOf(:string_param, String(s)))
_ip(i) = _ShmInf.InferParameter(; parameter_choice = ReactantServer.ProtoBuf.OneOf(:int64_param, Int64(i)))

# A unique POSIX shm name for this test process. macOS rejects names longer than PSHMNAMLEN
# (31 bytes), so the PID and the random suffix are base-36: at most 18 bytes including the slash.
_test_shm_key() = string("/rst-", string(getpid(); base = 36), "-", string(rand(UInt32); base = 36))

# Create a POSIX shm region; returns (handle, key, byte-view). volatile=true so the handle
# unlinks the object when finalized at the end of the test.
function _make_region(nbytes::Int)
    key = _test_shm_key()
    shm = IPC.SharedMemory(key, nbytes)
    view = unsafe_wrap(Array, convert(Ptr{UInt8}, pointer(shm)), nbytes; own = false)
    fill!(view, 0x00)
    return shm, key, view
end

@testset "shared memory registry read/write" begin
    shm, key, vbuf = _make_region(32)
    reg = ReactantServer.SharedMemoryRegistry()
    try
        ReactantServer.shm_register!(reg, "r", key, 0, 32)
        payload = collect(reinterpret(UInt8, Float32[1, 2, 3, 4]))

        # write through the locked accessor lands in the mapping
        ReactantServer.shm_write!(reg, "r", 16, payload)
        @test reinterpret(Float32, vbuf[17:32]) == Float32[1, 2, 3, 4]

        # read through the locked accessor returns a private copy, not an alias
        got = ReactantServer.shm_read(reg, "r", 16, 16)
        @test reinterpret(Float32, got) == Float32[1, 2, 3, 4]
        got[1] = 0xff
        @test vbuf[17] != 0xff

        # bounds errors stay ArgumentError; an unknown region is the distinct UnregisteredRegionError
        # (mapped to FAILED_PRECONDITION at the transport, so a client can re-register and retry).
        @test_throws ArgumentError ReactantServer.shm_read(reg, "r", 24, 16)
        @test_throws ArgumentError ReactantServer.shm_write!(reg, "r", 24, payload)
        @test_throws ReactantServer.UnregisteredRegionError ReactantServer.shm_read(reg, "missing", 0, 4)

        # unregister is idempotent: an unknown name is a successful no-op (so the gateway fan-out and
        # the client's pre-emptive cleanup unregister never error on a region that was never
        # registered), and a repeated unregister is harmless.
        @test ReactantServer.shm_unregister!(reg, "missing") === nothing
        @test ReactantServer.shm_unregister!(reg, "r") === nothing
        @test ReactantServer.shm_unregister!(reg, "r") === nothing   # already gone: still a no-op
        ReactantServer.shm_register!(reg, "r", key, 0, 32)           # re-register for the churn loop below
        # register still fails loudly (fail early, not at inference time): a region that does not fit
        # its shared-memory object is rejected at register time rather than silently accepted.
        @test_throws ArgumentError ReactantServer.shm_register!(reg, "bad", key, 0, 1024)

        # a concurrent unregister/re-register churn during repeated read/write must not fault:
        # the locked accessors copy under reg.lock, so the mapping can never be munmapped
        # mid-copy. Reaching the end without a segfault is the assertion.
        stop = Ref(false)
        worker = Threads.@spawn while !stop[]
            try
                ReactantServer.shm_read(reg, "r", 0, 16)
                ReactantServer.shm_write!(reg, "r", 0, payload)
            catch
                # region transiently unregistered between calls; retry
            end
        end
        for _ in 1:500
            ReactantServer.shm_register!(reg, "r", key, 0, 32)
            ReactantServer.shm_unregister!(reg, "r")
        end
        stop[] = true
        wait(worker)
        @test true
    finally
        finalize(shm)
    end
end

@testset "IsSameIPCNamespace probe" begin
    shm, key, _ = _make_region(64)
    try
        # The helper sees an object we created in this (the server's) own namespace.
        @test same_ipc_namespace(key) == true
        # Absent objects and empty names are "not same", never an error.
        @test same_ipc_namespace(string("/rst-none-", string(getpid(); base = 36))) == false
        @test same_ipc_namespace("") == false

        # The worker handler wraps the helper and returns an IsSameIPCNamespaceResponse.
        @test ReactantServer._handle_is_same_ipc_namespace(key).same == true
        @test ReactantServer._handle_is_same_ipc_namespace("/nope-$(getpid())").same == false
    finally
        finalize(shm)
    end
end

@testset "shared memory data plane (CPU)" begin
    mktempdir() do root
        manifest = """
        format_version: "2.0"
        name: scale4
        executable_inputs:
          - {name: x, dtype: f32, shape: c, dims: {c: 4}}
        executable_outputs:
          - {name: y, dtype: f32, shape: c, dims: {c: 4}}
        batching: {compiled_batch_sizes: [1]}
        """
        mlir = """
        module {
          func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.multiply %x, %w : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        write_bundle(
            root, "scale4"; manifest_yaml = manifest, mlir_text = mlir,
            weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"]
        )

        port = grpc_free_port()
        cfg = ReactantServer.ServerConfig(
            [root], "",
            ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", port)
        )
        srv = ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend(), blocking = false)
        sleep(0.3)

        rin, kin, vin = _make_region(16)               # input region
        rio, kio, vio = _make_region(32)                # input at [0,16), output at [16,32)
        try
            # --- register both regions ---
            for (rname, key, bytes) in (("rin", kin, 16), ("rio", kio, 32))
                resp = grpc_call(
                    WorkerRPC.SystemSharedMemoryRegister, port,
                    _ShmInf.SystemSharedMemoryRegisterRequest(; name = rname, key = key, offset = 0, byte_size = bytes)
                )
                @test resp isa _ShmInf.SystemSharedMemoryRegisterResponse
            end

            # status reports both regions
            st = grpc_call(
                WorkerRPC.SystemSharedMemoryStatus, port, _ShmInf.SystemSharedMemoryStatusRequest(; name = "")
            )
            @test haskey(st.regions, "rin") && haskey(st.regions, "rio")
            @test st.regions["rin"].byte_size == 16
            @test st.regions["rin"].key == kin

            # --- A: input from shm, output inline ---
            copyto!(vin, reinterpret(UInt8, Float32[1, 2, 3, 4]))
            inA = _ShmInf.var"ModelInferRequest.InferInputTensor"(;
                name = "x", datatype = "FP32", shape = Int64[4],
                parameters = Dict(
                    "shared_memory_region" => _sp("rin"),
                    "shared_memory_offset" => _ip(0),
                    "shared_memory_byte_size" => _ip(16)
                )
            )
            mA = grpc_call(
                WorkerRPC.ModelInfer, port,
                _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [inA])
            )
            @test collect(reinterpret(Float32, mA.raw_output_contents[1])) == Float32[2, 4, 6, 8]

            # --- B: input from shm, output written to shm at offset 16 ---
            copyto!(view(vio, 1:16), reinterpret(UInt8, Float32[3, 4, 5, 6]))
            inB = _ShmInf.var"ModelInferRequest.InferInputTensor"(;
                name = "x", datatype = "FP32", shape = Int64[4],
                parameters = Dict(
                    "shared_memory_region" => _sp("rio"),
                    "shared_memory_offset" => _ip(0),
                    "shared_memory_byte_size" => _ip(16)
                )
            )
            outB = _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(;
                name = "y",
                parameters = Dict(
                    "shared_memory_region" => _sp("rio"),
                    "shared_memory_offset" => _ip(16),
                    "shared_memory_byte_size" => _ip(16)
                )
            )
            mB = grpc_call(
                WorkerRPC.ModelInfer, port,
                _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [inB], outputs = [outB])
            )
            # output went to shm, not inline: an empty placeholder keeps raw_output_contents positional
            @test mB.raw_output_contents == [UInt8[]]
            @test mB.outputs[1].name == "y"
            # the server wrote the result into the region at offset 16
            @test reinterpret(Float32, view(vio, 17:32)) == Float32[6, 8, 10, 12]

            # A shared-memory parameter of the wrong type is a client error (INVALID_ARGUMENT): an
            # offset sent as a string must not be read as offset 0 and overwrite the input.
            for (k, bad) in (
                    ("shared_memory_offset", _sp("16")), ("shared_memory_byte_size", _sp("16")),
                    ("shared_memory_region", _ip(1)),
                )
                outBad = _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(;
                    name = "y", parameters = merge(outB.parameters, Dict(k => bad))
                )
                fill!(view(vio, 17:32), 0x00)
                try
                    grpc_call(
                        WorkerRPC.ModelInfer, port,
                        _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [inB], outputs = [outBad])
                    )
                    @test false
                catch ex
                    @test ex isa gRPCClient.gRPCServiceCallException
                    @test ex.grpc_status == Int(ReactantServer._G.StatusCode.INVALID_ARGUMENT)
                    @test occursin("invalid value type", ex.message)
                end
                @test reinterpret(Float32, view(vio, 1:16)) == Float32[3, 4, 5, 6]   # input untouched
                @test all(==(0x00), view(vio, 17:32))                                # nothing written
            end

            # --- unregister one, then confirm it is gone ---
            ru = grpc_call(
                WorkerRPC.SystemSharedMemoryUnregister, port, _ShmInf.SystemSharedMemoryUnregisterRequest(; name = "rin")
            )
            @test ru isa _ShmInf.SystemSharedMemoryUnregisterResponse
            st2 = grpc_call(
                WorkerRPC.SystemSharedMemoryStatus, port, _ShmInf.SystemSharedMemoryStatusRequest(; name = "")
            )
            @test !haskey(st2.regions, "rin") && haskey(st2.regions, "rio")

            # referencing an unregistered region is recoverable (the client re-registers), so it
            # surfaces as FAILED_PRECONDITION, distinct from a malformed-request INVALID_ARGUMENT.
            inBad = _ShmInf.var"ModelInferRequest.InferInputTensor"(;
                name = "x", datatype = "FP32", shape = Int64[4],
                parameters = Dict(
                    "shared_memory_region" => _sp("rin"),
                    "shared_memory_byte_size" => _ip(16)
                )
            )
            try
                grpc_call(
                    WorkerRPC.ModelInfer, port,
                    _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [inBad])
                )
                @test false
            catch ex
                @test ex isa gRPCClient.gRPCServiceCallException
                @test ex.grpc_status == Int(ReactantServer._G.StatusCode.FAILED_PRECONDITION)
            end

            # simulate a server restart: wipe the in-memory registry while the process keeps serving.
            # A request against the still-registered "rio" region now fails the same recoverable way,
            # which is exactly the stale-registration signal the client keys its recovery on.
            ReactantServer.shm_teardown!(srv.shm)
            inStale = _ShmInf.var"ModelInferRequest.InferInputTensor"(;
                name = "x", datatype = "FP32", shape = Int64[4],
                parameters = Dict(
                    "shared_memory_region" => _sp("rio"),
                    "shared_memory_offset" => _ip(0),
                    "shared_memory_byte_size" => _ip(16)
                )
            )
            try
                grpc_call(
                    WorkerRPC.ModelInfer, port,
                    _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [inStale])
                )
                @test false
            catch ex
                @test ex isa gRPCClient.gRPCServiceCallException
                @test ex.grpc_status == Int(ReactantServer._G.StatusCode.FAILED_PRECONDITION)
            end
        finally
            ReactantServer.stop!(srv)
            finalize(rin)
            finalize(rio)
        end
    end
end

@testset "shared memory bounds checks reject overflowing offsets and sizes" begin
    shm, key, _ = _make_region(32)
    reg = ReactantServer.SharedMemoryRegistry()
    try
        ReactantServer.shm_register!(reg, "r", key, 0, 32)

        # offset + byte_size would wrap Int64; the subtraction-form check must reject it.
        @test_throws ArgumentError ReactantServer.shm_read(reg, "r", Int64(2)^62, Int64(2)^62)
        @test_throws ArgumentError ReactantServer.shm_read(reg, "r", typemax(Int64), 8)
        @test_throws ArgumentError ReactantServer.shm_write!(reg, "r", Int64(2)^62, UInt8[1, 2])

        # Values that do not fit in Int64 are rejected at registration, not after the mmap.
        @test_throws ArgumentError ReactantServer.shm_register!(reg, "huge", key, typemax(UInt64), 8)
        @test_throws ArgumentError ReactantServer.shm_register!(reg, "huge", key, 0, typemax(UInt64))
        # ...and a wrapping offset+size pair within Int64 range is still out of bounds.
        @test_throws ArgumentError ReactantServer.shm_register!(reg, "huge", key, Int64(2)^62, Int64(2)^62)

        # In-bounds access still works.
        ReactantServer.shm_write!(reg, "r", 0, UInt8[1, 2, 3, 4])
        @test ReactantServer.shm_read(reg, "r", 0, 4) == UInt8[1, 2, 3, 4]
    finally
        ReactantServer.shm_teardown!(reg)
        finalize(shm)
    end
end

# The slot control protocol (CORE-3) on a real CPU worker, driven with a client-side BufferPool the
# way ReactantServerClient stages a chunk: a claimed attempt gets its output written and its word
# DONE; an attempt its client already gave up is dropped at dispatch (CANCELLED) and never written;
# malformed and stale control is refused with the status the client acts on.
@testset "slot control on a CPU worker (CORE-3)" begin
    if !ReactantServer.slot_control_supported()
        @test_skip false
    else
        mktempdir() do root
            write_bundle(
                root, "scale4"; manifest_yaml = """
                format_version: "2.0"
                name: scale4
                executable_inputs:
                  - {name: x, dtype: f32, shape: c, dims: {c: 4}}
                executable_outputs:
                  - {name: y, dtype: f32, shape: c, dims: {c: 4}}
                batching: {compiled_batch_sizes: [1]}
                """, mlir_text = """
                module {
                  func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %x, %w : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
                """, weights = Dict("w" => Float32[2, 2, 2, 2]), argument_order = ["w"]
            )
            port = grpc_free_port()
            cfg = ReactantServer.ServerConfig(
                [root], "",
                ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
                ReactantServer.SchedulerConfig(30.0, 64, 30.0),
                ReactantServer.EndpointsConfig("127.0.0.1", port)
            )
            srv = ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend(), blocking = false)
            sleep(0.3)
            pool = ReactantServer.BufferPool(1 << 16; n_slots = 2)
            name = ReactantServer.pool_region_name(pool)
            try
                grpc_call(
                    WorkerRPC.SystemSharedMemoryRegister, port,
                    _ShmInf.SystemSharedMemoryRegisterRequest(;
                        name = name, key = "/" * name, offset = 0, byte_size = ReactantServer.pool_region_bytes(pool)
                    )
                )
                s = ReactantServer.acquire_slot!(pool)
                xin = ReactantServer.subslot(s, 16; align = 4)
                yout = ReactantServer.subslot(s, 16; align = 4)
                ReactantServer.pool_view(xin, Float32, 4) .= Float32[1, 2, 3, 4]
                word() = unsafe_load(_RSCore._word_ptr(pool, s.index), 1, :acquire)
                ctl(t) = Dict(
                    ReactantServer.SLOT_CONTROL_REGION_PARAM => _sp(name),
                    ReactantServer.SLOT_CONTROL_OFFSET_PARAM => _ip(ReactantServer.slot_control_offset(s)),
                    ReactantServer.SLOT_CONTROL_TOKEN_PARAM => _ip(t)
                )
                shm(sub) = Dict(
                    "shared_memory_region" => _sp(name), "shared_memory_offset" => _ip(sub.offset),
                    "shared_memory_byte_size" => _ip(16)
                )
                input = _ShmInf.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[4], parameters = shm(xin))
                out_shm = _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y", parameters = shm(yout))
                out_inline = _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y")
                infer(params, out) = grpc_call(
                    WorkerRPC.ModelInfer, port,
                    _ShmInf.ModelInferRequest(; model_name = "scale4", inputs = [input], outputs = [out], parameters = params)
                )
                status(f) = try
                    f()
                    nothing
                catch ex
                    ex isa gRPCClient.gRPCServiceCallException ? ex.grpc_status : rethrow()
                end

                # Claimed and written: the output lands and the word reads DONE.
                t = ReactantServer.slot_arm!(s)
                resp = infer(ctl(t), out_shm)
                @test resp.raw_output_contents == [UInt8[]]
                @test word() == _RSCore.slot_word(_RSCore.SLOT_DONE, t)
                @test ReactantServer.slot_seal!(s)
                @test ReactantServer.pool_view(yout, Float32, 4) == Float32[2, 4, 6, 8]

                # Given up before it ran: dropped at dispatch. With an inline output only the dispatch
                # check can answer CANCELLED, so this is the scheduler's predicate at work.
                fill!(ReactantServer.pool_view(yout, UInt8, 16), 0x00)
                t2 = ReactantServer.slot_arm!(s)
                @test ReactantServer.slot_seal!(s)           # closes OWNED(t2), as giving up does
                @test status(() -> infer(ctl(t2), out_inline)) == Int(ReactantServer._G.StatusCode.CANCELLED)
                @test status(() -> infer(ctl(t2), out_shm)) == Int(ReactantServer._G.StatusCode.CANCELLED)
                @test all(==(0x00), ReactantServer.pool_view(yout, UInt8, 16))

                # Malformed control is a client error; a control word in an unknown region is stale.
                t3 = ReactantServer.slot_arm!(s)
                partial = filter(kv -> first(kv) != ReactantServer.SLOT_CONTROL_TOKEN_PARAM, ctl(t3))
                @test status(() -> infer(partial, out_shm)) == Int(ReactantServer._G.StatusCode.INVALID_ARGUMENT)
                typed = merge(ctl(t3), Dict(ReactantServer.SLOT_CONTROL_TOKEN_PARAM => _sp("1")))
                @test status(() -> infer(typed, out_shm)) == Int(ReactantServer._G.StatusCode.INVALID_ARGUMENT)
                stale = merge(ctl(t3), Dict(ReactantServer.SLOT_CONTROL_REGION_PARAM => _sp("gone")))
                @test status(() -> infer(stale, out_inline)) == Int(ReactantServer._G.StatusCode.FAILED_PRECONDITION)
                @test word() == _RSCore.slot_word(_RSCore.SLOT_OWNED, t3)   # untouched
                # A request without control still works (third-party clients).
                @test infer(Dict{String, _ShmInf.InferParameter}(), out_shm).outputs[1].name == "y"
                @test ReactantServer.pool_view(yout, Float32, 4) == Float32[2, 4, 6, 8]
                ReactantServer.release_slot!(s)
            finally
                ReactantServer.stop!(srv)
                close(pool)
            end
        end
    end
end

# The same give-up versus write race as the Core test, with the client and the writer on separate
# threads and random spins between every step, so the interleavings are not limited to yield points.
# Both spins start from a go signal, not from the spawn: `Threads.@spawn` returns only after waking
# each sleeping thread in turn, which keeps the client busy for over 100 us on a 64-thread host
# while the writer is already running, so nearly every write would finish before the client could
# give up. The busy-waits pass a GC safepoint, so a collection requested meanwhile cannot deadlock
# them. Iterations go on past 2000 until every outcome has been seen 20 times, up to 20000.
@testset "slot control under thread interleavings (CORE-3)" begin
    if !ReactantServer.slot_control_supported() || Threads.nthreads() < 2
        @test_skip false
    else
        pool = ReactantServer.BufferPool(1 << 16; n_slots = 1)
        reg = ReactantServer.SharedMemoryRegistry(; reap_interval = 0)
        ReactantServer.shm_register!(reg, "r", "/" * ReactantServer.pool_region_name(pool), 0, ReactantServer.pool_region_bytes(pool))
        r = ReactantServer.shm_regions(reg)["r"]
        function spin(ns)
            t0 = time_ns()
            while time_ns() - t0 < ns
            end
            return nothing
        end
        n = 4096
        outcomes = Dict{Symbol, Int}()
        kinds = (:refused, :written, :quarantined)
        bad = 0
        iters = 0
        try
            while iters < 2000 || (iters < 20_000 && any(k -> get(outcomes, k, 0) < 20, kinds))
                iters += 1
                s = ReactantServer.acquire_slot!(pool)
                out = ReactantServer.subslot(s, n)
                view = ReactantServer.pool_view(out, UInt8, n)
                fill!(view, 0x00)
                ReactantServer.slot_arm!(s)
                c = ReactantServer.SlotControl(r, ReactantServer.slot_control_offset(s), s.token)
                pattern = fill(UInt8(1 + iters % 200), n)
                ready = Threads.Atomic{Bool}(false)
                go = Threads.Atomic{Bool}(false)
                writer = Threads.@spawn begin
                    ready[] = true
                    while !go[]
                        GC.safepoint()
                    end
                    spin(rand(0:30_000))
                    try
                        _RSCore.shm_write_controlled!(c, [(pattern, out.offset, n)], "m"; chunk_bytes = 256, before_chunk = () -> spin(rand(0:2_000)))
                        :written
                    catch e
                        e isa ReactantServer.RequestCancelled || rethrow()
                        :refused
                    end
                end
                while !ready[]
                    GC.safepoint()
                end
                go[] = true                                      # both spins start here
                spin(rand(0:40_000))
                ReactantServer.release_slot!(s)                  # the client gives up
                if pool.quarantined[1]
                    result = fetch(writer)
                    reuse = ReactantServer.try_acquire_slot!(pool)
                    (result === :written && reuse !== nothing && view == pattern) || (bad += 1)
                    outcomes[:quarantined] = get(outcomes, :quarantined, 0) + 1
                else
                    reuse = ReactantServer.try_acquire_slot!(pool)
                    fill!(view, 0xee)                            # the next user's data
                    result = fetch(writer)
                    (reuse !== nothing && all(==(0xee), view)) || (bad += 1)
                    outcomes[result] = get(outcomes, result, 0) + 1
                end
                reuse === nothing || ReactantServer.release_slot!(reuse)
            end
        finally
            ReactantServer.shm_teardown!(reg)
            close(pool)
        end
        @info "slot control thread interleavings" iterations = iters outcomes
        @test bad == 0
        @test all(k -> get(outcomes, k, 0) >= 20, kinds)
    end
end

# ---- CORE-4: no SIGBUS from a region's pages ----
#
# The worker checks, before every copy into or out of a registered region and before touching a
# slot control word, that the object still covers the range and that the range's pages exist, so a
# shrunk object or a full /dev/shm fails the request instead of killing the worker with SIGBUS. The
# mock model is test_status_cancel.jl's `_sc_entry`: x and y are f32 (2, n), y = x .* [2, 3].

# A one-item request for model "m" whose x is read from `region` at `x_at` and whose y is written
# there at `y_at` (`nothing`: inline), with request parameters `ctl` (slot control).
function _bk_body(region, x_at, y_at; ctl = Dict{String, _ShmInf.InferParameter}())
    none = Dict{String, _ShmInf.InferParameter}()
    shm(off) = Dict(
        "shared_memory_region" => _sp(region), "shared_memory_offset" => _ip(off),
        "shared_memory_byte_size" => _ip(8)
    )
    req = _ShmInf.ModelInferRequest(;
        model_name = "m",
        inputs = [
            _ShmInf.var"ModelInferRequest.InferInputTensor"(;
                name = "x", datatype = "FP32", shape = Int64[1, 2], parameters = x_at === nothing ? none : shm(x_at)
            ),
        ],
        outputs = [
            _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(;
                name = "y", parameters = y_at === nothing ? none : shm(y_at)
            ),
        ],
        raw_input_contents = x_at === nothing ? [collect(reinterpret(UInt8, Float32[1, 2]))] : Vector{UInt8}[],
        parameters = ctl
    )
    return ReactantServer._encode_message(req)
end

_bk_ctl(region, at, t) = Dict(
    ReactantServer.SLOT_CONTROL_REGION_PARAM => _sp(region),
    ReactantServer.SLOT_CONTROL_OFFSET_PARAM => _ip(at),
    ReactantServer.SLOT_CONTROL_TOKEN_PARAM => _ip(t)
)

@testset "pages /dev/shm has no room for: RESOURCE_EXHAUSTED, nothing written (CORE-4)" begin
    sched, ctx = _sc_sched(_sc_entry("m"))
    ReactantServer.start!(sched)
    page = Int(ccall(:getpagesize, Cint, ()))
    key = _test_shm_key()
    # A third page past the registered window: the object's pages all exist, as a
    # ReactantServerClient pool's do, until the test frees this one.
    cfd, cptr, _ = _RSCore.shm_create_exclusive(key, 3 * page)
    full = nothing
    y() = [unsafe_load(Ptr{Float32}(cptr + page), i) for i in 1:2]
    word() = unsafe_load(Ptr{UInt64}(cptr + page + 64))
    try
        full = _RSCore._SHM_FALLOCATE[]
        ReactantServer.shm_register!(ctx.shm, "r", key, 0, 2 * page)
        unsafe_store!(Ptr{Float32}(cptr), 1.0f0, 1)
        unsafe_store!(Ptr{Float32}(cptr), 2.0f0, 2)
        @test _sc_call(ctx, _bk_body("r", 0, page)) isa Vector{UInt8}
        @test y() == Float32[2, 6]
        unsafe_store!(Ptr{UInt64}(cptr + page), UInt64(0))

        _RSCore._SHM_FALLOCATE[] = (fd, off, len) -> Libc.ENOSPC
        if Sys.islinux()
            # fstat proves every page there, so the worker allocates none and a full /dev/shm
            # (simulated) does not matter.
            @test _sc_call(ctx, _bk_body("r", 0, page)) isa Vector{UInt8}
            @test y() == Float32[2, 6]
            unsafe_store!(Ptr{UInt64}(cptr + page), UInt64(0))
            # A page freed anywhere in the object, even outside the window, and every copy allocates
            # again. Nothing below touches page 2.
            @test ccall(:fallocate, Cint, (Cint, Cint, Int64, Int64), cfd.fd, 0x03, 2 * page, page) == 0
        end
        @test _sc_code(_sc_call(ctx, _bk_body("r", 0, nothing))) == _SCStatus.RESOURCE_EXHAUSTED
        err = _sc_call(ctx, _bk_body("r", nothing, page))
        @test _sc_code(err) == _SCStatus.RESOURCE_EXHAUSTED
        @test occursin("/dev/shm", err.message)
        @test y() == Float32[0, 0]
        if ReactantServer.slot_control_supported()
            t = UInt64(99)
            unsafe_store!(Ptr{UInt64}(cptr + page + 64), _RSCore.slot_word(_RSCore.SLOT_OWNED, t))
            @test _sc_code(_sc_call(ctx, _bk_body("r", nothing, page; ctl = _bk_ctl("r", page + 64, t)))) ==
                _SCStatus.RESOURCE_EXHAUSTED
            @test word() == _RSCore.slot_word(_RSCore.SLOT_OWNED, t)   # refused before the claim
            @test y() == Float32[0, 0]
        end
    finally
        full === nothing || (_RSCore._SHM_FALLOCATE[] = full)
        ReactantServer.shutdown!(sched)
        ReactantServer.shm_teardown!(ctx.shm)
        _RSCore._shm_munmap(cptr, 3 * page)
        close(cfd)
        ccall(:shm_unlink, Cint, (Cstring,), key)
    end
end

# The real kernel cases, through the worker's ModelInfer and register handlers in a child process, so
# a regression (the worker dying with SIGBUS) fails the test instead of the suite. The child shrinks
# an object it registered, in the host's /dev/shm (a few pages); and, where unprivileged user and
# mount namespaces exist, it runs on a private 2 MiB /dev/shm, which it fills to the last page, and
# serves a region whose pages its client never allocated (an older or third-party client).
const _BK_WORKER_CHILD = raw"""
using ReactantServer
const RS = ReactantServer
const C = RS.ReactantServerCore
const IN = RS.inference
const PAGE = Int(ccall(:getpagesize, Cint, ()))
const MiB = 1 << 20

function entry()
    spec(n) = RS.TensorSpec(n, RS.F32, RS.Dim[RS.Dim(RS.FIXED, 2), RS.Dim(RS.BATCH)], 2)
    man = RS.Manifest(
        "2.0", "m", "", RS.TensorSpec[spec("x")], RS.TensorSpec[spec("y")], nothing, nothing,
        RS.BatchingSpec(Int[]), RS.Provenance(Dict{String, Any}()), 1
    )
    sig = RS.ModelSignature(["x"], DataType[Float32], ["w"], 1, ["y"], DataType[Float32], 1, Tuple{String, Int}[])
    exec = RS.MockExecutable(a -> [a[1] .* a[2]], 1)
    model = RS.LoadedModel(sig, Dict{Int, Any}(1 => exec, 4 => exec), Any[RS.MockBuffer(Float32[2, 3])])
    return RS.ModelEntry("m", man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
end
backend = RS.MockBackend()
mpool = RS.MemoryPool(backend, RS.MockClient(), RS.MockDevice(0), "mock", nothing)
models = RS.ModelRegistry()
models.by_name["m"] = entry()
sched = RS.Scheduler(models, backend, mpool, RS.SchedulerConfig(30.0, 64, 30.0; discipline = RS.FIFO))
ctx = RS.InferContext(sched, models, RS.SharedMemoryRegistry(; reap_interval = 0), "mock", nothing)
RS.start!(sched)

par(s::String) = IN.InferParameter(; parameter_choice = RS.ProtoBuf.OneOf(:string_param, s))
par(i::Integer) = IN.InferParameter(; parameter_choice = RS.ProtoBuf.OneOf(:int64_param, Int64(i)))
none() = Dict{String, IN.InferParameter}()
shm(region, at) = Dict(
    "shared_memory_region" => par(region), "shared_memory_offset" => par(at),
    "shared_memory_byte_size" => par(8)
)
function body(region, x_at, y_at; ctl = none())
    x = IN.var"ModelInferRequest.InferInputTensor"(;
        name = "x", datatype = "FP32", shape = Int64[1, 2], parameters = x_at === nothing ? none() : shm(region, x_at)
    )
    y = IN.var"ModelInferRequest.InferRequestedOutputTensor"(;
        name = "y", parameters = y_at === nothing ? none() : shm(region, y_at)
    )
    raw = x_at === nothing ? [collect(reinterpret(UInt8, Float32[1, 2]))] : Vector{UInt8}[]
    req = IN.ModelInferRequest(; model_name = "m", inputs = [x], outputs = [y], raw_input_contents = raw, parameters = ctl)
    return RS._encode_message(req)
end
ctl(region, at, t) = Dict(
    RS.SLOT_CONTROL_REGION_PARAM => par(region), RS.SLOT_CONTROL_OFFSET_PARAM => par(at),
    RS.SLOT_CONTROL_TOKEN_PARAM => par(t)
)
function status(tag, f)
    v = try
        f()
        "OK"
    catch e
        e isa RS.gRPCServer.GRPCError ? string(e.code) : "THROWN " * string(nameof(typeof(e)))
    end
    println(tag, " ", v)
    flush(stdout)
end
infer(b) = RS._handle_infer_raw(ctx, b, 0)
register(name, key, bytes) = RS._handle_shm_register(
    ctx.shm, IN.SystemSharedMemoryRegisterRequest(; name = name, key = key, offset = 0, byte_size = bytes)
)
store_x(ptr) = (unsafe_store!(Ptr{Float32}(ptr), 1.0f0, 1); unsafe_store!(Ptr{Float32}(ptr), 2.0f0, 2))

# A client (allocating its object, as ReactantServerClient does) shrinks it to one page after
# registering four.
key = ENV["BK_KEY"]
cfd, cptr, _ = C.shm_create_exclusive(key, 4 * PAGE)
store_x(cptr)
t = 5
unsafe_store!(Ptr{UInt64}(cptr + 3 * PAGE), C.slot_word(C.SLOT_OWNED, UInt64(t)))
status("REGISTER", () -> register("s", key, 4 * PAGE))
status("BEFORE", () -> infer(body("s", 0, 2 * PAGE)))
ccall(:ftruncate, Cint, (Cint, Int64), cfd.fd, PAGE)
status("SHRUNK_INPUT", () -> infer(body("s", 2 * PAGE, nothing)))
status("SHRUNK_OUTPUT", () -> infer(body("s", 0, 2 * PAGE)))
status("SHRUNK_CONTROL", () -> infer(body("s", 0, 64; ctl = ctl("s", 3 * PAGE, t))))       # word past the end
status("SHRUNK_INSIDE", () -> infer(body("s", 0, nothing)))
status("SHRUNK_REREGISTER", () -> register("s", key, 4 * PAGE))

if get(ENV, "BK_FULL", "0") == "1"
    # A client that sizes its object and never allocates it, on a /dev/shm then filled to the last page.
    fkey = "/bk-sparse"
    raw = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), fkey, 0o102, 0o600)    # O_RDWR | O_CREAT
    ccall(:ftruncate, Cint, (Cint, Int64), raw, 8MiB)
    fptr = ccall(:mmap, Ptr{UInt8}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64), C_NULL, 8MiB, 3, 1, raw, 0)
    store_x(fptr)                                                                # page 0 only
    status("FULL_REGISTER", () -> register("f", fkey, 8MiB))
    filler = ccall(:shm_open, Cint, (Cstring, Cint, UInt32), "/bk-filler", 0o102, 0o600)
    used = 0
    for chunk in (256 * 1024, PAGE)
        while ccall(:posix_fallocate, Cint, (Cint, Int64, Int64), filler, used, chunk) == 0
            global used += chunk
        end
    end
    status("FULL_OUTPUT", () -> infer(body("f", 0, 4MiB)))                     # a page nobody wrote
    status("FULL_INPUT", () -> infer(body("f", 4MiB, nothing)))                # one nobody wrote either
    status("FULL_ALLOCATED", () -> infer(body("f", 0, 64)))                    # page 0 has a page
    println("FULL_Y ", [unsafe_load(Ptr{Float32}(fptr + 64), i) for i in 1:2])
end
RS.shutdown!(sched)
"""

@testset "a shrunk region and a full /dev/shm fail the request, not the worker (CORE-4)" begin
    if !Sys.islinux()
        @test_skip false
    else
        unshare = Sys.which("unshare")
        full = unshare !== nothing && success(
            pipeline(
                `$unshare --user --map-root-user --mount sh -c 'mount -t tmpfs -o size=1m tmpfs /dev/shm'`;
                stdout = devnull, stderr = devnull
            )
        )
        full || @info "no unprivileged user and mount namespaces here: skipping the full /dev/shm part"
        key = _test_shm_key() * "c"         # sibling testsets draw the same `rand` key
        jl = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $_BK_WORKER_CHILD`
        cmd = if full
            mount = "mount -t tmpfs -o size=2m tmpfs /dev/shm && exec \"\$@\""
            `$unshare --user --map-root-user --mount sh -c $mount sh $jl`
        else
            jl
        end
        out, err = IOBuffer(), IOBuffer()
        env = ("BK_KEY" => key, "BK_FULL" => full ? "1" : "0")
        p = try
            run(pipeline(ignorestatus(addenv(cmd, env...)); stdout = out, stderr = err))
        finally
            ccall(:shm_unlink, Cint, (Cstring,), key)
        end
        text = String(take!(out))
        lines = Dict{String, String}()
        for l in split(text, '\n')
            parts = split(l, ' '; limit = 2)
            length(parts) == 2 && (lines[String(parts[1])] = String(parts[2]))
        end
        success(p) ||
            @error "worker child failed" signal = p.termsignal stdout = text stderr = last(String(take!(err)), 4000)
        @test p.termsignal != 7                       # not killed by SIGBUS
        @test success(p)
        @test get(lines, "REGISTER", "") == "OK"
        @test get(lines, "BEFORE", "") == "OK"
        # A shrunk region is a stale registration, for inputs, outputs and the control word alike...
        for tag in ("SHRUNK_INPUT", "SHRUNK_OUTPUT", "SHRUNK_CONTROL")
            @test get(lines, tag, "") == "FAILED_PRECONDITION"
        end
        @test get(lines, "SHRUNK_INSIDE", "") == "OK"
        # ...and registering it again is refused, so a client that re-registers on
        # FAILED_PRECONDITION (ReactantServerClient does, once) stops there.
        @test get(lines, "SHRUNK_REREGISTER", "") == "INVALID_ARGUMENT"
        if full
            @test get(lines, "FULL_REGISTER", "") == "OK"
            @test get(lines, "FULL_OUTPUT", "") == "RESOURCE_EXHAUSTED"
            @test get(lines, "FULL_INPUT", "") == "RESOURCE_EXHAUSTED"
            @test get(lines, "FULL_ALLOCATED", "") == "OK"
            @test get(lines, "FULL_Y", "") == "Float32[2.0, 6.0]"
        end
    end
end
