# Shared-memory data plane end to end: create a POSIX region, register it over gRPC, and run
# inference referencing it for inputs (inline output) and for both inputs and outputs (with a
# nonzero output offset). Mirrors the Triton system shared-memory extension.

import InterProcessCommunication as IPC

const _ShmInf = ReactantServer.inference

_sp(s) = _ShmInf.InferParameter(; parameter_choice=ReactantServer.ProtoBuf.OneOf(:string_param, String(s)))
_ip(i) = _ShmInf.InferParameter(; parameter_choice=ReactantServer.ProtoBuf.OneOf(:int64_param, Int64(i)))

# Create a POSIX shm region; returns (handle, key, byte-view). volatile=true so the handle
# unlinks the object when finalized at the end of the test.
function _make_region(nbytes::Int)
    key = "/reactantserver-test-$(getpid())-$(rand(UInt32))"
    shm = IPC.SharedMemory(key, nbytes)
    view = unsafe_wrap(Array, convert(Ptr{UInt8}, pointer(shm)), nbytes; own=false)
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

        # bounds and unknown-region errors are enforced
        @test_throws ArgumentError ReactantServer.shm_read(reg, "r", 24, 16)
        @test_throws ArgumentError ReactantServer.shm_write!(reg, "r", 24, payload)
        @test_throws ArgumentError ReactantServer.shm_read(reg, "missing", 0, 4)

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
        @test same_ipc_namespace("/reactantserver-test-no-such-object-$(getpid())") == false
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
        write_bundle(root, "scale4"; manifest_yaml=manifest, mlir_text=mlir,
            weights=Dict("w" => Float32[2, 2, 2, 2]), argument_order=["w"])

        port = grpc_free_port()
        cfg = ReactantServer.ServerConfig([root], "",
            ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true),
            ReactantServer.SchedulerConfig(30.0, 64, 30.0),
            ReactantServer.EndpointsConfig("127.0.0.1", port))
        srv = ReactantServer.serve(cfg; backend=ReactantServer.ReactantBackend(), blocking=false)
        sleep(0.3)

        rin, kin, vin = _make_region(16)               # input region
        rio, kio, vio = _make_region(32)                # input at [0,16), output at [16,32)
        try
            # --- register both regions ---
            for (rname, key, bytes) in (("rin", kin, 16), ("rio", kio, 32))
                resp = grpc_call(_ShmInf.SystemSharedMemoryRegisterRequest, _ShmInf.SystemSharedMemoryRegisterResponse,
                    "SystemSharedMemoryRegister", port,
                    _ShmInf.SystemSharedMemoryRegisterRequest(; name=rname, key=key, offset=0, byte_size=bytes))
                @test resp isa _ShmInf.SystemSharedMemoryRegisterResponse
            end

            # status reports both regions
            st = grpc_call(_ShmInf.SystemSharedMemoryStatusRequest, _ShmInf.SystemSharedMemoryStatusResponse,
                "SystemSharedMemoryStatus", port, _ShmInf.SystemSharedMemoryStatusRequest(; name=""))
            @test haskey(st.regions, "rin") && haskey(st.regions, "rio")
            @test st.regions["rin"].byte_size == 16
            @test st.regions["rin"].key == kin

            # --- A: input from shm, output inline ---
            copyto!(vin, reinterpret(UInt8, Float32[1, 2, 3, 4]))
            inA = _ShmInf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[4],
                parameters=Dict("shared_memory_region" => _sp("rin"),
                                "shared_memory_offset" => _ip(0),
                                "shared_memory_byte_size" => _ip(16)))
            mA = grpc_call(_ShmInf.ModelInferRequest, _ShmInf.ModelInferResponse, "ModelInfer", port,
                _ShmInf.ModelInferRequest(; model_name="scale4", inputs=[inA]))
            @test collect(reinterpret(Float32, mA.raw_output_contents[1])) == Float32[2, 4, 6, 8]

            # --- B: input from shm, output written to shm at offset 16 ---
            copyto!(view(vio, 1:16), reinterpret(UInt8, Float32[3, 4, 5, 6]))
            inB = _ShmInf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[4],
                parameters=Dict("shared_memory_region" => _sp("rio"),
                                "shared_memory_offset" => _ip(0),
                                "shared_memory_byte_size" => _ip(16)))
            outB = _ShmInf.var"ModelInferRequest.InferRequestedOutputTensor"(; name="y",
                parameters=Dict("shared_memory_region" => _sp("rio"),
                                "shared_memory_offset" => _ip(16),
                                "shared_memory_byte_size" => _ip(16)))
            mB = grpc_call(_ShmInf.ModelInferRequest, _ShmInf.ModelInferResponse, "ModelInfer", port,
                _ShmInf.ModelInferRequest(; model_name="scale4", inputs=[inB], outputs=[outB]))
            @test isempty(mB.raw_output_contents)                 # output went to shm, not inline
            @test mB.outputs[1].name == "y"
            # the server wrote the result into the region at offset 16
            @test reinterpret(Float32, view(vio, 17:32)) == Float32[6, 8, 10, 12]

            # --- unregister one, then confirm it is gone ---
            ru = grpc_call(_ShmInf.SystemSharedMemoryUnregisterRequest, _ShmInf.SystemSharedMemoryUnregisterResponse,
                "SystemSharedMemoryUnregister", port, _ShmInf.SystemSharedMemoryUnregisterRequest(; name="rin"))
            @test ru isa _ShmInf.SystemSharedMemoryUnregisterResponse
            st2 = grpc_call(_ShmInf.SystemSharedMemoryStatusRequest, _ShmInf.SystemSharedMemoryStatusResponse,
                "SystemSharedMemoryStatus", port, _ShmInf.SystemSharedMemoryStatusRequest(; name=""))
            @test !haskey(st2.regions, "rin") && haskey(st2.regions, "rio")

            # referencing an unregistered region is a client error (INVALID_ARGUMENT)
            inBad = _ShmInf.var"ModelInferRequest.InferInputTensor"(; name="x", datatype="FP32", shape=Int64[4],
                parameters=Dict("shared_memory_region" => _sp("rin"),
                                "shared_memory_byte_size" => _ip(16)))
            try
                grpc_call(_ShmInf.ModelInferRequest, _ShmInf.ModelInferResponse, "ModelInfer", port,
                    _ShmInf.ModelInferRequest(; model_name="scale4", inputs=[inBad]))
                @test false
            catch ex
                @test ex isa gRPCClient.gRPCServiceCallException
                @test ex.grpc_status == ReactantServer._G.GRPC_INVALID_ARGUMENT
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
