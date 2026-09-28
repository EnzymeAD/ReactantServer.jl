# The vendored executable-serialization and allocator-statistics bindings (runtime/xla_serialization.jl),
# checked directly on the CPU PJRT client, independently of the cache built on top of them. The cases
# follow Reactant's own test for the same API (EnzymeAD/Reactant.jl#3305, test/core/executable_serialization.jl).

using ReactantServer: VendoredXLA

@testset "vendored XLA serialization (CPU)" begin
    backend = ReactantServer.ReactantBackend()
    cfg = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
    pool = ReactantServer.resolve_client(backend, cfg)
    artifact = stablehlo_artifact(
        """
        func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
          %0 = stablehlo.sine %x : tensor<4xf32>
          %1 = stablehlo.add %0, %y : tensor<4xf32>
          return %1 : tensor<4xf32>
        }
        """
    )
    exec = ReactantServer.compile_artifact(backend, pool, artifact, 2, 1)
    x, y = Float32[0.1, 0.2, 0.3, 0.4], Float32[1, 2, 3, 4]
    function run(e)
        bufs = [ReactantServer.to_device(backend, pool.client, a, pool.device) for a in (x, y)]
        out = only(ReactantServer.execute_single_device(backend, e, pool.device, bufs, [false, false], 1))
        return ReactantServer.to_host!(backend, out, zeros(Float32, 4))
    end
    expected = run(exec)
    program = (;
        num_parameters = exec.num_parameters, num_outputs = exec.num_outputs,
        is_sharded = exec.is_sharded, num_replicas = exec.num_replicas, num_partitions = exec.num_partitions,
    )

    bytes = VendoredXLA.serialize_executable(exec)
    @test bytes isa Vector{UInt8}
    @test !isempty(bytes)

    @testset "round trip" begin
        loaded = VendoredXLA.load_serialized_executable(pool.client, bytes; program...)
        @test loaded isa typeof(exec)
        # The loaded program is the compiled program, so the results are bit-identical.
        @test run(loaded) == expected
    end

    @testset "compile options override" begin
        opts = Reactant.XLA.make_compile_options(; device_id = Int64(ReactantServer.device_ordinal(backend, pool.device)))
        loaded = VendoredXLA.load_serialized_executable(pool.client, bytes; compile_options = opts, program...)
        @test run(loaded) == expected
    end

    @testset "invalid bytes" begin
        @test_throws Reactant.XLA.ReactantInternalError VendoredXLA.load_serialized_executable(
            pool.client, UInt8[0x01, 0x02, 0x03]; program...
        )
    end

    @testset "compiled memory stats" begin
        stats = VendoredXLA.compiled_memory_stats(exec)
        @test stats isa VendoredXLA.CompiledMemoryStats
        @test stats.argument_size_in_bytes == 2 * 4 * sizeof(Float32)
        @test stats.output_size_in_bytes == 4 * sizeof(Float32)
        @test stats.temp_size_in_bytes >= 0
        @test stats.peak_memory_in_bytes >= 0
    end

    @testset "clear_memory_stats! on the CPU device" begin
        # The CPU allocator keeps no statistics, so there is nothing to reset and the C side throws.
        @test_throws Reactant.XLA.ReactantInternalError VendoredXLA.clear_memory_stats!(pool.device)
    end
end
