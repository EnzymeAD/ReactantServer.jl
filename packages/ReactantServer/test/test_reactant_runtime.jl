# End-to-end runtime on the CPU PJRT backend: load a real bundle from disk, compile the
# StableHLO with a weight as an explicit argument, and execute it.

@testset "reactant runtime (CPU)" begin
    backend = ReactantServer.ReactantBackend()
    cfg = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
    pool = ReactantServer.resolve_client(backend, cfg)

    mktempdir() do root
        manifest = """
        format_version: "2.0"
        name: scale4
        description: "elementwise multiply by a weight vector"
        executable_inputs:
          - name: x
            dtype: f32
            shape: c
            dims:
              c: 4
        executable_outputs:
          - name: y
            dtype: f32
            shape: c
            dims:
              c: 4
        batching:
          compiled_batch_sizes: [1]
        provenance:
          source: handwritten
        """
        mlir = """
        module {
          func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.multiply %x, %w : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        write_bundle(root, "scale4";
            manifest_yaml=manifest, mlir_text=mlir,
            weights=Dict("w" => Float32[2, 2, 2, 2]), argument_order=["w"])

        reg = ReactantServer.load_bundles([root])
        entry = ReactantServer.get_model(reg, "scale4")
        @test entry !== nothing

        entry.executable = ReactantServer.build_loaded_model(backend, pool, entry)
        @test entry.executable.sig.weight_names == ["w"]
        @test ReactantServer.num_parameters(entry.executable.sig) == 2

        out = ReactantServer.run_model(backend, pool, entry.executable,
            [ReactantServer.NamedTensor("x", Float32[1, 2, 3, 4])])
        @test length(out) == 1
        @test out[1].name == "y"
        @test out[1].data == Float32[2, 4, 6, 8]
    end

    # 1-D host<->device transfer round-trips exactly. The slice's models are 1-D, so this
    # is the property the runtime relies on.
    v = Float32[5, 6, 7, 8]
    vb = ReactantServer.to_device(backend, pool.client, v, pool.device)
    vd = Array{Float32}(undef, ReactantServer.buffer_size(backend, vb))
    ReactantServer.to_host!(backend, vb, vd)
    @test vd == v

    # Reactant transfers multi-dimensional arrays with reversed dims (column-major Julia
    # vs row-major XLA): a Julia (2,3) array becomes an XLA (3,2) tensor. Normalizing this
    # for client-facing multi-dimensional tensors is a documented follow-up (the layout
    # item flagged in planning). Here we pin the convention so a future change is noticed.
    A = Float32[1 2 3; 4 5 6]
    buf = ReactantServer.to_device(backend, pool.client, A, pool.device)
    @test ReactantServer.buffer_size(backend, buf) == (3, 2)
end
