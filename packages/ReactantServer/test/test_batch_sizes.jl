# runtime.batch_sizes and runtime.xla_flags on the worker: which compiled batch sizes a bundle load
# reads, how the executable cache treats the sizes it skipped, and how XLA flags are checked and
# applied. The compile tests run on Reactant's CPU client.

using ReactantServer: load_bundle_entry, executable_cache_slots, sync_mlir_hashes!, entry_path,
    store_entry, ExecutableCacheSlot, EXEC_CACHE_HASHES_FILE, sha256hex, resolve_xla_flags,
    _compile_options, _cache_policy, _coalesce_inputs, _slice_outputs, BATCH_SIZES_ALL,
    BATCH_SIZES_LARGEST, ConfigError, BundleError, VariantKey

# y = x .* w over a (4, n) batch, compiled per batch size n. StableHLO is row-major, so the Julia
# (4, n) tensor is tensor<n x 4>.
function _bs_mlir(n::Int)
    return """
    module {
      func.func @main(%x: tensor<$(n)x4xf32>, %w: tensor<4xf32>) -> tensor<$(n)x4xf32> {
        %0 = stablehlo.broadcast_in_dim %w, dims = [1] : (tensor<4xf32>) -> tensor<$(n)x4xf32>
        %1 = stablehlo.multiply %x, %0 : tensor<$(n)x4xf32>
        return %1 : tensor<$(n)x4xf32>
      }
    }
    """
end

function _write_multi_batch_bundle(root, name; sizes = [1, 4, 8], files = sizes)
    dir = joinpath(root, name)
    mkpath(dir)
    write(
        joinpath(dir, "manifest.yaml"), """
        format_version: "2.0"
        name: $name
        executable_inputs:
          - {name: x, dtype: f32, shape: cn, dims: {c: 4}}
        executable_outputs:
          - {name: y, dtype: f32, shape: cn, dims: {c: 4}}
        batching: {compiled_batch_sizes: [$(join(sizes, ", "))]}
        """
    )
    for n in files
        write(joinpath(dir, "model.b$n.mlir"), stablehlo_artifact(_bs_mlir(n)))
    end
    SafeTensors.serialize(
        joinpath(dir, "weights.safetensors"), Dict("w" => Float32[1, 2, 3, 4]),
        Dict("argument_order" => JSON3.write(["w"]))
    )
    return dir
end

_loaded_sizes(entry) = sort!(collect(keys(entry.mlir_bytes[VariantKey()])))

@testset "batch_sizes: which modules a bundle load reads" begin
    mktempdir() do root
        dir = _write_multi_batch_bundle(root, "bs")
        @test _loaded_sizes(load_bundle_entry(dir)) == [1, 4, 8]
        @test _loaded_sizes(load_bundle_entry(dir; batch_sizes = BATCH_SIZES_LARGEST)) == [8]

        # Under largest, the smaller declared files are not required at all; under all they are.
        partial = _write_multi_batch_bundle(root, "partial"; files = [8])
        @test _loaded_sizes(load_bundle_entry(partial; batch_sizes = BATCH_SIZES_LARGEST)) == [8]
        @test_throws BundleError load_bundle_entry(partial)

        # The largest declared size must exist.
        nolargest = _write_multi_batch_bundle(root, "nolargest"; files = [1, 4])
        @test_throws BundleError load_bundle_entry(nolargest; batch_sizes = BATCH_SIZES_LARGEST)

        # load_bundles passes the mode to every bundle.
        reg = ReactantServer.load_bundles([root]; include = ["bs"], batch_sizes = BATCH_SIZES_LARGEST)
        @test _loaded_sizes(ReactantServer.get_model(reg, "bs")) == [8]
    end
end

@testset "batch_sizes: largest keeps the cached programs of the sizes it skipped" begin
    mktempdir() do root
        dir = _write_multi_batch_bundle(root, "bs")
        cache = joinpath(dir, ".cache")
        # A worker loading every size has cached one program per size.
        full = executable_cache_slots(load_bundle_entry(dir))
        programs = Dict(
            sz => entry_path(full[(VariantKey(), sz)], "target", "k"^64) for sz in (1, 4, 8)
        )
        for p in values(programs)
            @test store_entry(p, UInt8[1, 2, 3])
        end
        record = ReactantServer._read_hashes(cache)
        @test sort!(collect(keys(record))) == ["model.b1.mlir", "model.b4.mlir", "model.b8.mlir"]

        # A worker loading only the largest size, on the same directory, deletes nothing.
        slots = executable_cache_slots(load_bundle_entry(dir; batch_sizes = BATCH_SIZES_LARGEST))
        @test collect(keys(slots)) == [(VariantKey(), 8)]
        @test all(isfile, values(programs))
        @test ReactantServer._read_hashes(cache) == record

        # A file that really left the bundle (undeclared and gone) is still swept as before.
        @test sync_mlir_hashes!(cache, Dict("model.b8.mlir" => record["model.b8.mlir"])) ==
            ["model.b1.mlir", "model.b4.mlir"]
        @test !isfile(programs[1]) && !isfile(programs[4]) && isfile(programs[8])
    end
end

@testset "xla_flags: checked against DebugOptions" begin
    backend = ReactantServer.ReactantBackend()
    flags = resolve_xla_flags(
        backend, Dict{String, Any}(
            "xla_gpu_exclude_nondeterministic_ops" => true,
            "xla_gpu_autotune_level" => 2,
            "xla_gpu_experimental_autotune_cache_mode" => "AUTOTUNE_CACHE_MODE_READ",
        )
    )
    @test first.(flags) == [
        :xla_gpu_autotune_level, :xla_gpu_exclude_nondeterministic_ops,
        :xla_gpu_experimental_autotune_cache_mode,
    ]
    @test flags[1].second === Int32(2)                # converted to the proto field's type
    @test flags[2].second === true
    @test string(Symbol(flags[3].second)) == "AUTOTUNE_CACHE_MODE_READ"

    @test_throws ConfigError resolve_xla_flags(backend, Dict{String, Any}("xla_gpu_no_such_flag" => true))
    @test_throws ConfigError resolve_xla_flags(backend, Dict{String, Any}("xla_gpu_deterministic_ops" => 1))
    @test_throws ConfigError resolve_xla_flags(backend, Dict{String, Any}("xla_gpu_autotune_level" => true))
    @test_throws ConfigError resolve_xla_flags(backend, Dict{String, Any}("xla_gpu_autotune_level" => 2^40))
    @test_throws ConfigError resolve_xla_flags(
        backend, Dict{String, Any}("xla_gpu_experimental_autotune_cache_mode" => "NOPE")
    )

    # The mock backend has no XLA to check against; it only converts and sorts.
    @test resolve_xla_flags(ReactantServer.MockBackend(), Dict{String, Any}("b" => 1, "a" => true)) ==
        [:a => true, :b => 1]
end

@testset "xla_flags + largest: compile, cache key, padded execution (CPU)" begin
    backend = ReactantServer.ReactantBackend()
    base = ReactantServer.RuntimeConfig(ReactantServer.CPU_BACKEND, 0, 0.9, true, true)
    flagged = ReactantServer.RuntimeConfig(
        (getfield(base, f) for f in fieldnames(ReactantServer.RuntimeConfig)[1:(end - 3)])...,
        ReactantServer.PROFILE_REGULATED, BATCH_SIZES_LARGEST,
        Dict{String, Any}("xla_gpu_exclude_nondeterministic_ops" => true),
    )
    pool0 = ReactantServer.resolve_client(backend, base)
    pool = ReactantServer.resolve_client(backend, flagged)
    @test pool.xla_flags == [:xla_gpu_exclude_nondeterministic_ops => true]

    # The flag reaches the compile options, and splits the executable cache key; no flags leaves
    # the key exactly as it was.
    opts = _compile_options(pool, 0)
    @test opts.executable_build_options.debug_options.xla_gpu_exclude_nondeterministic_ops
    @test !occursin("xla=", _cache_policy(pool0, false))
    @test endswith(_cache_policy(pool, false), ";xla=xla_gpu_exclude_nondeterministic_ops=true")

    mktempdir() do root
        dir = _write_multi_batch_bundle(root, "bs")
        entry = load_bundle_entry(dir; batch_sizes = flagged.batch_sizes)
        entry.executable = ReactantServer.build_loaded_model(backend, pool, entry; executable_cache = false)
        @test ReactantServer._all_batch_sizes(entry.executable) == [8]

        # Three one-row requests pad to the one compiled size and each gets its own row back.
        xs = [Float32[i, i, i, i][:, :] for i in 1:3]
        pres = [[ReactantServer.NamedTensor("x", x)] for x in xs]
        merged = _coalesce_inputs(entry, pres, 3, 8)
        @test size(merged[1].data) == (4, 8)
        out = ReactantServer.run_model(backend, pool, entry.executable, merged)
        for i in 1:3
            @test _slice_outputs(entry, out, i - 1, 1)[1].data == xs[i] .* Float32[1, 2, 3, 4]
        end
    end
end
