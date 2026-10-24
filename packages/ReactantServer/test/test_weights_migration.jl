# Weights in the program's axis order (`axis_order: xla`) and the migration that converts legacy
# bundles to it: the copy and replace modes, the proof that guards every step, crashes at each step,
# a redeployed original, a full disk, a read-only directory, concurrent workers, and a compiled
# program that must give identical results before and after.

const _RSM = ReactantServer

const _MIG_WEIGHTS = Dict{String, Array}(
    "w" => Float32[1 2 3; 4 5 6; 7 8 10],                       # square: a transpose would fit
    "v" => reshape(Float32.(1:24), 2, 3, 4),
    "b" => Float32[0.5, -0.0, 2],
    "h" => Float16[1 2; 3 4],
    "n" => reshape(reinterpret(Float32, UInt32[0x7fc00001, 0x80000000, 0x3f800000, 0x7f800001]), 2, 2),
)

const _MIG_MANIFEST = """
# a comment the migration must keep
format_version: "2.0"
name: NAME
executable_inputs:
  - {name: x, dtype: f32, shape: c, dims: {c: 3}}
executable_outputs:
  - {name: y, dtype: f32, shape: c, dims: {c: 3}}
batching:
  compiled_batch_sizes: [1]
"""

const _MATVEC_MLIR = """
module {
  func.func @main(%x: tensor<3xf32>, %w: tensor<3x3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.dot_general %w, %x, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""

function _mig_bundle(root, name; weights = _MIG_WEIGHTS, order = sort!(collect(keys(weights))))
    return write_bundle(
        root, name; manifest_yaml = replace(_MIG_MANIFEST, "NAME" => name), mlir_text = _MATVEC_MLIR,
        weights = Dict{String, Array}(weights), argument_order = order
    )
end

_mig_bytes(a) = reinterpret(UInt8, vec(collect(a)))

# Every tensor of the bundle's chosen weights file reads back as the expected Julia array.
function _mig_reads(dir, expected = _MIG_WEIGHTS)
    entry = _RSM.load_bundle_entry(dir)
    try
        keys(entry.weights) == keys(expected) || return false
        return all(_mig_bytes(_RSM.read_weight(entry.weights, k)) == _mig_bytes(v) for (k, v) in expected)
    finally
        _RSM.close_weights!(entry.weights)
    end
end

_xla(path) = _RSM._with_weights(w -> w.xla_order, path)
_mfst(dir) = _RSM.YAML.load_file(joinpath(dir, "manifest.yaml"))
_no_tmp(dir) = !any(f -> startswith(f, ".") && endswith(f, ".tmp"), readdir(dir))

# A safetensors file written by hand: `entries` is name => (dtype, header shape, data bytes).
function _raw_safetensors(path, entries, metadata)
    header = Dict{String, Any}("__metadata__" => metadata)
    off = 0
    for (n, (dt, shape, bytes)) in entries
        header[n] = Dict("dtype" => dt, "shape" => shape, "data_offsets" => [off, off + length(bytes)])
        off += length(bytes)
    end
    json = JSON3.write(header)
    json *= " "^((8 - sizeof(json) % 8) % 8)
    open(path, "w") do io
        write(io, htol(UInt64(sizeof(json))), json)
        for (_, (_, _, bytes)) in entries
            write(io, bytes)
        end
    end
    return path
end

@testset "weights in the program's axis order" begin
    mktempdir() do root
        dir = _mig_bundle(root, "layout")
        legacy = joinpath(dir, "weights.safetensors")
        md = Dict("argument_order" => JSON3.write(sort!(collect(keys(_MIG_WEIGHTS)))), "axis_order" => "xla")
        entries = [k => (k == "h" ? "F16" : "F32", reverse(collect(size(v))), Vector{UInt8}(_mig_bytes(v))) for (k, v) in _MIG_WEIGHTS]
        xla = _raw_safetensors(joinpath(root, "xla.safetensors"), entries, md)

        lw, _ = _RSM.open_weights_file(legacy)
        xw, _ = _RSM.open_weights_file(xla)
        @test !lw.xla_order && xw.xla_order
        for (k, v) in _MIG_WEIGHTS
            @test _RSM.weight_spec(xw, k) == (eltype(v), size(v))          # Julia shapes in both
            @test _mig_bytes(_RSM.read_weight(xw, k)) == _mig_bytes(v)
            dest = similar(v)
            @test _mig_bytes(_RSM.read_weight!(dest, xw, k)) == _mig_bytes(v)
        end
        @test isnothing(_RSM.verify_transposed(lw, xw; reference = legacy))
        # numpy/PyTorch view: SafeTensors.jl reads each tensor in the program's shape
        st = SafeTensors.deserialize(xla)
        @test size(st["v"]) == (4, 3, 2) && permutedims(collect(st["v"]), (3, 2, 1)) == _MIG_WEIGHTS["v"]
        _RSM.close_weights!(lw)
        _RSM.close_weights!(xw)

        md["axis_order"] = "fortran"
        bad = _raw_safetensors(joinpath(root, "bad.safetensors"), entries, md)
        @test_throws "unsupported weights axis_order" _RSM.open_weights_file(bad)
    end
end

@testset "a legacy layout warns once, naming the ways out" begin
    mktempdir() do root
        name = "warnlegacy_" * string(getpid())
        dir = _mig_bundle(root, name)
        delete!(_RSM._LEGACY_WARNED, name)
        e = @test_logs (:warn, r"legacy axis order.*weights_migration.*migrate_weights.jl.*ReactantServerExport 2.0") match_mode = :any _RSM.load_bundle_entry(dir)
        _RSM.close_weights!(e.weights)
        e = @test_logs min_level = Base.CoreLogging.Warn _RSM.load_bundle_entry(dir)     # once per model
        _RSM.close_weights!(e.weights)
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) == :replaced
        delete!(_RSM._LEGACY_WARNED, name)
        e = @test_logs min_level = Base.CoreLogging.Warn _RSM.load_bundle_entry(dir)     # converted: silent
        _RSM.close_weights!(e.weights)
        vec_only = _mig_bundle(root, name * "_1d"; weights = Dict{String, Array}("b" => Float32[1, 2, 3]))
        e = @test_logs min_level = Base.CoreLogging.Warn _RSM.load_bundle_entry(vec_only)  # no extra copy
        _RSM.close_weights!(e.weights)
    end
end

@testset "manifest weights_file and format 2.1" begin
    base = Dict{String, Any}(
        "format_version" => "2.1", "name" => "m",
        "executable_inputs" => [Dict("name" => "x", "dtype" => "f32", "shape" => "c", "dims" => Dict("c" => 3))],
        "executable_outputs" => [Dict("name" => "y", "dtype" => "f32", "shape" => "c", "dims" => Dict("c" => 3))],
    )
    m = _RSM.parse_manifest(base)
    @test m.weights_file == "weights.safetensors"
    @test (_RSM.validate_manifest(m, ".", false); true)          # 2.1 is accepted
    @test _RSM.parse_manifest(merge(base, Dict("weights_file" => "weights.T.safetensors"))).weights_file == "weights.T.safetensors"
    for bad in ("../weights.safetensors", "weights/../../x.safetensors", "model.bin", "/etc/weights.safetensors", 3)
        @test_throws _RSM.ManifestError _RSM.parse_manifest(merge(base, Dict("weights_file" => bad)))
    end
    @test_throws _RSM.ManifestError _RSM.validate_manifest(_RSM.parse_manifest(merge(base, Dict("format_version" => "2.2"))), ".", false)
end

@testset "the verifier refuses anything but an exact transpose" begin
    mktempdir() do root
        dir = _mig_bundle(root, "verify")
        legacy = joinpath(dir, "weights.safetensors")
        order = JSON3.write(sort!(collect(keys(_MIG_WEIGHTS))))
        good(k, v) = (k == "h" ? "F16" : "F32", reverse(collect(size(v))), Vector{UInt8}(_mig_bytes(v)))
        variant(f; md = Dict("argument_order" => order, "axis_order" => "xla")) =
            _raw_safetensors(joinpath(root, "v.safetensors"), [k => f(k, v) for (k, v) in _MIG_WEIGHTS], md)
        refused(path) = try
            _RSM.verify_transposed(legacy, path); false
        catch e
            e isa _RSM.WeightsMigrationError
        end

        @test !refused(variant(good))
        @test refused(variant((k, v) -> k == "w" ? (t = good(k, v); t[3][5] ⊻= 0x01; t) : good(k, v)))   # one bit
        @test refused(variant((k, v) -> k == "w" ? ("F32", [3, 3], Vector{UInt8}(_mig_bytes(permutedims(v)))) : good(k, v)))  # untransposed
        @test refused(variant((k, v) -> k == "v" ? ("F32", collect(size(v)), good(k, v)[3]) : good(k, v)))  # shape not reversed
        @test refused(variant((k, v) -> k == "b" ? ("F32", [3], Vector{UInt8}(_mig_bytes(Float32[0.5, 0.0, 2]))) : good(k, v)))  # -0.0 vs 0.0
        @test refused(variant(good; md = Dict("argument_order" => JSON3.write(["w", "v", "b", "h", "n"]), "axis_order" => "xla")))
        @test refused(variant(good; md = Dict("argument_order" => order, "axis_order" => "xla", "extra" => "1")))
        @test refused(variant(good; md = Dict("argument_order" => order)))              # not marked
        # two same-shape tensors swapped
        sw = Dict{String, Array}(_MIG_WEIGHTS)
        sw["w2"] = Float32[9 8 7; 6 5 4; 3 2 1]
        d2 = _mig_bundle(root, "verify2"; weights = sw)
        swapped = _raw_safetensors(
            joinpath(root, "s.safetensors"),
            [k => (k == "h" ? "F16" : "F32", reverse(collect(size(v))), Vector{UInt8}(_mig_bytes(sw[k == "w" ? "w2" : k == "w2" ? "w" : k]))) for (k, v) in sw],
            Dict("argument_order" => JSON3.write(sort!(collect(keys(sw)))), "axis_order" => "xla")
        )
        @test_throws _RSM.WeightsMigrationError _RSM.verify_transposed(joinpath(d2, "weights.safetensors"), swapped)
    end
end

@testset "copy mode" begin
    mktempdir() do root
        dir = _mig_bundle(root, "copym")
        legacy = joinpath(dir, "weights.safetensors")
        before = read(legacy)
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_OFF) == :off
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
        @test read(legacy) == before                                     # original untouched
        @test _xla(joinpath(dir, "weights.T.safetensors"))
        m = _mfst(dir)
        @test m["weights_file"] == "weights.T.safetensors" && m["format_version"] == "2.0"
        @test startswith(read(joinpath(dir, "manifest.yaml"), String), "# a comment the migration must keep")
        @test _RSM.load_bundle_entry(dir).weights_path == joinpath(dir, "weights.T.safetensors")
        @test _mig_reads(dir)
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :current
        @test _no_tmp(dir)
    end
end

@testset "replace mode" begin
    for path in (:direct, :after_copy)
        mktempdir() do root
            dir = _mig_bundle(root, "replm")
            path === :after_copy && @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
            @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) == :replaced
            @test _xla(joinpath(dir, "weights.safetensors"))
            @test !isfile(joinpath(dir, "weights.T.safetensors"))
            m = _mfst(dir)
            @test m["format_version"] == "2.1" && !haskey(m, "weights_file")
            @test startswith(read(joinpath(dir, "manifest.yaml"), String), "# a comment the migration must keep")
            @test _mig_reads(dir)
            @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) == :current
            @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :current
            @test _no_tmp(dir)
        end
    end
end

@testset "a worker holding the original keeps reading it through replace" begin
    mktempdir() do root
        dir = _mig_bundle(root, "holder")
        held = _RSM.load_bundle_entry(dir)
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) == :replaced
        @test !held.weights.xla_order
        @test all(_mig_bytes(_RSM.read_weight(held.weights, k)) == _mig_bytes(v) for (k, v) in _MIG_WEIGHTS)
        _RSM.close_weights!(held.weights)
    end
end

@testset "a crash at any step leaves a loadable bundle and resumes" begin
    for step in (:write_copy, :point_manifest, :bump_format, :swap, :unpoint_manifest, :unlink_copy)
        mktempdir() do root
            dir = _mig_bundle(root, "crash")
            _RSM._MIGRATION_STEP_HOOK[] = s -> s === step && error("simulated crash at $s")
            try
                @test (@test_logs (:error, r"weights migration failed") match_mode = :any _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE)) == :failed
            finally
                _RSM._MIGRATION_STEP_HOOK[] = nothing
            end
            @test _no_tmp(dir)
            @test _mig_reads(dir)                                   # a 0.2.0 server loads it correctly
            @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) in (:replaced, :current)
            @test _xla(joinpath(dir, "weights.safetensors")) && !isfile(joinpath(dir, "weights.T.safetensors"))
            @test _mfst(dir)["format_version"] == "2.1" && !haskey(_mfst(dir), "weights_file")
            @test _mig_reads(dir)
        end
    end
end

@testset "a redeployed original is never ignored" begin
    mktempdir() do root
        dir = _mig_bundle(root, "redeploy")
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
        newer = Dict{String, Array}(k => v .* 2 for (k, v) in _MIG_WEIGHTS)
        tmp = mktempdir()
        _mig_bundle(tmp, "redeploy"; weights = newer)
        mv(joinpath(tmp, "redeploy", "weights.safetensors"), joinpath(dir, "weights.safetensors"); force = true)
        # migration off: the stale copy is skipped and the new original loads
        @test (@test_logs (:warn, r"predates") match_mode = :any _mig_reads(dir, newer))
        # copy mode rebuilds the copy from the new original
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
        @test _RSM.load_bundle_entry(dir).weights_path == joinpath(dir, "weights.T.safetensors")
        @test _mig_reads(dir, newer)
    end
end

@testset "a corrupted copy is caught before anything is replaced" begin
    mktempdir() do root
        dir = _mig_bundle(root, "corrupt")
        legacy = joinpath(dir, "weights.safetensors")
        before = read(legacy)
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
        tpath = joinpath(dir, "weights.T.safetensors")
        tw, _ = _RSM.open_weights_file(tpath)
        off = tw.tensors["w"].offset
        _RSM.close_weights!(tw)
        open(tpath, "r+") do io
            seek(io, off + 8)
            b = read(io, UInt8)
            seek(io, off + 8)
            write(io, b ⊻ 0x10)
        end
        @test (@test_logs (:error, r"weights migration failed") match_mode = :any _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE)) == :failed
        @test read(legacy) == before                                     # the original survives
        @test _mfst(dir)["format_version"] == "2.0"
    end
end

@testset "migration falls back without failing the model" begin
    mktempdir() do root
        dir = _mig_bundle(root, "full")
        saved = _RSM._AVAILABLE_BYTES[]
        _RSM._AVAILABLE_BYTES[] = _ -> 0
        try
            @test (@test_logs (:error, r"weights migration failed") match_mode = :any _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY)) == :failed
        finally
            _RSM._AVAILABLE_BYTES[] = saved
        end
        @test !isfile(joinpath(dir, "weights.T.safetensors")) && _no_tmp(dir)
        @test !haskey(_mfst(dir), "weights_file")
        @test _mig_reads(dir)

        ro = _mig_bundle(root, "readonly")
        if ccall(:geteuid, Cuint, ()) == 0
            @test_skip "root ignores directory permissions"
        else
            chmod(ro, 0o555)
            try
                @test (@test_logs (:error, r"weights migration failed") match_mode = :any _RSM.migrate_bundle_weights!(ro, _RSM.MIGRATE_REPLACE)) == :failed
                @test _mig_reads(ro)
            finally
                chmod(ro, 0o755)
            end
        end

        # a model whose manifest names its own weights file is left alone
        custom = _mig_bundle(root, "custom")
        mv(joinpath(custom, "weights.safetensors"), joinpath(custom, "weights.mine.safetensors"))
        open(joinpath(custom, "manifest.yaml"), "a") do io
            write(io, "weights_file: weights.mine.safetensors\n")
        end
        @test _RSM.migrate_bundle_weights!(custom, _RSM.MIGRATE_REPLACE) == :skipped
        @test _mig_reads(custom)
    end
end

@testset "concurrent workers migrate a bundle once" begin
    if Threads.nthreads() < 2
        @test_skip "needs two threads"
    else
        mktempdir() do root
            dir = _mig_bundle(root, "race")
            results = fetch.([Threads.@spawn(_RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE)) for _ in 1:4])
            @test count(==(:replaced), results) == 1 && count(==(:current), results) == 3
            @test _mig_reads(dir)
        end
    end
end

@testset "a migrated bundle computes the same results (CPU)" begin
    backend = _RSM.ReactantBackend()
    cfg = _RSM.RuntimeConfig(_RSM.CPU_BACKEND, 0, 0.9, true, true)
    pool = _RSM.resolve_client(backend, cfg)
    W = Float32[1 2 3; 4 5 6; 7 8 10]
    x = Float32[1, -2, 3]
    run(dir) = begin
        entry = _RSM.load_bundle_entry(dir)
        entry.executable = _RSM.build_loaded_model(backend, pool, entry)
        _RSM.run_model(backend, pool, entry.executable, [_RSM.NamedTensor("x", x)])[1].data
    end
    mktempdir() do root
        dir = _mig_bundle(root, "matvec"; weights = Dict{String, Array}("w" => W))
        y_legacy = run(dir)
        @test y_legacy == permutedims(W) * x      # a Julia (3,3) weight is the program's W'
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_COPY) == :copied
        @test run(dir) == y_legacy
        @test _RSM.migrate_bundle_weights!(dir, _RSM.MIGRATE_REPLACE) == :replaced
        @test run(dir) == y_legacy
    end
end
