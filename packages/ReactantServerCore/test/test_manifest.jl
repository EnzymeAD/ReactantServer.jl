@testset "manifest parse + validate" begin
    good = Dict{String,Any}(
        "format_version" => "2.0",
        "name" => "resnet",
        "executable_inputs" => [Dict("name" => "pixel_values", "dtype" => "f32",
                                     "shape" => "nchw",
                                     "dims" => Dict("c" => 3, "h" => 224, "w" => 224))],
        "executable_outputs" => [Dict("name" => "logits", "dtype" => "f32",
                                      "shape" => "nk",
                                      "dims" => Dict("k" => 1000))],
        "batching" => Dict("compiled_batch_sizes" => [1, 2, 4]),
    )
    m = ReactantServer.parse_manifest(good)
    @test m.name == "resnet"
    @test length(m.executable_inputs) == 1
    @test m.executable_inputs[1].batch_axis == 1          # 'n' at Julia index 1 (0-based axis 0)
    @test m.executable_inputs[1].shape[2] == ReactantServer.Dim(ReactantServer.FIXED, 3)
    @test m.executable_inputs[1].shape[1] == ReactantServer.Dim(ReactantServer.BATCH)
    @test m.batching.compiled_batch_sizes == [1, 2, 4]
    @test m.input_batch_dim == 0
    @test ReactantServer.validate_manifest(m, "/models/resnet", false) === m

    # name must match the directory name
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(m, "/models/other", false)

    # 'b' is the alternate batch marker; variable dim with -1
    withclient = copy(good)
    withclient["client_inputs"] = [Dict("name" => "img", "dtype" => "u8",
                                        "shape" => "bd", "dims" => Dict("d" => -1))]
    mc = ReactantServer.parse_manifest(withclient)
    @test mc.client_inputs[1].shape[1] == ReactantServer.Dim(ReactantServer.BATCH)
    @test mc.client_inputs[1].shape[2] == ReactantServer.Dim(ReactantServer.VARIABLE)
    # client_inputs without model.jl is rejected
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(mc, "/models/resnet", false)
    @test ReactantServer.validate_manifest(mc, "/models/resnet", true) === mc   # allowed with model.jl

    # more than one batch marker in a shape
    twoN = copy(good)
    twoN["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                      "shape" => "nn", "dims" => Dict())]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(twoN)

    bothNB = copy(good)
    bothNB["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                       "shape" => "nb", "dims" => Dict())]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(bothNB)

    # inputs disagree on batch-axis position
    disagree = Dict{String,Any}(
        "format_version" => "2.0", "name" => "m",
        "executable_inputs" => [Dict("name" => "x", "dtype" => "f32",
                                     "shape" => "nc", "dims" => Dict("c" => 3)),
                                Dict("name" => "y", "dtype" => "f32",
                                     "shape" => "cn", "dims" => Dict("c" => 3))],
        "executable_outputs" => [Dict("name" => "z", "dtype" => "f32",
                                      "shape" => "n", "dims" => Dict())],
        "batching" => Dict("compiled_batch_sizes" => [1]),
    )
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(disagree)

    # missing dims entry for a non-batch letter
    miss = copy(good)
    miss["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                      "shape" => "ck", "dims" => Dict("c" => 3))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(miss)

    # orphan key in dims
    orphan = copy(good)
    orphan["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                        "shape" => "c", "dims" => Dict("c" => 3, "k" => 7))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(orphan)

    # reserved letter 'n' in dims is rejected
    resn = copy(good)
    resn["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                      "shape" => "nc", "dims" => Dict("n" => 4, "c" => 3))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(resn)

    # duplicate non-batch letter in shape
    dup = copy(good)
    dup["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                     "shape" => "cc", "dims" => Dict("c" => 3))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(dup)

    # non-letter character in shape
    nonletter = copy(good)
    nonletter["executable_inputs"] = [Dict("name" => "x", "dtype" => "f32",
                                          "shape" => "c1", "dims" => Dict("c" => 3))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(nonletter)

    # unsupported format version
    badver = copy(good); badver["format_version"] = "1.0"
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(ReactantServer.parse_manifest(badver), "/models/resnet", false)

    # unknown dtype
    baddt = copy(good)
    baddt["executable_inputs"] = [Dict("name" => "x", "dtype" => "float",
                                       "shape" => "c", "dims" => Dict("c" => 4))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(baddt)
end

@testset "manifest rejects FP8 client-facing dtypes" begin
    # An FP8 executable output with no model.jl becomes the client-facing output, which cannot
    # be advertised over KServe, so load must fail rather than crashing when a response is built.
    f8 = Dict{String,Any}(
        "format_version" => "2.0", "name" => "fp8model",
        "executable_inputs" => [Dict("name" => "x", "dtype" => "f32",
                                     "shape" => "c", "dims" => Dict("c" => 4))],
        "executable_outputs" => [Dict("name" => "y", "dtype" => "f8_e4m3",
                                      "shape" => "c", "dims" => Dict("c" => 4))],
        "batching" => Dict("compiled_batch_sizes" => [1]),
    )
    mf8 = ReactantServer.parse_manifest(f8)
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(mf8, "/models/fp8model", false)

    # FP8 internally is fine when model.jl maps the client-facing output to a wire dtype.
    okclient = copy(f8)
    okclient["client_outputs"] = [Dict("name" => "y", "dtype" => "f32",
                                       "shape" => "c", "dims" => Dict("c" => 4))]
    mok = ReactantServer.parse_manifest(okclient)
    @test ReactantServer.validate_manifest(mok, "/models/fp8model", true) === mok

    # FP8 declared directly as a client output is rejected.
    badclient = copy(f8)
    badclient["executable_outputs"] = [Dict("name" => "y", "dtype" => "f32",
                                            "shape" => "c", "dims" => Dict("c" => 4))]
    badclient["client_outputs"] = [Dict("name" => "y", "dtype" => "f8_e5m2",
                                        "shape" => "c", "dims" => Dict("c" => 4))]
    mbad = ReactantServer.parse_manifest(badclient)
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(mbad, "/models/fp8model", true)
end

@testset "input_shapes variants" begin
    # A detector compiled for several aspect ratios: w,h are variable (-1), enumerated by
    # input_shapes. The variant key is the variable-axis sizes in (input, axis) order, so for
    # shape "whn" with w at axis 1 and h at axis 2 the key is [w, h].
    base = Dict{String,Any}(
        "format_version" => "2.0",
        "name" => "detector",
        "executable_inputs" => [Dict("name" => "INPUT__0", "dtype" => "f32",
                                     "shape" => "whn", "dims" => Dict("w" => -1, "h" => -1))],
        "executable_outputs" => [Dict("name" => "feat", "dtype" => "f32",
                                      "shape" => "whcn", "dims" => Dict("w" => -1, "h" => -1, "c" => 256))],
        "batching" => Dict("compiled_batch_sizes" => [1]),
        "input_shapes" => [Dict("w" => 1024, "h" => 1024),
                           Dict("w" => 1448, "h" => 720),
                           Dict("w" => 720, "h" => 1448)],
    )
    m = ReactantServer.parse_manifest(base)
    @test m.input_shapes == [[1024, 1024], [1448, 720], [720, 1448]]
    @test ReactantServer.validate_manifest(m, "/models/detector", false) === m
    @test m.executable_inputs[1].shape[1] == ReactantServer.Dim(ReactantServer.VARIABLE)
    @test m.executable_inputs[1].shape[3] == ReactantServer.Dim(ReactantServer.BATCH)

    # Absent input_shapes => single fixed shape (empty variant list).
    nofix = copy(base)
    delete!(nofix, "input_shapes")
    nofix["executable_inputs"] = [Dict("name" => "INPUT__0", "dtype" => "f32",
                                       "shape" => "whn", "dims" => Dict("w" => 512, "h" => 512))]
    nofix["executable_outputs"] = [Dict("name" => "feat", "dtype" => "f32",
                                        "shape" => "whcn", "dims" => Dict("w" => 128, "h" => 128, "c" => 256))]
    mno = ReactantServer.parse_manifest(nofix)
    @test isempty(mno.input_shapes)
    @test ReactantServer.validate_manifest(mno, "/models/detector", false) === mno

    # A variable executable-input axis with no input_shapes is rejected at validation.
    novar = copy(base)
    delete!(novar, "input_shapes")
    @test_throws ReactantServer.ManifestError ReactantServer.validate_manifest(
        ReactantServer.parse_manifest(novar), "/models/detector", false)

    # A variant missing a size for a variable axis is rejected at parse time.
    missing_axis = copy(base)
    missing_axis["input_shapes"] = [Dict("w" => 1024)]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(missing_axis)

    # A variant naming a non-variable axis is rejected.
    extra_axis = copy(base)
    extra_axis["input_shapes"] = [Dict("w" => 1024, "h" => 1024, "c" => 3)]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(extra_axis)

    # input_shapes with no variable axis present is rejected.
    fixed_in = copy(base)
    fixed_in["executable_inputs"] = [Dict("name" => "INPUT__0", "dtype" => "f32",
                                          "shape" => "whn", "dims" => Dict("w" => 1024, "h" => 1024))]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(fixed_in)

    # Duplicate variants are rejected.
    dup = copy(base)
    dup["input_shapes"] = [Dict("w" => 1024, "h" => 1024), Dict("w" => 1024, "h" => 1024)]
    @test_throws ReactantServer.ManifestError ReactantServer.parse_manifest(dup)
end
