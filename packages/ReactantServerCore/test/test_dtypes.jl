@testset "dtypes" begin
    @test ReactantServer.dtype_from_token("f32") == ReactantServer.F32
    @test ReactantServer.dtype_from_token("bf16") == ReactantServer.BF16
    @test ReactantServer.julia_type(ReactantServer.F32) === Float32
    @test ReactantServer.julia_type(ReactantServer.I64) === Int64
    @test ReactantServer.julia_type(ReactantServer.BOOL) === Bool
    @test ReactantServer.dtype_of(Float32) == ReactantServer.F32
    @test ReactantServer.dtype_token(ReactantServer.U8) == "u8"
    @test ReactantServer.dtype_size(ReactantServer.F32) == 4
    @test ReactantServer.dtype_size(ReactantServer.F64) == 8

    @test ReactantServer.kserve_string(ReactantServer.F32) == "FP32"
    @test ReactantServer.kserve_string(ReactantServer.I64) == "INT64"
    @test ReactantServer.kserve_string(ReactantServer.BF16) == "BF16"
    @test ReactantServer.dtype_from_kserve("FP32") == ReactantServer.F32
    @test ReactantServer.dtype_from_kserve("UINT8") == ReactantServer.U8

    @test_throws ArgumentError ReactantServer.dtype_from_token("float32")
    @test_throws ArgumentError ReactantServer.dtype_from_kserve("BYTES")
    # FP8 has no KServe mapping by design.
    @test_throws ArgumentError ReactantServer.kserve_string(ReactantServer.F8E5M2)
end
