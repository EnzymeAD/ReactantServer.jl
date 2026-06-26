using Test
using ReactantServerClient
import ReactantServerCore
using ReactantServerCore: inference
using ProtoBuf: OneOf

# Minimal IO for exercising the driver helpers without a server. `specs` drives output_specs.
struct _TestIO <: AbstractInferenceIO
    n::Int
    ipb::Int
    specs::Vector{OutputSpec}
end
Base.length(io::_TestIO) = io.n
ReactantServerClient.item_input_bytes(io::_TestIO) = io.ipb
ReactantServerClient.output_specs(io::_TestIO) = io.specs

# Build an InferParameter the way the wire carries shared-memory pointers.
_strp(s) = inference.InferParameter(parameter_choice = OneOf(:string_param, String(s)))
_intp(i) = inference.InferParameter(parameter_choice = OneOf(:int64_param, Int64(i)))

# ---- IOs for the validate_io dry run. Model: 4 Float32 in, 4 Float32 out, one item per chunk. ----

const _VSPEC = ModelIOSpec(
    Dict("INPUT__0" => TensorMeta("INPUT__0", "FP32", [4, -1])),   # Julia col-major: feature, batch last
    Dict("OUTPUT__0" => TensorMeta("OUTPUT__0", "FP32", [4, -1])),
    ["INPUT__0"], ["OUTPUT__0"],
)

# Correct IO.
struct GoodIO <: AbstractInferenceIO
    data::Vector{Vector{Float32}}
    results::Vector{Vector{Float32}}
end
GoodIO(n) = GoodIO([Float32[i, i, i, i] for i in 1:n], [Float32[] for _ in 1:n])
Base.length(io::GoodIO) = length(io.data)
ReactantServerClient.item_input_bytes(::GoodIO) = 4 * sizeof(Float32)
function ReactantServerClient.infer_encode_chunk!(io::GoodIO, r, slot)
    n = length(r)
    sub = subslot(slot, n * 4 * sizeof(Float32))
    v = pool_view(sub, Float32, n * 4)
    k = 1
    for i in r
        v[k:k+3] .= io.data[i]
        k += 4
    end
    return [InferInput("INPUT__0", sub, [4, n], Float32)]   # Julia col-major: feature, batch last
end
function ReactantServerClient.infer_decode_chunk!(io::GoodIO, r, response)
    out = InferOutput("OUTPUT__0", response, Float32)   # (4, n) col-major
    for (j, i) in enumerate(r)
        io.results[i] = collect(out[:, j])
    end
    return nothing
end

# Encodes more bytes than item_input_bytes declares (overflows the slot).
struct OverflowIO <: AbstractInferenceIO end
Base.length(::OverflowIO) = 1
ReactantServerClient.item_input_bytes(::OverflowIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::OverflowIO, r, slot) =
    [InferInput("INPUT__0", subslot(slot, length(r) * 8 * sizeof(Float32)), [8, length(r)], Float32)]
ReactantServerClient.infer_decode_chunk!(::OverflowIO, r, response) = nothing

# Emits a descriptor with a name the model does not have.
struct WrongNameIO <: AbstractInferenceIO end
Base.length(::WrongNameIO) = 1
ReactantServerClient.item_input_bytes(::WrongNameIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::WrongNameIO, r, slot) =
    [InferInput("WRONG", subslot(slot, length(r) * 4 * sizeof(Float32)), [4, length(r)], Float32)]
ReactantServerClient.infer_decode_chunk!(::WrongNameIO, r, response) = nothing

# Reads past the model's actual output shape in the decode step.
struct BadDecodeIO <: AbstractInferenceIO end
Base.length(::BadDecodeIO) = 1
ReactantServerClient.item_input_bytes(::BadDecodeIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::BadDecodeIO, r, slot) =
    [InferInput("INPUT__0", subslot(slot, length(r) * 4 * sizeof(Float32)), [4, length(r)], Float32)]
function ReactantServerClient.infer_decode_chunk!(::BadDecodeIO, r, response)
    out = InferOutput("OUTPUT__0", response, Float32)   # (4, 1)
    return out[99, 1]                                   # BoundsError
end

# Same overread but under @inbounds: only safe to exercise when bounds checking is forced on
# (--check-bounds=yes), which is exactly what we are verifying neutralizes @inbounds.
struct InboundsBadDecodeIO <: AbstractInferenceIO end
Base.length(::InboundsBadDecodeIO) = 1
ReactantServerClient.item_input_bytes(::InboundsBadDecodeIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::InboundsBadDecodeIO, r, slot) =
    [InferInput("INPUT__0", subslot(slot, length(r) * 4 * sizeof(Float32)), [4, length(r)], Float32)]
function ReactantServerClient.infer_decode_chunk!(::InboundsBadDecodeIO, r, response)
    out = InferOutput("OUTPUT__0", response, Float32)   # (4, 1)
    @inbounds return out[99, 1]
end

# Same overread but using @infer_inbounds: validate_io enters with_bounds_checks, so the checked
# branch runs and this is caught in auto mode too (only --check-bounds=no would elide it).
struct MacroInboundsBadDecodeIO <: AbstractInferenceIO end
Base.length(::MacroInboundsBadDecodeIO) = 1
ReactantServerClient.item_input_bytes(::MacroInboundsBadDecodeIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::MacroInboundsBadDecodeIO, r, slot) =
    [InferInput("INPUT__0", subslot(slot, length(r) * 4 * sizeof(Float32)), [4, length(r)], Float32)]
function ReactantServerClient.infer_decode_chunk!(::MacroInboundsBadDecodeIO, r, response)
    out = InferOutput("OUTPUT__0", response, Float32)   # (4, 1)
    @infer_inbounds out[99, 1]
    return nothing
end

# Records whether forced bounds checking was active when decode ran.
mutable struct ContextProbeIO <: AbstractInferenceIO
    seen::Bool
end
Base.length(::ContextProbeIO) = 1
ReactantServerClient.item_input_bytes(::ContextProbeIO) = 4 * sizeof(Float32)
ReactantServerClient.infer_encode_chunk!(::ContextProbeIO, r, slot) =
    [InferInput("INPUT__0", subslot(slot, length(r) * 4 * sizeof(Float32)), [4, length(r)], Float32)]
function ReactantServerClient.infer_decode_chunk!(io::ContextProbeIO, r, response)
    io.seen = ReactantServerClient._FORCE_BOUNDS[]
    return nothing
end

@testset "ReactantServerClient" begin
    @testset "no Reactant dependency (headline goal)" begin
        # The client must be usable without the heavy Reactant/XLA stack. Assert Reactant is not
        # a dependency of the client package nor of its Reactant-free Core dependency.
        @test Base.identify_package(ReactantServerClient, "Reactant") === nothing
        @test Base.identify_package(ReactantServerCore, "Reactant") === nothing
    end

    @testset "InferenceBufferPool wraps the Core allocator" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)
        @test !ReactantServerClient.is_shm_backed(p)
        @test ReactantServerClient.slot_bytes(p) == 1024
        @test ReactantServerClient.n_slots(p) == 4
        @test sizeof(p) == 4096

        s = acquire_slot!(p)
        @test s.capacity == 1024
        view = pool_view(s, Float32, 256)
        view[1] = 3.5f0
        @test view[1] == 3.5f0
        release_slot!(s)
        # all slots returned: can drain all four again
        slots = [acquire_slot!(p) for _ in 1:4]
        @test length(unique(x.index for x in slots)) == 4
        foreach(release_slot!, slots)
    end

    @testset "scratch + pool_view staging (inline)" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)
        m = KServeModel("grpc://h:1", "x"; max_batch_size = 100000)
        s = acquire_slot!(p, 2)                                   # fresh slot: cursor at 0
        inputs = scratch(s, ["INPUT__0" => ((4, 3), Float32), "MASK" => ((3,), Int32)])

        # One contiguous block, carved into disjoint descriptors with the right shape/dtype/offset.
        @test inputs isa Vector{PoolInferInput}
        @test inputs[1].subslot.offset == s.offset
        @test inputs[2].subslot.offset == s.offset + 4 * 3 * sizeof(Float32)
        @test inputs[1].shape == [4, 3] && inputs[1].dtype == Float32
        @test inputs[2].shape == [3] && inputs[2].dtype == Int32

        feats, mask = pool_view(inputs...)            # splat the descriptors, destructure the views
        @test feats isa Matrix{Float32} && mask isa Vector{Int32}
        feats .= reshape(collect(Float32, 1:12), 4, 3)
        mask  .= Int32[7, 8, 9]

        # Inline materialization builds the same wire tensors the manual InferInput path would.
        wire = ReactantServerClient._materialize_inputs(inputs, m, p)
        @test wire[1].name == "INPUT__0" && wire[1].datatype == "FP32"
        @test wire[1].shape == [3, 4]                             # reversed col-major -> row-major
        @test wire[1].contents.fp32_contents == collect(Float32, 1:12)
        @test wire[2].datatype == "INT32"
        @test wire[2].contents.int_contents == Int32[7, 8, 9]
        release_slot!(s)
    end

    @testset "scratch staging (shm) references the pool region by offset" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = true, name = "rsc_scratch_pool")
        try
            m = KServeModel("grpc://h:1", "x"; max_batch_size = 100000)
            s = acquire_slot!(p, 2)
            inputs = scratch(s, ["A" => ((4,), Float32), "B" => ((2,), Int32)])
            wire = ReactantServerClient._materialize_inputs(inputs, m, p)
            pa = wire[1].parameters
            @test pa["shared_memory_region"].parameter_choice[] == ReactantServerClient.pool_name(p)
            @test pa["shared_memory_offset"].parameter_choice[] == s.offset
            @test pa["shared_memory_byte_size"].parameter_choice[] == 4 * sizeof(Float32)
            pb = wire[2].parameters
            @test pb["shared_memory_offset"].parameter_choice[] == s.offset + 4 * sizeof(Float32)
            @test pb["shared_memory_byte_size"].parameter_choice[] == 2 * sizeof(Int32)
            release_slot!(s)
        finally
            rm(p.pool.backing)
        end
    end

    @testset "scratch returns a homogeneous descriptor vector (no element promotion)" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)
        s = acquire_slot!(p, 2)
        # Mixed dtypes stay one concrete type: returning these never promotes/copies a buffer the
        # way a literal vector of differently-typed arrays would.
        inputs = scratch(s, ["F" => ((4,), Float32), "I" => ((2,), Int32)])
        @test inputs isa Vector{PoolInferInput}
        @test pool_view(inputs[1]) isa Vector{Float32}
        @test pool_view(inputs[2]) isa Vector{Int32}
        # Scalar form returns a single descriptor.
        one = scratch(s, "X", (3,), Float64)
        @test one isa PoolInferInput && one.shape == [3] && one.dtype == Float64
        release_slot!(s)
    end

    @testset "InferInput builds wire tensors" begin
        x = reshape(collect(Float32, 1:6), 2, 3)        # Julia (W=2, H=3)
        t = InferInput("INPUT__0", x)
        @test t.datatype == "FP32"
        @test t.shape == [3, 2]                          # reversed to network row-major
        @test t.contents.fp32_contents == vec(x)

        b = InferInput("MASK", [1, 4], collect(UInt8, 1:4))
        @test b.datatype == "UINT8"
        @test b.contents.uint_contents == collect(UInt8, 1:4)
    end

    @testset "InferOutput decodes a response" begin
        # Network sends row-major (N, ..., H, W); InferOutput reshapes directly to Julia order.
        data = collect(Float32, 1:6)
        raw = reinterpret(UInt8, data) |> collect
        out = inference.var"ModelInferResponse.InferOutputTensor"(; name = "OUTPUT__0",
            datatype = "FP32", shape = Int64[3, 2])
        resp = inference.ModelInferResponse(; model_name = "m", id = "1",
            outputs = [out], raw_output_contents = [raw])
        got = InferOutput("OUTPUT__0", resp, Float32)
        @test size(got) == (2, 3)                        # reverse of wire shape (3, 2)
        @test vec(got) == data
    end

    @testset "output_specs default empty -> inline fallback" begin
        io = _TestIO(4, 16, OutputSpec[])
        @test output_specs(io) == OutputSpec[]
        @test item_output_bytes(io) == 0

        # No declaration means no requested outputs and no output subslots: the request is
        # byte-for-byte what it was before this feature (server returns everything inline).
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)
        s = acquire_slot!(p)
        requested, subs =
            ReactantServerClient._build_requested_outputs(output_specs(io), s, 1:2, p)
        @test isempty(requested)
        @test isempty(subs)
        release_slot!(s)
    end

    @testset "item_output_bytes sums declared outputs" begin
        io = _TestIO(10, 16, [OutputSpec("OUTPUT__0", Float32, [4]),     # 16 bytes/item
                              OutputSpec("OUTPUT__1", UInt8, 2, 3)])     # 6 bytes/item
        @test item_output_bytes(io) == 16 + 6
        # Scalar (no per-item dims) counts as one element.
        @test item_output_bytes(_TestIO(1, 1, [OutputSpec("S", Float64, Int[])])) == 8
    end

    @testset "inline pool: name-only outputs, output bytes not charged to slot" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)     # 1024-byte slots
        io = _TestIO(10, 16, [OutputSpec("OUTPUT__0", Float32, [4])])
        m = KServeModel("grpc://h:1", "x"; max_batch_size = 100000)
        # Inline outputs travel in the response, not the slot, so chunk size ignores them.
        @test ReactantServerClient._chunk_size(io, m, p) == 1024 ÷ 16

        s = acquire_slot!(p)
        requested, subs =
            ReactantServerClient._build_requested_outputs(output_specs(io), s, 1:3, p)
        @test length(requested) == 1
        @test requested[1].name == "OUTPUT__0"
        @test isempty(requested[1].parameters)                         # name only, no SHM params
        @test isempty(subs)
        release_slot!(s)
    end

    @testset "shm pool: output requests carry params placed after inputs" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = true, name = "rsc_test_pool")
        try
            @test ReactantServerClient.is_shm_backed(p)
            s = acquire_slot!(p)
            subslot(s, 64)                                             # simulate inputs at the front
            specs = [OutputSpec("OUTPUT__0", Float32, [4])]            # 16 bytes/item
            requested, subs =
                ReactantServerClient._build_requested_outputs(specs, s, 1:2, p)
            @test length(requested) == 1
            prm = requested[1].parameters
            @test prm["shared_memory_region"].parameter_choice[] == ReactantServerClient.pool_name(p)
            @test prm["shared_memory_byte_size"].parameter_choice[] == 16 * 2   # per-item * length(r)
            @test prm["shared_memory_offset"].parameter_choice[] == s.offset + 64
            @test subs["OUTPUT__0"].offset == s.offset + 64
            release_slot!(s)
        finally
            rm(p.pool.backing)
        end
    end

    @testset "rehydrate reads shm outputs back into raw_output_contents" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = true, name = "rsc_test_pool2")
        try
            s = acquire_slot!(p)
            data = collect(Float32, 1:6)
            sub = subslot(s, sizeof(data))
            pool_view(sub, Float32, length(data)) .= data             # server would write here

            out = inference.var"ModelInferResponse.InferOutputTensor"(; name = "OUTPUT__0",
                datatype = "FP32", shape = Int64[3, 2],
                parameters = Dict("shared_memory_region" => _strp(ReactantServerClient.pool_name(p)),
                                  "shared_memory_offset" => _intp(sub.offset),
                                  "shared_memory_byte_size" => _intp(sizeof(data))))
            resp = inference.ModelInferResponse(; model_name = "m", id = "1",
                outputs = [out], raw_output_contents = Vector{UInt8}[])

            norm = ReactantServerClient._rehydrate_response(resp, Dict("OUTPUT__0" => sub))
            @test length(norm.raw_output_contents) == 1
            @test !haskey(norm.outputs[1].parameters, "shared_memory_region")   # params stripped
            got = InferOutput("OUTPUT__0", norm, Float32)
            @test size(got) == (2, 3)
            @test vec(got) == data
            release_slot!(s)
        finally
            rm(p.pool.backing)
        end
    end

    @testset "rehydrate aligns mixed shm + inline outputs" begin
        # The server's encoder only appends a raw entry per inline output, so raw_output_contents
        # is positionally compressed. Rehydration must realign it to the full outputs list.
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = true, name = "rsc_test_pool3")
        try
            s = acquire_slot!(p)
            shm_data = collect(Float32, 10:13)
            sub = subslot(s, sizeof(shm_data))
            pool_view(sub, Float32, length(shm_data)) .= shm_data

            inline_data = collect(Float32, 1:4)
            inline_raw = collect(reinterpret(UInt8, inline_data))
            inline_out = inference.var"ModelInferResponse.InferOutputTensor"(;
                name = "INLINE", datatype = "FP32", shape = Int64[4])
            shm_out = inference.var"ModelInferResponse.InferOutputTensor"(; name = "SHM",
                datatype = "FP32", shape = Int64[4],
                parameters = Dict("shared_memory_region" => _strp(ReactantServerClient.pool_name(p)),
                                  "shared_memory_offset" => _intp(sub.offset),
                                  "shared_memory_byte_size" => _intp(sizeof(shm_data))))
            # outputs has two entries; raw holds only the inline one (compressed).
            resp = inference.ModelInferResponse(; model_name = "m", id = "1",
                outputs = [inline_out, shm_out], raw_output_contents = [inline_raw])

            norm = ReactantServerClient._rehydrate_response(resp, Dict("SHM" => sub))
            @test length(norm.raw_output_contents) == 2
            @test vec(InferOutput("INLINE", norm, Float32)) == inline_data
            @test vec(InferOutput("SHM", norm, Float32)) == shm_data
            release_slot!(s)
        finally
            rm(p.pool.backing)
        end
    end

    @testset "_chunk_geometry spans slots for oversized items" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)    # 1024-byte slots
        m = KServeModel("grpc://h:1", "x"; max_batch_size = 100000)

        # Items within one slot: span 1, current behavior.
        chunk, span = ReactantServerClient._chunk_geometry(_TestIO(10, 16, OutputSpec[]), m, p)
        @test (chunk, span) == (1024 ÷ 16, 1)

        # An item larger than one slot spans contiguous slots instead of erroring.
        chunk, span = ReactantServerClient._chunk_geometry(_TestIO(10, 2000, OutputSpec[]), m, p)
        @test span == 2
        @test chunk == 1                                               # 2048 ÷ 2000

        # The span is minimal: one byte past a slot boundary takes the next slot, no more.
        chunk, span = ReactantServerClient._chunk_geometry(_TestIO(10, 1025, OutputSpec[]), m, p)
        @test (chunk, span) == (1, 2)
        chunk, span = ReactantServerClient._chunk_geometry(_TestIO(10, 2049, OutputSpec[]), m, p)
        @test (chunk, span) == (1, 3)

        # max_batch_size still caps the chunk.
        m1 = KServeModel("grpc://h:1", "x"; max_batch_size = 3)
        chunk, span = ReactantServerClient._chunk_geometry(_TestIO(10, 16, OutputSpec[]), m1, p)
        @test (chunk, span) == (3, 1)

        # Inline pools do not charge declared outputs to the slot; SHM pools would.
        io = _TestIO(10, 1000, [OutputSpec("OUTPUT__0", Float32, [32])])   # +128 output bytes
        chunk, span = ReactantServerClient._chunk_geometry(io, m, p)
        @test span == 1                                                # inline: 1000 <= 1024
    end

    @testset "_chunk_geometry errors when an item cannot fit the pool" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)    # 1024-byte slots
        m = KServeModel("grpc://h:1", "x"; max_batch_size = 1)
        io = _TestIO(10, 5000, OutputSpec[])                           # input alone exceeds pool
        @test_throws ErrorException ReactantServerClient._chunk_geometry(io, m, p)
        @test_throws ErrorException ReactantServerClient._chunk_size(io, m, p)
    end

    @testset "acquire_slot! forwards span through InferenceBufferPool" begin
        p = InferenceBufferPool(4096; n_slots = 4, use_shm = false)
        # A span larger than the pool is detected as unsatisfiable instead of blocking forever.
        @test_throws ArgumentError acquire_slot!(p, 5)
        s = acquire_slot!(p, 3)
        @test s.span == 3
        @test s.capacity == 3 * ReactantServerClient.slot_bytes(p)
        # Subslot carving works across the physical slot boundary.
        sub = subslot(s, 2000)
        v = pool_view(sub, UInt8, 2000)
        v .= 0x5a
        @test all(==(0x5a), pool_view(sub, UInt8, 2000))
        release_slot!(s)
        slots = [acquire_slot!(p) for _ in 1:4]                        # whole run was freed
        foreach(release_slot!, slots)
    end

    @testset "manifest_io_spec loads a manifest into a ModelIOSpec" begin
        # Hermetic: write a tiny manifest and load it (no bundle directory needed).
        dir = mktempdir()
        path = joinpath(dir, "manifest.yaml")
        write(path, """
        format_version: "2.0"
        name: "tinymodel"
        executable_inputs:
          - name: "INPUT__0"
            dtype: "f32"
            shape: "wcn"
            dims:
              w: 4
              c: 3
        executable_outputs:
          - name: "OUTPUT__0"
            dtype: "f32"
            shape: "wn"
            dims:
              w: 4
        batching:
          compiled_batch_sizes: [1]
        """)
        spec = manifest_io_spec(path)
        # The spec reports Julia col-major shapes, reading in the same axis order as the manifest
        # einsum letters ("wcn" -> (w, c, n)); the batch letter n is -1.
        @test spec.inputs["INPUT__0"].datatype == "FP32"
        @test spec.inputs["INPUT__0"].shape == [4, 3, -1]
        @test spec.outputs["OUTPUT__0"].shape == [4, -1]
        @test spec.input_order == ["INPUT__0"]
        @test spec.output_order == ["OUTPUT__0"]
    end

    @testset "validate_io dry-runs the IO against the spec" begin
        io = GoodIO(3)
        @test validate_io(_VSPEC, io) === nothing      # passes, no exception
        @test io.results[1] == zeros(Float32, 4)        # decode ran against the synthetic zeroed output

        # A decode that indexes past the model's output shape is caught.
        @test_throws ErrorException validate_io(_VSPEC, BadDecodeIO())
        # An encode that writes more than item_input_bytes declares overflows and is caught.
        @test_throws ErrorException validate_io(_VSPEC, OverflowIO())
        # A descriptor naming an input the model lacks is caught.
        @test_throws ErrorException validate_io(_VSPEC, WrongNameIO())

        # Empty io: warns and returns without error.
        @test validate_io(_VSPEC, GoodIO(0)) === nothing

        # Pkg.test runs with --check-bounds=yes (check_bounds == 1), which neutralizes @inbounds,
        # so a bare-@inbounds overread in the decode is still caught. Guard the overread so the test
        # is only exercised when bounds checking is actually forced (otherwise it is UB).
        if Base.JLOptions().check_bounds == 1
            @test_throws ErrorException validate_io(_VSPEC, InboundsBadDecodeIO())
        end

        # An @infer_inbounds overread is caught via the with_bounds_checks context, safely in auto
        # and yes modes (the executed branch bounds-checks); only --check-bounds=no would elide it.
        if Base.JLOptions().check_bounds != 2
            @test_throws ErrorException validate_io(_VSPEC, MacroInboundsBadDecodeIO())
        end

        # validate_io enters the forced-bounds context around the user's decode.
        probe = ContextProbeIO(false)
        validate_io(_VSPEC, probe)
        @test probe.seen
    end

    @testset "with_bounds_checks / @infer_inbounds" begin
        # The scoped flag is off by default and on inside the context, restored on exit.
        @test ReactantServerClient._FORCE_BOUNDS[] == false
        @test with_bounds_checks(() -> ReactantServerClient._FORCE_BOUNDS[]) == true
        @test ReactantServerClient._FORCE_BOUNDS[] == false

        # A valid @infer_inbounds access returns the value in both contexts.
        a = Float32[10, 20, 30, 40]
        @test (@infer_inbounds a[2]) == 20.0f0
        @test with_bounds_checks(() -> @infer_inbounds a[3]) == 30.0f0
    end

    @testset "@infer_inbounds elides in auto mode, checks in context (subprocess)" begin
        # Pkg.test forces --check-bounds=yes, so the elision can only be observed in a fresh process
        # under --check-bounds=auto. A probe array records whether its @boundscheck ran (no OOB).
        proj = dirname(dirname(pathof(ReactantServerClient)))
        # The accesses go through functions so getindex inlines and @inbounds can actually elide.
        code = raw"""
        using ReactantServerClient
        mutable struct ProbeArr <: AbstractVector{Float32}
            checked::Bool
        end
        Base.size(::ProbeArr) = (4,)
        Base.IndexStyle(::Type{ProbeArr}) = IndexLinear()
        Base.@propagate_inbounds function Base.getindex(a::ProbeArr, i::Int)
            @boundscheck (a.checked = true)
            return 0.0f0
        end
        f_elide(a::ProbeArr) = @infer_inbounds a[1]
        f_ctx(a::ProbeArr) = with_bounds_checks(() -> @infer_inbounds a[1])
        a = ProbeArr(false); f_elide(a)
        b = ProbeArr(false); f_ctx(b)
        exit((!a.checked && b.checked) ? 0 : 1)   # elided normally, checked in context
        """
        julia = Base.julia_cmd()[1]
        p = run(ignorestatus(`$julia --check-bounds=auto --project=$proj -e $code`))
        @test p.exitcode == 0
    end

    @testset "validation catches dtype and shape mismatches with a reversal hint" begin
        # Wrong declared dtype. Model shape is Julia col-major (feature 4, batch last).
        spec = ModelIOSpec(Dict{String,TensorMeta}(),
            Dict("O" => TensorMeta("O", "FP32", [4, -1])), String[], ["O"])
        @test_throws ErrorException ReactantServerClient._validate_output_specs(
            spec, [OutputSpec("O", Float64, [4])])

        # 2-D per-item output: declaring the axis order reversed is rejected with a hint, while the
        # correct column-major declaration passes. Model col-major shape (4, 3, batch).
        spec2 = ModelIOSpec(Dict{String,TensorMeta}(),
            Dict("O" => TensorMeta("O", "FP32", [4, 3, -1])), String[], ["O"])
        @test ReactantServerClient._validate_output_specs(spec2, [OutputSpec("O", Float32, [4, 3])]) === nothing
        err = try
            ReactantServerClient._validate_output_specs(spec2, [OutputSpec("O", Float32, [3, 4])])
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("reversed", sprint(showerror, err))
    end

    @testset "KServeModel URL parsing" begin
        m = KServeModel("grpc://host.example:8001", "mymodel"; max_batch_size = 8)
        @test m.host == "host.example"
        @test m.port == 0x1f41
        @test m.secure == false
        @test model_name(m) == "mymodel"
        ms = KServeModel("grpcs://h:9000", "x")
        @test ms.secure == true
    end
end

@testset "KServeModel gRPC message-size default + env fallback" begin
    # Default is 512 MiB in each direction.
    m = KServeModel("grpc://h:1", "x")
    @test m.max_send_message_length == 512 * 1024 * 1024
    @test m.max_receive_message_length == 512 * 1024 * 1024

    # An explicit kwarg wins.
    m2 = KServeModel("grpc://h:1", "x"; max_send_message_length = 123, max_receive_message_length = 456)
    @test m2.max_send_message_length == 123
    @test m2.max_receive_message_length == 456

    # The env var supplies the default when the kwarg is omitted (per direction); a kwarg still wins.
    withenv("REACTANT_CLIENT_GRPC_MAX_RECV_MSG_BYTES" => "4096",
            "REACTANT_CLIENT_GRPC_MAX_SEND_MSG_BYTES" => "8192") do
        me = KServeModel("grpc://h:1", "x")
        @test me.max_receive_message_length == 4096
        @test me.max_send_message_length == 8192
        mo = KServeModel("grpc://h:1", "x"; max_receive_message_length = 7)
        @test mo.max_receive_message_length == 7
        @test mo.max_send_message_length == 8192
    end
end

@testset "parse_grpc_url rejects malformed URLs with a clear error" begin
    @test_throws ErrorException ReactantServerClient.parse_grpc_url("not-a-url")
    @test_throws ErrorException ReactantServerClient.parse_grpc_url("ftp://h:1")
    @test_throws ErrorException ReactantServerClient.parse_grpc_url("grpc://h:not_a_port")
    @test_throws ErrorException ReactantServerClient.parse_grpc_url("h:p:q:r")
    err = try
        ReactantServerClient.parse_grpc_url("not-a-url")
        nothing
    catch e
        e
    end
    @test occursin("invalid gRPC URL", sprint(showerror, err))
    # plain host:port still parses, insecure
    h, p, sec = ReactantServerClient.parse_grpc_url("localhost:8080")
    @test h == "localhost" && p == 0x1f90 && sec == false
end

@testset "shared_memory mode: validation and :off routing" begin
    # Default is :auto; :on / :off accepted; anything else rejected at construction.
    @test KServeModel("grpc://h:1", "x").shared_memory == :auto
    @test KServeModel("grpc://h:1", "x"; shared_memory = :on).shared_memory == :on
    @test KServeModel("grpc://h:1", "x"; shared_memory = :off).shared_memory == :off
    @test_throws ArgumentError KServeModel("grpc://h:1", "x"; shared_memory = :sometimes)

    # :off never probes the server and always routes to a non-SHM pool. Small pools so the
    # test does not allocate the 256 MiB default.
    kserve_init(; pool_bytes = 4096, n_slots = 4)
    try
        m = KServeModel("grpc://127.0.0.1:1", "x"; shared_memory = :off)
        pool = ReactantServerClient._decide_pool!(m)
        @test !ReactantServerClient.is_shm_backed(pool)
    finally
        kserve_shutdown()
    end
end
