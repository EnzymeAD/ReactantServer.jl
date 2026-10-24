# WeightsFile (runtime/weights_file.jl): weights.safetensors read with pread. Tensors must come back
# exactly as SafeTensors.jl returns them, a file truncated after loading must be an error (it was
# SIGBUS through a memory mapping), and an atomically replaced file must keep reading the version
# that was opened.

const _BF16 = ReactantServer.BFloat16s.BFloat16

function _write_st(path, tensors::Dict{String, <:AbstractArray}; order = sort!(collect(keys(tensors))))
    SafeTensors.serialize(path, tensors, Dict("argument_order" => JSON3.write(order)))
    return path
end

@testset "WeightsFile reads exactly what SafeTensors.jl returns" begin
    mktempdir() do dir
        tensors = Dict{String, AbstractArray}(
            "v" => rand(Float32, 7),
            "m" => rand(Float32, 3, 5),
            "c" => rand(Float32, 2, 3, 4),
            "h" => rand(Float16, 4, 2),
            "b" => _BF16.(rand(Float32, 3, 3)),
            "i" => rand(Int8, 5, 2),
            "q" => rand(Bool, 6),
        )
        path = _write_st(joinpath(dir, "weights.safetensors"), tensors)
        ref = SafeTensors.deserialize(path; mmap = false)
        w, id = ReactantServer.open_weights_file(path)
        @test id == weights_file_id(path)
        @test Set(keys(w)) == Set(keys(tensors))
        @test ReactantServer.weight_order(w) == sort!(collect(keys(tensors)))      # argument_order metadata
        for name in keys(tensors)
            want = collect(ref[name])
            got = ReactantServer.read_weight(w, name)
            @test typeof(got) == typeof(want) && size(got) == size(want) && got == want
            @test ReactantServer.weight_spec(w, name) == (eltype(want), size(want))
            dest = similar(want)
            @test ReactantServer.read_weight!(dest, w, name) == want
        end
        @test ReactantServer.weights_nbytes(w, collect(keys(tensors))) ==
            sum(sizeof(collect(ref[n])) for n in keys(tensors))
        ReactantServer.close_weights!(w)
        ReactantServer.close_weights!(w)                                           # idempotent
        @test !isopen(w.io)
    end
end

@testset "a weights file truncated after loading is an error, not a crash" begin
    mktempdir() do dir
        path = _write_st(joinpath(dir, "weights.safetensors"), Dict("w" => rand(Float32, 1024)))
        w, _ = ReactantServer.open_weights_file(path)
        # What `cp new old` does first: truncate the destination in place.
        open(path, "w") do io
            write(io, UInt8[1, 2, 3])
        end
        @test_throws "ended early" ReactantServer.read_weight(w, "w")
        @test_throws "ended early" ReactantServer.read_weight!(Vector{Float32}(undef, 1024), w, "w")
        ReactantServer.close_weights!(w)
    end
end

@testset "an atomically replaced weights file keeps reading the opened version" begin
    mktempdir() do dir
        old = rand(Float32, 64)
        path = _write_st(joinpath(dir, "weights.safetensors"), Dict("w" => old))
        w, id = ReactantServer.open_weights_file(path)
        newpath = _write_st(joinpath(dir, "weights.new"), Dict("w" => rand(Float32, 64)))
        mv(newpath, path; force = true)
        @test weights_file_id(path) != id
        @test ReactantServer.read_weight(w, "w") == old
        ReactantServer.close_weights!(w)
    end
end

@testset "a malformed safetensors header is a BundleError" begin
    mktempdir() do dir
        p = joinpath(dir, "weights.safetensors")
        write(p, UInt8[1, 2, 3])
        @test_throws ReactantServer.BundleError ReactantServer.open_weights_file(p)
        write(p, vcat(reinterpret(UInt8, [UInt64(1 << 40)]), UInt8[0x7b, 0x7d]))   # absurd header length
        @test_throws ReactantServer.BundleError ReactantServer.open_weights_file(p)
        hdr = Vector{UInt8}("{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,64]}}")
        write(p, vcat(reinterpret(UInt8, [UInt64(length(hdr))]), hdr, zeros(UInt8, 16)))   # size mismatch
        @test_throws ReactantServer.BundleError ReactantServer.open_weights_file(p)
    end
end
