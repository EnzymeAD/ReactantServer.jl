# Generate a model repository of "heavy weights, trivial compute" bundles for the residency
# stress suite (see README.md beside this file).
#
#   julia --project=tools/residency_stress tools/residency_stress/gen_models.jl OUT_DIR [PROFILE]
#
# PROFILE is `standard` (default, about 7 GB of weights), `small` (about 1.2 GB, for smoke runs) or
# `tiny` (a few MB, for CPU runs of the driver itself). OUT_DIR receives:
#
#   OUT_DIR/models/<name>/          the bundles (the worker's model_repo)
#   OUT_DIR/alt/<name>/<tag>.safetensors   alternate weights for weights-only updates of <name>
#   OUT_DIR/catalog.toml            what every bundle and alternate is, with its expected outputs
#
# Every synthetic model computes, for each weight tensor k, an exact position-sensitive checksum
#
#     cs[k] = sum_i bits(w_k[i]) * (i * C1 + 2k + 1)   (mod 2^64, i = 0-based program-order index)
#
# where bits() reinterprets the element as an unsigned integer of its width, and returns
# y[k, b] = cs[k] + x[b] for the u64 nonce input x. Integer addition mod 2^64 is associative, so the
# result is exact whatever order the device reduces in: the client checks every response for
# equality with the checksums computed here. A model served with another model's weights, a stale
# version, a tensor in the wrong axis order, or a single corrupted element fails the check, and a
# response row that does not carry its own request's nonce exposes a mixed-up batch.
#
# The program reads every weight byte once per execution and does nothing else, so the work a
# request causes is dominated by getting the weights onto the device.

using ReactantServerExport: ReactantServerExport, IOSpec, write_bundle
using SafeTensors: SafeTensors
using BFloat16s: BFloat16
using Random: Xoshiro, rand!
using TOML: TOML

const C1 = 0x9e3779b97f4a7c15
c2(k::Integer) = UInt64(2k + 1)

ubits(::Type{Float32}) = UInt32
ubits(::Type{BFloat16}) = UInt16
mlir_elt(::Type{Float32}) = "f32"
mlir_elt(::Type{BFloat16}) = "bf16"
mlir_uelt(::Type{Float32}) = "ui32"
mlir_uelt(::Type{BFloat16}) = "ui16"

tensor_t(dims, t) = isempty(dims) ? "tensor<$t>" : "tensor<$(join(dims, "x"))x$t>"

"""
    host_checksum(A, k) -> UInt64

The checksum the program computes for weight tensor `k` (1-based) holding the Julia array `A`. The
program's row-major index over the reversed shape is the Julia linear index.
"""
function host_checksum(A::Array{T}, k::Integer) where {T}
    v = reinterpret(ubits(T), vec(A))
    acc = UInt64(0)
    ck = c2(k)
    @inbounds for i in eachindex(v)
        acc += UInt64(v[i]) * (UInt64(i - 1) * C1 + ck)
    end
    return acc
end

# The checksum section of the program: one exact u64 reduction per weight, concatenated to %cs.
function checksum_body(io::IO, wtypes, wshapes)
    println(io, "    %zero = stablehlo.constant dense<0> : tensor<ui64>")
    println(io, "    %c1 = stablehlo.constant dense<$(C1)> : tensor<ui64>")
    K = length(wtypes)
    for k in 1:K
        T = wtypes[k]
        pd = reverse(collect(wshapes[k]))
        N = prod(pd; init = 1)
        t, u = mlir_elt(T), mlir_uelt(T)
        n64 = tensor_t([N], "ui64")
        println(io, "    %r$k = stablehlo.reshape %w$k : ($(tensor_t(pd, t))) -> $(tensor_t([N], t))")
        println(io, "    %b$k = stablehlo.bitcast_convert %r$k : ($(tensor_t([N], t))) -> $(tensor_t([N], u))")
        println(io, "    %v$k = stablehlo.convert %b$k : ($(tensor_t([N], u))) -> $n64")
        println(io, "    %i$k = stablehlo.iota dim = 0 : $n64")
        println(io, "    %ca$k = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<ui64>) -> $n64")
        println(io, "    %cs$k = stablehlo.constant dense<$(c2(k))> : tensor<ui64>")
        println(io, "    %cb$k = stablehlo.broadcast_in_dim %cs$k, dims = [] : (tensor<ui64>) -> $n64")
        println(io, "    %m$k = stablehlo.multiply %i$k, %ca$k : $n64")
        println(io, "    %n$k = stablehlo.add %m$k, %cb$k : $n64")
        println(io, "    %p$k = stablehlo.multiply %v$k, %n$k : $n64")
        println(io, "    %s$k = stablehlo.reduce(%p$k init: %zero) applies stablehlo.add across dimensions = [0] : ($n64, tensor<ui64>) -> tensor<ui64>")
        println(io, "    %e$k = stablehlo.reshape %s$k : (tensor<ui64>) -> tensor<1xui64>")
    end
    operands = join(("%e$k" for k in 1:K), ", ")
    types = join(fill("tensor<1xui64>", K), ", ")
    println(io, "    %cs = stablehlo.concatenate $operands, dim = 0 : ($types) -> tensor<$(K)xui64>")
    return nothing
end

wargs(wtypes, wshapes) = join(
    ("%w$k: $(tensor_t(reverse(collect(wshapes[k])), mlir_elt(wtypes[k])))" for k in eachindex(wtypes)), ", "
)

# Batched program: x is u64[B] (or u64[B, L] for a multi-shape variant, summed over L), y is u64[B, K].
function batched_module(wtypes, wshapes, B::Int; L::Union{Int, Nothing} = nothing)
    K = length(wtypes)
    io = IOBuffer()
    xt = L === nothing ? "tensor<$(B)xui64>" : "tensor<$(B)x$(L)xui64>"
    println(io, "module {")
    println(io, "  func.func @main(%x: $xt, $(wargs(wtypes, wshapes))) -> tensor<$(B)x$(K)xui64> {")
    checksum_body(io, wtypes, wshapes)
    if L === nothing
        println(io, "    %xs = stablehlo.reshape %x : ($xt) -> tensor<$(B)xui64>")
    else
        println(io, "    %xs = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [1] : ($xt, tensor<ui64>) -> tensor<$(B)xui64>")
    end
    println(io, "    %bc = stablehlo.broadcast_in_dim %cs, dims = [1] : (tensor<$(K)xui64>) -> tensor<$(B)x$(K)xui64>")
    println(io, "    %bx = stablehlo.broadcast_in_dim %xs, dims = [0] : (tensor<$(B)xui64>) -> tensor<$(B)x$(K)xui64>")
    println(io, "    %y = stablehlo.add %bc, %bx : tensor<$(B)x$(K)xui64>")
    println(io, "    return %y : tensor<$(B)x$(K)xui64>")
    println(io, "  }")
    println(io, "}")
    return String(take!(io))
end

# Unbatched program: x is u64[1], y is u64[K].
function unbatched_module(wtypes, wshapes)
    K = length(wtypes)
    io = IOBuffer()
    println(io, "module {")
    println(io, "  func.func @main(%x: tensor<1xui64>, $(wargs(wtypes, wshapes))) -> tensor<$(K)xui64> {")
    checksum_body(io, wtypes, wshapes)
    println(io, "    %bx = stablehlo.broadcast_in_dim %x, dims = [0] : (tensor<1xui64>) -> tensor<$(K)xui64>")
    println(io, "    %y = stablehlo.add %cs, %bx : tensor<$(K)xui64>")
    println(io, "    return %y : tensor<$(K)xui64>")
    println(io, "  }")
    println(io, "}")
    return String(take!(io))
end

# ---- weights ------------------------------------------------------------------------------------

const ODD = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)

# Julia shapes for `n` tensors totalling about `bytes` of element type T, with `nd` axes each. `odd`
# shapes have small odd leading axes and a large last axis, so the legacy layout's permute and the
# program-order reshape both move every element.
function make_shapes(rng, T, bytes::Int, n::Int; nd = 2:4, odd::Bool = true)
    per = max(1, bytes ÷ (n * sizeof(T)))
    shapes = Tuple[]
    for _ in 1:n
        d = rand(rng, nd)
        if d == 0               # a one-element tensor (SafeTensors.jl cannot write a 0-d array)
            push!(shapes, (1,))
            continue
        end
        lead = odd ? Int[rand(rng, ODD) for _ in 1:(d - 1)] : Int[rand(rng, (8, 16, 32, 64)) for _ in 1:(d - 1)]
        last = max(1, per ÷ prod(lead; init = 1))
        push!(shapes, Tuple(vcat(lead, last)))
    end
    return shapes
end

function fill_weights(seed::Integer, wtypes, wshapes; special::Bool = false)
    ws = Array[]
    for k in eachindex(wtypes)
        T = wtypes[k]
        rng = Xoshiro(UInt64(seed) * 0x00010001 + UInt64(k))
        A = Array{T}(undef, wshapes[k]...)
        if T === Float32
            rand!(rng, A)
            A .-= 0.5f0
        else
            tmp = rand(rng, Float32, size(A))
            A .= BFloat16.(tmp .- 0.5f0)
        end
        if special && length(A) >= 64 && T === Float32
            # NaN payloads, signed zeros, infinities and subnormals: every byte must survive the
            # legacy permute, the migration, the host caches and the upload unchanged.
            u = reinterpret(UInt32, vec(A))
            for j in 1:max(1, length(u) ÷ 4096)
                i = 1 + ((j * 7919) % length(u))
                u[i] = (0x7fc00000 | UInt32(j & 0x003fffff), 0xffc00001, 0x80000000, 0x00000001, 0x7f800000, 0xff800000, 0x007fffff)[1 + j % 7]
            end
        end
        push!(ws, A)
    end
    return ws
end

# Weights in the layout write_bundle uses, written to an arbitrary path (alternate versions).
function write_weights(path::AbstractString, names, ws; layout::Symbol)
    meta = Dict("argument_order" => ReactantServerExport.JSON3.write(names))
    if layout === :xla
        meta["axis_order"] = "xla"
        data = Dict{String, AbstractArray}(n => ReactantServerExport._program_order(w) for (n, w) in zip(names, ws))
    else
        data = Dict{String, AbstractArray}(n => w for (n, w) in zip(names, ws))
    end
    tmp = path * ".tmp"
    SafeTensors.serialize(tmp, data, meta)
    mv(tmp, path; force = true)
    return path
end

hexs(x::UInt64) = string(x; base = 16, pad = 16)

# ---- catalog ------------------------------------------------------------------------------------

struct Spec
    name::String
    seed::Int
    wtypes::Vector{DataType}
    wshapes::Vector{Tuple}
    layout::Symbol
    batch_sizes::Vector{Int}       # empty: unbatched
    variants::Vector{Int}          # multi-shape input L values; empty: single shape
    special::Bool
    alts::Vector{Int}              # seeds of alternate weight versions (same shapes)
end

mb(x) = round(Int, x * 2^20)

function spec(
        name, seed, T, bytes, n; layout = :xla, nd = 2:4, odd = true, batch_sizes = [1, 4],
        variants = Int[], special = false, alts = Int[], mixed = false
    )
    rng = Xoshiro(seed)
    shapes = make_shapes(rng, T, bytes, n; nd, odd)
    types = DataType[mixed && isodd(k) ? BFloat16 : T for k in 1:n]
    return Spec(name, seed, types, shapes, layout, batch_sizes, variants, special, alts)
end

function profile(name::AbstractString)
    if name == "tiny"
        return Spec[
            spec("tiny-f32-a", 1, Float32, mb(2), 3),
            spec("tiny-f32-legacy", 2, Float32, mb(3), 4; layout = :legacy, nd = 3:5, alts = [201]),
            spec("tiny-bf16-many", 3, BFloat16, mb(1), 40),
            spec("tiny-mixed", 4, Float32, mb(2), 6; layout = :legacy, nd = 0:3, mixed = true),
            spec("tiny-special", 5, Float32, mb(1), 2; layout = :legacy, special = true),
            spec("tiny-variant", 6, Float32, mb(1), 2; variants = [1, 3]),
            spec("tiny-unbatched", 7, BFloat16, mb(1), 2; batch_sizes = Int[]),
            spec("tiny-upd", 8, Float32, mb(2), 3; alts = [801, 802]),
        ]
    elseif name == "small"
        return Spec[
            spec("sm-f32-a", 11, Float32, mb(120), 4),
            spec("sm-f32-legacy", 12, Float32, mb(150), 5; layout = :legacy, nd = 3:5, alts = [1201]),
            spec("sm-bf16-many", 13, BFloat16, mb(100), 200),
            spec("sm-mixed", 14, Float32, mb(80), 8; layout = :legacy, nd = 0:3, mixed = true),
            spec("sm-special", 15, Float32, mb(60), 2; layout = :legacy, special = true),
            spec("sm-variant", 16, Float32, mb(100), 3; variants = [1, 3]),
            spec("sm-big", 17, Float32, mb(300), 2; nd = 2:2),
            spec("sm-upd", 18, Float32, mb(150), 3; alts = [1801, 1802]),
            spec("sm-dup-a", 19, Float32, mb(100), 3),
        ]
    elseif name == "standard"
        return Spec[
            spec("syn-f32-a", 21, Float32, mb(400), 4),
            spec("syn-f32-legacy", 22, Float32, mb(600), 6; layout = :legacy, nd = 3:5),
            spec("syn-f32-big", 23, Float32, mb(1000), 2; nd = 2:2, odd = false),
            spec("syn-bf16-many", 24, BFloat16, mb(300), 400),
            spec("syn-bf16-legacy", 25, BFloat16, mb(500), 5; layout = :legacy, nd = 3:3, alts = [2501]),
            spec("syn-mixed-legacy", 26, Float32, mb(250), 12; layout = :legacy, nd = 0:3, mixed = true),
            spec("syn-special", 27, Float32, mb(120), 3; layout = :legacy, special = true),
            spec("syn-variant", 28, Float32, mb(200), 3; variants = [1, 3]),
            spec("syn-unbatched", 29, BFloat16, mb(150), 3; batch_sizes = Int[]),
            spec("syn-ver-1", 30, Float32, mb(350), 4),
            spec("syn-ver-2", 31, Float32, mb(350), 4),
            spec("syn-f32-c", 32, Float32, mb(700), 3),
            spec("syn-bf16-c", 33, BFloat16, mb(600), 4; layout = :legacy),
            spec("syn-upd", 34, Float32, mb(300), 3; alts = [3401, 3402, 3403]),
            spec("syn-dup-a", 35, Float32, mb(300), 3),
        ]
    end
    error("unknown profile $(repr(name)); use tiny, small or standard")
end

# Fix syn-ver-2's shapes to syn-ver-1's (two versions of one model), and add each profile's twin of
# its `*-dup-a` model (identical weights under another name: shared-store dedup).
function finish(specs::Vector{Spec})
    out = Spec[]
    for s in specs
        if endswith(s.name, "-ver-2")
            v1 = only(filter(x -> x.name == replace(s.name, "-ver-2" => "-ver-1"), specs))
            s = Spec(s.name, s.seed, v1.wtypes, v1.wshapes, s.layout, s.batch_sizes, s.variants, s.special, s.alts)
        end
        push!(out, s)
        if endswith(s.name, "-dup-a")
            push!(out, Spec(replace(s.name, "-dup-a" => "-dup-b"), s.seed, s.wtypes, s.wshapes, s.layout, s.batch_sizes, s.variants, s.special, s.alts))
        end
    end
    return out
end

function write_model(models_dir::AbstractString, alt_dir::AbstractString, s::Spec)
    K = length(s.wtypes)
    names = ["w$(lpad(k, 4, '0'))" for k in 1:K]
    ws = fill_weights(s.seed, s.wtypes, s.wshapes; special = s.special)
    cs = UInt64[host_checksum(ws[k], k) for k in 1:K]
    if isempty(s.batch_sizes)
        ins = [IOSpec("x", UInt64, [1])]
        outs = [IOSpec("y", UInt64, [K])]
        modules = Dict(0 => unbatched_module(s.wtypes, s.wshapes))
        input_shapes = nothing
    elseif isempty(s.variants)
        ins = [IOSpec("x", UInt64, [1]; batch_axis = 0)]
        outs = [IOSpec("y", UInt64, [K, 1]; batch_axis = 1)]
        modules = Dict(B => batched_module(s.wtypes, s.wshapes, B) for B in s.batch_sizes)
        input_shapes = nothing
    else
        ins = [IOSpec("x", UInt64, [-1, 1]; batch_axis = 1)]
        outs = [IOSpec("y", UInt64, [K, 1]; batch_axis = 1)]
        modules = Dict([L] => Dict(B => batched_module(s.wtypes, s.wshapes, B; L) for B in s.batch_sizes) for L in s.variants)
        input_shapes = [[L] for L in s.variants]
    end
    dir = joinpath(models_dir, s.name)
    isdir(dir) && rm(dir; recursive = true)
    write_bundle(
        dir; name = s.name, executable_inputs = ins, executable_outputs = outs, modules,
        weights = [n => w for (n, w) in zip(names, ws)], input_shapes, layout = s.layout
    )
    nbytes = sum(sizeof, ws)
    alts = Dict{String, Any}[]
    for a in s.alts
        aws = fill_weights(a, s.wtypes, s.wshapes)
        p = joinpath(alt_dir, s.name, "alt$(a).safetensors")
        mkpath(dirname(p))
        write_weights(p, names, aws; layout = s.layout)
        push!(alts, Dict{String, Any}("tag" => "alt$(a)", "path" => p, "checksums" => hexs.(UInt64[host_checksum(aws[k], k) for k in 1:K])))
    end
    # The original weights again, so an update can roll back to them.
    if !isempty(s.alts)
        p = joinpath(alt_dir, s.name, "orig.safetensors")
        write_weights(p, names, ws; layout = s.layout)
        push!(alts, Dict{String, Any}("tag" => "orig", "path" => p, "checksums" => hexs.(cs)))
    end
    entry = Dict{String, Any}(
        "kind" => "checksum", "name" => s.name, "dir" => dir, "outputs" => K,
        "batch_sizes" => s.batch_sizes, "variants" => s.variants, "layout" => string(s.layout),
        "weight_bytes" => nbytes, "tensors" => K,
        "dtypes" => sort(unique(string.(s.wtypes))), "checksums" => hexs.(cs), "alts" => alts,
    )
    @info "wrote model" s.name layout = s.layout tensors = K weights = Base.format_bytes(nbytes)
    return entry
end

function main(args)
    length(args) in (1, 2) || (println(stderr, "usage: gen_models.jl OUT_DIR [tiny|small|standard]"); return 2)
    out = abspath(args[1])
    prof = length(args) == 2 ? args[2] : "standard"
    models_dir = joinpath(out, "models")
    alt_dir = joinpath(out, "alt")
    mkpath(models_dir)
    mkpath(alt_dir)
    entries = Dict{String, Any}[]
    for s in finish(profile(prof))
        push!(entries, write_model(models_dir, alt_dir, s))
        GC.gc()
    end
    catalog = joinpath(out, "catalog.toml")
    # Merge with an existing catalog (e.g. the BiT exports) so both generators can share OUT_DIR.
    old = isfile(catalog) ? TOML.parsefile(catalog) : Dict{String, Any}()
    keep = filter(m -> m["kind"] != "checksum", get(old, "model", Any[]))
    open(catalog, "w") do io
        TOML.print(io, Dict("profile" => prof, "model" => vcat(keep, entries)))
    end
    @info "catalog written" catalog models = length(entries)
    return 0
end

abspath(PROGRAM_FILE) == @__FILE__() && exit(main(ARGS))
