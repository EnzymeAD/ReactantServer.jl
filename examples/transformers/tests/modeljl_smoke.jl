# Smoke-test the four bundle model.jl hooks outside the server: stub NamedTensor + NNlib +
# register_model, stage each model.jl next to the shared tokenizer (as the export driver does),
# include it, and round-trip realistic requests through preprocess/postprocess. Pure Julia, no
# packages:
#
#   julia --startup-file=no examples/transformers/tests/modeljl_smoke.jl
const EXAMPLE = dirname(@__DIR__)
const MODELS = joinpath(EXAMPLE, "models")
const TOKENIZER = joinpath(EXAMPLE, "tokenizer")

module FakeRS
struct NamedTensor
    name::String
    data::Array
end
module NNlib
sigmoid(x) = 1 / (1 + exp(-x))
function softmax(x::AbstractMatrix; dims::Int=1)
    m = maximum(x; dims=dims)
    e = exp.(x .- m)
    return e ./ sum(e; dims=dims)
end
end
const REGISTERED = Dict{String,Any}()
register_model(name; preprocess=identity, postprocess=identity) =
    (REGISTERED[name] = (; preprocess, postprocess); nothing)
end

# Mirror the server's model.jl sandbox: an isolated module with register_model and ReactantServer
# (here FakeRS) injected; the `using ReactantServer: NamedTensor` line is rewritten to the injected
# binding since FakeRS is not a loadable package.
function load_bundle(name)
    # Stage model.jl + shared tokenizer into a temp dir, exactly like the export driver's _stage.
    dir = mktempdir()
    cp(joinpath(MODELS, name, "model.jl"), joinpath(dir, "model.jl"))
    for f in ("bert_wordpiece.jl", "vocab.txt")
        cp(joinpath(TOKENIZER, f), joinpath(dir, f))
    end
    # basename(@__DIR__) is the registered name, so name the temp dir after the model.
    named = joinpath(dir, name); mkpath(named)
    for f in ("model.jl", "bert_wordpiece.jl", "vocab.txt")
        mv(joinpath(dir, f), joinpath(named, f))
    end
    src = read(joinpath(named, "model.jl"), String)
    src = replace(src, "using ReactantServer: NamedTensor" => "const NamedTensor = ReactantServer.NamedTensor")
    m = Module(Symbol("Bundle_" * name))
    Core.eval(m, :(const ReactantServer = $FakeRS))
    Core.eval(m, :(const register_model = $(FakeRS.register_model)))
    Base.include_string(m, src, joinpath(named, "model.jl"))
    return m
end

pad_rows(rows::Vector{Vector{UInt8}}) = begin
    mx = maximum(length, rows; init=1)
    out = zeros(UInt8, mx, length(rows))
    for (i, r) in enumerate(rows)
        out[1:length(r), i] = r
    end
    out
end
NT(name, data) = FakeRS.NamedTensor(name, data)

texts = ["The central bank raised interest rates.", "café STRASSE 漢字", ""]
rows = [Vector{UInt8}(codeunits(t)) for t in texts]
text_inputs() = FakeRS.NamedTensor[NT("texts", pad_rows(rows)), NT("text_lens", Int32[length(r) for r in rows])]

# --- SPLADE ---
load_bundle("splade")
sp = FakeRS.REGISTERED["splade"]
prep = sp.preprocess(text_inputs())
ids = prep[1].data; mask = prep[2].data
@assert prep[1].name == "input_ids" && size(ids) == (512, 3) "unexpected shape $(size(ids))"
@assert ids[1, :] == fill(101, 3) "rows must start with [CLS]"
@assert ids[2, 3] == 102 "empty text row is [CLS][SEP]"
scores = zeros(Float32, 30522, 3)
scores[5, 1] = 1.4f0; scores[17, 1] = 0.004f0; scores[100, 2] = 0.3f0   # 0.004 rounds to 0 -> dropped
post = sp.postprocess(FakeRS.NamedTensor[NT("scores", scores)])
d = Dict(t.name => t.data for t in post)
@assert d["indices"] == Int32[4, 99] && d["values"] == Float32[1.4, 0.3]
@assert d["row_offsets"] == Int64[0, 1, 2, 2]
println("splade ok: seq bucket $(size(ids,1)), CSR ", d)

# --- embedding ---
load_bundle("embedding")
eb = FakeRS.REGISTERED["embedding"]
prep = eb.preprocess(text_inputs())
@assert prep[1].name == "input_ids" && size(prep[1].data) == (512, 3)
emb = rand(Float32, 384, 3)
post = eb.postprocess(FakeRS.NamedTensor[NT("embedding", emb)])
@assert length(post) == 1 && post[1].name == "embedding" && post[1].data === emb "embedding passthrough"
println("embedding ok: dim $(size(post[1].data, 1)), batch $(size(post[1].data, 2))")

# --- cross encoder ---
load_bundle("cross_encoder")
ce = FakeRS.REGISTERED["cross_encoder"]
keys = ["Rest and fluids help with a cold.", "unrelated documentation", ""]
krows = [Vector{UInt8}(codeunits(k)) for k in keys]
prep = ce.preprocess(FakeRS.NamedTensor[
    NT("query", Vector{UInt8}(codeunits("how to treat a cold"))),
    NT("keys", pad_rows(krows)),
    NT("key_lens", Int32[length(r) for r in krows]),
])
cids = prep[1].data; tt = prep[3].data
@assert size(cids) == (512, 3)
@assert count(==(102), cids[:, 1]) == 2 "pair row has two [SEP]"
@assert count(==(102), cids[:, 3]) == 1 "empty key falls back to single encoding"
@assert any(tt[:, 1] .== 1) && all(tt[:, 3] .== 0)
post = ce.postprocess(FakeRS.NamedTensor[NT("logits", Float32[2.0, -2.0, 0.0])])
pd = Dict(t.name => t.data for t in post)
@assert pd["logits"] == Float32[2.0, -2.0, 0.0]
@assert isapprox(pd["prob"], 1 ./ (1 .+ exp.(-Float32[2.0, -2.0, 0.0])); atol=1e-6)
println("cross_encoder ok: prob = ", pd["prob"])

# --- sentiment ---
load_bundle("sentiment")
se = FakeRS.REGISTERED["sentiment"]
prep = se.preprocess(text_inputs())
@assert prep[1].name == "input_ids" && size(prep[1].data) == (512, 3)
logits = Float32[-2.0 1.0 0.0; 2.0 -1.0 0.5]   # (2 classes, 3 rows): row1 POSITIVE, row2 NEGATIVE, row3 POSITIVE
post = se.postprocess(FakeRS.NamedTensor[NT("logits", logits)])
sd = Dict(t.name => t.data for t in post)
@assert sd["label_id"] == Int32[1, 0, 1] "argmax label ids"
@assert size(sd["probs"]) == (2, 3)
@assert all(isapprox.(sum(sd["probs"]; dims=1), 1.0f0; atol=1e-6)) "softmax columns sum to 1"
println("sentiment ok: label_id = ", sd["label_id"], ", probs col1 = ", sd["probs"][:, 1])

println("\nsmoke test passed")
