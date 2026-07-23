# SPLADE bundle: tokenize raw UTF-8 text in preprocess, sparsify the dense term scores in
# postprocess. The executable is the traced BertForMaskedLM with log1p/relu/mask/max folded
# in, emitting dense f32 (vocab, batch).
#
# Wire format (row-major, KServe):
#   client inputs
#     texts     UINT8  [batch, max_bytes]   per-row UTF-8 bytes, zero-padded to max_bytes
#     text_lens INT32  [batch]              byte length of each row
#   client outputs (CSR-style, 0-based vocab indices to match the Python server)
#     indices     INT32 [K]                 all rows' nonzero vocab indices, concatenated
#     values      FP32  [K]                 matching weights (unrounded)
#     row_offsets INT64 [batch+1]           row b spans indices[row_offsets[b]+1 : row_offsets[b+1]] (0-based offsets)

using ReactantServer: NamedTensor

Base.include(@__MODULE__, joinpath(@__DIR__, "bert_wordpiece.jl"))
using .BertWordPiece

const TOKENIZER = load_tokenizer(joinpath(@__DIR__, "vocab.txt"))
const MAX_LEN = 512
# Single compiled sequence length: every request pads to 512 (bit-identical to a tight bucket
# because all ops are attention-mask aware). Keep in sync with the export driver's SEQ_LEN.
const SEQ_BUCKETS = (512,)
const VOCAB_SIZE = 30522

_bucket(n::Int) = something(findfirst(>=(n), SEQ_BUCKETS), length(SEQ_BUCKETS)) |> i -> SEQ_BUCKETS[i]

function preprocess(inputs::Vector{NamedTensor})
    byname = Dict(t.name => t for t in inputs)
    texts = byname["texts"].data::Matrix{UInt8}          # (max_bytes, batch) col-major
    lens = vec(byname["text_lens"].data)
    B = size(texts, 2)
    length(lens) == B || error("text_lens has $(length(lens)) entries for $B text rows")

    encoded = Vector{Vector{Int32}}(undef, B)
    maxlen = 1
    for b in 1:B
        n = Int(lens[b])
        0 <= n <= size(texts, 1) || error("text_lens[$b] = $n out of range")
        s = String(@view texts[1:n, b])
        ids = encode_single(TOKENIZER, s; max_len=MAX_LEN)
        maxlen = max(maxlen, length(ids))
        encoded[b] = ids
    end

    seq = _bucket(maxlen)
    input_ids = zeros(Int64, seq, B)                     # 0 == [PAD]
    attention_mask = zeros(Int64, seq, B)
    for b in 1:B, (i, id) in enumerate(encoded[b])
        input_ids[i, b] = id
        attention_mask[i, b] = 1
    end
    return NamedTensor[NamedTensor("input_ids", input_ids),
                       NamedTensor("attention_mask", attention_mask)]
end

function postprocess(out::Vector{NamedTensor})
    scores = out[1].data::Matrix{Float32}                # (vocab, batch) col-major
    size(scores, 1) == VOCAB_SIZE || error("expected $VOCAB_SIZE term scores per row")
    B = size(scores, 2)

    indices = Int32[]
    values = Float32[]
    row_offsets = Vector{Int64}(undef, B + 1)
    row_offsets[1] = 0
    @inbounds for b in 1:B
        for v in 1:VOCAB_SIZE
            x = scores[v, b]
            # Match the Python server's keep rule: round(x, decimals=2) != 0 with
            # half-to-even ties (torch.round semantics == Julia's default rounding).
            if round(x * 100.0f0) != 0.0f0
                push!(indices, Int32(v - 1))             # 0-based vocab ids on the wire
                push!(values, x)                         # unrounded weight
            end
        end
        row_offsets[b + 1] = length(indices)
    end
    return NamedTensor[NamedTensor("indices", indices),
                       NamedTensor("values", values),
                       NamedTensor("row_offsets", row_offsets)]
end

# The serving identity is the bundle directory's basename; register under it so the dated
# bundle name lives in exactly one place.
register_model(basename(@__DIR__); preprocess=preprocess, postprocess=postprocess)
