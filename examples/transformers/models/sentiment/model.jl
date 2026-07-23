# Sentiment classifier bundle: tokenize raw UTF-8 text in preprocess; the executable is the
# traced DistilBERT SST-2 classifier emitting raw f32 logits (2, batch). Per the classifier rule
# the softmax lives here, not in the traced graph, so the client gets the raw logits, the class
# probabilities, and the predicted label id (0 == NEGATIVE, 1 == POSITIVE for SST-2).
#
# Wire format (row-major, KServe):
#   client inputs
#     texts     UINT8  [batch, max_bytes]   per-row UTF-8 bytes, zero-padded to max_bytes
#     text_lens INT32  [batch]              byte length of each row
#   client outputs
#     logits    FP32   [batch, 2]           raw [NEGATIVE, POSITIVE] logits per row
#     probs     FP32   [batch, 2]           softmax probabilities per row
#     label_id  INT32  [batch]              argmax class id (0 NEGATIVE, 1 POSITIVE)

using ReactantServer: NamedTensor

Base.include(@__MODULE__, joinpath(@__DIR__, "bert_wordpiece.jl"))
using .BertWordPiece

const TOKENIZER = load_tokenizer(joinpath(@__DIR__, "vocab.txt"))
const MAX_LEN = 512
# Single compiled sequence length: every request pads to 512 (bit-identical to a tight bucket
# because all ops are attention-mask aware). Keep in sync with the export driver's SEQ_LEN.
const SEQ_BUCKETS = (512,)

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
    logits = out[1].data::Matrix{Float32}                # (2, batch) col-major: rows are classes
    probs = ReactantServer.NNlib.softmax(logits; dims=1)
    label_id = Int32[argmax(view(logits, :, b)) - 1 for b in 1:size(logits, 2)]  # 0-based class id
    return NamedTensor[NamedTensor("logits", logits),
                       NamedTensor("probs", probs),
                       NamedTensor("label_id", label_id)]
end

# The serving identity is the bundle directory's basename; register under it so the bundle name
# lives in exactly one place.
register_model(basename(@__DIR__); preprocess=preprocess, postprocess=postprocess)
