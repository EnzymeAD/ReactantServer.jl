# Cross-encoder bundle: one query scored against N keys in a single request. Preprocess
# tokenizes each (query, key) pair with the BERT pair template; the executable is the traced
# BertForSequenceClassification emitting raw f32 logits (batch,); sigmoid lives here per the
# classifier rule, and the client gets both tensors.
#
# Wire format (row-major, KServe):
#   client inputs
#     query    UINT8 [q_bytes]              the query's UTF-8 bytes (shared by all keys)
#     keys     UINT8 [batch, max_bytes]     per-row UTF-8 bytes, zero-padded to max_bytes
#     key_lens INT32 [batch]                byte length of each key row
#   client outputs
#     logits   FP32  [batch]                raw score per (query, key) pair
#     prob     FP32  [batch]                sigmoid similarity per (query, key) pair

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
    query = String(vec(copy(byname["query"].data::Array{UInt8})))
    keys_raw = byname["keys"].data::Matrix{UInt8}        # (max_bytes, batch) col-major
    lens = vec(byname["key_lens"].data)
    B = size(keys_raw, 2)
    length(lens) == B || error("key_lens has $(length(lens)) entries for $B key rows")

    encoded = Vector{Tuple{Vector{Int32},Vector{Int32}}}(undef, B)
    maxlen = 1
    for b in 1:B
        n = Int(lens[b])
        0 <= n <= size(keys_raw, 1) || error("key_lens[$b] = $n out of range")
        key = String(@view keys_raw[1:n, b])
        ids, type_ids = encode_pair(TOKENIZER, query, key; max_len=MAX_LEN)
        maxlen = max(maxlen, length(ids))
        encoded[b] = (ids, type_ids)
    end

    seq = _bucket(maxlen)
    input_ids = zeros(Int64, seq, B)                     # 0 == [PAD]
    attention_mask = zeros(Int64, seq, B)
    token_type_ids = zeros(Int64, seq, B)
    for b in 1:B
        ids, type_ids = encoded[b]
        for i in eachindex(ids)
            input_ids[i, b] = ids[i]
            attention_mask[i, b] = 1
            token_type_ids[i, b] = type_ids[i]
        end
    end
    return NamedTensor[NamedTensor("input_ids", input_ids),
                       NamedTensor("attention_mask", attention_mask),
                       NamedTensor("token_type_ids", token_type_ids)]
end

function postprocess(out::Vector{NamedTensor})
    logits = vec(out[1].data::Array{Float32})            # (batch,)
    prob = ReactantServer.NNlib.sigmoid.(logits)
    return NamedTensor[NamedTensor("logits", logits),
                       NamedTensor("prob", prob)]
end

# The serving identity is the bundle directory's basename; register under it so the dated
# bundle name lives in exactly one place.
register_model(basename(@__DIR__); preprocess=preprocess, postprocess=postprocess)
