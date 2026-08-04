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

# Shared BERT WordPiece tokenizer + wire padding; only vocab.txt is bundle-local.
const BT = ReactantServer.BertText

const TOKENIZER = BT.load_tokenizer(joinpath(@__DIR__, "vocab.txt"))
const MAX_LEN = 512
# Single compiled sequence length: every request pads to 512 (bit-identical to a tight bucket
# because all ops are attention-mask aware). Keep in sync with the export driver's SEQ_LEN.
const SEQ_BUCKETS = (512,)

function preprocess(inputs::Vector{NamedTensor})
    byname = Dict(t.name => t for t in inputs)
    input_ids, attention_mask, token_type_ids = BT.encode_pair_batch(TOKENIZER,
        BT.wire_text(byname["query"].data::Array{UInt8}),
        byname["keys"].data::Matrix{UInt8},              # (max_bytes, batch) col-major
        vec(byname["key_lens"].data);
        max_len=MAX_LEN, buckets=SEQ_BUCKETS, lens_name="key_lens")
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
