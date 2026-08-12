# Dense embedding bundle: tokenize raw UTF-8 text in preprocess; the executable is the traced
# sentence-transformers encoder with masked mean-pooling + L2 normalization folded in, emitting
# a unit-norm (dim, batch) embedding. Postprocess is a passthrough (the embedding is already
# normalized in-graph), kept explicit so every bundle in this example reads the same way.
#
# Wire format (row-major, KServe):
#   client inputs
#     texts     UINT8  [batch, max_bytes]   per-row UTF-8 bytes, zero-padded to max_bytes
#     text_lens INT32  [batch]              byte length of each row
#   client outputs
#     embedding FP32   [batch, dim]         unit-norm sentence embedding per row

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
    input_ids, attention_mask = BT.encode_text_batch(
        TOKENIZER,
        byname["texts"].data::Matrix{UInt8},             # (max_bytes, batch) col-major
        vec(byname["text_lens"].data);
        max_len = MAX_LEN, buckets = SEQ_BUCKETS, lens_name = "text_lens"
    )
    return NamedTensor[
        NamedTensor("input_ids", input_ids),
        NamedTensor("attention_mask", attention_mask),
    ]
end

function postprocess(out::Vector{NamedTensor})
    emb = out[1].data::Matrix{Float32}                   # (dim, batch) col-major, unit-norm
    return NamedTensor[NamedTensor("embedding", emb)]
end

# The serving identity is the bundle directory's basename; register under it so the bundle name
# lives in exactly one place.
register_model(basename(@__DIR__); preprocess = preprocess, postprocess = postprocess)
