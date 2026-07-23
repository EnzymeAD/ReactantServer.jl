# Step 3 of 3: query the running server. Exercises all four transformer bundles from a second
# terminal.
#
#   julia --project=examples/transformers/client examples/transformers/client/query.jl
#
# Connects to the server from step 2 on 127.0.0.1:$TX_PORT (default 8080). This environment has no
# Reactant and no PythonCall, so it loads fast. All four models tokenize server-side, so the client
# only sends raw UTF-8 text bytes plus per-row byte lengths.

using ReactantServerClient

const HERE = @__DIR__
const VOCAB_PATH = normpath(joinpath(HERE, "..", "tokenizer", "vocab.txt"))
const HOST = get(ENV, "TX_HOST", "127.0.0.1")
const PORT = parse(Int, get(ENV, "TX_PORT", "8080"))
const URL = "grpc://$HOST:$PORT"
const SENTIMENT_LABELS = ("NEGATIVE", "POSITIVE")   # DistilBERT SST-2 id2label

# The shared bert-base-uncased vocab, for turning SPLADE's 0-based vocab ids back into tokens.
const VOCAB = readlines(VOCAB_PATH)                 # VOCAB[id + 1] is the token for 0-based id
token_for(id::Integer) = (1 <= id + 1 <= length(VOCAB)) ? VOCAB[id + 1] : "<$id>"

# Pack a batch of strings into the wire tensors: a (max_bytes, batch) UInt8 matrix of zero-padded
# UTF-8 rows plus an Int32 byte-length per row. InferInput reverses the Julia (max_bytes, batch)
# to the network's row-major (batch, max_bytes).
function pack_texts(texts)
    rows = [Vector{UInt8}(codeunits(t)) for t in texts]
    maxb = maximum(length, rows; init = 1)
    mat = zeros(UInt8, maxb, length(rows))
    for (i, r) in enumerate(rows)
        mat[1:length(r), i] = r
    end
    return mat, Int32[length(r) for r in rows]
end

model(name) = KServeModel(URL, name; max_batch_size = 8)

function demo_embedding()
    println("\n== dense embedding (all-MiniLM-L6-v2) ==")
    texts = ["A dog runs across the park.",
             "A puppy sprints through the field.",
             "The quarterly earnings report was released today."]
    mat, lens = pack_texts(texts)
    resp = infer_sync(model("embedding"),
                      [InferInput("texts", mat), InferInput("text_lens", lens)])
    emb = InferOutput("embedding", resp, Float32)      # (dim, batch)
    cos(a, b) = sum(emb[:, a] .* emb[:, b])            # unit-norm rows -> dot product is cosine
    println("  embedding dim = $(size(emb, 1)), row norms ≈ ",
            round.([sqrt(sum(abs2, emb[:, b])) for b in 1:size(emb, 2)]; digits = 4))
    println("  cosine(1, 2) = $(round(cos(1, 2); digits = 4))  (near-paraphrase)")
    println("  cosine(1, 3) = $(round(cos(1, 3); digits = 4))  (unrelated)")
end

function demo_splade(; topk = 8)
    println("\n== SPLADE (naver/splade-cocondenser-ensembledistil) ==")
    text = "The central bank raised interest rates to curb inflation."
    mat, lens = pack_texts([text])
    resp = infer_sync(model("splade"),
                      [InferInput("texts", mat), InferInput("text_lens", lens)])
    indices = InferOutput("indices", resp, Int32)
    values = InferOutput("values", resp, Float32)
    offsets = InferOutput("row_offsets", resp, Int64)  # 0-based, length batch+1
    span = (offsets[1] + 1):offsets[2]                 # row 1
    order = sortperm(values[span]; rev = true)
    println("  text: ", text)
    println("  $(length(span)) expansion terms; top $topk by weight:")
    for j in order[1:min(topk, length(order))]
        id = indices[span[j]]
        println("    $(rpad(token_for(id), 16)) $(round(values[span[j]]; digits = 3))")
    end
end

function demo_cross()
    println("\n== cross encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) ==")
    query = "How do I treat a cold?"
    candidates = ["Rest, fluids, and over-the-counter medicine help with a cold.",
                  "The GDP grew by two percent last quarter.",
                  "Drinking warm tea can soothe a sore throat from a cold."]
    keys, key_lens = pack_texts(candidates)
    resp = infer_sync(model("cross_encoder"),
                      [InferInput("query", Vector{UInt8}(codeunits(query))),
                       InferInput("keys", keys), InferInput("key_lens", key_lens)])
    prob = InferOutput("prob", resp, Float32)          # (batch,)
    println("  query: ", query)
    for i in sortperm(prob; rev = true)
        println("    $(round(prob[i]; digits = 4))  $(candidates[i])")
    end
end

function demo_sentiment()
    println("\n== sentiment (distilbert-base-uncased-finetuned-sst-2-english) ==")
    texts = ["I absolutely loved this movie, it was fantastic!",
             "The plot was dull and the acting was terrible.",
             "It was fine, nothing special."]
    mat, lens = pack_texts(texts)
    resp = infer_sync(model("sentiment"),
                      [InferInput("texts", mat), InferInput("text_lens", lens)])
    probs = InferOutput("probs", resp, Float32)        # (2, batch)
    label_id = InferOutput("label_id", resp, Int32)    # (batch,)
    for b in eachindex(label_id)
        label = SENTIMENT_LABELS[label_id[b] + 1]
        println("    $(rpad(label, 8)) $(round(probs[label_id[b] + 1, b]; digits = 4))  \"$(texts[b])\"")
    end
end

function main()
    kserve_init()
    try
        @info "Querying transformer bundles" server = "$HOST:$PORT"
        demo_embedding()
        demo_splade()
        demo_cross()
        demo_sentiment()
        println("\ndone")
    finally
        kserve_shutdown()
    end
end

main()
