# Step 1 of 3: export the transformer text bundles (SPLADE, dense embedding, cross encoder,
# sentiment classifier) to StableHLO bundles from public HuggingFace checkpoints.
#
#   julia --project=examples/transformers/export examples/transformers/export/export_stablehlo.jl
#
# CPU trace: no GPU, no lease. One model.b{N}.mlir per batch size; all batch variants share one
# weights.safetensors. After each trace the staged model.jl and the shared tokenizer
# (bert_wordpiece.jl + vocab.txt) are copied into the bundle dir. Writes ./bundles/; skips a
# bundle that already exists (delete the bundles dir to re-export). Needs network on the first run
# for the four checkpoint downloads.

# Python init order (CRITICAL): torch / torchax must dlopen BEFORE Reactant loads, or Triton's
# static LLVM/MLIR registration SIGSEGVs. ReactantServerExport pulls in Reactant, so import the
# Python stack first.
using PythonCall
pyimport("torch"); pyimport("torch.export")
pyimport("torchax.export"); pyimport("torchax.ops.jaten")
try pyimport("triton._C.libtriton") catch end   # only present as a CUDA-wheel leftover
using ReactantServerExport
using ReactantServerExport: IOSpec

# Corporate SSL: HuggingFace downloads with Python's requests/urllib, which honor SSL_CERT_FILE.
# Point it at the OS trust store that holds the corporate CA (same bundle the Docker images use),
# so the checkpoint downloads trust a MitM proxy's CA. Mirrors examples/object_detection/export.
if !haskey(ENV, "SSL_CERT_FILE")
    for candidate in (get(ENV, "REQUESTS_CA_BUNDLE", ""), get(ENV, "CURL_CA_BUNDLE", ""),
                      get(ENV, "JULIA_SSL_CA_ROOTS_PATH", ""),
                      "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
        if !isempty(candidate) && isfile(candidate)
            ENV["SSL_CERT_FILE"] = candidate
            @info "Using system CA bundle for Python TLS" SSL_CERT_FILE = candidate
            break
        end
    end
end

const HERE = @__DIR__
const EXAMPLE = dirname(HERE)                            # examples/transformers
const TOKENIZER = joinpath(EXAMPLE, "tokenizer")         # shared bert_wordpiece.jl + vocab.txt
const MODELS = joinpath(EXAMPLE, "models")               # per-bundle model.jl sources
const OUT = joinpath(EXAMPLE, "bundles")
const REPO_ROOT = dirname(dirname(EXAMPLE))              # ReactantServer.jl checkout

# Single compiled sequence length (must match each model.jl SEQ_BUCKETS). Every request pads to
# SEQ_LEN, which is bit-identical to a tight bucket because all ops are attention-mask aware; the
# cost is wasted compute on short inputs, accepted here to cut compile time and command-buffer
# count. Two batch sizes -> two programs per bundle (model.b1.mlir, model.b8.mlir).
const SEQ_LEN = 512
const BATCH_SIZES = [1, 8]

# Example inputs are Julia arrays: (seq, batch), batch trailing. Values are irrelevant to the
# trace, but the sample forward runs for real, so use a valid token ([CLS]=101) and an all-ones
# mask to keep the pooling / softmax rows finite.
_ids(seq) = fill(Int64(101), seq, 1)
_ones(seq) = ones(Int64, seq, 1)
_zeros(seq) = zeros(Int64, seq, 1)

# Load the torch builders from export_model.py (put its dir on sys.path once).
function _builders()
    sys = pyimport("sys")
    any(p -> pyconvert(String, p) == HERE, sys.path) || sys.path.insert(0, HERE)
    return pyimport("export_model")
end

# Copy the staged serve-time files into the bundle; the frontend never writes model.jl. The
# tokenizer must live inside the bundle because model.jl `include`s it at serve time.
function _stage(bundle_dir, model_name)
    cp(joinpath(MODELS, model_name, "model.jl"), joinpath(bundle_dir, "model.jl"); force=true)
    for f in ("bert_wordpiece.jl", "vocab.txt")
        cp(joinpath(TOKENIZER, f), joinpath(bundle_dir, f); force=true)
    end
end

_prov(extra) = merge(ReactantServerExport.collect_provenance(REPO_ROOT), Dict{String,Any}(extra...))

function export_splade(; name="splade")
    em = _builders()
    model = em.build_splade()
    example = (_ids(SEQ_LEN), _ones(SEQ_LEN))
    dir = joinpath(OUT, name)
    export_bundle(:pytorch, model, example;
        dir, name,
        input_names  = ["input_ids", "attention_mask"],
        output_names = ["term_scores"],
        axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'],
                            "term_scores" => ['v']),
        batch_sizes = BATCH_SIZES,
        matmul_precision = "highest",
        client_inputs = [
            IOSpec("texts", UInt8, [1, -1]; batch_axis=0, letters=['c']),
            IOSpec("text_lens", Int32, [1]; batch_axis=0),
        ],
        client_outputs = [
            IOSpec("indices", Int32, [-1]; letters=['k']),
            IOSpec("values", Float32, [-1]; letters=['w']),
            IOSpec("row_offsets", Int64, [-1]; letters=['r']),
        ],
        provenance = _prov((
            "model" => "SPLADE (BertForMaskedLM + in-graph log1p/relu/mask/max term-score head)",
            "checkpoint" => "naver/splade-cocondenser-ensembledistil",
            "source" => "https://huggingface.co/naver/splade-cocondenser-ensembledistil",
            "hidden_size" => 768, "num_hidden_layers" => 12, "vocab_size" => 30522,
            "seq_len" => SEQ_LEN,
        )))
    _stage(dir, "splade")
    return dir
end

function export_embedding(; name="embedding")
    em = _builders()
    model = em.build_embedding()
    example = (_ids(SEQ_LEN), _ones(SEQ_LEN))
    dir = joinpath(OUT, name)
    export_bundle(:pytorch, model, example;
        dir, name,
        input_names  = ["input_ids", "attention_mask"],
        output_names = ["embedding"],
        axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'],
                            "embedding" => ['d']),
        batch_sizes = BATCH_SIZES,
        matmul_precision = "highest",
        client_inputs = [
            IOSpec("texts", UInt8, [1, -1]; batch_axis=0, letters=['c']),
            IOSpec("text_lens", Int32, [1]; batch_axis=0),
        ],
        client_outputs = [
            IOSpec("embedding", Float32, [1, 384]; batch_axis=0, letters=['d']),
        ],
        provenance = _prov((
            "model" => "dense embedding (BertModel + in-graph masked mean-pool + L2 normalize)",
            "checkpoint" => "sentence-transformers/all-MiniLM-L6-v2",
            "source" => "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            "hidden_size" => 384, "num_hidden_layers" => 6, "vocab_size" => 30522,
            "embedding_dim" => 384, "seq_len" => SEQ_LEN,
        )))
    _stage(dir, "embedding")
    return dir
end

function export_cross(; name="cross_encoder")
    em = _builders()
    model = em.build_cross()
    example = (_ids(SEQ_LEN), _ones(SEQ_LEN), _zeros(SEQ_LEN))
    dir = joinpath(OUT, name)
    export_bundle(:pytorch, model, example;
        dir, name,
        input_names  = ["input_ids", "attention_mask", "token_type_ids"],
        output_names = ["logits"],
        axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'],
                            "token_type_ids" => ['s']),
        batch_sizes = BATCH_SIZES,
        matmul_precision = "highest",
        client_inputs = [
            IOSpec("query", UInt8, [-1]; letters=['q']),
            IOSpec("keys", UInt8, [1, -1]; batch_axis=0, letters=['c']),
            IOSpec("key_lens", Int32, [1]; batch_axis=0),
        ],
        client_outputs = [
            IOSpec("logits", Float32, [1]; batch_axis=0),
            IOSpec("prob", Float32, [1]; batch_axis=0),
        ],
        provenance = _prov((
            "model" => "cross encoder (BertForSequenceClassification, raw logits; sigmoid in model.jl)",
            "checkpoint" => "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "source" => "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
            "hidden_size" => 384, "num_hidden_layers" => 6, "vocab_size" => 30522,
            "seq_len" => SEQ_LEN,
        )))
    _stage(dir, "cross_encoder")
    return dir
end

function export_sentiment(; name="sentiment")
    em = _builders()
    model = em.build_sentiment()
    example = (_ids(SEQ_LEN), _ones(SEQ_LEN))
    dir = joinpath(OUT, name)
    export_bundle(:pytorch, model, example;
        dir, name,
        input_names  = ["input_ids", "attention_mask"],
        output_names = ["logits"],
        axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'],
                            "logits" => ['c']),
        batch_sizes = BATCH_SIZES,
        matmul_precision = "highest",
        client_inputs = [
            IOSpec("texts", UInt8, [1, -1]; batch_axis=0, letters=['c']),
            IOSpec("text_lens", Int32, [1]; batch_axis=0),
        ],
        client_outputs = [
            IOSpec("logits", Float32, [1, 2]; batch_axis=0, letters=['c']),
            IOSpec("probs", Float32, [1, 2]; batch_axis=0, letters=['c']),
            IOSpec("label_id", Int32, [1]; batch_axis=0),
        ],
        provenance = _prov((
            "model" => "sentiment classifier (DistilBERT SST-2, raw logits; softmax in model.jl)",
            "checkpoint" => "distilbert-base-uncased-finetuned-sst-2-english",
            "source" => "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english",
            "hidden_size" => 768, "num_hidden_layers" => 6, "vocab_size" => 30522,
            "num_labels" => 2, "id2label" => "0=NEGATIVE, 1=POSITIVE", "seq_len" => SEQ_LEN,
        )))
    _stage(dir, "sentiment")
    return dir
end

function main()
    mkpath(OUT)
    for (name, f) in (("splade", export_splade), ("embedding", export_embedding),
                      ("cross_encoder", export_cross), ("sentiment", export_sentiment))
        if isdir(joinpath(OUT, name))
            @info "Skipping $name (already built; delete $(joinpath(OUT, name)) to re-export)"
            continue
        end
        @info "Exporting $name ..."
        dir = f(; name)
        println("exported: $dir")
    end
end

main()
