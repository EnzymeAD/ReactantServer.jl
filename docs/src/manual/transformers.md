```@meta
CurrentModule = ReactantServer
```

# Transformer Text Models (BERT)

BERT-family text models are the canonical case for the plain [bundle](bundles.md) path with a
`model.jl` that owns tokenization. The transformer itself is dense tensor math that
`torch.export` traces cleanly to a single static StableHLO program, but two things around it are
not tensor math: turning a string into token ids, and turning raw logits into the shape a caller
wants. ReactantServer keeps both in the bundle's `model.jl`, so the traced graph is exactly the
encoder (plus any in-graph numeric head), the client sends raw UTF-8 text, and no Python
tokenizer or activation library is needed at serve time.

This page walks through exporting four public HuggingFace checkpoints that cover the common
text-serving shapes. For the bundle/manifest/`model.jl` contract itself see
[Bundles & model.jl](bundles.md); for the PyTorch export mechanics see the
`export_bundle(:pytorch, ...)` section there.

!!! tip "Runnable example"
    `examples/transformers/` in the repository is a complete, runnable version of this
    walkthrough. It is split into three copy-pasteable commands, each with its own environment:
    **export** the four bundles, **serve** them on a single GPU, and a **client** that queries
    all four over KServe V2. See `examples/transformers/README.md`.

## The four models

| Bundle | HuggingFace checkpoint | torch class | In-graph head | `model.jl` |
| --- | --- | --- | --- | --- |
| `splade` | `naver/splade-cocondenser-ensembledistil` | `AutoModelForMaskedLM` | `log1p/relu/mask/max` term scores | tokenize; sparsify to CSR |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` | `AutoModel` | masked mean-pool + L2 normalize | tokenize; passthrough |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `AutoModelForSequenceClassification` | none (raw logit) | tokenize pair; add sigmoid |
| `sentiment` | `distilbert-base-uncased-finetuned-sst-2-english` | `AutoModelForSequenceClassification` | none (raw logits) | tokenize; add softmax + argmax |

All four share the `bert-base-uncased` WordPiece vocab (30522 tokens), so one tokenizer serves
every bundle: the code in `ReactantServer.BertText`, the vocab as a per-bundle `vocab.txt`.

## Tokenization lives in the bundle

Clients send raw UTF-8 bytes; the bundle's `model.jl` decodes them and tokenizes in Julia with
`ReactantServer.BertText`, a self-contained BERT WordPiece (uncased) reimplementation that depends
on nothing but `Base`. It matches the HuggingFace Rust tokenizer, including the awkward corners:
unassigned codepoints stay in the word, ASCII symbols like `` $ + < = > ^ ` | ~ `` count as
punctuation, `LongestFirst` pair truncation, and special-token literals in raw text.

`BertText` is the request-side counterpart to `ReactantServer.DetectionGlue`: shared code lives in
the package, per-model configuration is baked into `model.jl`, and the only tokenizer asset the
export driver copies into a bundle is the checkpoint's `vocab.txt` (loaded at serve time relative
to `model.jl`'s directory). A bundle exported this way needs a server that has `BertText`, the same
version coupling the detector bundles already accept for `DetectionGlue`.

`preprocess` turns the wire tensors into the executable's token-id inputs. The single-sequence
models (`splade`, `embedding`, `sentiment`) call `encode_text_batch` and emit `input_ids` +
`attention_mask`; the cross encoder is a text *pair* (one query scored against N keys) and calls
`encode_pair_batch`, adding `token_type_ids`. Both decode the padded UInt8 rows, encode with the
BERT template, and pad the batch up to the smallest compiled sequence bucket that fits:

```julia
const BT = ReactantServer.BertText
const TOKENIZER = BT.load_tokenizer(joinpath(@__DIR__, "vocab.txt"))

function preprocess(inputs::Vector{NamedTensor})
    byname = Dict(t.name => t for t in inputs)
    input_ids, attention_mask = BT.encode_text_batch(TOKENIZER,
        byname["texts"].data::Matrix{UInt8},             # (max_bytes, batch) col-major
        vec(byname["text_lens"].data);
        max_len=MAX_LEN, buckets=SEQ_BUCKETS, lens_name="text_lens")
    return NamedTensor[NamedTensor("input_ids", input_ids),
                       NamedTensor("attention_mask", attention_mask)]
end
```

A bundle that needs a shape these two do not cover can drop a level and compose the same pieces:
`wire_texts` decodes the padded rows to Strings, `encode_single` / `encode_pair` encode one text,
and `pad_batch` packs the result into `(seq, batch)` matrices at the right bucket.

## The raw-logits / classifier rule

The two classifiers (`cross_encoder`, `sentiment`) emit **raw logits** from the traced graph; the
activation is applied in `postprocess`, which returns both the raw logits and the probabilities.
Keeping the sigmoid/softmax out of the graph follows the same rule as every other classification
bundle in the package, and lets a caller read either value:

```julia
function postprocess(out::Vector{NamedTensor})           # sentiment
    logits = out[1].data::Matrix{Float32}                # (2, batch): rows are classes
    probs = ReactantServer.NNlib.softmax(logits; dims=1)
    label_id = Int32[argmax(view(logits, :, b)) - 1 for b in 1:size(logits, 2)]
    return NamedTensor[NamedTensor("logits", logits),
                       NamedTensor("probs", probs),
                       NamedTensor("label_id", label_id)]
end
```

By contrast, SPLADE's `log1p/relu/mask/max` and the embedding model's masked mean-pool + L2
normalization stay **in** the traced graph: they define the model's numeric output (and, for
SPLADE, do the `(batch, seq, vocab) -> (batch, vocab)` reduction), rather than being a
presentation-layer activation. SPLADE's `postprocess` then only sparsifies the dense term-score
vector to a CSR triple; the embedding `postprocess` is a passthrough.

## The wire contract

Because `model.jl` reshapes the wire I/O, the manifest declares `client_inputs`/`client_outputs`
(what the caller sends and receives) distinct from `executable_inputs`/`executable_outputs` (the
token tensors and raw logits the traced program sees). The export driver states this with
`IOSpec`s passed to `export_bundle(:pytorch, ...)`; for example the sentiment bundle:

```julia
export_bundle(:pytorch, model, (input_ids, attention_mask);
    dir, name,
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["logits"],
    axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'], "logits" => ['c']),
    batch_sizes = [1, 8],
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
    provenance = _prov(...))
```

`matmul_precision = "highest"` is required on the PyTorch path: JAX freezes float32 matmul
precision at trace time based on the export host, so a CPU trace without it would bake a lower
precision. See [Bundles & model.jl](bundles.md) for the full `IOSpec` and manifest encoding.

## Sequence length and batching

Each bundle compiles a **single** sequence length of 512 and batch sizes `[1, 8]`, i.e. two
programs per bundle (`model.b1.mlir`, `model.b8.mlir`). Every request pads to 512, which is
bit-identical to a tighter bucket because every op is attention-mask aware; the tradeoff is wasted
compute on short inputs in exchange for far fewer programs and lower compile time and
command-buffer count. The constant lives in two places that must agree: `SEQ_LEN` in the export
driver and `SEQ_BUCKETS = (512,)` in each `model.jl`. To trade compute back for more programs, add
sequence buckets or batch sizes in the driver (`export_bundle` supports multiple input shapes over
one weight set) and bump `SEQ_BUCKETS`/`max_batch` to match.

## See also

- `examples/transformers/` for the runnable end-to-end example (export, serve, client)
- [Bundles & model.jl](bundles.md) for the bundle contract, `IOSpec`, and the manifest encoding
- [Object Detection](object_detection.md) for the data-dependent (meta) export path
- [Client Usage](client_usage.md) for building requests with `InferInput`/`InferOutput`
