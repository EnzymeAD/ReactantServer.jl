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
    walkthrough: [Serve Transformer Text Models](tutorials/examples/transformers.md) exports the four
    bundles, serves them, and queries all four over KServe V2.

## The four models

| Bundle | HuggingFace checkpoint | torch class | In-graph head | `model.jl` |
| --- | --- | --- | --- | --- |
| `splade` | `naver/splade-cocondenser-ensembledistil` | `AutoModelForMaskedLM` | `log1p/relu/mask/max` term scores | tokenize; sparsify to CSR |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` | `AutoModel` | masked mean-pool + L2 normalize | tokenize; passthrough |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `AutoModelForSequenceClassification` | none (raw logit) | tokenize pair; add sigmoid |
| `sentiment` | `distilbert-base-uncased-finetuned-sst-2-english` | `AutoModelForSequenceClassification` | none (raw logits) | tokenize; add softmax + argmax |

The four cover the common text-serving shapes: learned sparse term expansion (SPLADE), a dense
384-dim unit-norm sentence embedding, query/document relevance scoring, and binary sentiment
classification. All four share the `bert-base-uncased` WordPiece vocab (30522 tokens), so one
tokenizer serves every bundle: the code in `ReactantServer.BertText`, the vocab as a per-bundle
`vocab.txt`.

## Tokenization lives in the bundle

Clients send raw UTF-8 bytes; the bundle's `model.jl` decodes them and tokenizes in Julia with
`ReactantServer.BertText`, a self-contained BERT WordPiece (uncased) reimplementation that depends
on nothing but `Base`. It matches the HuggingFace Rust tokenizer, including the awkward corners:
unassigned codepoints stay in the word, ASCII symbols like `` $ + < = > ^ ` | ~ `` count as
punctuation, `LongestFirst` pair truncation, and special-token literals in raw text.

Shared code lives in the package and per-model configuration is baked into `model.jl`: the only
tokenizer asset the export driver copies into a bundle is the checkpoint's `vocab.txt` (loaded at
serve time relative to `model.jl`'s directory). A bundle exported this way needs a server that has
`BertText`, so the bundle and the server version are coupled.

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

The `BertText` surface is a small set of composable helpers:

| Helper | What it does |
| --- | --- |
| `load_tokenizer(vocab_path)` | Load a HuggingFace `vocab.txt` (one token per line, line number minus 1 is the id); returns the `BertTokenizer` the other helpers take. |
| `wire_text(bytes)` / `wire_texts(texts, lens)` | Decode one UTF-8 tensor (the cross encoder's shared query) or a `(max_bytes, batch)` matrix of zero-padded rows using the per-row byte lengths. |
| `encode_single(t, text)` / `encode_pair(t, a, b)` | Encode one text as `[CLS] text [SEP]`, or a pair as `[CLS] a [SEP] b [SEP]` with HF `LongestFirst` truncation; the pair form also returns `token_type_ids`. |
| `encode_text_batch` / `encode_pair_batch` | The preprocess entry points: decode the wire rows, encode each, and pad to the smallest bucket that fits. |
| `pad_batch(encoded, buckets)` / `seq_bucket(n, buckets)` | Pack per-row token ids into `(seq, batch)` matrices at the smallest compiled sequence length that holds them; `pad_batch` errors clearly if an input exceeds the largest bucket. |

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

The cross encoder's `postprocess` is the same shape with a sigmoid instead: `prob =
sigmoid.(logits)` over the per-pair score vector. By contrast, SPLADE's `log1p/relu/mask/max` and
the embedding model's masked mean-pool + L2 normalization stay **in** the traced graph: they
define the model's numeric output (and, for SPLADE, do the `(batch, seq, vocab) -> (batch,
vocab)` reduction), rather than being a presentation-layer activation. SPLADE's `postprocess`
then only sparsifies the dense term-score vector to a CSR triple, applying the Python server's
keep rule (`round(x, decimals=2) != 0`, half-to-even ties, which `torch.round` semantics match
Julia's default rounding) and emitting 0-based vocab indices, unrounded weights, and row offsets;
the embedding `postprocess` is a passthrough.

## The wire contract

Because `model.jl` reshapes the wire I/O, the manifest declares `client_inputs`/`client_outputs`
(what the caller sends and receives) distinct from `executable_inputs`/`executable_outputs` (the
token tensors and raw logits the traced program sees). The export driver states this with
`IOSpec`s passed to `export_bundle(:pytorch, ...)`. Their shapes are in Julia (column-major)
order with the batch axis last, the reverse of the wire shapes in the table below: `texts` is
`[-1, 1]` with `batch_axis=1`, the Julia `(max_bytes, batch)` matrix `model.jl` reads. For
example the sentiment bundle:

```julia
export_bundle(:pytorch, model, (input_ids, attention_mask);
    dir, name,
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["logits"],
    axis_letters = Dict("input_ids" => ['s'], "attention_mask" => ['s'], "logits" => ['c']),
    batch_sizes = [1, 8],
    matmul_precision = "highest",
    client_inputs = [
        IOSpec("texts", UInt8, [-1, 1]; batch_axis=1, letters=['c']),
        IOSpec("text_lens", Int32, [1]; batch_axis=0),
    ],
    client_outputs = [
        IOSpec("logits", Float32, [2, 1]; batch_axis=1, letters=['c']),
        IOSpec("probs", Float32, [2, 1]; batch_axis=1, letters=['c']),
        IOSpec("label_id", Int32, [1]; batch_axis=0),
    ],
    provenance = _prov(...))
```

`matmul_precision = "highest"` is required on the PyTorch path: JAX freezes float32 matmul
precision at trace time based on the export host, so a CPU trace without it would bake a lower
precision. See [Bundles & model.jl](bundles.md) for the full `IOSpec` and manifest encoding.

The client-facing shapes below are in row-major (KServe/client) order; the wire codec converts
to the Julia column-major convention at the boundary.

| Bundle | Client inputs | Client outputs |
| --- | --- | --- |
| `splade` | `texts` UINT8 `[batch, max_bytes]` (zero-padded rows), `text_lens` INT32 `[batch]` | `indices` INT32 `[K]`, `values` FP32 `[K]`, `row_offsets` INT64 `[batch+1]` (CSR, 0-based vocab ids) |
| `embedding` | `texts`, `text_lens` (as above) | `embedding` FP32 `[batch, 384]` (unit-norm) |
| `cross_encoder` | `query` UINT8 `[q_bytes]` (shared across all keys), `keys` UINT8 `[batch, max_bytes]`, `key_lens` INT32 `[batch]` | `logits` FP32 `[batch]` (raw), `prob` FP32 `[batch]` (sigmoid) |
| `sentiment` | `texts`, `text_lens` | `logits` FP32 `[batch, 2]` (raw), `probs` FP32 `[batch, 2]` (softmax), `label_id` INT32 `[batch]` (0 NEGATIVE, 1 POSITIVE) |

All inputs are raw UTF-8 bytes; the bundle tokenizes them. Note the two text encodings in one
table: the single-sequence models read one `(max_bytes, batch)` matrix of zero-padded rows plus
one length per row, while the cross encoder's `query` is a single unpadded byte vector shared by
every `(query, key)` pair.

## Sequence length and batching

Each bundle compiles a **single** sequence length of 512 and batch sizes `[1, 8]`, i.e. two
programs per bundle (`model.b1.mlir`, `model.b8.mlir`), all batch variants sharing one
`weights.safetensors`. Every request pads to 512, which is bit-identical to a tighter bucket
because every op is attention-mask aware; the tradeoff is wasted compute on short inputs in
exchange for far fewer programs and lower compile time and command-buffer count. The constant
lives in two places that must agree: `SEQ_LEN` in the export driver and `SEQ_BUCKETS = (512,)`
in each `model.jl`. To trade compute back for more programs, add sequence buckets or batch sizes
in the driver (`export_bundle` supports multiple input shapes over one weight set) and bump
`SEQ_BUCKETS`/`max_batch` to match. An over-long text fails in `preprocess` with a clear error
naming the encoded length and the largest compiled bucket, instead of a `BoundsError` while
filling the padded matrices.

## Running the example

`examples/transformers/` in the repository exports all four models, serves them, and queries them;
[Serve Transformer Text Models](tutorials/examples/transformers.md) walks through running it.

## See also

- [Serve Transformer Text Models](tutorials/examples/transformers.md) to run the example end to end
- [Bundles & model.jl](bundles.md) for the bundle contract, `IOSpec`, and the manifest encoding
- [Object Detection](object_detection.md) for exporting a detector's data-dependent middle as one
  static program
- [Client Usage](client.md) for building requests with `InferInput`/`InferOutput`
