# Transformer text models demo

Serving BERT-family text models with ReactantServer, end to end: trace four public
HuggingFace checkpoints into StableHLO bundles, serve them on a single GPU, and query them
over KServe V2. The four models cover the common text-serving shapes:

| Bundle | HuggingFace checkpoint | Task |
|---|---|---|
| `splade` | `naver/splade-cocondenser-ensembledistil` | learned sparse (SPLADE) term expansion |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` | dense sentence embedding (384-dim, unit-norm) |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | query/document relevance scoring |
| `sentiment` | `distilbert-base-uncased-finetuned-sst-2-english` | binary sentiment classification |

All four share the `bert-base-uncased` WordPiece vocab, so one tokenizer
(`tokenizer/bert_wordpiece.jl` + `tokenizer/vocab.txt`) serves every bundle. Tokenization
runs in Julia **inside** each bundle's `model.jl`, so clients send raw UTF-8 text bytes,
never token ids.

It is split into three single-purpose Julia environments so each loads only what it needs
(and they stop invalidating each other's precompilation): **export** is the only one with
PythonCall + torch, **server** is the only one with Reactant, and **client** has neither.

Each environment resolves independently the first time you use it:

```sh
for env in export server client; do
  julia --project=examples/transformers/$env -e 'using Pkg; Pkg.instantiate()'
done
```

Then run the three steps in order (the server stays running; drive it from a second terminal):

```sh
# 1. Export the bundles (first time only; writes ./bundles/). Needs network for the checkpoints.
julia --project=examples/transformers/export examples/transformers/export/export_stablehlo.jl

# 2. Serve on a single GPU (blocks; Ctrl-C to stop). Add --cpu for a GPU-free smoke test.
CUDA_VISIBLE_DEVICES=0 julia --project=examples/transformers/server examples/transformers/server/serve.jl

# 3. In another terminal: query all four models.
julia --project=examples/transformers/client examples/transformers/client/query.jl
```

The server port defaults to 8080; set `TX_PORT` (and `TX_HOST` for the client) to change it
on both step 2 and step 3.

## Wire formats

All inputs are raw UTF-8 bytes; the bundle tokenizes them. Shapes below are row-major
(KServe/client) order.

- **`splade`** in: `texts UINT8 [batch, max_bytes]` (zero-padded rows), `text_lens INT32 [batch]`.
  out (CSR, 0-based vocab ids): `indices INT32 [K]`, `values FP32 [K]`, `row_offsets INT64 [batch+1]`.
- **`embedding`** in: `texts`, `text_lens` (as above). out: `embedding FP32 [batch, 384]` (unit-norm).
- **`cross_encoder`** in: `query UINT8 [q_bytes]` (shared across all keys), `keys UINT8 [batch, max_bytes]`,
  `key_lens INT32 [batch]`. out: `logits FP32 [batch]` (raw), `prob FP32 [batch]` (sigmoid).
- **`sentiment`** in: `texts`, `text_lens`. out: `logits FP32 [batch, 2]` (raw), `probs FP32 [batch, 2]`
  (softmax), `label_id INT32 [batch]` (0 NEGATIVE, 1 POSITIVE).

Per the classifier rule the cross encoder and sentiment executables emit raw logits; their
`model.jl` adds the sigmoid/softmax so the client gets both. SPLADE's `log1p/relu/mask/max`
term-score head and the embedding model's masked mean-pool + L2 normalization stay in the
traced graph because they define the model's numeric output, not a presentation activation.

## Notes

- **Sequence length:** every request pads to a single compiled sequence length of 512 and the
  bundles compile batch sizes `[1, 8]`, so each bundle is two programs (`model.b1.mlir`,
  `model.b8.mlir`). Padding to 512 is bit-identical to a tighter bucket because every op is
  attention-mask aware; the tradeoff is wasted compute on short inputs for far fewer programs
  and lower compile time. `SEQ_BUCKETS` in each `model.jl` and `SEQ_LEN` in the export driver
  must agree.
- **Python deps (export only):** torch/torchax/jax come from `ReactantServerExport`'s CondaPkg
  and `transformers` from `export/CondaPkg.toml`; CondaPkg resolves and installs them on the
  first export (needs network).
- **First run downloads** the four checkpoints from HuggingFace (needs network).
- **Corporate SSL:** `export_stablehlo.jl` points Python's TLS at the OS CA bundle
  (`SSL_CERT_FILE`, defaulting from `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/`JULIA_SSL_CA_ROOTS_PATH`
  or `/etc/ssl/certs/ca-certificates.crt`) so the checkpoint downloads trust a MitM proxy's CA.
- **CPU smoke test cache dir:** running the export or `serve.jl --cpu` outside Docker can hit an
  `EACCES` on Reactant's default compile cache; set a writable `REACTANT_CACHE_DIR` if so.
- **This bundle contract** (manifest, `model.jl`, `client_inputs`/`client_outputs`) is documented
  in the [Bundles & model.jl](../../docs/src/manual/bundles.md) manual page; the walkthrough is
  [Transformer Text Models](../../docs/src/manual/transformers.md).
