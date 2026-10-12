# Transformer text models demo

Serving BERT-family text models with ReactantServer, end to end: trace four public HuggingFace
checkpoints into StableHLO bundles, serve them, and query them over KServe V2 with raw text. It
runs on the CPU or a GPU.

| Bundle | HuggingFace checkpoint | Task |
|---|---|---|
| `splade` | `naver/splade-cocondenser-ensembledistil` | learned sparse (SPLADE) term expansion |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` | dense sentence embedding (384-dim, unit-norm) |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | query/document relevance scoring |
| `sentiment` | `distilbert-base-uncased-finetuned-sst-2-english` | binary sentiment classification |

Tokenization runs in Julia inside each bundle's `model.jl`, so clients send raw UTF-8 bytes,
never token ids. The example is three single-purpose Julia environments: `export/` holds the
export tooling with PythonCall and torch, `server/` holds the server, and `client/` holds only the
client, with neither Reactant nor torch.

- [Serve Transformer Text Models](https://enzymead.github.io/ReactantServer.jl/dev/tutorials/examples/transformers/) walks through running it.
- [Transformer Text Models](https://enzymead.github.io/ReactantServer.jl/dev/transformers/) is the reference: the four models, tokenization in
  the bundle, the raw-logits rule, the wire contract, and sequence-length batching.
