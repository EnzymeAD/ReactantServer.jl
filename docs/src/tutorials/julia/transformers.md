# Serve Transformer Text Models

This tutorial exports four public HuggingFace BERT-family checkpoints (a SPLADE sparse encoder, a
sentence embedder, a cross-encoder reranker, and a sentiment classifier) into ReactantServer
bundles, serves all four, and queries them with raw text. It runs on the CPU, so it needs no GPU,
and the same steps serve on a GPU by dropping one flag. Why tokenization lives in each bundle's
`model.jl`, and the wire contract the client follows, are covered in the
[Transformer Text Models](../../transformers.md) reference.

You need a clone of the repository (the example's scripts live in `examples/transformers/`) and
network access for the first export, which downloads the checkpoints and the Python packages the
PyTorch export uses.

`examples/transformers/` is split into three single-purpose Julia environments so each loads only
what it needs (and they stop invalidating each other's precompilation): **export** is the only
one with PythonCall + torch, **server** is the only one with Reactant, and **client** has
neither. Each environment resolves independently the first time you use it:

```sh
for env in export server client; do
  julia --project=examples/transformers/$env -e 'using Pkg; Pkg.instantiate()'
done
```

Then run the three steps in order (the server stays running; drive it from a second terminal):

```sh
# 1. Export the bundles (first time only; writes ./bundles/). Needs network for the checkpoints.
julia --project=examples/transformers/export examples/transformers/export/export_stablehlo.jl

# 2. Serve them on the CPU (blocks; Ctrl-C to stop). On a GPU host, drop --cpu and pick a device
#    with CUDA_VISIBLE_DEVICES=0.
julia --project=examples/transformers/server examples/transformers/server/serve.jl --cpu

# 3. In another terminal: query all four models.
julia --project=examples/transformers/client examples/transformers/client/query.jl
```

The server port defaults to 8080; set `TX_PORT` (and `TX_HOST` for the client) to change it on
both step 2 and step 3.

### Export (`export_stablehlo.jl`)

The driver imports torch/torchax through PythonCall **before** `ReactantServerExport` pulls in
Reactant; that init order is critical, because Triton's static LLVM/MLIR registration
SIGSEGVs if Reactant loads first. Each of the four builders wraps its checkpoint (torch class
per the table in the [Transformer Text Models](../../transformers.md) reference, plus the in-graph head for `splade`/`embedding`), traces it with
`export_bundle(:pytorch, ...)`, and stages the per-bundle `model.jl` plus the shared
`vocab.txt` into `./bundles/<name>/` (the tokenizer code itself lives in
`ReactantServer.BertText`, so it is not staged). Bundles that already exist are skipped; delete
`./bundles/` to re-export.

Python dependencies come from `ReactantServerExport`'s CondaPkg (torch, torchax, jax) plus
`transformers` from `export/CondaPkg.toml`; CondaPkg resolves and installs them on the first
export, and the first run downloads the four checkpoints from HuggingFace, so both need network.
For corporate proxies, the driver points Python's TLS at the OS CA bundle (`SSL_CERT_FILE`,
defaulting from `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/`JULIA_SSL_CA_ROOTS_PATH` or
`/etc/ssl/certs/ca-certificates.crt`) so the checkpoint downloads trust a MitM proxy's CA.

### Serve (`serve.jl`)

The server validates that all four bundles exist under `examples/transformers/bundles`, picks
`ReactantServer.CPU_BACKEND` or `CUDA_BACKEND` from `--cpu`, and builds a
`ReactantServer.ServerConfig` (runtime, scheduler, and endpoints on `127.0.0.1:$TX_PORT`) before
calling `ReactantServer.serve(...; backend = ReactantServer.ReactantBackend())`, which blocks
until Ctrl-C. Running the export or `serve.jl --cpu` outside Docker can hit an `EACCES` on
Reactant's default compile cache; set a writable `REACTANT_CACHE_DIR` if so.

### Client (`query.jl`)

The client depends only on `ReactantServerClient` (no Reactant, no PythonCall), so it loads
fast. It packs strings into the wire tensors itself (`pack_texts`: a `(max_bytes, batch)` UInt8
matrix plus an Int32 byte-length vector) and sends raw UTF-8 to all four models, then decodes
each response: cosine similarity between unit-norm embeddings, SPLADE's top expansion terms
turned back into tokens via the shared vocab, cross-encoder reranking of candidate answers, and
sentiment labels (0 NEGATIVE, 1 POSITIVE) with their probabilities.
