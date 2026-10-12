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
what it needs (and they stop invalidating each other's precompilation): **export** holds the
export tooling with PythonCall and torch, **server** holds `ReactantServer`, and **client** holds
only `ReactantServerClient`, with neither Reactant nor torch. Each environment resolves independently the first time you use it:

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

### Export

The export writes one bundle per model to `examples/transformers/bundles/`, and skips any bundle
that already exists; delete `bundles/` to export again. The first run downloads the four
checkpoints from HuggingFace and, through CondaPkg, the Python packages the export uses. Behind a
proxy that intercepts TLS, set `SSL_CERT_FILE` to a CA bundle that trusts it; by default the
script uses the system bundle.

### Serve

The server compiles all four models before it accepts requests. If it fails with `EACCES` on
Reactant's compile cache, set `REACTANT_CACHE_DIR` to a writable directory.

### Client

The client sends raw UTF-8 text to all four models and prints what each returns: cosine similarity
between sentence embeddings, SPLADE's top expansion terms, the cross-encoder's ranking of
candidate answers, and sentiment labels with their probabilities. The
[Transformer Text Models](../../transformers.md) reference describes the wire format the client
uses.
