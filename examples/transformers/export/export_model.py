"""Torch-side builders for the transformers StableHLO export.

Each builder returns an inference-only nn.Module that IS the executable contract:
token-id tensors in, dense raw tensors out. Tokenization lives in each bundle's Julia
model.jl, and the classifier activations (cross encoder sigmoid, sentiment softmax) live
there too, so neither the tokenizer nor the final activation appears in the traced graph.

Two heads stay in-graph on purpose because they define the model's numeric output rather
than being a presentation-layer activation:
  - SPLADE's log1p/relu/mask/max, which also does the (batch, seq, vocab) -> (batch, vocab)
    reduction that makes the output a fixed-width term-score vector.
  - the embedding model's masked mean-pool + L2 normalization, which turns per-token hidden
    states into one unit-norm sentence vector.

All checkpoints are public HuggingFace ids, downloaded on first run.
"""
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

# Public HuggingFace checkpoints. All four share the bert-base-uncased WordPiece vocab
# (30522 tokens), so the same Julia tokenizer + vocab.txt serves every bundle.
SPLADE_ID = "naver/splade-cocondenser-ensembledistil"
EMBEDDING_ID = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SENTIMENT_ID = "distilbert-base-uncased-finetuned-sst-2-english"


class SpladeExport(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):        # (batch, seq) int64 each
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        relu_log = torch.log1p(torch.relu(logits))        # (batch, seq, vocab)
        weighted = relu_log * attention_mask.unsqueeze(-1).to(relu_log.dtype)
        return weighted.max(dim=1).values                 # (batch, vocab) term scores


class EmbedExport(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):        # (batch, seq) int64 each
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)      # (batch, seq, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)   # masked mean pool
        return F.normalize(pooled, p=2, dim=1)            # (batch, dim) unit-norm


class CrossExport(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):   # (batch, seq) int64
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).logits    # (batch, 1)
        return logits.squeeze(-1)                          # (batch,) raw logits


class SentimentExport(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):        # (batch, seq) int64 each
        # DistilBERT takes no token_type_ids.
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits  # (batch, 2)


def build_splade(model_id=SPLADE_ID):
    bert = AutoModelForMaskedLM.from_pretrained(model_id)
    bert.eval()
    return SpladeExport(bert).eval()


def build_embedding(model_id=EMBEDDING_ID):
    bert = AutoModel.from_pretrained(model_id)
    bert.eval()
    return EmbedExport(bert).eval()


def build_cross(model_id=CROSS_ID):
    bert = AutoModelForSequenceClassification.from_pretrained(model_id)
    bert.eval()
    return CrossExport(bert).eval()


def build_sentiment(model_id=SENTIMENT_ID):
    bert = AutoModelForSequenceClassification.from_pretrained(model_id)
    bert.eval()
    return SentimentExport(bert).eval()
