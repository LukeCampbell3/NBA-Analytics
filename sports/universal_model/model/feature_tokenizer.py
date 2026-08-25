"""Typed feature tokenizer (spec section 6) -- NOT an NLP tokenizer.

Turns one batch of the fixed tensors produced by ``data/dataset.py`` into a
fixed-length sequence of ``N_TOKENS`` typed token embeddings per
observation: [SPORT] [ENTITY] [ROLE] [OPPORTUNITY] [TEMPORAL] [MARKET]
[UNCERTAINTY] [TARGET]. Numeric values get a learned projection conditioned
on which semantic family they belong to; categorical values get learned
embeddings; missingness gets an explicit additive mask embedding per spec
section 6 ("explicit missing-value mask/embedding").

This is deterministic given fixed vocab sizes (spec section 55.J,
tokenization determinism): the same input tensor always produces the same
token ids fed to the same embedding tables -- there is no randomness in
tokenization itself (only in model weight init/dropout).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
ENTITY_HASH_BUCKETS = 8192
NUMERIC_DIM = 5  # matches data/normalization.py NUMERIC_UNIVERSAL_COLUMNS

TOKEN_CLASSES = ["SPORT", "ENTITY", "ROLE", "OPPORTUNITY", "TEMPORAL", "MARKET", "UNCERTAINTY", "TARGET"]
# index into the 5-dim numeric vector: [line, american_price, sample_support_rows, days_since_last_history, recency_prior]
_NUMERIC_IDX = {"line": 0, "american_price": 1, "sample_support_rows": 2, "days_since_last_history": 3, "recency_prior": 4}


class FeatureTokenizer(nn.Module):
    def __init__(self, hidden_dim: int = 384, normalization_path: Path = MANIFESTS_DIR / "normalization_manifest.json"):
        super().__init__()
        norm = json.loads(normalization_path.read_text())
        vocabs = norm["vocabs"]
        self.hidden_dim = hidden_dim

        self.sport_emb = nn.Embedding(len(vocabs["sport"]) + 1, hidden_dim)
        self.entity_emb = nn.Embedding(ENTITY_HASH_BUCKETS, hidden_dim)
        self.role_emb = nn.Embedding(len(vocabs["role"]) + 2, hidden_dim)  # +1 unknown, +1 pad-shift
        self.home_emb = nn.Embedding(len(vocabs["home_away"]) + 2, hidden_dim)
        self.target_emb = nn.Embedding(max(len(vocabs["target"]), 1), hidden_dim)

        # Numeric projections, one per semantic family token (each sees the
        # full 5-dim numeric+missing vector but is conditioned on family
        # identity via a distinct learned projection -- spec section 6:
        # "learned numeric projection conditioned on feature identity /
        # semantic family").
        self.opportunity_proj = nn.Linear(NUMERIC_DIM * 2, hidden_dim)
        self.temporal_proj = nn.Linear(NUMERIC_DIM * 2, hidden_dim)
        self.market_proj = nn.Linear(NUMERIC_DIM * 2, hidden_dim)
        self.uncertainty_proj = nn.Linear(NUMERIC_DIM * 2, hidden_dim)

        self.missing_mask_emb = nn.Embedding(2, hidden_dim)  # additive, per-token "any missing in this family" flag
        self.token_type_emb = nn.Embedding(len(TOKEN_CLASSES), hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        numeric = batch["numeric"]  # (B, 5)
        missing = batch["missing"]  # (B, 5)
        numeric_and_mask = torch.cat([numeric, missing], dim=-1)  # (B, 10)
        B = numeric.shape[0]
        device = numeric.device

        sport_tok = self.sport_emb(batch["sport_id"])
        entity_tok = self.entity_emb(batch["entity_bucket"])
        role_tok = self.role_emb(batch["role_id"]) + self.home_emb(batch["home_id"])
        opportunity_tok = self.opportunity_proj(numeric_and_mask)
        temporal_tok = self.temporal_proj(numeric_and_mask)
        market_tok = self.market_proj(numeric_and_mask)
        market_missing_any = (missing[:, [_NUMERIC_IDX["line"], _NUMERIC_IDX["american_price"]]].sum(-1) > 0).long()
        market_tok = market_tok + self.missing_mask_emb(market_missing_any)
        uncertainty_tok = self.uncertainty_proj(numeric_and_mask)
        target_tok = self.target_emb(batch["target_id"])

        tokens = torch.stack(
            [sport_tok, entity_tok, role_tok, opportunity_tok, temporal_tok, market_tok, uncertainty_tok, target_tok],
            dim=1,
        )  # (B, 8, hidden_dim)
        type_ids = torch.arange(len(TOKEN_CLASSES), device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.token_type_emb(type_ids)
        return tokens
