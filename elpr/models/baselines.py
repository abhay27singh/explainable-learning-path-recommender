"""Comparison models for Table I.

Each consumes the same batches, the same splits and the same supervision as KG-DKT,
so differences come from the architecture rather than from the data pipeline. The
common interface is ``forward(batch, adjacency) -> logits [B, L]``; baselines that do
not use the graph simply ignore the adjacency argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from elpr.models.layers import GCNEncoder, build_attention_mask


@dataclass
class BaselineConfig:
    n_concepts: int
    n_student_features: int
    hidden_dim: int = 128
    concept_dim: int = 64
    n_heads: int = 4
    dropout: float = 0.2


class DKT(nn.Module):
    """Deep Knowledge Tracing (Piech et al., 2015).

    A single LSTM over one-hot (concept, response) pairs. No graph, no timing, no
    student features — the original formulation, and the reference point every
    knowledge-tracing paper is expected to report.
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.n_concepts = config.n_concepts
        self.input = nn.Embedding(2 * config.n_concepts + 1, config.hidden_dim)
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.hidden_dim, config.n_concepts)

    def forward(self, batch: dict, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        concepts = batch["concept_ids"]
        # Interaction token: concept identity combined with its outcome. Unlabelled
        # context events get their own bucket rather than being dropped.
        supervised = batch["kinds"].bool()
        token = torch.where(
            supervised,
            concepts + self.n_concepts * batch["labels"].clamp(min=0),
            torch.full_like(concepts, 2 * self.n_concepts),
        )
        h, _ = self.lstm(self.input(token))
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        logits = self.out(self.dropout(h))
        return logits.gather(-1, concepts.unsqueeze(-1)).squeeze(-1)


class SAKT(nn.Module):
    """Self-Attentive Knowledge Tracing (Pandey & Karypis, 2019).

    Included because comparing a transformer against LSTM-DKT alone invites the
    obvious objection that the gain comes from attention rather than from the
    knowledge graph. SAKT isolates that: attention, no graph, no time awareness.
    """

    def __init__(self, config: BaselineConfig, max_len: int = 200):
        super().__init__()
        self.n_concepts = config.n_concepts
        self.interaction = nn.Embedding(2 * config.n_concepts + 1, config.hidden_dim)
        self.query = nn.Embedding(config.n_concepts, config.hidden_dim)
        self.position = nn.Embedding(max_len, config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.out = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: dict, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        concepts = batch["concept_ids"]
        b, length = concepts.shape
        supervised = batch["kinds"].bool()
        token = torch.where(
            supervised,
            concepts + self.n_concepts * batch["labels"].clamp(min=0),
            torch.full_like(concepts, 2 * self.n_concepts),
        )
        positions = torch.arange(length, device=concepts.device).clamp(
            max=self.position.num_embeddings - 1
        )
        keys = self.interaction(token) + self.position(positions)[None]
        keys = torch.cat([torch.zeros_like(keys[:, :1]), keys[:, :-1]], dim=1)
        queries = self.query(concepts)

        causal = torch.ones(length, length, dtype=torch.bool, device=concepts.device).triu(1)
        attended, _ = self.attn(
            queries, keys, keys,
            attn_mask=causal, key_padding_mask=batch["padding_mask"], need_weights=False,
        )
        h = self.norm1(queries + attended)
        h = self.norm2(h + self.ff(h))
        return self.out(h).squeeze(-1)


class GNNRecommender(nn.Module):
    """GCN concept embeddings fed to an LSTM.

    The graph-based comparison point: prerequisite structure is used, but there is no
    time-aware attention. Isolates the contribution of the temporal mechanism.
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.graph = GCNEncoder(config.n_concepts, config.concept_dim, 2, config.dropout)
        self.response = nn.Embedding(3, config.concept_dim)
        self.lstm = nn.LSTM(config.concept_dim, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.concept_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: dict, adjacency: torch.Tensor) -> torch.Tensor:
        z = self.graph(adjacency)
        x = z[batch["concept_ids"]] + self.response(batch["responses"])
        h, _ = self.lstm(x)
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return self.head(torch.cat([self.dropout(h), z[batch["concept_ids"]]], dim=-1)).squeeze(-1)


class MajorityBaseline(nn.Module):
    """Predicts the training base rate for everything.

    Proves the reported AUC is not an artefact of class imbalance: this model scores
    exactly 0.5 AUC by construction while still achieving high accuracy.
    """

    def __init__(self, config: BaselineConfig | None = None, base_rate: float = 0.679):
        super().__init__()
        self.register_buffer(
            "logit", torch.tensor(float(np.log(base_rate / (1 - base_rate))))
        )
        # The optimiser needs a parameter, and backward needs the loss to depend on one.
        # This is multiplied by zero, so it keeps the graph connected without ever
        # moving the prediction off the base rate.
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        constant = self.logit.expand_as(batch["concept_ids"].float())
        return constant + 0.0 * self.unused


BASELINES = {
    "dkt": DKT,
    "sakt": SAKT,
    "gnn": GNNRecommender,
    "majority": MajorityBaseline,
}
