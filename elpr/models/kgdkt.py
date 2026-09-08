"""KG-DKT: knowledge-graph-conditioned, time-aware knowledge tracing.

Configuration follows the paper exactly — 2 transformer layers, hidden 128, 4 heads,
max sequence length 200, a 3-layer GCN producing 64-dimensional concept embeddings.

The prediction convention avoids label leakage: the hidden state at position t is
built from interactions 0..t only, and is used together with the *identity* of concept
t+1 (never its response) to predict the outcome at t+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from elpr.models.layers import EncoderLayer, GCNEncoder, build_attention_mask


@dataclass
class KGDKTConfig:
    n_concepts: int
    n_student_features: int
    concept_dim: int = 64
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    n_gcn_layers: int = 3
    max_len: int = 200
    dropout: float = 0.2
    # Ablation switches. Both default to the full model.
    use_graph: bool = True      # False -> free embedding table, no GCN
    use_time: bool = True       # False -> gamma fixed at 0, no elapsed-time features


class KGDKT(nn.Module):
    def __init__(self, config: KGDKTConfig):
        super().__init__()
        self.config = config

        if config.use_graph:
            self.graph_encoder = GCNEncoder(
                config.n_concepts, config.concept_dim, config.n_gcn_layers, config.dropout
            )
            self.free_embedding = None
        else:
            # Ablation: same parameter budget, no propagation over prerequisites.
            self.graph_encoder = None
            self.free_embedding = nn.Embedding(config.n_concepts, config.concept_dim)
            nn.init.normal_(self.free_embedding.weight, std=0.02)

        self.response_embedding = nn.Embedding(3, config.hidden_dim)   # correct / incorrect / unlabelled
        self.kind_embedding = nn.Embedding(2, config.hidden_dim)       # vle context / assessment
        self.concept_proj = nn.Linear(config.concept_dim, config.hidden_dim)
        # log1p(clicks), days since previous event, days since this concept was last seen,
        # position through the presentation
        self.numeric_proj = nn.Linear(4, config.hidden_dim)
        self.student_proj = nn.Linear(config.n_student_features, config.hidden_dim)

        self.layers = nn.ModuleList(
            EncoderLayer(config.hidden_dim, config.n_heads, config.dropout, config.use_time)
            for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.concept_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    # -- concept embeddings -----------------------------------------------------
    def concept_embeddings(self, adjacency: torch.Tensor) -> torch.Tensor:
        if self.graph_encoder is not None:
            return self.graph_encoder(adjacency)
        return self.free_embedding.weight

    # -- encoding ---------------------------------------------------------------
    def encode(self, batch: dict, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.concept_embeddings(adjacency)                 # [C, concept_dim]
        concepts = batch["concept_ids"]                        # [B, L]

        x = self.concept_proj(z[concepts])
        x = x + self.response_embedding(batch["responses"])
        x = x + self.kind_embedding(batch["kinds"])

        numeric = batch["numeric"]
        if not self.config.use_time:
            # Ablation: strip both elapsed-time channels, keep clicks and position.
            numeric = numeric.clone()
            numeric[..., 1:3] = 0.0
        x = x + self.numeric_proj(numeric)

        # Student context is added to every position rather than prepended, so that
        # positions keep a one-to-one correspondence with interactions.
        x = x + self.student_proj(batch["student_features"]).unsqueeze(1)

        # Built once per forward pass, not once per layer: at L=200 these are the
        # dominant allocations on CPU.
        padding = batch["padding_mask"]
        base_mask = build_attention_mask(padding, concepts.size(1))
        if self.config.use_time:
            days = batch["days"]
            gap = (days.unsqueeze(-1) - days.unsqueeze(-2)).clamp(min=0)
            log_time_gap = torch.log1p(gap).unsqueeze(1)
        else:
            log_time_gap = torch.zeros_like(base_mask)

        for layer in self.layers:
            x = layer(x, log_time_gap, base_mask)
        return self.norm(x), z

    # -- prediction -------------------------------------------------------------
    def forward(self, batch: dict, adjacency: torch.Tensor) -> torch.Tensor:
        """Logits for the outcome at each position, given everything strictly before it.

        Returns [B, L]: entry t is the prediction for interaction t, conditioned on
        history 0..t-1 plus the identity of concept t.
        """
        h, z = self.encode(batch, adjacency)
        # Shift: position t is predicted from the state after t-1.
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        query = z[batch["concept_ids"]]                        # [B, L, concept_dim]
        return self.head(torch.cat([h_prev, query], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def mastery(self, batch: dict, adjacency: torch.Tensor) -> torch.Tensor:
        """Mastery over every concept at the final position of each sequence: [B, C].

        This is the state the planner consumes, so it must cover all concepts, not
        only those the student has met.
        """
        h, z = self.encode(batch, adjacency)
        lengths = (~batch["padding_mask"]).sum(dim=1).clamp(min=1) - 1
        last = h[torch.arange(h.size(0), device=h.device), lengths]    # [B, D]
        b, c = last.size(0), z.size(0)
        pair = torch.cat(
            [last.unsqueeze(1).expand(b, c, last.size(-1)), z.unsqueeze(0).expand(b, c, z.size(-1))],
            dim=-1,
        )
        return torch.sigmoid(self.head(pair).squeeze(-1))
