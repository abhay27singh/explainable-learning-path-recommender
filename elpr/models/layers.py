"""Building blocks for KG-DKT: the graph encoder and time-aware attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNEncoder(nn.Module):
    """Graph convolution over the concept prerequisite graph.

    Three layers, LayerNorm + ReLU, 64-dimensional output — the configuration the
    paper states. The adjacency is dense: at 237 concepts a 237x237 matrix costs
    nothing and avoids sparse-operator gaps across CPU, MPS and CUDA.

    Concept embeddings are recomputed on every forward pass, so gradients flow into
    the graph structure. That is what makes the '-knowledge graph' ablation (swap this
    for a free embedding table) a real experiment rather than a formality.
    """

    def __init__(self, n_concepts: int, dim: int = 64, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_concepts, dim)
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.embedding.weight
        for linear, norm in zip(self.layers, self.norms):
            h = adjacency @ linear(h)
            h = self.dropout(F.relu(norm(h)))
        return h


class TimeAwareAttention(nn.Module):
    """Multi-head self-attention with a learned decay over elapsed time.

    The paper claims time-aware attention and forgetting effects but never specifies
    them. This is the mechanism:

        bias[h, i, j] = -softplus(gamma_h) * log1p(days_between(i, j))

    Older interactions are attenuated, at a rate the model learns separately per head,
    so one head can attend to recent work while another keeps a longer memory.
    Setting gamma to zero recovers ordinary attention, which is exactly the
    '-temporal attention' ablation.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.2, time_aware: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.time_aware = time_aware
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(n_heads)) if time_aware else None

    def forward(
        self,
        x: torch.Tensor,             # [B, L, D]
        log_time_gap: torch.Tensor,  # [B, 1, L, L] log1p of elapsed days, precomputed
        base_mask: torch.Tensor,     # [B, 1, L, L] additive float mask, causal + padding
    ) -> torch.Tensor:
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each [B, H, L, Dh]

        if self.time_aware:
            decay = F.softplus(self.gamma).view(1, self.n_heads, 1, 1)
            attn_mask = base_mask - decay * log_time_gap
        else:
            attn_mask = base_mask.expand(b, self.n_heads, length, length)

        # Fused attention: avoids materialising the softmax intermediate, which is the
        # dominant cost at L=200 on CPU.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, length, dim)
        return self.proj(out)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block wrapping the time-aware attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.2, time_aware: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TimeAwareAttention(dim, n_heads, dropout, time_aware)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, log_time_gap, base_mask):
        x = x + self.dropout(self.attn(self.norm1(x), log_time_gap, base_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


def build_attention_mask(
    padding_mask: torch.Tensor, length: int
) -> torch.Tensor:
    """Additive float mask combining causality and padding: [B, 1, L, L].

    Built once per forward pass rather than once per layer. Rows that would be fully
    masked (position 0 sees nothing before it) keep their diagonal open, so softmax
    never sees an all -inf row and cannot produce NaN.
    """
    device = padding_mask.device
    causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
    blocked = causal[None, :, :] | padding_mask[:, None, :]
    blocked = blocked & ~torch.eye(length, dtype=torch.bool, device=device)[None]
    return torch.zeros_like(blocked, dtype=torch.float32).masked_fill_(blocked, -1e9).unsqueeze(1)
