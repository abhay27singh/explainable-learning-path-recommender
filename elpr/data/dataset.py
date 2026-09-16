"""PyTorch dataset over the per-enrolment interaction sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Re-exported so every existing import of these names keeps working.
from elpr.data.columns import (  # noqa: F401
    CATEGORICAL,
    NUMERIC,
    RESPONSE_CORRECT,
    RESPONSE_INCORRECT,
    RESPONSE_UNLABELLED,
)


def build_student_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot the categoricals, standardise the numerics, return [N, F]."""
    blocks, names = [], []
    for col, size in CATEGORICAL.items():
        codes = features[col].fillna(0).astype(int).to_numpy()
        onehot = np.zeros((len(features), size), dtype=np.float32)
        onehot[np.arange(len(features)), np.clip(codes, 0, size - 1)] = 1.0
        blocks.append(onehot)
        names += [f"{col}={i}" for i in range(size)]

    numeric = features[NUMERIC].fillna(0).to_numpy(dtype=np.float32)
    # Counts are long-tailed; log1p before standardising keeps a few very active
    # students from dominating the scale.
    numeric[:, 3:] = np.log1p(np.clip(numeric[:, 3:], 0, None))
    mean, std = numeric.mean(axis=0), numeric.std(axis=0)
    numeric = (numeric - mean) / np.where(std > 0, std, 1.0)
    blocks.append(numeric)
    names += NUMERIC

    return np.concatenate(blocks, axis=1).astype(np.float32), names


class SequenceDataset(Dataset):
    """Windows of interactions, each carrying at least one supervised token.

    Sequences longer than ``max_len`` are cut into consecutive non-overlapping windows.
    Windows with no assessment contribute nothing to the loss and are dropped.
    """

    KEY = ["id_student", "code_module", "code_presentation"]

    def __init__(
        self,
        sequences: pd.DataFrame,
        features: pd.DataFrame,
        max_len: int = 200,
    ):
        self.max_len = max_len
        self.student_matrix, self.feature_names = build_student_matrix(features)
        index = {tuple(r): i for i, r in enumerate(features[self.KEY].itertuples(index=False))}

        self.windows: list[tuple[int, int, int]] = []   # (row in sequences, start, end)
        self.rows: list[dict] = []
        self.feature_index: list[int] = []

        for row_i, row in enumerate(sequences.itertuples(index=False)):
            key = (row.id_student, row.code_module, row.code_presentation)
            if key not in index:
                continue
            n = int(row.n_events)
            self.rows.append(
                {
                    "concept_ids": np.asarray(row.concept_ids, dtype=np.int64),
                    "days": np.asarray(row.days, dtype=np.float32),
                    "is_supervised": np.asarray(row.is_supervised, dtype=np.int64),
                    "labels": np.asarray(row.labels, dtype=np.int64),
                    "clicks": np.asarray(row.clicks, dtype=np.float32),
                }
            )
            for start in range(0, n, max_len):
                end = min(start + max_len, n)
                if self.rows[-1]["is_supervised"][start:end].sum() > 0:
                    self.windows.append((len(self.rows) - 1, start, end))
                    self.feature_index.append(index[key])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        row_i, start, end = self.windows[i]
        row = self.rows[row_i]
        sl = slice(start, end)

        concepts = row["concept_ids"][sl]
        days = row["days"][sl]
        supervised = row["is_supervised"][sl]
        labels = row["labels"][sl]
        clicks = row["clicks"][sl]
        length = len(concepts)

        responses = np.where(
            supervised == 1,
            np.where(labels == 1, RESPONSE_CORRECT, RESPONSE_INCORRECT),
            RESPONSE_UNLABELLED,
        ).astype(np.int64)

        gap_prev = np.zeros(length, dtype=np.float32)
        gap_prev[1:] = np.clip(np.diff(days), 0, None)

        # Days since this same concept was last seen — the signal the forgetting
        # mechanism is meant to act on.
        gap_concept = np.zeros(length, dtype=np.float32)
        last_seen: dict[int, float] = {}
        for t, (c, d) in enumerate(zip(concepts, days)):
            gap_concept[t] = d - last_seen[c] if c in last_seen else 0.0
            last_seen[c] = d
        gap_concept = np.clip(gap_concept, 0, None)

        span = max(float(days[-1] - days[0]), 1.0)
        progress = (days - days[0]) / span

        numeric = np.stack(
            [np.log1p(np.clip(clicks, 0, None)),
             np.log1p(gap_prev),
             np.log1p(gap_concept),
             progress],
            axis=-1,
        ).astype(np.float32)

        return {
            "concept_ids": torch.from_numpy(concepts),
            "responses": torch.from_numpy(responses),
            "kinds": torch.from_numpy(supervised),
            "numeric": torch.from_numpy(numeric),
            "days": torch.from_numpy(days),
            "labels": torch.from_numpy(np.where(supervised == 1, labels, 0)),
            "supervised": torch.from_numpy(supervised.astype(np.float32)),
            "student_features": torch.from_numpy(self.student_matrix[self.feature_index[i]]),
        }


def collate(items: list[dict]) -> dict:
    """Right-pad to the longest window in the batch and build the pairwise time gap."""
    batch_size = len(items)
    length = max(item["concept_ids"].numel() for item in items)

    out = {
        "concept_ids": torch.zeros(batch_size, length, dtype=torch.long),
        "responses": torch.full((batch_size, length), RESPONSE_UNLABELLED, dtype=torch.long),
        "kinds": torch.zeros(batch_size, length, dtype=torch.long),
        "numeric": torch.zeros(batch_size, length, 4),
        "labels": torch.zeros(batch_size, length, dtype=torch.long),
        "supervised": torch.zeros(batch_size, length),
        "padding_mask": torch.ones(batch_size, length, dtype=torch.bool),
        "student_features": torch.stack([item["student_features"] for item in items]),
        # Day offsets only. The pairwise [B, L, L] gap is derived on the model's device:
        # materialising it here costs a Python loop on the dataloader's CPU plus a
        # ~10 MB host-to-device copy per step, which dominates on a 2-vCPU runtime.
        "days": torch.zeros(batch_size, length),
    }

    for i, item in enumerate(items):
        n = item["concept_ids"].numel()
        out["concept_ids"][i, :n] = item["concept_ids"]
        out["responses"][i, :n] = item["responses"]
        out["kinds"][i, :n] = item["kinds"]
        out["numeric"][i, :n] = item["numeric"]
        out["labels"][i, :n] = item["labels"]
        out["supervised"][i, :n] = item["supervised"]
        out["padding_mask"][i, :n] = False
        out["days"][i, :n] = item["days"]

    return out


def load_splits(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequences = pd.read_parquet(processed / "sequences.parquet")
    features = pd.read_parquet(processed / "student_features.parquet")
    return sequences, features
