"""Mastery estimation and what-if simulation for a single learner.

Everything the planner and the explainer need reduces to two questions:

    what does this learner know now?
    what would change if they studied concept c next?

Both are answered by the frozen knowledge-tracing model. The second is asked once per
candidate concept, so it is evaluated as a single batched forward pass rather than one
call per candidate — otherwise a recommendation would take seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from elpr.data.dataset import (
    RESPONSE_CORRECT, RESPONSE_INCORRECT, RESPONSE_UNLABELLED,
)
from elpr.models.kgdkt import KGDKT, KGDKTConfig


@dataclass
class LearnerSequence:
    """One learner's interaction window, in the form the model consumes."""

    concept_ids: np.ndarray
    days: np.ndarray
    responses: np.ndarray
    kinds: np.ndarray
    clicks: np.ndarray
    student_features: np.ndarray

    def __len__(self) -> int:
        return len(self.concept_ids)


class MasteryEngine:
    def __init__(
        self,
        model: KGDKT,
        adjacency: torch.Tensor,
        max_len: int = 200,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.adjacency = adjacency.to(self.device)
        self.max_len = max_len
        self.n_concepts = adjacency.size(0)

    # -- batch construction -----------------------------------------------------
    def _build_batch(self, sequences: list[LearnerSequence]) -> dict:
        batch_size = len(sequences)
        length = max(len(s) for s in sequences)

        batch = {
            "concept_ids": torch.zeros(batch_size, length, dtype=torch.long),
            "responses": torch.full((batch_size, length), RESPONSE_UNLABELLED, dtype=torch.long),
            "kinds": torch.zeros(batch_size, length, dtype=torch.long),
            "numeric": torch.zeros(batch_size, length, 4),
            "days": torch.zeros(batch_size, length),
            "padding_mask": torch.ones(batch_size, length, dtype=torch.bool),
            "student_features": torch.zeros(batch_size, len(sequences[0].student_features)),
        }

        for i, seq in enumerate(sequences):
            n = len(seq)
            days = seq.days.astype(np.float32)
            batch["concept_ids"][i, :n] = torch.from_numpy(seq.concept_ids.astype(np.int64))
            batch["responses"][i, :n] = torch.from_numpy(seq.responses.astype(np.int64))
            batch["kinds"][i, :n] = torch.from_numpy(seq.kinds.astype(np.int64))
            batch["days"][i, :n] = torch.from_numpy(days)
            batch["padding_mask"][i, :n] = False
            batch["student_features"][i] = torch.from_numpy(
                seq.student_features.astype(np.float32)
            )

            gap_prev = np.zeros(n, dtype=np.float32)
            gap_prev[1:] = np.clip(np.diff(days), 0, None)
            gap_concept = np.zeros(n, dtype=np.float32)
            last_seen: dict[int, float] = {}
            for t, (c, d) in enumerate(zip(seq.concept_ids, days)):
                gap_concept[t] = d - last_seen[c] if c in last_seen else 0.0
                last_seen[int(c)] = d
            span = max(float(days[-1] - days[0]), 1.0)
            batch["numeric"][i, :n] = torch.from_numpy(
                np.stack([
                    np.log1p(np.clip(seq.clicks.astype(np.float32), 0, None)),
                    np.log1p(gap_prev),
                    np.log1p(np.clip(gap_concept, 0, None)),
                    (days - days[0]) / span,
                ], axis=-1).astype(np.float32)
            )

        return {k: v.to(self.device) for k, v in batch.items()}

    # -- queries ----------------------------------------------------------------
    @torch.no_grad()
    def mastery(self, sequence: LearnerSequence) -> np.ndarray:
        """Calibrated mastery over every concept: [C]."""
        batch = self._build_batch([sequence])
        return self.model.mastery(batch, self.adjacency)[0].cpu().numpy()

    def extend(
        self, sequence: LearnerSequence, concept: int, correct: bool, gap_days: float = 7.0
    ) -> LearnerSequence:
        """Append a hypothetical study interaction with ``concept``."""
        response = RESPONSE_CORRECT if correct else RESPONSE_INCORRECT
        keep = slice(1, None) if len(sequence) >= self.max_len else slice(0, None)
        return LearnerSequence(
            concept_ids=np.append(sequence.concept_ids[keep], concept),
            days=np.append(sequence.days[keep], sequence.days[-1] + gap_days),
            responses=np.append(sequence.responses[keep], response),
            kinds=np.append(sequence.kinds[keep], 1),
            clicks=np.append(sequence.clicks[keep], 0),
            student_features=sequence.student_features,
        )

    def idle(self, sequence: LearnerSequence, gap_days: float = 7.0) -> LearnerSequence:
        """The same elapsed time, with no study.

        This is the baseline any recommendation must be measured against. Appending an
        interaction also advances the clock, and the model's forgetting mechanism then
        decays every earlier interaction — a drop that is far larger than the gain from
        one concept. Comparing "studied, a week later" against "right now" therefore
        makes every option look harmful. Comparing against "a week later, having
        studied nothing" isolates the effect of the choice itself.

        Constructed to mirror ``extend`` exactly — same window shift, same elapsed time,
        same sequence length — differing only in that the appended event is unlabelled
        context rather than a graded attempt. Any other construction compares sequences
        of different shapes and the difference stops meaning anything.
        """
        keep = slice(1, None) if len(sequence) >= self.max_len else slice(0, None)
        return LearnerSequence(
            concept_ids=np.append(sequence.concept_ids[keep], sequence.concept_ids[-1]),
            days=np.append(sequence.days[keep], sequence.days[-1] + gap_days),
            responses=np.append(sequence.responses[keep], RESPONSE_UNLABELLED),
            kinds=np.append(sequence.kinds[keep], 0),
            clicks=np.append(sequence.clicks[keep], 0),
            student_features=sequence.student_features,
        )

    @torch.no_grad()
    def mastery_idle(self, sequence: LearnerSequence, gap_days: float = 7.0) -> np.ndarray:
        """Mastery after ``gap_days`` of no study: [C]."""
        return self.mastery(self.idle(sequence, gap_days))

    @torch.no_grad()
    def mastery_after(
        self, sequence: LearnerSequence, concepts: list[int], correct: bool = True
    ) -> np.ndarray:
        """Mastery after studying each candidate concept: [len(concepts), C].

        One batched forward pass over all candidates. Evaluating them one at a time
        would make a single recommendation take seconds on CPU.
        """
        if not concepts:
            return np.zeros((0, self.n_concepts), dtype=np.float32)
        extended = [self.extend(sequence, c, correct) for c in concepts]
        batch = self._build_batch(extended)
        return self.model.mastery(batch, self.adjacency).cpu().numpy()

    # -- loading ----------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, path: Path, adjacency: torch.Tensor, device: torch.device | None = None
    ) -> "MasteryEngine":
        blob = torch.load(path, map_location="cpu")
        config = KGDKTConfig(**blob["config"])
        model = KGDKT(config)
        model.load_state_dict(blob["state_dict"])
        return cls(model, adjacency, max_len=config.max_len, device=device)
