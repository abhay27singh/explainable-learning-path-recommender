"""Path-quality metrics for Table II.

The paper reports completion rate, NDCG and average knowledge gain without defining
any of them. Definitions here, and the distinction that matters most:

    NDCG and hit rate are measured against WHAT LEARNERS ACTUALLY DID NEXT.
    Completion rate and knowledge gain are measured IN SIMULATION.

Only the first pair is independent of the model being evaluated. The second pair uses
the frozen tracer as an environment, so a planner optimised against that tracer is
partly grading its own work. Reporting them together, clearly labelled, is the honest
treatment — and agreement between the two is then genuine evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dcg(relevances: list[float]) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(relevances)))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 5) -> float:
    """Normalised discounted cumulative gain against the learner's actual next steps."""
    if not relevant:
        return float("nan")
    gains = [1.0 if c in relevant else 0.0 for c in recommended[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator > 0 else float("nan")


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int = 5) -> float:
    if not relevant:
        return float("nan")
    return float(any(c in relevant for c in recommended[:k]))


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 5) -> float:
    if not relevant or not recommended:
        return float("nan")
    top = recommended[:k]
    return float(sum(c in relevant for c in top) / len(top))


@dataclass
class PathMetrics:
    planner: str
    ndcg: float
    ndcg_std: float
    hit_rate: float
    precision: float
    knowledge_gain: float
    prerequisite_violations: float
    coverage: float
    n_decisions: int

    def row(self) -> dict:
        return {
            "planner": self.planner,
            "ndcg": round(self.ndcg, 4),
            "ndcg_std": round(self.ndcg_std, 4),
            "hit_rate": round(self.hit_rate, 4),
            "precision": round(self.precision, 4),
            "knowledge_gain": round(self.knowledge_gain, 4),
            "prerequisite_violations": round(self.prerequisite_violations, 4),
            "coverage": round(self.coverage, 4),
            "n_decisions": self.n_decisions,
        }


def prerequisite_violation_rate(
    graph, mastery: np.ndarray, recommended: list[int], threshold: float
) -> float:
    """Fraction of recommendations made past an unmet prerequisite.

    Should be exactly zero for every masked planner. It is reported rather than
    assumed, because a pedagogical guarantee that is never checked is not a guarantee.
    """
    if not recommended:
        return float("nan")
    violations = 0
    for concept in recommended:
        prereqs = graph.prereqs(int(concept))
        if prereqs and np.any(mastery[prereqs] < threshold):
            violations += 1
    return violations / len(recommended)
