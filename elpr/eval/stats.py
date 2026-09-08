"""Statistical comparison for Table I.

The paper promises paired t-tests at p < 0.05 and reports no p-values. It also
compares one proposed model against several baselines on several metrics without any
multiple-comparison correction, and reports no effect sizes. All three are standard
reviewer objections; all three are cheap to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Comparison:
    name: str
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_raw: float
    p_holm: float
    cohens_d: float
    significant: bool


def paired_comparison(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """Paired t-test plus Cohen's d for repeated measures, across folds."""
    difference = a - b
    t_statistic, p_value = stats.ttest_rel(a, b)
    # Cohen's d for paired samples: mean difference over its own standard deviation.
    spread = difference.std(ddof=1)
    d = float(difference.mean() / spread) if spread > 0 else 0.0
    return {
        "name": name,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "difference": float(difference.mean()),
        "t_statistic": float(t_statistic),
        "p_raw": float(p_value),
        "cohens_d": d,
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down correction.

    Comparing the proposed model against five baselines on the same folds is five
    dependent tests. Uncorrected, the chance of at least one false positive at
    alpha=0.05 is far above 5%. Holm controls the family-wise error rate while being
    uniformly more powerful than plain Bonferroni.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running_max = max(running_max, value)   # enforce monotonicity
        adjusted[index] = running_max

    return adjusted.tolist(), [p <= alpha for p in adjusted]


def compare_against_reference(
    fold_scores: dict[str, np.ndarray], reference: str, alpha: float = 0.05
) -> list[Comparison]:
    """Compare every model against the reference, correcting across the family."""
    others = [name for name in fold_scores if name != reference]
    raw = [
        paired_comparison(fold_scores[reference], fold_scores[name], name)
        for name in others
    ]
    adjusted, significant = holm_bonferroni([r["p_raw"] for r in raw], alpha)

    return [
        Comparison(
            name=r["name"],
            mean_a=r["mean_a"],
            mean_b=r["mean_b"],
            difference=r["difference"],
            t_statistic=r["t_statistic"],
            p_raw=r["p_raw"],
            p_holm=p_adj,
            cohens_d=r["cohens_d"],
            significant=sig,
        )
        for r, p_adj, sig in zip(raw, adjusted, significant)
    ]


def interpret_d(d: float) -> str:
    """Conventional labels, so a large p-value with a tiny effect is not oversold."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"
