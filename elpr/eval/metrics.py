"""Metrics for knowledge tracing, including the calibration the paper lacks."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def auc_rmse(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float(np.sqrt(np.mean((y_prob - y_true) ** 2)))
    return (
        float(roc_auc_score(y_true, y_prob)),
        float(np.sqrt(np.mean((y_prob - y_true) ** 2))),
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Mean gap between confidence and accuracy, weighted by bin population.

    Mastery values are shown to students inside explanations, so a probability that
    does not mean what it says is a correctness problem, not a cosmetic one.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(y_prob, edges[1:-1])
    error = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(y_prob, edges[1:-1])
    conf, acc, count = [], [], []
    for b in range(n_bins):
        mask = bins == b
        conf.append(y_prob[mask].mean() if mask.any() else np.nan)
        acc.append(y_true[mask].mean() if mask.any() else np.nan)
        count.append(int(mask.sum()))
    return np.array(conf), np.array(acc), np.array(count)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
    """Single-parameter temperature scaling, fitted by LBFGS on held-out logits."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(y_true, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z / log_t.exp(), y
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def cold_start_auc(
    y_true: np.ndarray, y_prob: np.ndarray, n_prior: np.ndarray,
    buckets: tuple[int, ...] = (0, 1, 5, 10, 20),
) -> dict[str, float]:
    """AUC as a function of how much history a learner had at prediction time.

    The paper criticises cold start in prior systems and never reports its own.
    """
    out: dict[str, float] = {}
    edges = list(buckets) + [10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (n_prior >= lo) & (n_prior < hi)
        label = f"{lo}" if hi == 10**9 else f"{lo}-{hi - 1}"
        if mask.sum() > 50 and len(np.unique(y_true[mask])) > 1:
            out[label] = float(roc_auc_score(y_true[mask], y_prob[mask]))
        else:
            out[label] = float("nan")
    return out
