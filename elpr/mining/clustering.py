"""Learner-profile clustering.

Groups students by how they *start* a module — session frequency, click volume,
regularity — using only the first 28 days, so a profile is available at prediction
time and cannot encode the outcome it will be used to predict.

The profiles serve three purposes: a model feature, a cold-start prior for students
with no interaction history, and an ingredient in explanations ("learners who start
the way you do typically...").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "early_sessions",
    "early_clicks",
    "early_clicks_per_session",
    "early_span",
    "early_mean_gap",
    "early_gap_std",
    "early_active_days",
]


@dataclass
class ProfileResult:
    labels: np.ndarray
    k: int
    silhouette: float
    stability: float
    scores: dict[int, float]
    centroids: pd.DataFrame


def fit_learner_profiles(
    features: pd.DataFrame,
    k_range: tuple[int, ...] = (3, 4, 5, 6),
    seed: int = 42,
    sample_for_selection: int = 8000,
) -> ProfileResult:
    """Select k by silhouette, then report stability across seeds.

    Silhouette is evaluated on a subsample because it is O(n^2) in memory and this
    machine has 8 GB; the final fit uses every row.
    """
    LOG_COLS = [FEATURES.index("early_clicks"), FEATURES.index("early_clicks_per_session")]

    x = features[FEATURES].to_numpy(dtype=np.float64)
    # Click counts are heavily skewed; log1p keeps a handful of very active students
    # from defining the entire clustering.
    x[:, LOG_COLS] = np.log1p(x[:, LOG_COLS])
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(x), size=sample_for_selection, replace=False)
        if len(x) > sample_for_selection
        else np.arange(len(x))
    )

    scores: dict[int, float] = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x[idx])
        scores[k] = float(silhouette_score(x[idx], labels))

    best_k = max(scores, key=scores.get)
    final = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(x)

    # Stability: refit under a different seed and compare assignments. A clustering
    # that moves under reseeding is not a finding.
    alt = KMeans(n_clusters=best_k, n_init=10, random_state=seed + 1).fit_predict(x)
    stability = float(adjusted_rand_score(final.labels_, alt))

    # Invert the exact pipeline that was fitted — unscale first, then undo the log on
    # the columns it was applied to. Fitting a fresh scaler on the raw features here
    # would invert a transform that was never applied, and yields impossible values
    # such as negative click counts.
    centers = scaler.inverse_transform(final.cluster_centers_)
    centers[:, LOG_COLS] = np.expm1(centers[:, LOG_COLS])
    centroids = pd.DataFrame(centers, columns=FEATURES)
    centroids.insert(0, "profile", range(best_k))
    centroids.insert(1, "n_enrolments", np.bincount(final.labels_, minlength=best_k))

    return ProfileResult(
        labels=final.labels_,
        k=best_k,
        silhouette=scores[best_k],
        stability=stability,
        scores=scores,
        centroids=centroids,
    )
