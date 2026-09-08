#!/usr/bin/env python
"""Stage 0, part 3: features, sequences, learner profiles, and cross-validation splits.

Runs sql/08, 09 and 11, fits learner profiles on early-window behaviour, and assigns
stratified group-aware folds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.duckdb_runner import SQLRunner  # noqa: E402
from elpr.mining.clustering import fit_learner_profiles  # noqa: E402

RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
N_FOLDS = 5
SEED = 42


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    runner = SQLRunner()

    print("Running SQL (sql/08, 09, 11):")
    runner.run_range(8, 11)

    features = runner.df("SELECT * FROM student_features")
    sequences = runner.df(
        "SELECT id_student, code_module, code_presentation, n_events, n_supervised "
        "FROM sequences"
    )

    print("\nFitting learner profiles (first 28 days only):")
    profiles = fit_learner_profiles(features, seed=SEED)
    features["learner_profile"] = profiles.labels
    print(f"  k selected           {profiles.k}   (silhouette by k: "
          + ", ".join(f"{k}={v:.3f}" for k, v in profiles.scores.items()) + ")")
    print(f"  silhouette           {profiles.silhouette:.3f}")
    print(f"  stability (ARI)      {profiles.stability:.3f}")

    # Folds are assigned over enrolments but grouped by student: a student taking two
    # modules must not appear in both train and test, or the model sees them twice.
    trainable = features.merge(
        sequences[["id_student", "code_module", "code_presentation"]],
        on=["id_student", "code_module", "code_presentation"],
        how="inner",
    ).reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trainable["fold"] = -1
    for fold, (_, test_idx) in enumerate(
        splitter.split(trainable, trainable.final_result, groups=trainable.id_student)
    ):
        trainable.loc[test_idx, "fold"] = fold
    assert (trainable.fold >= 0).all(), "every enrolment must be assigned a fold"

    runner.con.register("trainable_df", trainable)
    runner.con.execute("CREATE OR REPLACE TABLE splits AS SELECT * FROM trainable_df")

    trainable.to_parquet(PROCESSED / "student_features.parquet")
    runner.df("SELECT * FROM sequences").to_parquet(PROCESSED / "sequences.parquet")

    seq_stats = runner.df("SELECT * FROM sequence_stats").to_dict("records")[0]
    fold_table = (
        trainable.groupby(["fold", "final_result"]).size().unstack(fill_value=0)
    )
    leak = trainable.groupby("id_student").fold.nunique().max()

    stats = {
        "sequences": seq_stats,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_distribution": fold_table.to_dict(),
        "max_folds_per_student": int(leak),
        "learner_profiles": {
            "k": profiles.k,
            "silhouette": round(profiles.silhouette, 4),
            "silhouette_by_k": {str(k): round(v, 4) for k, v in profiles.scores.items()},
            "stability_ari": round(profiles.stability, 4),
            "centroids": profiles.centroids.round(2).to_dict("records"),
        },
    }
    (RESULTS / "sequence_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    print("\nSequences:")
    for k, v in seq_stats.items():
        print(f"  {k:<24} {v:,.0f}" if isinstance(v, (int, float)) else f"  {k:<24} {v}")

    print("\nFold sizes by outcome:")
    print(fold_table.to_string())
    print(f"\n  students spanning >1 fold: {leak - 1 if leak > 1 else 0}")

    print("\nLearner profiles:")
    print(profiles.centroids.round(2).to_string(index=False))

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
