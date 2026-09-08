#!/usr/bin/env python
"""Stage 2: Table II — learning path quality.

Evaluation protocol
-------------------
For each held-out learner, truncate their history at several points and ask each
planner what to recommend next. Ground truth is what the learner ACTUALLY engaged with
over the following interactions — real behaviour, not a simulation.

Only learners who passed or achieved distinction contribute ground truth. A failing
learner's next step is not evidence of a good next step.

    python scripts/09_eval_paths.py --fold 0 --learners 150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.data.dataset import (  # noqa: E402
    RESPONSE_CORRECT, RESPONSE_INCORRECT, RESPONSE_UNLABELLED,
    SequenceDataset, build_student_matrix,
)
from elpr.eval.path import (  # noqa: E402
    PathMetrics, hit_rate_at_k, ndcg_at_k, precision_at_k, prerequisite_violation_rate,
)
from elpr.graph.concept_graph import ConceptGraph  # noqa: E402
from elpr.planner.baselines import (  # noqa: E402
    CurriculumPlanner, PopularityPlanner, RandomPlanner, WeakestFirstPlanner,
)
from elpr.planner.engine import LearnerSequence, MasteryEngine  # noqa: E402
from elpr.planner.greedy import GreedyPlanner  # noqa: E402
from elpr.planner.state import State, enrolment_mask  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

SUCCESS = {"Pass", "Distinction"}


def make_sequence(row, features_row, matrix, index, upto: int) -> LearnerSequence:
    concepts = np.asarray(row.concept_ids)[:upto]
    supervised = np.asarray(row.is_supervised)[:upto]
    labels = np.asarray(row.labels)[:upto]
    days = np.asarray(row.days, dtype=np.float32)[:upto]
    clicks = np.asarray(row.clicks, dtype=np.float32)[:upto]
    responses = np.where(
        supervised == 1,
        np.where(labels == 1, RESPONSE_CORRECT, RESPONSE_INCORRECT),
        RESPONSE_UNLABELLED,
    )
    window = slice(max(0, len(concepts) - 200), len(concepts))
    key = (row.id_student, row.code_module, row.code_presentation)
    return LearnerSequence(
        concept_ids=concepts[window], days=days[window], responses=responses[window],
        kinds=supervised[window], clicks=clicks[window],
        student_features=matrix[index[key]],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learners", type=int, default=150)
    parser.add_argument("--cuts", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=25, help="events counted as ground truth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = ConceptGraph.load(ARTIFACTS / "graph")
    adjacency = torch.load(ARTIFACTS / "graph" / "adjacency.pt")["prerequisite"]
    sequences = pd.read_parquet(PROCESSED / "sequences.parquet")
    features = pd.read_parquet(PROCESSED / "student_features.parquet")
    matrix, _ = build_student_matrix(features)
    index = {
        tuple(r): i
        for i, r in enumerate(features[SequenceDataset.KEY].itertuples(index=False))
    }

    checkpoint = ARTIFACTS / "models" / f"kgdkt_proposed_fold{args.fold}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing {checkpoint}")
    engine = MasteryEngine.from_checkpoint(checkpoint, adjacency)

    # Held-out learners only, and only those whose trajectory is worth imitating.
    held_out = features[(features.fold == args.fold) & (features.final_result.isin(SUCCESS))]
    pool = sequences.merge(
        held_out[SequenceDataset.KEY], on=SequenceDataset.KEY, how="inner"
    )
    pool = pool[pool.n_events >= 60]
    pool = pool.sample(min(args.learners, len(pool)), random_state=args.seed)
    print(f"{len(pool)} held-out successful learners, fold {args.fold}")

    # Popularity is computed from TRAINING learners only, or it leaks the test set.
    train_keys = features[(features.fold != args.fold) & (features.final_result.isin(SUCCESS))]
    train_seqs = sequences.merge(train_keys[SequenceDataset.KEY], on=SequenceDataset.KEY)
    popularity = np.zeros(graph.n_concepts, dtype=np.float64)
    for row in train_seqs.itertuples(index=False):
        np.add.at(popularity, np.unique(np.asarray(row.concept_ids)), 1.0)
    popularity /= max(popularity.max(), 1.0)

    planners = {
        "random": RandomPlanner(graph, seed=args.seed),
        "curriculum": CurriculumPlanner(graph),
        "popularity": PopularityPlanner(graph, popularity),
        "weakest-first": WeakestFirstPlanner(graph),
        "greedy (proposed)": GreedyPlanner(engine, graph),
    }
    records: dict[str, list[dict]] = {name: [] for name in planners}

    start = time.perf_counter()
    for n, row in enumerate(pool.itertuples(index=False), start=1):
        total = int(row.n_events)
        cut_points = [
            int(total * frac) for frac in np.linspace(0.3, 0.7, args.cuts)
        ]
        for cut in cut_points:
            if cut < 30 or cut + args.horizon > total:
                continue
            future = set(int(c) for c in np.asarray(row.concept_ids)[cut:cut + args.horizon])
            past = set(int(c) for c in np.asarray(row.concept_ids)[:cut])
            relevant = future - past          # genuinely NEW concepts they moved to
            if not relevant:
                continue

            sequence = make_sequence(row, None, matrix, index, cut)
            mastery = engine.mastery(sequence)
            state = State.create(
                mastery, sequence, enrolled=enrolment_mask(graph, [row.code_module])
            )
            idle = engine.mastery_idle(sequence)

            for name, planner in planners.items():
                scored = planner.score(state)[: args.k]
                if not scored:
                    continue
                recommended = [s.concept for s in scored]
                gain = float(np.mean([
                    np.log(np.clip(mastery[c], 1e-6, 1 - 1e-6) /
                           (1 - np.clip(mastery[c], 1e-6, 1 - 1e-6)))
                    - np.log(np.clip(idle[c], 1e-6, 1 - 1e-6) /
                             (1 - np.clip(idle[c], 1e-6, 1 - 1e-6)))
                    for c in recommended
                ]))
                records[name].append({
                    "ndcg": ndcg_at_k(recommended, relevant, args.k),
                    "hit": hit_rate_at_k(recommended, relevant, args.k),
                    "precision": precision_at_k(recommended, relevant, args.k),
                    "gain": scored[0].total_gain,
                    "violations": prerequisite_violation_rate(
                        graph, mastery, recommended, state.thresholds.prerequisite
                    ),
                    "concepts": recommended,
                })
        if n % 25 == 0:
            print(f"  {n}/{len(pool)} learners ({time.perf_counter() - start:.0f}s)")

    table = []
    for name, rows in records.items():
        if not rows:
            continue
        ndcgs = np.array([r["ndcg"] for r in rows], dtype=float)
        seen = {c for r in rows for c in r["concepts"]}
        table.append(PathMetrics(
            planner=name,
            ndcg=float(np.nanmean(ndcgs)),
            ndcg_std=float(np.nanstd(ndcgs)),
            hit_rate=float(np.nanmean([r["hit"] for r in rows])),
            precision=float(np.nanmean([r["precision"] for r in rows])),
            knowledge_gain=float(np.nanmean([r["gain"] for r in rows])),
            prerequisite_violations=float(np.nanmean([r["violations"] for r in rows])),
            coverage=len(seen) / graph.n_concepts,
            n_decisions=len(rows),
        ))

    table.sort(key=lambda m: m.ndcg)
    width = max(len(m.planner) for m in table) + 2
    print("\n" + "=" * (width + 74))
    print(f"{'Planner':<{width}}{'NDCG@5':>16}{'Hit@5':>9}{'Prec@5':>9}"
          f"{'Gain':>9}{'PrereqViol':>12}{'Coverage':>10}{'n':>9}")
    print("=" * (width + 74))
    for m in table:
        print(f"{m.planner:<{width}}{m.ndcg:>9.4f} ± {m.ndcg_std:.3f}{m.hit_rate:>9.4f}"
              f"{m.precision:>9.4f}{m.knowledge_gain:>9.3f}"
              f"{m.prerequisite_violations:>12.4f}{m.coverage:>10.3f}{m.n_decisions:>9}")
    print("=" * (width + 74))
    print("NDCG/Hit/Precision measured against concepts the learner actually moved to next.")
    print("Gain is log-odds improvement over an idle week, from the frozen tracer.")

    # Significance. Decisions are paired: every planner answered the same questions,
    # so a paired test is the correct one and is far more powerful than comparing
    # means with wide standard deviations.
    from scipy import stats as _stats
    from elpr.eval.stats import holm_bonferroni, interpret_d

    reference = "greedy (proposed)"
    if reference in records:
        base = np.array([r["ndcg"] for r in records[reference]], dtype=float)
        others, raw = [], []
        for name, rows in records.items():
            if name == reference:
                continue
            other = np.array([r["ndcg"] for r in rows], dtype=float)
            n = min(len(base), len(other))
            mask = ~(np.isnan(base[:n]) | np.isnan(other[:n]))
            if mask.sum() < 3:
                continue
            t, p = _stats.ttest_rel(base[:n][mask], other[:n][mask])
            diff = base[:n][mask] - other[:n][mask]
            d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            others.append((name, float(diff.mean()), d))
            raw.append(float(p))
        adjusted, _ = holm_bonferroni(raw)
        print(f"\nPaired comparison of NDCG@5 against '{reference}' (Holm-corrected):")
        for (name, diff, d), p_adj in zip(others, adjusted):
            flag = "significant" if p_adj < 0.05 else "not significant"
            print(f"  vs {name:<18} +{diff:.4f}   p = {p_adj:.4f}   "
                  f"d = {d:+.2f} ({interpret_d(d)})   {flag}")
        significance = [
            {"vs": name, "difference": diff, "cohens_d": d, "p_holm": p_adj}
            for (name, diff, d), p_adj in zip(others, adjusted)
        ]
    else:
        significance = []

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "table2.json").write_text(json.dumps(
        {"fold": args.fold, "k": args.k, "horizon": args.horizon,
         "n_learners": len(pool), "rows": [m.row() for m in table],
         "significance": significance}, indent=2))
    print(f"\nWrote {(RESULTS / 'table2.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
