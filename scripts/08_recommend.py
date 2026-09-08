#!/usr/bin/env python
"""Stage 2: produce explained recommendations for a real learner.

    python scripts/08_recommend.py --student 11391 --k 3

Uses a trained checkpoint when one is present in artifacts/models/. Without one it
falls back to an untrained model so the pipeline can still be exercised — the numbers
are then meaningless and the script says so.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from elpr.explain.explainer import Explainer  # noqa: E402
from elpr.graph.concept_graph import ConceptGraph  # noqa: E402
from elpr.models.kgdkt import KGDKT, KGDKTConfig  # noqa: E402
from elpr.planner.engine import LearnerSequence, MasteryEngine  # noqa: E402
from elpr.planner.greedy import BeamPlanner, GreedyPlanner  # noqa: E402
from elpr.planner.state import State, enrolment_mask  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"


def load_learner(sequences: pd.DataFrame, features: pd.DataFrame, student: int | None):
    rows = sequences if student is None else sequences[sequences.id_student == student]
    if rows.empty:
        raise SystemExit(f"no sequence found for student {student}")
    # Prefer a learner with enough history to be interesting.
    row = rows.sort_values("n_events", ascending=False).iloc[0]

    matrix, _ = build_student_matrix(features)
    key = (row.id_student, row.code_module, row.code_presentation)
    index = {
        tuple(r): i
        for i, r in enumerate(
            features[SequenceDataset.KEY].itertuples(index=False)
        )
    }
    if key not in index:
        raise SystemExit(f"no features for {key}")

    concepts = np.asarray(row.concept_ids, dtype=np.int64)
    supervised = np.asarray(row.is_supervised, dtype=np.int64)
    labels = np.asarray(row.labels, dtype=np.int64)
    responses = np.where(
        supervised == 1,
        np.where(labels == 1, RESPONSE_CORRECT, RESPONSE_INCORRECT),
        RESPONSE_UNLABELLED,
    )

    window = slice(max(0, len(concepts) - 200), len(concepts))
    return row, LearnerSequence(
        concept_ids=concepts[window],
        days=np.asarray(row.days, dtype=np.float32)[window],
        responses=responses[window],
        kinds=supervised[window],
        clicks=np.asarray(row.clicks, dtype=np.float32)[window],
        student_features=matrix[index[key]],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--variant", default="proposed")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--beam", action="store_true", help="also plan a multi-step path")
    args = parser.parse_args()

    graph = ConceptGraph.load(ARTIFACTS / "graph")
    labels_path = ROOT / "graph" / "concept_labels.yaml"
    if labels_path.exists():
        import yaml
        catalogue = yaml.safe_load(labels_path.read_text())
        graph._labels = {int(k): v["label"] for k, v in catalogue.items()}

    adjacency = torch.load(ARTIFACTS / "graph" / "adjacency.pt")["prerequisite"]
    sequences = pd.read_parquet(PROCESSED / "sequences.parquet")
    features = pd.read_parquet(PROCESSED / "student_features.parquet")

    checkpoint = ARTIFACTS / "models" / f"kgdkt_{args.variant}_fold{args.fold}.pt"
    if checkpoint.exists():
        engine = MasteryEngine.from_checkpoint(checkpoint, adjacency)
        trained = True
    else:
        matrix, _ = build_student_matrix(features)
        config = KGDKTConfig(
            n_concepts=graph.n_concepts, n_student_features=matrix.shape[1]
        )
        engine = MasteryEngine(KGDKT(config), adjacency)
        trained = False
        print(f"WARNING: {checkpoint.name} not found — using an UNTRAINED model.")
        print("         The pipeline runs, but every number below is meaningless.\n")

    row, sequence = load_learner(sequences, features, args.student)
    mastery = engine.mastery(sequence)
    # Restrict to the modules this learner is enrolled in.
    modules = sequences[sequences.id_student == row.id_student].code_module.unique().tolist()
    state = State.create(mastery, sequence, enrolled=enrolment_mask(graph, modules))

    planner = GreedyPlanner(engine, graph)
    explainer = Explainer(engine, graph)
    actions = planner.score(state)[: args.k]

    print(f"Learner {row.id_student} · {row.code_module} {row.code_presentation}")
    print(f"  {row.n_events} interactions, {row.n_supervised} assessments")
    print(f"  mastery: mean {mastery.mean():.3f}, "
          f"{int((mastery >= state.thresholds.mastered).sum())} of {len(mastery)} concepts mastered")
    print(f"  enrolled modules: {', '.join(modules)}")
    print(f"  {len(planner.candidates(state)[0])} concepts currently eligible\n")

    payload = []
    for rank, action in enumerate(actions, start=1):
        explanation = explainer.explain(state, action)
        payload.append(explanation.to_dict())
        print(f"{rank}. {explanation.text}\n")

    if args.beam:
        path = BeamPlanner(engine, graph).plan(state)
        print("Suggested path:")
        for step, concept in enumerate(path, start=1):
            print(f"  {step}. {graph.label(concept)}")

    if trained:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"recommendations_student{row.id_student}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
