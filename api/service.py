"""Model and data loading for the inference API.

Everything expensive happens once, at process start: the checkpoint, the graph, the
sequence table. Per-request work is then a single batched forward pass, which is what
keeps the demo interactive. This is the whole reason the demo is not Streamlit — that
execution model re-runs the script on every interaction.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from elpr.data.dataset import (
    RESPONSE_CORRECT, RESPONSE_INCORRECT, RESPONSE_UNLABELLED,
    SequenceDataset, build_student_matrix,
)
from elpr.explain.explainer import Explainer
from elpr.graph.concept_graph import ConceptGraph
from elpr.planner.baselines import CurriculumPlanner, WeakestFirstPlanner
from elpr.planner.engine import LearnerSequence, MasteryEngine
from elpr.planner.greedy import BeamPlanner, GreedyPlanner
from elpr.planner.state import State, enrolment_mask
from elpr.db.app_store import AppStore, User

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


class Service:
    def __init__(self, variant: str = "proposed", fold: int = 0):
        self.graph = ConceptGraph.load(ARTIFACTS / "graph")

        labels_path = ROOT / "graph" / "concept_labels.yaml"
        self.catalogue = {}
        if labels_path.exists():
            self.catalogue = yaml.safe_load(labels_path.read_text())
            self.graph._labels = {
                int(k): v["label"] for k, v in self.catalogue.items()
            }

        self.adjacency = torch.load(ARTIFACTS / "graph" / "adjacency.pt")["prerequisite"]
        self.sequences = pd.read_parquet(PROCESSED / "sequences.parquet")
        self.features = pd.read_parquet(PROCESSED / "student_features.parquet")
        self.matrix, _ = build_student_matrix(self.features)
        self.index = {
            tuple(r): i
            for i, r in enumerate(
                self.features[SequenceDataset.KEY].itertuples(index=False)
            )
        }

        checkpoint = ARTIFACTS / "models" / f"kgdkt_{variant}_fold{fold}.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"checkpoint not found: {checkpoint}")
        self.engine = MasteryEngine.from_checkpoint(checkpoint, self.adjacency)
        self.checkpoint = checkpoint.name

        self.planners = {
            "greedy": GreedyPlanner(self.engine, self.graph),
            "beam": BeamPlanner(self.engine, self.graph),
            "curriculum": CurriculumPlanner(self.graph),
            "weakest": WeakestFirstPlanner(self.graph),
        }
        self.explainer = Explainer(self.engine, self.graph)

        # Index sequences for lookup without scanning the frame each request.
        self.by_student = {
            int(s): g for s, g in self.sequences.groupby("id_student")
        }

        self.store = AppStore()
        # A registered learner has no OULAD demographics. Rather than invent any, use
        # the dataset mean, which is the honest "no information" position and is what
        # the cold-start prior should fall back to.
        self.default_features = self.matrix.mean(axis=0).astype(np.float32)
        self.modules = sorted(self.graph.concepts.code_module.unique().tolist())

    # -- registered learners ----------------------------------------------------
    def is_registered(self, student: int) -> bool:
        return int(student) >= 90_000_000

    def registered_sequence(self, user: User) -> LearnerSequence:
        """Build a model-ready sequence from a registered learner's recorded activity.

        A brand-new account has no events at all. Rather than fail, seed a single
        neutral context event so the model has something to condition on — this is the
        genuine cold-start case, and the mastery estimate that comes back is the
        learner-profile prior rather than anything personalised.
        """
        events = self.store.events(user.id)
        module = user.module or self.modules[0]
        first_concept = int(
            self.graph.concepts[self.graph.concepts.code_module == module]
            .sort_values("week").concept_id.iloc[0]
        )
        if not events:
            return LearnerSequence(
                concept_ids=np.array([first_concept], dtype=np.int64),
                days=np.array([0.0], dtype=np.float32),
                responses=np.array([RESPONSE_UNLABELLED], dtype=np.int64),
                kinds=np.array([0], dtype=np.int64),
                clicks=np.array([0.0], dtype=np.float32),
                student_features=self.default_features,
            )

        concepts, days, responses, kinds, clicks = [], [], [], [], []
        for e in events:
            concepts.append(int(e["concept_id"]))
            days.append(float(e["day"]))
            clicks.append(1.0)
            if e["kind"] == "assessment":
                kinds.append(1)
                responses.append(RESPONSE_CORRECT if e["correct"] else RESPONSE_INCORRECT)
            else:
                kinds.append(0)
                responses.append(RESPONSE_UNLABELLED)

        window = slice(max(0, len(concepts) - 200), len(concepts))
        return LearnerSequence(
            concept_ids=np.array(concepts, dtype=np.int64)[window],
            days=np.array(days, dtype=np.float32)[window],
            responses=np.array(responses, dtype=np.int64)[window],
            kinds=np.array(kinds, dtype=np.int64)[window],
            clicks=np.array(clicks, dtype=np.float32)[window],
            student_features=self.default_features,
        )

    def registered_state(self, user: User, overrides: tuple = ()):
        sequence = self.registered_sequence(user)
        mastery = self.engine.mastery(sequence)
        for concept in overrides:
            mastery[int(concept)] = 0.999
        module = user.module or self.modules[0]
        state = State.create(
            mastery, sequence, enrolled=enrolment_mask(self.graph, [module])
        )
        return module, state

    # -- learners ---------------------------------------------------------------
    def search(self, query: str = "", limit: int = 40) -> list[dict]:
        frame = self.sequences
        if query:
            frame = frame[
                frame.id_student.astype(str).str.startswith(query)
                | frame.code_module.str.upper().str.startswith(query.upper())
            ]
        frame = frame.nlargest(limit, "n_events")
        outcomes = self.features.set_index(SequenceDataset.KEY).final_result
        rows = []
        for r in frame.itertuples(index=False):
            key = (r.id_student, r.code_module, r.code_presentation)
            rows.append({
                "id_student": int(r.id_student),
                "module": r.code_module,
                "presentation": r.code_presentation,
                "n_events": int(r.n_events),
                "n_assessments": int(r.n_supervised),
                "outcome": str(outcomes.get(key, "unknown")),
            })
        return rows

    def _row(self, student: int, module: str | None = None):
        group = self.by_student.get(int(student))
        if group is None or group.empty:
            raise KeyError(f"no learner {student}")
        if module:
            filtered = group[group.code_module == module]
            if not filtered.empty:
                group = filtered
        return group.sort_values("n_events", ascending=False).iloc[0]

    def sequence_for(self, student: int, module: str | None = None, upto: float = 1.0):
        row = self._row(student, module)
        concepts = np.asarray(row.concept_ids)
        cut = max(30, int(len(concepts) * float(np.clip(upto, 0.05, 1.0))))
        cut = min(cut, len(concepts))

        supervised = np.asarray(row.is_supervised)[:cut]
        labels = np.asarray(row.labels)[:cut]
        responses = np.where(
            supervised == 1,
            np.where(labels == 1, RESPONSE_CORRECT, RESPONSE_INCORRECT),
            RESPONSE_UNLABELLED,
        )
        window = slice(max(0, cut - 200), cut)
        key = (row.id_student, row.code_module, row.code_presentation)
        return row, LearnerSequence(
            concept_ids=concepts[:cut][window],
            days=np.asarray(row.days, dtype=np.float32)[:cut][window],
            responses=responses[window],
            kinds=supervised[window],
            clicks=np.asarray(row.clicks, dtype=np.float32)[:cut][window],
            student_features=self.matrix[self.index[key]],
        )

    @functools.lru_cache(maxsize=256)
    def _cached_state(self, student: int, module: str | None, upto: float, overrides: tuple):
        row, sequence = self.sequence_for(student, module, upto)
        mastery = self.engine.mastery(sequence)
        # Advisor override: mark concepts the adviser says are already known.
        for concept in overrides:
            mastery[int(concept)] = 0.999
        state = State.create(
            mastery, sequence, enrolled=enrolment_mask(self.graph, [row.code_module])
        )
        return row, state

    def state_for(
        self, student: int, module: str | None = None, upto: float = 1.0,
        overrides: tuple = (),
    ):
        return self._cached_state(int(student), module, float(upto), tuple(sorted(overrides)))

    # -- recommendations --------------------------------------------------------
    def recommend(
        self, student: int, k: int = 3, planner: str = "greedy",
        module: str | None = None, upto: float = 1.0, overrides: tuple = (),
    ) -> dict:
        row, state = self.state_for(student, module, upto, overrides)
        engine = self.planners.get(planner, self.planners["greedy"])
        actions = engine.score(state)[:k]
        explanations = [self.explainer.explain(state, a).to_dict() for a in actions]
        return {
            "student": int(row.id_student),
            "module": row.code_module,
            "presentation": row.code_presentation,
            "planner": planner,
            "n_eligible": int(len(engine.candidates(state)[0])),
            "thresholds": {
                "mastered": round(state.thresholds.mastered, 4),
                "prerequisite": round(state.thresholds.prerequisite, 4),
            },
            "recommendations": explanations,
        }

    def mastery_for(
        self, student: int, module: str | None = None, upto: float = 1.0,
        overrides: tuple = (),
    ) -> dict:
        row, state = self.state_for(student, module, upto, overrides)
        scope = state.scope()
        return {
            "student": int(row.id_student),
            "module": row.code_module,
            "n_events": int(row.n_events),
            "thresholds": {
                "mastered": round(state.thresholds.mastered, 4),
                "prerequisite": round(state.thresholds.prerequisite, 4),
            },
            "concepts": [
                {
                    "concept": int(c),
                    "label": self.graph.label(int(c)),
                    "week": int(self.graph.concepts.week.iloc[int(c)]),
                    "module": str(self.graph.concepts.code_module.iloc[int(c)]),
                    "mastery": round(float(state.mastery[c]), 4),
                    "in_scope": bool(scope[c]),
                }
                for c in range(self.graph.n_concepts)
            ],
        }

    def graph_payload(self, module: str | None = None) -> dict:
        concepts = self.graph.concepts
        keep = concepts.code_module == module if module else pd.Series(True, index=concepts.index)
        nodes = [
            {"id": int(r.concept_id), "label": self.graph.label(int(r.concept_id)),
             "module": r.code_module, "week": int(r.week), "n_students": int(r.n_students)}
            for r in concepts[keep].itertuples(index=False)
        ]
        allowed = {n["id"] for n in nodes}
        edges = [
            {"source": int(r.src), "target": int(r.dst), "type": r.edge_type,
             "weight": round(float(r.weight), 3)}
            for r in self.graph.edges.itertuples(index=False)
            if int(r.src) in allowed and int(r.dst) in allowed
        ]
        return {"nodes": nodes, "edges": edges}
