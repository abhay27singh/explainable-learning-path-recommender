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
from elpr.modules import codes_matching
from elpr.profile import ProfileEncoder

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
        self.profile_encoder = ProfileEncoder(self.features, self.default_features)
        self.modules = sorted(self.graph.concepts.code_module.unique().tolist())

    # -- registered learners ----------------------------------------------------
    def is_registered(self, student: int) -> bool:
        return int(student) >= 90_000_000

    def features_for_user(self, user: User) -> np.ndarray:
        """The learner's own background answers plus their course. Fields they left
        blank stay at the dataset average, the honest no-information value."""
        return self.profile_encoder.encode(self.store.get_profile(user.id), user.module)

    def registered_sequence(self, user: User, events: list | None = None) -> LearnerSequence:
        """Build a model-ready sequence from a registered learner's recorded activity.

        A brand-new account has no events at all. Rather than fail, seed a single
        neutral context event so the model has something to condition on — this is the
        genuine cold-start case, and the mastery estimate that comes back is the
        learner-profile prior rather than anything personalised.
        """
        events = self.store.events(user.id) if events is None else events
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
                student_features=self.features_for_user(user),
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
            student_features=self.features_for_user(user),
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
                | frame.code_module.isin(codes_matching(query))
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

    def done_concepts(self, user: User) -> tuple:
        """Weeks the student has finished, oldest first: studied or passed, unless the
        latest record for that week says they found it hard."""
        last = {}
        for e in self.store.events(user.id):
            concept = int(e["concept_id"])
            last.pop(concept, None)
            last[concept] = e["kind"] == "study" or bool(e["correct"])
        return tuple(c for c, ok in last.items() if ok)

    def course_path(self, user: User, steps: int = 5, known: tuple = ()) -> dict:
        """A registered student's course, week by week from the first unfinished week.

        The order is the course's own, so a new student starts at its first week. The
        model supplies what each week builds on and what it opens up, with the earlier
        steps treated as done: the same override an adviser uses for "already knows this"."""
        module = user.module or self.modules[0]
        weeks = [n["concept"] for n in self.module_graph(module)["nodes"]]
        done = list(dict.fromkeys([*(int(c) for c in known), *self.done_concepts(user)]))
        todo = [c for c in weeks if c not in done][:max(1, min(int(steps), 8))]
        curriculum = self.planners["curriculum"]
        out, first = [], None
        for i, concept in enumerate(todo):
            _, state = self.registered_state(user, tuple(done + todo[:i]))
            first = first or state
            action = curriculum._wrap(state, np.array([concept]), np.array([0.0]), False)[0]
            step = self.explainer.explain(state, action).to_dict()
            step["prerequisites_met"] = not step["gaps"]
            out.append(step)
        if first is None:
            _, first = self.registered_state(user, tuple(done))
        return {
            "module": module,
            "n_weeks": len(weeks),
            "n_done": len(set(done) & set(weeks)),
            "thresholds": {"mastered": round(first.thresholds.mastered, 4),
                           "prerequisite": round(first.thresholds.prerequisite, 4)},
            "steps": out,
        }

    def learning_path(self, make_state, steps: int = 5, start: tuple = ()) -> dict:
        """The next few steps, in order, each chosen as if the earlier steps were done.

        Step 1 is the ordinary top recommendation. Each later step re-runs the planner
        with the earlier steps marked as known, the same override an adviser uses for
        "already knows this", so every step is real model output under a stated
        assumption rather than a guess. make_state(overrides) returns (module, state)."""
        chosen = [int(c) for c in start]
        steps_out, module, first = [], None, None
        for _ in range(max(1, min(int(steps), 8))):
            module, state = make_state(tuple(chosen))
            first = first or state
            actions = self.planners["greedy"].score(state)[:1]
            if not len(actions):
                break
            step = self.explainer.explain(state, actions[0]).to_dict()
            concept = int(step["concept"])
            if concept in chosen:
                break
            chosen.append(concept)
            steps_out.append(step)
        return {
            "module": module,
            "thresholds": {"mastered": round(first.thresholds.mastered, 4),
                           "prerequisite": round(first.thresholds.prerequisite, 4)},
            "n_eligible": int(len(self.planners["greedy"].candidates(first)[0])),
            "steps": steps_out,
        }

    def progress_for(self, user: User, now: float | None = None) -> dict:
        """Streak, activity calendar, badges and a weekly summary for a registered student.

        Built only from the study events the student recorded and the model's own mastery
        estimates, so every number in the summary traces to data.

        Change is measured as the model's predicted chance of doing well, with and without
        the last seven days of events, and only counted when it moves by at least five
        points. Levels are not compared: they are relative to the student's own weeks, so
        one week rising re-labels others lower, which would report change that is not
        real. A student with no activity before this week is not compared at all, because
        the only baseline would be the model's generic starting estimate."""
        import time as _time
        from datetime import date

        from elpr.progress import badges, calendar_days, streaks

        now = _time.time() if now is None else float(now)
        today = date.fromtimestamp(now)
        events = self.store.events(user.id)
        stamps = [e["created_at"] for e in events]
        streak = streaks(stamps, today)
        window = now - 7 * 86400
        recent = [e for e in events if e["created_at"] >= window]
        older = [e for e in events if e["created_at"] < window]

        short = lambda c: self.graph.label(int(c)).split(" (")[0]
        module, state = self.registered_state(user)
        scope = state.scope()
        first_week = not older
        up, down = [], []
        if not first_week and recent:
            before = self.engine.mastery(self.registered_sequence(user, events=older))
            for c in range(self.graph.n_concepts):
                if not scope[c]:
                    continue
                change = float(state.mastery[c]) - float(before[c])
                if change >= 0.05:
                    up.append(short(c))
                elif change <= -0.05:
                    down.append(short(c))

        # the same next step the student's path shows
        path = self.course_path(user, 1)["steps"] if user.module else []
        next_step = path[0]["label"].split(" (")[0] if path else None

        n = len(recent)
        studied = sorted({short(e["concept_id"]) for e in recent})
        passed = sum(1 for e in recent if e["kind"] == "assessment" and e["correct"])
        hard = sum(1 for e in recent if e["kind"] == "assessment"
                   and e["correct"] is not None and not e["correct"])
        plural = lambda k, one, many: f"{k} {one if k == 1 else many}"
        lines = []
        if n == 0:
            lines.append("No activity recorded in the last 7 days.")
        else:
            lines.append(f"In the last 7 days you recorded {plural(n, 'activity', 'activities')} "
                         f"across {plural(len(studied), 'week', 'weeks')} of your course.")
            if passed or hard:
                lines.append(f"You passed {plural(passed, 'test', 'tests')} and marked "
                             f"{hard} as hard.")
            if first_week:
                lines.append("This is your first week of activity, so there is no earlier "
                             "week to compare with.")
            else:
                if up:
                    lines.append(f"Your chance of doing well rose by 5 points or more in "
                                 f"{plural(len(up), 'week', 'weeks')} of the course.")
                if down:
                    lines.append(f"It fell by 5 points or more in "
                                 f"{plural(len(down), 'week', 'weeks')}.")
                if not up and not down:
                    lines.append("No week's chance of doing well changed by 5 points or more.")
        if next_step:
            lines.append(f"Suggested next step: {next_step}.")

        all_passed = sum(1 for e in events if e["kind"] == "assessment" and e["correct"])
        return {
            "module": module,
            "streak": streak,
            "calendar": calendar_days(stamps, today),
            "badges": badges(len(events), streak["longest"], all_passed),
            "week": {"activities": n, "weeks_studied": studied, "passed": passed, "hard": hard,
                     "first_week": first_week, "moved_up": up, "moved_down": down,
                     "next_step": next_step, "summary": lines},
        }

    def module_graph(self, module: str) -> dict:
        """One course's weeks and the prerequisite links between them, for the
        roadmap view. Uses the de-duplicated prerequisite DAG (724 links in total);
        corequisite links are excluded because they carry no ordering."""
        if module not in self.modules:
            raise KeyError(module)
        concepts = self.graph.concepts
        mine = concepts[concepts.code_module == module].sort_values("week")
        ids = set(int(c) for c in mine.concept_id)
        kinds = {}
        for r in self.graph.edges.itertuples(index=False):
            if str(r.edge_type).startswith("prereq") and int(r.src) in ids:
                kinds.setdefault((int(r.src), int(r.dst)), []).append(
                    str(r.edge_type).replace("prereq_", ""))
        edges = [
            {"src": int(a), "dst": int(b), "kinds": sorted(set(kinds.get((int(a), int(b)), [])))}
            for a, b in self.graph.prereq.edges()
            if int(a) in ids and int(b) in ids
        ]
        return {
            "module": module,
            "nodes": [
                {"concept": int(r.concept_id), "week": int(r.week),
                 "label": self.graph.label(int(r.concept_id))}
                for r in mine.itertuples(index=False)
            ],
            "edges": edges,
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
