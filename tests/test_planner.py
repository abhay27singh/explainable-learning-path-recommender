"""Invariants for the planner and the explainer."""

from __future__ import annotations

import numpy as np
import pytest

from elpr.explain.backtrack import backtrack, unlocked_by
from elpr.planner.state import State, Thresholds, action_mask, enrolment_mask


@pytest.fixture
def sequence():
    from elpr.planner.engine import LearnerSequence

    n = 12
    return LearnerSequence(
        concept_ids=np.arange(n) % 5,
        days=np.arange(n, dtype=np.float32) * 3,
        responses=np.full(n, 2),
        kinds=np.zeros(n, dtype=np.int64),
        clicks=np.ones(n, dtype=np.float32),
        student_features=np.zeros(56, dtype=np.float32),
    )


def make_state(graph, mastery, sequence, **kwargs):
    return State.create(mastery, sequence, **kwargs)


def test_mask_excludes_mastered_concepts(graph, sequence):
    mastery = np.full(graph.n_concepts, 0.95, dtype=np.float32)
    state = make_state(graph, mastery, sequence)
    assert not action_mask(state, graph).any(), "nothing should be legal when all mastered"


def test_mask_excludes_unmet_prerequisites(graph, sequence):
    """The core pedagogical guarantee: never recommend past an unmet prerequisite."""
    mastery = np.full(graph.n_concepts, 0.10, dtype=np.float32)
    state = make_state(graph, mastery, sequence)
    legal = action_mask(state, graph)

    for concept in np.flatnonzero(legal):
        prereqs = graph.prereqs(int(concept))
        assert all(mastery[p] >= state.thresholds.prerequisite for p in prereqs), (
            f"concept {concept} is legal despite unmet prerequisites"
        )


def test_mask_respects_enrolment(graph, sequence):
    """A learner in one module must never be pointed at another module's material."""
    module = graph.concepts.code_module.iloc[0]
    mastery = np.full(graph.n_concepts, 0.30, dtype=np.float32)
    state = make_state(
        graph, mastery, sequence, enrolled=enrolment_mask(graph, [module])
    )
    for concept in np.flatnonzero(action_mask(state, graph)):
        assert graph.concepts.code_module.iloc[concept] == module


def test_covered_concepts_are_not_repeated(graph, sequence):
    mastery = np.full(graph.n_concepts, 0.30, dtype=np.float32)
    state = make_state(graph, mastery, sequence)
    legal = action_mask(state, graph)
    first = int(np.flatnonzero(legal)[0])
    state.covered[first] = True
    assert not action_mask(state, graph)[first]


def test_backtracking_returns_only_true_ancestors(graph):
    mastery = np.full(graph.n_concepts, 0.30, dtype=np.float32)
    targets = [c for c in graph.concepts.concept_id if graph.prereqs(int(c))][:15]
    for target in targets:
        gaps, met = backtrack(graph, mastery, int(target))
        ancestors = set(graph.ancestors(int(target)))
        for finding in gaps + met:
            assert finding.concept in ancestors, "cited a concept that is not a prerequisite"
            assert finding.concept != target


def test_gaps_are_ranked_by_priority(graph):
    rng = np.random.default_rng(0)
    mastery = rng.uniform(0, 1, size=graph.n_concepts).astype(np.float32)
    targets = [c for c in graph.concepts.concept_id if graph.prereqs(int(c))][:15]
    for target in targets:
        gaps, _ = backtrack(graph, mastery, int(target))
        priorities = [g.priority for g in gaps]
        assert priorities == sorted(priorities, reverse=True)


def test_deficit_is_zero_exactly_when_met(graph):
    rng = np.random.default_rng(1)
    mastery = rng.uniform(0, 1, size=graph.n_concepts).astype(np.float32)
    target = next(int(c) for c in graph.concepts.concept_id if graph.prereqs(int(c)))
    gaps, met = backtrack(graph, mastery, target, threshold=0.6)
    assert all(g.mastery < 0.6 and g.deficit > 0 for g in gaps)
    assert all(m.mastery >= 0.6 and m.deficit == 0 for m in met)


def test_unlocked_concepts_are_real_successors(graph):
    mastery = np.full(graph.n_concepts, 0.90, dtype=np.float32)
    for concept in list(graph.concepts.concept_id)[:20]:
        for successor in unlocked_by(graph, mastery, int(concept)):
            assert graph.prereq.has_edge(int(concept), successor)


def test_thresholds_are_ordered():
    t = Thresholds()
    assert t.prerequisite < t.mastered, (
        "readiness must be a lower bar than mastery, or nothing is ever recommendable"
    )
