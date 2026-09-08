"""Comparison planners for Table II.

Every one obeys the same prerequisite action mask, so differences reflect the
*ranking* rather than one planner being allowed illegal moves. Two of these are
deliberately trivial: a recommender that cannot beat "suggest the next week of the
course" is not doing anything a syllabus does not already do.
"""

from __future__ import annotations

import numpy as np

from elpr.graph.concept_graph import ConceptGraph
from elpr.planner.greedy import ScoredAction
from elpr.planner.state import State, action_mask, relaxed_action_mask


class _MaskedPlanner:
    """Shared candidate generation, so every planner faces identical legal moves."""

    name = "base"

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def candidates(self, state: State) -> tuple[np.ndarray, bool]:
        legal = action_mask(state, self.graph)
        relaxed = False
        if not legal.any():
            legal = relaxed_action_mask(state, self.graph)
            relaxed = True
        return np.flatnonzero(legal), relaxed

    def _wrap(self, state, indices, scores, relaxed) -> list[ScoredAction]:
        order = np.argsort(-scores)
        return [
            ScoredAction(
                concept=int(indices[i]),
                total_gain=float(scores[i]),
                own_gain=float(scores[i]),
                transfer_gain=0.0,
                predicted_success=float(state.mastery[indices[i]]),
                prerequisites_met=not relaxed,
            )
            for i in order
        ]

    def recommend(self, state: State, k: int = 1) -> list[int]:
        return [a.concept for a in self.score(state)[:k]]


class RandomPlanner(_MaskedPlanner):
    """Uniform choice among legal concepts. The floor any method must clear."""

    name = "random"

    def __init__(self, graph: ConceptGraph, seed: int = 42):
        super().__init__(graph)
        self.rng = np.random.default_rng(seed)

    def score(self, state: State) -> list[ScoredAction]:
        indices, relaxed = self.candidates(state)
        if len(indices) == 0:
            return []
        return self._wrap(state, indices, self.rng.random(len(indices)), relaxed)


class PopularityPlanner(_MaskedPlanner):
    """Recommend whatever successful learners engage with most.

    A strong, unglamorous baseline. Any personalised method has to beat "do what
    works for most people" to justify the machinery.
    """

    name = "popularity"

    def __init__(self, graph: ConceptGraph, popularity: np.ndarray):
        super().__init__(graph)
        self.popularity = popularity

    def score(self, state: State) -> list[ScoredAction]:
        indices, relaxed = self.candidates(state)
        if len(indices) == 0:
            return []
        return self._wrap(state, indices, self.popularity[indices], relaxed)


class CurriculumPlanner(_MaskedPlanner):
    """Recommend the earliest unstudied week, i.e. follow the syllabus.

    The honest control. A learning-path recommender that cannot beat reading the
    course in order has not earned its place, whatever its AUC.
    """

    name = "curriculum"

    def __init__(self, graph: ConceptGraph):
        super().__init__(graph)
        self.week = graph.concepts.sort_values("concept_id").week.to_numpy()

    def score(self, state: State) -> list[ScoredAction]:
        indices, relaxed = self.candidates(state)
        if len(indices) == 0:
            return []
        # Earlier weeks first, so negate.
        return self._wrap(state, indices, -self.week[indices].astype(float), relaxed)


class WeakestFirstPlanner(_MaskedPlanner):
    """Recommend the concept the learner is weakest on.

    Needs no simulation at all — just the mastery vector. Included because it is the
    obvious thing to do, and the expensive lookahead planner should beat it.
    """

    name = "weakest"

    def score(self, state: State) -> list[ScoredAction]:
        indices, relaxed = self.candidates(state)
        if len(indices) == 0:
            return []
        return self._wrap(state, indices, -state.mastery[indices], relaxed)
