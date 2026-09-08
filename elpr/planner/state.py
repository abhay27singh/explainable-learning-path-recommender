"""Planner state and the prerequisite action mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from elpr.graph.concept_graph import ConceptGraph
from elpr.planner.engine import LearnerSequence


@dataclass
class Thresholds:
    """Where 'mastered' and 'ready' sit on the mastery scale.

    ABSOLUTE THRESHOLDS DO NOT WORK HERE, and the reason matters for the paper.

    Because non-submission is the negative class, the model predicts whether a learner
    will successfully complete an assessment — which is dominated by engagement. The
    resulting mastery distribution is bimodal: an engaged learner scores ~0.98 on
    essentially every concept, a disengaged one ~0.00. A fixed bar of 0.80 therefore
    marks everything as mastered for one group and nothing for the other, and the
    planner has no candidates in either case.

    Relative mode instead ranks each learner against their own distribution: the
    weakest concepts are the ones worth studying, whoever they are. Absolute mode is
    retained for the sensitivity analysis and for datasets where it applies.
    """

    mastered: float = 0.80
    prerequisite: float = 0.60
    relative: bool = True
    mastered_percentile: float = 70.0      # above this, treat as already known
    prerequisite_percentile: float = 30.0  # above this, treat as ready enough

    def resolve(self, mastery: np.ndarray, scope: np.ndarray | None = None) -> "Thresholds":
        """Return thresholds calibrated to this learner's own mastery spread."""
        if not self.relative:
            return self
        values = mastery[scope] if scope is not None and scope.any() else mastery
        if values.size == 0 or float(np.ptp(values)) < 1e-6:
            return Thresholds(self.mastered, self.prerequisite, relative=False)
        return Thresholds(
            mastered=float(np.percentile(values, self.mastered_percentile)),
            prerequisite=float(np.percentile(values, self.prerequisite_percentile)),
            relative=False,
        )


@dataclass
class State:
    mastery: np.ndarray                  # [C], calibrated
    sequence: LearnerSequence
    covered: np.ndarray                  # [C] bool, recommended already this session
    # Concepts the learner could actually study. Concepts are module-scoped, so a
    # learner enrolled in DDD must never be pointed at week 1 of AAA — those are
    # different courses, not an easier starting point.
    enrolled: np.ndarray | None = None   # [C] bool
    budget: int = 10
    learner_profile: int | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    @classmethod
    def create(
        cls, mastery: np.ndarray, sequence: LearnerSequence, **kwargs
    ) -> "State":
        state = cls(
            mastery=mastery,
            sequence=sequence,
            covered=np.zeros(len(mastery), dtype=bool),
            **kwargs,
        )
        # Calibrate thresholds to this learner before any masking happens.
        state.thresholds = state.thresholds.resolve(mastery, state.scope())
        return state

    def scope(self) -> np.ndarray:
        """Concepts in scope for this learner: [C] bool."""
        if self.enrolled is None:
            return np.ones(len(self.mastery), dtype=bool)
        return self.enrolled


def enrolment_mask(graph: ConceptGraph, modules: list[str]) -> np.ndarray:
    """Concepts belonging to the modules a learner is enrolled in."""
    return graph.concepts.code_module.isin(modules).to_numpy()


def action_mask(state: State, graph: ConceptGraph) -> np.ndarray:
    """Legal next concepts: [C] bool.

    A concept is legal when it is not already mastered, has not been recommended this
    session, and every one of its prerequisites is at or above the readiness
    threshold. Encoding pedagogy in the mask rather than hoping a learned policy
    discovers it means an unprepared recommendation is impossible by construction —
    not merely unlikely.
    """
    legal = (state.mastery < state.thresholds.mastered) & ~state.covered & state.scope()

    for concept in np.flatnonzero(legal):
        prereqs = graph.prereqs(int(concept))
        if prereqs and np.any(state.mastery[prereqs] < state.thresholds.prerequisite):
            legal[concept] = False

    return legal


def relaxed_action_mask(state: State, graph: ConceptGraph) -> np.ndarray:
    """Fallback when nothing is legal.

    Early in a course every concept can have an unmet prerequisite, which would leave
    the learner with no recommendation at all. Dropping the prerequisite condition
    keeps the system responsive; the explanation then says plainly that prerequisites
    are outstanding.
    """
    return (state.mastery < state.thresholds.mastered) & ~state.covered & state.scope()
