"""Greedy and beam planners.

This is the planner the demo runs on. It always works, it needs no training, and it
doubles as the non-RL baseline that any reinforcement-learning agent has to beat
before its inclusion in the paper can be justified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from elpr.graph.concept_graph import ConceptGraph
from elpr.planner.engine import MasteryEngine
from elpr.planner.state import State, action_mask, relaxed_action_mask


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class ScoredAction:
    concept: int
    total_gain: float          # summed mastery gain across all concepts
    own_gain: float            # gain on the recommended concept itself
    transfer_gain: float       # gain on everything else — what the graph propagates
    predicted_success: float   # mastery of this concept before studying it
    prerequisites_met: bool
    # Concepts receiving the most transferred benefit, largest first. Measured
    # sufficiency is ~0.002, meaning almost all the value of a recommendation lands
    # on concepts other than the one named — so an explanation that omits them is
    # true but tells the learner almost nothing about why.
    beneficiaries: list[tuple[int, float]] = field(default_factory=list)


class GreedyPlanner:
    """Rank concepts by one-step predicted total mastery gain."""

    name = "greedy"

    def __init__(self, engine: MasteryEngine, graph: ConceptGraph, max_candidates: int = 60):
        self.engine = engine
        self.graph = graph
        # Scoring is one batched forward over candidates; capping keeps a single
        # recommendation interactive on CPU.
        self.max_candidates = max_candidates

    def candidates(self, state: State) -> tuple[np.ndarray, bool]:
        legal = action_mask(state, self.graph)
        relaxed = False
        if not legal.any():
            legal = relaxed_action_mask(state, self.graph)
            relaxed = True
        indices = np.flatnonzero(legal)
        if len(indices) > self.max_candidates:
            # Prefer the concepts closest to being learnable: highest current mastery
            # below the mastered threshold.
            indices = indices[np.argsort(-state.mastery[indices])[: self.max_candidates]]
        return indices, relaxed

    def score(self, state: State) -> list[ScoredAction]:
        indices, relaxed = self.candidates(state)
        if len(indices) == 0:
            return []

        after = self.engine.mastery_after(state.sequence, [int(i) for i in indices], correct=True)

        # Measured against DOING NOTHING for the same elapsed time, in LOG-ODDS.
        #
        # Two corrections, both load-bearing:
        #
        # 1. The baseline is an idle week, not the present moment. Studying advances
        #    the clock, and the forgetting mechanism then decays every prior
        #    interaction; against a present-moment baseline every option looks harmful.
        # 2. Log-odds rather than probability. Engaged learners sit near 0.98 on
        #    almost every concept, where probability differences round to zero and the
        #    ranking becomes arbitrary.
        idle = _logit(self.engine.mastery_idle(state.sequence))
        gains = _logit(after) - idle[None, :]

        # Only concepts the learner could actually study count toward transfer.
        # Summing over all 237 sweeps in other modules the learner will never see,
        # whose predictions collapse toward zero and dominate the total.
        scope = state.scope()
        gains = gains * scope[None, :]

        scored = []
        for row, concept in enumerate(indices):
            own = float(gains[row, concept])
            total = float(gains[row].sum())
            ranked = np.argsort(-gains[row])
            top = [
                (int(c), float(gains[row, c]))
                for c in ranked[:4] if c != concept and gains[row, c] > 0
            ][:3]
            scored.append(
                ScoredAction(
                    concept=int(concept),
                    total_gain=total,
                    own_gain=own,
                    transfer_gain=total - own,
                    predicted_success=float(state.mastery[concept]),
                    prerequisites_met=not relaxed,
                    beneficiaries=top,
                )
            )
        scored.sort(key=lambda s: s.total_gain, reverse=True)
        return scored

    def recommend(self, state: State, k: int = 1) -> list[int]:
        return [s.concept for s in self.score(state)[:k]]


class BeamPlanner(GreedyPlanner):
    """Greedy extended to multi-step lookahead.

    Greedy optimises the next step only, which can miss a prerequisite worth taking
    now because it unlocks several concepts later. Beam search keeps the best few
    partial paths and extends them.
    """

    name = "beam"

    def __init__(self, engine, graph, depth: int = 3, width: int = 3, max_candidates: int = 30):
        super().__init__(engine, graph, max_candidates)
        self.depth = depth
        self.width = width

    def plan(self, state: State, length: int | None = None) -> list[int]:
        length = length or self.depth
        beams: list[tuple[float, list[int], State]] = [(0.0, [], state)]

        for _ in range(length):
            expanded: list[tuple[float, list[int], State]] = []
            for score_so_far, path, current in beams:
                for action in self.score(current)[: self.width]:
                    nxt = State(
                        mastery=self.engine.mastery_after(
                            current.sequence, [action.concept], correct=True
                        )[0],
                        sequence=self.engine.extend(current.sequence, action.concept, True),
                        covered=current.covered.copy(),
                        budget=current.budget - 1,
                        learner_profile=current.learner_profile,
                        thresholds=current.thresholds,
                    )
                    nxt.covered[action.concept] = True
                    expanded.append((score_so_far + action.total_gain, path + [action.concept], nxt))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0], reverse=True)
            beams = expanded[: self.width]

        return beams[0][1] if beams else []
