"""Post-hoc explanation for a single recommendation.

Post-hoc is stated deliberately. The explanation runs after the planner has chosen,
which is what the paper's abstract says and what the architecture actually does. The
methodology section's claim that explainability is 'integrated rather than post-hoc'
is not supported by this design and must be corrected.

What makes the explanation more than a saliency map is that it is grounded in the
prerequisite graph and in counterfactuals the model can actually evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from elpr.explain.backtrack import PrerequisiteFinding, backtrack, unlocked_by
from elpr.graph.concept_graph import ConceptGraph
from elpr.planner.engine import MasteryEngine
from elpr.planner.greedy import ScoredAction
from elpr.planner.state import State


@dataclass
class Counterfactual:
    concept: int
    label: str
    gain_now: float
    gain_if_mastered: float

    @property
    def improvement(self) -> float:
        return self.gain_if_mastered - self.gain_now


@dataclass
class Explanation:
    concept: int
    label: str
    mastery: float
    total_gain: float
    own_gain: float
    transfer_gain: float
    prerequisites_met: bool
    gaps: list[PrerequisiteFinding] = field(default_factory=list)
    satisfied: list[PrerequisiteFinding] = field(default_factory=list)
    unlocks: list[tuple[int, str]] = field(default_factory=list)
    beneficiaries: list[tuple[int, str, float]] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "label": self.label,
            "mastery": round(self.mastery, 4),
            "total_gain": round(self.total_gain, 4),
            "own_gain": round(self.own_gain, 4),
            "transfer_gain": round(self.transfer_gain, 4),
            "prerequisites_met": self.prerequisites_met,
            "gaps": [
                {"concept": g.concept, "label": g.label, "mastery": round(g.mastery, 4),
                 "deficit": round(g.deficit, 4), "distance": g.distance,
                 "priority": round(g.priority, 4)}
                for g in self.gaps
            ],
            "satisfied": [
                {"concept": s.concept, "label": s.label, "mastery": round(s.mastery, 4)}
                for s in self.satisfied
            ],
            "unlocks": [{"concept": c, "label": lbl} for c, lbl in self.unlocks],
            "beneficiaries": [
                {"concept": c, "label": lbl, "gain": round(v, 4)}
                for c, lbl, v in self.beneficiaries
            ],
            "counterfactuals": [
                {"concept": c.concept, "label": c.label, "gain_now": round(c.gain_now, 4),
                 "gain_if_mastered": round(c.gain_if_mastered, 4),
                 "improvement": round(c.improvement, 4)}
                for c in self.counterfactuals
            ],
            "text": self.text,
        }


class Explainer:
    def __init__(
        self,
        engine: MasteryEngine,
        graph: ConceptGraph,
        max_counterfactuals: int = 2,
    ):
        self.engine = engine
        self.graph = graph
        self.max_counterfactuals = max_counterfactuals

    def explain(self, state: State, action: ScoredAction) -> Explanation:
        gaps, satisfied = backtrack(
            self.graph, state.mastery, action.concept,
            threshold=state.thresholds.prerequisite,
        )
        unlocks = [
            (c, self.graph.label(c))
            for c in unlocked_by(
                self.graph, state.mastery, action.concept,
                threshold=state.thresholds.prerequisite,
            )
        ][:4]

        explanation = Explanation(
            concept=action.concept,
            label=self.graph.label(action.concept),
            mastery=action.predicted_success,
            total_gain=action.total_gain,
            own_gain=action.own_gain,
            transfer_gain=action.transfer_gain,
            prerequisites_met=action.prerequisites_met,
            gaps=gaps,
            satisfied=satisfied,
            unlocks=unlocks,
            beneficiaries=[
                (c, self.graph.label(c), v) for c, v in action.beneficiaries
            ],
            counterfactuals=self._counterfactuals(state, action, gaps),
        )
        explanation.text = render(explanation)
        return explanation

    def _counterfactuals(
        self, state: State, action: ScoredAction, gaps: list[PrerequisiteFinding]
    ) -> list[Counterfactual]:
        """Ask the model what the recommendation would be worth if a gap were closed.

        Implemented by lifting the gap concept's mastery to the mastered threshold and
        re-scoring — so the counterfactual is computed, not asserted.
        """
        out = []
        for gap in gaps[: self.max_counterfactuals]:
            hypothetical = state.mastery.copy()
            hypothetical[gap.concept] = state.thresholds.mastered
            sequence = self.engine.extend(state.sequence, gap.concept, correct=True)
            after = self.engine.mastery_after(sequence, [action.concept], correct=True)[0]
            gain_if = float(np.clip(after - hypothetical, 0.0, None).sum())
            out.append(
                Counterfactual(
                    concept=gap.concept,
                    label=gap.label,
                    gain_now=action.total_gain,
                    gain_if_mastered=gain_if,
                )
            )
        return out


def render(explanation: Explanation) -> str:
    """Templated pedagogical prose.

    Deliberately not a language model: every clause is traceable to a number, which is
    what makes the explanation auditable.
    """
    lines = [f"Recommended: {explanation.label}"]

    if explanation.gaps:
        worst = explanation.gaps[0]
        lines.append(
            f"Your weakest prerequisite is {worst.label} at {worst.mastery:.2f}, "
            f"{worst.deficit:.2f} below the readiness level."
        )
        if len(explanation.gaps) > 1:
            others = ", ".join(f"{g.label} ({g.mastery:.2f})" for g in explanation.gaps[1:3])
            lines.append(f"Also outstanding: {others}.")
    elif explanation.satisfied:
        best = explanation.satisfied[0]
        lines.append(
            f"Every prerequisite is in place — {best.label} sits at {best.mastery:.2f} — "
            f"so this is ready to study now."
        )
    else:
        lines.append("This concept has no recorded prerequisites, so nothing blocks it.")

    # Report the per-concept effect, not the sum. A total of "207 log-odds" is the
    # sum over every concept in the module and means nothing to a reader; the size of
    # the effect on this concept, plus how many others benefit, does.
    n_helped = len(explanation.beneficiaries)
    lines.append(
        f"Studying it is predicted to improve this concept by {explanation.own_gain:.2f} "
        f"log-odds"
        + (f", and to carry benefit to {n_helped} related concepts." if n_helped else ".")
    )

    if explanation.beneficiaries:
        names = ", ".join(label for _, label, _ in explanation.beneficiaries[:2])
        lines.append(
            f"Most of that benefit lands elsewhere — chiefly {names} — because these "
            f"concepts share prerequisites with it."
        )

    for cf in explanation.counterfactuals:
        if cf.improvement > 0.001:
            lines.append(
                f"If {cf.label} were secured first, the predicted gain here would rise "
                f"from {cf.gain_now:.3f} to {cf.gain_if_mastered:.3f}."
            )

    if explanation.unlocks:
        names = ", ".join(label for _, label in explanation.unlocks[:2])
        lines.append(f"Completing this opens up: {names}.")

    if not explanation.prerequisites_met:
        lines.append(
            "Note: no concept currently has all prerequisites satisfied, so this "
            "recommendation relaxes that requirement."
        )

    return " ".join(lines)
