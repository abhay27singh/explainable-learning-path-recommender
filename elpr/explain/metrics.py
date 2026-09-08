"""Objective quality measures for explanations.

The paper evaluates its headline contribution — explanation quality — with a Likert
survey and nothing else. A survey measures whether people *liked* the explanation, not
whether it was *true of the model*. These three measure the latter:

    fidelity     do the cited prerequisites actually drive the prediction?
    sufficiency  how much of the recommendation do the cited factors account for?
    stability    does a small change in the learner's state rewrite the explanation?

An explanation can be persuasive and score badly on all three. That is exactly the
failure mode worth detecting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from elpr.explain.explainer import Explainer, Explanation
from elpr.planner.greedy import GreedyPlanner
from elpr.planner.state import State


@dataclass
class ExplanationQuality:
    fidelity: float
    sufficiency: float
    stability: float
    n_evaluated: int


def fidelity(explanations: list[Explanation]) -> float:
    """Rank correlation between cited deficit and measured counterfactual improvement.

    If the explanation says "Stack Frames is your biggest gap", then closing Stack
    Frames should produce the biggest improvement when the model is actually asked.
    Where it does not, the explanation is telling a story the model does not follow.
    """
    deficits, improvements = [], []
    for explanation in explanations:
        for counterfactual in explanation.counterfactuals:
            gap = next(
                (g for g in explanation.gaps if g.concept == counterfactual.concept), None
            )
            if gap is not None:
                deficits.append(gap.deficit)
                improvements.append(counterfactual.improvement)

    if len(deficits) < 3 or np.std(deficits) == 0 or np.std(improvements) == 0:
        return float("nan")
    return float(spearmanr(deficits, improvements).statistic)


def sufficiency(explanation: Explanation) -> float:
    """Share of the predicted gain attributable to the concept the explanation names.

    Near 1 means the explanation accounts for the recommendation on its own. Near 0
    means most of the value came from transfer the explanation never mentioned — the
    text is then true but incomplete.
    """
    if explanation.total_gain <= 0:
        return float("nan")
    return float(explanation.own_gain / explanation.total_gain)


def stability(
    planner: GreedyPlanner,
    explainer: Explainer,
    state: State,
    n_perturbations: int = 5,
    noise: float = 0.02,
    seed: int = 42,
) -> float:
    """Jaccard overlap of cited prerequisites under small mastery perturbations.

    Mastery estimates carry uncertainty. If nudging them by a couple of percentage
    points rewrites which prerequisites are blamed, the explanation is an artefact of
    noise rather than a property of the learner.
    """
    baseline_actions = planner.score(state)
    if not baseline_actions:
        return float("nan")
    baseline = explainer.explain(state, baseline_actions[0])
    baseline_set = {g.concept for g in baseline.gaps}
    if not baseline_set:
        return float("nan")

    rng = np.random.default_rng(seed)
    overlaps = []
    for _ in range(n_perturbations):
        perturbed = State(
            mastery=np.clip(
                state.mastery + rng.normal(0, noise, size=state.mastery.shape), 0.0, 1.0
            ),
            sequence=state.sequence,
            covered=state.covered.copy(),
            enrolled=state.enrolled,
            budget=state.budget,
            learner_profile=state.learner_profile,
            thresholds=state.thresholds,
        )
        actions = planner.score(perturbed)
        if not actions:
            continue
        current = {g.concept for g in explainer.explain(perturbed, actions[0]).gaps}
        union = baseline_set | current
        overlaps.append(len(baseline_set & current) / len(union) if union else 1.0)

    return float(np.mean(overlaps)) if overlaps else float("nan")


def evaluate(
    planner: GreedyPlanner,
    explainer: Explainer,
    states: list[State],
    n_perturbations: int = 5,
) -> ExplanationQuality:
    explanations, sufficiencies, stabilities = [], [], []

    for state in states:
        actions = planner.score(state)
        if not actions:
            continue
        explanation = explainer.explain(state, actions[0])
        explanations.append(explanation)
        value = sufficiency(explanation)
        if not np.isnan(value):
            sufficiencies.append(value)
        score = stability(planner, explainer, state, n_perturbations)
        if not np.isnan(score):
            stabilities.append(score)

    return ExplanationQuality(
        fidelity=fidelity(explanations),
        sufficiency=float(np.mean(sufficiencies)) if sufficiencies else float("nan"),
        stability=float(np.mean(stabilities)) if stabilities else float("nan"),
        n_evaluated=len(explanations),
    )
