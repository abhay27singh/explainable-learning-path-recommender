"""Priority backtracking over the prerequisite graph.

Given a recommended concept, walk its prerequisite ancestry and rank what is missing.
A gap two steps back matters less than the same gap immediately before, so priority
decays with graph distance:

    priority(a) = deficit(a) x decay^(distance from a to the recommendation)

This is the mechanism the paper names but never specifies.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from elpr.graph.concept_graph import ConceptGraph


@dataclass
class PrerequisiteFinding:
    concept: int
    label: str
    mastery: float
    deficit: float          # how far below the readiness threshold, 0 if met
    distance: int           # steps back through the prerequisite graph
    priority: float
    met: bool


def backtrack(
    graph: ConceptGraph,
    mastery: np.ndarray,
    target: int,
    threshold: float = 0.60,
    decay: float = 0.6,
    max_depth: int = 4,
    top_k: int = 5,
) -> tuple[list[PrerequisiteFinding], list[PrerequisiteFinding]]:
    """Return (unmet gaps, satisfied prerequisites), each ranked by priority.

    Satisfied prerequisites are returned too: an explanation that only lists problems
    reads as criticism, and a learner needs to see what they have already secured.
    """
    distances: dict[int, int] = {}
    frontier = [(target, 0)]
    seen = {target}

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for parent in graph.prereq.predecessors(node):
            if parent not in seen:
                seen.add(parent)
                distances[parent] = depth + 1
                frontier.append((parent, depth + 1))
            else:
                distances[parent] = min(distances.get(parent, depth + 1), depth + 1)

    findings = []
    for concept, distance in distances.items():
        value = float(mastery[concept])
        deficit = max(0.0, threshold - value)
        findings.append(
            PrerequisiteFinding(
                concept=concept,
                label=graph.label(concept),
                mastery=value,
                deficit=deficit,
                distance=distance,
                priority=deficit * (decay ** (distance - 1)),
                met=deficit == 0.0,
            )
        )

    gaps = sorted([f for f in findings if not f.met], key=lambda f: -f.priority)[:top_k]
    met = sorted([f for f in findings if f.met], key=lambda f: (f.distance, -f.mastery))[:top_k]
    return gaps, met


def unlocked_by(
    graph: ConceptGraph, mastery: np.ndarray, concept: int, threshold: float = 0.60
) -> list[int]:
    """Concepts that would become reachable once ``concept`` is mastered.

    This is what turns an explanation from backward-looking ("you are missing X") into
    forward-looking ("studying this opens up Y and Z"), which is the more useful thing
    to tell a learner.
    """
    unlocked = []
    for successor in graph.prereq.successors(concept):
        others = [p for p in graph.prereq.predecessors(successor) if p != concept]
        if all(mastery[p] >= threshold for p in others):
            unlocked.append(int(successor))
    return unlocked


def is_ancestor(graph: ConceptGraph, ancestor: int, descendant: int) -> bool:
    return nx.has_path(graph.prereq, ancestor, descendant)
