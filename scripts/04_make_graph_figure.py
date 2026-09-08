#!/usr/bin/env python
"""Stage 0, part 4: render the concept graph.

Nodes are laid out by their actual meaning — week of study on x, module on y — so the
figure shows the curriculum structure rather than a force-directed hairball.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.graph.concept_graph import ConceptGraph  # noqa: E402

GRAPH_DIR = ROOT / "artifacts" / "graph"
OUT = ROOT / "results" / "figures"

EDGE_STYLE = {
    "prereq_sequence": ("#1F5E7A", 0.9, 1.1),
    "prereq_assessment": ("#B07A2A", 0.5, 0.7),
    "prereq_mined": ("#8A5CA8", 0.45, 0.7),
    "corequisite": ("#2A6350", 0.35, 0.6),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    graph = ConceptGraph.load(GRAPH_DIR)
    concepts = graph.concepts
    modules = sorted(concepts.code_module.unique())
    y_of = {m: i for i, m in enumerate(modules)}

    pos = {
        row.concept_id: (row.week, y_of[row.code_module])
        for row in concepts.itertuples()
    }

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=160)

    # Edges, drawn weakest-first so the structural spine stays legible on top.
    #
    # All edges are now within-module, so every one shares its endpoints' y value and a
    # straight line would collapse onto the module's own row — hiding exactly the
    # non-adjacent structure the mining produced. Edges spanning more than one week are
    # therefore drawn as arcs, with height scaled to the week gap. Corequisites arc
    # downward to separate them from prerequisites.
    def arc(x0, x1, y, direction, color, alpha, width):
        span = abs(x1 - x0)
        if span <= 1:
            ax.plot([x0, x1], [y, y], color=color, alpha=alpha, lw=width, zorder=2)
            return
        height = direction * min(0.42, 0.05 + 0.020 * span)
        t = np.linspace(0.0, 1.0, 32)
        xs = x0 + (x1 - x0) * t
        ys = y + height * np.sin(np.pi * t)
        ax.plot(xs, ys, color=color, alpha=alpha, lw=width, zorder=1)

    for edge_type in ("prereq_mined", "corequisite", "prereq_assessment", "prereq_sequence"):
        color, alpha, width = EDGE_STYLE[edge_type]
        direction = -1.0 if edge_type == "corequisite" else 1.0
        subset = graph.edges[graph.edges.edge_type == edge_type]
        for row in subset.itertuples():
            if row.src not in pos or row.dst not in pos:
                continue
            (x0, y0), (x1, _) = pos[row.src], pos[row.dst]
            arc(x0, x1, y0, direction, color, alpha, width)

    sizes = 12 + 60 * (concepts.n_students / concepts.n_students.max())
    ax.scatter(
        [pos[c][0] for c in concepts.concept_id],
        [pos[c][1] for c in concepts.concept_id],
        s=sizes, c="#14181F", edgecolors="white", linewidths=0.6, zorder=3,
    )

    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=10)
    ax.set_xlabel("Week of study", fontsize=10)
    ax.set_ylabel("Module", fontsize=10)
    ax.set_title(
        f"Concept prerequisite graph — {graph.n_concepts} concepts, "
        f"{graph.prereq.number_of_edges()} prerequisite edges, "
        f"{graph.coreq.number_of_edges()} corequisite edges",
        fontsize=11, pad=14,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E6EBEF", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Line2D([], [], color=EDGE_STYLE[t][0], lw=1.6, label=lbl)
            for t, lbl in [
                ("prereq_sequence", "sequence (week w → w+1)"),
                ("prereq_assessment", "assessment coverage"),
                ("prereq_mined", "mined from behaviour"),
                ("corequisite", "corequisite"),
            ]
        ],
        loc="upper right", frameon=False, fontsize=9,
    )
    fig.tight_layout()
    out = OUT / "concept_graph.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out.relative_to(ROOT)}  ({graph.n_concepts} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
