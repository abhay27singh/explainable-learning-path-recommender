#!/usr/bin/env python
"""Stage 0, part 2: build the concept prerequisite graph.

Runs sql/06-07 (structural edges, sequential prerequisite mining), mines corequisite
edges by FP-Growth, enforces the DAG, and writes the adjacency the GCN consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.duckdb_runner import SQLRunner  # noqa: E402
from elpr.graph.concept_graph import ConceptGraph  # noqa: E402
from elpr.mining.association import mine_corequisites  # noqa: E402

RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts" / "graph"


def main() -> int:
    runner = SQLRunner()

    print("Running SQL (sql/06-07):")
    runner.run_range(6, 7)

    concepts = runner.df("SELECT * FROM concepts ORDER BY concept_id")
    structural = runner.df(
        "SELECT src, dst, edge_type, weight FROM edges_structural"
    )
    mined = runner.df(
        "SELECT src, dst, edge_type, weight, support, precedence, lift FROM edges_mined"
    )
    pair_stats = runner.df("SELECT src, dst, precedence FROM pair_stats")
    first_touch = runner.df("SELECT id_student, concept_id FROM first_touch")

    print("\nMining corequisite edges (FP-Growth):")
    coreq = mine_corequisites(first_touch, pair_stats)
    print(f"  corequisite edges: {len(coreq):,}")

    edges = pd.concat(
        [
            structural[["src", "dst", "edge_type", "weight"]],
            mined[["src", "dst", "edge_type", "weight"]],
            coreq[["src", "dst", "edge_type", "weight"]],
        ],
        ignore_index=True,
    )

    print("\nBuilding graph and enforcing the DAG:")
    graph = ConceptGraph(concepts, edges)
    report = graph.cycle_report
    print(f"  cycles found and broken: {report.n_cycles}")
    print(f"  edges removed:           {len(report.removed)}")
    assert graph.is_dag(), "prerequisite graph is not acyclic after cycle breaking"

    graph.save(ARTIFACTS)

    stats = {
        "n_concepts": graph.n_concepts,
        "n_edges_total": int(graph.prereq.number_of_edges() + graph.coreq.number_of_edges()),
        "n_prerequisite_edges": int(graph.prereq.number_of_edges()),
        "n_corequisite_edges": int(graph.coreq.number_of_edges()),
        "edges_by_rule": edges.edge_type.value_counts().to_dict(),
        "n_cycles_broken": report.n_cycles,
        "edges_removed_for_dag": [
            {"src": int(u), "dst": int(v), "weight": w} for u, v, w in report.removed
        ],
        "mining": runner.df("SELECT * FROM mining_stats").to_dict("records")[0],
        "mean_in_degree": round(
            graph.prereq.number_of_edges() / max(graph.n_concepts, 1), 3
        ),
        "n_concepts_with_assessments": int(
            runner.scalar(
                "SELECT COUNT(DISTINCT concept_id) FROM events WHERE kind='assessment'"
            )
        ),
    }
    out = RESULTS / "graph_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=str))

    print("\nGraph statistics:")
    for k in (
        "n_concepts", "n_prerequisite_edges", "n_corequisite_edges",
        "n_cycles_broken", "mean_in_degree", "n_concepts_with_assessments",
    ):
        print(f"  {k:<32} {stats[k]}")
    print(f"  edges by rule                    {stats['edges_by_rule']}")
    print(f"\nWrote {out.relative_to(ROOT)} and {ARTIFACTS.relative_to(ROOT)}/")

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
