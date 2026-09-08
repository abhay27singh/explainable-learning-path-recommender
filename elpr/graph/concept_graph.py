"""The concept prerequisite graph: construction, DAG enforcement, and adjacency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch


@dataclass
class CycleBreakReport:
    n_cycles: int
    removed: list[tuple[int, int, float]]


def break_cycles(graph: nx.DiGraph) -> CycleBreakReport:
    """Force a DAG by repeatedly removing the weakest edge on a detected cycle.

    Mined edges are statistical, so they can contradict one another and close a loop.
    Removing the lowest-weight edge of each cycle keeps the best-supported ordering.
    Every removal is logged, because silently discarding evidence is not acceptable
    in a graph the paper will describe.
    """
    removed: list[tuple[int, int, float]] = []
    n_cycles = 0
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        n_cycles += 1
        edges = [(u, v) for u, v, *_ in cycle]
        u, v = min(edges, key=lambda e: graph[e[0]][e[1]].get("weight", 1.0))
        removed.append((u, v, float(graph[u][v].get("weight", 1.0))))
        graph.remove_edge(u, v)
    return CycleBreakReport(n_cycles=n_cycles, removed=removed)


class ConceptGraph:
    """Prerequisite DAG over concepts, plus the dense adjacency the GCN consumes."""

    def __init__(self, concepts: pd.DataFrame, edges: pd.DataFrame):
        self.concepts = concepts.sort_values("concept_id").reset_index(drop=True)
        self.edges = edges
        self.n_concepts = len(self.concepts)
        self._labels = dict(zip(self.concepts.concept_id, self.concepts.concept_key))

        self.prereq = nx.DiGraph()
        self.prereq.add_nodes_from(self.concepts.concept_id.tolist())
        prereq_edges = edges[edges.edge_type != "corequisite"]
        for row in prereq_edges.itertuples():
            # Keep the strongest weight when several rules propose the same edge.
            if self.prereq.has_edge(row.src, row.dst):
                self.prereq[row.src][row.dst]["weight"] = max(
                    self.prereq[row.src][row.dst]["weight"], float(row.weight)
                )
            else:
                self.prereq.add_edge(row.src, row.dst, weight=float(row.weight))

        self.cycle_report = break_cycles(self.prereq)

        self.coreq = nx.Graph()
        self.coreq.add_nodes_from(self.concepts.concept_id.tolist())
        for row in edges[edges.edge_type == "corequisite"].itertuples():
            self.coreq.add_edge(row.src, row.dst, weight=float(row.weight))

    # -- queries ----------------------------------------------------------------
    def prereqs(self, c: int) -> list[int]:
        return list(self.prereq.predecessors(c))

    def ancestors(self, c: int) -> list[int]:
        return list(nx.ancestors(self.prereq, c))

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.prereq)

    def label(self, c: int) -> str:
        return self._labels.get(c, str(c))

    # -- tensors ----------------------------------------------------------------
    def adjacency(self, edge_type: str = "prerequisite") -> torch.Tensor:
        """Symmetric normalized adjacency  Â = D^-1/2 (A + I) D^-1/2.

        Dense rather than sparse: at 237 concepts this is a 237x237 matrix, which
        costs nothing and avoids sparse-operator gaps on CPU and MPS backends.
        """
        n = self.n_concepts
        a = np.zeros((n, n), dtype=np.float32)
        graph = self.prereq if edge_type == "prerequisite" else self.coreq
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            a[u, v] = w
            a[v, u] = w  # message passing is undirected; direction lives in the DAG
        a += np.eye(n, dtype=np.float32)
        deg = a.sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
        a_hat = a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        return torch.from_numpy(a_hat)

    # -- persistence ------------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.concepts.to_parquet(out_dir / "concepts.parquet")
        self.edges.to_parquet(out_dir / "edges.parquet")
        torch.save(
            {
                "prerequisite": self.adjacency("prerequisite"),
                "corequisite": self.adjacency("corequisite"),
                "n_concepts": self.n_concepts,
            },
            out_dir / "adjacency.pt",
        )

    @classmethod
    def load(cls, out_dir: Path) -> "ConceptGraph":
        return cls(
            pd.read_parquet(out_dir / "concepts.parquet"),
            pd.read_parquet(out_dir / "edges.parquet"),
        )
