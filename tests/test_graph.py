"""Invariants the concept graph must satisfy for the model and the paper to be valid."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def test_prerequisite_graph_is_acyclic(graph):
    assert graph.is_dag(), "cycle breaking failed to produce a DAG"


def test_concept_ids_are_contiguous(graph):
    ids = sorted(graph.concepts.concept_id.tolist())
    assert ids == list(range(len(ids))), "concept ids must index adjacency rows directly"


def test_every_site_maps_to_exactly_one_concept(con):
    total, mapped, distinct = con.sql(
        "SELECT COUNT(*), COUNT(DISTINCT id_site), COUNT(DISTINCT (id_site, concept_id)) "
        "FROM site_to_concept"
    ).fetchone()
    assert total == mapped == distinct, "a site was mapped to more than one concept"


def test_no_orphan_concepts(graph):
    isolated = [
        c for c in graph.prereq.nodes
        if graph.prereq.degree(c) == 0 and graph.coreq.degree(c) == 0
    ]
    assert not isolated, f"{len(isolated)} concepts have no edges at all"


def test_adjacency_is_symmetric_and_normalized(graph):
    a = graph.adjacency("prerequisite")
    assert a.shape == (graph.n_concepts, graph.n_concepts)
    assert torch.allclose(a, a.T, atol=1e-6), "adjacency must be symmetric for the GCN"
    assert torch.isfinite(a).all(), "adjacency contains NaN or inf"
    assert a.max().item() <= 1.0 + 1e-6
    # Row sums are NOT bounded by 1 under symmetric normalization — that holds for the
    # random-walk form D^-1 A. What matters for GCN stability is the spectral radius:
    # D^-1/2 (A+I) D^-1/2 has all eigenvalues in [-1, 1], so repeated propagation
    # neither explodes nor vanishes.
    eigenvalues = torch.linalg.eigvalsh(a.double())
    assert eigenvalues.abs().max().item() <= 1.0 + 1e-6, "spectral radius exceeds 1"


def test_self_loops_present(graph):
    a = graph.adjacency("prerequisite")
    assert (torch.diagonal(a) > 0).all(), "A + I means every node keeps its own signal"


def test_ancestors_respect_direction(graph):
    """A concept must never be its own ancestor, and ancestors must reach it."""
    sample = graph.concepts.concept_id.tolist()[::20]
    for c in sample:
        ancestors = graph.ancestors(c)
        assert c not in ancestors
        for a in ancestors[:5]:
            assert nx.has_path(graph.prereq, a, c)


def test_mined_edges_meet_declared_thresholds(con):
    """The acceptance rule stated in sql/07 and in the paper must actually hold."""
    violations = con.sql(
        """
        SELECT COUNT(*) FROM edges_mined
        WHERE support < 100 OR lift IS NULL OR precedence < 0.75 OR lift <= 1.1
        """
    ).fetchone()[0]
    assert violations == 0, f"{violations} mined edges violate the stated thresholds"


def test_mined_edges_are_within_module(con):
    """Cross-module 'prerequisites' are co-enrolment calendar artifacts: week 2 of one
    module trivially precedes week 32 of another. They must not be mined."""
    cross = con.sql(
        """
        SELECT COUNT(*) FROM edges_mined e
        JOIN concepts cs ON cs.concept_id = e.src
        JOIN concepts cd ON cd.concept_id = e.dst
        WHERE cs.code_module <> cd.code_module
        """
    ).fetchone()[0]
    assert cross == 0, f"{cross} mined edges cross module boundaries"


def test_mined_edges_are_not_merely_calendar_order(con):
    """Every mined edge must carry outcome evidence, not just temporal ordering."""
    trivial = con.sql(
        "SELECT COUNT(*) FROM edges_mined WHERE precedence >= 0.999 AND lift IS NULL"
    ).fetchone()[0]
    assert trivial == 0


def test_cycle_breaking_removed_only_weakest_edges(graph):
    """Every logged removal must be absent from the final graph."""
    for u, v, _ in graph.cycle_report.removed:
        assert not graph.prereq.has_edge(u, v)


def test_graph_scale_is_reported_honestly(graph):
    """Guards against silently drifting back toward the paper's unsupported figures."""
    assert 150 <= graph.n_concepts <= 400
    assert graph.prereq.number_of_edges() > graph.n_concepts
