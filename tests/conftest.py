from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "oulad.duckdb"
GRAPH_DIR = ROOT / "artifacts" / "graph"


@pytest.fixture(scope="session")
def con():
    if not DB.exists():
        pytest.skip("run scripts/01_prepare_data.py first")
    connection = duckdb.connect(str(DB), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def graph():
    from elpr.graph.concept_graph import ConceptGraph

    if not (GRAPH_DIR / "concepts.parquet").exists():
        pytest.skip("run scripts/02_build_graph.py first")
    return ConceptGraph.load(GRAPH_DIR)
