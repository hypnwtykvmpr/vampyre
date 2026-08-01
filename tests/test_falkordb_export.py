"""Hermetic contract tests for the optional FalkorDB exporter."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import pytest
import falkordb

FIXTURES = Path(__file__).parent / "fixtures"
GRAPH_NAME = "graphify_test"


class _Result:
    def __init__(self, value: int):
        self.result_set = [[value]]


class _FakeGraph:
    def __init__(self):
        self.nodes: set[object] = set()
        self.edges: set[tuple[object, object, str, object]] = set()

    def delete(self):
        self.nodes.clear()
        self.edges.clear()

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        if cypher.startswith("MERGE (n:"):
            self.nodes.add(params["id"])
            return _Result(0)
        if "MERGE (a)-[r:" in cypher:
            relation = re.search(r"MERGE \(a\)-\[r:([A-Z0-9_]+)", cypher)
            assert relation is not None
            self.edges.add(
                (
                    params["src"],
                    params["tgt"],
                    relation.group(1),
                    params.get("edge_key"),
                )
            )
            return _Result(0)
        if "count(n)" in cypher:
            return _Result(len(self.nodes))
        if "count(r)" in cypher:
            return _Result(len(self.edges))
        raise AssertionError(f"unexpected query: {cypher}")


class _FakeFalkorDB:
    def __init__(self):
        self.graphs: dict[str, _FakeGraph] = {}

    def select_graph(self, name: str):
        return self.graphs.setdefault(name, _FakeGraph())


@pytest.fixture()
def db(monkeypatch):
    client = _FakeFalkorDB()
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **_kwargs: client)
    return client


def test_push_to_falkordb_creates_expected_graph(db):
    from graphify.build import build_from_json
    from graphify.export import push_to_falkordb

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)

    result = push_to_falkordb(G, uri="localhost:6379", graph_name=GRAPH_NAME)

    assert result["nodes"] == G.number_of_nodes()
    assert result["edges"] == G.number_of_edges()

    graph = db.select_graph(GRAPH_NAME)
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    edge_count = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]

    assert node_count == G.number_of_nodes()
    assert edge_count == G.number_of_edges()


def test_push_to_falkordb_is_idempotent(db):
    """MERGE-based push is safe to re-run - counts must not grow."""
    from graphify.build import build_from_json
    from graphify.export import push_to_falkordb

    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)

    push_to_falkordb(G, uri="localhost:6379", graph_name=GRAPH_NAME)
    push_to_falkordb(G, uri="localhost:6379", graph_name=GRAPH_NAME)

    graph = db.select_graph(GRAPH_NAME)
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    edge_count = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]

    assert node_count == G.number_of_nodes()
    assert edge_count == G.number_of_edges()


def test_push_to_falkordb_preserves_parallel_edge_identity(db):
    from graphify.export import push_to_falkordb

    graph = nx.MultiDiGraph()
    graph.add_node("caller", file_type="function")
    graph.add_node("callee", file_type="function")
    graph.add_edge(
        "caller",
        "callee",
        key="call:L5",
        relation="calls",
        source_file="app.py",
        source_location="L5",
    )
    graph.add_edge(
        "caller",
        "callee",
        key="call:L9",
        relation="calls",
        source_file="app.py",
        source_location="L9",
    )

    push_to_falkordb(graph, uri="localhost:6379", graph_name=GRAPH_NAME)
    push_to_falkordb(graph, uri="localhost:6379", graph_name=GRAPH_NAME)

    stored = db.select_graph(GRAPH_NAME).edges
    assert len(stored) == 2
    assert {edge_key for *_prefix, edge_key in stored} == {"call:L5", "call:L9"}
