"""Hermetic contract tests for the optional Neo4j direct exporter."""

from __future__ import annotations

import networkx as nx
import neo4j


class _FakeSession:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query: str, **params):
        self.runs.append((query, params))


class _FakeDriver:
    def __init__(self):
        self.session_instance = _FakeSession()
        self.closed = False

    def session(self):
        return self.session_instance

    def close(self):
        self.closed = True


def test_push_to_neo4j_preserves_parallel_edge_identity(monkeypatch):
    from graphify.export import push_to_neo4j

    driver = _FakeDriver()
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *_args, **_kwargs: driver)
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

    result = push_to_neo4j(graph, "bolt://localhost", "neo4j", "password")

    edge_runs = [
        (query, params)
        for query, params in driver.session_instance.runs
        if "MERGE (a)-[r:" in query
    ]
    assert result == {"nodes": 2, "edges": 2}
    assert driver.closed is True
    assert len(edge_runs) == 2
    assert all("{edge_key: $edge_key}" in query for query, _params in edge_runs)
    assert {params["edge_key"] for _query, params in edge_runs} == {"call:L5", "call:L9"}
