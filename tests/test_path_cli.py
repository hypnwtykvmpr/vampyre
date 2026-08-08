"""Regression tests for `graphify path` arrow direction (#849)."""

from __future__ import annotations
import json
import graphify.__main__ as mainmod


def _write_graph(tmp_path):
    graph_data = {
        "schema_version": 1,
        "directed": True,
        "multigraph": False,
        "graph": {"graphify_profile": {"graph_type": "digraph"}},
        "nodes": [
            {
                "id": "create_patch",
                "label": "createPatchHandler()",
                "source_file": "server/create-patch-handler.ts",
                "community": 0,
            },
            {
                "id": "validate",
                "label": "validateSanitySession()",
                "source_file": "server/sanity-validate-session.ts",
                "community": 0,
            },
        ],
        "links": [
            {
                "source": "create_patch",
                "target": "validate",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "_origin": "ast",
            },
        ],
        "hyperedges": [],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph_data), encoding="utf-8")
    return p


def _run(monkeypatch, graph_path, src, tgt, capsys):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys, "argv", ["graphify", "path", src, tgt, "--graph", str(graph_path)]
    )
    mainmod.main()
    return capsys.readouterr().out


def test_forward_arrow(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    out = _run(monkeypatch, p, "createPatchHandler", "validateSanitySession", capsys)
    assert "Shortest path (1 hops):" in out
    assert "createPatchHandler() --calls [EXTRACTED]--> validateSanitySession()" in out


def test_reverse_arrow(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    out = _run(monkeypatch, p, "validateSanitySession", "createPatchHandler", capsys)
    assert "Shortest path (1 hops):" in out
    assert "validateSanitySession() <--calls [EXTRACTED]-- createPatchHandler()" in out
    assert "validateSanitySession() --calls [EXTRACTED]--> createPatchHandler()" not in out


def test_same_relation_parallel_records_are_not_collapsed(monkeypatch, tmp_path, capsys):
    graph_data = {
        "schema_version": 1,
        "directed": True,
        "multigraph": True,
        "graph": {"graphify_profile": {"graph_type": "multidigraph"}},
        "nodes": [
            {"id": "a", "label": "Alpha"},
            {"id": "b", "label": "Beta"},
        ],
        "links": [
            {
                "source": "a",
                "target": "b",
                "key": "call-L5",
                "relation": "calls",
                "source_location": "L5",
                "confidence": "EXTRACTED",
                "context": "call",
                "_origin": "ast",
            },
            {
                "source": "a",
                "target": "b",
                "key": "call-L9",
                "relation": "calls",
                "source_location": "L9",
                "confidence": "INFERRED",
                "context": "callback",
                "provenance": {"provider": "semantic"},
                "_origin": "semantic",
            },
        ],
        "hyperedges": [],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_data), encoding="utf-8")

    out = _run(monkeypatch, graph_path, "Alpha", "Beta", capsys)

    assert "2 records" in out
    assert "key=call-L5" in out
    assert "key=call-L9" in out
