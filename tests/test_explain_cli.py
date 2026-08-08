"""Regression tests for `graphify explain` arrow direction (#853)."""

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
                "id": "validate",
                "label": "validateSanitySession()",
                "source_file": "server/sanity-validate-session.ts",
                "community": 0,
            },
            {
                "id": "create_patch",
                "label": "createPatchHandler()",
                "source_file": "server/create-patch-handler.ts",
                "community": 0,
            },
            {
                "id": "create_edit",
                "label": "createEditHandler()",
                "source_file": "server/create-edit-handler.ts",
                "community": 0,
            },
            {
                "id": "stable_stringify",
                "label": "stableStringify()",
                "source_file": "shared/stringify.ts",
                "community": 0,
            },
        ],
        "hyperedges": [],
        "links": [
            {
                "source": "create_patch",
                "target": "validate",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "_origin": "ast",
            },
            {
                "source": "create_edit",
                "target": "validate",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "_origin": "ast",
            },
            {
                "source": "validate",
                "target": "stable_stringify",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "_origin": "ast",
            },
        ],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph_data), encoding="utf-8")
    return p


def _run(monkeypatch, graph_path, label, capsys):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys, "argv", ["graphify", "explain", label, "--graph", str(graph_path)]
    )
    mainmod.main()
    return capsys.readouterr().out


def test_callee_shows_callers_as_inbound(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    out = _run(monkeypatch, p, "validateSanitySession", capsys)
    assert "<-- createPatchHandler() [calls]" in out
    assert "<-- createEditHandler() [calls]" in out
    assert "--> stableStringify() [calls]" in out
    assert "--> createPatchHandler() [calls]" not in out
    assert "--> createEditHandler() [calls]" not in out


def test_caller_shows_callee_as_outbound(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    out = _run(monkeypatch, p, "createPatchHandler", capsys)
    assert "--> validateSanitySession() [calls]" in out
    assert "<-- " not in out


def test_explain_source_file_path_prefers_file_level_node(monkeypatch, tmp_path, capsys):
    source_file = "app/api/example/route.ts"
    graph_data = {
        "schema_version": 1,
        "directed": True,
        "multigraph": False,
        "graph": {"graphify_profile": {"graph_type": "digraph"}},
        "nodes": [
            {
                "id": "example_route_get",
                "label": "GET()",
                "source_file": source_file,
                "source_location": "L42",
                "community": 0,
            },
            {
                "id": "example_route",
                "label": "route.ts",
                "source_file": source_file,
                "source_location": "L1",
                "community": 0,
            },
        ],
        "links": [
            {
                "source": "example_route",
                "target": "example_route_get",
                "relation": "contains",
                "confidence": "EXTRACTED",
                "_origin": "ast",
            },
        ],
        "hyperedges": [],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph_data), encoding="utf-8")

    out = _run(monkeypatch, p, source_file, capsys)

    assert "Node: route.ts" in out
    assert "ID:        example_route" in out
    assert f"Source:    {source_file} L1" in out
    assert "Node: GET()" not in out


# --- work-memory overlay Lesson line ------------------------------------------


def _write_sidecar(tmp_path, nodes):
    (tmp_path / ".graphify_learning.json").write_text(
        json.dumps({"version": 1, "generated_at": "2026-06-01T00:00:00+00:00", "nodes": nodes}),
        encoding="utf-8",
    )


def test_explain_shows_preferred_lesson_line(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    _write_sidecar(
        tmp_path,
        {
            "validate": {
                "status": "preferred",
                "score": 2.4,
                "uses": 3,
                "label": "validateSanitySession()",
                "source_file": "",
                "code_fingerprint": "",
                "provenance": [],
            },
        },
    )
    out = _run(monkeypatch, p, "validateSanitySession", capsys)
    assert "Lesson: preferred source (start here) — 3 useful, score=2.4" in out
    assert "code changed" not in out


def test_explain_shows_contested_and_stale_lesson(monkeypatch, tmp_path, capsys):
    p = _write_graph(tmp_path)
    # source_file points at a path that does not exist -> loader marks it stale.
    _write_sidecar(
        tmp_path,
        {
            "validate": {
                "status": "contested",
                "score": -0.1,
                "uses": 2,
                "neg": 1,
                "verdict": "dead end",
                "label": "validateSanitySession()",
                "source_file": "server/sanity-validate-session.ts",
                "code_fingerprint": "deadbeef",
                "provenance": [],
            },
        },
    )
    out = _run(monkeypatch, p, "validateSanitySession", capsys)
    assert "Lesson: contested (useful 2 / dead-end 1)" in out
    assert "[code changed since — re-verify]" in out


def test_explain_no_lesson_line_for_unannotated_node(monkeypatch, tmp_path, capsys):
    """No sidecar => no Lesson line; output identical to pre-feature."""
    p = _write_graph(tmp_path)
    out = _run(monkeypatch, p, "validateSanitySession", capsys)
    assert "Lesson:" not in out


def test_explain_supports_undirected_graph(monkeypatch, tmp_path, capsys):
    graph_data = {
        "schema_version": 1,
        "directed": False,
        "multigraph": False,
        "graph": {"graphify_profile": {"graph_type": "simple"}},
        "nodes": [{"id": "a", "label": "Alpha"}, {"id": "b", "label": "Beta"}],
        "links": [{"source": "a", "target": "b", "relation": "calls", "_origin": "ast"}],
        "hyperedges": [],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_data), encoding="utf-8")

    out = _run(monkeypatch, graph_path, "Alpha", capsys)

    assert "--> Beta [calls] []" in out


def test_explain_shows_same_relation_parallel_details(monkeypatch, tmp_path, capsys):
    graph_data = {
        "schema_version": 1,
        "directed": True,
        "multigraph": True,
        "graph": {"graphify_profile": {"graph_type": "multidigraph"}},
        "nodes": [{"id": "a", "label": "Alpha"}, {"id": "b", "label": "Beta"}],
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

    out = _run(monkeypatch, graph_path, "Alpha", capsys)

    assert "2 records" in out
    assert "key=call-L5" in out
    assert "key=call-L9" in out
