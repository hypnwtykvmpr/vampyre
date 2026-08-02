"""Edge `_origin` provenance (#1521 transition to an authoritative edge-marker).

Covers the core increment:
- extract.py stamps `_origin="ast"` on every AST edge (mirrors the node stamp),
  and it survives to graph.json.
- llm.py `_merge_into` stamps `_origin="semantic"` on LLM edges at the single
  merge choke point (setdefault — keeps an explicit tag).
- watch.py `_dedupe_rebuilt_edge_records` excludes `_origin` from its fingerprint
  (the must-fix: a fresh AST-stamped edge and a preserved legacy un-stamped edge
  for the same relationship MUST still collapse — no first-rebuild inflation).
- build.py `_norm_source_file` canonicalizes `./` / `..` (F2).
- backward compatibility: an edge with no `_origin` still round-trips/loads, and a
  future tertiary `_origin` value is tolerated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# --- extract.py stamps edges "ast" (and it reaches graph.json) ---


def test_extract_stamps_every_edge_ast(tmp_path):
    from graphify.extract import extract

    (tmp_path / "a.py").write_text(
        "import b\n\ndef use():\n    return b.thing()\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text("def thing():\n    return 1\n", encoding="utf-8")
    result = extract([tmp_path / "a.py", tmp_path / "b.py"], cache_root=tmp_path, parallel=False)
    edges = result.get("edges", [])
    assert edges, "corpus must produce at least one AST edge"
    assert all(e.get("_origin") == "ast" for e in edges), (
        "every AST-extracted edge must carry _origin='ast'"
    )
    # nodes keep their existing AST stamp too (no regression)
    assert all(n.get("_origin") == "ast" for n in result.get("nodes", []))


@pytest.mark.skipif(sys.platform == "win32", reason="git CLI behaviour varies on Windows runners")
def test_ast_edge_origin_survives_to_graph_json(tmp_path):
    """The stamp must flow through the build/write path to the on-disk graph."""
    from graphify.watch import _rebuild_code

    _git_init(tmp_path)
    (tmp_path / "a.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n", encoding="utf-8"
    )
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False) is True
        data = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)
    links = data.get("links", data.get("edges", []))
    assert links, "expected AST edges on disk"
    assert all(e.get("_origin") == "ast" for e in links), (
        "AST edge _origin must survive to graph.json (not stripped on the JSON path)"
    )


# --- llm.py _merge_into stamps "semantic" at the choke point ---


def test_merge_into_stamps_semantic():
    from graphify.llm import _merge_into

    merged = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    _merge_into(merged, {"edges": [{"source": "x", "target": "y", "relation": "depends_on"}]})
    assert merged["edges"][0]["_origin"] == "semantic", "LLM edge must be stamped semantic"


def test_merge_into_setdefault_keeps_explicit_origin():
    from graphify.llm import _merge_into

    merged = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    _merge_into(
        merged, {"edges": [{"source": "x", "target": "y", "relation": "calls", "_origin": "ast"}]}
    )
    assert merged["edges"][0]["_origin"] == "ast", "an explicit _origin must not be clobbered"


# --- watch.py dedup MUST exclude _origin (the must-fix) ---


def test_dedupe_collapses_stamped_and_unstamped_same_edge():
    """A fresh AST-stamped edge and a preserved legacy un-stamped edge for the same
    relationship must collapse to one — else the first rebuild after stamping lands
    inflates the graph with duplicate edge records."""
    from graphify.watch import _dedupe_rebuilt_edge_records

    base = {
        "source": "a",
        "target": "b",
        "relation": "calls",
        "source_file": "a.py",
        "source_location": "L5",
    }
    preserved_legacy = dict(base)  # no _origin (old graph)
    fresh_stamped = dict(base, _origin="ast")  # new extraction
    out = _dedupe_rebuilt_edge_records([preserved_legacy, fresh_stamped])
    assert len(out) == 1, f"stamped + legacy duplicate must collapse, got {len(out)}"


def test_dedupe_keeps_distinct_relationships():
    from graphify.watch import _dedupe_rebuilt_edge_records

    e1 = {
        "source": "a",
        "target": "b",
        "relation": "calls",
        "source_file": "a.py",
        "_origin": "ast",
    }
    e2 = {
        "source": "a",
        "target": "b",
        "relation": "imports",
        "source_file": "a.py",
        "_origin": "ast",
    }
    out = _dedupe_rebuilt_edge_records([e1, e2])
    assert len(out) == 2, "distinct relations must not collapse"


# --- build.py _norm_source_file canonicalization (F2) ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a.py", "a.py"),
        ("./a.py", "a.py"),
        ("pkg/../a.py", "a.py"),
        ("pkg/./sub/x.py", "pkg/sub/x.py"),
        ("pkg\\sub\\x.py", "pkg/sub/x.py"),  # backslash -> posix
    ],
)
def test_norm_source_file_canonicalizes(raw, expected):
    from graphify.build import _norm_source_file

    assert _norm_source_file(raw) == expected


def test_norm_source_file_none_passthrough():
    from graphify.build import _norm_source_file

    assert _norm_source_file(None) is None
    assert _norm_source_file("") == ""


# --- backward compatibility + future tertiary _origin ---


@pytest.mark.skipif(sys.platform == "win32", reason="git CLI behaviour varies on Windows runners")
def test_unmarked_and_tertiary_origin_edges_load(tmp_path):
    """A legacy graph with un-stamped edges AND an edge with a FUTURE tertiary
    _origin value must both load and survive a full rebuild (no hard binary
    assumption; loader tolerates any/no marker)."""
    from graphify.watch import _rebuild_code
    from graphify.graph_loader import load_graph
    import networkx as nx

    _git_init(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False) is True
        gpath = tmp_path / "graphify-out" / "graph.json"
        data = json.loads(gpath.read_text(encoding="utf-8"))
        foo = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("foo("))
        bar = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("bar("))
        links = data.get("links", data.get("edges", []))
        # un-stamped legacy semantic edge (no _origin) — sourceless so it always survives
        links.append(
            {"source": foo, "target": bar, "relation": "relates_to", "confidence": "INFERRED"}
        )
        # FUTURE tertiary _origin value — must be tolerated, not crash, and (non-ast) preserved
        links.append(
            {
                "source": bar,
                "target": foo,
                "relation": "derived_link",
                "_origin": "derived",
                "confidence": "INFERRED",
            }
        )
        data["links"] = links
        data["multigraph"] = True
        data["directed"] = True
        gpath.write_text(json.dumps(data), encoding="utf-8")

        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False) is True
        after = json.loads(gpath.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    after_links = after.get("links", after.get("edges", []))
    rels = {e.get("relation") for e in after_links}
    assert "relates_to" in rels, "un-stamped legacy edge must survive (backward compat)"
    assert "derived_link" in rels, (
        "future tertiary _origin edge must survive (no binary assumption)"
    )
    reloaded = load_graph(after)
    assert isinstance(reloaded, nx.MultiDiGraph)


# --- CLI semantic merge stamps cached edges (doc-pipeline gap) ---


@pytest.mark.skipif(sys.platform == "win32", reason="git CLI behaviour varies on Windows runners")
def test_cli_semantic_merge_stamps_cached_edges(tmp_path):
    """Cached semantic edges bypass llm._merge_into (check_semantic_cache path),
    so the CLI merge must stamp them _origin='semantic' itself. Unstamped, a doc
    edge between two AST-resuppliable endpoints is wrongly evicted on an AST-only
    full rebuild now that markdown is structurally extracted (#1521 doc gap).
    """
    from graphify.cache import save_semantic_cache

    _git_init(tmp_path)
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "NOTES.md").write_text("# Notes\n\nalpha is documented here.\n", encoding="utf-8")
    # Seed the semantic cache with UNSTAMPED nodes/edges (the pre-stamp cache
    # format) so check_semantic_cache merges them without any LLM call.
    save_semantic_cache(
        nodes=[
            {
                "id": "notes_doc",
                "label": "NOTES",
                "type": "document",
                "source_file": "NOTES.md",
                "file_type": "document",
            },
            {
                "id": "notes_topic",
                "label": "alpha docs",
                "type": "concept",
                "source_file": "NOTES.md",
                "file_type": "document",
            },
        ],
        edges=[
            {
                "source": "notes_doc",
                "target": "notes_topic",
                "relation": "references",
                "source_file": "NOTES.md",
                "confidence": "EXTRACTED",
                "weight": 1.0,
            },
        ],
        root=tmp_path,
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.endswith("_API_KEY") or k.startswith(("AWS_", "AZURE_", "OLLAMA")))
    }
    # The CLI demands a key whenever doc files exist, BEFORE consulting the
    # cache. The seeded cache hit means this dummy key is never used — a cache
    # MISS would hit the backend with a bogus key and fail the returncode
    # assertion loudly (which is what we want: the test's premise broke).
    env["GEMINI_API_KEY"] = "dummy-never-used-cache-hit-expected"
    proc = subprocess.run(
        [sys.executable, "-m", "graphify", "extract", str(tmp_path), "--no-cluster"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"extract failed:\n{proc.stdout}\n{proc.stderr}"
    assert "semantic cache: 1 hit" in proc.stdout, (
        f"expected a pure cache hit (no LLM call):\n{proc.stdout}\n{proc.stderr}"
    )
    data = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    links = data.get("links", data.get("edges", []))
    sem = [e for e in links if e.get("source") == "notes_doc"]
    assert sem, f"cached semantic edge must reach graph.json\n{proc.stdout}\n{proc.stderr}"
    assert all(e.get("_origin") == "semantic" for e in sem), (
        "cached semantic edges must be stamped _origin='semantic' by the CLI merge"
    )


# --- helper ---


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"], check=True)
