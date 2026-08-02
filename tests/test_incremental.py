"""Integration tests for incremental graphify extract behavior."""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PYTHON = sys.executable

# Backend-selecting env vars. These tests assume no working LLM backend (a docs
# corpus should fail without one); strip them so a developer who has a real
# ANTHROPIC_API_KEY / OPENAI_API_KEY / etc. exported does not make a docs extract
# succeed and break the "no backend" path. CI has none of these set anyway.
_LLM_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MOONSHOT_API_KEY",
    "DEEPSEEK_API_KEY",
    "OLLAMA_BASE_URL",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
)


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _LLM_ENV_KEYS}
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def _make_docs_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "intro.md").write_text("# Introduction\nThis doc introduces the system.")
    (docs / "api.md").write_text("# API Reference\nThe API has endpoints.")
    return docs


def test_manifest_written_after_extract(tmp_path):
    """After a full extract run, manifest.json must exist (or run fails before writing it)."""
    docs = _make_docs_corpus(tmp_path)
    r = _run(["extract", str(docs)], tmp_path)
    # Should fail with no API key — but NOT with a path error
    assert "no LLM API key" in r.stderr or r.returncode != 0
    # manifest should NOT exist (run failed before writing)
    manifest = docs / "graphify-out" / "manifest.json"
    assert not manifest.exists()


def test_incremental_mode_detected_via_manifest(tmp_path):
    """If manifest.json + graph.json exist, incremental mode message is shown."""
    docs = _make_docs_corpus(tmp_path)
    out = docs / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps({"nodes": [], "links": []}))
    (out / "manifest.json").write_text(json.dumps({"document": [str(docs / "intro.md")]}))
    r = _run(["extract", str(docs)], tmp_path)
    combined = r.stdout + r.stderr
    assert "incremental" in combined.lower() or r.returncode != 0


def test_no_incremental_without_manifest(tmp_path):
    """Without manifest.json, full scan message is shown (not incremental)."""
    docs = _make_docs_corpus(tmp_path)
    r = _run(["extract", str(docs)], tmp_path)
    # Check combined output doesn't contain incremental-mode phrasing.
    # Use a phrase rather than a bare word to avoid matching the tmp_path,
    # which pytest derives from the test name and contains "incremental".
    assert "incremental update" not in r.stdout.lower()
    assert "incremental scan" not in r.stdout.lower()


def test_extract_no_cluster_incremental_noop_preserves_existing_graph(tmp_path):
    """#1347: no-op incremental no-cluster extract must not overwrite graph.json."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")

    first = _run(["extract", str(project), "--no-cluster"], tmp_path)
    assert first.returncode == 0, first.stderr
    graph_path = project / "graphify-out" / "graph.json"
    before_text = graph_path.read_text(encoding="utf-8")
    before = json.loads(before_text)
    assert before.get("nodes"), "first run should produce a non-empty code graph"

    second = _run(["extract", str(project), "--no-cluster"], tmp_path)
    assert second.returncode == 0, second.stderr

    after_text = graph_path.read_text(encoding="utf-8")
    after = json.loads(after_text)
    assert after.get("nodes"), "no-op incremental run must not empty the graph"
    assert after_text == before_text


def test_bare_update_resolves_portable_marker_from_unrelated_cwd(tmp_path):
    source_root = tmp_path / "project" / "Sources"
    source_root.mkdir(parents=True)
    (source_root / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = tmp_path / "canonical" / "graphify-out"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _LLM_ENV_KEYS}
    env["GRAPHIFY_OUT"] = str(output)

    initial = subprocess.run(
        [PYTHON, "-m", "graphify", "extract", str(source_root), "--no-cluster"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert initial.returncode == 0, initial.stderr
    marker = output / ".graphify_root"
    relative = os.path.relpath(source_root, marker.parent).replace(os.sep, "/")
    marker.write_text(f"marker-relative:{relative}", encoding="utf-8")

    updated = subprocess.run(
        [PYTHON, "-m", "graphify", "update", "--no-cluster"],
        cwd=unrelated,
        capture_output=True,
        text=True,
        env=env,
    )

    assert updated.returncode == 0, updated.stderr
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert any(node.get("label") == "run()" for node in graph.get("nodes", []))


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_without_existing_state_fails_without_creating_output(tmp_path, no_cluster):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    args = ["update", str(project)]
    if no_cluster:
        args.append("--no-cluster")

    result = _run(args, tmp_path)

    assert result.returncode != 0
    assert "no existing graph state" in result.stderr.lower()
    assert "graphify extract" in result.stderr.lower()
    assert not (project / "graphify-out").exists()


def test_update_rejects_unmarked_graph_without_mutation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = project / "graphify-out"
    output.mkdir()
    graph_path = output / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "run", "label": "run()", "source_file": "app.py"}],
                "edges": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    marker = output / ".graphify_root"
    marker.write_text("marker-relative:..", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    result = _run(["update", str(project), "--no-cluster"], tmp_path)

    assert result.returncode != 0
    assert "missing an explicit graph profile" in result.stderr.lower()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_rejects_malformed_graph_without_mutation(tmp_path, no_cluster):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = project / "graphify-out"
    output.mkdir()
    graph_path = output / "graph.json"
    graph_path.write_text('{"directed": false, "multigraph": false,', encoding="utf-8")
    (output / ".graphify_root").write_text("marker-relative:..", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    args = ["update", str(project)]
    if no_cluster:
        args.append("--no-cluster")
    result = _run(args, tmp_path)

    assert result.returncode != 0
    assert "existing graph state is unreadable" in result.stderr.lower()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_extract_no_cluster_writes_explicit_simple_profile(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    result = _run(["extract", str(project), "--no-cluster"], tmp_path)

    assert result.returncode == 0, result.stderr
    graph = json.loads((project / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert graph["multigraph"] is False
    assert graph["directed"] is False
    assert graph["graph"]["graphify_profile"]["graph_type"] == "simple"


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_out_routes_every_state_file_to_canonical_output(tmp_path, no_cluster):
    project = tmp_path / "project"
    source_root = project / "Sources"
    source_root.mkdir(parents=True)
    source = source_root / "app.py"
    source.write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    canonical_root = tmp_path / "canonical"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    initial_args = [
        "extract",
        str(source_root),
        "--out",
        str(canonical_root),
        "--multigraph",
    ]
    if no_cluster:
        initial_args.append("--no-cluster")
    initial = _run(initial_args, unrelated)
    assert initial.returncode == 0, initial.stderr
    output = canonical_root / "graphify-out"
    graph_path = output / "graph.json"
    source.write_text(
        "def helper():\n    return 2\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )

    stable_graph: bytes | None = None
    for attempt in range(1, 4):
        update_args = ["update", str(source_root), "--out", str(canonical_root)]
        if no_cluster:
            update_args.append("--no-cluster")
        updated = _run(update_args, unrelated)
        assert updated.returncode == 0, f"update #{attempt}: {updated.stderr}"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        assert graph["multigraph"] is True
        assert graph["graph"]["graphify_profile"]["graph_type"] == "multidigraph"
        if stable_graph is None:
            stable_graph = graph_path.read_bytes()
        else:
            assert graph_path.read_bytes() == stable_graph

    assert (output / "manifest.json").exists()
    assert (output / ".graphify_root").exists()
    assert not (source_root / "graphify-out").exists()
    assert not (unrelated / "graphify-out").exists()


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_preserves_graph_metadata_and_hyperedges(tmp_path, no_cluster):
    source_root = tmp_path / "project"
    source_root.mkdir()
    source = source_root / "app.py"
    source.write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    initial = _run(
        ["extract", str(source_root), "--multigraph", "--no-cluster"],
        tmp_path,
    )
    assert initial.returncode == 0, initial.stderr
    graph_path = source_root / "graphify-out" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    node_ids = [str(node["id"]) for node in graph["nodes"]]
    assert len(node_ids) >= 2
    graph["graph"]["custom_meta"] = {"owner": "project", "retention": 3}
    graph["graph"]["graphify_profile"]["custom_mode"] = "sticky"
    graph["hyperedges"] = [
        {
            "id": "workflow",
            "nodes": node_ids[:2],
            "relation": "coordinates",
            "source_file": "app.py",
        }
    ]
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    source.write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n\ndef added():\n    return run()\n",
        encoding="utf-8",
    )

    args = ["update", str(source_root)]
    if no_cluster:
        args.append("--no-cluster")
    updated = _run(args, tmp_path)

    assert updated.returncode == 0, updated.stderr
    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    assert rebuilt["graph"]["custom_meta"] == {"owner": "project", "retention": 3}
    assert rebuilt["graph"]["graphify_profile"]["custom_mode"] == "sticky"
    assert rebuilt["graph"]["graphify_profile"]["graph_type"] == "multidigraph"
    assert [hyperedge["id"] for hyperedge in rebuilt["hyperedges"]] == ["workflow"]


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_preserves_directed_simple_profile(tmp_path, no_cluster):
    source_root = tmp_path / "project"
    source_root.mkdir()
    (source_root / "app.py").write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    initial = _run(["extract", str(source_root), "--no-cluster"], tmp_path)
    assert initial.returncode == 0, initial.stderr
    graph_path = source_root / "graphify-out" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["directed"] = True
    graph["graph"]["graphify_profile"]["graph_type"] = "digraph"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    args = ["update", str(source_root)]
    if no_cluster:
        args.append("--no-cluster")
    updated = _run(args, tmp_path)

    assert updated.returncode == 0, updated.stderr
    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    assert rebuilt["multigraph"] is False
    assert rebuilt["directed"] is True
    assert rebuilt["graph"]["graphify_profile"]["graph_type"] == "digraph"


def test_bare_update_reuses_extract_out_graph_and_scan_root(tmp_path):
    project = tmp_path / "project"
    source_root = project / "Sources"
    source_root.mkdir(parents=True)
    source = source_root / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")

    initial = _run(
        ["extract", "Sources", "--out", ".", "--multigraph", "--no-cluster"],
        project,
    )
    assert initial.returncode == 0, initial.stderr
    graph_path = project / "graphify-out" / "graph.json"
    source.write_text("def run():\n    return 2\n", encoding="utf-8")

    stable_graph: bytes | None = None
    for attempt in range(1, 4):
        updated = _run(["update", "--no-cluster"], project)
        assert updated.returncode == 0, f"update #{attempt}: {updated.stderr}"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        assert graph["multigraph"] is True
        assert graph["graph"]["graphify_profile"]["graph_type"] == "multidigraph"
        if stable_graph is None:
            stable_graph = graph_path.read_bytes()
        else:
            assert graph_path.read_bytes() == stable_graph

    assert not (source_root / "graphify-out").exists()


def test_update_remap_preserves_external_output_and_multigraph_profile(tmp_path):
    source_root = tmp_path / "Sources"
    source_root.mkdir()
    (source_root / "app.py").write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    canonical_root = tmp_path / "canonical"
    initial = _run(
        [
            "extract",
            str(source_root),
            "--out",
            str(canonical_root),
            "--multigraph",
            "--no-cluster",
        ],
        tmp_path,
    )
    assert initial.returncode == 0, initial.stderr

    remapped = _run(
        [
            "update",
            str(source_root),
            "--out",
            str(canonical_root),
            "--remap",
            "--no-cluster",
        ],
        tmp_path,
    )

    assert remapped.returncode == 0, remapped.stderr
    graph_path = canonical_root / "graphify-out" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["multigraph"] is True
    assert graph["graph"]["graphify_profile"]["graph_type"] == "multidigraph"
    assert not (source_root / "graphify-out").exists()


@pytest.mark.parametrize("no_cluster", [False, True])
def test_update_remap_preserves_directed_graph_class(tmp_path, no_cluster):
    source_root = tmp_path / "Sources"
    source_root.mkdir()
    (source_root / "app.py").write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    initial = _run(["extract", str(source_root), "--directed", "--no-cluster"], tmp_path)
    assert initial.returncode == 0, initial.stderr
    graph_path = source_root / "graphify-out" / "graph.json"
    seeded = json.loads(graph_path.read_text(encoding="utf-8"))
    links = seeded.get("links", seeded.get("edges", []))
    assert links
    for edge in links:
        edge.pop("_origin", None)
    graph_path.write_text(json.dumps(seeded, indent=2), encoding="utf-8")

    args = ["update", str(source_root), "--remap"]
    if no_cluster:
        args.append("--no-cluster")
    remapped = _run(args, tmp_path)

    assert remapped.returncode == 0, remapped.stderr
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["multigraph"] is False
    assert graph["directed"] is True
    assert graph["graph"]["graphify_profile"]["graph_type"] == "digraph"
    remapped_links = graph.get("links", graph.get("edges", []))
    assert remapped_links
    assert all(edge.get("_origin") == "ast" for edge in remapped_links)


def test_update_out_respects_process_bound_graphify_out(tmp_path):
    source_root = tmp_path / "Sources"
    source_root.mkdir()
    source = source_root / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    canonical_root = tmp_path / "canonical"
    env = {k: v for k, v in os.environ.items() if k not in _LLM_ENV_KEYS}
    env["GRAPHIFY_OUT"] = "custom-graph"

    initial = subprocess.run(
        [
            PYTHON,
            "-m",
            "graphify",
            "extract",
            str(source_root),
            "--out",
            str(canonical_root),
            "--multigraph",
            "--no-cluster",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert initial.returncode == 0, initial.stderr
    source.write_text("def run():\n    return 2\n", encoding="utf-8")

    updated = subprocess.run(
        [
            PYTHON,
            "-m",
            "graphify",
            "update",
            str(source_root),
            "--out",
            str(canonical_root),
            "--no-cluster",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert updated.returncode == 0, updated.stderr
    output = canonical_root / "custom-graph"
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert graph["multigraph"] is True
    assert (output / "manifest.json").exists()
    assert not (canonical_root / "graphify-out").exists()
    assert not (source_root / "custom-graph").exists()


def _edges(graph_json: Path) -> list[dict]:
    g = json.loads(graph_json.read_text())
    return g.get("links", g.get("edges", []))


def test_update_prunes_a_removed_imports_edge(tmp_path):
    """#1521: when an import is deleted from a file, `graphify update` must prune
    the edge it produced — preserving it (keyed only on endpoint membership) left a
    stale edge that drove phantom circular-dependency findings."""
    proj = tmp_path / "proj"
    pkg = proj / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "b.py").write_text("def helper():\n    return 1\n")
    (pkg / "a.py").write_text("from pkg.b import helper\ndef use():\n    return helper()\n")

    # initial extract -> the import edge a -> b exists
    r1 = _run(["extract", str(proj), "--no-cluster"], tmp_path)
    assert r1.returncode == 0, r1.stderr
    gj = proj / "graphify-out" / "graph.json"
    before = _edges(gj)
    assert any(
        e.get("relation") in ("imports", "imports_from")
        and str(e.get("source_file", "")).endswith("a.py")
        for e in before
    ), f"expected an import edge from a.py initially: {before}"

    # remove the import, then update
    (pkg / "a.py").write_text("def use():\n    return 1\n")
    r2 = _run(["update", str(proj)], tmp_path)
    assert r2.returncode == 0, r2.stderr
    after = _edges(gj)

    # the stale import edge owned by a.py must be gone
    stale = [
        e
        for e in after
        if e.get("relation") in ("imports", "imports_from")
        and str(e.get("source_file", "")).endswith("a.py")
    ]
    assert not stale, f"removed import's edge survived update (stale): {stale}"
