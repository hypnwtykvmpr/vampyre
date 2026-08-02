"""`graphify merge-graphs` tolerates inputs that disagree on graph type (#1606).

Per-repo graph.json files written by different extract paths at different times
don't always agree on the `directed` / `multigraph` flags. compose requires one
uniform type, so a mixed set used to crash with an unhandled NetworkXError.

Fork semantics (diverges from upstream's fix): instead of collapsing every
input to a plain undirected Graph, `normalize_graphs_for_global` lifts the
batch to the common LOSSLESS class — multidigraph if any input is multi, else
digraph if any is directed — so no parallel edges or edge directions are
silently dropped. Upstream's collapse behavior remains available behind the
explicit `--simple` flag (which warns before projecting).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(args, cwd):
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args, cwd=cwd, capture_output=True, text=True
    )


def _write(p: Path, directed: bool, multigraph: bool, node_id: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "directed": directed,
                "multigraph": multigraph,
                "graph": {},
                "nodes": [{"id": node_id}],
                "links": [],
            }
        )
    )


def test_merge_graphs_mixed_directed_and_multigraph(tmp_path):
    a = tmp_path / "r1" / "graphify-out" / "graph.json"
    b = tmp_path / "r2" / "graphify-out" / "graph.json"
    c = tmp_path / "r3" / "graphify-out" / "graph.json"
    _write(a, directed=True, multigraph=False, node_id="x")  # DiGraph
    _write(b, directed=False, multigraph=False, node_id="y")  # Graph
    _write(c, directed=False, multigraph=True, node_id="z")  # MultiGraph
    out = tmp_path / "merged.json"

    r = _run(["merge-graphs", str(a), str(b), str(c), "--out", str(out)], tmp_path)
    assert r.returncode == 0, f"merge crashed: {r.stderr}"
    assert out.exists()
    data = json.loads(out.read_text())
    ids = {n["id"] for n in data["nodes"]}
    # every input's node survives, lifted into one common LOSSLESS class:
    # any-multi + any-directed => multidigraph (no silent collapse)
    assert {"r1::x", "r2::y", "r3::z"} <= ids or len(ids) == 3
    assert data.get("directed") is True
    assert data.get("multigraph") is True
    profile = (data.get("graph") or {}).get("graphify_profile") or {}
    assert profile.get("graph_type") == "multidigraph"


def test_merge_graphs_mixed_explicit_simple_collapses(tmp_path):
    """Upstream's collapse-to-simple behavior stays available behind --simple."""
    a = tmp_path / "r1" / "graphify-out" / "graph.json"
    b = tmp_path / "r2" / "graphify-out" / "graph.json"
    c = tmp_path / "r3" / "graphify-out" / "graph.json"
    _write(a, directed=True, multigraph=False, node_id="x")  # DiGraph
    _write(b, directed=False, multigraph=False, node_id="y")  # Graph
    _write(c, directed=False, multigraph=True, node_id="z")  # MultiGraph
    out = tmp_path / "merged.json"

    r = _run(["merge-graphs", str(a), str(b), str(c), "--simple", "--out", str(out)], tmp_path)
    assert r.returncode == 0, f"merge crashed: {r.stderr}"
    data = json.loads(out.read_text())
    ids = {n["id"] for n in data["nodes"]}
    assert {"r1::x", "r2::y", "r3::z"} <= ids or len(ids) == 3
    assert data.get("directed") is False
    assert data.get("multigraph") is False
    # projecting a multigraph input down is loud, never silent
    assert "WARNING" in r.stderr
