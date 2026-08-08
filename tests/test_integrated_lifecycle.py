from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import networkx as nx
import pytest

from graphify.export import to_json
from graphify.graph_loader import load_graph_state_file
from graphify.graph_state import DecodeMode, GraphStateError
from graphify.projections import edge_records_between


pytestmark = pytest.mark.xdist_group("integrated-lifecycle")
_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path, *, env: dict[str, str] | None = None):
    clean = dict(os.environ) if env is None else dict(env)
    for key in tuple(clean):
        if key.endswith("_API_KEY") or key in {"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"}:
            clean.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "graphify", *args],
        cwd=cwd,
        env=clean,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def _graph_type(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["graph"]["graphify_profile"]["graph_type"]


async def _mcp_graph_stats(graph_path: Path, stderr_path: Path) -> str:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "graphify.serve", str(graph_path)],
        cwd=str(_ROOT),
    )
    with stderr_path.open("w", encoding="utf-8") as stderr:
        async with stdio_client(parameters, errlog=stderr) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool("graph_stats", {})
    assert result.isError is False
    return "\n".join(getattr(item, "text", "") for item in result.content)


@pytest.mark.parametrize(
    ("class_flag", "expected_type", "no_cluster", "external_output"),
    [
        ("--simple", "simple", False, False),
        ("--simple", "simple", True, True),
        ("--directed", "digraph", False, True),
        ("--directed", "digraph", True, False),
        ("--multigraph", "multidigraph", False, False),
        ("--multigraph", "multidigraph", True, True),
    ],
)
def test_extract_update_read_export_lifecycle_matrix(
    tmp_path: Path,
    class_flag: str,
    expected_type: str,
    no_cluster: bool,
    external_output: bool,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def helper():\n    return 1\n\ndef main():\n    helper()\n    return helper()\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "canonical" if external_output else corpus
    output_args = ["--out", str(output_root)] if external_output else []
    mode_args = [class_flag, *(["--no-cluster"] if no_cluster else []), *output_args]

    _assert_ok(_run(["extract", str(corpus), *mode_args], tmp_path))
    graph_path = output_root / "graphify-out" / "graph.json"
    state = load_graph_state_file(graph_path, mode=DecodeMode.STRICT_CURRENT)
    assert state.graph_type == expected_type
    assert _graph_type(graph_path) == expected_type
    assert (output_root / "graphify-out" / ".graphify_root").exists()
    assert (output_root / "graphify-out" / "manifest.json").exists()

    if expected_type == "multidigraph":
        call_groups: dict[tuple[object, object], int] = {}
        for source, target, data in state.graph.edges(data=True):
            if data.get("relation") == "calls":
                pair = (source, target)
                call_groups[pair] = call_groups.get(pair, 0) + 1
        assert max(call_groups.values(), default=0) == 2

    update_args = ["update", str(corpus), *(["--no-cluster"] if no_cluster else []), *output_args]
    stable_bytes = []
    for _ in range(3):
        _assert_ok(_run(update_args, tmp_path))
        stable_bytes.append(graph_path.read_bytes())
    assert stable_bytes[0] == stable_bytes[1] == stable_bytes[2]

    for command in (
        ["query", "helper", "--graph", str(graph_path)],
        ["path", "main", "helper", "--graph", str(graph_path)],
        ["explain", "helper", "--graph", str(graph_path)],
    ):
        _assert_ok(_run(command, tmp_path))

    stats = asyncio.run(_mcp_graph_stats(graph_path, tmp_path / "mcp-stderr.log"))
    assert "Nodes:" in stats and "Edges:" in stats

    _assert_ok(
        _run(
            [
                "cluster-only",
                str(corpus),
                "--graph",
                str(graph_path),
                "--no-viz",
                "--no-label",
            ],
            tmp_path,
        )
    )
    clustered = load_graph_state_file(graph_path, mode=DecodeMode.STRICT_CURRENT)
    assert clustered.graph_type == expected_type
    assert clustered.nodes
    assert all(isinstance(record.get("community"), int) for record in clustered.nodes)
    assert (graph_path.parent / "GRAPH_REPORT.md").exists()
    assert (graph_path.parent / ".graphify_analysis.json").exists()
    if external_output:
        assert not (corpus / "graphify-out").exists()

    _assert_ok(_run(["export", "wiki", "--graph", str(graph_path)], tmp_path))
    assert (graph_path.parent / "wiki" / "index.md").exists()


def _write_multigraph(path: Path, prefix: str) -> None:
    graph = nx.MultiDiGraph()
    graph.graph.update(
        {
            "graphify_profile": {"graph_type": "multidigraph", "extension": "keep"},
            "custom_meta": {"owner": prefix},
            "hyperedges": [{"id": f"{prefix}-flow", "nodes": [f"{prefix}a", f"{prefix}b"]}],
        }
    )
    graph.add_node(f"{prefix}a", label=f"{prefix}a", source_file=f"{prefix}.py", _origin="ast")
    graph.add_node(f"{prefix}b", label=f"{prefix}b", source_file=f"{prefix}.md", _origin="semantic")
    graph.add_edge(
        f"{prefix}a",
        f"{prefix}b",
        key="call-L4",
        relation="calls",
        source_file=f"{prefix}.py",
        source_location="L4",
        _origin="ast",
    )
    graph.add_edge(
        f"{prefix}a",
        f"{prefix}b",
        key="call-L8",
        relation="calls",
        source_file=f"{prefix}.py",
        source_location="L8",
        _origin="semantic",
        semantic_provenance={"backend": "test", "model": "deterministic"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    assert to_json(graph, {0: list(graph)}, str(path), force=True)


def test_merge_global_and_parallel_read_lifecycle(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    current = tmp_path / "current.json"
    other = tmp_path / "other.json"
    empty = nx.MultiDiGraph()
    assert to_json(empty, {}, str(base), force=True)
    _write_multigraph(current, "x")
    _write_multigraph(other, "y")

    snapshots = []
    for _ in range(3):
        _assert_ok(_run(["merge-driver", str(base), str(current), str(other)], tmp_path))
        snapshots.append(current.read_bytes())
    assert snapshots[0] == snapshots[1] == snapshots[2]

    merged = load_graph_state_file(current, mode=DecodeMode.STRICT_CURRENT)
    assert merged.graph_type == "multidigraph"
    assert len(merged.hyperedges) == 2
    assert len(edge_records_between(merged.graph, "xa", "xb")) == 2
    assert {
        record["source_location"] for record in edge_records_between(merged.graph, "xa", "xb")
    } == {
        "L4",
        "L8",
    }

    merged_path = tmp_path / "merged.json"
    _assert_ok(
        _run(
            [
                "merge-graphs",
                str(current),
                "--as",
                "current",
                str(other),
                "--as",
                "other",
                "--out",
                str(merged_path),
            ],
            tmp_path,
        )
    )
    merged_batch = load_graph_state_file(merged_path, mode=DecodeMode.STRICT_CURRENT)
    assert merged_batch.graph_type == "multidigraph"
    assert merged_batch.graph.number_of_edges() >= 4

    global_dir = tmp_path / "global"
    monkeypatch.setattr("graphify.global_graph._GLOBAL_DIR", global_dir)
    monkeypatch.setattr("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json")
    monkeypatch.setattr(
        "graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"
    )
    from graphify.global_graph import _load_global_graph, global_add, global_remove

    first = global_add(current, "repo-x")
    assert first["skipped"] is False
    assert global_add(current, "repo-x")["skipped"] is True
    assert global_add(other, "repo-y")["skipped"] is False
    composed = _load_global_graph()
    assert isinstance(composed, nx.MultiDiGraph)
    # repo-x is the already merged x+y state (4 edges, 2 hyperedges); repo-y is
    # an independently namespaced copy of y (2 edges, 1 hyperedge).
    assert composed.number_of_edges() == 6
    assert len(composed.graph["hyperedges"]) == 3
    assert global_remove("repo-x") == 4
    assert all(data.get("repo") == "repo-y" for _, data in _load_global_graph().nodes(data=True))


def test_current_legacy_and_ambiguous_state_matrix(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    _write_multigraph(current, "z")
    assert (
        load_graph_state_file(current, mode=DecodeMode.STRICT_CURRENT).graph_type == "multidigraph"
    )

    payload = json.loads(current.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    payload["edges"] = payload.pop("links")
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    migrated = load_graph_state_file(legacy, mode=DecodeMode.MIGRATE_LEGACY)
    assert migrated.graph_type == "multidigraph"
    assert migrated.graph.number_of_edges() == 2

    ambiguous_payload = dict(payload)
    ambiguous_payload["links"] = [
        {"source": "za", "target": "zb", "key": "different", "relation": "imports"}
    ]
    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps(ambiguous_payload), encoding="utf-8")
    with pytest.raises(GraphStateError, match="edges.*links|links.*edges"):
        load_graph_state_file(ambiguous, mode=DecodeMode.MIGRATE_LEGACY)
