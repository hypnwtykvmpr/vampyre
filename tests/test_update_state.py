"""Tests for strict update-state resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify.paths import write_scan_root_marker
from graphify.update_state import UpdateStateError, resolve_update_context


def _write_state(
    source_root: Path,
    output_root: Path,
    *,
    graph_type: str = "simple",
) -> Path:
    source_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "graphify-out"
    output.mkdir(parents=True, exist_ok=True)
    multigraph = graph_type == "multidigraph"
    directed = graph_type in {"digraph", "multidigraph"}
    (output / "graph.json").write_text(
        json.dumps(
            {
                "directed": directed,
                "multigraph": multigraph,
                "graph": {"graphify_profile": {"graph_type": graph_type}},
                "nodes": [],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    write_scan_root_marker(output / ".graphify_root", source_root)
    return output


@pytest.mark.parametrize("graph_type", ["simple", "digraph", "multidigraph"])
def test_resolve_update_context_preserves_declared_graph_type(tmp_path, graph_type):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_state(source, output_root, graph_type=graph_type)

    context = resolve_update_context(source, output_root=output_root)

    assert context.scan_root == source.resolve()
    assert context.output_dir == output.resolve()
    assert context.graph_path == output.resolve() / "graph.json"
    assert context.manifest_path == output.resolve() / "manifest.json"
    assert context.graph_type == graph_type


def test_resolve_update_context_rejects_conflicting_explicit_scan(tmp_path):
    source = tmp_path / "Sources"
    other = tmp_path / "Other"
    other.mkdir()
    output_root = tmp_path / "canonical"
    _write_state(source, output_root)

    with pytest.raises(UpdateStateError, match="conflicts with the graph marker"):
        resolve_update_context(other, output_root=output_root)


@pytest.mark.parametrize("marker_value", ["", "marker-relative:", "marker-relative:/absolute"])
def test_resolve_update_context_rejects_invalid_marker(tmp_path, marker_value):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_state(source, output_root)
    (output / ".graphify_root").write_text(marker_value, encoding="utf-8")

    with pytest.raises(UpdateStateError, match="marker is unreadable or invalid"):
        resolve_update_context(output_root=output_root)


def test_resolve_update_context_rejects_profile_conflict(tmp_path):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_state(source, output_root)
    graph_path = output / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["graph"]["graphify_profile"]["graph_type"] = "multidigraph"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(UpdateStateError, match="profile conflicts"):
        resolve_update_context(output_root=output_root)


def test_resolve_update_context_accepts_runtime_output_directory(tmp_path):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_state(source, output_root)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    context = resolve_update_context(output_dir=output, cwd=unrelated)

    assert context.scan_root == source.resolve()
    assert context.output_root == output_root.resolve()
    assert context.output_dir == output.resolve()
