"""Tests for strict update-state resolution."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from graphify.paths import resolve_scan_root_marker, write_scan_root_marker
from graphify.update_state import UpdateStateError, repair_update_state, resolve_update_context


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


@pytest.mark.parametrize("profile", [None, {}, {"custom": "value"}])
def test_resolve_update_context_rejects_graph_without_explicit_fork_profile(tmp_path, profile):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_state(source, output_root)
    graph_path = output / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if profile is None:
        graph["graph"].pop("graphify_profile")
    else:
        graph["graph"]["graphify_profile"] = profile
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(UpdateStateError, match="explicit graphify profile"):
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


def _write_repairable_state(source_root: Path, output_root: Path) -> Path:
    source_root.mkdir(parents=True, exist_ok=True)
    source = source_root / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    output = _write_state(source_root, output_root, graph_type="multidigraph")
    (output / ".graphify_root").unlink()
    digest = hashlib.md5(source.read_bytes(), usedforsecurity=False).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "app.py": {
                    "mtime": source.stat().st_mtime,
                    "ast_hash": digest,
                    "semantic_hash": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    return output


def test_repair_update_state_writes_only_missing_marker(tmp_path):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_repairable_state(source, output_root)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}

    context = repair_update_state(source, output_root=output_root)

    assert context.scan_root == source.resolve()
    assert context.graph_type == "multidigraph"
    assert resolve_scan_root_marker(output / ".graphify_root") == source.resolve()
    after = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file() and path.name != ".graphify_root"
    }
    assert after == before


@pytest.mark.parametrize("defect", ["missing_manifest", "invalid_profile", "wrong_root"])
def test_repair_update_state_refuses_unverifiable_state_without_mutation(tmp_path, defect):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_repairable_state(source, output_root)
    repair_source = source
    if defect == "missing_manifest":
        (output / "manifest.json").unlink()
    elif defect == "invalid_profile":
        graph_path = output / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["graph"]["graphify_profile"]["graph_type"] = "simple"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
    else:
        repair_source = tmp_path / "Other"
        repair_source.mkdir()

    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    with pytest.raises(UpdateStateError):
        repair_update_state(repair_source, output_root=output_root)

    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (output / ".graphify_root").exists()


def test_repair_update_state_rejects_conflicting_existing_marker(tmp_path):
    source = tmp_path / "Sources"
    other = tmp_path / "Other"
    other.mkdir()
    output_root = tmp_path / "canonical"
    output = _write_repairable_state(source, output_root)
    write_scan_root_marker(output / ".graphify_root", other)
    before = (output / ".graphify_root").read_bytes()

    with pytest.raises(UpdateStateError, match="conflicts"):
        repair_update_state(source, output_root=output_root)

    assert (output / ".graphify_root").read_bytes() == before


def test_repair_update_state_rejects_partially_matching_manifest(tmp_path):
    source = tmp_path / "Sources"
    output_root = tmp_path / "canonical"
    output = _write_repairable_state(source, output_root)
    second = source / "second.py"
    second.write_text("def second():\n    return 2\n", encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["second.py"] = {
        "mtime": second.stat().st_mtime,
        "ast_hash": "0" * 32,
        "semantic_hash": "0" * 32,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UpdateStateError, match="does not match the requested scan root"):
        repair_update_state(source, output_root=output_root)

    assert not (output / ".graphify_root").exists()
