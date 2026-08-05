"""Tests for `graphify extract` CLI dispatch path in graphify.__main__."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

import graphify.__main__ as mainmod
from graphify.build import build_from_json
from graphify.export import to_json
from graphify.graph_loader import load_graph
from graphify.llm import BACKENDS, _backend_env_keys


PYTHON = sys.executable
EXTRACT_MODULE = importlib.import_module("graphify.extract")


def _make_corpus(tmp_path):
    """Minimal corpus: one Go code file + one Markdown doc.

    Both file types are needed so semantic extraction is requested
    (docs path triggers the LLM step we want to assert against).
    """
    (tmp_path / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Notes\nThe main function entry point.\n", encoding="utf-8"
    )
    return tmp_path


def test_extract_exits_nonzero_when_all_semantic_chunks_fail(monkeypatch, tmp_path, capsys):
    """When every semantic chunk errors (e.g. backend SDK not installed),
    the CLI must exit non-zero instead of silently writing an AST-only graph.

    A missing optional backend dependency must not produce a successful
    AST-only result after all semantic chunks fail.
    """
    corpus = _make_corpus(tmp_path)
    out_dir = tmp_path / "out"

    # Stub the API-key check so the backend gate doesn't reject before we
    # reach the semantic-extraction step.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    # Patch extract_corpus_parallel to simulate "all chunks failed":
    # return an empty merged accumulator without ever invoking on_chunk_done.
    # This matches the real behavior of extract_corpus_parallel when every
    # chunk raises (the per-chunk failures print to stderr and the loop
    # continues without calling the success callback).
    def _all_chunks_failed(paths, **kwargs):
        return {
            "nodes": [],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _all_chunks_failed)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--out", str(out_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1, (
        f"expected exit code 1 when all semantic chunks fail, got {exc_info.value.code}"
    )

    stderr = capsys.readouterr().err
    assert "all semantic chunks failed" in stderr
    assert "claude" in stderr

    # No graph.json should have been written - the failure must abort before
    # the merge/cluster/write phase, not after.
    assert not (out_dir / "graphify-out" / "graph.json").exists(), (
        "graph.json must not be written when semantic extraction fails"
    )


@pytest.mark.parametrize("failure_mode", ["reported", "raised"])
def test_extract_ast_failure_exits_before_writing_state(
    monkeypatch, tmp_path, capsys, failure_mode
):
    corpus = _code_only_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"

    def failed_extract(paths, **kwargs):
        if failure_mode == "raised":
            raise RuntimeError("parser process crashed")
        return {
            "nodes": [],
            "edges": [],
            "failed_files": [str(paths[0])],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    monkeypatch.setattr(EXTRACT_MODULE, "extract", failed_extract)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--out",
            str(out_dir),
            "--no-cluster",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert "AST extraction failed" in capsys.readouterr().err
    output = out_dir / "graphify-out"
    assert not (output / "graph.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()


def test_extract_refuses_ast_result_when_source_changes_during_extraction(
    monkeypatch, tmp_path, capsys
):
    corpus = _code_only_corpus(tmp_path / "corpus")
    source = corpus / "auth.py"
    out_root = tmp_path / "out"

    def unstable_extract(paths, **kwargs):
        source.write_text("def changed_after_detection():\n    return 2\n", encoding="utf-8")
        return {
            "nodes": [
                {
                    "id": "stale",
                    "label": "stale",
                    "file_type": "code",
                    "source_file": str(paths[0]),
                }
            ],
            "edges": [],
            "failed_files": [],
            "_deferred_cache_entries": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    import importlib

    extract_module = importlib.import_module("graphify.extract")
    monkeypatch.setattr(extract_module, "extract", unstable_extract)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--out", str(out_root), "--no-cluster"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert "source files changed during extraction" in capsys.readouterr().err
    output = out_root / "graphify-out"
    assert not (output / "graph.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()


def test_extract_refuses_semantic_result_when_source_changes_during_extraction(
    monkeypatch, tmp_path, capsys
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "notes.md"
    source.write_text("# Before\n", encoding="utf-8")
    out_root = tmp_path / "out"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def unstable_semantic(paths, **kwargs):
        source.write_text("# After\n", encoding="utf-8")
        result = {
            "nodes": [
                {
                    "id": "notes",
                    "label": "Notes",
                    "file_type": "document",
                    "source_file": str(paths[0]),
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 1,
            "output_tokens": 1,
            "failed_chunks": 0,
            "incomplete_chunks": 0,
        }
        kwargs["on_chunk_done"](0, 1, result)
        return result

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", unstable_semantic)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--backend",
            "claude",
            "--out",
            str(out_root),
            "--no-cluster",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert "source files changed during extraction" in capsys.readouterr().err
    output = out_root / "graphify-out"
    assert not (output / "graph.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()


@pytest.mark.parametrize("no_cluster", [True, False])
def test_extract_manifest_failure_rolls_back_all_published_state(
    monkeypatch, tmp_path, capsys, no_cluster
):
    corpus = _code_only_corpus(tmp_path / "corpus")
    output = corpus / "graphify-out"
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--no-cluster"],
    )
    with pytest.raises(SystemExit) as seed_exit:
        mainmod.main()
    assert seed_exit.value.code == 0

    (output / ".graphify_semantic_marker").write_text("{}", encoding="utf-8")
    source = corpus / "auth.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\ndef added():\n    return 1\n", encoding="utf-8"
    )
    tree_before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    def fail_manifest(*args, **kwargs):
        raise OSError("simulated manifest publication failure")

    import importlib

    detect_module = importlib.import_module("graphify.detect")
    monkeypatch.setattr(detect_module, "save_manifest", fail_manifest)
    argv = ["graphify", "extract", str(corpus)]
    if no_cluster:
        argv.append("--no-cluster")
    monkeypatch.setattr(mainmod.sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert "simulated manifest publication failure" in capsys.readouterr().err
    tree_after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before


def test_extract_corrupt_existing_profile_is_refused_without_mutation(
    monkeypatch, tmp_path, capsys
):
    corpus = _code_only_corpus(tmp_path / "corpus")
    output = corpus / "graphify-out"
    output.mkdir()
    graph = output / "graph.json"
    original = b"{not-json"
    graph.write_bytes(original)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--no-cluster"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert "could not inspect existing graph.json profile" in capsys.readouterr().err
    assert graph.read_bytes() == original
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()
    assert list((output / "cache").rglob("*.json")) == []


def test_extract_succeeds_when_at_least_one_chunk_completes(monkeypatch, tmp_path):
    """Sanity counter-test: a successful chunk run keeps exit 0. Confirms the
    new guard only fires on the all-failed path, not on every extract."""
    corpus = _make_corpus(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def _one_chunk_succeeded(paths, **kwargs):
        on_chunk = kwargs.get("on_chunk_done")
        if on_chunk:
            on_chunk(0, 1, {"nodes": [], "edges": [], "hyperedges": []})
        return {
            "nodes": [],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _one_chunk_succeeded)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--out", str(out_dir)],
    )

    # extract may still raise SystemExit at the end (clean exit code 0)
    # depending on platform; accept either no exception or SystemExit(0).
    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"unexpected exit code {exc.code}"

    # graph.json should exist on the happy path
    assert (out_dir / "graphify-out" / "graph.json").exists(), (
        "graph.json must be written on the happy path"
    )


@pytest.mark.parametrize(
    ("failed_chunks", "incomplete_chunks", "expected"),
    [(1, 0, "failed"), (0, 1, "incomplete")],
)
def test_extract_refuses_partial_semantic_result_before_writing_state(
    monkeypatch,
    tmp_path,
    capsys,
    failed_chunks,
    incomplete_chunks,
    expected,
):
    corpus = _make_corpus(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def partial_result(paths, **kwargs):
        on_chunk = kwargs.get("on_chunk_done")
        if on_chunk:
            on_chunk(0, 2, {"nodes": [{"id": "partial"}], "edges": []})
        return {
            "nodes": [{"id": "partial"}],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 100,
            "output_tokens": 50,
            "failed_chunks": failed_chunks,
            "incomplete_chunks": incomplete_chunks,
        }

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", partial_result)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--out", str(out_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 1
    assert expected in capsys.readouterr().err
    output = out_dir / "graphify-out"
    assert not (output / "graph.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()
    assert list((output / "cache" / "semantic").glob("*.json")) == []


def _code_only_corpus(tmp_path):
    """A corpus with only code — no docs/papers/images."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "auth.py").write_text(
        "def login(user):\n    return validate(user)\n\ndef validate(user):\n    return True\n",
        encoding="utf-8",
    )
    return tmp_path


def _clear_backend_keys(monkeypatch):
    """Clear every env var that detect_backend() or _get_backend_api_key() reads."""
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
        # bedrock: presence of any of these is treated as a valid credential
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        # ollama: a set OLLAMA_BASE_URL triggers backend detection
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_extract_codeonly_succeeds_without_api_key(monkeypatch, tmp_path):
    """A code-only corpus must run with no LLM API key.

    Regression: graphify extract validated a backend upfront and exited 1 with
    'no LLM API key found' even for a code-only corpus that never calls a model.
    The keyless AST path now runs to a written graph.json (#1122).
    """
    corpus = _code_only_corpus(tmp_path)
    out_dir = tmp_path / "out"
    _clear_backend_keys(monkeypatch)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--out", str(out_dir)],
    )

    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"unexpected exit code {exc.code}"

    graph = out_dir / "graphify-out" / "graph.json"
    assert graph.exists(), "code-only extract must write graph.json without a key"
    import json

    assert len(json.loads(graph.read_text(encoding="utf-8")).get("nodes", [])) > 0


def test_update_remap_routes_to_full_extract_and_stamps_ast(tmp_path):
    """`graphify update --remap` re-dispatches to the full `extract` pipeline so
    every edge is re-stamped with provenance (#1521). With no LLM key it degrades
    to AST-only and every edge must carry _origin='ast'. Guards the new --remap
    routing, that the extract edge-stamp reaches graph.json, and that --no-viz is
    surfaced (not silently ignored) on the remap path."""
    corpus = _make_code_corpus(tmp_path)
    seeded = _run(["extract", str(corpus), "--no-cluster"], corpus)
    assert seeded.returncode == 0, f"initial extract should succeed: {seeded.stderr}"
    graph = corpus / "graphify-out" / "graph.json"
    seeded_data = json.loads(graph.read_text(encoding="utf-8"))
    seeded_links = seeded_data.get("links", seeded_data.get("edges", []))
    assert seeded_links, "fixture must contain AST edges before provenance is removed"
    for edge in seeded_links:
        edge.pop("_origin", None)
    graph.write_text(json.dumps(seeded_data, indent=2), encoding="utf-8")

    r = _run(["update", "--remap", "--no-viz", str(corpus)], corpus)
    assert r.returncode == 0, f"update --remap should succeed: {r.stderr}"
    assert "Re-mapping" in (r.stdout + r.stderr), (
        "update --remap must route through the full-extract (remap) path, not AST-only"
    )
    assert "--no-viz is not applied with --remap" in (r.stdout + r.stderr), (
        "--no-viz must be surfaced as a note on the remap path, not silently dropped"
    )
    assert graph.exists(), "remap must write graph.json"
    data = json.loads(graph.read_text(encoding="utf-8"))
    links = data.get("links", data.get("edges", []))
    assert links, "expected AST edges from the code corpus"
    assert all(e.get("_origin") == "ast" for e in links), (
        "every edge must be stamped _origin='ast' after a remap re-extraction"
    )


def test_update_remap_forwards_semantic_reproducibility_options(monkeypatch, tmp_path):
    corpus = _code_only_corpus(tmp_path / "corpus")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--no-cluster", "--multigraph"],
    )
    with pytest.raises(SystemExit) as seed_exit:
        mainmod.main()
    assert seed_exit.value.code == 0

    captured: list[list[str]] = []
    monkeypatch.setattr(mainmod, "_cmd_extract", lambda: captured.append(list(mainmod.sys.argv)))
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "update",
            str(corpus),
            "--remap",
            "--backend",
            "claude",
            "--model=test-model",
            "--mode",
            "deep",
            "--max-workers",
            "3",
            "--token-budget=4096",
            "--max-concurrency",
            "2",
            "--api-timeout=30",
            "--resolution",
            "1.5",
            "--exclude-hubs=0.2",
            "--exclude",
            "vendor",
            "--google-workspace",
            "--dedup-llm",
            "--cargo",
            "--timing",
        ],
    )

    mainmod.main()

    assert len(captured) == 1
    forwarded = captured[0]
    for option in (
        "--backend",
        "claude",
        "--model=test-model",
        "--mode",
        "deep",
        "--max-workers",
        "3",
        "--token-budget=4096",
        "--max-concurrency",
        "2",
        "--api-timeout=30",
        "--resolution",
        "1.5",
        "--exclude-hubs=0.2",
        "--exclude",
        "vendor",
        "--google-workspace",
        "--dedup-llm",
        "--cargo",
        "--timing",
    ):
        assert option in forwarded


def test_update_rejects_remap_only_options_without_remap(monkeypatch, tmp_path, capsys):
    corpus = _code_only_corpus(tmp_path / "corpus")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--no-cluster"],
    )
    with pytest.raises(SystemExit) as seed_exit:
        mainmod.main()
    assert seed_exit.value.code == 0

    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "update", str(corpus), "--backend", "claude"],
    )
    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 2
    assert "requires --remap" in capsys.readouterr().err


def test_update_repair_state_restores_validated_marker_without_reextracting(
    monkeypatch, tmp_path, capsys
):
    corpus = _code_only_corpus(tmp_path / "corpus")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--no-cluster", "--multigraph"],
    )
    with pytest.raises(SystemExit) as seed_exit:
        mainmod.main()
    assert seed_exit.value.code == 0

    output = corpus / "graphify-out"
    marker = output / ".graphify_root"
    marker.unlink()
    graph_before = (output / "graph.json").read_bytes()
    manifest_before = (output / "manifest.json").read_bytes()
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "update", str(corpus), "--repair-state"],
    )

    mainmod.main()

    assert marker.exists()
    assert (output / "graph.json").read_bytes() == graph_before
    assert (output / "manifest.json").read_bytes() == manifest_before
    assert "repaired scan-root marker" in capsys.readouterr().out


def test_update_repair_state_rejects_force(monkeypatch, tmp_path, capsys):
    corpus = _code_only_corpus(tmp_path / "corpus")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "update", str(corpus), "--repair-state", "--force"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 2
    assert "repair-only" in capsys.readouterr().err


def test_extract_rejects_unknown_options(monkeypatch, tmp_path, capsys):
    corpus = _code_only_corpus(tmp_path / "corpus")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--definitely-not-real"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 2
    assert "unknown extract option" in capsys.readouterr().err
    assert not (corpus / "graphify-out").exists()


def test_check_update_cli_routes_external_output(monkeypatch, tmp_path, capsys):
    corpus = _code_only_corpus(tmp_path / "corpus")
    out_root = tmp_path / "canonical"
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--out",
            str(out_root),
            "--no-cluster",
        ],
    )
    with pytest.raises(SystemExit) as seed_exit:
        mainmod.main()
    assert seed_exit.value.code == 0

    output = out_root / "graphify-out"
    (output / "needs_update").write_text("pending", encoding="utf-8")
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "check-update", str(corpus), "--out", str(out_root)],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()

    assert exc_info.value.code == 0
    message = capsys.readouterr().out
    assert "Pending non-code changes" in message
    assert f"--out {out_root.resolve()}" in message


def test_extract_out_keeps_project_root_clean(monkeypatch, tmp_path):
    """`extract --out DIR` routes every artifact to DIR/graphify-out/ and the
    scanned project must not grow a graphify-out/ (or anything else) beside
    its sources.

    Guards the centralized-output workflow: run from the project root with
    --out pointing outside the repo, and the repo stays byte-identical.
    """
    project = tmp_path / "project"
    project.mkdir()
    corpus = _code_only_corpus(project)
    external = tmp_path / "external-graphs"

    _clear_backend_keys(monkeypatch)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.chdir(corpus)  # run from the project root, like a real user
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", ".", "--out", str(external)],
    )

    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"unexpected exit code {exc.code}"

    out = external / "graphify-out"
    assert (out / "graph.json").exists(), "graph.json must land under --out"
    assert (out / "manifest.json").exists(), "manifest.json must land under --out"
    assert not (corpus / "graphify-out").exists(), (
        "scanned project must not grow a graphify-out/ when --out is set"
    )
    assert sorted(p.name for p in corpus.iterdir()) == ["auth.py"], (
        "no stray files may appear in the project root"
    )


def _cross_file_python_corpus(root: Path) -> Path:
    root.mkdir()
    (root / "a.py").write_text(
        "from b import beta\n\ndef alpha():\n    return beta()\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text("def beta():\n    return 1\n", encoding="utf-8")
    return root


def _topology_signature(graph_path: Path) -> tuple[list[str], list[tuple[str, str, str]]]:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = sorted(str(node["id"]) for node in data.get("nodes", []))
    edges = sorted(
        (
            str(edge["source"]),
            str(edge["target"]),
            str(edge.get("relation", "")),
        )
        for edge in data.get("links", data.get("edges", []))
    )
    return nodes, edges


def test_extract_out_preserves_ids_and_cross_file_edges(monkeypatch, tmp_path):
    """Changing only output storage must not change extracted graph content."""
    plain = _cross_file_python_corpus(tmp_path / "plain")
    external_source = _cross_file_python_corpus(tmp_path / "external-source")
    external_root = tmp_path / "external-output"

    _clear_backend_keys(monkeypatch)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)

    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(plain), "--multigraph", "--no-cluster"],
    )
    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)

    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(external_source),
            "--out",
            str(external_root),
            "--multigraph",
            "--no-cluster",
        ],
    )
    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)

    plain_signature = _topology_signature(plain / "graphify-out" / "graph.json")
    external_signature = _topology_signature(external_root / "graphify-out" / "graph.json")
    assert plain_signature == external_signature
    assert plain_signature[0] == ["a", "a_alpha", "b", "b_beta"]
    assert ("a", "b", "imports_from") in plain_signature[1]
    assert ("a", "b_beta", "imports") in plain_signature[1]
    assert not any(str(tmp_path) in node_id for node_id in external_signature[0])


def test_extract_persists_portable_scan_root_marker(monkeypatch, tmp_path):
    """Plain and external-output extraction must record their source root."""
    plain = _code_only_corpus(tmp_path / "plain")
    external_source = _code_only_corpus(tmp_path / "external-source")
    external_root = tmp_path / "external-output"

    _clear_backend_keys(monkeypatch)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)

    for argv in (
        ["graphify", "extract", str(plain), "--no-cluster"],
        [
            "graphify",
            "extract",
            str(external_source),
            "--out",
            str(external_root),
            "--no-cluster",
        ],
    ):
        monkeypatch.setattr(mainmod.sys, "argv", argv)
        try:
            mainmod.main()
        except SystemExit as exc:
            assert exc.code in (None, 0)

    plain_marker = plain / "graphify-out" / ".graphify_root"
    external_marker = external_root / "graphify-out" / ".graphify_root"
    assert plain_marker.read_text(encoding="utf-8") == "marker-relative:.."
    assert external_marker.read_text(encoding="utf-8") == (
        "marker-relative:"
        + os.path.relpath(external_source.resolve(), external_marker.parent.resolve()).replace(
            os.sep, "/"
        )
    )


def test_extract_out_respects_process_bound_graphify_out(tmp_path):
    corpus = _cross_file_python_corpus(tmp_path / "corpus")
    external_root = tmp_path / "external-output"
    env = _clean_env()
    env["GRAPHIFY_OUT"] = "custom-graph"

    result = _run(
        [
            "extract",
            str(corpus),
            "--out",
            str(external_root),
            "--multigraph",
            "--no-cluster",
        ],
        tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    custom_out = external_root / "custom-graph"
    nodes, edges = _topology_signature(custom_out / "graph.json")
    assert nodes == ["a", "a_alpha", "b", "b_beta"]
    assert ("a", "b", "imports_from") in edges
    assert ("a", "b_beta", "imports") in edges
    marker = custom_out / ".graphify_root"
    assert marker.read_text(encoding="utf-8") == (
        "marker-relative:"
        + os.path.relpath(corpus.resolve(), marker.parent.resolve()).replace(os.sep, "/")
    )
    assert (custom_out / "cache" / "stat-index.json").exists()
    assert not (external_root / "graphify-out").exists()
    assert not (corpus / "graphify-out").exists()


def test_extract_out_semantic_cache_survives_prune_and_reloads(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text("# Notes\n\nStable body.\n", encoding="utf-8")
    external_root = tmp_path / "external-output"
    calls = 0

    def _semantic_result(paths, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["on_chunk_done"](0, 1, {})
        return {
            "nodes": [
                {
                    "id": "notes",
                    "label": "Notes",
                    "type": "document",
                    "source_file": "notes.md",
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 1,
            "output_tokens": 1,
        }

    _clear_backend_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _semantic_result)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    argv = [
        "graphify",
        "extract",
        str(corpus),
        "--backend",
        "claude",
        "--out",
        str(external_root),
        "--no-cluster",
    ]

    for attempt in range(2):
        monkeypatch.setattr(mainmod.sys, "argv", argv)
        try:
            mainmod.main()
        except SystemExit as exc:
            assert exc.code in (None, 0)
        if attempt == 0:
            (external_root / "graphify-out" / "manifest.json").unlink()

    assert calls == 1
    entries = list((external_root / "graphify-out" / "cache" / "semantic").glob("*.json"))
    assert len(entries) == 1
    cached = json.loads(entries[0].read_text(encoding="utf-8"))
    assert cached["nodes"][0]["source_file"] == "notes.md"
    assert not (corpus / "graphify-out").exists()


def test_extract_without_key_still_errors_when_docs_present(monkeypatch, tmp_path, capsys):
    """Key requirement still fires when semantic work is needed.

    A corpus with a Markdown doc needs LLM semantic extraction, so a keyless
    extract must exit 1 with clear guidance (#1122).
    """
    corpus = _make_corpus(tmp_path)  # includes a Markdown doc
    out_dir = tmp_path / "out"
    _clear_backend_keys(monkeypatch)
    # Patch detect_backend too so ambient AWS/ollama env can't slip through.
    monkeypatch.setattr("graphify.llm.detect_backend", lambda: None)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--out", str(out_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "no LLM API key found" in err
    assert "code-only corpus needs no key" in err
    assert not (out_dir / "graphify-out" / "graph.json").exists()


def test_extract_warm_semantic_cache_does_not_require_backend_credentials(monkeypatch, tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    corpus = _make_corpus(corpus_root)
    out_dir = tmp_path / "out"
    calls = 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)

    def semantic_extract(paths, **kwargs):
        nonlocal calls
        calls += 1
        on_chunk = kwargs.get("on_chunk_done")
        result = {
            "nodes": [
                {
                    "id": "notes",
                    "label": "Notes",
                    "file_type": "document",
                    "source_file": str(paths[0]),
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 1,
            "output_tokens": 1,
            "failed_chunks": 0,
            "incomplete_chunks": 0,
        }
        if on_chunk:
            on_chunk(0, 1, result)
        return result

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", semantic_extract)
    first_argv = [
        "graphify",
        "extract",
        str(corpus),
        "--backend",
        "claude",
        "--out",
        str(out_dir),
        "--no-cluster",
    ]
    monkeypatch.setattr(mainmod.sys, "argv", first_argv)
    with pytest.raises(SystemExit) as first_exit:
        mainmod.main()
    assert first_exit.value.code == 0
    assert calls == 1

    output = out_dir / "graphify-out"
    (output / "manifest.json").unlink()
    _clear_backend_keys(monkeypatch)
    monkeypatch.setattr("graphify.llm.detect_backend", lambda: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--out", str(out_dir), "--no-cluster"],
    )

    with pytest.raises(SystemExit) as second_exit:
        mainmod.main()

    assert second_exit.value.code == 0
    assert calls == 1, "a warm semantic cache must not call or require an LLM backend"


@pytest.mark.parametrize("new_event_during_extract", [False, True])
def test_extract_acknowledges_only_the_semantic_signal_generation_it_started_with(
    monkeypatch,
    tmp_path,
    new_event_during_extract,
):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    corpus = _make_corpus(corpus_root)
    out_root = tmp_path / "out"
    output = out_root / "graphify-out"
    output.mkdir(parents=True)
    signal = output / "needs_update"
    signal.write_text("pending-before-extract", encoding="utf-8")
    original = signal.read_bytes()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)

    def semantic_extract(paths, **kwargs):
        if new_event_during_extract:
            from graphify.watch import _notify_only

            _notify_only(corpus, output_dir=output)
        result = {
            "nodes": [
                {
                    "id": "notes",
                    "label": "Notes",
                    "file_type": "document",
                    "source_file": str(paths[0]),
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 1,
            "output_tokens": 1,
            "failed_chunks": 0,
            "incomplete_chunks": 0,
        }
        on_chunk = kwargs.get("on_chunk_done")
        if on_chunk:
            on_chunk(0, 1, result)
        return result

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", semantic_extract)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--backend",
            "claude",
            "--out",
            str(out_root),
            "--no-cluster",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 0

    if new_event_during_extract:
        assert signal.exists()
        assert signal.read_bytes() != original
    else:
        assert not signal.exists()


def test_extract_timing_flag_emits_stage_timings(monkeypatch, tmp_path, capsys):
    """--timing prints per-stage `[graphify timing]` lines to stderr (#1490); omitting
    it prints none, so default output is unchanged. Code-only corpus => no API key."""
    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text(
        "def a():\n    return b()\ndef b():\n    return 1\n", encoding="utf-8"
    )

    # with --timing
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(code),
            "--no-cluster",
            "--out",
            str(tmp_path / "o1"),
            "--timing",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mainmod.main()
    assert exc.value.code == 0
    err = capsys.readouterr().err
    assert "[graphify timing] detect:" in err
    assert "[graphify timing] total:" in err

    # without --timing => no timing lines
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(code), "--no-cluster", "--out", str(tmp_path / "o2")],
    )
    with pytest.raises(SystemExit) as exc2:
        mainmod.main()
    assert exc2.value.code == 0
    assert "graphify timing" not in capsys.readouterr().err


# ── Multigraph CLI helpers (forked additions) ───────────────────────────────


def _clean_env() -> dict:
    """Return os.environ with every backend API key stripped out.

    Mirrors tests/test_incremental._clean_env so subprocess runs do not pick up
    a real key from the developer's shell and accidentally hit a live LLM.
    """
    env = dict(os.environ)
    for backend in BACKENDS:
        for env_key in _backend_env_keys(backend):
            env.pop(env_key, None)
    for extra in (
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
    ):
        env.pop(extra, None)
    return env


def _run(args: list[str], cwd: Path, *, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `python -m graphify <args>` as a sanitized subprocess."""
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env if env is not None else _clean_env(),
        encoding="utf-8",
    )


def _make_code_corpus(tmp_path: Path) -> Path:
    """A tiny AST-only code corpus — no docs, so semantic/LLM extraction never runs.

    The functions reference each other so AST extraction produces real edges.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def helper():\n    return 1\n\n\n"
        "def main():\n    return helper()\n\n\n"
        "def extra():\n    return main()\n",
        encoding="utf-8",
    )
    return corpus


def _write_multidigraph_graph_json(corpus: Path) -> Path:
    """Seed corpus/graphify-out/graph.json as a multidigraph with parallel edges.

    Built exactly the way the pipeline persists it (build_from_json multigraph=True
    -> export.to_json), so the file carries the top-level ``multigraph: true`` flag
    and ``graphify_profile.graph_type == multidigraph``. Two parallel main->helper
    edges (different relations) prove parallels survive a sticky re-extract.
    """
    nodes = [
        {
            "id": n,
            "label": f"{n}()",
            "file_type": "code",
            "source_file": "app.py",
            "source_location": "L1",
        }
        for n in ("main", "helper")
    ]
    edges = [
        {
            "source": "main",
            "target": "helper",
            "relation": rel,
            "confidence": "EXTRACTED",
            "source_file": "app.py",
            "source_location": f"L{i}",
        }
        for i, rel in enumerate(["calls", "imports"])
    ]
    G = build_from_json({"nodes": nodes, "edges": edges}, multigraph=True)
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_edges("main", "helper") == 2
    out = corpus / "graphify-out"
    out.mkdir(exist_ok=True)
    graph_json = out / "graph.json"
    to_json(G, {0: ["main", "helper"]}, str(graph_json), force=True)
    # Persist the scan root so a later `update` (no path arg) can recover it.
    (out / ".graphify_root").write_text(str(corpus), encoding="utf-8")
    return graph_json


def _graph_type(graph_data: dict) -> str | None:
    return graph_data.get("graph", {}).get("graphify_profile", {}).get("graph_type")


def _parallel_edges(graph_data: dict, src: str, tgt: str) -> list[dict]:
    links = graph_data.get("links", graph_data.get("edges", []))
    return [e for e in links if e.get("source") == src and e.get("target") == tgt]


# ── Multigraph CLI tests (forked additions) ─────────────────────────────────


def test_extract_simple_default(tmp_path):
    """No flag on a fresh corpus → a simple graph (historical behavior).

    A fresh corpus has no existing graph.json to inherit, so the sticky default
    collapses to the historical simple build: multigraph:false / graph_type simple.
    """
    corpus = _make_code_corpus(tmp_path)
    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"  # code-only corpus never calls the LLM
    r = _run(["extract", str(corpus), "--backend", "claude"], tmp_path, env=env)
    assert r.returncode == 0, f"fresh simple extract should succeed: {r.stderr}"

    graph_json = corpus / "graphify-out" / "graph.json"
    assert graph_json.exists(), f"graph.json must be written: {r.stderr}"
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data.get("multigraph") is False, "default fresh build must be a simple graph"
    assert _graph_type(data) == "simple"


def test_extract_multigraph_flag(tmp_path):
    """`extract --multigraph` → graph.json is a keyed MultiDiGraph.

    Real end-to-end CLI subprocess: multigraph:true + graphify_profile.graph_type
    == "multidigraph", and it reloads as an actual nx.MultiDiGraph.
    """
    corpus = _make_code_corpus(tmp_path)
    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"
    r = _run(["extract", str(corpus), "--backend", "claude", "--multigraph"], tmp_path, env=env)
    assert r.returncode == 0, f"extract --multigraph should succeed: {r.stderr}"

    graph_json = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data.get("multigraph") is True, "--multigraph must produce a multigraph graph.json"
    assert data.get("directed") is True, "a MultiDiGraph is always directed"
    assert _graph_type(data) == "multidigraph"
    # Reloads as a real MultiDiGraph.
    G = load_graph(data)
    assert G.is_multigraph(), "graph.json must reload as a MultiDiGraph"


def test_extract_multigraph_then_update_sticky(tmp_path):
    """`extract --multigraph`, then default re-extract/update STAYS multigraph.

    The second build is run WITHOUT any flag 3 times in a row; the profile must
    stay multidigraph each time (idempotence-under-repeat), with the keyed
    parallel-edge capability intact — never a silent collapse to simple.
    """
    corpus = _make_code_corpus(tmp_path)
    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"

    r0 = _run(["extract", str(corpus), "--backend", "claude", "--multigraph"], tmp_path, env=env)
    assert r0.returncode == 0, f"initial --multigraph extract failed: {r0.stderr}"
    graph_json = corpus / "graphify-out" / "graph.json"

    # Seed two parallel main->helper edges so we can prove parallels persist.
    _write_multidigraph_graph_json(corpus)
    seeded = json.loads(graph_json.read_text(encoding="utf-8"))
    assert seeded.get("multigraph") is True
    assert len(_parallel_edges(seeded, "main", "helper")) == 2

    # Default re-extract (NO flag) 3×; sticky must keep it multigraph every time.
    for attempt in range(1, 4):
        r = _run(["extract", str(corpus), "--backend", "claude"], tmp_path, env=env)
        assert r.returncode == 0, f"sticky re-extract #{attempt} failed: {r.stderr}"
        data = json.loads(graph_json.read_text(encoding="utf-8"))
        assert data.get("multigraph") is True, (
            f"re-extract #{attempt} must STAY multigraph (sticky), "
            f"got multigraph={data.get('multigraph')!r}"
        )
        assert _graph_type(data) == "multidigraph", f"re-extract #{attempt} profile drifted"
        # Parallel edges are not collapsed away by the sticky rebuild.
        par = _parallel_edges(data, "main", "helper")
        assert len(par) == 2, f"re-extract #{attempt} must preserve keyed parallel edges"
        assert sorted(e["relation"] for e in par) == ["calls", "imports"]
        # Reloads as a MultiDiGraph with the parallels intact.
        G = load_graph(data)
        assert G.is_multigraph()
        assert G.number_of_edges("main", "helper") == 2

    # A default `update` (the watch entrypoint) also stays multigraph.
    ru = _run(["update", str(corpus)], tmp_path, env=env)
    assert ru.returncode == 0, f"sticky update failed: {ru.stderr}"
    after_update = json.loads(graph_json.read_text(encoding="utf-8"))
    assert after_update.get("multigraph") is True, "update must inherit the multigraph profile"
    assert _graph_type(after_update) == "multidigraph"


def test_extract_multigraph_no_cluster_sticky_idempotent(tmp_path):
    """`--no-cluster` still preserves a sticky multigraph across no-op re-runs.

    A no-cluster incremental scan with no changed files produces an empty fresh
    extraction. The command must merge that empty delta with the saved graph,
    not overwrite graph.json with zero nodes/edges.
    """
    corpus = _make_code_corpus(tmp_path)
    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"

    r0 = _run(
        ["extract", str(corpus), "--backend", "claude", "--multigraph", "--no-cluster"],
        tmp_path,
        env=env,
    )
    assert r0.returncode == 0, f"initial no-cluster --multigraph failed: {r0.stderr}"

    graph_json = corpus / "graphify-out" / "graph.json"
    first = json.loads(graph_json.read_text(encoding="utf-8"))
    first_nodes = len(first.get("nodes", []))
    first_edges = len(first.get("links", first.get("edges", [])))
    assert first.get("multigraph") is True
    assert _graph_type(first) == "multidigraph"
    assert first_nodes > 0
    assert first_edges > 0

    for attempt in range(1, 4):
        r = _run(
            ["extract", str(corpus), "--backend", "claude", "--no-cluster"],
            tmp_path,
            env=env,
        )
        assert r.returncode == 0, f"sticky no-cluster re-extract #{attempt} failed: {r.stderr}"
        data = json.loads(graph_json.read_text(encoding="utf-8"))
        assert data.get("multigraph") is True
        assert _graph_type(data) == "multidigraph"
        assert len(data.get("nodes", [])) == first_nodes
        assert len(data.get("links", data.get("edges", []))) == first_edges


def test_extract_explicit_simple_downgrade_warns(tmp_path):
    """Existing multigraph graph.json + `extract --simple` → builds simple AND warns.

    The downgrade collapses parallel edges, so it requires explicit intent and a
    loud lossy-collapse WARNING — never a silent collapse. A manifest is seeded so
    the run takes the incremental (preserve+merge) path, where the existing
    multigraph's parallel edges are loaded and then collapsed under the simple
    target — the real lossy projection we want to prove.
    """
    from graphify.detect import save_manifest

    corpus = _make_code_corpus(tmp_path)
    graph_json = _write_multidigraph_graph_json(corpus)
    out = corpus / "graphify-out"
    save_manifest(
        {"code": [str(corpus / "app.py")]},
        manifest_path=str(out / "manifest.json"),
        kind="both",
    )
    before = json.loads(graph_json.read_text(encoding="utf-8"))
    assert before.get("multigraph") is True
    assert len(_parallel_edges(before, "main", "helper")) == 2

    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"
    r = _run(["extract", str(corpus), "--backend", "claude", "--simple"], tmp_path, env=env)
    assert r.returncode == 0, f"--simple downgrade should succeed: {r.stderr}"
    # Lossy-collapse WARNING must be printed (explicit, audible downgrade).
    assert "WARNING" in r.stderr and "--simple" in r.stderr, (
        f"explicit --simple downgrade must warn about lossy collapse, got: {r.stderr}"
    )
    assert "collaps" in r.stderr.lower()

    after = json.loads(graph_json.read_text(encoding="utf-8"))
    assert after.get("multigraph") is False, "--simple must produce a non-multigraph graph"
    assert _graph_type(after) != "multidigraph"
    # The two parallel edges from the seeded multigraph collapse onto a single
    # main->helper edge (the lossy projection — one survivor, not two parallels).
    assert len(_parallel_edges(after, "main", "helper")) == 1, (
        "explicit --simple must collapse the existing parallel edges onto one"
    )


def test_extract_explicit_simple_noop_still_downgrades(tmp_path):
    """A no-change incremental scan must still apply an explicit class change."""
    corpus = _make_code_corpus(tmp_path)
    env = _clean_env()
    env["ANTHROPIC_API_KEY"] = "sk-test-fake-key"
    graph_json = corpus / "graphify-out" / "graph.json"

    seed = _run(
        ["extract", str(corpus), "--backend", "claude", "--multigraph", "--no-cluster"],
        tmp_path,
        env=env,
    )
    assert seed.returncode == 0, seed.stderr
    assert json.loads(graph_json.read_text(encoding="utf-8"))["multigraph"] is True

    downgrade = _run(
        ["extract", str(corpus), "--backend", "claude", "--simple", "--no-cluster"],
        tmp_path,
        env=env,
    )

    assert downgrade.returncode == 0, downgrade.stderr
    assert "WARNING" in downgrade.stderr and "--simple" in downgrade.stderr
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data["multigraph"] is False
    assert _graph_type(data) != "multidigraph"


def test_extract_multigraph_capability_failure_message(monkeypatch, tmp_path, capsys):
    """A MultiDiGraph capability failure surfaces as a clean CLI error, exit 1.

    The probe RuntimeError must be caught and printed (no traceback), and no
    graph.json may be written. Run in-process so we can monkeypatch the probe.
    """
    corpus = _make_code_corpus(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def _boom():
        raise RuntimeError(
            "error: --multigraph requires NetworkX keyed MultiDiGraph node-link "
            "round-trip support. Simulated capability failure."
        )

    # Patch where the extract handler imports it from.
    monkeypatch.setattr("graphify.multigraph_compat.require_multigraph_capabilities", _boom)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--multigraph"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 1, f"capability failure must exit 1, got {exc_info.value.code}"

    err = capsys.readouterr().err
    assert "--multigraph requires" in err, f"clean capability message expected, got: {err}"
    assert "Traceback" not in err, "capability failure must not leak a traceback"
    assert not (corpus / "graphify-out" / "graph.json").exists(), (
        "no graph.json may be written when the capability gate fails"
    )


def test_extract_multigraph_query_roundtrip(tmp_path, capsys, monkeypatch):
    """End-to-end public workflow: a multigraph corpus with same-endpoint different
    relations exposes the parallel relationships through the public query/path path.

    Builds the multigraph graph.json the way `extract --multigraph` persists it,
    then runs `graphify path` (a public query surface) and asserts BOTH parallel
    relations show — the parallel relationships are visible, not collapsed.
    """
    corpus = _make_code_corpus(tmp_path)
    graph_json = _write_multidigraph_graph_json(corpus)

    # Sanity: the persisted file is a multidigraph with both parallel relations.
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data.get("multigraph") is True
    G = load_graph(data)
    assert G.is_multigraph() and G.number_of_edges("main", "helper") == 2

    # Public query surface: `graphify path main helper` bundles all relations.
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "path", "main", "helper", "--graph", str(graph_json)],
    )
    mainmod.main()
    out = capsys.readouterr().out
    assert "calls" in out, f"parallel 'calls' relation must appear in path output: {out}"
    assert "imports" in out, f"parallel 'imports' relation must appear in path output: {out}"


def test_extract_no_cluster_refuses_to_zero_populated_graph(monkeypatch, tmp_path, capsys):
    """RISK 4 — Guard 3: the non-incremental no-cluster simple path must NOT wipe a
    populated graph.json with a 0-node extraction.

    The bug: with an existing populated (simple) graph.json but NO manifest.json
    (so the run is non-incremental) the ``--no-cluster`` branch falls to the raw
    ``graph_json_path.write_text(json.dumps(merged, ...))`` ``else`` case. That raw
    write bypasses both existing empty-merge guards (``export.to_json`` /
    ``watch._check_shrink``). When AST extraction aborts (returns 0 nodes) the raw
    write overwrites the saved graph with an EMPTY one — a failed extraction
    silently destroys real data. The clustered sibling already guards this with
    ``if G.number_of_nodes() == 0: ... sys.exit(1)``; the no-cluster simple path
    must do the same. The command must instead exit non-zero, print the byte-
    identical guard message, and leave the populated graph.json untouched.
    """
    corpus = _make_code_corpus(tmp_path)
    out = corpus / "graphify-out"
    out.mkdir(exist_ok=True)
    graph_json = out / "graph.json"

    # Seed a POPULATED *simple* graph.json the way the pipeline persists it
    # (build_from_json default-simple -> to_json). Simple (not multigraph) so the
    # sticky profile resolves to non-multigraph and the run takes the raw-write
    # ``else`` branch — exactly the unguarded site. NO manifest.json is written,
    # so the run is non-incremental (the path the incremental build_merge floor
    # never protects).
    seed_nodes = [
        {
            "id": n,
            "label": f"{n}()",
            "file_type": "code",
            "source_file": "app.py",
            "source_location": "L1",
        }
        for n in ("main", "helper", "extra")
    ]
    seed_edges = [
        {
            "source": "main",
            "target": "helper",
            "relation": "calls",
            "confidence": "EXTRACTED",
            "source_file": "app.py",
            "source_location": "L5",
        }
    ]
    G_seed = build_from_json({"nodes": seed_nodes, "edges": seed_edges})
    assert not G_seed.is_multigraph(), "seed must be a simple graph (non-multigraph)"
    to_json(G_seed, {0: ["main", "helper", "extra"]}, str(graph_json), force=True)
    before = json.loads(graph_json.read_text(encoding="utf-8"))
    seeded_n = len(before.get("nodes", []))
    assert seeded_n == 3, "seed graph.json must start populated with 3 nodes"
    assert before.get("multigraph") is False, "seed graph.json must be simple"
    assert not (out / "manifest.json").exists(), "no manifest → non-incremental run"
    (out / ".graphify_semantic_marker").write_text("{}", encoding="utf-8")
    tree_before = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }

    # Force the AST extraction to abort so the merged extraction yields 0 nodes.
    # This mirrors the real trigger (a parser/extractor blowing up): the extract
    # handler's ``except`` resets ast_result to an empty dict, and a code-only
    # corpus has no semantic pass, so ``merged`` collapses to 0 nodes. The extract
    # handler imports ``extract`` from graphify.extract at call time, so patching
    # the source symbol is picked up.
    def _empty_ast_result(paths, **kwargs):
        return {
            "nodes": [],
            "edges": [],
            "failed_files": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    import graphify.extract as _extract_mod

    monkeypatch.setattr(_extract_mod, "extract", _empty_ast_result)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")  # code-only: LLM never called
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--no-cluster"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 1, (
        f"a 0-node no-cluster extraction over a populated graph must exit 1, "
        f"got {exc_info.value.code}"
    )

    err = capsys.readouterr().err
    # Byte-identical to the Guard 1 / Guard 2 message.
    assert (
        f"[graphify] ERROR: refusing to overwrite a populated graph.json "
        f"({seeded_n} nodes) with an EMPTY (0-node) graph - this is a "
        f"failed/aborted extraction, not a real result. The previous graph "
        f"is preserved." in err
    ), f"guard message must match Guards 1/2 byte-for-byte, got: {err!r}"

    # The populated graph.json must be PRESERVED — not wiped to an empty graph.
    after = json.loads(graph_json.read_text(encoding="utf-8"))
    assert len(after.get("nodes", [])) == seeded_n, (
        "the populated graph.json must NOT be overwritten with a 0-node graph"
    )
    tree_after = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before, "a refused extraction must not mutate any output state"


def test_extract_no_cluster_incremental_zero_merge_exits_nonzero_and_preserves_graph(
    monkeypatch, tmp_path, capsys
):
    """RISK 4 — Guard 1 signaling gap: the INCREMENTAL no-cluster path must SIGNAL
    failure (exit non-zero, no false-success line) when the merge yields 0 nodes.

    The incremental no-cluster branch writes through
    ``to_json(_nc_graph, {}, ..., force=True)`` (Guard 1). When ``build_merge``
    collapses to a 0-node graph over a populated graph.json, Guard 1's empty-merge
    floor correctly *returns False and PRESERVES the data* — but the caller ignored
    that return value: it fell through, printed the success line
    ``[graphify extract] wrote ... graph.json — 0 nodes, 0 edges (no clustering)``
    and exited 0. The data was safe, but a failed/aborted extraction reported a
    misleading false success (wrong exit code + message).

    The fix captures Guard 1's ``False`` return at the no-cluster incremental write
    site and, on refusal only, emits an aborted-extraction stderr note and exits 1
    — never the bogus "wrote ... 0 nodes" success line. A populated graph.json plus
    a manifest.json makes the run incremental; ``build_merge`` is forced to yield an
    empty graph to model the aborted/pruned-to-empty merge. The legitimate sticky
    no-cluster case (``test_extract_multigraph_no_cluster_sticky_idempotent``) keeps
    exit 0 because ``build_merge`` preserves the existing nodes there (True return).
    """
    from graphify.detect import save_manifest

    corpus = _make_code_corpus(tmp_path)
    out = corpus / "graphify-out"
    out.mkdir(exist_ok=True)
    graph_json = out / "graph.json"

    # Seed a POPULATED *simple* graph.json the way the pipeline persists it.
    seed_nodes = [
        {
            "id": n,
            "label": f"{n}()",
            "file_type": "code",
            "source_file": "app.py",
            "source_location": "L1",
        }
        for n in ("main", "helper", "extra")
    ]
    seed_edges = [
        {
            "source": "main",
            "target": "helper",
            "relation": "calls",
            "confidence": "EXTRACTED",
            "source_file": "app.py",
            "source_location": "L5",
        }
    ]
    G_seed = build_from_json({"nodes": seed_nodes, "edges": seed_edges})
    assert not G_seed.is_multigraph(), "seed must be a simple graph (non-multigraph)"
    to_json(G_seed, {0: ["main", "helper", "extra"]}, str(graph_json), force=True)

    # A manifest.json alongside the populated graph.json makes the run INCREMENTAL
    # (incremental_mode = manifest.exists() and graph.json.exists()), so the write
    # routes through the incremental ``to_json(..., force=True)`` site, not the
    # raw-write else-branch the Guard 3 sibling covers.
    save_manifest(
        {"code": [str(corpus / "app.py")]},
        manifest_path=str(out / "manifest.json"),
        kind="both",
    )

    # Modify app.py AFTER the manifest is saved so the incremental scan detects a
    # CHANGED file and proceeds to build_merge. Upstream's "no incremental changes
    # detected" early-exit otherwise short-circuits with exit 0 before the merge,
    # so the empty-merge guard under test would never run.
    (corpus / "app.py").write_text(
        (corpus / "app.py").read_text(encoding="utf-8") + "\n\ndef added():\n    return helper()\n",
        encoding="utf-8",
    )

    before = json.loads(graph_json.read_text(encoding="utf-8"))
    seeded_n = len(before.get("nodes", []))
    assert seeded_n == 3, "seed graph.json must start populated with 3 nodes"
    assert before.get("multigraph") is False, "seed graph.json must be simple"
    assert (out / "manifest.json").exists(), "manifest → incremental run"
    (out / ".graphify_semantic_marker").write_text("{}", encoding="utf-8")
    tree_before = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }

    # Force the incremental merge to yield a 0-node graph (aborted / pruned-to-empty
    # extraction). The no-cluster incremental branch imports build_merge from
    # graphify.build at call time, so patching the source symbol is picked up.
    def _empty_merge(*args, **kwargs):
        return build_from_json({"nodes": [], "edges": []})

    import graphify.build as _build_mod

    monkeypatch.setattr(_build_mod, "build_merge", _empty_merge)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")  # code-only: LLM never called
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--no-cluster"],
    )

    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 1, (
        f"a 0-node incremental no-cluster merge over a populated graph must exit 1, "
        f"got {exc_info.value.code}"
    )

    captured = capsys.readouterr()
    # The misleading false-success line must NOT be printed.
    assert "0 nodes, 0 edges" not in captured.out, (
        f"a 0-node aborted merge must NOT print the 'wrote ... 0 nodes' success "
        f"line, got stdout: {captured.out!r}"
    )

    # The populated graph.json must be PRESERVED — not wiped to an empty graph.
    after = json.loads(graph_json.read_text(encoding="utf-8"))
    assert len(after.get("nodes", [])) == seeded_n, (
        "the populated graph.json must NOT be overwritten with a 0-node graph"
    )
    assert after == before, "graph.json must be byte-for-byte unchanged after the refused write"
    tree_after = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before, "a refused extraction must not mutate any output state"
