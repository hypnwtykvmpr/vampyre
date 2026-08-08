"""Tests for watch.py - file watcher helpers (no watchdog required)."""

import json
import os
import subprocess
import time
from pathlib import Path
import pytest

from graphify.watch import _notify_only, _WATCHED_EXTENSIONS, _rebuild_lock, _check_shrink


# --- _notify_only ---


def test_notify_only_creates_flag(tmp_path):
    _notify_only(tmp_path)
    flag = tmp_path / "graphify-out" / "needs_update"
    assert flag.exists()
    assert flag.read_text(encoding="utf-8").strip()


def test_notify_only_creates_flag_dir(tmp_path):
    # graphify-out dir does not exist yet
    assert not (tmp_path / "graphify-out").exists()
    _notify_only(tmp_path)
    assert (tmp_path / "graphify-out").is_dir()


def test_notify_only_idempotent(tmp_path):
    _notify_only(tmp_path)
    flag = tmp_path / "graphify-out" / "needs_update"
    first = flag.read_bytes()
    _notify_only(tmp_path)
    assert flag.read_bytes() != first


# --- _WATCHED_EXTENSIONS ---


def test_watched_extensions_includes_code():
    assert ".py" in _WATCHED_EXTENSIONS
    assert ".ts" in _WATCHED_EXTENSIONS
    assert ".go" in _WATCHED_EXTENSIONS
    assert ".rs" in _WATCHED_EXTENSIONS


def test_watched_extensions_includes_docs():
    assert ".md" in _WATCHED_EXTENSIONS
    assert ".txt" in _WATCHED_EXTENSIONS
    assert ".pdf" in _WATCHED_EXTENSIONS


def test_watched_extensions_includes_images():
    assert ".png" in _WATCHED_EXTENSIONS
    assert ".jpg" in _WATCHED_EXTENSIONS


def test_watched_extensions_excludes_noise():
    # .json is now indexed (bash/JSON extractors added in #866)
    assert ".json" in _WATCHED_EXTENSIONS
    assert ".sh" in _WATCHED_EXTENSIONS
    assert ".pyc" not in _WATCHED_EXTENSIONS
    assert ".log" not in _WATCHED_EXTENSIONS


# --- watch() import error without watchdog ---


def test_check_update_no_flag_returns_true(tmp_path):
    """check_update returns True and is silent when needs_update flag is absent."""
    from graphify.watch import check_update

    assert check_update(tmp_path) is True


@pytest.mark.parametrize(
    "foreign",
    [r"C:\repo\src\app.py", r"\\server\share\app.py", "/repo/src/app.py"],
)
def test_missing_local_source_does_not_guess_across_host_path_conventions(tmp_path, foreign):
    from graphify.watch import _missing_local_source

    expected = Path(foreign).is_absolute()
    assert _missing_local_source(foreign, tmp_path) is expected


def test_check_update_with_flag_returns_true_and_prints(tmp_path, capsys):
    """check_update returns True and prints notification when flag exists."""
    from graphify.watch import check_update

    flag = tmp_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1", encoding="utf-8")
    result = check_update(tmp_path)
    assert result is True
    out = capsys.readouterr().out
    assert "graphify update" in out
    assert "--remap" in out


def test_check_update_external_output_prints_routed_remap_command(tmp_path, capsys):
    from graphify.watch import check_update

    source = tmp_path / "Sources"
    source.mkdir()
    output_root = tmp_path / "canonical"
    output = output_root / "graphify-out"
    output.mkdir(parents=True)
    (output / "needs_update").write_text("pending", encoding="utf-8")

    assert check_update(source, output_dir=output) is True

    message = capsys.readouterr().out
    assert str(source) in message
    assert f"--out {output_root}" in message
    assert "--remap" in message


def test_check_update_does_not_clear_flag(tmp_path):
    """check_update never removes the needs_update flag (clearing is LLM's job)."""
    from graphify.watch import check_update

    flag = tmp_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1", encoding="utf-8")
    check_update(tmp_path)
    assert flag.exists()


def test_watch_raises_without_watchdog(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "watchdog.observers" or name == "watchdog.events":
            raise ImportError("mocked missing watchdog")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from graphify.watch import watch

    with pytest.raises(ImportError, match="watchdog not installed"):
        watch(tmp_path)


# --- _rebuild_lock (GH-858) ---


def test_rebuild_lock_writes_pid_with_newline(tmp_path):
    out = tmp_path / "graphify-out"
    lock_path = out / ".rebuild.lock"
    with _rebuild_lock(out) as got:
        assert got is True
        assert lock_path.exists()
        contents = lock_path.read_text(encoding="utf-8")
        assert contents == f"{os.getpid()}\n", contents


def test_rebuild_lock_removed_after_release(tmp_path):
    """GH-858: lock file must be unlinked once the rebuild completes so
    downstream waiters that poll for its absence unblock promptly."""
    out = tmp_path / "graphify-out"
    lock_path = out / ".rebuild.lock"
    with _rebuild_lock(out) as got:
        assert got is True
    assert not lock_path.exists(), "lock file should be unlinked after release"


def test_rebuild_lock_does_not_accumulate_pids_across_runs(tmp_path):
    """GH-858: each acquisition truncates and rewrites the PID line rather
    than appending, so the file never grows into a digit-concatenation."""
    out = tmp_path / "graphify-out"
    lock_path = out / ".rebuild.lock"
    expected = f"{os.getpid()}\n"
    for _ in range(5):
        with _rebuild_lock(out) as got:
            assert got is True
            assert lock_path.read_text(encoding="utf-8") == expected
        assert not lock_path.exists()


def test_graphify_root_preserves_relative_when_invoked_with_relative_path(tmp_path, monkeypatch):
    """#777: marker paths are portable and anchored to the marker directory."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lib.py").write_text("def f(): pass\n", encoding="utf-8")

    monkeypatch.chdir(corpus)
    assert _rebuild_code(Path("."), acquire_lock=False) is True

    saved = (corpus / "graphify-out" / ".graphify_root").read_text(encoding="utf-8")
    assert saved == "marker-relative:.."


def test_graphify_root_preserves_absolute_when_user_supplied(tmp_path):
    """Absolute caller paths are persisted in the same portable marker form."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lib.py").write_text("def f(): pass\n", encoding="utf-8")
    assert _rebuild_code(corpus, acquire_lock=False) is True

    saved = (corpus / "graphify-out" / ".graphify_root").read_text(encoding="utf-8")
    assert saved == "marker-relative:.."


def test_clustered_rebuild_repairs_and_stabilizes_analysis_sidecar(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def helper():\n    return 1\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    output = corpus / "graphify-out"
    output.mkdir()
    analysis_path = output / ".graphify_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "communities": {"999": ["ghost"]},
                "cohesion": {"999": 1.0},
                "gods": [],
                "surprises": [],
            }
        ),
        encoding="utf-8",
    )

    assert _rebuild_code(corpus, no_viz=True, acquire_lock=False)

    def _assert_current() -> bytes:
        from graphify.graph_loader import load_graph_state_file
        from graphify.graph_state import DecodeMode, graph_analysis_fingerprint

        graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        expected: dict[str, set[str]] = {}
        for node in graph["nodes"]:
            expected.setdefault(str(node["community"]), set()).add(str(node["id"]))
        actual = {
            key: {str(node_id) for node_id in members}
            for key, members in analysis["communities"].items()
        }
        assert actual == expected
        assert set(analysis["cohesion"]) == set(expected)
        assert "built_at_commit" in analysis
        assert analysis["built_at_commit"] == graph.get("built_at_commit")
        assert analysis["graph_fingerprint"] == graph_analysis_fingerprint(
            load_graph_state_file(
                output / "graph.json",
                mode=DecodeMode.STRICT_CURRENT,
            )
        )
        return analysis_path.read_bytes()

    first = _assert_current()
    valid_analysis = json.loads(first)
    analysis_path.write_text(
        json.dumps(
            {
                "built_at_commit": valid_analysis["built_at_commit"],
                "graph_fingerprint": valid_analysis["graph_fingerprint"],
                "communities": {"999": ["ghost"]},
                "cohesion": {"999": 1.0},
                "gods": [],
                "surprises": [],
            }
        ),
        encoding="utf-8",
    )
    assert _rebuild_code(corpus, no_viz=True, acquire_lock=False)
    second = _assert_current()
    assert second == first

    invalid_fingerprint = json.loads(second)
    invalid_fingerprint["graph_fingerprint"] = "0" * 64
    analysis_path.write_text(json.dumps(invalid_fingerprint), encoding="utf-8")
    assert _rebuild_code(corpus, no_viz=True, acquire_lock=False)
    assert _assert_current() == second

    assert _rebuild_code(corpus, no_viz=True, acquire_lock=False)
    assert _assert_current() == second


def test_no_cluster_rebuild_invalidates_cluster_derived_outputs(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "app.py"
    source.write_text("def first():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(corpus, acquire_lock=False)

    output = corpus / "graphify-out"
    derived = [
        output / ".graphify_analysis.json",
        output / "GRAPH_REPORT.md",
        output / "graph.html",
    ]
    assert all(path.exists() for path in derived)
    callflow = output / "app-callflow.html"
    callflow.write_text("stale", encoding="utf-8")
    before = {path: path.read_bytes() for path in [*derived, callflow]}
    graph_path = output / "graph.json"
    legacy_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    legacy_graph["input_tokens"] = 7
    legacy_graph["output_tokens"] = 3
    graph_path.write_text(json.dumps(legacy_graph), encoding="utf-8")

    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False)
    assert {path: path.read_bytes() for path in before} == before
    migrated_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "input_tokens" not in migrated_graph
    assert "output_tokens" not in migrated_graph

    source.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False)
    assert all(not path.exists() for path in derived)
    assert not callflow.exists()
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert all("community" not in node for node in graph["nodes"])
    assert "input_tokens" not in graph
    assert "output_tokens" not in graph
    assert "graphify_identity_diagnostics" in graph


def test_rebuild_code_evicts_nodes_from_deleted_files(tmp_path):
    """#1007: graphify update (_rebuild_code with no changed_paths) must remove
    nodes and edges from files deleted since the last run."""
    import json
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "auth.py").write_text("def login(): pass\ndef logout(): pass\n", encoding="utf-8")
    (corpus / "utils.py").write_text("def format_date(): pass\n", encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False) is True
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    node_labels_before = {n["label"] for n in data.get("nodes", [])}
    assert "format_date()" in node_labels_before

    (corpus / "utils.py").unlink()

    assert _rebuild_code(corpus, acquire_lock=False) is True
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    node_labels_after = {n["label"] for n in data.get("nodes", [])}
    assert "format_date()" not in node_labels_after, (
        "stale function node from deleted file must be evicted"
    )
    assert "login()" in node_labels_after, "nodes from surviving file must be kept"


def test_rebuild_code_evicts_removed_symbol_from_surviving_file(tmp_path):
    """#1116: graphify update (_rebuild_code with no changed_paths) must prune a
    symbol removed from a file that still exists — and its inbound call edge —
    without dropping genuine semantic nodes that share the surviving file."""
    import json
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "a.py").write_text("def foo(): pass\ndef bar(): pass\n", encoding="utf-8")
    (corpus / "b.py").write_text(
        "from a import foo\n\ndef caller():\n    foo()\n", encoding="utf-8"
    )

    assert _rebuild_code(corpus, acquire_lock=False) is True
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))

    def labels(d):
        return {n["label"] for n in d.get("nodes", [])}

    def id_for(d, label):
        return next(n["id"] for n in d.get("nodes", []) if n["label"] == label)

    def edges(d):
        return d.get("links", d.get("edges", []))

    before = labels(data)
    assert {"foo()", "bar()", "caller()"} <= before
    foo_id = id_for(data, "foo()")
    caller_id = id_for(data, "caller()")
    assert any({e.get("source"), e.get("target")} == {caller_id, foo_id} for e in edges(data)), (
        "cross-file caller->foo call edge must exist before removal"
    )

    # Pre-seed a semantic node on the surviving a.py (no AST id, no _origin
    # marker). A naive "evict every re-extracted file's nodes by source_file"
    # fix would wrongly delete this; the identity-based fix must keep it.
    data["nodes"].append(
        {
            "id": "a_authconcept",
            "label": "AuthConcept",
            "file_type": "concept",
            "source_file": "a.py",
        }
    )
    graph_path.write_text(json.dumps(data), encoding="utf-8")

    # Remove foo() from a.py (keep bar); leave b.py untouched.
    (corpus / "a.py").write_text("def bar(): pass\n", encoding="utf-8")

    # No force=True: a symbol removed from a re-extracted file is a legitimate
    # shrink, so the shrink-guard must let `graphify update` refresh the graph
    # without --force (the lost node belongs to a rebuilt source).
    assert _rebuild_code(corpus, acquire_lock=False) is True
    after_data = json.loads(graph_path.read_text(encoding="utf-8"))
    after = labels(after_data)

    assert "foo()" not in after, "removed symbol must be pruned from surviving file"
    assert not any(
        e.get("source") == foo_id or e.get("target") == foo_id for e in edges(after_data)
    ), "dangling edge to the removed symbol must be dropped"
    assert "bar()" in after, "surviving symbol in the same file must be kept"
    assert "caller()" in after, "unchanged file's nodes must be kept"
    assert "AuthConcept" in after, "semantic node on a surviving file must not be evicted"


def test_rebuild_code_preupgrade_marker_less_node_one_cycle_lag(tmp_path):
    """#1118 backward-compat: a graph.json built before #1116 has no `_origin`
    markers. On the first `graphify update` after upgrading, a symbol removed
    from a surviving file is NOT pruned that cycle — its old node carries no
    marker, so the new drop-rule skips it. This is a deliberate one-cycle lag
    (no data loss); it self-heals once the node has been stamped `_origin="ast"`
    (which a full re-extraction does for every surviving symbol)."""
    import json
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("def bar(): pass\n", encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False) is True
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))

    def labels(d):
        return {n["label"] for n in d.get("nodes", [])}

    # Simulate a pre-#1116 graph: strip every `_origin` marker, then inject a
    # stale AST node for a symbol no longer present in a.py's source — also
    # marker-less, exactly as a pre-upgrade graph would carry it.
    for n in data["nodes"]:
        n.pop("_origin", None)
    data["nodes"].append(
        {
            "id": "a_foo",
            "label": "foo()",
            "file_type": "function",
            "source_file": "a.py",
        }
    )
    graph_path.write_text(json.dumps(data), encoding="utf-8")

    # First update after "upgrade" (full rebuild, no changed_paths): the stale
    # node has no marker, so the drop-rule skips it and it survives this cycle.
    assert _rebuild_code(corpus, acquire_lock=False, force=True) is True
    after = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "foo()" in labels(after), (
        "pre-upgrade marker-less stale node must survive the first update — "
        "documented one-cycle backward-compat lag (#1118)"
    )

    # Once stamped (a full re-extraction stamps every surviving symbol), the
    # drop-rule applies on the next update and the stale node self-heals away.
    for n in after["nodes"]:
        if n["label"] == "foo()":
            n["_origin"] = "ast"
    graph_path.write_text(json.dumps(after), encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False, force=True) is True
    healed = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "foo()" not in labels(healed), (
        "once carrying _origin=ast, the stale node is pruned on the next update (self-heal)"
    )
    assert "bar()" in labels(healed), "surviving symbol must be kept throughout"


def test_full_rebuild_prunes_stale_sourceless_ast_node(tmp_path):
    """A complete AST result is authoritative even when an old AST node has no owner."""
    import json
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def live():\n    return 1\n", encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False, no_cluster=True) is True
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    data["nodes"].append(
        {
            "id": "stale_import_stub",
            "label": "StaleImport",
            "file_type": "code",
            "source_file": "",
            "source_location": "",
            "_origin": "ast",
        }
    )
    graph_path.write_text(json.dumps(data), encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False, no_cluster=True, force=True) is True
    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in rebuilt["nodes"]}

    assert "stale_import_stub" not in node_ids
    assert any(node.get("label") == "live()" for node in rebuilt["nodes"])


def test_rebuild_lock_non_blocking_does_not_clobber_holder(tmp_path):
    """GH-858: a non-blocking caller that fails to acquire the lock must not
    truncate the holder's PID payload."""
    out = tmp_path / "graphify-out"
    lock_path = out / ".rebuild.lock"
    with _rebuild_lock(out) as outer:
        assert outer is True
        held_contents = lock_path.read_text(encoding="utf-8")
        with _rebuild_lock(out, blocking=False) as inner:
            assert inner is False
            # Holder's PID line must still be intact.
            assert lock_path.read_text(encoding="utf-8") == held_contents


def test_rebuild_code_is_idempotent_when_cluster_ids_flap(tmp_path, monkeypatch):
    from graphify import cluster as cluster_mod
    from graphify.watch import _rebuild_code

    src = tmp_path / "app.py"
    src.write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n", encoding="utf-8"
    )

    calls = {"n": 0}

    def flaky_cluster(G):
        calls["n"] += 1
        nodes = sorted(G.nodes())
        if calls["n"] % 2 == 1:
            return {100: nodes}
        return {7: nodes}

    monkeypatch.setattr(cluster_mod, "cluster", flaky_cluster)
    monkeypatch.setattr(cluster_mod, "score_all", lambda _G, comm: {cid: 1.0 for cid in comm})

    assert _rebuild_code(tmp_path)
    graph_path = tmp_path / "graphify-out" / "graph.json"
    report_path = tmp_path / "graphify-out" / "GRAPH_REPORT.md"
    first_graph = graph_path.read_text(encoding="utf-8")
    first_report = report_path.read_text(encoding="utf-8")

    assert _rebuild_code(tmp_path)
    second_graph = graph_path.read_text(encoding="utf-8")
    second_report = report_path.read_text(encoding="utf-8")

    assert first_graph == second_graph
    assert first_report == second_report


def test_rebuild_report_receives_full_detected_corpus(tmp_path, monkeypatch):
    from graphify import report as report_mod
    from graphify.watch import _rebuild_code

    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    note = tmp_path / "notes.txt"
    note.write_text("Operational context that belongs in corpus totals.\n", encoding="utf-8")
    captured = {}

    def capture_generate(
        _graph,
        _communities,
        _cohesion,
        _labels,
        _gods,
        _surprises,
        detection,
        _token_cost,
        _root,
        **_kwargs,
    ):
        captured.update(detection)
        return "# Stable report\n"

    monkeypatch.setattr(report_mod, "generate", capture_generate)

    assert _rebuild_code(tmp_path, no_viz=True)
    assert str(note) in captured["files"]["document"]


def test_rebuild_code_skips_cluster_when_topology_unchanged(tmp_path, monkeypatch):
    from graphify import cluster as cluster_mod
    from graphify.watch import _rebuild_code

    src = tmp_path / "app.py"
    src.write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n", encoding="utf-8"
    )

    calls = {"n": 0}

    def cluster_once(G):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("cluster() should be skipped when topology is unchanged")
        return {0: sorted(G.nodes())}

    monkeypatch.setattr(cluster_mod, "cluster", cluster_once)
    monkeypatch.setattr(cluster_mod, "score_all", lambda _G, comm: {cid: 1.0 for cid in comm})

    assert _rebuild_code(tmp_path)
    assert _rebuild_code(tmp_path)
    assert calls["n"] == 1


def test_rebuild_code_no_viz_removes_stale_html_and_skips_export(tmp_path, monkeypatch, capsys):
    from graphify import export as export_mod
    from graphify.watch import _rebuild_code

    (tmp_path / "app.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    out = tmp_path / "graphify-out"
    out.mkdir()
    stale_html = out / "graph.html"
    stale_html.write_text("<html/>", encoding="utf-8")

    def fail_to_html(*_args, **_kwargs):
        raise AssertionError("to_html should not be called when no_viz=True")

    monkeypatch.setattr(export_mod, "to_html", fail_to_html)

    assert _rebuild_code(tmp_path, no_viz=True)
    assert not stale_html.exists()
    assert "Skipped graph.html" not in capsys.readouterr().out


# --- .graphifyignore honored in watch handler (gh-928) ---


def test_watch_handler_honors_graphifyignore(tmp_path, monkeypatch):
    """gh-928: the watch Handler must short-circuit paths matching
    .graphifyignore so busy volumes (node_modules churn, build artefacts,
    Time Machine writes, …) don't wake the rebuild pipeline.
    """
    import threading
    from graphify import watch as watch_mod

    watch_root = tmp_path / ".hidden-parent" / "corpus"
    watch_root.mkdir(parents=True)
    (watch_root / ".graphifyignore").write_text("node_modules/\nbuild/\n", encoding="utf-8")
    (watch_root / "node_modules").mkdir()
    (watch_root / "build").mkdir()

    rebuild_calls: list[tuple[Path, dict]] = []
    notify_calls: list[Path] = []
    monkeypatch.setattr(
        watch_mod,
        "_rebuild_code",
        lambda p, **kw: rebuild_calls.append((p, kw)) or True,
    )
    monkeypatch.setattr(watch_mod, "_notify_only", lambda p: notify_calls.append(p))

    # Run watch() in a thread with a short debounce so we can verify the
    # post-debounce dispatch path actually runs on real events.
    t = threading.Thread(
        target=watch_mod.watch,
        args=(watch_root,),
        kwargs={"debounce": 0.2},
        daemon=True,
    )
    t.start()
    time.sleep(0.5)  # let observer.start() settle

    # Ignored writes — handler must drop these.
    (watch_root / "node_modules" / "junk.js").write_text("// noise\n", encoding="utf-8")
    (watch_root / "build" / "out.py").write_text("x = 1\n", encoding="utf-8")
    time.sleep(1.0)
    assert rebuild_calls == [], "ignored writes triggered a rebuild"
    assert notify_calls == [], "ignored writes triggered a notify"

    # Non-ignored write — handler must accept and (after debounce) dispatch.
    (watch_root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not rebuild_calls:
        time.sleep(0.1)
    assert rebuild_calls, "non-ignored .py write should have triggered _rebuild_code"
    changed_paths = rebuild_calls[0][1].get("changed_paths")
    assert changed_paths is not None
    assert {Path(path).name for path in changed_paths} == {"app.py"}


def test_watch_loads_graphifyignore_once(tmp_path, monkeypatch):
    """gh-928: .graphifyignore must be parsed exactly once at watch() startup,
    not per filesystem event. Otherwise busy volumes re-read the file
    thousands of times per second.
    """
    import threading
    from graphify import watch as watch_mod
    from graphify import detect as detect_mod

    (tmp_path / ".graphifyignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()

    calls = {"n": 0}
    real_loader = detect_mod._load_graphifyignore

    def counting_loader(root):
        calls["n"] += 1
        return real_loader(root)

    # Patch the symbol the watch module imported at module-load time.
    monkeypatch.setattr(watch_mod, "_load_graphifyignore", counting_loader)
    monkeypatch.setattr(watch_mod, "_rebuild_code", lambda p, **kw: True)
    monkeypatch.setattr(watch_mod, "_notify_only", lambda p: None)

    t = threading.Thread(
        target=watch_mod.watch, args=(tmp_path,), kwargs={"debounce": 0.2}, daemon=True
    )
    t.start()
    time.sleep(0.5)

    # Generate many events; loader must not be called again.
    for i in range(50):
        (tmp_path / "ignored" / f"f{i}.py").write_text("x\n", encoding="utf-8")
    time.sleep(0.7)
    assert calls["n"] == 1, f"_load_graphifyignore called {calls['n']} times; expected 1"


# --- _check_shrink: silent-corruption guard with explicit-deletion bypass ---


def _shrink_payload(n: int) -> dict:
    """Build a minimal graph-data dict with *n* placeholder nodes."""
    return {"nodes": [{"id": f"n{i}"} for i in range(n)], "links": []}


def test_check_shrink_blocks_silent_shrink(capsys):
    """Default case: smaller new graph + no force + no declared deletions = refuse."""
    ok = _check_shrink(
        force=False,
        existing_data=_shrink_payload(100),
        new_data=_shrink_payload(80),
    )
    assert ok is False
    captured = capsys.readouterr()
    assert "Refusing to overwrite" in captured.err
    assert "80 nodes" in captured.err and "100" in captured.err


def test_check_shrink_allows_force_override():
    """force=True bypasses the guard regardless of node delta."""
    ok = _check_shrink(
        force=True,
        existing_data=_shrink_payload(100),
        new_data=_shrink_payload(1),
    )
    assert ok is True


def test_check_shrink_allows_losses_owned_by_explicit_deletions(capsys):
    """A declared deletion allows only nodes owned by that deleted source."""
    existing = {
        "nodes": [
            {"id": "gone", "source_file": "gone.py"},
            {"id": "live", "source_file": "live.py"},
        ],
        "links": [],
    }
    new = {"nodes": [{"id": "live", "source_file": "live.py"}], "links": []}
    ok = _check_shrink(
        force=False,
        existing_data=existing,
        new_data=new,
        had_explicit_deletions=True,
        rebuilt_sources={"gone.py"},
    )
    assert ok is True
    assert "Refusing to overwrite" not in capsys.readouterr().err


def test_check_shrink_blocks_unexplained_loss_in_mixed_deletion_batch(capsys):
    existing = {
        "nodes": [
            {"id": "gone", "source_file": "gone.py"},
            {"id": "lost", "source_file": "untouched.py"},
            {"id": "live", "source_file": "live.py"},
        ],
        "links": [],
    }
    new = {"nodes": [{"id": "live", "source_file": "live.py"}], "links": []}

    ok = _check_shrink(
        False,
        existing,
        new,
        had_explicit_deletions=True,
        rebuilt_sources={"gone.py"},
    )

    assert ok is False
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_check_shrink_allows_no_existing_data():
    """First-run case: no existing graph → guard inert."""
    ok = _check_shrink(
        force=False,
        existing_data={},
        new_data=_shrink_payload(50),
    )
    assert ok is True


def test_check_shrink_allows_shrink_within_rebuilt_sources(capsys):
    """#1116: a symbol removed from a re-extracted file is a legitimate shrink —
    every lost node belongs to a rebuilt source, so the write proceeds (no --force)."""
    existing = {
        "nodes": [
            {"id": "a", "source_file": "m.py"},
            {"id": "b", "source_file": "m.py"},
            {"id": "c", "source_file": "other.py"},
        ],
        "links": [],
    }
    new = {
        "nodes": [
            {"id": "a", "source_file": "m.py"},
            {"id": "c", "source_file": "other.py"},
        ],
        "links": [],
    }
    ok = _check_shrink(False, existing, new, rebuilt_sources={"m.py"})
    assert ok is True
    assert "Refusing to overwrite" not in capsys.readouterr().err


def test_check_shrink_blocks_shrink_outside_rebuilt_sources(capsys):
    """The guard's real job is intact: a node lost from a file we did NOT re-extract
    (the failed-chunk signal) is still refused even with rebuilt_sources set."""
    existing = {
        "nodes": [
            {"id": "a", "source_file": "m.py"},
            {"id": "z", "source_file": "untouched.py"},
        ],
        "links": [],
    }
    new = {"nodes": [{"id": "a", "source_file": "m.py"}], "links": []}
    ok = _check_shrink(False, existing, new, rebuilt_sources={"m.py"})
    assert ok is False
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_check_shrink_allows_growth():
    """new > existing is always fine."""
    ok = _check_shrink(
        force=False,
        existing_data=_shrink_payload(50),
        new_data=_shrink_payload(60),
    )
    assert ok is True


def test_check_shrink_unlinks_tmp_on_refuse(tmp_path):
    """When refusing, the temp graph file gets cleaned up so it can't leak across runs."""
    tmp = tmp_path / "graph.tmp.json"
    tmp.write_text("{}", encoding="utf-8")
    ok = _check_shrink(
        force=False,
        existing_data=_shrink_payload(100),
        new_data=_shrink_payload(80),
        tmp=tmp,
    )
    assert ok is False
    assert not tmp.exists()


def test_check_shrink_keeps_tmp_when_deletions_declared(tmp_path):
    """Mirror of the above: if the caller declared deletions, the tmp file is NOT unlinked
    because the caller is going to swap it into place. Regression guard against a future
    bug where the tmp cleanup leaks out of the refuse branch.
    """
    tmp = tmp_path / "graph.tmp.json"
    tmp.write_text("{}", encoding="utf-8")
    existing = {
        "nodes": [
            {"id": "gone", "source_file": "gone.py"},
            {"id": "live", "source_file": "live.py"},
        ],
        "links": [],
    }
    new = {"nodes": [{"id": "live", "source_file": "live.py"}], "links": []}
    ok = _check_shrink(
        False,
        existing,
        new,
        tmp=tmp,
        had_explicit_deletions=True,
        rebuilt_sources={"gone.py"},
    )
    assert ok is True
    assert tmp.exists()


# --- _rebuild_code integration: post-commit delete scenario ---


def test_rebuild_code_prunes_deleted_file_nodes(tmp_path):
    """End-to-end probe of the post-commit-delete bug fix.

    Build a tiny graph, delete one of its source files, then call _rebuild_code
    with the deleted path in changed_paths. Without the fix this raises the
    shrink guard and refuses to write; with the fix the deleted file's nodes
    are pruned and graph.json is rewritten.
    """
    from graphify.watch import _rebuild_code

    # Set up a minimal "project" with two Python files in a git repo so detect
    # treats it as a real corpus.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )

    keep = tmp_path / "keep.py"
    drop = tmp_path / "drop.py"
    keep.write_text("def keep_fn():\n    return 1\n", encoding="utf-8")
    drop.write_text("def drop_fn():\n    return 2\n", encoding="utf-8")

    # Initial build covers both files.
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ok = _rebuild_code(tmp_path, no_cluster=True)
        assert ok is True
        graph_path = tmp_path / "graphify-out" / "graph.json"
        assert graph_path.exists()
        before = json.loads(graph_path.read_text(encoding="utf-8"))
        before_sources = {n.get("source_file") for n in before.get("nodes", [])}
        assert "drop.py" in before_sources

        # Now delete drop.py and re-run with it in the change list. This is what
        # the post-commit hook does when git diff --name-only HEAD~1 HEAD includes
        # a deletion: the path is passed to _rebuild_code even though it no
        # longer exists on disk.
        drop.unlink()
        ok = _rebuild_code(
            tmp_path,
            changed_paths=[Path("drop.py")],
            no_cluster=True,
        )
        assert ok is True, "rebuild should succeed even though the graph shrinks"

        after = json.loads(graph_path.read_text(encoding="utf-8"))
        after_sources = {n.get("source_file") for n in after.get("nodes", [])}
        assert "drop.py" not in after_sources, "deleted file's nodes should be pruned"
        assert "keep.py" in after_sources, "untouched file's nodes should survive"
    finally:
        os.chdir(cwd)


def test_rebuild_code_accepts_repo_relative_changed_path_for_subdir_root(tmp_path):
    """#1348: git-hook paths are repo-root-relative even when the graph root is a subdir."""
    from graphify.watch import _rebuild_code

    src = tmp_path / "src"
    src.mkdir()
    app = src / "app.py"
    app.write_text("def old_name():\n    return 1\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert _rebuild_code(Path("src"), no_cluster=True, acquire_lock=False) is True
        graph_path = src / "graphify-out" / "graph.json"
        before = json.loads(graph_path.read_text(encoding="utf-8"))
        assert "old_name()" in {n.get("label") for n in before.get("nodes", [])}

        app.write_text("def new_name():\n    return 2\n", encoding="utf-8")
        assert (
            _rebuild_code(
                Path("src"),
                changed_paths=[Path("src/app.py")],
                no_cluster=True,
                acquire_lock=False,
                force=True,
            )
            is True
        )

        after = json.loads(graph_path.read_text(encoding="utf-8"))
        labels = {n.get("label") for n in after.get("nodes", [])}
        assert "old_name()" not in labels
        assert "new_name()" in labels
    finally:
        os.chdir(cwd)


# --- #1059: pending-changes queue prevents commit drops under lock contention ---


def test_queue_and_drain_pending_round_trip(tmp_path):
    """Queued path records round-trip and the claimed batch is removed."""
    from graphify.watch import _queue_pending, _drain_pending, _PENDING_FILENAME

    out = tmp_path / "graphify-out"
    paths = [Path("a.py"), Path("sub/b.py"), Path("c.md")]
    _queue_pending(out, paths)

    pending_file = out / _PENDING_FILENAME
    assert pending_file.exists()
    records = [json.loads(line) for line in pending_file.read_text(encoding="utf-8").splitlines()]
    assert records == [{"path": "a.py"}, {"path": "sub/b.py"}, {"path": "c.md"}]

    drained = _drain_pending(out)
    assert drained == paths
    # Drain unlinks so subsequent callers see an empty queue.
    assert not pending_file.exists()
    assert _drain_pending(out) == []


def test_queue_and_drain_pending_round_trips_newline_in_path(tmp_path):
    from graphify.watch import _drain_pending, _queue_pending

    out = tmp_path / "graphify-out"
    unusual = Path("src/line\nbreak.py")

    _queue_pending(out, [unusual])

    assert _drain_pending(out) == [unusual]


def test_drain_pending_dedupes_and_skips_blank_lines(tmp_path):
    """Repeated appends across concurrent contenders must dedupe; partial
    writes leaving blank lines must not poison the merge."""
    from graphify.watch import _queue_pending, _drain_pending

    out = tmp_path / "graphify-out"
    _queue_pending(out, [Path("a.py"), Path("b.py")])
    _queue_pending(out, [Path("b.py"), Path("c.py")])
    # Simulate a torn write leaving an empty line.
    with open(out / ".pending_changes", "a", encoding="utf-8") as fh:
        fh.write("\n   \n")

    drained = _drain_pending(out)
    assert drained == [Path("a.py"), Path("b.py"), Path("c.py")]


def test_queue_pending_noop_on_empty_list(tmp_path):
    """Empty change set must not create an empty .pending_changes file."""
    from graphify.watch import _queue_pending, _PENDING_FILENAME

    out = tmp_path / "graphify-out"
    _queue_pending(out, [])
    assert not (out / _PENDING_FILENAME).exists()


def test_rebuild_code_queues_on_lock_contention(tmp_path, monkeypatch, capsys):
    """#1059: when the rebuild lock is held, an incremental hook must queue
    its changed_paths to .pending_changes and print 'queued' instead of
    silently dropping the change set."""
    from graphify.watch import (
        _PENDING_FILENAME,
        _pending_paths,
        _rebuild_code,
        _rebuild_lock,
    )

    out = tmp_path / "graphify-out"
    out.mkdir()

    # Hold the lock so the next non-blocking attempt fails. Use a real
    # _rebuild_lock context manager in this same process — flock on the same
    # file descriptor would otherwise be re-entrant on Linux, so we open
    # the file ourselves via the lock helper.
    with _rebuild_lock(out, blocking=False) as outer_got:
        assert outer_got is True

        ok = _rebuild_code(
            tmp_path,
            changed_paths=[Path("a.py"), Path("b.py")],
        )
        assert ok is False

        # Output should say "queued", not "skipping".
        captured = capsys.readouterr().out
        assert "queued" in captured.lower()
        assert "skipping" not in captured.lower()

        # And the paths must have been written to the pending file so the
        # eventual lock-holder can drain them.
        pending = out / _PENDING_FILENAME
        assert pending.exists()
        assert _pending_paths([pending.read_text(encoding="utf-8")]) == [
            Path("a.py"),
            Path("b.py"),
        ]


def test_rebuild_code_merges_pending_on_acquire(tmp_path, monkeypatch):
    """#1059: the process that acquires the lock must drain .pending_changes
    and pass the merged change set to the inner rebuild call."""
    from graphify import watch as watch_mod

    out = tmp_path / "graphify-out"
    out.mkdir()
    # Pre-populate the queue as if an earlier contender had dropped its paths.
    watch_mod._queue_pending(out, [Path("queued1.py"), Path("queued2.py")])

    # Snapshot the original BEFORE monkeypatching so we can drive the outer
    # dispatch path while the inner recursive call resolves to our spy.
    orig_rebuild = watch_mod._rebuild_code
    inner_calls: list[list[str]] = []

    def recording_inner(watch_path, **kwargs):
        if kwargs.get("acquire_lock") is False:
            paths = kwargs.get("changed_paths") or []
            inner_calls.append([p.as_posix() for p in paths])
        return True

    monkeypatch.setattr(watch_mod, "_rebuild_code", recording_inner)

    ok = orig_rebuild(
        tmp_path,
        changed_paths=[Path("own.py"), Path("queued1.py")],
    )
    assert ok is True

    # The first inner call must have received the merged + deduped set:
    # own.py first (caller's order preserved), then drained queued1/queued2,
    # with queued1.py deduped against own's prior occurrence.
    assert inner_calls, "inner _rebuild_code should have been called"
    assert inner_calls[0] == ["own.py", "queued1.py", "queued2.py"]

    # And .pending_changes was drained.
    assert not (out / watch_mod._PENDING_FILENAME).exists()


def test_rebuild_code_drains_late_arrivals(tmp_path, monkeypatch):
    """#1059: after the primary rebuild, the lock-holder must loop and drain
    any paths queued by hooks that arrived mid-rebuild."""
    from graphify import watch as watch_mod
    from graphify.watch import _rebuild_code as orig_rebuild

    out = tmp_path / "graphify-out"
    out.mkdir()

    inner_calls: list[list[str]] = []
    call_state = {"i": 0}

    def fake_inner(watch_path, **kwargs):
        if kwargs.get("acquire_lock") is False:
            paths = [p.as_posix() for p in (kwargs.get("changed_paths") or [])]
            inner_calls.append(paths)
            # Simulate a late-arriving hook that queues during the FIRST
            # inner rebuild only. The outer drain loop must see it.
            call_state["i"] += 1
            if call_state["i"] == 1:
                watch_mod._queue_pending(out, [Path("late.py")])
        return True

    monkeypatch.setattr(watch_mod, "_rebuild_code", fake_inner)

    ok = orig_rebuild(tmp_path, changed_paths=[Path("own.py")])
    assert ok is True

    # First inner call covers our own change set; second is the late-drain
    # pass that picks up "late.py".
    assert len(inner_calls) >= 2
    assert inner_calls[0] == ["own.py"]
    assert inner_calls[1] == ["late.py"]
    # And the queue is now empty (no further late drains).
    assert not (out / watch_mod._PENDING_FILENAME).exists()


def test_rebuild_code_keeps_claimed_changes_after_failed_rebuild(tmp_path, monkeypatch):
    """A failed rebuild must leave its claimed paths durable for the next run."""
    from graphify import watch as watch_mod
    from graphify.watch import _rebuild_code as orig_rebuild

    out = tmp_path / "graphify-out"
    out.mkdir()
    calls: list[list[str]] = []
    outcomes = iter((False, True))

    def fake_inner(watch_path, **kwargs):
        if kwargs.get("acquire_lock") is False:
            calls.append([p.as_posix() for p in (kwargs.get("changed_paths") or [])])
            return next(outcomes)
        return True

    monkeypatch.setattr(watch_mod, "_rebuild_code", fake_inner)

    assert orig_rebuild(tmp_path, changed_paths=[Path("lost.py")]) is False
    durable = [
        path for path in out.glob(".pending_changes*") if path.name != ".pending_changes.guard"
    ]
    assert durable, "failed rebuild discarded its pending change set"

    assert orig_rebuild(tmp_path, changed_paths=[]) is True
    assert calls == [["lost.py"], ["lost.py"]]
    assert list(out.glob(".pending_changes.inflight.*")) == []
    assert not (out / watch_mod._PENDING_FILENAME).exists()


def test_rebuild_code_recovers_unacknowledged_claim_after_restart(tmp_path, monkeypatch):
    """A claimed batch remains recoverable when a process dies before acknowledgment."""
    from graphify import watch as watch_mod
    from graphify.watch import _rebuild_code as orig_rebuild

    out = tmp_path / "graphify-out"
    out.mkdir()
    watch_mod._queue_pending(out, [Path("crash.py")])
    claimed, claim_files = watch_mod._claim_pending(out)
    assert claimed == [Path("crash.py")]
    assert claim_files and all(path.exists() for path in claim_files)

    calls: list[list[str]] = []

    def successful_inner(watch_path, **kwargs):
        if kwargs.get("acquire_lock") is False:
            calls.append([p.as_posix() for p in (kwargs.get("changed_paths") or [])])
        return True

    monkeypatch.setattr(watch_mod, "_rebuild_code", successful_inner)

    assert orig_rebuild(tmp_path, changed_paths=[]) is True
    assert calls == [["crash.py"]]
    assert all(not path.exists() for path in claim_files)


def test_rebuild_code_full_corpus_skips_pending_queue(tmp_path, monkeypatch):
    """#1059: changed_paths=None means a full-corpus rebuild — the queue
    must not be touched on the failure path because there is nothing
    incremental to preserve."""
    from graphify import watch as watch_mod
    from graphify.watch import _rebuild_code as orig_rebuild

    out = tmp_path / "graphify-out"
    out.mkdir()

    # Pre-existing queued paths from an earlier incremental hook.
    watch_mod._queue_pending(out, [Path("earlier.py")])

    # Force the inner call to record what it saw.
    seen: list = []

    def fake_inner(watch_path, **kwargs):
        if kwargs.get("acquire_lock") is False:
            seen.append(kwargs.get("changed_paths"))
        return True

    monkeypatch.setattr(watch_mod, "_rebuild_code", fake_inner)

    ok = orig_rebuild(tmp_path, changed_paths=None)
    assert ok is True
    # Full-corpus rebuild passes None to the inner call (does not merge in
    # the queued paths — a full rebuild already covers them).
    assert seen == [None]
    # The queue still gets drained on entry so stale entries don't leak,
    # but no late-arrival loop runs for the full-corpus path.
    assert not (out / watch_mod._PENDING_FILENAME).exists()


def test_merge_changed_paths_dedupes_in_order():
    """_merge_changed_paths preserves first-seen order and drops dupes."""
    from graphify.watch import _merge_changed_paths

    merged = _merge_changed_paths(
        [Path("a.py"), Path("b.py")],
        None,
        [Path("b.py"), Path("c.py")],
        [Path("a.py")],
    )
    assert [p.as_posix() for p in merged] == ["a.py", "b.py", "c.py"]


# --- PR 7: MultiDiGraph keyed parallel-edge eviction + canonical comparison ----
#
# These exercise the incremental-update path of _rebuild_code against an on-disk
# MultiDiGraph graph.json. _rebuild_code's eviction logic (preserved_edges)
# operates on the raw on-disk "links" records BEFORE any graph build, and each
# parallel edge is one record carrying its own `key` + `source_file`, so the
# logic is naturally key-aware. The go/no-go gate: "Incremental update preserves
# and evicts keyed parallel edges intentionally, with no silent fallback to
# simple graph behavior."


def _git_init(path: Path) -> None:
    """Initialise a throwaway git repo so detect() treats `path` as a real corpus."""
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


def _build_multigraph_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a repo whose graph.json is a MultiDiGraph with two stable endpoints.

    Returns (repo_dir, a_id, b_id). Nodes ``afn``/``bfn`` live in dedicated,
    never-changed files (amod.py / bmod.py) so re-extraction of an edge-
    contributing file does not re-emit or evict them. The edge-contributing
    files (file1.py / file2.py / edgesrc.py) exist as tracked code so detect()
    keeps them in the corpus; parallel A->B edges are injected directly into the
    on-disk "links" so each carries its own `key` + `source_file`.
    """
    from graphify.watch import _rebuild_code

    _git_init(tmp_path)
    (tmp_path / "amod.py").write_text("def afn():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bmod.py").write_text("def bfn():\n    return 2\n", encoding="utf-8")
    (tmp_path / "file1.py").write_text("x1 = 1\n", encoding="utf-8")
    (tmp_path / "file2.py").write_text("x2 = 2\n", encoding="utf-8")
    (tmp_path / "edgesrc.py").write_text("y = 1\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert _rebuild_code(tmp_path, no_cluster=True) is True
    finally:
        os.chdir(cwd)

    graph_path = tmp_path / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    a_id = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("afn("))
    b_id = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("bfn("))
    return graph_path, a_id, b_id


def _promote_to_multidigraph(data: dict) -> None:
    links = data.get("links", data.get("edges", []))
    for index, edge in enumerate(links):
        edge.setdefault("key", f"fixture-edge-{index}")
    data["multigraph"] = True
    data["directed"] = True
    data.setdefault("graph", {}).setdefault("graphify_profile", {})["graph_type"] = "multidigraph"


def _set_links(graph_path: Path, base_data: dict, a_id: str, b_id: str, edges: list) -> None:
    """Append `edges` (A->B parallel records) and stamp multigraph flags on disk."""
    links = base_data.get("links", base_data.get("edges", []))
    for edge in edges:
        edge.setdefault("_origin", "ast")
    links += edges
    base_data["links"] = links
    _promote_to_multidigraph(base_data)
    graph_path.write_text(json.dumps(base_data, indent=2), encoding="utf-8")


def _ab_links(graph_path: Path, a_id: str, b_id: str, source_file: str | None = None) -> list:
    """Return the surviving A->B link records on disk, optionally filtered by source_file."""
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    links = data.get("links", data.get("edges", []))
    out = [e for e in links if e.get("source") == a_id and e.get("target") == b_id]
    if source_file is not None:
        out = [e for e in out if e.get("source_file") == source_file]
    return out


def test_watch_multigraph_unchanged_file_parallel_edges_persist(tmp_path):
    """A pair with 3 parallel edges from a file that is NOT changed must keep all
    3 across an incremental rebuild triggered by an unrelated file."""
    from graphify.watch import _rebuild_code

    graph_path, a_id, b_id = _build_multigraph_repo(tmp_path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    _set_links(
        graph_path,
        data,
        a_id,
        b_id,
        [
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "edgesrc.py",
                "source_location": "L1",
                "key": "k1",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": "edgesrc.py",
                "source_location": "L2",
                "key": "k2",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "references",
                "confidence": "EXTRACTED",
                "source_file": "edgesrc.py",
                "source_location": "L3",
                "key": "k3",
            },
        ],
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Change an UNRELATED file; edgesrc.py (the edge contributor) is untouched.
        (tmp_path / "other_change.py").write_text("def newfn():\n    return 0\n", encoding="utf-8")
        assert _rebuild_code(tmp_path, changed_paths=[Path("other_change.py")], no_cluster=True)
    finally:
        os.chdir(cwd)

    survivors = _ab_links(graph_path, a_id, b_id, source_file="edgesrc.py")
    assert len(survivors) == 3, "all 3 parallel edges from the unchanged file must persist"
    assert {e["relation"] for e in survivors} == {"calls", "imports", "references"}


def test_watch_multigraph_changed_file_evicts_its_parallel_edges(tmp_path):
    """A pair A->B with parallel edges from file1 AND file2; changing file1 must
    evict file1's parallel edges between A->B while file2's survive (keyed,
    per-source_file eviction — no collapse to one-edge-per-pair behaviour)."""
    from graphify.watch import _rebuild_code

    graph_path, a_id, b_id = _build_multigraph_repo(tmp_path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    _set_links(
        graph_path,
        data,
        a_id,
        b_id,
        [
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "file1.py",
                "source_location": "L1",
                "key": "k_f1_a",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": "file1.py",
                "source_location": "L2",
                "key": "k_f1_b",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "file2.py",
                "source_location": "L9",
                "key": "k_f2_a",
            },
        ],
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        (tmp_path / "file1.py").write_text("x1 = 99\n", encoding="utf-8")
        assert _rebuild_code(tmp_path, changed_paths=[Path("file1.py")], no_cluster=True)
    finally:
        os.chdir(cwd)

    assert _ab_links(graph_path, a_id, b_id, source_file="file1.py") == [], (
        "file1's parallel A->B edges must be evicted when file1 changes"
    )
    file2_survivors = _ab_links(graph_path, a_id, b_id, source_file="file2.py")
    assert len(file2_survivors) == 1, "file2's parallel A->B edge must survive selectively"
    assert file2_survivors[0]["relation"] == "calls"


def test_watch_multigraph_changed_file_evicts_stale_cross_file_edge(tmp_path):
    """The FIX 3 gap: an edge between two SURVIVING nodes that was CONTRIBUTED by
    the changed file must be evicted. The old endpoints-only check wrongly kept
    it because both A and B (defined in unchanged files) still exist."""
    from graphify.watch import _rebuild_code

    graph_path, a_id, b_id = _build_multigraph_repo(tmp_path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    # A single stale cross-file edge contributed by file1.py between A and B,
    # both of which live in amod.py / bmod.py and therefore survive the change.
    _set_links(
        graph_path,
        data,
        a_id,
        b_id,
        [
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "file1.py",
                "source_location": "L1",
                "key": "stale",
            },
        ],
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        (tmp_path / "file1.py").write_text("x1 = 99\n", encoding="utf-8")
        assert _rebuild_code(tmp_path, changed_paths=[Path("file1.py")], no_cluster=True)
    finally:
        os.chdir(cwd)

    assert _ab_links(graph_path, a_id, b_id, source_file="file1.py") == [], (
        "stale cross-file edge contributed by the changed file must be evicted "
        "even though both endpoints survive (FIX 3)"
    )


def test_watch_multigraph_deleted_file_removes_all_its_edge_records(tmp_path):
    """Deleting a file must remove ALL its edge records, including parallels,
    while leaving another file's parallel between the same pair intact."""
    from graphify.watch import _rebuild_code

    graph_path, a_id, b_id = _build_multigraph_repo(tmp_path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    _set_links(
        graph_path,
        data,
        a_id,
        b_id,
        [
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "file1.py",
                "source_location": "L1",
                "key": "d1",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": "file1.py",
                "source_location": "L2",
                "key": "d2",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "EXTRACTED",
                "source_file": "file2.py",
                "source_location": "L3",
                "key": "keep",
            },
        ],
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Delete file1.py and pass it in changed_paths (post-commit-hook style).
        (tmp_path / "file1.py").unlink()
        assert _rebuild_code(tmp_path, changed_paths=[Path("file1.py")], no_cluster=True)
    finally:
        os.chdir(cwd)

    assert _ab_links(graph_path, a_id, b_id, source_file="file1.py") == [], (
        "all of the deleted file's edge records (incl. parallels) must be removed"
    )
    assert len(_ab_links(graph_path, a_id, b_id, source_file="file2.py")) == 1, (
        "the surviving file's parallel edge between the same pair must be kept"
    )


def test_watch_canonical_comparison_distinguishes_parallel_edges():
    """Two multigraphs differing ONLY in a parallel edge's presence must canonical-
    compare as DIFFERENT (FIX 2). Identical multigraphs must compare EQUAL, and
    two parallels that differ ONLY in `key` must stay distinct (key is the
    load-bearing identity field that keeps parallels from collapsing)."""
    from graphify.watch import _canonical_topology_for_compare

    nodes = [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}]
    e1 = {
        "source": "A",
        "target": "B",
        "relation": "calls",
        "source_file": "f1.py",
        "source_location": "L1",
        "key": "k1",
    }
    e2 = {
        "source": "A",
        "target": "B",
        "relation": "calls",
        "source_file": "f1.py",
        "source_location": "L2",
        "key": "k2",
    }
    profile = {"graphify_profile": {"graph_type": "multidigraph"}}

    two = {"nodes": nodes, "links": [dict(e1), dict(e2)], "graph": dict(profile)}
    one = {"nodes": nodes, "links": [dict(e1)], "graph": dict(profile)}
    two_again = {"nodes": nodes, "links": [dict(e1), dict(e2)], "graph": dict(profile)}

    def canon(g: dict) -> str:
        return json.dumps(_canonical_topology_for_compare(g), sort_keys=True)

    assert canon(two) != canon(one), "adding a parallel edge must register as a change"
    assert canon(two) == canon(two_again), "identical multigraphs must compare equal"

    # Two parallels identical in every field EXCEPT key must remain distinct.
    twin_a = {
        "source": "A",
        "target": "B",
        "relation": "calls",
        "source_file": "f1.py",
        "source_location": "L1",
        "key": "ka",
    }
    twin_b = {
        "source": "A",
        "target": "B",
        "relation": "calls",
        "source_file": "f1.py",
        "source_location": "L1",
        "key": "kb",
    }
    twins = {"nodes": nodes, "links": [twin_a, twin_b]}
    canon_twins = _canonical_topology_for_compare(twins)
    assert len(canon_twins["links"]) == 2, "key-only-different parallels must not collapse"
    assert all("key" in e for e in canon_twins["links"]), "canonical edge must retain `key`"
    single_twin = {"nodes": nodes, "links": [dict(twin_a)]}
    assert canon(twins) != canon(single_twin), (
        "removing a key-only-different parallel must register as a change"
    )


def test_watch_simple_mode_unchanged_regression(tmp_path, monkeypatch):
    """Simple-graph watch rebuild behaves as before: a topology-unchanged second
    pass still skips cluster(). Guards the FIX 1 regression (graph-level
    graphify_profile metadata must not be read as a topology change)."""
    from graphify import cluster as cluster_mod
    from graphify.watch import _rebuild_code

    (tmp_path / "app.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n", encoding="utf-8"
    )

    calls = {"n": 0}

    def cluster_once(G):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("cluster() must be skipped when topology is unchanged")
        return {0: sorted(G.nodes())}

    monkeypatch.setattr(cluster_mod, "cluster", cluster_once)
    monkeypatch.setattr(cluster_mod, "score_all", lambda _G, comm: {cid: 1.0 for cid in comm})

    assert _rebuild_code(tmp_path)
    assert _rebuild_code(tmp_path)
    assert calls["n"] == 1, "topology-unchanged simple-graph rebuild must not re-cluster"


def test_full_state_compare_detects_metadata_only_changes():
    from graphify.watch import _canonical_graph_for_compare, _canonical_topology_for_compare

    left = {
        "nodes": [{"id": "a"}],
        "links": [],
        "hyperedges": [],
        "graph": {"graphify_profile": {"graph_type": "simple"}, "diagnostic": "old"},
        "graphify_identity_diagnostics": {"collisions": 1},
    }
    right = json.loads(json.dumps(left))
    right["graph"]["diagnostic"] = "fresh"
    right["graphify_identity_diagnostics"] = {"collisions": 0}

    assert _canonical_graph_for_compare(left) != _canonical_graph_for_compare(right)
    assert _canonical_topology_for_compare(left) == _canonical_topology_for_compare(right)


@pytest.mark.parametrize("no_cluster", [False, True])
def test_rebuild_publishes_fresh_diagnostics_and_keeps_custom_metadata(
    tmp_path, monkeypatch, no_cluster
):
    import graphify.extract as extract_mod
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=no_cluster, no_viz=True, acquire_lock=False)
    graph_path = corpus / "graphify-out" / "graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload.setdefault("graph", {})["custom_meta"] = "keep"
    payload["graph"]["graphify_identity_diagnostics"] = {"generation": "old"}
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    real_extract = extract_mod.extract

    def extract_with_fresh_diagnostics(*args, **kwargs):
        result = real_extract(*args, **kwargs)
        result["graphify_identity_diagnostics"] = {"generation": "fresh"}
        return result

    monkeypatch.setattr(extract_mod, "extract", extract_with_fresh_diagnostics)
    assert _rebuild_code(corpus, no_cluster=no_cluster, no_viz=True, acquire_lock=False)

    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    assert rebuilt["graph"]["custom_meta"] == "keep"
    diagnostics = rebuilt.get("graphify_identity_diagnostics") or rebuilt["graph"].get(
        "graphify_identity_diagnostics"
    )
    assert diagnostics != {"generation": "old"}
    if no_cluster:
        assert diagnostics == {"generation": "fresh"}
    else:
        assert "skipped_contested_endpoints" in diagnostics


def test_watch_multigraph_full_rebuild_preserves_profile_flag(tmp_path, monkeypatch):
    """Regression for the DEFERRED silent collapse to simple graph.

    A MultiDiGraph graph.json with keyed parallel edges, put through a
    TOPOLOGY-CHANGING rebuild (a new file is added so _rebuild_code does NOT
    early-return on the unchanged-topology check and actually rewrites
    graph.json via to_json), must rewrite a graph.json that still:
      - declares ``multigraph == true`` (the flag load_graph keys on),
      - carries ``graphify_profile.graph_type == "multidigraph"``,
      - keeps the parallel A->B edge records, and
      - reloads via the production loader as a MultiDiGraph with all parallels
        intact (NOT collapsed to one edge per pair).

    Before the inherit-multigraph fix, _rebuild_code built a simple DiGraph, so
    to_json wrote a graph.json with no multigraph flag and a "simple" profile —
    the next load_graph would collapse the preserved parallel links to a single
    edge (the PR 7 go/no-go violation: "no silent fallback to simple graph
    behavior"). This test fails on that regression and passes once _rebuild_code
    inherits the saved multigraph class.
    """
    from graphify.watch import _rebuild_code
    from graphify.graph_loader import load_graph, GRAPHIFY_PROFILE_KEY
    from graphify import cluster as cluster_mod
    import networkx as nx

    monkeypatch.setattr(
        cluster_mod,
        "cluster",
        lambda G: {0: sorted(G.nodes(), key=str)},
    )
    monkeypatch.setattr(cluster_mod, "score_all", lambda _G, comm: {cid: 1.0 for cid in comm})

    graph_path, a_id, b_id = _build_multigraph_repo(tmp_path)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    # Three keyed parallel A->B SEMANTIC edges (``_origin="semantic"``) sourced from
    # amod.py. A full rebuild re-extracts amod.py, but the LLM/semantic pass is NOT
    # re-run from the CLI, so the #1521 hybrid must PRESERVE these (an AST-only
    # rebuild can never re-supply them) while the clustered rewrite still keeps the
    # multidigraph profile + parallel records intact. (Pre-#1521 this test injected
    # AST parallels and relied on preserve-everything-by-endpoint; #1521 evicts AST
    # edges of re-extracted files, so the substrate is now the semantic layer the
    # hybrid is designed to protect.)
    _set_links(
        graph_path,
        data,
        a_id,
        b_id,
        [
            {
                "source": a_id,
                "target": b_id,
                "relation": "calls",
                "confidence": "INFERRED",
                "_origin": "semantic",
                "source_file": "amod.py",
                "source_location": "L1",
                "key": "mk1",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "imports",
                "confidence": "INFERRED",
                "_origin": "semantic",
                "source_file": "amod.py",
                "source_location": "L2",
                "key": "mk2",
            },
            {
                "source": a_id,
                "target": b_id,
                "relation": "references",
                "confidence": "INFERRED",
                "_origin": "semantic",
                "source_file": "amod.py",
                "source_location": "L3",
                "key": "mk3",
            },
        ],
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Add a NEW file: this changes topology so the rebuild does NOT hit the
        # "no topology change" early return and genuinely rewrites graph.json via
        # the clustered to_json path. no_viz keeps the test fast.
        (tmp_path / "newmod.py").write_text("def newfn():\n    return 9\n", encoding="utf-8")
        assert _rebuild_code(tmp_path, no_viz=True) is True
    finally:
        os.chdir(cwd)

    rewritten = json.loads(graph_path.read_text(encoding="utf-8"))
    # 1. The multigraph flag survived the rewrite.
    assert rewritten.get("multigraph") is True, (
        "rewritten graph.json must keep multigraph=true (else next load collapses parallels)"
    )
    # 2. The multidigraph profile survived (Phase A persists it from the instance).
    profile = (rewritten.get("graph") or {}).get(GRAPHIFY_PROFILE_KEY) or {}
    assert profile.get("graph_type") == "multidigraph", (
        f"rewritten graphify_profile must be multidigraph, got {profile!r}"
    )
    # 3. The parallel edge records are still present (3 A->B records from amod.py).
    ab = _ab_links(graph_path, a_id, b_id, source_file="amod.py")
    assert len(ab) == 3, f"all 3 parallel A->B edge records must persist, got {len(ab)}"
    # 4. The new file's node landed (proves the rebuild actually re-ran, not a no-op).
    new_labels = {n.get("label", "") for n in rewritten.get("nodes", [])}
    assert any(lbl.startswith("newfn(") for lbl in new_labels), (
        "the topology-changing new file must have been extracted into the rewrite"
    )
    # 5. The production loader reloads it as a MultiDiGraph with parallels intact —
    #    the definitive proof there is no deferred collapse to simple.
    reloaded = load_graph(rewritten)
    assert isinstance(reloaded, nx.MultiDiGraph), (
        f"reloaded graph must be a MultiDiGraph, got {type(reloaded).__name__}"
    )
    assert reloaded.number_of_edges(a_id, b_id) == 3, (
        "reloaded MultiDiGraph must keep all 3 parallel A->B edges (NOT collapsed to 1)"
    )


def test_watch_no_cluster_full_rebuild_does_not_duplicate_links(tmp_path):
    """A full raw rebuild must be idempotent for links.

    The full no-cluster path re-extracts every code file and also preserves
    existing links. Without a dedupe pass, each full rebuild appends another copy
    of the same AST edge records.
    """
    from graphify.watch import _rebuild_code

    (tmp_path / "app.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n",
        encoding="utf-8",
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False)
        graph_path = tmp_path / "graphify-out" / "graph.json"
        first = json.loads(graph_path.read_text(encoding="utf-8"))
        first_links = first.get("links", [])

        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False)
        second = json.loads(graph_path.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    second_links = second.get("links", [])
    assert len(second_links) == len(first_links)

    def fingerprint(edge: dict) -> str:
        comparable = dict(edge)
        comparable.pop("key", None)
        comparable.pop("confidence_score", None)
        return json.dumps(comparable, sort_keys=True, ensure_ascii=False)

    assert len({fingerprint(edge) for edge in second_links}) == len(second_links)


def test_watch_full_rebuild_preserves_real_call_site_parallels(tmp_path):
    """A real multigraph corpus where one function calls another at THREE distinct
    source locations yields three parallel call edges (distinct source_location ->
    distinct stable keys). A full re-extracting rebuild must keep all three (the
    fresh AST pass re-supplies them) and the production loader must reload them as a
    MultiDiGraph with three parallel edges — NOT collapsed to one.

    Replaces the old synthetic ``parallel-a``/``parallel-b`` injection test (#1521):
    real call-site parallels are AST-AST edges, so the hybrid evicts them on the
    full rebuild and the fresh AST pass re-emits them identically (dedupe collapses
    the keyless duplicates)."""
    from graphify.watch import _rebuild_code
    from graphify.graph_loader import load_graph
    import networkx as nx

    _git_init(tmp_path)
    # beta calls alpha at L5, L6, L7 -> three real call-site edges.
    (tmp_path / "app.py").write_text(
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "def beta():\n"
        "    alpha()\n"
        "    x = alpha()\n"
        "    return alpha()\n",
        encoding="utf-8",
    )

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Initial build, then promote the on-disk graph to multigraph so the
        # no-cluster rebuild keeps parallels instead of collapsing them.
        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False) is True
        graph_path = tmp_path / "graphify-out" / "graph.json"
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        _promote_to_multidigraph(data)
        graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Full re-extraction (changed_paths is None) re-extracts app.py and must
        # re-supply all three parallels via the fresh AST pass.
        assert _rebuild_code(tmp_path, no_cluster=True, acquire_lock=False) is True
        rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    def links(d):
        return d.get("links", d.get("edges", []))

    a = next(n["id"] for n in rebuilt["nodes"] if n.get("label", "").startswith("alpha("))
    b = next(n["id"] for n in rebuilt["nodes"] if n.get("label", "").startswith("beta("))
    calls = [
        e
        for e in links(rebuilt)
        if e.get("source") == b and e.get("target") == a and e.get("relation") == "calls"
    ]
    assert len(calls) == 3, f"all three real call-site parallels must survive, got {len(calls)}"
    assert {e.get("source_location") for e in calls} == {"L5", "L6", "L7"}

    reloaded = load_graph(rebuilt)
    assert isinstance(reloaded, nx.MultiDiGraph)
    assert reloaded.number_of_edges(b, a) == 3, "reloaded MultiDiGraph must keep 3 parallels"


def test_watch_full_rebuild_prunes_removed_import_edge(tmp_path):
    """A removed import between two surviving files must NOT linger as a phantom
    stale edge on a full rebuild (the #1521 motivation). Both endpoint nodes still
    exist, so endpoint-membership preservation alone would keep the dead edge;
    source_file-scoped eviction must drop it. The import edge is an AST-AST edge,
    so the #1521 hybrid still evicts it (only semantic / non-AST-touching edges are
    spared)."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _git_init(corpus)
    (corpus / "a.py").write_text("import b\n\ndef use():\n    return b.thing\n", encoding="utf-8")
    (corpus / "b.py").write_text("thing = 1\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(corpus)
        assert _rebuild_code(corpus, acquire_lock=False) is True
        graph_path = corpus / "graphify-out" / "graph.json"
        before = json.loads(graph_path.read_text(encoding="utf-8"))
        before_links = before.get("links", before.get("edges", []))
        assert any(e.get("relation") == "imports" for e in before_links), (
            "the import edge must exist before removal"
        )

        # Remove the import; both a.py and b.py survive.
        (corpus / "a.py").write_text("def use():\n    return 1\n", encoding="utf-8")
        assert _rebuild_code(corpus, acquire_lock=False) is True
        after = json.loads(graph_path.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    after_links = after.get("links", after.get("edges", []))
    assert not any(e.get("relation") == "imports" for e in after_links), (
        "removed import must not survive as a phantom stale edge on full rebuild"
    )


@pytest.mark.parametrize("no_cluster", [False, True])
def test_watch_full_rebuild_preserves_non_ast_edges(tmp_path, no_cluster):
    """A full rebuild is AST-only (the LLM/semantic pass is NOT re-run from the
    CLI), so a semantic / INFERRED (_origin != "ast") edge whose endpoints survive
    must be PRESERVED across a full rebuild. #1521's blanket source_file eviction
    over-evicts a *sourced* semantic edge (it is not re-supplied by the AST pass);
    the hybrid scopes eviction to AST/structural edges only. The sourceless
    semantic edge already survives today and guards the no-regression direction."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _git_init(corpus)
    (corpus / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (corpus / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(corpus)
        assert _rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
        graph_path = corpus / "graphify-out" / "graph.json"
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        foo = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("foo("))
        bar = next(n["id"] for n in data["nodes"] if n.get("label", "").startswith("bar("))
        links = data.get("links", data.get("edges", []))
        # Sourced semantic edge — the AST pass will NOT re-emit it.
        links.append(
            {
                "source": foo,
                "target": bar,
                "relation": "depends_on",
                "confidence": "INFERRED",
                "_origin": "semantic",
                "source_file": "a.py",
            }
        )
        # A legacy edge between AST endpoints still is not AST-owned. Inferring
        # its producer from endpoint types silently loses relationships that the
        # fresh AST pass does not recreate.
        links.append(
            {
                "source": foo,
                "target": bar,
                "relation": "legacy_context",
                "confidence": "EXTRACTED",
                "_origin": "legacy",
                "source_file": "a.py",
                "key": "legacy-context-key",
            }
        )
        # Sourceless semantic edge — must also survive.
        links.append(
            {
                "source": foo,
                "target": bar,
                "relation": "relates_to",
                "confidence": "INFERRED",
                "_origin": "semantic",
            }
        )
        data["links"] = links
        # Multigraph so the two semantic edges on the same foo->bar pair coexist as
        # parallels — a simple DiGraph collapses them to one, masking the assertion.
        # The fork's real semantic layer lives in a MultiDiGraph anyway.
        _promote_to_multidigraph(data)
        graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Full rebuild re-extracts a.py and b.py (AST only).
        assert _rebuild_code(corpus, no_cluster=no_cluster, acquire_lock=False) is True
        after = json.loads(graph_path.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    after_links = after.get("links", after.get("edges", []))
    assert any(e.get("relation") == "relates_to" for e in after_links), (
        "sourceless semantic edge must survive a full rebuild"
    )
    assert any(e.get("relation") == "depends_on" for e in after_links), (
        "sourced semantic (_origin!=ast) edge must survive a full rebuild "
        "(hybrid: scope eviction to AST/structural edges only)"
    )
    assert any(
        e.get("relation") == "legacy_context" and e.get("key") == "legacy-context-key"
        for e in after_links
    ), "legacy edges are not AST-owned merely because both endpoints are AST nodes"


def test_watch_incremental_rebuild_preserves_semantic_edge_and_prunes_stale_ast(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _git_init(corpus)
    source = corpus / "a.py"
    source.write_text("import b\n\ndef use():\n    return b.thing\n", encoding="utf-8")
    (corpus / "b.py").write_text("thing = 1\n", encoding="utf-8")

    assert _rebuild_code(
        corpus,
        graph_type="multidigraph",
        no_cluster=True,
        acquire_lock=False,
    )
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    import_edge = next(edge for edge in data["links"] if edge.get("relation") == "imports")
    data["links"].append(
        {
            "source": import_edge["source"],
            "target": import_edge["target"],
            "key": "semantic:dependency",
            "relation": "depends_on",
            "confidence": "INFERRED",
            "_origin": "semantic",
            "source_file": "a.py",
        }
    )
    graph_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    source.write_text("def use():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(
        corpus,
        changed_paths=[source],
        no_cluster=True,
        acquire_lock=False,
    )

    updated = json.loads(graph_path.read_text(encoding="utf-8"))
    assert not any(edge.get("relation") == "imports" for edge in updated["links"])
    assert any(edge.get("relation") == "depends_on" for edge in updated["links"])


@pytest.mark.parametrize("no_cluster", [True, False])
def test_watch_manifest_failure_rolls_back_graph_state(tmp_path, monkeypatch, no_cluster):
    import importlib

    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "app.py"
    source.write_text("def before():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    output = corpus / "graphify-out"
    (output / ".graphify_semantic_marker").write_text("{}", encoding="utf-8")
    source.write_text("def after():\n    return 2\n", encoding="utf-8")
    tree_before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    detect_module = importlib.import_module("graphify.detect")

    def fail_manifest(*args, **kwargs):
        raise OSError("simulated watch manifest failure")

    monkeypatch.setattr(detect_module, "save_manifest", fail_manifest)

    assert (
        _rebuild_code(
            corpus,
            no_cluster=no_cluster,
            no_viz=True,
            acquire_lock=False,
        )
        is False
    )

    tree_after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before


def test_watch_incremental_manifest_does_not_advance_unrebuilt_source(tmp_path, monkeypatch):
    import importlib

    from graphify.detect import detect_incremental
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first = corpus / "first.py"
    second = corpus / "second.py"
    first.write_text("def first():\n    return 1\n", encoding="utf-8")
    second.write_text("def second():\n    return 2\n", encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False)

    first.write_text("def first():\n    return 10\n", encoding="utf-8")
    extract_module = importlib.import_module("graphify.extract")
    real_extract = extract_module.extract

    def extract_while_other_source_changes(paths, **kwargs):
        result = real_extract(paths, **kwargs)
        second.write_text("def second():\n    return 20\n", encoding="utf-8")
        return result

    monkeypatch.setattr(extract_module, "extract", extract_while_other_source_changes)

    assert _rebuild_code(
        corpus,
        changed_paths=[first],
        no_cluster=True,
        acquire_lock=False,
    )

    incremental = detect_incremental(
        corpus,
        manifest_path=str(corpus / "graphify-out" / "manifest.json"),
        kind="ast",
    )
    changed = {Path(path).name for path in incremental["new_files"]["code"]}
    assert "second.py" in changed
    assert "first.py" not in changed


@pytest.mark.parametrize("no_cluster", [True, False])
def test_watch_refuses_manifest_for_source_changed_after_extraction(
    tmp_path, monkeypatch, no_cluster
):
    import importlib

    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "app.py"
    source.write_text("def before():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )

    output = corpus / "graphify-out"
    source.write_text("def after():\n    return 2\n", encoding="utf-8")
    tree_before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    extract_module = importlib.import_module("graphify.extract")
    real_extract = extract_module.extract

    def extract_then_mutate(paths, **kwargs):
        result = real_extract(paths, **kwargs)
        source.write_text("def changed_after_parse():\n    return 3\n", encoding="utf-8")
        return result

    monkeypatch.setattr(extract_module, "extract", extract_then_mutate)

    assert (
        _rebuild_code(
            corpus,
            changed_paths=[source],
            no_cluster=no_cluster,
            no_viz=True,
            acquire_lock=False,
        )
        is False
    )

    tree_after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before


def test_watch_corrupt_existing_graph_is_refused_without_mutation(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    output = corpus / "graphify-out"
    output.mkdir()
    graph = output / "graph.json"
    original = b"{not-json"
    graph.write_bytes(original)

    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is False

    assert graph.read_bytes() == original
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()
    assert list((output / "cache").rglob("*.json")) == []


def test_watch_full_rebuild_prunes_stale_semantic_sources_and_hyperedges(tmp_path):
    """A full AST rebuild keeps valid semantic data but removes records whose
    local source disappeared and hyperedges that no longer have two members."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def live():\n    return 1\n", encoding="utf-8")
    docs = corpus / "docs"
    docs.mkdir()
    (docs / "live.md").write_text("# Live semantic source\n", encoding="utf-8")

    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    live_ast = next(
        node["id"] for node in data["nodes"] if node.get("label", "").startswith("live(")
    )
    data["nodes"].extend(
        [
            {
                "id": "semantic_live",
                "label": "Live semantic node",
                "file_type": "concept",
                "source_file": "docs/live.md",
                "_origin": "semantic",
            },
            {
                "id": "semantic_legacy",
                "name": "Legacy semantic node",
                "file_type": "concept",
                "source_file": "docs/live.md",
                "_origin": "semantic",
            },
            {
                "id": "semantic_stale",
                "name": "Stale semantic node",
                "file_type": "concept",
                "source_file": "docs/deleted.md",
                "_origin": "semantic",
            },
        ]
    )
    links = data.get("links", data.get("edges", []))
    links.extend(
        [
            {
                "source": "semantic_live",
                "target": live_ast,
                "relation": "documents",
                "source_file": "docs/live.md",
                "_origin": "semantic",
            },
            {
                "source": "semantic_stale",
                "target": live_ast,
                "relation": "documents",
                "source_file": "docs/deleted.md",
                "_origin": "semantic",
            },
        ]
    )
    data["links"] = links
    data["hyperedges"] = [
        {
            "id": "live_group",
            "nodes": ["semantic_live", live_ast],
            "source_file": "docs/live.md",
        },
        {
            "id": "stale_group",
            "nodes": ["semantic_stale", live_ast],
            "source_file": "docs/deleted.md",
        },
        {
            "id": "dangling_group",
            "nodes": ["semantic_live", "semantic_stale"],
            "source_file": "docs/live.md",
        },
    ]
    _promote_to_multidigraph(data)
    graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))

    rebuilt_ids = {node["id"] for node in rebuilt["nodes"]}
    assert "semantic_live" in rebuilt_ids
    legacy = next(node for node in rebuilt["nodes"] if node["id"] == "semantic_legacy")
    assert legacy["label"] == "Legacy semantic node"
    assert "semantic_stale" not in rebuilt_ids
    rebuilt_links = rebuilt.get("links", rebuilt.get("edges", []))
    assert any(edge.get("source") == "semantic_live" for edge in rebuilt_links)
    assert not any(edge.get("source") == "semantic_stale" for edge in rebuilt_links)
    assert [edge.get("id") for edge in rebuilt["hyperedges"]] == ["live_group"]


@pytest.mark.parametrize("no_cluster", [False, True])
def test_full_rebuild_prunes_sources_newly_excluded_by_graphifyignore(tmp_path, no_cluster):
    """A full update must evict previously indexed records once their source is ignored."""
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    private_dir = corpus / "private"
    private_dir.mkdir()
    private_doc = private_dir / "notes.md"
    private_doc.write_text("# Private notes\n", encoding="utf-8")
    (private_dir / "manifest-only.txt").write_text("ignored later\n", encoding="utf-8")

    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    live_ids = [
        node["id"] for node in data["nodes"] if node.get("label") in {"first()", "second()"}
    ]
    assert len(live_ids) == 2

    private_source = "private/notes.md"
    data["nodes"].append(
        {
            "id": "private_semantic_node",
            "label": "Private semantic node",
            "file_type": "concept",
            "source_file": private_source,
            "_origin": "semantic",
        }
    )
    links = data.get("links", data.get("edges", []))
    links.append(
        {
            "source": live_ids[0],
            "target": live_ids[1],
            "relation": "documents",
            "confidence": "INFERRED",
            "source_file": private_source,
            "_origin": "semantic",
        }
    )
    data["links"] = links
    data["hyperedges"] = [
        {
            "id": "private_group",
            "nodes": live_ids,
            "source_file": private_source,
        }
    ]
    graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (corpus / ".graphifyignore").write_text("/private/\n", encoding="utf-8")

    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    cleaned = json.loads(graph_path.read_text(encoding="utf-8"))
    cleaned_links = cleaned.get("links", cleaned.get("edges", []))
    manifest = json.loads((corpus / "graphify-out" / "manifest.json").read_text(encoding="utf-8"))
    assert all(node.get("source_file") != private_source for node in cleaned["nodes"])
    assert all(edge.get("source_file") != private_source for edge in cleaned_links)
    assert all(
        hyperedge.get("source_file") != private_source
        for hyperedge in cleaned.get("hyperedges", [])
    )
    assert private_source not in manifest
    assert "private/manifest-only.txt" not in manifest

    stable_bytes = graph_path.read_bytes()
    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    assert graph_path.read_bytes() == stable_bytes


def test_incremental_non_code_change_preserves_semantic_records_and_signals(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    notes = corpus / "notes.txt"
    notes.write_text("Notes\n", encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, no_viz=True, acquire_lock=False)

    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    data["nodes"].append(
        {
            "id": "notes_semantic",
            "label": "Notes semantic",
            "file_type": "concept",
            "source_file": "notes.txt",
            "_origin": "semantic",
        }
    )
    graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    notes.write_text("# Notes changed\n", encoding="utf-8")
    assert _rebuild_code(
        corpus,
        changed_paths=[notes],
        no_cluster=True,
        no_viz=True,
        acquire_lock=False,
    )

    preserved = json.loads(graph_path.read_text(encoding="utf-8"))
    assert any(node.get("id") == "notes_semantic" for node in preserved["nodes"])
    assert (corpus / "graphify-out" / "needs_update").exists()


def test_incremental_deleted_extensionless_source_is_evicted(tmp_path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    script = corpus / "run-tool"
    script.write_text("#!/usr/bin/env python3\nprint('run')\n", encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, no_viz=True, acquire_lock=False)
    graph_path = corpus / "graphify-out" / "graph.json"
    before = json.loads(graph_path.read_text(encoding="utf-8"))
    before["nodes"].append(
        {
            "id": "extensionless_semantic",
            "label": "Extensionless semantic",
            "file_type": "concept",
            "source_file": "run-tool",
            "_origin": "semantic",
        }
    )
    graph_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
    assert any(node.get("source_file") == "run-tool" for node in before["nodes"])

    script.unlink()
    assert _rebuild_code(
        corpus,
        changed_paths=[script],
        no_cluster=True,
        no_viz=True,
        acquire_lock=False,
    )
    after = json.loads(graph_path.read_text(encoding="utf-8"))
    assert all(node.get("source_file") != "run-tool" for node in after["nodes"])


def test_rebuild_git_provenance_uses_scan_root_from_unrelated_cwd(tmp_path, monkeypatch):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    _git_init(corpus)
    (corpus / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(corpus), "add", "--", "app.py"], check=True)
    subprocess.run(["git", "-C", str(corpus), "commit", "-qm", "corpus"], check=True)
    corpus_commit = subprocess.run(
        ["git", "-C", str(corpus), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()

    caller = tmp_path / "caller"
    _git_init(caller)
    (caller / "caller.txt").write_text("caller\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(caller), "add", "--", "caller.txt"], check=True)
    subprocess.run(["git", "-C", str(caller), "commit", "-qm", "caller"], check=True)
    monkeypatch.chdir(caller)

    output_root = tmp_path / "external"
    output = output_root / "graphify-out"
    output.mkdir(parents=True)
    for _ in range(3):
        assert _rebuild_code(
            corpus,
            output_dir=output,
            output_root=output_root,
            no_cluster=True,
            no_viz=True,
            acquire_lock=False,
        )
        payload = json.loads((output / "graph.json").read_text(encoding="utf-8"))
        assert payload["built_at_commit"] == corpus_commit


@pytest.mark.parametrize("no_cluster", [False, True])
def test_full_rebuild_preserves_foreign_absolute_semantic_sources(tmp_path, no_cluster):
    from graphify.paths import is_absolute_path
    from graphify.watch import _rebuild_code

    foreign_source = (
        "/foreign-posix/repo/context.py"
        if os.name == "nt"
        else r"C:\foreign-windows\repo\context.py"
    )
    assert is_absolute_path(foreign_source)
    assert not Path(foreign_source).is_absolute()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )

    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    live_ids = [
        node["id"] for node in data["nodes"] if node.get("label") in {"first()", "second()"}
    ]
    assert len(live_ids) == 2
    data["nodes"].append(
        {
            "id": "foreign_semantic_node",
            "label": "Foreign semantic node",
            "file_type": "concept",
            "source_file": foreign_source,
            "_origin": "semantic",
        }
    )
    links = data.get("links", data.get("edges", []))
    links.append(
        {
            "source": live_ids[0],
            "target": live_ids[1],
            "relation": "documents",
            "confidence": "INFERRED",
            "source_file": foreign_source,
            "_origin": "semantic",
        }
    )
    data["links"] = links
    data["hyperedges"] = [
        {
            "id": "foreign_group",
            "nodes": live_ids,
            "source_file": foreign_source,
        }
    ]
    graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    assert _rebuild_code(
        corpus,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
    rebuilt_links = rebuilt.get("links", rebuilt.get("edges", []))
    canonical_source = foreign_source.replace("\\", "/")
    assert any(
        str(node.get("source_file", "")).replace("\\", "/") == canonical_source
        for node in rebuilt["nodes"]
    )
    assert any(
        str(edge.get("source_file", "")).replace("\\", "/") == canonical_source
        for edge in rebuilt_links
    )
    assert any(
        str(hyperedge.get("source_file", "")).replace("\\", "/") == canonical_source
        for hyperedge in rebuilt.get("hyperedges", [])
    )


@pytest.mark.parametrize("no_cluster", [False, True])
def test_full_rebuild_preserves_integer_hyperedge_members(tmp_path, no_cluster):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def first():\n    return 1\n\ndef second():\n    return first()\n",
        encoding="utf-8",
    )
    assert _rebuild_code(corpus, no_cluster=no_cluster, no_viz=True, acquire_lock=False)
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    live_id = next(node["id"] for node in data["nodes"] if node.get("label") == "first()")
    data["nodes"].append(
        {
            "id": 7,
            "label": "external seven",
            "file_type": "concept",
            "source_file": "https://example.invalid/context",
            "_origin": "semantic",
        }
    )
    data["hyperedges"] = [
        {
            "id": "mixed-members",
            "nodes": [7, live_id],
            "source_file": "https://example.invalid/context",
        }
    ]
    graph_path.write_text(json.dumps(data), encoding="utf-8")

    for _ in range(2):
        assert _rebuild_code(corpus, no_cluster=no_cluster, no_viz=True, acquire_lock=False)
        rebuilt = json.loads(graph_path.read_text(encoding="utf-8"))
        mixed = next(item for item in rebuilt["hyperedges"] if item["id"] == "mixed-members")
        assert 7 in mixed["nodes"]


def test_watch_refuses_current_state_with_dangling_hyperedge_member(tmp_path, capsys):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "app.py").write_text(
        "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    docs = corpus / "docs"
    docs.mkdir()
    (docs / "design.md").write_text("# Design\n", encoding="utf-8")

    assert _rebuild_code(corpus, no_viz=True, acquire_lock=False)
    graph_path = corpus / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    run_node = next(
        node["id"] for node in data["nodes"] if node.get("label", "").startswith("run(")
    )
    stale = {
        "id": "app_group",
        "label": "App group",
        "nodes": ["app_py_node", run_node],
        "source_file": "docs/design.md",
    }
    data["hyperedges"] = [stale]
    data.setdefault("graph", {})["hyperedges"] = [dict(stale)]
    data["graph"]["custom_meta"] = "kept"
    graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    before = graph_path.read_bytes()

    assert not _rebuild_code(corpus, no_viz=True, acquire_lock=False)
    assert graph_path.read_bytes() == before
    assert "dangling member" in capsys.readouterr().err


# --- RISK 3: no-cluster compare must not flap on a legacy edges-keyed graph.json ---


def _downgrade_to_legacy_edges(graph_path: Path) -> None:
    """Rewrite ``graph_path`` in the pre-modern on-disk shape that triggered the
    no-cluster flap: the edge list keyed as ``edges`` (not ``links``), a
    ``confidence_score`` stamped on every edge (a recomputed/volatile field), and
    the top-level ``hyperedges`` key dropped entirely (null-vs-[] history).

    All three deviations are required to reproduce the full bug: a fixture that
    only renames ``links``->``edges`` (without injecting ``confidence_score`` and
    without dropping ``hyperedges``) would falsely pass once the key fold lands,
    masking the volatile-field and missing-hyperedges legs of the same flap.
    """
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    links = data.pop("links", data.pop("edges", []))
    data["edges"] = [{**edge, "confidence_score": 0.9} for edge in links]
    data.pop("hyperedges", None)
    data.pop("schema_version", None)
    data.pop("graphify_state_diagnostics", None)
    graph_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_rebuild_code_no_cluster_does_not_flap_on_legacy_edges_key(tmp_path):
    """RISK 3: a no-op ``--no-cluster`` rebuild over a graph.json written in the
    legacy ``edges``-keyed shape must detect "no change" and leave graph.json
    byte-for-byte untouched (no flap).

    The legacy downgrade renames ``links``->``edges``, stamps a volatile
    ``confidence_score`` on each edge, and drops the top-level ``hyperedges``
    key. ``_canonical_graph_for_compare`` must fold all three back so the
    on-disk legacy graph compares EQUAL to the freshly-extracted candidate;
    otherwise every watcher tick rewrites graph.json forever.
    """
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    _git_init(repo)
    (repo / "app.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return alpha()\n",
        encoding="utf-8",
    )

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        # Real build to get an authentic no-cluster graph.json for this corpus.
        assert _rebuild_code(repo, no_cluster=True, acquire_lock=False) is True
        graph_path = repo / "graphify-out" / "graph.json"
        assert graph_path.exists()

        # The idempotence requirement is >=3 consecutive no-op rebuilds.  We
        # re-apply the legacy downgrade IMMEDIATELY before each measured run
        # because a buggy _rebuild_code rewrites edges->links on the first
        # flap; without re-downgrading, subsequent runs would compare links vs
        # links and falsely pass (masking the regression).
        #
        # Each run captures its own before-state (mtime + bytes) AFTER the
        # downgrade but BEFORE the sleep+rebuild, then asserts the rebuild
        # leaves the file untouched.  Comparing within each run (not across
        # runs) is correct because _downgrade_to_legacy_edges itself writes the
        # file and therefore changes its mtime — only the rebuild must be a
        # no-op.
        for run_idx in range(3):
            _downgrade_to_legacy_edges(graph_path)
            pre_bytes = graph_path.read_bytes()
            pre_mtime = graph_path.stat().st_mtime_ns

            # No source change — a correct compare must short-circuit to "no change".
            time.sleep(0.01)  # ensure any rewrite would move mtime measurably
            assert _rebuild_code(repo, no_cluster=True, acquire_lock=False) is True

            post_bytes = graph_path.read_bytes()
            post_mtime = graph_path.stat().st_mtime_ns

            assert post_bytes == pre_bytes, (
                f"run {run_idx + 1}: legacy edges-keyed graph.json was rewritten "
                "on a no-op no-cluster rebuild — the canonical compare flapped "
                "(edges->links / confidence_score / missing-hyperedges not folded)"
            )
            assert post_mtime == pre_mtime, (
                f"run {run_idx + 1}: graph.json mtime changed on a no-op rebuild (flap)"
            )
    finally:
        os.chdir(cwd)


# --- RISK 4 Guard 2: a failed/aborted extraction must not wipe a populated graph ---


def test_watch_no_cluster_delete_all_preserves_graph(tmp_path, monkeypatch):
    """RISK 4: when a declared-deletion rebuild ends with 0 nodes because the
    remaining files' extraction aborted (a failed/half-written extraction, not a
    real empty result), the no-cluster raw-write site must REFUSE to overwrite a
    populated graph.json and preserve the previous graph.

    Reproduction: a two-file corpus is built, ``a.py`` is deleted (declared via
    ``changed_paths`` so ``had_explicit_deletions`` is True and the existing
    shrink guard is bypassed), ``b.py`` stays on disk (so the "no code files"
    early return does NOT fire), and ``extract`` is stubbed to return nothing
    (the aborted extraction). Without the 0-floor the graph is wiped to 0 nodes.
    """
    import graphify.extract as extract_mod
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    _git_init(repo)
    (repo / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        assert _rebuild_code(repo, no_cluster=True, acquire_lock=False) is True
        graph_path = repo / "graphify-out" / "graph.json"
        before = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes_before = len(before.get("nodes", []))
        assert nodes_before > 0
        before_bytes = graph_path.read_bytes()
        marker = graph_path.parent / ".graphify_root"
        marker.write_text(str(repo), encoding="utf-8")
        marker_before = marker.read_bytes()

        # Delete a.py (declared deletion -> had_explicit_deletions=True). Keep
        # b.py on disk so detect() still returns code files, but make extraction
        # abort to empty so the merged candidate has 0 nodes.
        (repo / "a.py").unlink()

        def aborted_extract(_targets, **_kwargs):
            return {
                "nodes": [],
                "edges": [],
                "hyperedges": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        monkeypatch.setattr(extract_mod, "extract", aborted_extract)
        result = _rebuild_code(
            repo,
            changed_paths=[Path("a.py"), Path("b.py")],
            no_cluster=True,
            acquire_lock=False,
        )

        after = json.loads(graph_path.read_text(encoding="utf-8"))
        after_bytes = graph_path.read_bytes()
    finally:
        os.chdir(cwd)

    assert result is False, "rebuild must refuse the empty overwrite"
    assert len(after.get("nodes", [])) == nodes_before, (
        "populated graph.json must be preserved when a failed extraction yields 0 nodes"
    )
    assert after_bytes == before_bytes, "graph.json must be byte-for-byte untouched"
    assert marker.read_bytes() == marker_before, "refused rebuild must not rewrite marker state"


def test_watch_clustered_delete_all_preserves_graph(tmp_path, monkeypatch):
    """RISK 4: the clustered ``tmp.replace`` write site must likewise refuse to
    overwrite a populated graph.json with an empty (0-node) graph produced by a
    failed/aborted extraction during a declared-deletion rebuild.

    Same reproduction as the no-cluster sibling, exercising the clustered path
    (the ``graph_tmp.replace(existing_graph)`` write guarded by ``_check_shrink``).
    """
    import graphify.extract as extract_mod
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    _git_init(repo)
    (repo / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        # no_viz keeps the clustered path fast (skips graph.html generation).
        assert _rebuild_code(repo, no_viz=True, acquire_lock=False) is True
        graph_path = repo / "graphify-out" / "graph.json"
        before = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes_before = len(before.get("nodes", []))
        assert nodes_before > 0
        before_bytes = graph_path.read_bytes()
        marker = graph_path.parent / ".graphify_root"
        marker.write_text(str(repo), encoding="utf-8")
        marker_before = marker.read_bytes()

        (repo / "a.py").unlink()

        def aborted_extract(_targets, **_kwargs):
            return {
                "nodes": [],
                "edges": [],
                "hyperedges": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        monkeypatch.setattr(extract_mod, "extract", aborted_extract)
        result = _rebuild_code(
            repo,
            changed_paths=[Path("a.py"), Path("b.py")],
            no_viz=True,
            acquire_lock=False,
        )

        after = json.loads(graph_path.read_text(encoding="utf-8"))
        after_bytes = graph_path.read_bytes()
    finally:
        os.chdir(cwd)

    assert result is False, "clustered rebuild must refuse the empty overwrite"
    assert len(after.get("nodes", [])) == nodes_before, (
        "populated graph.json must be preserved when a failed extraction yields 0 nodes "
        "(clustered path)"
    )
    assert after_bytes == before_bytes, (
        "graph.json must be byte-for-byte untouched (clustered path)"
    )
    assert marker.read_bytes() == marker_before, "refused rebuild must not rewrite marker state"


@pytest.mark.parametrize("no_cluster", [False, True])
def test_ast_only_rebuild_does_not_acknowledge_semantic_pending_flag(tmp_path, no_cluster):
    """Only a successful semantic extraction may clear ``needs_update``."""
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    assert _rebuild_code(
        repo,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )
    flag = repo / "graphify-out" / "needs_update"
    flag.write_text("pending-semantic-work", encoding="utf-8")

    assert _rebuild_code(
        repo,
        no_cluster=no_cluster,
        no_viz=True,
        acquire_lock=False,
    )

    assert flag.read_text(encoding="utf-8") == "pending-semantic-work"


def test_ast_file_failure_refuses_watch_rebuild_without_state_mutation(tmp_path, monkeypatch):
    from graphify import extract as extract_mod
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    source = repo / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(
        extract_mod,
        "extract",
        lambda paths, **kwargs: {
            "nodes": [],
            "edges": [],
            "failed_files": [str(paths[0])],
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )

    assert (
        _rebuild_code(
            repo,
            no_cluster=True,
            no_viz=True,
            acquire_lock=False,
        )
        is False
    )
    output = repo / "graphify-out"
    assert not (output / "graph.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / ".graphify_root").exists()


def test_full_rebuild_snapshots_every_manifest_source(tmp_path):
    from graphify.watch import _rebuild_code

    repo = tmp_path / "corpus"
    repo.mkdir()
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    document = repo / "notes.txt"
    document.write_text("semantic-only source\n", encoding="utf-8")

    assert _rebuild_code(
        repo,
        no_cluster=True,
        no_viz=True,
        acquire_lock=False,
    )

    manifest = json.loads((repo / "graphify-out" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"app.py", "notes.txt"}


@pytest.mark.parametrize("no_cluster", [False, True])
def test_full_ast_update_losslessly_migrates_legacy_identity_references(
    tmp_path, capsys, no_cluster
):
    from graphify.ids import CURRENT_AST_NODE_ID_SCHEMA, make_id
    from graphify.semantic_schema import LEGACY_SEMANTIC_NODE_ID_SCHEMA
    from graphify.watch import _rebuild_code

    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Semantic package context\n", encoding="utf-8")
    assert _rebuild_code(
        tmp_path,
        graph_type="multidigraph",
        no_cluster=True,
        acquire_lock=False,
    )

    graph_path = tmp_path / "graphify-out" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    current_id = make_id("pkg", "python", "demo")
    legacy_id = "pkg_demo"
    assert legacy_id != current_id
    for node in graph["nodes"]:
        if node["id"] == current_id:
            node["id"] = legacy_id
    for edge in graph["links"]:
        if edge["source"] == current_id:
            edge["source"] = legacy_id
        if edge["target"] == current_id:
            edge["target"] = legacy_id
    graph["nodes"].append(
        {
            "id": "semantic_owner",
            "label": "Semantic owner",
            "file_type": "concept",
            "source_file": "notes.txt",
            "_origin": "semantic",
        }
    )
    graph["links"].append(
        {
            "source": "semantic_owner",
            "target": legacy_id,
            "relation": "references",
            "confidence": "INFERRED",
            "confidence_score": 0.85,
            "source_file": "notes.txt",
            "_origin": "semantic",
            "key": "semantic-package-reference",
        }
    )
    graph["hyperedges"] = [
        {
            "id": "semantic-package-group",
            "nodes": ["semantic_owner", legacy_id],
            "source_file": "notes.txt",
        }
    ]
    graph.pop("schema_version")
    graph["graph"]["graphify_profile"].pop("ast_node_id_schema")
    graph["graph"]["graphify_profile"].pop("semantic_node_id_schema")
    graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    output_dir = graph_path.parent
    before = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    manifest.write_text(
        '[project]\nname = "demo"\nversion = "2.0"\n',
        encoding="utf-8",
    )
    assert not _rebuild_code(
        tmp_path,
        changed_paths=[manifest],
        no_cluster=True,
        acquire_lock=False,
    )
    assert "ordinary `graphify update`" in capsys.readouterr().err
    after_refusal = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after_refusal == before

    stable_bytes = None
    for iteration in range(3):
        assert _rebuild_code(
            tmp_path,
            no_cluster=no_cluster,
            no_viz=True,
            acquire_lock=False,
        )
        updated = json.loads(graph_path.read_text(encoding="utf-8"))
        profile = updated["graph"]["graphify_profile"]
        assert profile["ast_node_id_schema"] == CURRENT_AST_NODE_ID_SCHEMA
        assert profile["semantic_node_id_schema"] == LEGACY_SEMANTIC_NODE_ID_SCHEMA
        package_nodes = [node for node in updated["nodes"] if node.get("type") == "package"]
        assert [(node["id"], node.get("version")) for node in package_nodes] == [
            (current_id, "2.0")
        ]
        assert legacy_id not in {node["id"] for node in updated["nodes"]}
        semantic_edge = next(
            edge for edge in updated["links"] if edge.get("key") == "semantic-package-reference"
        )
        assert semantic_edge["source"] == "semantic_owner"
        assert semantic_edge["target"] == current_id
        assert len(updated["hyperedges"]) == 1
        semantic_group = updated["hyperedges"][0]
        assert semantic_group["id"] == "semantic-package-group"
        assert set(semantic_group["nodes"]) == {"semantic_owner", current_id}
        assert semantic_group["source_file"] == "notes.txt"
        current_bytes = graph_path.read_bytes()
        if iteration == 0:
            continue
        if stable_bytes is not None:
            assert current_bytes == stable_bytes
        stable_bytes = current_bytes


def test_legacy_ast_reference_migration_refuses_ambiguous_matches_without_mutation(tmp_path):
    from copy import deepcopy

    from graphify.watch import _legacy_ast_reference_remap

    old_nodes = [
        {
            "id": "legacy_item",
            "label": "item()",
            "type": "function",
            "file_type": "code",
            "source_file": "app.py",
            "source_location": "L1",
            "_origin": "ast",
        }
    ]
    fresh_nodes = [
        {
            **old_nodes[0],
            "id": "current_item_a",
            "source_location": "L2",
        },
        {
            **old_nodes[0],
            "id": "current_item_b",
            "source_location": "L3",
        },
    ]
    edge = {
        "source": "semantic_owner",
        "target": "legacy_item",
        "relation": "references",
        "_origin": "semantic",
    }

    unique_edge = deepcopy(edge)
    assert _legacy_ast_reference_remap(
        old_nodes,
        fresh_nodes[:1],
        [unique_edge],
        [],
        root=tmp_path,
    ) == {"legacy_item": "current_item_a"}
    assert unique_edge["target"] == "current_item_a"

    ambiguous_edge = deepcopy(edge)
    before = deepcopy(ambiguous_edge)
    with pytest.raises(ValueError, match="could not be mapped uniquely"):
        _legacy_ast_reference_remap(
            old_nodes,
            fresh_nodes,
            [ambiguous_edge],
            [],
            root=tmp_path,
        )
    assert ambiguous_edge == before


def test_raw_edge_filter_drops_only_fresh_ast_dangling_edges():
    from graphify.watch import _filter_raw_dangling_edges

    nodes = [{"id": "a"}, {"id": "b"}]
    live = {"source": "a", "target": "b", "relation": "calls", "_origin": "ast"}
    external = {
        "source": "a",
        "target": "stdlib_external",
        "relation": "calls",
        "_origin": "ast",
    }

    assert _filter_raw_dangling_edges(nodes, [external, live]) == ([live], 1)

    carried = {**external, "_origin": "semantic"}
    with pytest.raises(ValueError, match="preserved non-AST edge"):
        _filter_raw_dangling_edges(nodes, [carried])


def test_update_cli_passes_no_viz_to_rebuild(tmp_path, monkeypatch):
    """`graphify update --no-viz <path>` must parse the flag and forward
    no_viz=True to _rebuild_code. Regression: the flag was dropped from the
    update CLI arm even though watch._rebuild_code still accepts it."""
    import sys as _sys
    import graphify.watch as _watch
    import graphify.__main__ as _main
    from graphify.ids import CURRENT_AST_NODE_ID_SCHEMA
    from graphify.semantic_schema import PROMPT_SCHEMA_VERSION

    (tmp_path / "x.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    output.mkdir()
    (output / "graph.json").write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {
                    "graphify_profile": {
                        "graph_type": "simple",
                        "ast_node_id_schema": CURRENT_AST_NODE_ID_SCHEMA,
                        "semantic_node_id_schema": PROMPT_SCHEMA_VERSION,
                    }
                },
                "nodes": [],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    (output / ".graphify_root").write_text("marker-relative:..", encoding="utf-8")
    captured: dict = {}

    def _fake_rebuild(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return True

    monkeypatch.setattr(_watch, "_rebuild_code", _fake_rebuild)
    monkeypatch.setattr(_sys, "argv", ["graphify", "update", "--no-viz", str(tmp_path)])
    try:
        _main.main()
    except SystemExit as exc:  # must NOT be the unknown-option exit(2)
        assert exc.code in (0, None), f"update --no-viz errored: exit {exc.code}"
    assert captured.get("no_viz") is True


def test_watch_cli_passes_resolved_external_output_context(tmp_path, monkeypatch):
    import sys as _sys
    import graphify.watch as _watch_module
    import graphify.__main__ as _main
    from graphify.ids import CURRENT_AST_NODE_ID_SCHEMA
    from graphify.semantic_schema import PROMPT_SCHEMA_VERSION

    source_root = tmp_path / "Sources"
    source_root.mkdir()
    output_root = tmp_path / "canonical"
    output = output_root / "graphify-out"
    output.mkdir(parents=True)
    (output / "graph.json").write_text(
        json.dumps(
            {
                "directed": True,
                "multigraph": True,
                "graph": {
                    "graphify_profile": {
                        "graph_type": "multidigraph",
                        "ast_node_id_schema": CURRENT_AST_NODE_ID_SCHEMA,
                        "semantic_node_id_schema": PROMPT_SCHEMA_VERSION,
                    }
                },
                "nodes": [],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    relative = os.path.relpath(source_root, output).replace(os.sep, "/")
    (output / ".graphify_root").write_text(f"marker-relative:{relative}", encoding="utf-8")
    captured: dict = {}

    def _fake_watch(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)

    monkeypatch.setattr(_watch_module, "watch", _fake_watch)
    monkeypatch.setattr(
        _sys,
        "argv",
        ["graphify", "watch", str(source_root), "--out", str(output_root)],
    )

    _main.main()

    assert captured["path"] == source_root.resolve()
    assert captured["output_dir"] == output.resolve()
    assert captured["output_root"] == output_root.resolve()
    assert captured["graph_type"] == "multidigraph"
