from __future__ import annotations

import io
import errno
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


def test_atomic_write_text_replaces_complete_payload(tmp_path):
    from graphify.persistence import atomic_write_text

    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_publish_graph_state_writes_only_completed_canonical_bytes(tmp_path):
    from graphify.graph_state import DecodeMode, decode_graph_state, encode_graph_state_bytes
    from graphify.persistence import publish_graph_state

    payload = {
        "schema_version": 1,
        "directed": True,
        "multigraph": False,
        "graph": {"graphify_profile": {"graph_type": "digraph"}},
        "nodes": [{"id": "b"}, {"id": "a"}],
        "links": [{"source": "a", "target": "b", "_origin": "ast"}],
        "hyperedges": [],
    }
    state = decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)
    target = tmp_path / "graph.json"

    publish_graph_state(target, state)

    assert target.read_bytes() == encode_graph_state_bytes(state)
    assert list(tmp_path.glob(".graph.json.*.tmp")) == []


def test_atomic_write_text_preserves_target_when_replace_fails(tmp_path, monkeypatch):
    from graphify import persistence

    target = tmp_path / "state.json"
    target.write_text("stable", encoding="utf-8")

    def fail_replace(source: str | os.PathLike, destination: str | os.PathLike) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        persistence.atomic_write_text(target, "partial")

    assert target.read_text(encoding="utf-8") == "stable"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_text_preserves_mode_without_fchmod(tmp_path, monkeypatch):
    from graphify import persistence

    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    monkeypatch.delattr(persistence.os, "fchmod", raising=False)

    persistence.atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o640


def test_atomic_write_text_new_file_uses_normal_creation_mode(tmp_path):
    from graphify.persistence import atomic_write_text

    ordinary = tmp_path / "ordinary.json"
    target = tmp_path / "state.json"
    if os.name == "nt":
        ordinary.write_text("ordinary", encoding="utf-8")
        atomic_write_text(target, "new")
    else:
        previous_umask = os.umask(0o027)
        try:
            ordinary.write_text("ordinary", encoding="utf-8")
            atomic_write_text(target, "new")
        finally:
            os.umask(previous_umask)

    assert target.stat().st_mode & 0o777 == ordinary.stat().st_mode & 0o777


def test_windows_blocking_lock_retries_past_msvcrt_retry_window(monkeypatch):
    from graphify import persistence

    attempts = 0

    def locking(_fd, _mode, _size):
        nonlocal attempts
        attempts += 1
        if attempts <= 11:
            raise OSError(errno.EACCES, "simulated lock contention")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(persistence.time, "sleep", lambda _seconds: None)

    class LockStream(io.BytesIO):
        def fileno(self):
            return 17

    assert persistence._acquire_windows_lock(LockStream(), blocking=True) is True
    assert attempts == 12


def test_windows_nonblocking_lock_reports_contention_once(monkeypatch):
    from graphify import persistence

    attempts = 0

    def locking(_fd, _mode, _size):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EACCES, "simulated lock contention")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        persistence.time,
        "sleep",
        lambda _seconds: pytest.fail("nonblocking locks must not retry"),
    )

    class LockStream(io.BytesIO):
        def fileno(self):
            return 17

    assert persistence._acquire_windows_lock(LockStream(), blocking=False) is False
    assert attempts == 1


def test_windows_blocking_lock_does_not_retry_permanent_errors(monkeypatch):
    from graphify import persistence

    def locking(_fd, _mode, _size):
        raise OSError(errno.EBADF, "simulated invalid descriptor")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        persistence.time,
        "sleep",
        lambda _seconds: pytest.fail("permanent lock errors must not be retried"),
    )

    class LockStream(io.BytesIO):
        def fileno(self):
            return 17

    with pytest.raises(OSError, match="invalid descriptor"):
        persistence._acquire_windows_lock(LockStream(), blocking=True)


def test_locked_file_reports_contention_and_recovers(tmp_path):
    from graphify.persistence import locked_file

    lock_path = tmp_path / "state.guard"
    with locked_file(lock_path, blocking=False) as first:
        assert first is True
        with locked_file(lock_path, blocking=False) as second:
            assert second is False

    with locked_file(lock_path, blocking=False) as acquired_again:
        assert acquired_again is True
    assert Path(lock_path).exists()


def test_pending_signal_acknowledges_only_the_captured_generation(tmp_path):
    from graphify.persistence import (
        acknowledge_pending_signal,
        capture_pending_signal,
        write_pending_signal,
    )

    signal = tmp_path / "needs_update"
    first = write_pending_signal(signal)
    assert capture_pending_signal(signal) == first

    second = write_pending_signal(signal)
    assert second != first
    assert acknowledge_pending_signal(signal, first) is False
    assert capture_pending_signal(signal) == second

    assert acknowledge_pending_signal(signal, second) is True
    assert not signal.exists()


def test_file_state_transaction_restores_existing_and_removes_new_files(tmp_path):
    from graphify.persistence import FileStateTransaction, atomic_write_text

    existing = tmp_path / "graph.json"
    created = tmp_path / "manifest.json"
    existing.write_text("before", encoding="utf-8")

    with pytest.raises(RuntimeError, match="publish failed"):
        with FileStateTransaction([existing, created]):
            atomic_write_text(existing, "after")
            atomic_write_text(created, "new")
            raise RuntimeError("publish failed")

    assert existing.read_text(encoding="utf-8") == "before"
    assert not created.exists()


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_file_state_transaction_attempts_every_restore_before_reporting_failure(
    tmp_path, monkeypatch, failed_index
):
    from graphify import persistence

    paths = [tmp_path / name for name in ("graph.json", "manifest.json", "marker")]
    for index, path in enumerate(paths):
        path.write_bytes(f"before-{index}".encode())
    transaction = persistence.FileStateTransaction(paths)
    for index, path in enumerate(paths):
        path.write_bytes(f"mutated-{index}".encode())

    failed_path = paths[failed_index]
    attempted: list[Path] = []
    real_atomic_write = persistence.atomic_write_bytes

    def fail_one_restore(path: Path, payload: bytes) -> None:
        target = Path(path)
        attempted.append(target)
        if target == failed_path:
            raise PermissionError(errno.EACCES, "simulated rollback failure", target)
        real_atomic_write(target, payload)

    monkeypatch.setattr(persistence, "atomic_write_bytes", fail_one_restore)

    with pytest.raises(OSError, match=failed_path.name):
        transaction.rollback()

    assert attempted == list(reversed(paths))
    for index, path in enumerate(paths):
        expected = f"mutated-{index}" if path == failed_path else f"before-{index}"
        assert path.read_text(encoding="utf-8") == expected


def test_file_state_transaction_reports_all_cleanup_failures_after_best_effort(
    tmp_path, monkeypatch
):
    from graphify import persistence

    graph = tmp_path / "graph.json"
    manifest = tmp_path / "manifest.json"
    created = tmp_path / "new-marker"
    graph.write_bytes(b"graph-before")
    manifest.write_bytes(b"manifest-before")
    transaction = persistence.FileStateTransaction([graph, manifest, created])
    graph.write_bytes(b"graph-mutated")
    manifest.write_bytes(b"manifest-mutated")
    created.write_bytes(b"created")

    real_atomic_write = persistence.atomic_write_bytes
    real_unlink = Path.unlink

    def fail_manifest_restore(path: Path, payload: bytes) -> None:
        target = Path(path)
        if target == manifest:
            raise PermissionError(errno.EACCES, "simulated write-back failure", target)
        real_atomic_write(target, payload)

    def fail_created_cleanup(path: Path, *args, **kwargs) -> None:
        if path == created:
            raise PermissionError(errno.EACCES, "simulated unlink failure", path)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(persistence, "atomic_write_bytes", fail_manifest_restore)
    monkeypatch.setattr(Path, "unlink", fail_created_cleanup)

    with pytest.raises(persistence.FileRollbackError) as raised:
        transaction.rollback()

    assert [path for path, _error in raised.value.failures] == [created, manifest]
    assert graph.read_bytes() == b"graph-before"
    assert manifest.read_bytes() == b"manifest-mutated"
    assert created.read_bytes() == b"created"


def test_file_state_transaction_keeps_committed_files(tmp_path):
    from graphify.persistence import FileStateTransaction, atomic_write_text

    target = tmp_path / "graph.json"
    with FileStateTransaction([target]) as transaction:
        atomic_write_text(target, "published")
        transaction.commit()

    assert target.read_text(encoding="utf-8") == "published"


def test_output_state_lock_is_reentrant_for_the_same_directory(tmp_path):
    from graphify.persistence import output_state_lock

    with output_state_lock(tmp_path) as outer:
        assert outer is not None
        with output_state_lock(tmp_path) as inner:
            assert inner is outer

    assert not (tmp_path / ".publication-generation").exists()


def test_file_state_transaction_refuses_stale_generation_rollback(tmp_path):
    from graphify.persistence import FileStateTransaction, atomic_write_text, output_state_lock

    target = tmp_path / "graph.json"
    target.write_text("baseline", encoding="utf-8")
    with output_state_lock(tmp_path) as first_lease:
        transaction = FileStateTransaction([target], lease=first_lease)
        atomic_write_text(target, "first-writer")

    with output_state_lock(tmp_path):
        atomic_write_text(target, "newer-writer")

    assert transaction.rollback() is False
    assert target.read_text(encoding="utf-8") == "newer-writer"


def test_file_state_transaction_rolls_back_under_current_generation(tmp_path):
    from graphify.persistence import FileStateTransaction, atomic_write_text, output_state_lock

    target = tmp_path / "graph.json"
    target.write_text("baseline", encoding="utf-8")
    with pytest.raises(RuntimeError, match="publish failed"):
        with output_state_lock(tmp_path) as lease:
            with FileStateTransaction([target], lease=lease):
                atomic_write_text(target, "partial")
                raise RuntimeError("publish failed")

    assert target.read_text(encoding="utf-8") == "baseline"


def test_output_state_lock_serializes_real_processes(tmp_path):
    events = tmp_path / "events.txt"
    release = tmp_path / "release-first"
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from graphify.persistence import locked_file, output_state_lock

        output = Path(sys.argv[1])
        events = Path(sys.argv[2])
        name = sys.argv[3]
        release_arg = sys.argv[4]
        probe_contention = sys.argv[5] == "probe"

        def write_event(value):
            with events.open("a", encoding="utf-8") as stream:
                stream.write(value + "\\n")
                stream.flush()

        if probe_contention:
            with locked_file(output / ".rebuild.guard", blocking=False) as acquired:
                write_event(f"{name}:probe:{'acquired' if acquired else 'contended'}")
            if acquired:
                raise SystemExit("probe unexpectedly acquired the held output lock")

        with output_state_lock(output):
            write_event(f"{name}:start")
            if release_arg != "-":
                release = Path(release_arg)
                deadline = time.monotonic() + 10
                while not release.exists():
                    if time.monotonic() >= deadline:
                        raise SystemExit("timed out waiting for parent release")
                    time.sleep(0.01)
            write_event(f"{name}:end")
        """
    )

    def wait_for_event(value: str, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if events.exists() and value in events.read_text(encoding="utf-8").splitlines():
                return
            return_code = process.poll()
            assert return_code is None, f"child exited {return_code} before {value!r}"
            time.sleep(0.01)
        pytest.fail(f"timed out waiting for child event {value!r}")

    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "out"),
            str(events),
            "first",
            str(release),
            "no-probe",
        ]
    )
    wait_for_event("first:start", first)

    second = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "out"),
            str(events),
            "second",
            "-",
            "probe",
        ]
    )
    try:
        wait_for_event("second:probe:contended", second)
        assert events.read_text(encoding="utf-8").splitlines() == [
            "first:start",
            "second:probe:contended",
        ]
    finally:
        release.touch()

    assert first.wait(timeout=10) == 0
    assert second.wait(timeout=10) == 0
    assert events.read_text(encoding="utf-8").splitlines() == [
        "first:start",
        "second:probe:contended",
        "first:end",
        "second:start",
        "second:end",
    ]


def test_output_state_lock_keeps_same_guard_file_for_future_waiters(tmp_path):
    from graphify.persistence import output_state_lock

    output = tmp_path / "new-output"
    with output_state_lock(output):
        guard = output / ".rebuild.guard"
        assert guard.exists()
        first_handle = guard.open("rb")

    try:
        assert guard.exists()
        with output_state_lock(output):
            with guard.open("rb") as second_handle:
                assert os.path.sameopenfile(first_handle.fileno(), second_handle.fileno())
    finally:
        first_handle.close()


def test_output_state_lock_is_released_when_holder_process_crashes(tmp_path):
    from graphify.persistence import output_state_lock

    output = tmp_path / "out"
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from graphify.persistence import output_state_lock

        with output_state_lock(Path(sys.argv[1])) as lease:
            assert lease is not None
            print("locked", flush=True)
            os._exit(17)
        """
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )

    assert crashed.returncode == 17
    assert crashed.stdout == "locked\n"
    assert (output / ".publication-generation").exists()
    with output_state_lock(output, blocking=False) as recovered:
        assert recovered is not None
    assert not (output / ".publication-generation").exists()
