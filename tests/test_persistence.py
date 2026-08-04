from __future__ import annotations

import io
import errno
import os
import sys
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


def test_file_state_transaction_keeps_committed_files(tmp_path):
    from graphify.persistence import FileStateTransaction, atomic_write_text

    target = tmp_path / "graph.json"
    with FileStateTransaction([target]) as transaction:
        atomic_write_text(target, "published")
        transaction.commit()

    assert target.read_text(encoding="utf-8") == "published"
