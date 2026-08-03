"""Crash-safe state writes and small cross-process file locks."""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


def _sync_directory(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` only after a complete, synced same-directory write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous_mode: int | None = None
    try:
        previous_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        pass

    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    stream = None
    try:
        stream = open(temporary, "x+b")
        fd = stream.fileno()
        desired_mode = previous_mode
        if desired_mode is None:
            desired_mode = stat.S_IMODE(os.fstat(fd).st_mode)
        fchmod = getattr(os, "fchmod", None)
        if callable(fchmod):
            fchmod(fd, desired_mode)
        else:
            os.chmod(temporary, desired_mode)
        stream.write(payload)
        stream.flush()
        os.fsync(fd)
        stream.close()
        stream = None
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(Path(path), text.encode(encoding))


@dataclass(frozen=True)
class _FileSnapshot:
    existed: bool
    payload: bytes = b""
    mode: int | None = None


class FileStateTransaction:
    """Restore a small known set of state files if publication fails."""

    def __init__(self, paths: Iterable[Path] = ()) -> None:
        self._files: dict[Path, _FileSnapshot] = {}
        self._directories: dict[Path, bool] = {}
        self._committed = False
        for path in paths:
            self.capture(path)

    def capture(self, path: Path) -> None:
        target = Path(path)
        if target in self._files:
            return
        try:
            self._files[target] = _FileSnapshot(
                True,
                target.read_bytes(),
                stat.S_IMODE(target.stat().st_mode),
            )
        except FileNotFoundError:
            self._files[target] = _FileSnapshot(False)

    def capture_directory(self, path: Path) -> None:
        directory = Path(path)
        self._directories.setdefault(directory, directory.exists())

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> None:
        if self._committed:
            return
        for path, snapshot in reversed(list(self._files.items())):
            if snapshot.existed:
                atomic_write_bytes(path, snapshot.payload)
                if snapshot.mode is not None:
                    os.chmod(path, snapshot.mode)
            else:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
        for directory, existed in reversed(list(self._directories.items())):
            if not existed:
                with contextlib.suppress(OSError):
                    directory.rmdir()

    def __enter__(self) -> FileStateTransaction:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False


def _acquire_windows_lock(stream: BinaryIO, *, blocking: bool) -> bool:
    """Acquire one byte with true blocking semantics on Windows."""
    import msvcrt

    locking = getattr(msvcrt, "locking")
    nonblocking_mode = getattr(msvcrt, "LK_NBLCK")
    while True:
        stream.seek(0)
        try:
            locking(stream.fileno(), nonblocking_mode, 1)
        except OSError as exc:
            contention = (
                exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
                or getattr(exc, "winerror", None) == 33
            )
            if not contention:
                raise
            if not blocking:
                return False
            time.sleep(0.05)
            continue
        return True


@contextlib.contextmanager
def locked_file(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Yield whether an exclusive process lock was acquired.

    The lock file is persistent so every process synchronizes on the same file
    identity. Callers may maintain a separate transient status file when they
    need absence to signal completion.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(lock_path, "a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if not _acquire_windows_lock(stream, blocking=blocking):
                yield False
                return
        else:
            import fcntl

            flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(stream.fileno(), flags)
            except BlockingIOError:
                yield False
                return
        acquired = True
        yield True
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                with contextlib.suppress(OSError):
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with contextlib.suppress(OSError):
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _pending_signal_guard(path: Path) -> Path:
    signal = Path(path)
    return signal.with_name(f"{signal.name}.guard")


def capture_pending_signal(path: Path) -> bytes | None:
    """Read one pending-work generation while synchronized with its writers."""
    signal = Path(path)
    # An event created after this observation is a new, unacknowledged
    # generation. Avoid creating a persistent guard artifact when there is no
    # work to capture.
    if not signal.exists():
        return None
    with locked_file(_pending_signal_guard(signal)) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-signal lock")
        try:
            return signal.read_bytes()
        except FileNotFoundError:
            return None


def write_pending_signal(path: Path) -> bytes:
    """Atomically publish a unique pending-work generation."""
    signal = Path(path)
    payload = f"{time.time_ns()}:{os.getpid()}:{uuid.uuid4().hex}\n".encode("ascii")
    with locked_file(_pending_signal_guard(signal)) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-signal lock")
        atomic_write_bytes(signal, payload)
    return payload


def acknowledge_pending_signal(path: Path, expected: bytes | None) -> bool:
    """Remove a signal only when it still matches the captured generation."""
    if expected is None:
        return False
    signal = Path(path)
    with locked_file(_pending_signal_guard(signal)) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-signal lock")
        try:
            current = signal.read_bytes()
        except FileNotFoundError:
            return False
        if current != expected:
            return False
        signal.unlink()
        _sync_directory(signal.parent)
        return True
