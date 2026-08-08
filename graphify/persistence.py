"""Crash-safe state writes and small cross-process file locks."""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import time
import uuid
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from graphify.graph_state import GraphState


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


def publish_graph_state(path: Path, state: GraphState) -> None:
    """Publish one fully encoded graph state under its output-directory lease."""
    from graphify.graph_state import encode_graph_state_bytes

    payload = encode_graph_state_bytes(state)
    target = Path(path)
    with output_state_lock(target.parent) as lease:
        if lease is None:
            raise RuntimeError("could not acquire graph publication lock")
        atomic_write_bytes(target, payload)


@dataclass(frozen=True)
class _FileSnapshot:
    existed: bool
    payload: bytes = b""
    mode: int | None = None


@dataclass(frozen=True)
class PublicationLease:
    """One generation of exclusive authority over an output directory."""

    output_dir: Path
    generation_path: Path
    generation: bytes

    def is_current(self) -> bool:
        try:
            return self.generation_path.read_bytes() == self.generation
        except OSError:
            return False


class FileRollbackError(OSError):
    """Report every state path that could not be restored."""

    def __init__(self, failures: Iterable[tuple[Path, OSError]]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(f"{path}: {error}" for path, error in self.failures)
        super().__init__(
            errno.EIO, f"rollback incomplete for {len(self.failures)} path(s): {details}"
        )


_active_publication_lease: ContextVar[PublicationLease | None] = ContextVar(
    "graphify_publication_lease",
    default=None,
)


def holds_output_state_lock(output_dir: Path) -> bool:
    """Return whether this context owns the resolved output-directory lease."""
    active = _active_publication_lease.get()
    if active is None:
        return False
    output = Path(output_dir).expanduser().resolve(strict=False)
    return active.output_dir == output


class FileStateTransaction:
    """Restore a small known set of state files if publication fails."""

    def __init__(
        self,
        paths: Iterable[Path] = (),
        *,
        lease: PublicationLease | None = None,
    ) -> None:
        self._files: dict[Path, _FileSnapshot] = {}
        self._directories: dict[Path, bool] = {}
        self._committed = False
        self._lease = lease or _active_publication_lease.get()
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

    def rollback(self) -> bool:
        if self._committed:
            return True
        if self._lease is not None and not self._lease.is_current():
            return False
        failures: list[tuple[Path, OSError]] = []
        for path, snapshot in reversed(list(self._files.items())):
            try:
                if snapshot.existed:
                    atomic_write_bytes(path, snapshot.payload)
                    if snapshot.mode is not None:
                        os.chmod(path, snapshot.mode)
                else:
                    path.unlink()
            except FileNotFoundError as exc:
                if snapshot.existed:
                    failures.append((path, exc))
            except OSError as exc:
                failures.append((path, exc))
        for directory, existed in reversed(list(self._directories.items())):
            if not existed:
                try:
                    directory.rmdir()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    failures.append((directory, exc))
        if failures:
            raise FileRollbackError(failures)
        return True

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


@contextlib.contextmanager
def output_state_lock(
    output_dir: Path,
    *,
    blocking: bool = True,
    reentrant: bool = True,
) -> Iterator[PublicationLease | None]:
    """Hold one reentrant lock across a complete output-state lifecycle.

    Lock ordering is publication guard first, then any pending-signal or queue
    guard. Non-blocking hook writers may publish queue records before attempting
    this lock, but they release the queue guard before acquisition.
    """
    output = Path(output_dir).expanduser().resolve(strict=False)
    active = _active_publication_lease.get()
    if reentrant and active is not None and active.output_dir == output:
        yield active
        return

    with locked_file(output / ".rebuild.guard", blocking=blocking) as acquired:
        if not acquired:
            yield None
            return
        generation = f"{time.time_ns()}:{os.getpid()}:{uuid.uuid4().hex}\n".encode("ascii")
        generation_path = output / ".publication-generation"
        atomic_write_bytes(generation_path, generation)
        lease = PublicationLease(output, generation_path, generation)
        token = _active_publication_lease.set(lease)
        try:
            yield lease
        finally:
            _active_publication_lease.reset(token)
            try:
                if generation_path.read_bytes() == generation:
                    generation_path.unlink()
                    _sync_directory(output)
            except OSError:
                pass
            # Keep the guard inode stable for the lifetime of the output path.
            # Unlinking it while another process has the file open creates a
            # second lock domain when a third process recreates the pathname.


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
