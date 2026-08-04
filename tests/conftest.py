"""Shared test configuration."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path_factory, monkeypatch) -> Path:
    """Give each test an isolated home and platform config directories."""
    home = tmp_path_factory.mktemp("sandbox-home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture
def path_alias() -> Iterator[Callable[[Path, Path], Path]]:
    """Create a real symlink, or a Windows junction when links need privilege."""
    junctions: list[Path] = []

    def create(alias: Path, target: Path) -> Path:
        alias = Path(alias)
        target = Path(target).resolve()
        try:
            alias.symlink_to(target, target_is_directory=target.is_dir())
            return alias
        except (OSError, NotImplementedError):
            if os.name != "nt":
                raise

        junction = alias if target.is_dir() else alias.with_name(f"{alias.stem}-junction")
        junction.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(junction),
                str(target if target.is_dir() else target.parent),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        junctions.append(junction)
        return junction if target.is_dir() else junction / target.name

    yield create

    for junction in reversed(junctions):
        try:
            junction.rmdir()
        except OSError:
            pass
