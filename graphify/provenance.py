"""Corpus-bound provenance helpers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_GIT_OBJECT_ID = re.compile(r"[0-9a-fA-F]{40,64}")


def git_head(scan_root: str | Path) -> str | None:
    """Return the Git HEAD containing ``scan_root``, or ``None`` outside Git."""
    root = Path(scan_root).resolve()
    if not root.is_dir():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
            encoding="utf-8",
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    if completed.returncode != 0 or _GIT_OBJECT_ID.fullmatch(commit) is None:
        return None
    return commit.lower()
