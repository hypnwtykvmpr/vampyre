"""Vampyre contract guard.

Every assertion here is a Vampyre-specific guarantee that must survive future
maintenance. This is the tripwire for contract drift:
when we re-base on a newer graphify, this file fails fast if our own surface or
privacy posture regressed.
"""

import subprocess
import sys
from pathlib import Path

from graphify.build import build_from_json


def _graphify(*args: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "graphify", *args], capture_output=True, text=True, encoding="utf-8"
    )
    return r.stdout + r.stderr


def _nodes() -> list[dict]:
    return [
        {"id": "a", "label": "a", "file_type": "code", "source_file": "a.py"},
        {"id": "b", "label": "b", "file_type": "code", "source_file": "b.py"},
    ]


def test_multigraph_build_preserves_parallel_edges() -> None:
    """The headline feature: distinct relations between the same pair must not collapse."""
    edges = [
        {
            "source": "a",
            "target": "b",
            "relation": "calls",
            "confidence": "EXTRACTED",
            "source_file": "a.py",
        },
        {
            "source": "a",
            "target": "b",
            "relation": "imports",
            "confidence": "EXTRACTED",
            "source_file": "a.py",
        },
    ]
    G = build_from_json({"nodes": _nodes(), "edges": edges}, multigraph=True)
    assert G.is_multigraph(), "build_from_json(multigraph=True) must yield a multigraph"
    assert G.number_of_edges("a", "b") == 2, "parallel edges collapsed -> multigraph regressed"


def test_simple_build_still_collapses() -> None:
    """Non-multigraph path must remain a simple graph (no accidental always-multi)."""
    edges = [
        {
            "source": "a",
            "target": "b",
            "relation": "calls",
            "confidence": "EXTRACTED",
            "source_file": "a.py",
        },
        {
            "source": "a",
            "target": "b",
            "relation": "imports",
            "confidence": "EXTRACTED",
            "source_file": "a.py",
        },
    ]
    G = build_from_json({"nodes": _nodes(), "edges": edges}, multigraph=False)
    assert not G.is_multigraph()


def test_query_logging_is_opt_in_by_default(monkeypatch) -> None:
    """Vampyre privacy policy: query logging is off unless explicitly enabled."""
    from graphify import querylog

    monkeypatch.delenv("GRAPHIFY_QUERY_LOG", raising=False)
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)
    assert querylog._log_path() is None, "query logging must default OFF (opt-in) in Vampyre"


def test_cli_advertises_multigraph_surface() -> None:
    h = _graphify("--help")
    assert "--multigraph" in h, "Vampyre --multigraph flag missing from CLI help"
    assert "--simple" in h, "Vampyre --simple flag missing from CLI help"
    assert "merge-graphs" in h, "Vampyre merge-graphs command missing from CLI help"


def test_private_agent_files_not_tracked() -> None:
    """AGENTS.md/CLAUDE.md/GEMINI.md were scrubbed; a re-base must not re-track them."""
    root = Path(__file__).resolve().parent.parent
    assert (root / ".git").exists(), "repository contract tests require a git checkout"
    r = subprocess.run(
        ["git", "-C", str(root), "ls-files", "AGENTS.md", "CLAUDE.md", "GEMINI.md"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tracked = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert not tracked, f"private agent files were tracked: {tracked}"
