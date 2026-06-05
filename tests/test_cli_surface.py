"""CLI surface-parity guard.

Catches the class of regression where a rebuild/merge silently DROPS a CLI command
or flag. Real example this guards against: `graphify update --no-viz` was dropped
when graphify/__main__.py was rebuilt on upstream v8 -- the unit tests only covered
watch._rebuild_code(no_viz=...), never the CLI arg-parse, so it shipped broken.

Each flag is probed against a nonexistent path so we exercise arg-PARSING, not
execution: a parseable flag yields "path not found" (or a backend error); a DROPPED
flag yields "unknown ... option: <flag>" -> the test fails.
"""
import subprocess
import sys

import pytest


def _graphify(*args: str) -> str:
    r = subprocess.run([sys.executable, "-m", "graphify", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr)


# Commands that must always exist in the CLI surface.
REQUIRED_COMMANDS = [
    "install", "uninstall", "update", "extract", "watch", "query", "path",
    "explain", "affected", "merge-graphs", "merge-driver", "global", "diagnose",
    "cluster-only", "label", "tree", "export", "add", "save-result", "check-update",
]

# Boolean flags that must stay parseable per command. Dropping one regresses the
# CLI surface even when the underlying function still supports the behavior.
COMMAND_FLAGS = {
    "update": ["--force", "--no-cluster", "--no-viz"],
    "extract": ["--multigraph", "--simple", "--no-cluster", "--global"],
    "merge-graphs": ["--multigraph", "--simple"],
    "cluster-only": ["--no-viz"],
}

_BAD_PATH = "/no/such/graphify/path/xyz-123"


@pytest.mark.parametrize("cmd", REQUIRED_COMMANDS)
def test_command_present_in_help(cmd: str) -> None:
    assert cmd in _graphify("--help"), f"CLI command '{cmd}' disappeared from `graphify --help`"


@pytest.mark.parametrize(
    "cmd,flag",
    [(c, f) for c, flags in COMMAND_FLAGS.items() for f in flags],
)
def test_flag_is_parseable(cmd: str, flag: str) -> None:
    out = _graphify(cmd, _BAD_PATH, flag).lower()
    rejected = "unknown" in out and flag.lower() in out
    assert not rejected, f"`graphify {cmd} {flag}` is rejected as an unknown option (CLI surface regression)"
