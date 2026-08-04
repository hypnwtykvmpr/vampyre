"""Repository-wide test isolation and zero-waiver outcome policy."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_COLLECTION_HOME = Path(tempfile.mkdtemp(prefix="graphify-pytest-home-")).resolve()
atexit.register(shutil.rmtree, _COLLECTION_HOME, ignore_errors=True)

# Apply isolation before test modules import graphify. Several modules read user
# config and provider settings at import time, before fixtures can run.
os.environ["HOME"] = str(_COLLECTION_HOME)
os.environ["USERPROFILE"] = str(_COLLECTION_HOME)
os.environ["LOCALAPPDATA"] = str(_COLLECTION_HOME / "AppData" / "Local")
os.environ["APPDATA"] = str(_COLLECTION_HOME / "AppData" / "Roaming")
os.environ["XDG_CONFIG_HOME"] = str(_COLLECTION_HOME / ".config")
os.environ.pop("CLAUDE_CONFIG_DIR", None)
os.environ["GIT_CEILING_DIRECTORIES"] = str(Path(tempfile.gettempdir()).resolve())

_deselected_nodeids: set[str] = set()


def _enforce_forbidden_report(report, nodeid: str) -> bool:
    """Turn skips and expected-failure outcomes into ordinary failures."""
    was_xfail = hasattr(report, "wasxfail")
    if not report.skipped and not (report.passed and was_xfail):
        return False
    kind = "xfail" if report.skipped and was_xfail else "xpass" if was_xfail else "skip"
    report.outcome = "failed"
    if was_xfail:
        del report.wasxfail
    report.longrepr = (
        f"POLICY VIOLATION: {nodeid} produced {kind}. "
        "Make the test hermetic and require a real pass."
    )
    return True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert runtime skips, xfails, and xpasses into hard failures."""
    outcome = yield
    report = outcome.get_result()
    _enforce_forbidden_report(report, item.nodeid)


def pytest_collectreport(report) -> None:
    """Collection-time skips, including importorskip, are failures too."""
    _enforce_forbidden_report(report, report.nodeid)


def pytest_configure(config) -> None:
    """Reject explicit filters before xdist launches collection workers."""
    filters = (
        ("-k", getattr(config.option, "keyword", "")),
        ("-m", getattr(config.option, "markexpr", "")),
        ("--deselect", getattr(config.option, "deselect", None)),
    )
    for option, value in filters:
        if value:
            raise pytest.UsageError(
                f"POLICY VIOLATION: test selection via {option} deselects tests; "
                "the suite must run in full."
            )


def pytest_deselected(items) -> None:
    _deselected_nodeids.update(item.nodeid for item in items)
    if not items:
        return
    worker_output = getattr(items[0].config, "workeroutput", None)
    if worker_output is not None:
        worker_output["graphify_deselected_nodeids"] = sorted(_deselected_nodeids)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """Transfer worker-side deselections without masking worker crashes."""
    worker_output = getattr(node, "workeroutput", None)
    if not worker_output:
        return
    _deselected_nodeids.update(worker_output.get("graphify_deselected_nodeids", ()))


def pytest_sessionfinish(session, exitstatus) -> None:
    if _deselected_nodeids:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter) -> None:
    if _deselected_nodeids:
        terminalreporter.write_sep(
            "=",
            f"POLICY VIOLATION: {len(_deselected_nodeids)} test(s) were deselected",
            red=True,
        )
