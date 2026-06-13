"""Fork test policy: we do not ship skips.

Upstream ships two tests that are environment-/state-gated and therefore
*conditionally skip* — which this fork does not allow to surface as "skipped":

  * ``tests/test_falkordb_integration.py`` — an opt-in integration test that
    needs a live FalkorDB server (``docker run -p 6379:6379 falkordb/falkordb``)
    plus the optional ``falkordb`` SDK. Its module top-level calls
    ``pytest.importorskip("falkordb")``, so it skips at *import* time offline.
  * ``tests/test_install_references.py::test_unbuilt_bundle_host_falls_back_to_monolith``
    — only meaningful while some progressive host bundle has *not* shipped yet.
    All currently ship, so it ``pytest.skip(...)``s at runtime.

Neither touches fork-specific code; both skip identically on a clean
``upstream/v8`` checkout. Rather than let them report as skips, we **deselect**
them from the default run (kept in the repo, runnable on demand) and **escalate
every other skip to a hard failure** so no skip ever ships silently again.

Run the opt-in gated tests explicitly with::

    GRAPHIFY_RUN_OPTIN_TESTS=1 uv run --frozen pytest \
        tests/test_falkordb_integration.py \
        tests/test_install_references.py

To add a new gated exception, append a precise nodeid substring to
``_OPTIN_ONLY`` *with a comment naming the concrete external/state condition* —
never to hide a real failure.
"""
from __future__ import annotations

import os

import pytest

_OPT_IN = os.environ.get("GRAPHIFY_RUN_OPTIN_TESTS") == "1"

# nodeid substrings for the documented environment/state gates above.
_OPTIN_ONLY = (
    "tests/test_falkordb_integration.py",  # needs a live FalkorDB server (docker)
    # progressive-host fallback: only runs when a bundle has NOT shipped yet
    "tests/test_install_references.py::test_unbuilt_bundle_host_falls_back_to_monolith",
)

# The falkordb module skips at *import* (module-level importorskip), so it never
# reaches collection-item filtering — drop the whole file unless opting in.
collect_ignore: list[str] = []
if not _OPT_IN:
    collect_ignore.append("tests/test_falkordb_integration.py")


def _is_optin(nodeid: str) -> bool:
    return any(sub in nodeid for sub in _OPTIN_ONLY)


def pytest_collection_modifyitems(config, items):
    """Deselect the documented opt-in gated tests from the default run."""
    if _OPT_IN:
        return
    keep, deselected = [], []
    for item in items:
        (deselected if _is_optin(item.nodeid) else keep).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Policy gate: a skip that is not a documented opt-in gate is a failure."""
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not getattr(report, "wasxfail", False) and not _is_optin(item.nodeid):
        report.outcome = "failed"
        report.longrepr = (
            "POLICY VIOLATION: this fork treats test skips as failures, but "
            f"{item.nodeid} was skipped. Fix the test, or add a documented "
            "opt-in gate to conftest.py._OPTIN_ONLY if it is genuinely "
            "environment/state-conditional."
        )
