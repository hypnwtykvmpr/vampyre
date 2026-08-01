"""Fork test policy: skipped tests are failures, never hidden or deselected."""

from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert every runtime skip into a hard failure."""
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not getattr(report, "wasxfail", False):
        report.outcome = "failed"
        report.longrepr = (
            "POLICY VIOLATION: this fork treats test skips as failures, but "
            f"{item.nodeid} was skipped. Make the test hermetic or satisfy its "
            "required environment."
        )
