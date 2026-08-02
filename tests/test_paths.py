"""Tests for graphify.paths — the shared test-path classifier (#1553)."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify.paths import (
    _is_test_path,
    disambiguate_ambiguous_candidates,
    resolve_scan_root_marker,
    write_scan_root_marker,
)


@pytest.mark.parametrize(
    "path",
    [
        # test dir segments
        "tests/foo.py",
        "src/tests/foo.py",
        "test/foo.go",
        "spec/foo.rb",
        "specs/foo.rb",
        "app/__tests__/foo.js",
        "a/b/TESTS/foo.py",  # case-insensitive segment
        # test filename conventions
        "src/test_service.py",
        "pkg/service_test.go",
        "src/service.test.ts",
        "src/service.spec.ts",
        "src/service_spec.rb",
        "ps/Module.Tests.ps1",
        "java/FooTest.java",
        "java/FooTests.java",
        "cs/FooTests.cs",
        # windows separators
        "src\\tests\\foo.py",
        "src\\service_test.py",
    ],
)
def test_is_test_path_positive(path: str) -> None:
    assert _is_test_path(path) is True, path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "latest.py",
        "contest.py",
        "src/contest.py",
        "src/greatest/x.py",
        "src/service.py",
        "lib/helper.go",
        "src/attestation.py",  # "test" only as substring, not a segment
        "src/testimony.py",  # filename starts with "test" but no underscore
        "src/contest/x.py",  # "contest" is not "test"
        "src/greatest.cs",  # ends with "test" but not "Tests.cs"
        "src/protest.java",  # not "*Test.java"
        "config/manifest.json",
    ],
)
def test_is_test_path_negative(path: str) -> None:
    assert _is_test_path(path) is False, path


def test_disambiguate_drops_test_candidate_for_nontest_call_site() -> None:
    winner = disambiguate_ambiguous_candidates(
        ["src", "mock"],
        {"src": "src/service.py", "mock": "tests/test_service.py"},
        "src/caller.py",
    )
    assert winner == "src"


def test_disambiguate_bails_on_two_nontest_candidates() -> None:
    winner = disambiguate_ambiguous_candidates(
        ["a", "b"],
        {"a": "alpha/a.py", "b": "beta/b.py"},
        "pkg/caller.py",
    )
    assert winner is None


def test_disambiguate_test_call_site_prefers_test_local() -> None:
    winner = disambiguate_ambiguous_candidates(
        ["src", "local"],
        {"src": "src/service.py", "local": "tests/test_service.py"},
        "tests/test_service.py",
    )
    assert winner == "local"


def test_disambiguate_path_proximity_same_dir() -> None:
    # Two non-test candidates; the one in the call site's directory wins.
    winner = disambiguate_ambiguous_candidates(
        ["near", "far"],
        {"near": "pkg/a/service.py", "far": "pkg/b/service.py"},
        "pkg/a/caller.py",
    )
    assert winner == "near"


def test_scan_root_marker_round_trips_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "project" / "Sources"
    source_root.mkdir(parents=True)
    marker = tmp_path / "canonical" / "graphify-out" / ".graphify_root"
    marker.parent.mkdir(parents=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    write_scan_root_marker(marker, source_root)
    monkeypatch.chdir(unrelated)

    assert marker.read_text(encoding="utf-8").startswith("marker-relative:")
    assert resolve_scan_root_marker(marker) == source_root.resolve()


def test_resolve_scan_root_marker_supports_phase1_relative_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "project" / "Sources"
    marker = source_root / "graphify-out" / ".graphify_root"
    marker.parent.mkdir(parents=True)
    marker.write_text(".", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    assert resolve_scan_root_marker(marker) == source_root.resolve()


def test_resolve_scan_root_marker_supports_legacy_absolute_marker(tmp_path: Path) -> None:
    source_root = tmp_path / "project" / "Sources"
    source_root.mkdir(parents=True)
    marker = tmp_path / "canonical" / "graphify-out" / ".graphify_root"
    marker.parent.mkdir(parents=True)
    marker.write_text(str(source_root), encoding="utf-8")

    assert resolve_scan_root_marker(marker) == source_root.resolve()


def test_resolve_scan_root_marker_supports_legacy_cwd_relative_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source_root = project / "Sources"
    source_root.mkdir(parents=True)
    marker = tmp_path / "canonical" / "graphify-out" / ".graphify_root"
    marker.parent.mkdir(parents=True)
    marker.write_text("Sources", encoding="utf-8")
    monkeypatch.chdir(project)

    assert resolve_scan_root_marker(marker) == source_root.resolve()


@pytest.mark.parametrize(
    "recorded", ["", "marker-relative:", "bad\x00path", "marker-relative:bad\x00path"]
)
def test_resolve_scan_root_marker_rejects_malformed_values(tmp_path: Path, recorded: str) -> None:
    marker = tmp_path / ".graphify_root"
    marker.write_text(recorded, encoding="utf-8")

    assert resolve_scan_root_marker(marker) is None
