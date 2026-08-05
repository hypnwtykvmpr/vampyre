"""Repository policy checks for the independent Vampyre fork."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORK_SOURCE = "git+https://github.com/hypnwtykvmpr/vampyre.git@v9"
GUIDANCE_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".sh", ".ps1"}


def _tracked_guidance_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    paths: list[Path] = []
    for name in tracked:
        path = ROOT / name
        if name == "CHANGELOG.md" or name.startswith("tests/"):
            continue
        if path.name == "Dockerfile" or path.suffix.lower() in GUIDANCE_SUFFIXES:
            paths.append(path)
    return paths


def test_tracked_guidance_uses_uv_and_installs_the_fork() -> None:
    forbidden_install = re.compile(
        r"\b(?:p" + r"ip3?\s+(?:install|uninstall)|python[^\n]*-m\s+p" + r"ip\b|p" + r"ipx\b)",
        re.IGNORECASE,
    )
    ambiguous_uv = re.compile(
        r"\b(?:uvx|uv\s+tool\s+(?:install|upgrade|run))\b[^\n]*\bgraphifyy(?:\[[^]]+\])?\b",
        re.IGNORECASE,
    )
    violations: list[str] = []

    for path in _tracked_guidance_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if forbidden_install.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: forbidden manager")
            if ambiguous_uv.search(line) and FORK_SOURCE not in line:
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: non-fork uv source")

    assert violations == []


def test_project_metadata_points_to_the_fork() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "https://github.com/hypnwtykvmpr/vampyre" in metadata
    assert "https://github.com/safishamsi/graphify" not in metadata


def test_ci_targets_active_branches_and_security_findings_block() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    branch_lines = [line for line in workflow.splitlines() if "branches:" in line]

    assert branch_lines
    assert all('"v9"' in line and '"v8"' not in line for line in branch_lines)
    assert "continue-on-error" not in workflow
    assert "--audit-coverage" not in workflow
    assert "--monolith-roundtrip" not in workflow
    assert "--always-on-roundtrip" not in workflow


def test_ci_runs_full_strict_suite_on_windows_macos_and_linux() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    test_job = workflow.split("\n  test:", 1)[1].split("\n  quality:", 1)[0]

    assert "ubuntu-latest" in test_job
    assert "macos-latest" in test_job
    assert "windows-latest" in test_job
    assert 'python-version: ["3.10", "3.13"]' in test_job
    assert "fail-fast: false" in test_job
    assert "pytest tests/ -n auto" in test_job
    assert "-k " not in test_job
    assert "PYTHONUTF8" not in test_job
    assert "PYTHONIOENCODING" not in test_job


def test_pytest_parallel_scheduler_preserves_xdist_groups_by_default() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pytest_config = metadata.split("[tool.pytest.ini_options]", 1)[1].split("\n[tool.", 1)[0]

    assert "--dist loadgroup" in pytest_config


def test_ci_enforces_explicit_text_encodings_across_the_tracked_python_tree() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    quality_job = workflow.split("\n  quality:", 1)[1].split("\n  security-scan:", 1)[0]

    assert "ruff check . --select PLW1514 --preview" in quality_job
    assert "python -m tools.text_encoding_check --check" in quality_job


def test_parallel_ci_jobs_have_a_single_uv_cache_writer_per_python_version() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    skillgen_job = workflow.split("  skillgen-check:", 1)[1].split("\n  test:", 1)[0]
    test_job = workflow.split("\n  test:", 1)[1].split("\n  quality:", 1)[0]
    quality_job = workflow.split("\n  quality:", 1)[1].split("\n  security-scan:", 1)[0]
    security_job = workflow.split("\n  security-scan:", 1)[1]

    assert 'save-cache: "false"' in skillgen_job
    assert 'save-cache: "false"' in quality_job
    assert 'save-cache: "false"' in security_job
    assert 'save-cache: "false"' not in test_job


def test_skill_generator_uses_current_snapshots_not_frozen_v8() -> None:
    generator = (ROOT / "tools" / "skillgen" / "gen.py").read_text(encoding="utf-8")
    manifest = (ROOT / "tools" / "skillgen" / "platforms.toml").read_text(encoding="utf-8")
    retired = (
        "origin/v8",
        "_V8_BASELINE_SHA",
        "roundtrip_ref",
        "--audit-coverage",
        "--monolith-roundtrip",
        "--always-on-roundtrip",
    )

    for marker in retired:
        assert marker not in generator
        assert marker not in manifest


def test_security_suppressions_are_absent() -> None:
    violations: list[str] = []
    for path in (ROOT / "graphify").rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "# nosec" in line:
                violations.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert violations == []


def test_removed_graph_backend_stays_absent() -> None:
    forbidden = "falkor" + "db"
    paths = _tracked_guidance_files()
    paths.extend((ROOT / "tests").glob("*.py"))
    paths.append(ROOT / "uv.lock")

    violations = [
        str(path.relative_to(ROOT))
        for path in paths
        if forbidden in path.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert violations == []
