"""Repository policy checks for standalone Vampyre."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
VAMPYRE_SOURCES = {
    "git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5",
    "git+https://github.com/hypnwtykvmpr/vampyre.git@main",
}
GUIDANCE_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".sh", ".ps1"}
PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("ARCHITECTURE.md"),
    Path("SECURITY.md"),
    Path("PROJECT.md"),
    Path("docs/how-it-works.md"),
    Path("docs/command-reference.md"),
    Path("docs/agent-integrations.md"),
    Path("docs/releases/0.9.5.md"),
    Path("packaging/release/INSTALL.md"),
)


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
        if path.is_file() and (
            path.name == "Dockerfile" or path.suffix.lower() in GUIDANCE_SUFFIXES
        ):
            paths.append(path)
    return paths


def test_tracked_guidance_uses_uv_and_installs_vampyre() -> None:
    forbidden_install = re.compile(
        r"\b(?:p"
        + r"ip3?\s+(?:install|uninstall)|python[^\n]*-m\s+(?:p"
        + r"ip|venv)\b|p"
        + r"ipx\b|poetry\s+(?:install|add|remove|update)|conda\s+(?:install|create|env))",
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
            if ambiguous_uv.search(line) and not any(source in line for source in VAMPYRE_SOURCES):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: non-Vampyre uv source")

    assert violations == []


def test_project_metadata_points_to_vampyre() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "https://github.com/hypnwtykvmpr/vampyre" in metadata
    assert "https://github.com/safishamsi/graphify" not in metadata


def test_version_remains_human_controlled() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'version = "0.9.5"' in metadata


def test_public_branch_guidance_uses_main_for_development() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tracked_markdown = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    retired_checkout = [
        name
        for name in tracked_markdown
        if "git checkout v8" in (ROOT / name).read_text(encoding="utf-8", errors="replace")
    ]

    assert "git checkout main" in readme
    assert "Active development happens on `main`" in readme
    assert "Development snapshots install from `main`" in readme
    assert "@v9" not in readme
    assert retired_checkout == []


def test_public_docs_describe_a_standalone_repository() -> None:
    forbidden = (
        "independent fork",
        "this fork",
        "active fork",
        "upstream synchronization",
        "github.com/safishamsi/graphify",
        "github.com/graphify-labs/graphify",
        "graphifylabs.ai",
        "@v9",
    )
    authority_docs = (
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "PROJECT.md",
        ROOT / "docs" / "releases" / "0.9.5.md",
        ROOT / "packaging" / "release" / "INSTALL.md",
    )

    assert not (ROOT / "FORK.md").exists()
    assert not (ROOT / "docs" / "translations").exists()
    assert not (ROOT / "docs" / "node-summaries-rfc.md").exists()
    for relative_path in PUBLIC_DOCUMENTS:
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker not in text, f"{path.relative_to(ROOT)} still contains {marker!r}"
    for path in authority_docs:
        text = path.read_text(encoding="utf-8").lower()
        assert "https://github.com/hypnwtykvmpr/vampyre" in text
        assert "canonical" in text


def test_retired_translations_have_public_migration_guidance() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    project = (ROOT / "PROJECT.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "releases" / "0.9.5.md").read_text(encoding="utf-8")

    assert "## Documentation Language" in readme
    assert "Pre-standalone translations were retired" in readme
    assert "English documentation is authoritative" in project
    assert "Pre-standalone translations" in project
    assert "translated READMEs were retired" in release_notes


def test_readme_links_current_command_and_platform_references() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    command_reference = (ROOT / "docs" / "command-reference.md").read_text(encoding="utf-8")
    integrations = (ROOT / "docs" / "agent-integrations.md").read_text(encoding="utf-8")

    assert "[Command reference](docs/command-reference.md)" in readme
    assert "[Agent integration matrix](docs/agent-integrations.md)" in readme
    for command in ("extract", "update", "query", "export", "provider", "serve"):
        assert f"`graphify {command}" in command_reference
    for platform in ("Claude Code", "Codex", "GitHub Copilot CLI", "VS Code Copilot Chat"):
        assert platform in integrations


def test_public_references_cover_the_live_top_level_command_inventory() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    commands = {
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.match(r"^  ([a-z][a-z0-9-]*)(?:\s|$)", line))
    }
    references = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (Path("docs/command-reference.md"), Path("docs/agent-integrations.md"))
    )
    undocumented = sorted(
        command for command in commands if f"graphify {command}" not in references
    )

    assert commands
    assert undocumented == []


def test_agent_matrix_covers_the_live_install_platform_inventory() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "install", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    platform_line = next(
        line.removeprefix("Platforms: ")
        for line in result.stdout.splitlines()
        if line.startswith("Platforms: ")
    )
    platforms = platform_line.split(", ")
    integrations = (ROOT / "docs" / "agent-integrations.md").read_text(encoding="utf-8")

    assert platforms
    assert (
        sorted(platform for platform in platforms if f"--platform {platform}" not in integrations)
        == []
    )


def test_public_install_references_distinguish_distribution_from_command() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    references = [
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (Path("docs/command-reference.md"), Path("docs/agent-integrations.md"))
    ]

    assert 'name = "graphifyy"' in metadata
    for text in references:
        assert "The distribution name is `graphifyy`; the installed command is `graphify`." in text
        assert 'uv tool install --force "graphifyy @ git+' in text


def test_public_documentation_local_links_resolve() -> None:
    missing: list[str] = []
    link_pattern = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")

    for relative in PUBLIC_DOCUMENTS:
        document = ROOT / relative
        assert document.is_file(), f"missing public document: {relative}"
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(target.split("#", 1)[0])
            if path_text and not (document.parent / path_text).resolve().exists():
                missing.append(f"{relative}: {target}")

    assert missing == []


def test_live_docs_use_current_security_and_privacy_model() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "Query logging is off by default" in readme
    assert "Non-loopback binds require all three" in readme
    assert "API key" in security
    assert "TLS certificate and key" in security
    assert "0.9.5" in security
    assert "no network listener" not in security.lower()


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


def test_tracked_home_paths_are_only_explicit_portability_examples() -> None:
    home_pattern = re.compile(
        r"[A-Za-z]:(?:\\Users\\|/Users/)[A-Za-z0-9._-]+"
        r"|/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+",
        re.IGNORECASE,
    )
    allowed = {
        "CHANGELOG.md": {"/Users/..."},
        "graphify/extract.py": {"/home/victim"},
        "tests/test_claude_cli_backend.py": {"/home/tester", "/home/alice"},
        "tests/test_detect.py": {"/home/user"},
        "tests/test_hooks.py": {r"C:\Users\u", r"c:/Users/u"},
        "tests/test_paths.py": {r"C:\Users\dev"},
        "tests/test_prs.py": {"/home/user"},
    }
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    violations: list[str] = []
    for name in tracked:
        if name == "tests/test_repository_policy.py":
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for match in home_pattern.findall(line):
                if match not in allowed.get(name, set()):
                    violations.append(f"{name}:{line_number}: {match}")

    assert violations == []


def test_force_guidance_matches_narrow_shrink_semantics() -> None:
    surfaces = {
        name: (ROOT / name).read_text(encoding="utf-8", errors="replace")
        for name in ("README.md", "graphify/serve.py", "graphify/cache.py")
    }
    assert "graphify extract --full --force" in surfaces["graphify/serve.py"]
    assert "clear ghost duplicates" not in surfaces["README.md"]
    assert "bypass" not in "\n".join(
        line.lower()
        for line in surfaces["graphify/cache.py"].splitlines()
        if "force" in line.lower()
    )

    contradictions: list[str] = []
    unclassified_extract_force: list[str] = []
    for path in _tracked_guidance_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            lowered = line.lower()
            if "full build with --force" in lowered:
                contradictions.append(f"{path.relative_to(ROOT)}:{line_number}")
            if "graphify extract" in lowered and "--force" in lowered:
                is_full = "--full --force" in lowered
                is_narrow = any(
                    marker in lowered
                    for marker in (
                        "non-empty shrink",
                        "verified non-empty",
                        "непорожнє зменшення",
                    )
                )
                if not (is_full or is_narrow):
                    unclassified_extract_force.append(
                        f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                    )

    assert contradictions == []
    assert unclassified_extract_force == []
