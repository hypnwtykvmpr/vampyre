"""CLI surface-parity guard.

Catches the class of regression where a rebuild/merge silently DROPS a CLI command
or flag. Real example this guards against: `graphify update --no-viz` was dropped
when graphify/__main__.py was rebuilt from an older baseline -- the unit tests only covered
watch._rebuild_code(no_viz=...), never the CLI arg-parse, so it shipped broken.

Legacy probes use a nonexistent path where the command parses options first. The
strict extract/update/watch probes use an existing temporary path followed by an
unknown sentinel, so they prove each candidate flag was consumed before execution.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest


def _graphify(*args: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "graphify", *args], capture_output=True, text=True, encoding="utf-8"
    )
    return r.stdout + r.stderr


# Commands that must always exist in the CLI surface.
REQUIRED_COMMANDS = [
    "install",
    "uninstall",
    "update",
    "extract",
    "watch",
    "query",
    "path",
    "explain",
    "affected",
    "merge-graphs",
    "merge-driver",
    "global",
    "diagnose",
    "cluster-only",
    "label",
    "tree",
    "export",
    "add",
    "save-result",
    "check-update",
    "provider",
    "prs",
    "serve",
]

# Boolean flags that must stay parseable per command. Dropping one regresses the
# CLI surface even when the underlying function still supports the behavior.
COMMAND_FLAGS = {
    "merge-graphs": ["--multigraph", "--simple"],
    "cluster-only": ["--no-viz"],
}


def test_serve_subcommand_delegates_to_installed_server(monkeypatch) -> None:
    import graphify.__main__ as mainmod
    import graphify.serve as serve

    received: list[list[str]] = []
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _path: None)
    monkeypatch.setattr(serve, "_main", lambda argv=None: received.append(list(argv or [])))
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        ["graphify", "serve", "custom.json", "--transport", "stdio"],
    )

    mainmod.main()

    assert received == [["custom.json", "--transport", "stdio"]]


# These parsers reject unknown options before doing work. Every documented flag
# for them is exercised with a final unknown sentinel, proving that the tested
# flag was consumed rather than merely appearing in help text.
STRICT_COMMAND_FLAG_VALUES: dict[str, dict[str, str | None]] = {
    "extract": {
        "--api-timeout": "1",
        "--as": "repo",
        "--backend": "openai",
        "--cargo": None,
        "--dedup-llm": None,
        "--directed": None,
        "--exclude": "ignored/**",
        "--exclude-hubs": "95",
        "--force": None,
        "--full": None,
        "--global": None,
        "--google-workspace": None,
        "--max-concurrency": "1",
        "--max-workers": "1",
        "--mode": "deep",
        "--model": "test-model",
        "--multigraph": None,
        "--no-cluster": None,
        "--out": "{tmp}",
        "--postgres": "postgresql://example.invalid/db",
        "--resolution": "1",
        "--simple": None,
        "--timing": None,
        "--token-budget": "1",
    },
    "update": {
        "--api-timeout": "1",
        "--backend": "openai",
        "--cargo": None,
        "--dedup-llm": None,
        "--exclude": "ignored/**",
        "--exclude-hubs": "95",
        "--force": None,
        "--google-workspace": None,
        "--max-concurrency": "1",
        "--max-workers": "1",
        "--mode": "deep",
        "--model": "test-model",
        "--no-cluster": None,
        "--no-viz": None,
        "--out": "{tmp}",
        "--postgres": "postgresql://example.invalid/db",
        "--rebuild": None,
        "--remap": None,
        "--repair-state": None,
        "--resolution": "1",
        "--timing": None,
        "--token-budget": "1",
    },
    "watch": {"--out": "{tmp}"},
}

_BAD_PATH = "/no/such/graphify/path/xyz-123"


@pytest.mark.parametrize("cmd", REQUIRED_COMMANDS)
def test_command_present_in_help(cmd: str) -> None:
    help_text = _graphify("--help")
    assert re.search(rf"(?m)^  {re.escape(cmd)}(?:\s|$)", help_text), (
        f"CLI command '{cmd}' disappeared from `graphify --help`"
    )


def test_prs_help_reaches_command_specific_usage() -> None:
    output = _graphify("prs", "--help")

    assert "graphify prs --base <branch>" in output
    assert "Run 'graphify --help' for full usage." not in output


def test_provider_help_reaches_command_specific_usage() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "provider", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "graphify provider [add|list|show|remove]" in output
    assert "Run 'graphify --help' for full usage." not in output


@pytest.mark.parametrize(
    "cmd,flag",
    [(c, f) for c, flags in COMMAND_FLAGS.items() for f in flags],
)
def test_flag_is_parseable(cmd: str, flag: str) -> None:
    out = _graphify(cmd, _BAD_PATH, flag).lower()
    rejected = "unknown" in out and flag.lower() in out
    assert not rejected, (
        f"`graphify {cmd} {flag}` is rejected as an unknown option (CLI surface regression)"
    )


@pytest.mark.parametrize(
    "command,flag,value",
    [
        (command, flag, value)
        for command, flags in STRICT_COMMAND_FLAG_VALUES.items()
        for flag, value in flags.items()
    ],
)
def test_strict_command_flag_is_consumed_before_unknown_sentinel(
    tmp_path, command: str, flag: str, value: str | None
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sentinel = "--graphify-parser-sentinel"
    args = [command, str(source), flag]
    if value is not None:
        args.append(value.format(tmp=str(tmp_path / "output")))
    args.append(sentinel)

    output = _graphify(*args).lower()

    assert f"unknown {command} option: {sentinel}" in output
    assert f"unknown {command} option: {flag}" not in output


def test_tracked_guidance_uses_only_parser_proven_strict_command_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    guidance_suffixes = {".md", ".py", ".toml", ".yml", ".yaml", ".sh", ".ps1"}
    invocation = re.compile(
        r"(?<![/\w\[])graphify[ \t]+(extract|update|watch)\b([^`\n|;]*)",
        re.IGNORECASE,
    )
    flag_pattern = re.compile(r"(?<!\w)(--[a-z][\w-]*)", re.IGNORECASE)
    violations: list[str] = []

    for name in tracked:
        path = root / name
        if not path.is_file():
            continue
        if name == "CHANGELOG.md" or name.startswith("tests/"):
            continue
        if path.name != "Dockerfile" and path.suffix.lower() not in guidance_suffixes:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for match in invocation.finditer(line):
                command = match.group(1).lower()
                for flag in flag_pattern.findall(match.group(2)):
                    if flag not in STRICT_COMMAND_FLAG_VALUES[command]:
                        violations.append(f"{name}:{line_number}: graphify {command} {flag}")

    assert violations == []


@pytest.mark.parametrize("command", ["cluster-only", "label"])
def test_explicit_graph_selects_its_own_lifecycle_lock_directory(tmp_path, command):
    from graphify.__main__ import _stateful_command_output_dir

    graph_path = tmp_path / "canonical" / "graph.json"
    graph_path.parent.mkdir()
    graph_path.write_text("{}", encoding="utf-8")

    assert _stateful_command_output_dir(["graphify", command, "--graph", str(graph_path)]) == (
        graph_path.parent
    )
