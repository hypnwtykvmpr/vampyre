"""Security contract for the shared Claude Code CLI launcher."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


def _subject():
    return importlib.import_module("graphify.claude_cli")


def _valid_help() -> str:
    return " ".join(
        (
            "--safe-mode",
            "--tools",
            "--strict-mcp-config",
            "--mcp-config",
            "--setting-sources",
            "--settings",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "--permission-mode",
            "--no-chrome",
        )
    )


def _result_envelope(result: str = "ok") -> str:
    return json.dumps(
        {
            "type": "result",
            "result": result,
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "modelUsage": {"test-model": {}},
            "stop_reason": "end_turn",
        }
    )


def test_shared_launcher_module_exists():
    assert importlib.util.find_spec("graphify.claude_cli") is not None


def test_text_launcher_removes_ambient_authority(monkeypatch):
    subject = _subject()
    subject._validated_executable.cache_clear()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "/tools/claude")
    monkeypatch.setenv("PATH", "/tools")
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("MCP_TOKEN", "must-not-leak")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        if "--help" in args:
            return SimpleNamespace(returncode=0, stdout=_valid_help(), stderr="")
        cwd = Path(kwargs["cwd"])
        assert cwd.is_dir()
        assert list(cwd.iterdir()) == []
        return SimpleNamespace(returncode=0, stdout=_result_envelope(), stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    envelope = subject.run_claude_cli("untrusted prompt", timeout=7.0)

    assert envelope["result"] == "ok"
    assert len(calls) == 2
    command, kwargs = calls[-1]
    for flag in (
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--no-chrome",
    ):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--add-dir" not in command
    assert kwargs["input"] == "untrusted prompt"
    assert kwargs["timeout"] == 7.0
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["env"]["PATH"] == "/tools"
    assert kwargs["env"]["HOME"] == "/home/tester"
    assert "CLAUDECODE" not in kwargs["env"]
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert "MCP_TOKEN" not in kwargs["env"]


def test_launcher_fails_closed_when_required_flag_is_missing(monkeypatch):
    subject = _subject()
    subject._validated_executable.cache_clear()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "/tools/claude")
    calls = 0

    def fake_run(args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=0,
            stdout=_valid_help().replace("--strict-mcp-config", ""),
            stderr="",
        )

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    with pytest.raises(subject.ClaudeCLIError, match="required isolation controls"):
        subject.run_claude_cli("prompt", timeout=1.0)
    assert calls == 1


@pytest.mark.parametrize(
    ("stdout", "returncode", "stderr"),
    [
        ("not json: TOP-SECRET /home/alice/project", 0, ""),
        ("", 17, "TOP-SECRET /home/alice/project/source.py"),
    ],
)
def test_launcher_errors_do_not_disclose_output_or_paths(monkeypatch, stdout, returncode, stderr):
    subject = _subject()
    subject._validated_executable.cache_clear()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "/tools/claude")

    def fake_run(args, **_kwargs):
        if "--help" in args:
            return SimpleNamespace(returncode=0, stdout=_valid_help(), stderr="")
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    with pytest.raises(subject.ClaudeCLIError) as exc_info:
        subject.run_claude_cli("prompt", timeout=1.0)
    message = str(exc_info.value)
    assert "TOP-SECRET" not in message
    assert "/home/alice" not in message
    assert "source.py" not in message


def test_launcher_accepts_stream_event_array(monkeypatch):
    subject = _subject()
    subject._validated_executable.cache_clear()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "/tools/claude")
    stream = json.dumps([{"type": "system"}, json.loads(_result_envelope("done"))])

    def fake_run(args, **_kwargs):
        if "--help" in args:
            return SimpleNamespace(returncode=0, stdout=_valid_help(), stderr="")
        return SimpleNamespace(returncode=0, stdout=stream, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    assert subject.run_claude_cli("prompt", timeout=1.0)["result"] == "done"


@pytest.mark.parametrize(
    "envelope",
    [
        {"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "partial"},
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": False,
            "result": "partial",
        },
        {"type": "result", "subtype": "success", "is_error": "false", "result": "partial"},
    ],
)
def test_launcher_rejects_error_or_malformed_result_envelopes(monkeypatch, envelope):
    subject = _subject()
    subject._validated_executable.cache_clear()
    monkeypatch.setattr(subject.shutil, "which", lambda _name: "/tools/claude")

    def fake_run(args, **_kwargs):
        if "--help" in args:
            return SimpleNamespace(returncode=0, stdout=_valid_help(), stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    with pytest.raises(subject.ClaudeCLIError, match="result envelope"):
        subject.run_claude_cli("prompt", timeout=1.0)


def test_launcher_rejects_image_mode_before_starting_a_process(monkeypatch, tmp_path):
    subject = _subject()
    subject._validated_executable.cache_clear()
    image = tmp_path / "private.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("image mode must fail before process launch"),
    )
    with pytest.raises(subject.ClaudeCLIError, match="inline-vision API backend"):
        subject.run_claude_cli("prompt", timeout=1.0, image_paths=[image])


def test_windows_resolver_prefers_executable_shim(monkeypatch):
    subject = _subject()
    monkeypatch.setattr(subject.platform, "system", lambda: "Windows")
    candidates = {
        "claude.exe": None,
        "claude.cmd": "C:/tools/claude.cmd",
        "claude": "C:/tools/claude.ps1",
    }
    monkeypatch.setattr(subject.shutil, "which", candidates.get)
    assert subject._resolve_executable() == "C:/tools/claude.cmd"


def test_extraction_backend_delegates_to_shared_launcher(monkeypatch):
    from graphify import claude_cli, llm

    calls: list[dict] = []

    def fake_launcher(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return {
            "type": "result",
            "result": '{"nodes":[],"edges":[],"hyperedges":[]}',
            "usage": {},
            "modelUsage": {},
            "stop_reason": "end_turn",
        }

    monkeypatch.setattr(claude_cli, "run_claude_cli", fake_launcher)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("llm.py must not launch Claude directly"),
    )
    result = llm._call_claude_cli("CORPUS")
    assert result["nodes"] == []
    assert len(calls) == 1
    assert "CORPUS" in calls[0]["prompt"]


def test_secondary_llm_dispatch_delegates_to_shared_launcher(monkeypatch):
    from graphify import claude_cli, llm

    calls: list[dict] = []

    def fake_launcher(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return {"type": "result", "result": "answer"}

    monkeypatch.setattr(claude_cli, "run_claude_cli", fake_launcher)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("llm.py must not launch Claude directly"),
    )
    assert llm._call_llm("question", backend="claude-cli") == "answer"
    assert calls[0]["prompt"] == "question"


def test_pr_triage_delegates_to_shared_launcher(monkeypatch, capsys):
    from graphify import claude_cli, prs

    candidate = cast(
        prs.PRInfo,
        SimpleNamespace(
            base_branch="main",
            status="READY",
            blast_radius="",
            number=1,
            ci_status="SUCCESS",
            review_decision="",
            days_old=1,
            author="alice",
            title="Safe change",
        ),
    )
    calls: list[str] = []

    def fake_launcher(prompt, **_kwargs):
        calls.append(prompt)
        return {"type": "result", "result": "#1 - review now"}

    monkeypatch.setattr(prs, "_resolve_triage_backend", lambda: ("claude-cli", "claude-code-plan"))
    monkeypatch.setattr(claude_cli, "run_claude_cli", fake_launcher)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("prs.py must not launch Claude directly"),
    )
    prs.triage_with_opus([candidate], "main")
    captured = capsys.readouterr()
    assert "review now" in captured.out
    assert "Triage failed" not in captured.err
    assert len(calls) == 1


def test_invalid_model_json_log_does_not_echo_model_or_source_content(capsys):
    from graphify import llm

    secret = "TOP-SECRET from /home/alice/project/source.py"
    assert llm._parse_llm_json(secret) == {"nodes": [], "edges": [], "hyperedges": []}
    error = capsys.readouterr().err
    assert "invalid JSON" in error
    assert "TOP-SECRET" not in error
    assert "/home/alice" not in error
