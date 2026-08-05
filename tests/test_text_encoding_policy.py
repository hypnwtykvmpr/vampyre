"""Regression coverage for locale-independent tracked Python text I/O."""

from __future__ import annotations

from pathlib import Path

from tools.text_encoding_check import scan_paths, scan_source


ROOT = Path(__file__).resolve().parents[1]


def test_scanner_detects_compound_path_calls_without_encoding() -> None:
    findings = scan_source(
        '(Path("root") / "child.txt").read_text()\n'
        'tmp_path.joinpath("out.txt").write_text("content")\n',
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (1, "read_text"),
        (2, "write_text"),
    ]


def test_scanner_accepts_explicit_encodings() -> None:
    findings = scan_source(
        'path.read_text(encoding="utf-8")\n'
        'path.write_text("content", encoding="utf-8", errors="strict")\n',
        filename="example.py",
    )

    assert findings == []


def test_scanner_detects_textual_subprocess_calls_without_encoding() -> None:
    findings = scan_source(
        "import subprocess as sp\n"
        "from subprocess import Popen as process\n"
        'sp.run(["tool"], capture_output=True, text=True)\n'
        'process(["tool"], universal_newlines=True)\n',
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (3, "subprocess.run"),
        (4, "subprocess.Popen"),
    ]


def test_scanner_accepts_explicit_subprocess_encoding() -> None:
    findings = scan_source(
        'import subprocess\nsubprocess.run(["tool"], text=True, encoding="utf-8")\n',
        filename="example.py",
    )

    assert findings == []


def test_scanner_rejects_locale_only_process_apis_and_indirection() -> None:
    findings = scan_source(
        "import os as operating_system\n"
        "import subprocess as sp\n"
        "from subprocess import getoutput as shell_output\n"
        "runner = sp.run\n"
        'shell_output("tool")\n'
        'sp.getstatusoutput("tool")\n'
        'operating_system.popen("tool")\n'
        'runner(["tool"], text=True)\n'
        'sp.run(["tool"], **options)\n',
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (5, "subprocess.getoutput"),
        (6, "subprocess.getstatusoutput"),
        (7, "os.popen"),
        (8, "subprocess.run"),
        (9, "subprocess.run.**kwargs"),
    ]


def test_scanner_rejects_subprocess_star_import() -> None:
    findings = scan_source(
        "from subprocess import *\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (1, "subprocess.*"),
    ]


def test_scanner_ignores_unrelated_text_mode_apis() -> None:
    findings = scan_source(
        'runner.run(["tool"], text=True)\n',
        filename="example.py",
    )

    assert findings == []


def test_tracked_python_text_io_is_locale_independent() -> None:
    findings = scan_paths([ROOT / "graphify", ROOT / "tools", ROOT / "tests", ROOT / "conftest.py"])

    assert findings == []
