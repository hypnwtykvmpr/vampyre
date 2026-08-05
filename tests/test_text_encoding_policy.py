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


def test_scanner_detects_ambiguous_text_mode_open_calls() -> None:
    findings = scan_source(
        "def read_file(path):\n"
        "    return path.open()\n"
        "def read_named_file(path):\n"
        "    return path.open('r')\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (2, "open"),
        (4, "open"),
    ]


def test_scanner_detects_io_open_calls_without_encoding() -> None:
    findings = scan_source(
        "import io as text_io\n"
        "from io import open as io_open\n"
        "text_io.open('first.txt')\n"
        "io_open('second.txt', 'r')\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (3, "io.open"),
        (4, "io.open"),
    ]


def test_scanner_accepts_binary_and_explicit_encoding_open_calls() -> None:
    findings = scan_source(
        "import io\n"
        "path.open('rb')\n"
        "path.open(mode='wb')\n"
        "path.open('r', encoding='utf-8')\n"
        "io.open('payload.bin', 'rb')\n"
        "io.open('payload.txt', encoding='utf-8')\n",
        filename="example.py",
    )

    assert findings == []


def test_scanner_accepts_known_non_filesystem_open_calls() -> None:
    findings = scan_source(
        "import os\n"
        "import zipfile\n"
        "def fetch(opener, request):\n"
        "    return opener.open(request, timeout=5)\n"
        "def read_member(archive_path, member):\n"
        "    with zipfile.ZipFile(archive_path) as archive:\n"
        "        return archive.open(member)\n"
        "os.open('payload.bin', os.O_RDONLY)\n",
        filename="example.py",
    )

    assert findings == []


def test_scanner_accepts_assigned_zipfile_receivers() -> None:
    findings = scan_source(
        "import zipfile\n"
        "archive = zipfile.ZipFile(archive_path)\n"
        "archive.open(member)\n"
        "class Reader:\n"
        "    def __init__(self, path):\n"
        "        self.archive = zipfile.ZipFile(path)\n"
        "    def read(self, member):\n"
        "        return self.archive.open(member)\n",
        filename="example.py",
    )

    assert findings == []


def test_scanner_detects_path_rebound_inside_zipfile_context() -> None:
    findings = scan_source(
        "import zipfile\n"
        "from pathlib import Path\n"
        "with zipfile.ZipFile(archive_path) as resource:\n"
        "    resource = Path('plain.txt')\n"
        "    resource.open()\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [(5, "open")]


def test_scanner_detects_zipfile_name_shadowed_by_nested_scopes() -> None:
    findings = scan_source(
        "import zipfile\n"
        "with zipfile.ZipFile(archive_path) as resource:\n"
        "    callback = lambda resource: resource.open()\n"
        "    handles = [resource.open() for resource in resources]\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (3, "open"),
        (4, "open"),
    ]


def test_scanner_classifies_module_open_functions_by_signature() -> None:
    findings = scan_source(
        "import codecs\n"
        "import gzip\n"
        "import tarfile\n"
        "import shelve\n"
        "codecs.open('text.txt', 'r', 'utf-8')\n"
        "gzip.open('payload.gz', 'rb')\n"
        "gzip.open('payload.gz', 'rt', encoding='utf-8')\n"
        "tarfile.open('archive.tar', encoding='utf-8')\n"
        "shelve.open('records')\n",
        filename="example.py",
    )

    assert findings == []


def test_scanner_rejects_text_module_open_without_encoding() -> None:
    findings = scan_source(
        "import codecs as text_codecs\n"
        "import tarfile\n"
        "from gzip import open as gzip_open\n"
        "text_codecs.open('first.txt', 'r')\n"
        "gzip_open('second.gz', mode)\n"
        "tarfile.open('archive.tar')\n",
        filename="example.py",
    )

    assert [(finding.line, finding.method) for finding in findings] == [
        (4, "codecs.open"),
        (5, "gzip.open"),
        (6, "tarfile.open"),
    ]


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
