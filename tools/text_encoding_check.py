"""Reject ``Path`` text I/O that relies on the process locale."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ROOTS = ("graphify", "tools", "tests", "conftest.py")
EXCLUDED_PARTS = frozenset({".venv", "venv", "build", "dist", "__pycache__"})
TEXT_METHODS = frozenset({"read_text", "write_text"})
SUBPROCESS_METHODS = frozenset({"call", "check_call", "check_output", "Popen", "run"})
LOCALE_ONLY_SUBPROCESS_METHODS = frozenset({"getoutput", "getstatusoutput"})
LOCALE_ONLY_OS_METHODS = frozenset({"popen"})


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    column: int
    method: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: {self.method}() uses locale-dependent text I/O"
        )


def scan_source(source: str, *, filename: str) -> list[Finding]:
    """Return implicit-encoding ``Path`` calls found in Python source."""
    tree = ast.parse(source, filename=filename)
    subprocess_modules: set[str] = set()
    subprocess_functions: dict[str, str] = {}
    os_modules: set[str] = set()
    os_functions: dict[str, str] = {}
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_modules.add(alias.asname or alias.name)
                elif alias.name == "os":
                    os_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                if alias.name == "*":
                    findings.append(
                        Finding(filename, node.lineno, node.col_offset + 1, "subprocess.*")
                    )
                elif alias.name in SUBPROCESS_METHODS | LOCALE_ONLY_SUBPROCESS_METHODS:
                    subprocess_functions[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in LOCALE_ONLY_OS_METHODS:
                    os_functions[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if (
            not isinstance(value, ast.Attribute)
            or not isinstance(value.value, ast.Name)
            or value.value.id not in subprocess_modules
            or value.attr not in SUBPROCESS_METHODS | LOCALE_ONLY_SUBPROCESS_METHODS
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                subprocess_functions[target.id] = value.attr

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        has_encoding = any(keyword.arg == "encoding" for keyword in node.keywords)
        if isinstance(node.func, ast.Attribute) and node.func.attr in TEXT_METHODS:
            if not has_encoding:
                findings.append(Finding(filename, node.lineno, node.col_offset + 1, node.func.attr))
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in subprocess_modules
        ):
            namespace = "subprocess"
            method = node.func.attr
        elif isinstance(node.func, ast.Name) and node.func.id in subprocess_functions:
            namespace = "subprocess"
            method = subprocess_functions[node.func.id]
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in os_modules
        ):
            namespace = "os"
            method = node.func.attr
        elif isinstance(node.func, ast.Name) and node.func.id in os_functions:
            namespace = "os"
            method = os_functions[node.func.id]
        else:
            continue
        if namespace == "subprocess" and method in LOCALE_ONLY_SUBPROCESS_METHODS:
            findings.append(
                Finding(filename, node.lineno, node.col_offset + 1, f"subprocess.{method}")
            )
            continue
        if namespace == "os" and method in LOCALE_ONLY_OS_METHODS:
            findings.append(Finding(filename, node.lineno, node.col_offset + 1, f"os.{method}"))
            continue
        if method not in SUBPROCESS_METHODS:
            continue
        if any(keyword.arg is None for keyword in node.keywords) and not has_encoding:
            findings.append(
                Finding(
                    filename,
                    node.lineno,
                    node.col_offset + 1,
                    f"subprocess.{method}.**kwargs",
                )
            )
            continue
        if has_encoding:
            continue
        text_keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg in {"text", "universal_newlines"}
        }
        if not text_keywords:
            continue
        if all(
            isinstance(value, ast.Constant) and not value.value for value in text_keywords.values()
        ):
            continue
        findings.append(Finding(filename, node.lineno, node.col_offset + 1, f"subprocess.{method}"))
    return sorted(findings)


def _python_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    for candidate in sorted(path.rglob("*.py")):
        if not EXCLUDED_PARTS.intersection(candidate.parts):
            yield candidate


def scan_paths(paths: Iterable[Path], *, display_root: Path | None = None) -> list[Finding]:
    """Scan Python files below paths, reporting names relative to display_root."""
    root = display_root.resolve() if display_root is not None else None
    findings: list[Finding] = []
    for supplied in paths:
        for path in _python_files(supplied):
            resolved = path.resolve()
            if root is not None:
                try:
                    filename = resolved.relative_to(root).as_posix()
                except ValueError:
                    filename = resolved.as_posix()
            else:
                filename = resolved.as_posix()
            source = path.read_text(encoding="utf-8")
            findings.extend(scan_source(source, filename=filename))
    return sorted(findings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--check", action="store_true", help="fail when findings exist")
    args = parser.parse_args(argv)

    repo_root = Path.cwd().resolve()
    paths = [repo_root / item for item in args.roots]
    missing = [item for item, path in zip(args.roots, paths) if not path.exists()]
    if missing:
        parser.error(f"missing scan roots: {', '.join(missing)}")

    findings = scan_paths(paths, display_root=repo_root)
    for finding in findings:
        print(finding)
    print(f"text encoding check: {len(findings)} finding(s)")
    return 1 if args.check and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
