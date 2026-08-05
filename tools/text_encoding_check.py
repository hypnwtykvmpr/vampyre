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
PATH_OPEN_KEYWORDS = frozenset({"mode", "buffering", "encoding", "errors", "newline"})
NON_TEXT_OPEN_MODULES = frozenset({"shelve", "webbrowser"})


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


@dataclass(frozen=True)
class _OpenFunctionSpec:
    method: str
    mode_position: int
    encoding_position: int | None
    default_mode: str


OPEN_FUNCTION_SPECS = {
    "bz2": _OpenFunctionSpec("bz2.open", 1, 3, "rb"),
    "codecs": _OpenFunctionSpec("codecs.open", 1, 2, "r"),
    "gzip": _OpenFunctionSpec("gzip.open", 1, 3, "rb"),
    "io": _OpenFunctionSpec("io.open", 1, 3, "r"),
    "lzma": _OpenFunctionSpec("lzma.open", 1, None, "rb"),
    "tarfile": _OpenFunctionSpec("tarfile.open", 1, None, "r"),
}


def _uses_text_mode(call: ast.Call, *, positional_index: int, default: str = "r") -> bool:
    mode: ast.expr = ast.Constant(default)
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
            break
    else:
        if len(call.args) > positional_index:
            mode = call.args[positional_index]
    return not (
        isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value
    )


def _has_encoding(call: ast.Call, *, positional_index: int | None = None) -> bool:
    if any(keyword.arg == "encoding" for keyword in call.keywords):
        return True
    return positional_index is not None and len(call.args) > positional_index


def _looks_like_open_mode(value: ast.expr) -> bool:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return True
    mode = value.value
    return bool(mode) and set(mode) <= set("rwaxbt+") and sum(c in mode for c in "rwax") == 1


def _receiver_key(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _receiver_key(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _is_zipfile_constructor(
    node: ast.expr,
    *,
    modules: set[str],
    classes: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in modules
        and function.attr == "ZipFile"
    ) or (isinstance(function, ast.Name) and function.id in classes)


def _assignment_targets(node: ast.AST) -> Iterable[ast.expr]:
    if isinstance(node, ast.Assign):
        yield from node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        yield node.target


class _ZipFileBindingVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, *, modules: set[str], classes: set[str]) -> None:
        self.modules = modules
        self.classes = classes
        self.open_call_ids: set[int] = set()
        self._bindings: list[dict[str, bool]] = [{}]
        states: dict[str, set[bool]] = {}
        for node in ast.walk(tree):
            value = getattr(node, "value", None)
            for target in _assignment_targets(node):
                key = _receiver_key(target)
                if key is None or "." not in key:
                    continue
                states.setdefault(key, set()).add(
                    _is_zipfile_constructor(value, modules=modules, classes=classes)
                )
        self._stable_attributes = {key for key, values in states.items() if values == {True}}

    @property
    def _current(self) -> dict[str, bool]:
        return self._bindings[-1]

    def _bind(self, target: ast.expr, *, is_zipfile: bool) -> None:
        key = _receiver_key(target)
        if key is not None:
            self._current[key] = is_zipfile
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind(item, is_zipfile=False)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        is_zipfile = _is_zipfile_constructor(node.value, modules=self.modules, classes=self.classes)
        for target in node.targets:
            self._bind(target, is_zipfile=is_zipfile)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._bind(
            node.target,
            is_zipfile=node.value is not None
            and _is_zipfile_constructor(node.value, modules=self.modules, classes=self.classes),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind(node.target, is_zipfile=False)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(
            node.target,
            is_zipfile=_is_zipfile_constructor(
                node.value, modules=self.modules, classes=self.classes
            ),
        )

    def visit_With(self, node: ast.With) -> None:
        saved: list[tuple[str, bool | None]] = []
        for item in node.items:
            self.visit(item.context_expr)
            target = item.optional_vars
            if target is None:
                continue
            key = _receiver_key(target)
            if key is not None:
                saved.append((key, self._current.get(key)))
            self._bind(
                target,
                is_zipfile=_is_zipfile_constructor(
                    item.context_expr, modules=self.modules, classes=self.classes
                ),
            )
        for statement in node.body:
            self.visit(statement)
        for key, previous in reversed(saved):
            if previous is None:
                self._current.pop(key, None)
            else:
                self._current[key] = previous

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._bindings.append(self._current.copy())
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._current[argument.arg] = False
        if node.args.vararg is not None:
            self._current[node.args.vararg.arg] = False
        if node.args.kwarg is not None:
            self._current[node.args.kwarg.arg] = False
        for statement in node.body:
            self.visit(statement)
        self._bindings.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._bindings.append(self._current.copy())
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._current[argument.arg] = False
        if node.args.vararg is not None:
            self._current[node.args.vararg.arg] = False
        if node.args.kwarg is not None:
            self._current[node.args.kwarg.arg] = False
        self.visit(node.body)
        self._bindings.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._bindings.append(self._current.copy())
        for statement in node.body:
            self.visit(statement)
        self._bindings.pop()

    def _visit_comprehension(
        self,
        generators: Sequence[ast.comprehension],
        results: Sequence[ast.expr],
    ) -> None:
        if not generators:
            for result in results:
                self.visit(result)
            return
        self.visit(generators[0].iter)
        self._bindings.append(self._current.copy())
        for index, generator in enumerate(generators):
            if index:
                self.visit(generator.iter)
            self._bind(generator.target, is_zipfile=False)
            for condition in generator.ifs:
                self.visit(condition)
        for result in results:
            self.visit(result)
        self._bindings.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            key = _receiver_key(node.func.value)
            if key is not None and (
                self._current.get(key, False) or key in self._stable_attributes
            ):
                self.open_call_ids.add(id(node))
        self.generic_visit(node)


def scan_source(source: str, *, filename: str) -> list[Finding]:
    """Return implicit-encoding ``Path`` calls found in Python source."""
    tree = ast.parse(source, filename=filename)
    subprocess_modules: set[str] = set()
    subprocess_functions: dict[str, str] = {}
    os_modules: set[str] = set()
    os_functions: dict[str, str] = {}
    open_modules: dict[str, _OpenFunctionSpec] = {}
    open_functions: dict[str, _OpenFunctionSpec] = {}
    non_text_open_modules: set[str] = set()
    zipfile_modules: set[str] = set()
    zipfile_classes: set[str] = set()
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_modules.add(alias.asname or alias.name)
                elif alias.name == "os":
                    os_modules.add(alias.asname or alias.name)
                elif alias.name == "zipfile":
                    zipfile_modules.add(alias.asname or alias.name)
                elif alias.name in OPEN_FUNCTION_SPECS:
                    open_modules[alias.asname or alias.name] = OPEN_FUNCTION_SPECS[alias.name]
                elif alias.name in NON_TEXT_OPEN_MODULES:
                    non_text_open_modules.add(alias.asname or alias.name)
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
        elif isinstance(node, ast.ImportFrom) and node.module in OPEN_FUNCTION_SPECS:
            for alias in node.names:
                if alias.name == "open":
                    open_functions[alias.asname or alias.name] = OPEN_FUNCTION_SPECS[node.module]
        elif isinstance(node, ast.ImportFrom) and node.module == "zipfile":
            for alias in node.names:
                if alias.name == "ZipFile":
                    zipfile_classes.add(alias.asname or alias.name)

    zipfile_visitor = _ZipFileBindingVisitor(
        tree,
        modules=zipfile_modules,
        classes=zipfile_classes,
    )
    zipfile_visitor.visit(tree)

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
        has_encoding = _has_encoding(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in TEXT_METHODS:
            if not has_encoding:
                findings.append(Finding(filename, node.lineno, node.col_offset + 1, node.func.attr))
            continue
        if isinstance(node.func, ast.Name) and node.func.id in open_functions:
            spec = open_functions[node.func.id]
            if not _has_encoding(node, positional_index=spec.encoding_position) and _uses_text_mode(
                node,
                positional_index=spec.mode_position,
                default=spec.default_mode,
            ):
                findings.append(Finding(filename, node.lineno, node.col_offset + 1, spec.method))
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in open_modules:
                spec = open_modules[receiver.id]
                if not _has_encoding(
                    node, positional_index=spec.encoding_position
                ) and _uses_text_mode(
                    node,
                    positional_index=spec.mode_position,
                    default=spec.default_mode,
                ):
                    findings.append(
                        Finding(filename, node.lineno, node.col_offset + 1, spec.method)
                    )
                continue
            if isinstance(receiver, ast.Name) and (
                receiver.id in os_modules or receiver.id in non_text_open_modules
            ):
                continue
            if id(node) in zipfile_visitor.open_call_ids:
                continue
            if _is_zipfile_constructor(receiver, modules=zipfile_modules, classes=zipfile_classes):
                continue
            if (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id in zipfile_modules
                and receiver.attr == "ZipFile"
            ) or (isinstance(receiver, ast.Name) and receiver.id in zipfile_classes):
                continue
            if any(
                keyword.arg is not None and keyword.arg not in PATH_OPEN_KEYWORDS
                for keyword in node.keywords
            ):
                continue
            if node.args and not _looks_like_open_mode(node.args[0]):
                continue
            if not has_encoding and _uses_text_mode(node, positional_index=0):
                findings.append(Finding(filename, node.lineno, node.col_offset + 1, "open"))
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
