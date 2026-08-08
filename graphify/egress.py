"""Fail-closed source egress decisions for semantic extraction."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graphify.persistence import atomic_write_bytes
from graphify.semantic_schema import PROMPT_SCHEMA_VERSION, endpoint_digest

_SENSITIVE_DIRECTORIES = frozenset(
    {".ssh", ".gnupg", ".aws", ".gcloud", "secrets", ".secrets", "credentials"}
)
_KEY_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".cert", ".crt", ".der", ".p8"})
_CODE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".m",
        ".mm",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_CREDENTIAL_WORDS = frozenset(
    {
        "credential",
        "credentials",
        "creds",
        "password",
        "passwords",
        "passwd",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)
_CONFIG_WORDS = frozenset({"config", "configuration", "store", "vault"})
_SPECIAL_NAMES = frozenset(
    {
        ".htpasswd",
        ".netrc",
        ".pgpass",
        "aws_credentials",
        "gcloud_credentials",
        "kubeconfig",
        "service-account",
        "service_account",
    }
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_KNOWN_CREDENTIALS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
)
_ASSIGNMENT_HEAD = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[\"']?([A-Za-z_][A-Za-z0-9_-]*)[\"']?"
    r"[ \t]*[:=][ \t]*"
)
_CREDENTIAL_IDENTIFIER = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd|private[_-]?key|secret|token)",
    re.IGNORECASE,
)
_PLACEHOLDER_SUBSTRINGS = (
    "changeme",
    "dummy",
    "example",
    "fake",
    "placeholder",
    "redacted",
    "replace",
    "your-",
    "your_",
)
_PLACEHOLDER_TOKENS = frozenset({"test"})
_TRIPLE_QUOTE_DELIMITERS = ('"""', "'''")
_COLLECTION_DELIMITERS = {"[": "]", "{": "}", "(": ")"}
_HEREDOC_START = re.compile(r"^<<[-~]?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[ \t]*(?:#.*)?$")


@dataclass(frozen=True)
class EgressDecision:
    eligible: bool
    reason_code: str
    safe_relative_path: str
    content_digest: str
    backend: str
    endpoint_class: str
    endpoint_digest: str
    model: str
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def credential_path_reason(path: Path) -> str | None:
    """Return a stable reason for high-confidence credential paths."""
    candidate = Path(path)
    parent_parts = [part.casefold() for part in candidate.parts[:-1]]
    if any(part in _SENSITIVE_DIRECTORIES for part in parent_parts):
        return "credential_directory"

    name = candidate.name.casefold()
    if re.match(r"^\.(?:env|envrc)(?:\.|$)", name):
        return "credential_dotfile"
    if candidate.suffix.casefold() in _KEY_SUFFIXES:
        return "credential_key_file"
    if name in _SPECIAL_NAMES or any(name.startswith(f"{special}.") for special in _SPECIAL_NAMES):
        return "credential_filename"

    base = name.lstrip(".")
    if "." in base:
        base = base.rsplit(".", 1)[0]
    if base in _SPECIAL_NAMES:
        return "credential_filename"
    if candidate.suffix.casefold() in _CODE_SUFFIXES:
        return None
    if "private_key" in base or "private-key" in base:
        return "credential_filename"
    words = [word for word in re.split(r"[._\-\s]+", base) if word]
    if not words:
        return None
    if words[-1] in _CREDENTIAL_WORDS:
        return "credential_filename"
    if words[0] in _CREDENTIAL_WORDS and any(word in _CONFIG_WORDS for word in words[1:]):
        return "credential_filename"
    return None


def _looks_like_secret_value(value: str) -> bool:
    folded = value.casefold()
    placeholder_tokens = set(re.split(r"[^a-z0-9]+", folded))
    if (
        any(marker in folded for marker in _PLACEHOLDER_SUBSTRINGS)
        or placeholder_tokens & _PLACEHOLDER_TOKENS
    ):
        return False
    if "_" in value and re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        return False
    if folded.startswith(("env.", "os.", "process.", "request.", "self.", "user.")):
        return False
    classes = sum(
        (
            any(char.islower() for char in value),
            any(char.isupper() for char in value),
            any(char.isdigit() for char in value),
            any(not char.isalnum() for char in value),
        )
    )
    return len(value) >= 12 and classes >= 3


def _inline_assignment_value(raw: str) -> str:
    """Return one assignment value without consuming another source line."""
    candidate = raw.lstrip(" \t")
    if not candidate:
        return ""
    if candidate[0] in {'"', "'"}:
        quote = candidate[0]
        value: list[str] = []
        index = 1
        while index < len(candidate):
            char = candidate[index]
            if char == "\\" and index + 1 < len(candidate):
                value.append(candidate[index + 1])
                index += 2
                continue
            if char == quote and quote == "'" and candidate[index : index + 2] == "''":
                value.append(quote)
                index += 2
                continue
            if char == quote:
                break
            value.append(char)
            index += 1
        return "".join(value)
    return candidate.split(maxsplit=1)[0]


def _leading_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _candidate_values_from_line(line: str) -> Iterable[str]:
    """Yield scalar candidates from one owned continuation line."""
    stripped = line.strip()
    if stripped.startswith(("- ", "? ")):
        stripped = stripped[2:].lstrip()
    if not stripped:
        return
    token: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if quote is not None:
            if char == "\\" and index + 1 < len(stripped):
                token.append(stripped[index + 1])
                index += 2
                continue
            if char == quote:
                if quote == "'" and stripped[index : index + 2] == "''":
                    token.append(quote)
                    index += 2
                    continue
                if token:
                    yield "".join(token)
                    token.clear()
                quote = None
                index += 1
                continue
            token.append(char)
            index += 1
            continue
        if char in {'"', "'"}:
            if token:
                yield "".join(token)
                token.clear()
            quote = char
        elif char.isspace() or char in ",[]{}()":
            if token:
                yield "".join(token)
                token.clear()
        else:
            token.append(char)
        index += 1
    if token:
        yield "".join(token)


def _indented_continuation_values(
    lines: list[str],
    line_number: int,
    owner_indent: int,
    *,
    allow_unindented_first: bool = False,
) -> Iterable[str]:
    """Yield values owned by a YAML/JSON-style indented continuation."""
    for continuation in lines[line_number + 1 :]:
        if not continuation.strip():
            continue
        if _leading_indent(continuation) <= owner_indent:
            if allow_unindented_first:
                yield from _candidate_values_from_line(continuation)
            break
        yield from _candidate_values_from_line(continuation)


def _triple_quoted_values(
    raw_value: str, lines: list[str], line_number: int, delimiter: str
) -> Iterable[str]:
    """Yield candidates inside one TOML-style multiline quoted value."""
    remainder = raw_value.lstrip()[len(delimiter) :]
    value, closed, _tail = remainder.partition(delimiter)
    yield from _candidate_values_from_line(value)
    if closed:
        return
    for continuation in lines[line_number + 1 :]:
        value, closed, _tail = continuation.partition(delimiter)
        yield from _candidate_values_from_line(value)
        if closed:
            return


def _delimited_values(
    raw_value: str,
    lines: list[str],
    line_number: int,
    opening: str,
    closing: str,
) -> Iterable[str]:
    """Yield candidates from a multiline bracketed collection."""
    collection_lines = [raw_value.lstrip()[len(opening) :], *lines[line_number + 1 :]]
    depth = 1
    quote: str | None = None
    escaped = False
    for collection_line in collection_lines:
        value: list[str] = []
        for char in collection_line:
            if quote is not None:
                value.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
                value.append(char)
            elif char == opening:
                depth += 1
                value.append(char)
            elif char == closing:
                depth -= 1
                if depth == 0:
                    yield from _candidate_values_from_line("".join(value))
                    return
                value.append(char)
            else:
                value.append(char)
        # Scan incrementally so a truncated or malformed collection cannot hide
        # a credential that appeared before its missing outer delimiter.
        yield from _candidate_values_from_line("".join(value))


def _heredoc_values(lines: list[str], line_number: int, delimiter: str) -> Iterable[str]:
    """Yield candidates until a shell/HCL-style heredoc terminator."""
    for continuation in lines[line_number + 1 :]:
        if continuation.strip() == delimiter:
            return
        yield from _candidate_values_from_line(continuation)


def _credential_assignment_values(text: str) -> Iterable[str]:
    """Yield inline and continued values owned by credential keys."""
    lines = text.splitlines()
    for line_number, line in enumerate(lines):
        for match in _ASSIGNMENT_HEAD.finditer(line):
            if _CREDENTIAL_IDENTIFIER.search(match.group(1)) is None:
                continue
            raw_value = line[match.end() :]
            stripped_value = raw_value.lstrip()
            triple_delimiter = next(
                (
                    delimiter
                    for delimiter in _TRIPLE_QUOTE_DELIMITERS
                    if stripped_value.startswith(delimiter)
                ),
                None,
            )
            if triple_delimiter is not None:
                yield from _triple_quoted_values(
                    raw_value,
                    lines,
                    line_number,
                    triple_delimiter,
                )
                continue
            heredoc = _HEREDOC_START.fullmatch(stripped_value)
            if heredoc is not None:
                yield from _heredoc_values(lines, line_number, heredoc.group(2))
                continue
            collection_closing = _COLLECTION_DELIMITERS.get(stripped_value[:1])
            if collection_closing is not None:
                yield from _delimited_values(
                    raw_value,
                    lines,
                    line_number,
                    stripped_value[0],
                    collection_closing,
                )
                continue
            inline_value = _inline_assignment_value(raw_value)
            if inline_value:
                yield inline_value
            yield from _indented_continuation_values(
                lines,
                line_number,
                _leading_indent(line),
                allow_unindented_first=(not stripped_value or stripped_value.startswith("#")),
            )


def credential_content_reason(content: bytes) -> str | None:
    """Return a stable reason when exact outbound bytes contain credentials."""
    text = content.decode("utf-8", errors="ignore")
    if _PRIVATE_KEY.search(text):
        return "private_key_material"
    if any(pattern.search(text) for pattern in _KNOWN_CREDENTIALS):
        return "provider_credential"
    if any(_looks_like_secret_value(value) for value in _credential_assignment_values(text)):
        return "credential_assignment"
    return None


def classify_endpoint(
    endpoint: str,
    *,
    resolver: Callable[..., list[tuple[Any, ...]]] | None = None,
) -> str:
    """Classify an outbound endpoint from its resolved address set."""
    if endpoint.startswith("managed://"):
        return "external_managed"
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        return "external_unresolved"
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None and literal.is_unspecified:
        return "external_wildcard"
    if literal is not None:
        addresses = {literal}
    else:
        active_resolver = resolver or socket.getaddrinfo
        try:
            rows = active_resolver(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except OSError:
            return "external_unresolved"
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for row in rows:
            try:
                addresses.add(ipaddress.ip_address(str(row[4][0]).split("%", 1)[0]))
            except (IndexError, TypeError, ValueError):
                continue
    if not addresses:
        return "external_unresolved"
    if all(address.is_loopback for address in addresses):
        return "local_loopback"
    if any(address.is_loopback for address in addresses):
        return "external_mixed"
    if all(address.is_private for address in addresses):
        return "external_private"
    if all(address.is_global for address in addresses):
        return "external_public"
    return "external_mixed"


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return "<outside-root>"


def decide_egress(
    path: Path,
    *,
    root: Path,
    content: bytes,
    backend: str,
    endpoint: str,
    model: str,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    endpoint_class: str | None = None,
    precomputed_content_digest: str | None = None,
) -> EgressDecision:
    safe_path = _safe_relative_path(Path(path), Path(root))
    path_reason = credential_path_reason(Path(path))
    content_reason = credential_content_reason(content)
    resolved_class = endpoint_class or classify_endpoint(endpoint)
    parsed = urlparse(endpoint)
    endpoint_reason: str | None = None
    if resolved_class in {"external_wildcard", "external_mixed", "external_unresolved"}:
        endpoint_reason = "unsafe_endpoint_class"
    elif resolved_class != "external_managed" and parsed.scheme not in {"http", "https"}:
        endpoint_reason = "unsafe_endpoint_scheme"
    elif resolved_class != "local_loopback" and parsed.scheme == "http":
        endpoint_reason = "plaintext_remote_endpoint"
    reason = (
        path_reason
        or ("credential_content" if content_reason else None)
        or endpoint_reason
        or "eligible"
    )
    digest = precomputed_content_digest or hashlib.sha256(content).hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("precomputed content digest must be a SHA-256 hex digest")
    return EgressDecision(
        eligible=reason == "eligible" and safe_path != "<outside-root>",
        reason_code="outside_root" if safe_path == "<outside-root>" else reason,
        safe_relative_path=safe_path,
        content_digest=digest,
        backend=backend,
        endpoint_class=resolved_class,
        endpoint_digest=endpoint_digest(endpoint),
        model=model,
        prompt_schema_version=prompt_schema_version,
    )


def write_egress_manifest(path: Path, decisions: Iterable[EgressDecision | dict[str, Any]]) -> None:
    """Atomically write deterministic content-free egress provenance."""
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for decision in decisions:
        row = decision.to_dict() if isinstance(decision, EgressDecision) else dict(decision)
        key = tuple(row.get(field) for field in EgressDecision.__dataclass_fields__)
        rows[key] = row
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row.get("safe_relative_path", ""),
            row.get("backend", ""),
            row.get("model", ""),
            row.get("content_digest", ""),
        ),
    )
    payload = json.dumps(
        {"schema_version": 1, "decisions": ordered},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_bytes(Path(path), (payload + "\n").encode("utf-8"))
