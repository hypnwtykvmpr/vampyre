"""Executable contract for model-produced semantic graph fragments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, NoReturn, cast

PROMPT_SCHEMA_VERSION = "semantic-v2"
SEMANTIC_NODE_ID_SCHEMA_PROFILE_KEY = "semantic_node_id_schema"
LEGACY_SEMANTIC_NODE_ID_SCHEMA = "legacy-preserved"

FILE_TYPES = ("code", "document", "paper", "image", "rationale", "concept")
EDGE_RELATIONS = (
    "calls",
    "implements",
    "references",
    "cites",
    "conceptually_related_to",
    "shares_data_with",
    "semantically_similar_to",
    "rationale_for",
)
HYPEREDGE_RELATIONS = ("participate_in", "implement", "form")
CONFIDENCE_LEVELS = ("EXTRACTED", "INFERRED", "AMBIGUOUS")
INFERRED_SCORES = (0.95, 0.85, 0.75, 0.65, 0.55)
AMBIGUOUS_SCORES = (0.3, 0.2, 0.1)

NODE_FIELDS = (
    "id",
    "label",
    "file_type",
    "source_file",
    "source_location",
    "source_url",
    "captured_at",
    "author",
    "contributor",
    "rationale",
)
EDGE_FIELDS = (
    "source",
    "target",
    "relation",
    "confidence",
    "confidence_score",
    "source_file",
    "source_location",
    "weight",
    "context",
)
HYPEREDGE_FIELDS = (
    "id",
    "label",
    "nodes",
    "relation",
    "confidence",
    "confidence_score",
    "source_file",
    "source_location",
    "weight",
    "context",
)

_TOP_LEVEL_FIELDS = frozenset({"nodes", "edges", "hyperedges", "input_tokens", "output_tokens"})
_REQUIRED_TOP_LEVEL_FIELDS = frozenset({"nodes", "edges", "hyperedges"})
_REQUIRED_NODE_FIELDS = frozenset({"id", "label", "file_type", "source_file"})
_REQUIRED_EDGE_FIELDS = frozenset(
    {
        "source",
        "target",
        "relation",
        "confidence",
        "confidence_score",
        "source_file",
    }
)
_REQUIRED_HYPEREDGE_FIELDS = frozenset(
    {"id", "label", "nodes", "relation", "confidence", "confidence_score", "source_file"}
)
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class SemanticSchemaError(ValueError):
    """Raised when model output violates the semantic extraction contract."""


def _fail(path: str, reason: str) -> NoReturn:
    raise SemanticSchemaError(f"semantic schema violation at {path}: {reason}")


def _record(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected object")
    return cast(dict[str, Any], value)


def _required_fields(record: dict[str, Any], required: frozenset[str], path: str) -> None:
    missing = sorted(required - record.keys())
    if missing:
        _fail(path, f"missing field {missing[0]}")


def _known_fields(record: dict[str, Any], allowed: tuple[str, ...], path: str) -> None:
    unknown = sorted(record.keys() - set(allowed))
    if unknown:
        _fail(f"{path}.{unknown[0]}", "unknown field")


def _string(value: object, path: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(path, "expected nonempty string")


def _identifier(value: object, path: str) -> None:
    _string(value, path)
    assert isinstance(value, str)
    if _ID_RE.fullmatch(value) is None:
        _fail(path, "expected lowercase ASCII identifier")


def _number(value: object, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        _fail(path, "expected finite number")
    if not isinstance(value, (int, float)):
        _fail(path, "expected finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail(path, "expected finite number")
    if nonnegative and number < 0:
        _fail(path, "expected nonnegative number")
    return number


def _confidence(record: dict[str, Any], path: str, *, hyperedge: bool = False) -> None:
    confidence = record["confidence"]
    allowed = CONFIDENCE_LEVELS[:-1] if hyperedge else CONFIDENCE_LEVELS
    if confidence not in allowed:
        _fail(f"{path}.confidence", "unknown confidence")
    score = _number(record["confidence_score"], f"{path}.confidence_score", nonnegative=True)
    expected = {
        "EXTRACTED": (1.0,),
        "INFERRED": INFERRED_SCORES,
        "AMBIGUOUS": AMBIGUOUS_SCORES,
    }[confidence]
    if score not in expected:
        _fail(f"{path}.confidence_score", f"score does not match {confidence}")


def _optional_text_fields(record: dict[str, Any], fields: tuple[str, ...], path: str) -> None:
    for field in fields:
        if field in record:
            _string(record[field], f"{path}.{field}", nullable=True)


def validate_semantic_fragment(fragment: object) -> dict[str, Any]:
    """Return a defensive copy after strict model-output validation.

    Diagnostics contain only schema paths and stable reason codes; model-provided
    values are never reflected into logs or exception text.
    """
    root = _record(fragment, "root")
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - root.keys())
    if missing:
        _fail("root", f"missing field {missing[0]}")
    unknown = sorted(root.keys() - _TOP_LEVEL_FIELDS)
    if unknown:
        _fail(unknown[0], "unknown field")

    for bucket in ("nodes", "edges", "hyperedges"):
        if not isinstance(root[bucket], list):
            _fail(bucket, "expected array")
    if len(root["hyperedges"]) > 3:
        _fail("hyperedges", "maximum 3 records")
    for token_field in ("input_tokens", "output_tokens"):
        if token_field in root:
            value = root[token_field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(token_field, "expected nonnegative integer")

    for index, raw in enumerate(root["nodes"]):
        path = f"nodes[{index}]"
        record = _record(raw, path)
        _required_fields(record, _REQUIRED_NODE_FIELDS, path)
        _known_fields(record, NODE_FIELDS, path)
        _identifier(record["id"], f"{path}.id")
        _string(record["label"], f"{path}.label")
        if record["file_type"] not in FILE_TYPES:
            _fail(f"{path}.file_type", "unknown file type")
        _string(record["source_file"], f"{path}.source_file")
        _optional_text_fields(
            record,
            (
                "source_location",
                "source_url",
                "captured_at",
                "author",
                "contributor",
                "rationale",
            ),
            path,
        )

    for index, raw in enumerate(root["edges"]):
        path = f"edges[{index}]"
        record = _record(raw, path)
        _required_fields(record, _REQUIRED_EDGE_FIELDS, path)
        _known_fields(record, EDGE_FIELDS, path)
        _identifier(record["source"], f"{path}.source")
        _identifier(record["target"], f"{path}.target")
        if record["relation"] not in EDGE_RELATIONS:
            _fail(f"{path}.relation", "unknown relation")
        _confidence(record, path)
        _string(record["source_file"], f"{path}.source_file")
        _optional_text_fields(record, ("source_location", "context"), path)
        if "weight" in record:
            _number(record["weight"], f"{path}.weight", nonnegative=True)

    for index, raw in enumerate(root["hyperedges"]):
        path = f"hyperedges[{index}]"
        record = _record(raw, path)
        _required_fields(record, _REQUIRED_HYPEREDGE_FIELDS, path)
        _known_fields(record, HYPEREDGE_FIELDS, path)
        _identifier(record["id"], f"{path}.id")
        _string(record["label"], f"{path}.label")
        members = record["nodes"]
        if not isinstance(members, list) or len(members) < 3:
            _fail(f"{path}.nodes", "expected at least 3 node identifiers")
        for member_index, member in enumerate(members):
            _identifier(member, f"{path}.nodes[{member_index}]")
        if len(set(members)) != len(members):
            _fail(f"{path}.nodes", "duplicate node identifier")
        if record["relation"] not in HYPEREDGE_RELATIONS:
            _fail(f"{path}.relation", "unknown relation")
        _confidence(record, path, hyperedge=True)
        _string(record["source_file"], f"{path}.source_file")
        _optional_text_fields(record, ("source_location", "context"), path)
        if "weight" in record:
            _number(record["weight"], f"{path}.weight", nonnegative=True)

    return copy.deepcopy(root)


def validate_semantic_source_paths(
    fragment: Mapping[str, object], allowed_source_files: set[str]
) -> None:
    """Require every model record to name one source actually sent to the model."""
    for bucket in ("nodes", "edges", "hyperedges"):
        records = fragment.get(bucket, [])
        if not isinstance(records, list):
            _fail(bucket, "expected array")
        for index, raw in enumerate(records):
            record = _record(raw, f"{bucket}[{index}]")
            if record.get("source_file") not in allowed_source_files:
                _fail(f"{bucket}[{index}].source_file", "source was not supplied")


def endpoint_digest(endpoint: str) -> str:
    """Return exact endpoint identity without persisting endpoint text."""
    return hashlib.sha256(endpoint.strip().encode("utf-8")).hexdigest()


def semantic_provenance(
    *,
    backend: str,
    model: str,
    endpoint: str,
    deep_mode: bool = False,
    token_budget: int | None = 60_000,
    chunk_size: int = 20,
    max_retry_depth: int = 3,
) -> dict[str, str]:
    return {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "endpoint_digest": endpoint_digest(endpoint),
        "extraction_mode": "deep" if deep_mode else "standard",
        "token_budget": "none" if token_budget is None else str(token_budget),
        "chunk_size": str(chunk_size),
        "max_retry_depth": str(max_retry_depth),
    }


def normalize_semantic_provenance(value: Mapping[str, object] | None) -> dict[str, str]:
    """Normalize cache provenance, using an explicit local-API identity if absent."""
    if value is not None and not isinstance(value, Mapping):
        _fail("identity", "expected object")
    raw = value or {}
    identity = {
        "prompt_schema_version": str(raw.get("prompt_schema_version") or PROMPT_SCHEMA_VERSION),
        "backend": str(raw.get("backend") or "unspecified"),
        "model": str(raw.get("model") or "unspecified"),
        "endpoint_digest": str(raw.get("endpoint_digest") or endpoint_digest("unspecified")),
        "extraction_mode": str(raw.get("extraction_mode") or "standard"),
        "token_budget": str(raw.get("token_budget") or "60000"),
        "chunk_size": str(raw.get("chunk_size") or "20"),
        "max_retry_depth": str(raw.get("max_retry_depth") or "3"),
    }
    if identity["prompt_schema_version"] != PROMPT_SCHEMA_VERSION:
        _fail("identity.prompt_schema_version", "unsupported schema version")
    if _HEX_DIGEST_RE.fullmatch(identity["endpoint_digest"]) is None:
        _fail("identity.endpoint_digest", "expected SHA-256 digest")
    for field in (
        "backend",
        "model",
        "extraction_mode",
        "token_budget",
        "chunk_size",
        "max_retry_depth",
    ):
        _string(identity[field], f"identity.{field}")
    if identity["extraction_mode"] not in {"standard", "deep"}:
        _fail("identity.extraction_mode", "expected standard or deep")
    return identity


def semantic_provenance_digest(value: Mapping[str, object] | None) -> str:
    identity = normalize_semantic_provenance(value)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_semantic_schema() -> str:
    """Render the schema block shared by runtime prompts and generated references."""
    relation_values = "|".join(EDGE_RELATIONS)
    hyperedge_values = "|".join(HYPEREDGE_RELATIONS)
    file_type_values = "|".join(FILE_TYPES)
    inferred_values = ", ".join(str(score) for score in INFERRED_SCORES)
    ambiguous_values = ", ".join(str(score) for score in AMBIGUOUS_SCORES)
    example = {
        "nodes": [
            {
                "id": "auth_session_validatetoken",
                "label": "Validate Token",
                "file_type": file_type_values,
                "source_file": "supplied/path",
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
                "rationale": None,
            }
        ],
        "edges": [
            {
                "source": "node_id",
                "target": "node_id",
                "relation": relation_values,
                "confidence": "EXTRACTED|INFERRED|AMBIGUOUS",
                "confidence_score": 1.0,
                "source_file": "supplied/path",
                "source_location": None,
                "weight": 1.0,
                "context": None,
            }
        ],
        "hyperedges": [
            {
                "id": "snake_case_id",
                "label": "Human Readable Label",
                "nodes": ["node_id1", "node_id2", "node_id3"],
                "relation": hyperedge_values,
                "confidence": "EXTRACTED|INFERRED",
                "confidence_score": 0.75,
                "source_file": "supplied/path",
                "source_location": None,
                "weight": 1.0,
                "context": None,
            }
        ],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    schema_json = json.dumps(example, ensure_ascii=True, separators=(",", ":"))
    rendered = (
        f"Canonical semantic schema `{PROMPT_SCHEMA_VERSION}`:\n"
        "- file_type: exactly `code`, `document`, `paper`, `image`, `rationale`, "
        f"`concept` (`{file_type_values}`).\n"
        f"- edge relation: exactly `{relation_values}`.\n"
        f"- hyperedge relation: exactly `{hyperedge_values}`.\n"
        "- Node and hyperedge IDs are lowercase producer IDs matching `[a-z0-9_]+`. "
        "Use the full supplied path stem plus the entity label; Graphify owns final "
        "canonical AST identity. Never invent chunk or sequence suffixes.\n"
        "  Example: `src/auth/session.py` + `ValidateToken` → "
        "`src_auth_session_validatetoken`.\n"
        "- source_file must copy one supplied source path exactly.\n"
        "- Every edge requires confidence and confidence_score. EXTRACTED uses 1.0; "
        f"INFERRED uses exactly one of {inferred_values}; AMBIGUOUS uses exactly one of "
        f"{ambiguous_values}.\n"
        "- Hyperedges require at least 3 distinct nodes, use only EXTRACTED or INFERRED, "
        "and are limited to 3 per response.\n"
        "- Emit only the listed fields. Unknown fields and relations are rejected.\n"
        f"Output exactly this JSON shape:\n{schema_json}"
    )
    decoded = json.loads(rendered.rsplit("\n", 1)[-1])
    expected_fields = {
        "nodes": set(NODE_FIELDS),
        "edges": set(EDGE_FIELDS),
        "hyperedges": set(HYPEREDGE_FIELDS),
    }
    for bucket, fields in expected_fields.items():
        records = decoded.get(bucket)
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise RuntimeError(f"rendered semantic schema lacks one {bucket} example")
        if set(records[0]) != fields:
            raise RuntimeError(f"rendered semantic schema {bucket} fields drifted")
    if not rendered.strip() or PROMPT_SCHEMA_VERSION not in rendered:
        raise RuntimeError("rendered semantic schema is empty or unversioned")
    return rendered
