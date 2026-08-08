"""Strict state resolution for incremental graph updates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .graph_state import (
    DecodeMode,
    GraphStateError,
    decode_graph_state,
    read_graph_state_payload,
)
from .ids import AST_NODE_ID_SCHEMA_PROFILE_KEY, CURRENT_AST_NODE_ID_SCHEMA
from .paths import GRAPHIFY_OUT, resolve_scan_root_marker, write_scan_root_marker
from .semantic_schema import (
    LEGACY_SEMANTIC_NODE_ID_SCHEMA,
    PROMPT_SCHEMA_VERSION,
    SEMANTIC_NODE_ID_SCHEMA_PROFILE_KEY,
)

GraphType = Literal["simple", "digraph", "multidigraph"]


class UpdateStateError(ValueError):
    """Raised when an update target cannot be resolved without guessing."""


@dataclass(frozen=True)
class UpdateContext:
    """All paths and profile state needed by an incremental update."""

    scan_root: Path
    output_root: Path
    output_dir: Path
    graph_path: Path
    manifest_path: Path
    graph_type: GraphType

    @property
    def multigraph(self) -> bool:
        return self.graph_type == "multidigraph"

    @property
    def directed(self) -> bool:
        return self.graph_type in {"digraph", "multidigraph"}


@dataclass(frozen=True)
class IdentitySchemas:
    """Producer-specific identity contracts declared by serialized state."""

    ast: str | None
    semantic: str | None


def _identity_schemas(data: object) -> IdentitySchemas:
    """Read identity declarations without treating one producer as another."""
    if not isinstance(data, dict):
        raise UpdateStateError("graph.json must contain a JSON object")
    nested = data.get("graph")
    profiles = [data.get("graphify_profile")]
    if isinstance(nested, dict):
        profiles.append(nested.get("graphify_profile"))
    profiles = [profile for profile in profiles if profile is not None]
    if not profiles or any(not isinstance(profile, dict) for profile in profiles):
        raise UpdateStateError("graph.json is missing a valid explicit graphify profile")

    def _declaration(key: str, label: str) -> str | None:
        values = [profile.get(key) for profile in profiles if profile.get(key) is not None]
        if len({repr(value) for value in values}) > 1:
            raise UpdateStateError(f"top-level and nested {label} node-ID schemas conflict")
        value = values[0] if values else None
        if value is not None and not isinstance(value, str):
            raise UpdateStateError(f"{label} node-ID schema must be a string")
        return value

    return IdentitySchemas(
        ast=_declaration(AST_NODE_ID_SCHEMA_PROFILE_KEY, "AST"),
        semantic=_declaration(SEMANTIC_NODE_ID_SCHEMA_PROFILE_KEY, "semantic"),
    )


def require_identity_schemas(
    data: object,
    *,
    allow_unmarked_legacy_ast_ids: bool = False,
    require_current_semantic: bool = False,
) -> IdentitySchemas:
    """Validate producer schemas for the mutation a caller intends to perform."""
    schemas = _identity_schemas(data)
    if schemas.ast is None and not allow_unmarked_legacy_ast_ids:
        raise UpdateStateError(
            "graph.json predates the current AST node-ID schema; run ordinary "
            "`graphify update` to migrate deterministic IDs and preserved references"
        )
    if schemas.ast not in (None, CURRENT_AST_NODE_ID_SCHEMA):
        raise UpdateStateError(
            f"unsupported AST node-ID schema {schemas.ast!r}; upgrade Graphify before "
            "modifying this graph"
        )
    if schemas.semantic not in (
        None,
        PROMPT_SCHEMA_VERSION,
        LEGACY_SEMANTIC_NODE_ID_SCHEMA,
    ):
        raise UpdateStateError(
            f"unsupported semantic node-ID schema {schemas.semantic!r}; upgrade Graphify "
            "before modifying this graph"
        )
    if require_current_semantic and schemas.semantic != PROMPT_SCHEMA_VERSION:
        raise UpdateStateError(
            "graph.json contains unmarked or legacy semantic identity; run "
            "`graphify update --remap` before incremental semantic extraction"
        )
    return schemas


def _output_dir(output_root: Path) -> Path:
    configured = Path(GRAPHIFY_OUT)
    return (
        configured.resolve() if configured.is_absolute() else (output_root / configured).resolve()
    )


def _output_root_from_dir(output_dir: Path) -> Path:
    """Recover the storage root from a concrete configured output directory."""
    configured = Path(GRAPHIFY_OUT)
    if configured.is_absolute():
        return output_dir.parent
    parts = tuple(part for part in configured.parts if part not in ("", "."))
    candidate = output_dir
    if parts and tuple(candidate.parts[-len(parts) :]) == parts:
        for _ in parts:
            candidate = candidate.parent
        return candidate
    return output_dir.parent


def _strict_graph_type(data: object) -> GraphType:
    if not isinstance(data, dict):
        raise UpdateStateError("graph.json must contain a JSON object")
    nested = data.get("graph")
    profile = data.get("graphify_profile")
    if profile is None and isinstance(nested, dict):
        profile = nested.get("graphify_profile")
    if not isinstance(profile, dict) or profile.get("graph_type") not in {
        "simple",
        "digraph",
        "multidigraph",
    }:
        raise UpdateStateError("graph.json is missing a valid explicit graphify profile")
    try:
        return decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY).graph_type
    except GraphStateError as exc:
        message = str(exc)
        if message in {
            "graphify profile is required",
            "graphify profile is missing a valid graph_type",
        }:
            message = "graph.json is missing a valid explicit graphify profile"
        raise UpdateStateError(message) from exc


def _graph_type_from_file(graph_path: Path) -> GraphType:
    try:
        return _strict_graph_type(read_graph_state_payload(graph_path))
    except GraphStateError as exc:
        message = str(exc)
        if message in {
            "graphify profile is required",
            "graphify profile is missing a valid graph_type",
        }:
            message = "graph.json is missing a valid explicit graphify profile"
        raise UpdateStateError(message) from exc


def resolve_update_context(
    scan_path: Path | None = None,
    *,
    output_root: Path | None = None,
    output_dir: Path | None = None,
    cwd: Path | None = None,
    allow_unmarked_legacy_ast_ids: bool = False,
) -> UpdateContext:
    """Resolve an existing graph update without creating or guessing state."""
    working_dir = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    explicit_scan = Path(scan_path).resolve() if scan_path is not None else None
    if explicit_scan is not None and not explicit_scan.is_dir():
        raise UpdateStateError(f"scan path is not a directory: {explicit_scan}")

    if output_root is not None and output_dir is not None:
        raise UpdateStateError("output root and output directory cannot both be specified")
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = working_dir / resolved_output_dir
        resolved_output_dir = resolved_output_dir.resolve()
        resolved_output_root = _output_root_from_dir(resolved_output_dir)
    else:
        if output_root is not None:
            resolved_output_root = Path(output_root).resolve()
        elif explicit_scan is not None:
            resolved_output_root = explicit_scan
        else:
            resolved_output_root = working_dir
        resolved_output_dir = _output_dir(resolved_output_root)
    graph_path = resolved_output_dir / "graph.json"
    marker_path = resolved_output_dir / ".graphify_root"
    manifest_path = resolved_output_dir / "manifest.json"

    if not graph_path.is_file():
        raise UpdateStateError(
            f"no existing graph state at {graph_path}; run graphify extract <path> "
            "with --multigraph, --directed, or --simple first"
        )
    if not marker_path.is_file():
        raise UpdateStateError(
            f"existing graph state at {graph_path} has no scan-root marker; "
            "re-run graphify extract with the intended graph profile"
        )

    marker_scan = resolve_scan_root_marker(marker_path, cwd=working_dir)
    if marker_scan is None or not marker_scan.is_dir():
        raise UpdateStateError(f"scan-root marker is unreadable or invalid: {marker_path}")
    if explicit_scan is not None and marker_scan != explicit_scan:
        raise UpdateStateError(
            f"scan path {explicit_scan} conflicts with the graph marker {marker_scan}"
        )

    try:
        graph_payload = read_graph_state_payload(graph_path)
        graph_type = _strict_graph_type(graph_payload)
        require_identity_schemas(
            graph_payload,
            allow_unmarked_legacy_ast_ids=allow_unmarked_legacy_ast_ids,
        )
    except UpdateStateError:
        raise
    except (GraphStateError, OSError, UnicodeError, ValueError) as exc:
        raise UpdateStateError(f"existing graph state is unreadable: {exc}") from exc

    return UpdateContext(
        scan_root=marker_scan,
        output_root=resolved_output_root,
        output_dir=resolved_output_dir,
        graph_path=graph_path,
        manifest_path=manifest_path,
        graph_type=graph_type,
    )


def repair_update_state(
    scan_path: Path,
    *,
    output_root: Path | None = None,
    output_dir: Path | None = None,
    cwd: Path | None = None,
) -> UpdateContext:
    """Restore only a missing scan-root marker after validating existing state."""
    working_dir = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    scan_root = Path(scan_path).resolve()
    if not scan_root.is_dir():
        raise UpdateStateError(f"scan path is not a directory: {scan_root}")
    if output_root is not None and output_dir is not None:
        raise UpdateStateError("output root and output directory cannot both be specified")
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = working_dir / resolved_output_dir
        resolved_output_dir = resolved_output_dir.resolve()
        resolved_output_root = _output_root_from_dir(resolved_output_dir)
    else:
        resolved_output_root = Path(output_root).resolve() if output_root is not None else scan_root
        resolved_output_dir = _output_dir(resolved_output_root)

    graph_path = resolved_output_dir / "graph.json"
    manifest_path = resolved_output_dir / "manifest.json"
    marker_path = resolved_output_dir / ".graphify_root"
    if not graph_path.is_file():
        raise UpdateStateError(f"no existing graph state at {graph_path}")
    if not manifest_path.is_file():
        raise UpdateStateError(f"existing graph state has no manifest: {manifest_path}")

    try:
        graph_type = _graph_type_from_file(graph_path)
    except (OSError, UnicodeError, UpdateStateError, ValueError) as exc:
        raise UpdateStateError(f"existing graph state is unreadable: {exc}") from exc

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdateStateError(f"existing manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest:
        raise UpdateStateError("existing manifest must be a non-empty JSON object")

    matched_live_file = False
    mismatched_live_files: list[str] = []
    for raw_path, entry in manifest.items():
        if not isinstance(raw_path, str) or not raw_path or not isinstance(entry, dict):
            raise UpdateStateError("existing manifest contains a malformed entry")
        stored_path = Path(raw_path)
        candidate = stored_path if stored_path.is_absolute() else scan_root / stored_path
        try:
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(scan_root)
        except (OSError, ValueError) as exc:
            raise UpdateStateError(
                f"manifest path escapes the requested scan root: {raw_path}"
            ) from exc
        if not resolved_candidate.is_file():
            continue
        hashes = {
            value
            for key in ("ast_hash", "semantic_hash", "hash")
            if isinstance((value := entry.get(key)), str) and value
        }
        if not hashes:
            mismatched_live_files.append(raw_path)
            continue
        digest = hashlib.md5(
            resolved_candidate.read_bytes(),
            usedforsecurity=False,
        ).hexdigest()
        if digest in hashes:
            matched_live_file = True
        else:
            mismatched_live_files.append(raw_path)
    if mismatched_live_files:
        preview = ", ".join(mismatched_live_files[:3])
        if len(mismatched_live_files) > 3:
            preview += f", and {len(mismatched_live_files) - 3} more"
        raise UpdateStateError(
            "manifest content does not match the requested scan root: " + preview
        )
    if not matched_live_file:
        raise UpdateStateError(
            "manifest does not contain a live content-hash match under the requested scan root"
        )

    if marker_path.exists():
        marker_scan = resolve_scan_root_marker(marker_path, cwd=working_dir)
        if marker_scan is None:
            raise UpdateStateError(f"scan-root marker is unreadable or invalid: {marker_path}")
        if marker_scan != scan_root:
            raise UpdateStateError(
                f"scan path {scan_root} conflicts with the graph marker {marker_scan}"
            )
    else:
        write_scan_root_marker(marker_path, scan_root)

    return UpdateContext(
        scan_root=scan_root,
        output_root=resolved_output_root,
        output_dir=resolved_output_dir,
        graph_path=graph_path,
        manifest_path=manifest_path,
        graph_type=graph_type,
    )
