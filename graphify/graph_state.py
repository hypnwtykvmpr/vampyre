"""Authoritative validation and realization of serialized Graphify state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast, overload

import networkx as nx

from .edge_identity import strip_schema_key
from .ids import make_id, normalize_id
from .paths import is_absolute_path
from .validate import is_hashable

GRAPHIFY_PROFILE_KEY = "graphify_profile"
CURRENT_SCHEMA_VERSION = "1"
STATE_DIAGNOSTICS_KEY = "graphify_state_diagnostics"
TOP_LEVEL_METADATA_CARRIER = "_graphify_top_level_metadata"

GraphType = Literal["simple", "digraph", "multidigraph"]
NodeId = Hashable


class GraphStateError(ValueError):
    """Serialized graph state cannot be interpreted without data loss."""


class DecodeMode(str, Enum):
    STRICT_CURRENT = "strict-current"
    MIGRATE_LEGACY = "migrate-legacy"
    READ_ONLY_LEGACY = "read-only-legacy"


class CompatibilityStatus(str, Enum):
    CURRENT = "current"
    MIGRATED = "migrated"
    READ_ONLY_LEGACY = "read-only-legacy"


class MetadataPolicy(str, Enum):
    NAMESPACE_CONFLICTS = "namespace-conflicts"
    REFUSE_CONFLICTS = "refuse-conflicts"


@dataclass(frozen=True)
class StateDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class GraphState:
    graph: nx.Graph
    nodes: tuple[dict[str, object], ...]
    edges: tuple[dict[str, object], ...]
    hyperedges: tuple[dict[str, object], ...]
    graph_metadata: Mapping[str, object]
    graph_type: GraphType
    schema_version: str | None
    compatibility_status: CompatibilityStatus
    diagnostics: tuple[StateDiagnostic, ...]
    top_level_metadata: Mapping[str, object]


@dataclass(frozen=True)
class NamedGraphState:
    namespace: str
    state: GraphState
    alias: str | None = None
    apply_namespace: bool = True
    node_remap: Mapping[NodeId, NodeId] = dataclass_field(default_factory=dict)


_OWNED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "directed",
    "multigraph",
    "graph",
    "nodes",
    "links",
    "edges",
    "hyperedges",
    GRAPHIFY_PROFILE_KEY,
    STATE_DIAGNOSTICS_KEY,
}

_HYPEREDGE_MEMBER_FIELDS = ("nodes", "members", "node_ids")

_SCALAR_NODE_REFERENCE_FIELDS = frozenset(
    {"node_id", "caller_nid", "nid", "parent_id", "container_id", "_src", "_tgt"}
)
_LIST_NODE_REFERENCE_FIELDS = frozenset({"node_ids", "source_nodes"})


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GraphStateError(f"state contains a non-JSON value: {exc}") from exc


def _canonicalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _canonicalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    return copy.deepcopy(value)


def _json_order_key(value: object) -> tuple[int, str]:
    if value is None:
        tag = 0
    elif isinstance(value, bool):
        tag = 1
    elif isinstance(value, int):
        tag = 2
    elif isinstance(value, float):
        tag = 3
    elif isinstance(value, str):
        tag = 4
    elif isinstance(value, list):
        tag = 5
    elif isinstance(value, dict):
        tag = 6
    else:
        raise GraphStateError(f"unsupported canonical-order type: {type(value).__name__}")
    return tag, _canonical_json(_canonicalize_json(value))


def _remap_node_id(node_id: NodeId, node_remap: Mapping[Any, Any]) -> NodeId:
    seen: set[NodeId] = set()
    current = node_id
    while current in node_remap:
        if current in seen:
            raise GraphStateError("node remap contains a cycle")
        seen.add(current)
        current = node_remap[current]
        if not is_hashable(current):
            raise GraphStateError("node remap produced an unhashable member")
    return current


def _hyperedge_members(record: Mapping[str, object]) -> list[NodeId]:
    present = [(field, record[field]) for field in _HYPEREDGE_MEMBER_FIELDS if field in record]
    if not present:
        raise GraphStateError("hyperedge is missing a member list")
    normalized: list[list[NodeId]] = []
    for field, value in present:
        if not isinstance(value, list):
            raise GraphStateError(f"hyperedge {field} must be a list")
        members: list[NodeId] = []
        for member in value:
            if member is None or not is_hashable(member):
                raise GraphStateError("hyperedge member must be a non-null hashable JSON value")
            members.append(member)
        normalized.append(members)
    if any(members != normalized[0] for members in normalized[1:]):
        raise GraphStateError("hyperedge member fields conflict")
    return normalized[0]


def reconcile_hyperedges(
    records: Iterable[Mapping[str, object]],
    *,
    live_node_ids: Collection[NodeId],
    node_remap: Mapping[Any, Any] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return canonical, live hyperedges after a node remap or prune."""
    remap = node_remap or {}
    live = set(live_node_ids)
    reconciled: dict[str, tuple[str, dict[str, object]]] = {}

    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise GraphStateError(f"hyperedges[{index}] must be an object")
        record = copy.deepcopy(dict(raw))
        _validate_json_value(record, path=f"hyperedges[{index}]")
        raw_members = _hyperedge_members(record)
        members: list[NodeId] = []
        seen_members: set[NodeId] = set()
        for raw_member in raw_members:
            member = _remap_node_id(raw_member, remap)
            if member not in live or member in seen_members:
                continue
            seen_members.add(member)
            members.append(member)
        members.sort(key=_json_order_key)
        if len(members) < 2:
            continue

        for field in _HYPEREDGE_MEMBER_FIELDS:
            record.pop(field, None)
        record["nodes"] = members
        explicit_id = record.get("id")
        if explicit_id is not None and (not isinstance(explicit_id, str) or not explicit_id):
            raise GraphStateError("hyperedge id must be a non-empty string")
        if explicit_id is None:
            identity_payload = {key: value for key, value in record.items() if key != "id"}
            digest = hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()
            explicit_id = f"hyperedge:v1:{digest}"
            record["id"] = explicit_id

        fingerprint = _canonical_json(record)
        previous = reconciled.get(explicit_id)
        if previous is not None:
            if previous[0] == fingerprint:
                continue
            raise GraphStateError(f"conflicting hyperedge id: {explicit_id!r}")
        reconciled[explicit_id] = (fingerprint, record)

    return tuple(
        record
        for _fingerprint, record in sorted(
            reconciled.values(),
            key=lambda item: (str(item[1]["id"]), item[0]),
        )
    )


def repair_legacy_file_node_hyperedge_aliases(
    nodes: Iterable[Mapping[str, object]],
    hyperedges: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Repair only unambiguous legacy path-derived file-node members."""
    aliases: dict[str, NodeId | None] = {}
    valid_ids: set[NodeId] = set()
    for node in nodes:
        node_id = node.get("id")
        if is_hashable(node_id):
            valid_ids.add(node_id)
        if not isinstance(node_id, str) or node.get("_origin") != "ast":
            continue
        if node.get("source_location") != "L1":
            continue
        source_file = node.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            continue
        relative = Path(source_file)
        if is_absolute_path(source_file) or not relative.name:
            continue
        label = str(node.get("label") or "")
        if label not in {relative.name, relative.as_posix()}:
            continue
        path_alias = make_id(relative.as_posix())
        legacy_path_node_alias = normalize_id(f"{path_alias}_node")
        for alias in (path_alias, legacy_path_node_alias, make_id(path_alias, "node")):
            normalized = normalize_id(alias)
            prior = aliases.get(normalized)
            if prior is None and normalized not in aliases:
                aliases[normalized] = node_id
            elif prior != node_id:
                aliases[normalized] = None

    repaired: list[dict[str, object]] = []
    for hyperedge in hyperedges:
        item = copy.deepcopy(dict(hyperedge))
        members = _hyperedge_members(item)
        remapped_members: list[object] = []
        for member in members:
            replacement = None
            if isinstance(member, str) and member not in valid_ids:
                replacement = aliases.get(normalize_id(member))
            remapped_members.append(replacement if replacement is not None else member)
        for field in _HYPEREDGE_MEMBER_FIELDS:
            item.pop(field, None)
        item["nodes"] = remapped_members
        repaired.append(item)
    return repaired


def _repair_legacy_hyperedge_id_conflicts(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for index, record in enumerate(records):
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id:
            grouped.setdefault(record_id, []).append((index, _canonical_json(record)))
    conflicting_indexes: set[int] = set()
    for entries in grouped.values():
        if len({fingerprint for _index, fingerprint in entries}) > 1:
            conflicting_indexes.update(index for index, _fingerprint in entries)
    if not conflicting_indexes:
        return records, 0
    repaired = copy.deepcopy(records)
    for index in conflicting_indexes:
        repaired[index].pop("id", None)
    return repaired, len(conflicting_indexes)


def _persisted_diagnostics(payload: Mapping[str, object]) -> list[StateDiagnostic]:
    raw = payload.get(STATE_DIAGNOSTICS_KEY, [])
    if not isinstance(raw, list):
        raise GraphStateError(f"{STATE_DIAGNOSTICS_KEY} must be a list")
    diagnostics: list[StateDiagnostic] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise GraphStateError(f"{STATE_DIAGNOSTICS_KEY}[{index}] must be an object")
        code = item.get("code")
        message = item.get("message")
        if not isinstance(code, str) or not code or not isinstance(message, str):
            raise GraphStateError(
                f"{STATE_DIAGNOSTICS_KEY}[{index}] requires non-empty code and string message"
            )
        diagnostics.append(StateDiagnostic(code, message))
    return diagnostics


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphStateError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise GraphStateError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise GraphStateError(f"{path} contains unsupported {type(value).__name__}")


def _profile_candidates(payload: Mapping[str, object]) -> list[dict[str, object]]:
    nested = payload.get("graph")
    nested_profile = nested.get(GRAPHIFY_PROFILE_KEY) if isinstance(nested, dict) else None
    top_profile = payload.get(GRAPHIFY_PROFILE_KEY)
    profiles: list[dict[str, object]] = []
    for profile in (top_profile, nested_profile):
        if profile is None:
            continue
        if not isinstance(profile, dict):
            raise GraphStateError("graphify profile must be an object")
        profiles.append(copy.deepcopy(profile))
    return profiles


def resolve_graph_type_fields(
    payload: Mapping[str, object],
    *,
    require_explicit: bool,
    allow_direction_rescue: bool = False,
) -> GraphType:
    """Resolve class flags/profile under one strict field policy."""
    profiles = _profile_candidates(payload)
    declarations: list[str] = []
    for profile in profiles:
        declared = profile.get("graph_type")
        if declared not in {"simple", "digraph", "multidigraph"}:
            raise GraphStateError("graphify profile is missing a valid graph_type")
        declarations.append(str(declared))
    if len(set(declarations)) > 1:
        raise GraphStateError("top-level and nested graphify profiles conflict")

    flags_present = "multigraph" in payload and "directed" in payload
    multigraph = payload.get("multigraph")
    directed = payload.get("directed")
    for name, value in (("multigraph", multigraph), ("directed", directed)):
        if name in payload and not isinstance(value, bool):
            raise GraphStateError(f"{name} must be a boolean")

    declared_type = declarations[0] if declarations else None
    if flags_present:
        if multigraph is True and directed is False:
            if declared_type == "multidigraph" and allow_direction_rescue:
                return "multidigraph"
            raise GraphStateError("multigraph=true conflicts with directed=false")
        flag_type: GraphType = (
            "multidigraph" if multigraph is True else "digraph" if directed is True else "simple"
        )
        if declared_type is None:
            if require_explicit:
                raise GraphStateError("graphify profile is required")
            return flag_type
        if declared_type != flag_type:
            raise GraphStateError("graphify profile conflicts with class flags")
        return flag_type

    if require_explicit:
        raise GraphStateError("directed and multigraph flags are required")
    if declared_type is not None:
        if "multigraph" in payload:
            declared_multi = declared_type == "multidigraph"
            if multigraph is not declared_multi:
                raise GraphStateError("graphify profile conflicts with multigraph flag")
        if "directed" in payload:
            declared_directed = declared_type in {"digraph", "multidigraph"}
            if directed is not declared_directed:
                if not (
                    allow_direction_rescue and declared_type == "multidigraph" and directed is False
                ):
                    raise GraphStateError("graphify profile conflicts with directed flag")
        return declared_type  # type: ignore[return-value]
    if "multigraph" in payload:
        return "multidigraph" if multigraph is True else "simple"
    if "directed" in payload:
        return "digraph" if directed is True else "simple"
    return "simple"


def _resolve_edge_records(
    payload: Mapping[str, object], *, strict: bool
) -> list[dict[str, object]]:
    present: dict[str, list[dict[str, object]]] = {}
    for field in ("links", "edges"):
        if field not in payload:
            continue
        raw = payload[field]
        if not isinstance(raw, list):
            raise GraphStateError(f"{field} must be a list")
        records: list[dict[str, object]] = []
        for index, record in enumerate(raw):
            if not isinstance(record, dict):
                raise GraphStateError(f"{field}[{index}] must be an object")
            _validate_json_value(record, path=f"{field}[{index}]")
            records.append(copy.deepcopy(record))
        present[field] = records
    if strict and not present:
        raise GraphStateError("current graph state requires a links edge list")
    if len(present) == 2:
        if Counter(map(_canonical_json, present["links"])) != Counter(
            map(_canonical_json, present["edges"])
        ):
            raise GraphStateError("edges and links contain conflicting records")
    return present.get("links", present.get("edges", []))


def _resolve_hyperedges(
    payload: Mapping[str, object],
    graph_metadata: dict[str, object],
    *,
    strict: bool,
    diagnostics: list[StateDiagnostic],
) -> list[dict[str, object]]:
    nested = graph_metadata.get("hyperedges")
    top_present = "hyperedges" in payload
    top = payload.get("hyperedges")
    for label, value in (("top-level hyperedges", top), ("nested hyperedges", nested)):
        if value is not None and not isinstance(value, list):
            raise GraphStateError(f"{label} must be a list")
    if top_present:
        records = top if isinstance(top, list) else []
        if nested is not None and _canonical_json(records) != _canonical_json(nested):
            if strict:
                raise GraphStateError("top-level and nested hyperedges conflict")
            diagnostics.append(
                StateDiagnostic(
                    "legacy-hyperedge-conflict",
                    "top-level hyperedges were authoritative over legacy nested metadata",
                )
            )
    else:
        if strict:
            raise GraphStateError("current graph state requires top-level hyperedges")
        records = nested if isinstance(nested, list) else []
        if nested is not None:
            diagnostics.append(
                StateDiagnostic(
                    "legacy-nested-hyperedges",
                    "migrated hyperedges from nested graph metadata",
                )
            )
    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise GraphStateError(f"hyperedges[{index}] must be an object")
        _validate_json_value(record, path=f"hyperedges[{index}]")
        normalized.append(copy.deepcopy(record))
    graph_metadata.pop("hyperedges", None)
    return normalized


def _node_records(payload: Mapping[str, object]) -> tuple[list[dict[str, object]], set[object]]:
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise GraphStateError("nodes must be a list")
    records: list[dict[str, object]] = []
    node_ids: set[object] = set()
    for index, record in enumerate(raw_nodes):
        if not isinstance(record, dict) or "id" not in record:
            raise GraphStateError(f"nodes[{index}] must be an object with an id")
        _validate_json_value(record, path=f"nodes[{index}]")
        node_id = record["id"]
        if node_id is None or not is_hashable(node_id):
            raise GraphStateError(f"nodes[{index}].id must be a non-null hashable JSON value")
        if node_id in node_ids:
            raise GraphStateError(f"duplicate node id: {node_id!r}")
        node_ids.add(node_id)
        records.append(copy.deepcopy(record))
    return records, node_ids


def _normalize_edges(
    records: list[dict[str, object]],
    *,
    graph_type: GraphType,
    node_ids: set[object],
    strict: bool,
    rescue_direction: bool,
    diagnostics: list[StateDiagnostic],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen: dict[object, str] = {}
    repaired_keys = 0
    repaired_provenance = 0
    for index, raw in enumerate(records):
        record = copy.deepcopy(raw)
        origin = record.get("_origin")
        if "_origin" not in record:
            if strict:
                raise GraphStateError(f"edge {index} is missing provenance")
            record["_origin"] = "legacy"
            repaired_provenance += 1
        elif not isinstance(origin, str) or not origin:
            raise GraphStateError(f"edge {index} provenance must be a non-empty string")
        if "source" in record and "target" in record:
            source = record["source"]
            target = record["target"]
        elif not strict and "from" in record and "to" in record:
            source = record.pop("from")
            target = record.pop("to")
            record["source"] = source
            record["target"] = target
            diagnostics.append(
                StateDiagnostic("legacy-edge-endpoints", "migrated from/to edge endpoints")
            )
        else:
            raise GraphStateError(f"edge {index} is missing source or target")
        if not is_hashable(source) or not is_hashable(target):
            raise GraphStateError(f"edge {index} has an unhashable endpoint")
        if source not in node_ids or target not in node_ids:
            raise GraphStateError(f"edge {index} has a dangling endpoint")
        if rescue_direction:
            rescued_source = record.get("_src")
            rescued_target = record.get("_tgt")
            if {rescued_source, rescued_target} != {source, target}:
                raise GraphStateError(
                    "legacy undirected edge lacks unambiguous _src/_tgt direction"
                )
            source = rescued_source
            target = rescued_target
            record["source"] = source
            record["target"] = target
            record.pop("_src", None)
            record.pop("_tgt", None)
        if graph_type == "multidigraph":
            key, attrs = strip_schema_key(
                {k: v for k, v in record.items() if k not in {"source", "target"}}
            )
            if key is None:
                if strict:
                    raise GraphStateError("current multidigraph edge is missing a key")
                digest = hashlib.sha256(_canonical_json(record).encode()).hexdigest()
                key = f"edge:v1:{digest}"
                repaired_keys += 1
            if key is None or not is_hashable(key):
                raise GraphStateError(
                    "multidigraph edge key must be a non-null hashable JSON value"
                )
            record = {"source": source, "target": target, "key": key, **attrs}
            identity: object = (source, target, key)
        else:
            record = {
                "source": source,
                "target": target,
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"source", "target", "from", "to", "key"}
                },
            }
            endpoint_key = (
                tuple(sorted((source, target), key=lambda value: _canonical_json(value)))
                if graph_type == "simple"
                else (source, target)
            )
            identity = endpoint_key
        fingerprint = _canonical_json(record)
        previous = seen.get(identity)
        if previous is not None:
            if previous == fingerprint:
                continue
            label = (
                "duplicate keyed edge identity"
                if graph_type == "multidigraph"
                else "duplicate edge identity"
            )
            raise GraphStateError(f"{label} has conflicting attributes")
        seen[identity] = fingerprint
        normalized.append(record)
    if repaired_keys:
        diagnostics.append(
            StateDiagnostic(
                "legacy-missing-edge-key",
                f"generated deterministic keys for {repaired_keys} multigraph edge(s)",
            )
        )
    if repaired_provenance:
        diagnostics.append(
            StateDiagnostic(
                "legacy-missing-edge-provenance",
                f"marked {repaired_provenance} edge(s) with explicit legacy provenance",
            )
        )
    return normalized


def decode_graph_state(
    payload: Mapping[str, object],
    *,
    mode: DecodeMode,
) -> GraphState:
    if not isinstance(payload, Mapping):
        raise GraphStateError("serialized graph state must be a JSON object")
    _validate_json_value(dict(payload), path="graph")
    schema_raw = payload.get("schema_version")
    schema_version = str(schema_raw) if schema_raw is not None else None
    current = schema_raw == int(CURRENT_SCHEMA_VERSION) and not isinstance(schema_raw, bool)
    if schema_raw is not None and not current:
        raise GraphStateError(f"unsupported graph schema version: {schema_raw!r}")
    if mode is DecodeMode.STRICT_CURRENT and not current:
        raise GraphStateError("current graph state requires schema_version 1")
    strict = current
    if mode is DecodeMode.MIGRATE_LEGACY and not current:
        profiles_present = bool(_profile_candidates(payload))
        flags_present = "multigraph" in payload and "directed" in payload
        if not profiles_present and not flags_present:
            raise GraphStateError("legacy state has no unambiguous graph class")
    require_explicit = strict
    graph_type = resolve_graph_type_fields(
        payload,
        require_explicit=require_explicit,
        allow_direction_rescue=not current,
    )
    profiles = _profile_candidates(payload)
    if (
        strict
        and len(profiles) == 2
        and _canonical_json(profiles[0]) != _canonical_json(profiles[1])
    ):
        raise GraphStateError("top-level and nested graphify profiles conflict")
    profile = copy.deepcopy(profiles[0]) if profiles else {"graph_type": graph_type}
    profile["graph_type"] = graph_type

    diagnostics = _persisted_diagnostics(payload)
    if not current:
        diagnostics.append(
            StateDiagnostic(
                "legacy-schema",
                "decoded pre-schema graph state; re-encode before stateful use",
            )
        )
    graph_metadata_raw = payload.get("graph", {})
    if not isinstance(graph_metadata_raw, dict):
        raise GraphStateError("graph metadata must be an object")
    graph_metadata = copy.deepcopy(graph_metadata_raw)
    graph_metadata[GRAPHIFY_PROFILE_KEY] = profile
    hyperedges = _resolve_hyperedges(
        payload,
        graph_metadata,
        strict=strict,
        diagnostics=diagnostics,
    )
    nodes, node_ids = _node_records(payload)
    if not strict:
        hyperedges = repair_legacy_file_node_hyperedge_aliases(nodes, hyperedges)
        hyperedges, repaired_conflicts = _repair_legacy_hyperedge_id_conflicts(hyperedges)
        if repaired_conflicts:
            diagnostics.append(
                StateDiagnostic(
                    "legacy-hyperedge-id-conflict",
                    f"assigned content identities to {repaired_conflicts} conflicting hyperedge records",
                )
            )
    if strict:
        for index, record in enumerate(hyperedges):
            members = _hyperedge_members(record)
            if any(member not in node_ids for member in members):
                raise GraphStateError(f"hyperedge {index} has a dangling member")
            if len(dict.fromkeys(members)) < 2:
                raise GraphStateError(f"hyperedge {index} has fewer than two distinct members")
    reconciled_hyperedges = list(reconcile_hyperedges(hyperedges, live_node_ids=node_ids))
    if not strict and _canonical_json(reconciled_hyperedges) != _canonical_json(hyperedges):
        diagnostics.append(
            StateDiagnostic(
                "legacy-hyperedge-repair",
                "canonicalized legacy hyperedge identities and live members",
            )
        )
    hyperedges = reconciled_hyperedges
    edge_records = _resolve_edge_records(payload, strict=strict)
    rescue_direction = (
        not strict
        and payload.get("multigraph") is True
        and payload.get("directed") is False
        and graph_type == "multidigraph"
    )
    edges = _normalize_edges(
        edge_records,
        graph_type=graph_type,
        node_ids=node_ids,
        strict=strict,
        rescue_direction=rescue_direction,
        diagnostics=diagnostics,
    )

    graph_class: type[nx.Graph]
    if graph_type == "multidigraph":
        graph_class = nx.MultiDiGraph
    elif graph_type == "digraph":
        graph_class = nx.DiGraph
    else:
        graph_class = nx.Graph
    graph = graph_class()
    graph.graph.update(copy.deepcopy(graph_metadata))
    graph.graph["hyperedges"] = copy.deepcopy(hyperedges)
    for record in nodes:
        graph.add_node(
            record["id"], **{k: copy.deepcopy(v) for k, v in record.items() if k != "id"}
        )
    for record in edges:
        source = record["source"]
        target = record["target"]
        attrs = {k: copy.deepcopy(v) for k, v in record.items() if k not in {"source", "target"}}
        if graph_type == "multidigraph":
            key, attrs = strip_schema_key(attrs)
            graph.add_edge(source, target, key=key, **attrs)
        else:
            graph.add_edge(source, target, **attrs)

    status = (
        CompatibilityStatus.CURRENT
        if current
        else CompatibilityStatus.MIGRATED
        if mode is DecodeMode.MIGRATE_LEGACY
        else CompatibilityStatus.READ_ONLY_LEGACY
    )
    top_level_metadata = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in _OWNED_TOP_LEVEL_FIELDS
    }
    state = GraphState(
        graph=graph,
        nodes=tuple(nodes),
        edges=tuple(edges),
        hyperedges=tuple(hyperedges),
        graph_metadata=graph_metadata,
        graph_type=graph_type,
        schema_version=schema_version,
        compatibility_status=status,
        diagnostics=tuple(diagnostics),
        top_level_metadata=top_level_metadata,
    )
    validate_graph_state(state)
    return state


def validate_graph_state(state: GraphState) -> None:
    expected_type = {
        "simple": nx.Graph,
        "digraph": nx.DiGraph,
        "multidigraph": nx.MultiDiGraph,
    }[state.graph_type]
    if type(state.graph) is not expected_type:
        raise GraphStateError("live graph class conflicts with graph_type")
    live_nodes = set(state.graph.nodes)
    if live_nodes != {record["id"] for record in state.nodes}:
        raise GraphStateError("live graph nodes conflict with serialized node records")
    for index, edge in enumerate(state.edges):
        if edge.get("source") not in live_nodes or edge.get("target") not in live_nodes:
            raise GraphStateError(f"edge {index} has a dangling endpoint")
    reconciled = reconcile_hyperedges(state.hyperedges, live_node_ids=live_nodes)
    if reconciled != state.hyperedges:
        raise GraphStateError("serialized hyperedges are not canonical and live")
    for diagnostic in state.diagnostics:
        if not diagnostic.code or not isinstance(diagnostic.message, str):
            raise GraphStateError("state diagnostic requires non-empty code and string message")


def _stable_union(values: Sequence[list[object]]) -> list[object]:
    merged: list[object] = []
    seen: set[str] = set()
    for items in values:
        for item in items:
            fingerprint = _canonical_json(item)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(copy.deepcopy(item))
    return merged


def _merge_named_metadata(
    named: Sequence[tuple[str, Mapping[str, object]]],
    *,
    path: str,
    policy: MetadataPolicy,
    conflicts: dict[str, dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    keys = sorted({key for _namespace, values in named for key in values})
    for key in keys:
        claims = [(namespace, values[key]) for namespace, values in named if key in values]
        fingerprints = {_canonical_json(value) for _namespace, value in claims}
        if len(fingerprints) == 1:
            result[key] = copy.deepcopy(claims[0][1])
            continue
        if all(isinstance(value, list) for _namespace, value in claims):
            result[key] = _stable_union([value for _namespace, value in claims])  # type: ignore[list-item]
            continue
        if all(isinstance(value, Mapping) for _namespace, value in claims):
            nested = _merge_named_metadata(
                [(namespace, value) for namespace, value in claims],  # type: ignore[list-item]
                path=f"{path}.{key}",
                policy=policy,
                conflicts=conflicts,
            )
            if nested:
                result[key] = nested
            continue
        conflict_path = f"{path}.{key}"
        if policy is MetadataPolicy.REFUSE_CONFLICTS:
            raise GraphStateError(f"conflicting composition metadata at {conflict_path}")
        # Keep a canonical usable value while preserving every claim. Omitting
        # the field makes repeated composition non-idempotent: the next pass sees
        # only the later input's value and silently restores it.
        result[key] = copy.deepcopy(claims[0][1])
        conflicts[conflict_path] = {namespace: copy.deepcopy(value) for namespace, value in claims}
    return result


def merge_graph_metadata(
    existing: Mapping[str, object],
    generated: Mapping[str, object],
    *,
    target_type: GraphType,
    policy: MetadataPolicy = MetadataPolicy.NAMESPACE_CONFLICTS,
) -> dict[str, object]:
    """Merge graph metadata without allowing stale class markers to win."""
    prepared: list[tuple[str, dict[str, object]]] = []
    for namespace, source in (("existing", existing), ("generated", generated)):
        values = copy.deepcopy(dict(source))
        values.pop("hyperedges", None)
        values.pop(TOP_LEVEL_METADATA_CARRIER, None)
        profile = values.get(GRAPHIFY_PROFILE_KEY)
        if profile is not None and not isinstance(profile, dict):
            raise GraphStateError("graphify profile must be an object")
        if isinstance(profile, dict):
            profile = copy.deepcopy(profile)
            profile.pop("graph_type", None)
            if profile:
                values[GRAPHIFY_PROFILE_KEY] = profile
            else:
                values.pop(GRAPHIFY_PROFILE_KEY, None)
        prepared.append((namespace, values))

    conflicts: dict[str, dict[str, object]] = {}
    merged = _merge_named_metadata(
        prepared,
        path="graph",
        policy=policy,
        conflicts=conflicts,
    )
    profile = merged.get(GRAPHIFY_PROFILE_KEY)
    profile = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    profile["graph_type"] = target_type
    merged[GRAPHIFY_PROFILE_KEY] = profile
    if conflicts:
        merged["graphify_composition_provenance"] = conflicts
    return merged


def _mapped_reference(value: object, node_map: Mapping[NodeId, NodeId]) -> object:
    try:
        return node_map.get(value, value)  # type: ignore[arg-type]
    except TypeError:
        return value


@overload
def _remap_reference_metadata(
    value: dict[str, object],
    node_map: Mapping[NodeId, NodeId],
) -> dict[str, object]: ...


@overload
def _remap_reference_metadata(
    value: list[object],
    node_map: Mapping[NodeId, NodeId],
) -> list[object]: ...


@overload
def _remap_reference_metadata(
    value: object,
    node_map: Mapping[NodeId, NodeId],
) -> object: ...


def _remap_reference_metadata(
    value: object,
    node_map: Mapping[NodeId, NodeId],
) -> object:
    if isinstance(value, list):
        return [_remap_reference_metadata(item, node_map) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    remapped: dict[str, object] = {}
    for key, item in value.items():
        if key in _SCALAR_NODE_REFERENCE_FIELDS:
            remapped[key] = _mapped_reference(item, node_map)
        elif key in _LIST_NODE_REFERENCE_FIELDS and isinstance(item, list):
            remapped[key] = [_mapped_reference(member, node_map) for member in item]
        else:
            remapped[key] = _remap_reference_metadata(item, node_map)
    return remapped


def _namespaced_node_id(namespace: str, node_id: NodeId) -> str:
    suffix = node_id if isinstance(node_id, str) else f"json:{_canonical_json(node_id)}"
    return f"{namespace}::{suffix}"


def _state_already_uses_namespace(item: NamedGraphState, namespace: str) -> bool:
    """Recognize only namespace provenance previously emitted by composition."""
    if not item.apply_namespace or not item.state.nodes:
        return False
    for record in item.state.nodes:
        if record.get("repo") != namespace or "local_id" not in record:
            return False
        local_id = record["local_id"]
        if not is_hashable(local_id):
            return False
        if record["id"] != _namespaced_node_id(namespace, cast(NodeId, local_id)):
            return False
    return True


def _merge_composition_provenance(
    records: Iterable[object],
) -> dict[str, dict[str, object]]:
    """Combine prior conflict evidence without overwriting repeated claims."""
    merged: dict[str, dict[str, object]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        for path, raw_claims in raw.items():
            if not isinstance(path, str) or not isinstance(raw_claims, Mapping):
                continue
            claims = merged.setdefault(path, {})
            for namespace, value in raw_claims.items():
                if not isinstance(namespace, str):
                    continue
                prior = claims.get(namespace)
                if prior is None or _canonical_json(prior) == _canonical_json(value):
                    claims[namespace] = copy.deepcopy(value)
                    continue
                digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
                claims[f"{namespace}:{digest}"] = copy.deepcopy(value)
    return merged


def _effective_namespaces(inputs: Sequence[NamedGraphState]) -> list[str]:
    implicit: dict[str, list[int]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item.namespace, str) or not item.namespace.strip():
            raise GraphStateError("composition namespace must be non-empty")
        implicit.setdefault(item.namespace.strip(), []).append(index)
    for namespace, indexes in implicit.items():
        if len(indexes) > 1 and any(not inputs[index].alias for index in indexes):
            raise GraphStateError(f"duplicate implicit namespace requires aliases: {namespace!r}")
    effective: list[str] = []
    for item in inputs:
        namespace = item.alias.strip() if isinstance(item.alias, str) else item.namespace.strip()
        if not namespace:
            raise GraphStateError("composition alias must be non-empty")
        effective.append(namespace)
    duplicates = sorted(name for name, count in Counter(effective).items() if count > 1)
    if duplicates:
        raise GraphStateError(f"duplicate effective composition namespace: {duplicates!r}")
    return effective


def compose_graph_states(
    inputs: Sequence[NamedGraphState],
    *,
    target_type: GraphType,
    metadata_policy: MetadataPolicy,
) -> GraphState:
    """Compose validated states without a lossy NetworkX round trip."""
    if not inputs:
        raise GraphStateError("composition requires at least one graph state")
    if target_type not in {"simple", "digraph", "multidigraph"}:
        raise GraphStateError(f"unsupported composition target type: {target_type!r}")
    if not isinstance(metadata_policy, MetadataPolicy):
        raise GraphStateError("metadata_policy must be a MetadataPolicy")
    namespaces = _effective_namespaces(inputs)

    for item in inputs:
        validate_graph_state(item.state)
        if target_type != "multidigraph" and item.state.graph_type == "multidigraph":
            message = (
                f"composition projects multigraph input {item.namespace!r} to "
                f"{target_type!r}; parallel edges will be collapsed"
            )
            warnings.warn(message, stacklevel=2)
            print(f"[graphify] WARNING: {message}", file=sys.stderr)

    already_namespaced = [
        _state_already_uses_namespace(item, namespace)
        for item, namespace in zip(inputs, namespaces)
    ]
    node_maps: list[dict[NodeId, NodeId]] = []
    node_claims: dict[NodeId, list[tuple[str, dict[str, object]]]] = {}
    for item, namespace, preserve_namespace in zip(inputs, namespaces, already_namespaced):
        node_map: dict[NodeId, NodeId] = {}
        for record in item.state.nodes:
            node_id = record["id"]
            mapped = item.node_remap.get(
                node_id,
                (
                    node_id
                    if preserve_namespace or not item.apply_namespace
                    else _namespaced_node_id(namespace, node_id)
                ),
            )
            if mapped is None or not is_hashable(mapped):
                raise GraphStateError("composition node remap produced an invalid id")
            if mapped in node_map.values() and node_map.get(node_id) != mapped:
                raise GraphStateError(f"composition node remap collides at {mapped!r}")
            node_map[node_id] = mapped
        node_maps.append(node_map)
        for record in item.state.nodes:
            mapped_id = node_map[record["id"]]
            attrs = {key: value for key, value in record.items() if key != "id"}
            attrs = _remap_reference_metadata(attrs, node_map)  # type: ignore[assignment]
            if (
                item.apply_namespace
                and not preserve_namespace
                and record["id"] not in item.node_remap
            ):
                attrs.setdefault("repo", namespace)
                attrs.setdefault("local_id", copy.deepcopy(record["id"]))
            node_claims.setdefault(mapped_id, []).append((namespace, {"id": mapped_id, **attrs}))

    nodes: list[dict[str, object]] = []
    for node_id in sorted(node_claims, key=_json_order_key):
        claims = node_claims[node_id]
        if len(claims) == 1:
            nodes.append(copy.deepcopy(claims[0][1]))
            continue
        prior_provenance = [
            record.get("graphify_composition_provenance") for _namespace, record in claims
        ]
        conflicts: dict[str, dict[str, object]] = {}
        attrs = _merge_named_metadata(
            [
                (
                    namespace,
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"id", "graphify_composition_provenance"}
                    },
                )
                for namespace, record in claims
            ],
            path=f"node.{_canonical_json(node_id)}",
            policy=metadata_policy,
            conflicts=conflicts,
        )
        provenance = _merge_composition_provenance([*prior_provenance, conflicts])
        if provenance:
            attrs["graphify_composition_provenance"] = provenance
        nodes.append({"id": node_id, **attrs})

    edge_claims: dict[object, list[tuple[str, str, dict[str, object]]]] = {}
    hyperedges: list[dict[str, object]] = []
    for item, namespace, node_map, preserve_namespace in zip(
        inputs, namespaces, node_maps, already_namespaced
    ):
        for record in item.state.edges:
            edge = copy.deepcopy(record)
            source = node_map[edge.pop("source")]
            target = node_map[edge.pop("target")]
            if item.state.graph_type == "simple" and target_type != "simple":
                rescued_source = edge.pop("_src", None)
                rescued_target = edge.pop("_tgt", None)
                if (
                    rescued_source is None
                    or rescued_target is None
                    or {rescued_source, rescued_target} != {record["source"], record["target"]}
                ):
                    raise GraphStateError(
                        "undirected-to-directed composition requires unambiguous _src/_tgt"
                    )
                source = node_map[rescued_source]
                target = node_map[rescued_target]
            elif item.state.graph_type != "simple" and target_type == "simple":
                edge.setdefault("_src", source)
                edge.setdefault("_tgt", target)
            edge = _remap_reference_metadata(edge, node_map)  # type: ignore[assignment]
            if target_type == "multidigraph":
                key = edge.get("key")
                if key is None:
                    key_payload = {"source": source, "target": target, **edge}
                    digest = hashlib.sha256(_canonical_json(key_payload).encode()).hexdigest()
                    key = f"composition:v1:{digest}"
                normalized = {"source": source, "target": target, "key": key, **edge}
                identity: object = (source, target, key)
            else:
                edge.pop("key", None)
                normalized = {"source": source, "target": target, **edge}
                identity = (
                    tuple(sorted((source, target), key=_json_order_key))
                    if target_type == "simple"
                    else (source, target)
                )
            edge_claims.setdefault(identity, []).append(
                (namespace, item.state.graph_type, normalized)
            )

        for record in item.state.hyperedges:
            hyperedge = copy.deepcopy(record)
            raw_id = hyperedge.get("id")
            if item.apply_namespace and not preserve_namespace and isinstance(raw_id, str):
                hyperedge["id"] = f"{namespace}::{raw_id}"
            members = _hyperedge_members(hyperedge)
            for field_name in _HYPEREDGE_MEMBER_FIELDS:
                hyperedge.pop(field_name, None)
            hyperedge["nodes"] = [node_map[member] for member in members]
            hyperedge = _remap_reference_metadata(hyperedge, node_map)  # type: ignore[assignment]
            hyperedges.append(hyperedge)

    edges: list[dict[str, object]] = []
    for identity in sorted(edge_claims, key=lambda value: _canonical_json(value)):
        claims = edge_claims[identity]
        fingerprints = {_canonical_json(record) for _namespace, _source_type, record in claims}
        claim_namespaces = {namespace for namespace, _source_type, _record in claims}
        if len(fingerprints) > 1 and (target_type == "multidigraph" or len(claim_namespaces) > 1):
            label = "keyed edge" if target_type == "multidigraph" else "edge"
            raise GraphStateError(f"conflicting {label} identity: {identity!r}")
        edges.append(
            min(
                (record for _namespace, _source_type, record in claims),
                key=_canonical_json,
            )
        )

    live_node_ids = {record["id"] for record in nodes}
    reconciled_hyperedges = reconcile_hyperedges(hyperedges, live_node_ids=live_node_ids)

    conflicts: dict[str, dict[str, object]] = {}
    graph_metadata_inputs: list[tuple[str, Mapping[str, object]]] = []
    prior_graph_provenance: list[object] = []
    top_metadata_inputs: list[tuple[str, Mapping[str, object]]] = []
    for item, namespace, node_map in zip(inputs, namespaces, node_maps):
        graph_metadata = copy.deepcopy(dict(item.state.graph_metadata))
        prior_graph_provenance.append(graph_metadata.pop("graphify_composition_provenance", None))
        profile = graph_metadata.pop(GRAPHIFY_PROFILE_KEY, {})
        if isinstance(profile, dict):
            profile = {key: value for key, value in profile.items() if key != "graph_type"}
            if profile:
                graph_metadata[GRAPHIFY_PROFILE_KEY] = profile
        graph_metadata_inputs.append(
            (
                namespace,
                _remap_reference_metadata(graph_metadata, node_map),  # type: ignore[arg-type]
            )
        )
        top_metadata_inputs.append(
            (
                namespace,
                _remap_reference_metadata(dict(item.state.top_level_metadata), node_map),  # type: ignore[arg-type]
            )
        )
    graph_metadata = _merge_named_metadata(
        graph_metadata_inputs,
        path="graph",
        policy=metadata_policy,
        conflicts=conflicts,
    )
    top_level_metadata = _merge_named_metadata(
        top_metadata_inputs,
        path="top",
        policy=metadata_policy,
        conflicts=conflicts,
    )
    profile = graph_metadata.get(GRAPHIFY_PROFILE_KEY)
    profile = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    profile["graph_type"] = target_type
    graph_metadata[GRAPHIFY_PROFILE_KEY] = profile
    provenance = _merge_composition_provenance([*prior_graph_provenance, conflicts])
    if provenance:
        graph_metadata["graphify_composition_provenance"] = provenance

    diagnostic_records = {
        (diagnostic.code, diagnostic.message)
        for item in inputs
        for diagnostic in item.state.diagnostics
    }
    if conflicts:
        diagnostic_records.add(
            (
                "composition-metadata-conflict",
                f"preserved {len(conflicts)} metadata conflict(s) by input namespace",
            )
        )
    payload: dict[str, object] = {
        **top_level_metadata,
        "schema_version": int(CURRENT_SCHEMA_VERSION),
        "directed": target_type in {"digraph", "multidigraph"},
        "multigraph": target_type == "multidigraph",
        "graph": graph_metadata,
        "nodes": nodes,
        "links": edges,
        "hyperedges": list(reconciled_hyperedges),
        STATE_DIAGNOSTICS_KEY: [
            {"code": code, "message": message} for code, message in sorted(diagnostic_records)
        ],
    }
    return decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)


_THREE_WAY_MISSING = object()


def _three_way_equal(left: object, right: object) -> bool:
    if left is _THREE_WAY_MISSING or right is _THREE_WAY_MISSING:
        return left is right
    return _canonical_json(left) == _canonical_json(right)


def _copy_three_way(value: object) -> object:
    return _THREE_WAY_MISSING if value is _THREE_WAY_MISSING else copy.deepcopy(value)


def _merge_three_way_value(
    base: object,
    current: object,
    other: object,
    *,
    path: str,
    conflicts: dict[str, dict[str, object]] | None = None,
) -> object:
    if _three_way_equal(current, other):
        return _copy_three_way(current)
    if _three_way_equal(current, base):
        return _copy_three_way(other)
    if _three_way_equal(other, base):
        return _copy_three_way(current)

    if base is _THREE_WAY_MISSING and isinstance(current, Mapping) and isinstance(other, Mapping):
        base = {}
    if base is _THREE_WAY_MISSING:
        if conflicts is not None:
            conflicts[path] = {
                "current": _copy_three_way(current),
                "other": _copy_three_way(other),
            }
            return _copy_three_way(current)
        raise GraphStateError(f"three-way merge has conflicting additions at {path}")
    if current is _THREE_WAY_MISSING or other is _THREE_WAY_MISSING:
        if conflicts is not None:
            conflicts[path] = {
                "current": (
                    {"deleted": True} if current is _THREE_WAY_MISSING else copy.deepcopy(current)
                ),
                "other": (
                    {"deleted": True} if other is _THREE_WAY_MISSING else copy.deepcopy(other)
                ),
            }
            return _copy_three_way(current)
        raise GraphStateError(f"three-way merge has delete/modify conflict at {path}")
    if isinstance(base, Mapping) and isinstance(current, Mapping) and isinstance(other, Mapping):
        merged: dict[str, object] = {}
        keys = sorted(set(base) | set(current) | set(other))
        for key in keys:
            value = _merge_three_way_value(
                base.get(key, _THREE_WAY_MISSING),
                current.get(key, _THREE_WAY_MISSING),
                other.get(key, _THREE_WAY_MISSING),
                path=f"{path}.{key}",
                conflicts=conflicts,
            )
            if value is not _THREE_WAY_MISSING:
                merged[key] = value
        return merged
    if conflicts is not None:
        conflicts[path] = {
            "current": copy.deepcopy(current),
            "other": copy.deepcopy(other),
        }
        return copy.deepcopy(current)
    raise GraphStateError(f"three-way merge has conflicting changes at {path}")


def _state_record_map(
    records: Sequence[Mapping[str, object]],
    *,
    identity,
    label: str,
) -> dict[object, dict[str, object]]:
    mapped: dict[object, dict[str, object]] = {}
    for record in records:
        key = identity(record)
        if key in mapped:
            raise GraphStateError(f"duplicate {label} identity during three-way merge: {key!r}")
        mapped[key] = copy.deepcopy(dict(record))
    return mapped


def _merge_three_way_records(
    base: Sequence[Mapping[str, object]],
    current: Sequence[Mapping[str, object]],
    other: Sequence[Mapping[str, object]],
    *,
    identity,
    label: str,
) -> list[dict[str, object]]:
    base_map = _state_record_map(base, identity=identity, label=label)
    current_map = _state_record_map(current, identity=identity, label=label)
    other_map = _state_record_map(other, identity=identity, label=label)
    merged: list[dict[str, object]] = []
    keys = sorted(set(base_map) | set(current_map) | set(other_map), key=_canonical_json)
    for key in keys:
        value = _merge_three_way_value(
            base_map.get(key, _THREE_WAY_MISSING),
            current_map.get(key, _THREE_WAY_MISSING),
            other_map.get(key, _THREE_WAY_MISSING),
            path=f"{label}.{_canonical_json(key)}",
        )
        if value is _THREE_WAY_MISSING:
            continue
        if not isinstance(value, dict):
            raise GraphStateError(f"three-way merge produced invalid {label} record")
        merged.append(value)
    return merged


def merge_graph_states_three_way(
    base: GraphState,
    current: GraphState,
    other: GraphState,
    *,
    target_type: GraphType | None = None,
) -> GraphState:
    """Merge graph state using Git-style base/current/other semantics."""
    for state in (base, current, other):
        validate_graph_state(state)
    resolved_type = target_type or infer_composition_target_type((base, current, other))

    normalized = [
        compose_graph_states(
            [NamedGraphState(namespace, state, apply_namespace=False)],
            target_type=resolved_type,
            metadata_policy=MetadataPolicy.REFUSE_CONFLICTS,
        )
        for namespace, state in (("base", base), ("current", current), ("other", other))
    ]
    base_state, current_state, other_state = normalized

    def edge_identity(record: Mapping[str, object]) -> object:
        source = record["source"]
        target = record["target"]
        if resolved_type == "multidigraph":
            return source, target, record["key"]
        if resolved_type == "digraph":
            return source, target
        return tuple(sorted((source, target), key=_json_order_key))

    nodes = _merge_three_way_records(
        base_state.nodes,
        current_state.nodes,
        other_state.nodes,
        identity=lambda record: record["id"],
        label="node",
    )
    edges = _merge_three_way_records(
        base_state.edges,
        current_state.edges,
        other_state.edges,
        identity=edge_identity,
        label="edge",
    )
    hyperedges = _merge_three_way_records(
        base_state.hyperedges,
        current_state.hyperedges,
        other_state.hyperedges,
        identity=lambda record: record["id"],
        label="hyperedge",
    )
    base_graph_metadata = copy.deepcopy(dict(base_state.graph_metadata))
    current_graph_metadata = copy.deepcopy(dict(current_state.graph_metadata))
    other_graph_metadata = copy.deepcopy(dict(other_state.graph_metadata))
    prior_graph_provenance = [
        base_graph_metadata.pop("graphify_composition_provenance", None),
        current_graph_metadata.pop("graphify_composition_provenance", None),
        other_graph_metadata.pop("graphify_composition_provenance", None),
    ]
    metadata_conflicts: dict[str, dict[str, object]] = {}
    graph_metadata = _merge_three_way_value(
        base_graph_metadata,
        current_graph_metadata,
        other_graph_metadata,
        path="graph",
        conflicts=metadata_conflicts,
    )
    top_level_metadata = _merge_three_way_value(
        base_state.top_level_metadata,
        current_state.top_level_metadata,
        other_state.top_level_metadata,
        path="top",
        conflicts=metadata_conflicts,
    )
    if not isinstance(graph_metadata, dict) or not isinstance(top_level_metadata, dict):
        raise GraphStateError("three-way merge produced invalid metadata")
    profile = graph_metadata.get(GRAPHIFY_PROFILE_KEY)
    profile = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    profile["graph_type"] = resolved_type
    graph_metadata[GRAPHIFY_PROFILE_KEY] = profile
    provenance = _merge_composition_provenance([*prior_graph_provenance, metadata_conflicts])
    if provenance:
        graph_metadata["graphify_composition_provenance"] = provenance

    diagnostics = sorted(
        {
            (diagnostic.code, diagnostic.message)
            for state in normalized
            for diagnostic in state.diagnostics
        }
    )
    payload: dict[str, object] = {
        **top_level_metadata,
        "schema_version": int(CURRENT_SCHEMA_VERSION),
        "directed": resolved_type in {"digraph", "multidigraph"},
        "multigraph": resolved_type == "multidigraph",
        "graph": graph_metadata,
        "nodes": nodes,
        "links": edges,
        "hyperedges": hyperedges,
        STATE_DIAGNOSTICS_KEY: [
            {"code": code, "message": message} for code, message in diagnostics
        ],
    }
    return decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)


def infer_composition_target_type(inputs: Sequence[GraphState]) -> GraphType:
    """Return the least-lossy common class for a set of validated states."""
    if not inputs:
        raise GraphStateError("composition requires at least one graph state")
    if any(state.graph_type == "multidigraph" for state in inputs):
        return "multidigraph"
    if any(state.graph_type == "digraph" for state in inputs):
        return "digraph"
    return "simple"


def prune_graph_state(state: GraphState, remove_node_ids: Collection[NodeId]) -> GraphState:
    """Return current state with selected nodes and all dangling records removed."""
    validate_graph_state(state)
    removed = set(remove_node_ids)
    if not removed:
        return state
    nodes = [record for record in state.nodes if record["id"] not in removed]
    live = {record["id"] for record in nodes}
    edges = [
        record for record in state.edges if record["source"] in live and record["target"] in live
    ]
    hyperedges = reconcile_hyperedges(state.hyperedges, live_node_ids=live)
    payload = encode_graph_state(state)
    payload["nodes"] = nodes
    payload["links"] = edges
    payload["hyperedges"] = list(hyperedges)
    return decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)


def graph_state_from_graph(
    graph: nx.Graph,
    *,
    top_level_metadata: Mapping[str, object] | None = None,
) -> GraphState:
    """Capture a live NetworkX graph as validated current state without mutating it."""
    graph_type: GraphType
    if graph.is_multigraph():
        graph_type = "multidigraph"
    elif graph.is_directed():
        graph_type = "digraph"
    else:
        graph_type = "simple"

    graph_metadata = copy.deepcopy(dict(graph.graph))
    carried_top = graph_metadata.pop(TOP_LEVEL_METADATA_CARRIER, {})
    if carried_top is not None and not isinstance(carried_top, Mapping):
        raise GraphStateError("top-level metadata carrier must be an object")
    merged_top = copy.deepcopy(dict(carried_top or {}))
    merged_top.update(copy.deepcopy(dict(top_level_metadata or {})))
    hyperedges = graph_metadata.pop("hyperedges", [])
    if not isinstance(hyperedges, list):
        raise GraphStateError("graph hyperedges metadata must be a list")

    nodes = [{"id": node_id, **copy.deepcopy(attrs)} for node_id, attrs in graph.nodes(data=True)]
    hyperedges = repair_legacy_file_node_hyperedge_aliases(nodes, hyperedges)
    hyperedges = list(reconcile_hyperedges(hyperedges, live_node_ids=set(graph.nodes)))
    edges: list[dict[str, object]] = []
    if graph.is_multigraph():
        multigraph = cast(nx.MultiGraph, graph)
        for source, target, key, attrs in multigraph.edges(keys=True, data=True):
            edge_attrs = copy.deepcopy(attrs)
            edge_attrs.setdefault("_origin", "legacy")
            if not graph.is_directed() and source != target:
                rescued_source = edge_attrs.get("_src")
                rescued_target = edge_attrs.get("_tgt")
                if {rescued_source, rescued_target} != {source, target}:
                    raise GraphStateError(
                        "undirected multigraph edge lacks unambiguous _src/_tgt direction"
                    )
                source, target = rescued_source, rescued_target
                edge_attrs.pop("_src", None)
                edge_attrs.pop("_tgt", None)
            edges.append({"source": source, "target": target, "key": key, **edge_attrs})
    else:
        for source, target, attrs in graph.edges(data=True):
            edge_attrs = copy.deepcopy(attrs)
            edge_attrs.setdefault("_origin", "legacy")
            edges.append({"source": source, "target": target, **edge_attrs})

    profile = graph_metadata.get(GRAPHIFY_PROFILE_KEY)
    profile = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    profile["graph_type"] = graph_type
    graph_metadata[GRAPHIFY_PROFILE_KEY] = profile
    payload: dict[str, object] = {
        **merged_top,
        "schema_version": int(CURRENT_SCHEMA_VERSION),
        "directed": graph_type in {"digraph", "multidigraph"},
        "multigraph": graph_type == "multidigraph",
        "graph": graph_metadata,
        "nodes": nodes,
        "links": edges,
        "hyperedges": hyperedges,
    }
    return decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)


def encode_graph_state(state: GraphState) -> dict[str, object]:
    """Return a canonical current-schema payload without mutating ``state``."""
    validate_graph_state(state)
    top_level = copy.deepcopy(dict(state.top_level_metadata))
    collisions = sorted(set(top_level) & _OWNED_TOP_LEVEL_FIELDS)
    if collisions:
        raise GraphStateError(f"top-level metadata conflicts with owned fields: {collisions!r}")

    graph_metadata = copy.deepcopy(dict(state.graph_metadata))
    graph_metadata.pop("hyperedges", None)
    existing_profile = graph_metadata.get(GRAPHIFY_PROFILE_KEY)
    if existing_profile is not None and not isinstance(existing_profile, dict):
        raise GraphStateError("graphify profile must be an object")
    profile = copy.deepcopy(existing_profile) if isinstance(existing_profile, dict) else {}
    profile["graph_type"] = state.graph_type
    graph_metadata[GRAPHIFY_PROFILE_KEY] = profile

    nodes = sorted(
        (copy.deepcopy(record) for record in state.nodes),
        key=lambda record: (_json_order_key(record["id"]), _canonical_json(record)),
    )
    edges = sorted(
        (copy.deepcopy(record) for record in state.edges),
        key=lambda record: (
            _json_order_key(record["source"]),
            _json_order_key(record["target"]),
            _json_order_key(record.get("key")),
            _canonical_json(record),
        ),
    )
    hyperedges = sorted(
        (copy.deepcopy(record) for record in state.hyperedges),
        key=lambda record: (_json_order_key(record["id"]), _canonical_json(record)),
    )
    diagnostics = sorted(
        ({"code": item.code, "message": item.message} for item in state.diagnostics),
        key=lambda item: (_json_order_key(item["code"]), _json_order_key(item["message"])),
    )

    payload: dict[str, object] = {
        **top_level,
        "schema_version": int(CURRENT_SCHEMA_VERSION),
        "directed": state.graph_type in {"digraph", "multidigraph"},
        "multigraph": state.graph_type == "multidigraph",
        "graph": graph_metadata,
        "nodes": nodes,
        "links": edges,
        "hyperedges": hyperedges,
        STATE_DIAGNOSTICS_KEY: diagnostics,
    }
    _validate_json_value(payload, path="graph")
    return _canonicalize_json(payload)  # type: ignore[return-value]


def encode_graph_state_bytes(state: GraphState) -> bytes:
    """Encode canonical graph state as fixed UTF-8 JSON with one trailing newline."""
    payload = encode_graph_state(state)
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def graph_analysis_fingerprint(state: GraphState) -> str:
    """Return the SHA-256 identity of graph content consumed by analysis."""
    payload = encode_graph_state(state)
    analysis_input = {
        "directed": payload["directed"],
        "multigraph": payload["multigraph"],
        "nodes": payload["nodes"],
        "links": payload["links"],
        "hyperedges": payload["hyperedges"],
    }
    return hashlib.sha256(_canonical_json(analysis_input).encode("utf-8")).hexdigest()


def read_graph_state_payload(
    path: str | Path,
    *,
    enforce_size_cap: bool = True,
) -> dict[str, object]:
    """Read one Graphify state payload without interpreting legacy semantics."""
    from .security import check_graph_file_size_cap

    graph_path = Path(path)
    if enforce_size_cap:
        check_graph_file_size_cap(graph_path)
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphStateError(f"cannot read graph state at {graph_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GraphStateError("serialized graph state must be a JSON object")
    return payload


def decode_graph_state_file(
    path: str | Path,
    *,
    mode: DecodeMode,
    enforce_size_cap: bool = True,
) -> GraphState:
    """Read one size-capped graph file through the authoritative decoder."""
    payload = read_graph_state_payload(path, enforce_size_cap=enforce_size_cap)
    return decode_graph_state(payload, mode=mode)
