"""Typed registry for node-reference fields carried by extraction producers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

NodeId = Hashable


@dataclass
class ExtractionState:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    hyperedges: list[dict] = field(default_factory=list)
    raw_calls: list[dict] = field(default_factory=list)
    swift_extensions: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class IdChannelAdapter:
    name: str
    collect: Callable[[ExtractionState], Iterable[NodeId]]
    remap: Callable[[ExtractionState, Mapping[Any, Any]], None]


def _lookup(remap: Mapping[Any, Any], value: Any) -> Any:
    try:
        return remap.get(value, value)
    except TypeError:
        return value


def _field_adapter(name: str, collection: str, fields: tuple[str, ...]) -> IdChannelAdapter:
    def collect(state: ExtractionState) -> Iterable[NodeId]:
        for record in getattr(state, collection):
            for field_name in fields:
                value = record.get(field_name)
                if value is not None:
                    yield value

    def remap(state: ExtractionState, mapping: Mapping[Any, Any]) -> None:
        for record in getattr(state, collection):
            for field_name in fields:
                if field_name in record:
                    record[field_name] = _lookup(mapping, record[field_name])

    return IdChannelAdapter(name=name, collect=collect, remap=remap)


def _collect_hyperedge_members(state: ExtractionState) -> Iterable[NodeId]:
    for record in state.hyperedges:
        for field_name in ("nodes", "members", "node_ids"):
            members = record.get(field_name)
            if isinstance(members, list):
                yield from members


def _remap_hyperedge_members(state: ExtractionState, remap: Mapping[Any, Any]) -> None:
    for record in state.hyperedges:
        for field_name in ("nodes", "members", "node_ids"):
            members = record.get(field_name)
            if isinstance(members, list):
                record[field_name] = [_lookup(remap, member) for member in members]


ID_CHANNELS: tuple[IdChannelAdapter, ...] = (
    _field_adapter("nodes", "nodes", ("id",)),
    _field_adapter("edges", "edges", ("source", "target")),
    IdChannelAdapter(
        name="hyperedges",
        collect=_collect_hyperedge_members,
        remap=_remap_hyperedge_members,
    ),
    _field_adapter("raw_calls", "raw_calls", ("caller_nid",)),
    _field_adapter("swift_extensions", "swift_extensions", ("nid",)),
)

_ADAPTER_BY_NAME = {adapter.name: adapter for adapter in ID_CHANNELS}
_CLASSIFIED_SIDE_CHANNEL_ID_FIELDS = {
    ("raw_calls", "caller_nid"),
    ("swift_extensions", "nid"),
}
SIDE_CHANNEL_CLASSIFICATIONS = {
    "raw_calls": "caller_nid is a node reference; callee and receiver fields are labels",
    "swift_extensions": "nid is a node reference; label is descriptive text",
    "swift_type_table": "path is a source path; table keys and values are type names",
    "ts_type_table": "path is a source path; table keys and values are type names",
    "cpp_type_table": "path is a source path; table keys and values are type names",
    "csharp_type_table": "path is a source path; table keys and values are type names",
    "objc_type_table": "path is a source path; table keys and values are type names",
}


def apply_id_remap(
    state: ExtractionState,
    remap: Mapping[Any, Any],
    *,
    channels: Iterable[str] | None = None,
) -> None:
    """Apply ``remap`` only to declared node-reference channels."""
    if not remap:
        return
    adapters = (
        ID_CHANNELS
        if channels is None
        else tuple(_ADAPTER_BY_NAME[channel] for channel in channels)
    )
    for adapter in adapters:
        adapter.remap(state, remap)


def apply_edge_reference_remap(
    edge: dict,
    *,
    source_remap: Mapping[Any, Any],
    target_remap: Mapping[Any, Any],
) -> None:
    """Apply independently selected source/target remaps to one edge record."""
    if "source" in edge:
        edge["source"] = _lookup(source_remap, edge["source"])
    if "target" in edge:
        edge["target"] = _lookup(target_remap, edge["target"])


def collect_unresolved_references(state: ExtractionState) -> dict[str, set[NodeId]]:
    """Capture references already unresolved before a documented later resolver."""
    live = {node.get("id") for node in state.nodes if node.get("id") is not None}
    unresolved: dict[str, set[NodeId]] = {}
    for adapter in ID_CHANNELS:
        if adapter.name == "nodes":
            continue
        for reference in adapter.collect(state):
            try:
                is_live = reference in live
            except TypeError:
                is_live = False
            if not is_live:
                try:
                    unresolved.setdefault(adapter.name, set()).add(reference)
                except TypeError:
                    continue
    return unresolved


def assert_registered_references_live(
    state: ExtractionState,
    *,
    allowed_unresolved: Mapping[str, set[NodeId]] | None = None,
) -> None:
    """Reject dangling references introduced by an ID-remap phase."""
    allowed = allowed_unresolved or {}
    live = {node.get("id") for node in state.nodes if node.get("id") is not None}
    for adapter in ID_CHANNELS:
        if adapter.name == "nodes":
            continue
        dangling = []
        for reference in adapter.collect(state):
            try:
                is_live = reference in live
                was_unresolved = reference in allowed.get(adapter.name, set())
            except TypeError:
                is_live = False
                was_unresolved = False
            if not is_live and not was_unresolved:
                dangling.append(reference)
        if dangling:
            sample = sorted((repr(value) for value in dangling))[:5]
            raise ValueError(
                f"registered ID channel {adapter.name!r} contains dangling references: {sample}"
            )


def assert_no_unclassified_id_fields(results: Sequence[Mapping[str, object]]) -> None:
    """Reject newly emitted side-channel fields that look like node references."""
    for result_index, result in enumerate(results):
        for channel_name, channel_value in result.items():
            if channel_name in {"nodes", "edges", "hyperedges"}:
                continue
            records: list[Mapping[str, object]] = []
            if isinstance(channel_value, list):
                records = [item for item in channel_value if isinstance(item, Mapping)]
            elif isinstance(channel_value, Mapping):
                records = [channel_value]
            for record_index, record in enumerate(records):
                for field_name in record:
                    looks_like_node_id = field_name == "nid" or field_name.endswith("_nid")
                    if (
                        looks_like_node_id
                        and (
                            channel_name,
                            field_name,
                        )
                        not in _CLASSIFIED_SIDE_CHANNEL_ID_FIELDS
                    ):
                        raise ValueError(
                            "unclassified ID-bearing producer field: "
                            f"result[{result_index}].{channel_name}[{record_index}].{field_name}"
                        )
