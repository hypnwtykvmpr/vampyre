"""Projection helpers for graph consumers that need explicit edge semantics."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Hashable, Iterable
from typing import Any, Literal, TypedDict, cast

import networkx as nx

WeightMode = Literal["confidence", "count", "sum"]


class RelationshipEnvelope(TypedDict):
    """Stable schema for one endpoint pair's complete relationship bundle."""

    count: int
    shown: list[dict[str, Any]]
    truncated: int
    relations: list[str]
    confidences: list[str]


# Stable default for the capped relationship display envelope. Surfaces (CLI text,
# MCP structured arrays) may override per call, but this is the canonical default
# so "A relates to B through N relationships" renders consistently everywhere.
DEFAULT_RELATIONSHIP_CAP = 3

_CONFIDENCE_SCORE = {
    "EXTRACTED": 1.0,
    "INFERRED": 0.5,
    "AMBIGUOUS": 0.2,
}


def _confidence_score(data: dict[str, Any]) -> float:
    raw_score = data.get("confidence_score")
    if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):  # Python 3.10+
        return float(raw_score)
    raw_confidence = data.get("confidence")
    if isinstance(raw_confidence, str):
        return _CONFIDENCE_SCORE.get(raw_confidence.upper(), 0.0)
    return 0.0


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return repr(value)


def stable_value_key(value: Any) -> tuple[str, str]:
    """Return a total, type-aware key for JSON-safe graph values."""
    value_type = type(value)
    return (f"{value_type.__module__}.{value_type.__qualname__}", _canonical_json(value))


def _edge_sort_key(data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_confidence_score(data),
        stable_value_key(data.get("relation", "")),
        stable_value_key(data.get("source_file", "")),
        stable_value_key(data.get("source_location", "")),
        stable_value_key(data.get("context", "")),
        stable_value_key(data.get("key")),
        _canonical_json(data),
    )


def _iter_edge_data(G: nx.Graph) -> Iterable[tuple[Any, Any, Any, dict[str, Any]]]:
    if isinstance(G, nx.MultiGraph | nx.MultiDiGraph):  # Python 3.10+
        yield from G.edges(keys=True, data=True)
        return
    for u, v, data in G.edges(data=True):
        yield u, v, None, data


def _copy_graph_skeleton(G: nx.Graph, graph_type: type[nx.Graph]) -> nx.Graph:
    H = graph_type()
    H.graph.update(G.graph)
    H.add_nodes_from((node, attrs.copy()) for node, attrs in G.nodes(data=True))
    return H


def _unordered_pair(u: Any, v: Any) -> tuple[Any, Any]:
    if stable_value_key(u) <= stable_value_key(v):
        return u, v
    return v, u


def _merged_edge_attrs(records: list[dict[str, Any]], weight_mode: WeightMode) -> dict[str, Any]:
    if weight_mode not in ("confidence", "count", "sum"):
        raise ValueError("weight_mode must be one of: confidence, count, sum")
    sorted_records = sorted(records, key=_edge_sort_key)
    representative = sorted_records[0].copy()
    scores = [_confidence_score(record) for record in records]
    if weight_mode == "confidence":
        weight = max(scores, default=0.0)
    elif weight_mode == "count":
        weight = float(len(records))
    else:
        weight = float(sum(scores))
    representative["weight"] = weight
    representative["parallel_edge_count"] = len(records)
    return representative


def project_for_community(G: nx.Graph, *, weight_mode: WeightMode = "confidence") -> nx.Graph:
    """Return a simple undirected projection for clustering and community metrics."""
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for u, v, _key, data in _iter_edge_data(G):
        if u == v:
            continue
        pair = _unordered_pair(u, v)
        groups.setdefault(pair, []).append(dict(data))

    H = _copy_graph_skeleton(G, nx.Graph)
    for (u, v), records in sorted(
        groups.items(),
        key=lambda item: (stable_value_key(item[0][0]), stable_value_key(item[0][1])),
    ):
        H.add_edge(u, v, **_merged_edge_attrs(records, weight_mode))
    return H


def project_for_path(G: nx.Graph) -> nx.Graph:
    """Return a simple undirected topology projection for path search."""
    return project_for_community(G, weight_mode="count")


def project_for_callflow(
    G: nx.Graph,
    *,
    relations: frozenset[str] | set[str] | None = None,
) -> nx.DiGraph:
    """Return a simple directed projection for callflow-style consumers."""
    relation_filter = set(relations) if relations is not None else None
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for u, v, _key, data in _iter_edge_data(G):
        relation = data.get("relation")
        # Guard against non-string `relation`; relation_filter is set[str], and
        # an unhashable relation would TypeError on the `in` membership test.
        if relation_filter is not None and (
            not isinstance(relation, str) or relation not in relation_filter
        ):
            continue
        src = data.get("_src", u)
        tgt = data.get("_tgt", v)
        if src == tgt:
            continue
        groups.setdefault((src, tgt), []).append(dict(data))

    H = cast(nx.DiGraph, _copy_graph_skeleton(G, nx.DiGraph))
    for (src, tgt), records in sorted(
        groups.items(),
        key=lambda item: (stable_value_key(item[0][0]), stable_value_key(item[0][1])),
    ):
        if src not in H:
            H.add_node(src)
        if tgt not in H:
            H.add_node(tgt)
        H.add_edge(src, tgt, **_merged_edge_attrs(records, "confidence"))
    return H


def _normalize_contexts(contexts: Iterable[str] | str | None) -> set[str] | None:
    if contexts is None:
        return None
    raw_contexts = [contexts] if isinstance(contexts, str) else contexts
    normalized = {str(context).strip().lower() for context in raw_contexts if str(context).strip()}
    return normalized or None


def project_for_context(G: nx.Graph, *, contexts: Iterable[str] | str | None = None) -> nx.Graph:
    """Return a graph copy containing only edges whose context matches the filter."""
    filters = _normalize_contexts(contexts)
    H = _copy_graph_skeleton(G, G.__class__)
    for u, v, key, data in _iter_edge_data(G):
        if filters is not None and str(data.get("context", "")).strip().lower() not in filters:
            continue
        if isinstance(H, nx.MultiGraph | nx.MultiDiGraph):  # Python 3.10+
            H.add_edge(u, v, key=key, **data)
        else:
            H.add_edge(u, v, **data)
    return H


def edge_records_between(
    G: nx.Graph, u: Hashable, v: Hashable, *, directed_only: bool = False
) -> list[dict[str, Any]]:
    """Return shallow copies of all edge records connecting two nodes.

    By default (``directed_only=False``) a directed graph contributes records in
    BOTH directions (``u->v`` and ``v->u``), which is correct for symmetric
    "how are A and B related" queries. Set ``directed_only=True`` to collect only
    the ``u->v`` direction, which is what directional arrow surfaces need (path
    hops ``A-->B``, explain "out"/"in" connections). On undirected graphs the
    flag is a no-op, as there is no separate reverse direction to collect.
    """
    records: list[dict[str, Any]] = []

    def collect(src: Hashable, tgt: Hashable) -> None:
        if not G.has_edge(src, tgt):
            return
        raw = G.get_edge_data(src, tgt)
        if not isinstance(raw, dict):
            return
        if isinstance(G, nx.MultiGraph | nx.MultiDiGraph):  # Python 3.10+
            for key, data in raw.items():
                if isinstance(data, dict):
                    record = dict(data)
                    record["key"] = key
                    records.append(record)
        else:
            records.append(dict(raw))

    collect(u, v)
    if not directed_only and G.is_directed() and u != v:
        collect(v, u)
    return sorted(records, key=_edge_sort_key)


def edge_summary_between(G: nx.Graph, u: Hashable, v: Hashable) -> dict[str, Any]:
    """Summarize all relationships between two nodes for display consumers."""
    records = edge_records_between(G, u, v)
    representative = records[0].copy() if records else {}
    return {
        "count": len(records),
        "relations": sorted(
            {str(record.get("relation")) for record in records if record.get("relation")}
        ),
        "confidences": sorted(
            {str(record.get("confidence")) for record in records if record.get("confidence")}
        ),
        "representative": representative,
    }


def relationship_envelope(
    G: nx.Graph,
    u: Hashable,
    v: Hashable,
    *,
    cap: int = DEFAULT_RELATIONSHIP_CAP,
    directed_only: bool = False,
) -> RelationshipEnvelope:
    """Bundle all parallel relationships between two nodes into a capped envelope.

    Returns a structured dict suitable for MCP serialization (arrays, not a
    representative-only dict) and as the basis for text rendering::

        {
            "count": int,            # total parallel relationships (u->v plus v->u if directed)
            "shown": list[dict],     # up to ``cap`` full edge-record dicts, in edge_records_between order
            "truncated": int,        # max(0, count - len(shown))
            "relations": list[str],  # ALL unique relations across every record, sorted
            "confidences": list[str],# ALL unique confidences across every record, sorted
        }

    ``relations``/``confidences`` summarize the FULL set even when ``shown`` is
    capped, so callers can say "calls, imports, +2 more (5 total)" accurately.

    A ``cap`` below 1 shows zero records (``shown == []``) while still reporting
    the full ``count``/``relations``/``confidences``; negative caps are clamped to
    zero rather than slicing from the tail.

    ``directed_only`` is threaded through to :func:`edge_records_between`: with the
    default ``False`` a directed graph bundles both directions (symmetric view),
    while ``True`` restricts the envelope to the ``u->v`` direction for directional
    arrow surfaces. It is a no-op on undirected graphs.
    """
    records = edge_records_between(G, u, v, directed_only=directed_only)
    effective_cap = cap if cap > 0 else 0
    shown = records[:effective_cap]
    return {
        "count": len(records),
        "shown": shown,
        "truncated": max(0, len(records) - len(shown)),
        "relations": sorted(
            {str(record.get("relation")) for record in records if record.get("relation")}
        ),
        "confidences": sorted(
            {str(record.get("confidence")) for record in records if record.get("confidence")}
        ),
    }


def format_relationship_envelope(
    G: nx.Graph,
    u: Hashable,
    v: Hashable,
    *,
    cap: int = DEFAULT_RELATIONSHIP_CAP,
    directed_only: bool = False,
) -> str:
    """Render a stable one-line summary of all relationships between two nodes.

    Examples::

        single relation:        "calls"
        single with confidence: "calls (EXTRACTED)"  # confidence shown only if present
        multiple within cap:    "calls, contains, imports"
        capped:                 "calls, contains, imports (+2 more, 5 total)"
        none:                   ""  # empty string when no edge exists

    A single record retains the compact historical representation. Multiple
    records are rendered individually, including their stable key and useful
    provenance fields, so same-relation parallels cannot collapse visually.

    ``directed_only`` is threaded through to :func:`relationship_envelope`: with
    the default ``False`` directed graphs render the symmetric both-direction
    summary, while ``True`` restricts the line to the ``u->v`` direction for
    directional arrow surfaces. It is a no-op on undirected graphs.
    """
    envelope = relationship_envelope(G, u, v, cap=cap, directed_only=directed_only)
    count = envelope["count"]
    if count == 0:
        return ""

    if count == 1:
        record = envelope["shown"][0] if envelope["shown"] else {}
        relation = str(record.get("relation", ""))
        confidence = record.get("confidence")
        return f"{relation} ({confidence})" if confidence else relation

    def render_record(record: dict[str, Any]) -> str:
        relation = str(record.get("relation", ""))
        details: list[str] = []
        for output_name, field_name in (
            ("key", "key"),
            ("location", "source_location"),
            ("confidence", "confidence"),
            ("context", "context"),
        ):
            value = record.get(field_name)
            if value not in (None, ""):
                details.append(f"{output_name}={value}")
        provenance = record.get("provenance", record.get("_origin"))
        if provenance not in (None, ""):
            rendered = (
                _canonical_json(provenance)
                if isinstance(provenance, dict | list)
                else str(provenance)
            )
            details.append(f"provenance={rendered}")
        return f"{relation} [{', '.join(details)}]" if details else relation

    rendered_records = " | ".join(render_record(record) for record in envelope["shown"])
    suffix = ""
    if envelope["truncated"]:
        suffix = f" (+{envelope['truncated']} more, {count} total)"
    return f"{count} records: {rendered_records}{suffix}"


def distinct_neighbor_degree(G: nx.Graph, node: Hashable) -> int:
    """Count unique adjacent nodes without inflating parallel edges."""
    if node not in G:
        return 0
    if G.is_directed():
        directed = cast(nx.DiGraph, G)
        return len(set(directed.predecessors(node)) | set(directed.successors(node)))
    return len(set(G.neighbors(node)))


def stable_shortest_path(G: nx.Graph, source: Hashable, target: Hashable) -> list[Hashable]:
    """Return the canonical shortest path over undirected topology."""
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"source or target is not in the graph: {source!r}, {target!r}")
    if source == target:
        return [source]

    parents: dict[Hashable, Hashable | None] = {source: None}
    pending: deque[Hashable] = deque([source])
    while pending:
        node = pending.popleft()
        if G.is_directed():
            directed = cast(nx.DiGraph, G)
            neighbors = set(directed.predecessors(node)) | set(directed.successors(node))
        else:
            neighbors = set(G.neighbors(node))
        for neighbor in sorted(neighbors, key=stable_value_key):
            if neighbor in parents:
                continue
            parents[neighbor] = node
            if neighbor == target:
                path: list[Hashable] = [target]
                cursor: Hashable = target
                while parents[cursor] is not None:
                    cursor = cast(Hashable, parents[cursor])
                    path.append(cursor)
                path.reverse()
                return path
            pending.append(neighbor)
    raise nx.NetworkXNoPath(f"No path between {source!r} and {target!r}")


def directed_edge_endpoints(
    G: nx.Graph,
    u: Hashable,
    v: Hashable,
    data: dict[str, Any],
) -> tuple[Hashable, Hashable]:
    """Return authoritative endpoints when lifting an edge to a directed graph."""
    if G.is_directed() or u == v:
        return u, v
    source = data.get("_src")
    target = data.get("_tgt")
    if source is None or target is None or {source, target} != {u, v}:
        raise ValueError("undirected edge lacks unambiguous _src/_tgt direction")
    return source, target


def normalize_to_multidigraph(G: nx.Graph) -> nx.MultiDiGraph:
    """Return a MultiDiGraph copy without guessing undirected edge direction."""
    H = nx.MultiDiGraph()
    H.graph.update(G.graph)
    H.add_nodes_from((node, attrs.copy()) for node, attrs in G.nodes(data=True))
    if isinstance(G, nx.MultiGraph | nx.MultiDiGraph):  # Python 3.10+
        for u, v, key, data in G.edges(keys=True, data=True):
            source, target = directed_edge_endpoints(G, u, v, data)
            attrs = data.copy()
            attrs.pop("_src", None)
            attrs.pop("_tgt", None)
            H.add_edge(source, target, key=key, **attrs)
    else:
        for u, v, data in G.edges(data=True):
            source, target = directed_edge_endpoints(G, u, v, data)
            attrs = data.copy()
            attrs.pop("_src", None)
            attrs.pop("_tgt", None)
            H.add_edge(source, target, **attrs)
    return H
