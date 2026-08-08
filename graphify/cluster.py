"""Community detection with native Leiden and a NetworkX Louvain fallback."""

from __future__ import annotations
import inspect
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import networkx as nx

from graphify.projections import project_for_community, stable_value_key

NodeId = Any


def _sorted_nodes(nodes: Iterable[NodeId]) -> list[NodeId]:
    """Return node IDs in the repository-wide heterogeneous total order."""
    return sorted(nodes, key=stable_value_key)


def _member_identity(nodes: Iterable[NodeId]) -> tuple[tuple[str, str], ...]:
    return tuple(stable_value_key(node) for node in _sorted_nodes(nodes))


def _normalize_partition(partition: dict[NodeId, Any]) -> dict[NodeId, int]:
    """Replace backend community labels with stable membership-derived IDs."""
    groups: dict[Any, list[NodeId]] = {}
    for node, cid in partition.items():
        groups.setdefault(cid, []).append(node)
    ordered = sorted(groups.values(), key=lambda nodes: (-len(nodes), _member_identity(nodes)))
    return {node: cid for cid, nodes in enumerate(ordered) for node in _sorted_nodes(nodes)}


def _is_usable_weight(raw: object) -> bool:
    """True when *raw* is a weight the native Leiden engine can consume.

    Only a finite, non-negative real number qualifies. ``bool`` is excluded
    because it subclasses ``int``.
    """
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return False
    return math.isfinite(raw) and raw >= 0


def _effective_weight(data: dict) -> float:
    """Weight leiden will actually use for an edge: its usable ``weight``, else 1.0.

    Mirrors the previous wrapper's default so a total-weight check here matches what
    the partitioner sees.
    """
    raw = data.get("weight")
    return float(raw) if _is_usable_weight(raw) else 1.0  # type: ignore[arg-type]


def _partition(G: nx.Graph, resolution: float = 1.0) -> dict[NodeId, int]:
    """Run community detection. Returns {node_id: community_id}.

    Tries the native Leiden engine first, then falls back to NetworkX Louvain.

    resolution > 1.0 → more, smaller communities.
    resolution < 1.0 → fewer, larger communities.

    """
    stable = nx.Graph()
    stable.add_nodes_from(_sorted_nodes(G.nodes()))

    # Canonicalize orientation before sorting. Sorting the raw orientation leaves
    # adjacency insertion order dependent on the input graph's node insertion
    # order, and both Leiden and Louvain can distinguish that order despite their
    # random seeds being fixed.
    edge_rows: list[tuple[NodeId, NodeId, float]] = []
    for raw_src, raw_tgt, data in G.edges(data=True):
        src, tgt = raw_src, raw_tgt
        if stable_value_key(tgt) < stable_value_key(src):
            src, tgt = tgt, src
        edge_rows.append((src, tgt, _effective_weight(data)))
    edge_rows.sort(key=lambda row: (stable_value_key(row[0]), stable_value_key(row[1]), row[2]))

    # The native engine can panic on a legitimate all-zero graph. Invalid values
    # already map to the established default 1.0 in _effective_weight; if every
    # explicit usable weight is zero, use uniform weights for both backends.
    if edge_rows and sum(weight for _, _, weight in edge_rows) <= 0:
        edge_rows = [(src, tgt, 1.0) for src, tgt, _weight in edge_rows]
    for src, tgt, weight in edge_rows:
        stable.add_edge(src, tgt, weight=weight)

    try:
        from graspologic_native import leiden

        # Native Leiden only accepts string IDs. Stable synthetic IDs preserve
        # distinct JSON node identities such as 1 and "1" without exposing their
        # representation or relying on backend label numbering.
        node_by_native: dict[str, NodeId] = {}
        native_by_node: dict[NodeId, str] = {}
        for index, node in enumerate(stable.nodes()):
            native_id = f"n{index}"
            node_by_native[native_id] = node
            native_by_node[node] = native_id

        native_edges = [
            (native_by_node[source], native_by_node[target], weight)
            for source, target, weight in edge_rows
        ]
        # The native API returns (quality, {string_node_id: community_id}).
        _quality, partition = leiden(
            edges=native_edges,
            starting_communities=None,
            resolution=resolution,
            randomness=0.001,
            iterations=1,
            use_modularity=True,
            seed=42,
            trials=1,
        )
        decoded = {node_by_native[node]: cid for node, cid in partition.items()}
        return _normalize_partition(decoded)
    except (ImportError, SyntaxError):
        pass

    # Fallback: networkx louvain (available since networkx 2.7).
    # Inspect kwargs to stay compatible across NetworkX versions — max_level
    # was added in a later release and prevents hangs on large sparse graphs.
    kwargs: dict = {"seed": 42, "threshold": 1e-4, "resolution": resolution}
    if "max_level" in inspect.signature(nx.community.louvain_communities).parameters:
        kwargs["max_level"] = 10
    communities = nx.community.louvain_communities(stable, **kwargs)
    return _normalize_partition(
        {node: cid for cid, nodes in enumerate(communities) for node in nodes}
    )


_MAX_COMMUNITY_FRACTION = 0.25  # communities larger than 25% of graph get split
_MIN_SPLIT_SIZE = 10  # only split if community has at least this many nodes
_COHESION_SPLIT_THRESHOLD = 0.05  # re-split communities with cohesion below this
_COHESION_SPLIT_MIN_SIZE = 50  # only cohesion-split if community has at least this many nodes


def label_communities_by_hub(
    G: nx.Graph, communities: Mapping[int, Sequence[NodeId]]
) -> dict[int, str]:
    """Deterministic, LLM-free community labels: name each community after its
    highest-degree member — the structural hub — so a report reads ``auth`` /
    ``log_action`` instead of ``Community 70``. Degree is measured on the full graph
    ``G``; ties break by node id for run-to-run stability. A community whose members
    are all absent from ``G`` falls back to ``Community {cid}``.

    Used as the default (no-backend) labeler; an LLM naming pass, when configured,
    overrides these with richer names.
    """
    labels: dict[int, str] = {}
    for cid, members in communities.items():
        present = [n for n in members if n in G]
        if not present:
            labels[cid] = f"Community {cid}"
            continue
        # highest degree wins; ties broken by node id (ascending) for determinism
        hub = min(present, key=lambda n: (-G.degree(n), stable_value_key(n)))
        name = str(G.nodes[hub].get("label") or hub).strip()
        if name.endswith("()"):
            name = name[:-2]
        labels[cid] = name or f"Community {cid}"
    return labels


def community_member_sigs(communities: Mapping[int, Sequence[NodeId]]) -> dict[int, str]:
    """Per-community membership fingerprints: ``{cid: sha256(sorted member ids)}``.

    Persisted next to ``.graphify_labels.json`` so a later ``cluster-only`` can tell
    which communities actually changed since labeling. A cid whose members no longer
    hash the same is a different community — reusing its old (LLM) label there is the
    "stale label after re-scoping" bug this guards against. Deterministic; independent
    of cid index, node order, and machine.
    """
    import hashlib

    sigs: dict[int, str] = {}
    for cid, members in communities.items():
        h = hashlib.sha256()
        for node in _sorted_nodes(members):
            encoded = json.dumps(stable_value_key(node), ensure_ascii=True, separators=(",", ":"))
            h.update(encoded.encode("utf-8"))
            h.update(b"\x00")
        sigs[cid] = h.hexdigest()[:16]
    return sigs


def cluster(
    G: nx.Graph,
    resolution: float = 1.0,
    exclude_hubs_percentile: float | None = None,
) -> dict[int, list[NodeId]]:
    """Run Leiden community detection. Returns {community_id: [node_ids]}.

    Community IDs are stable across runs: 0 = largest community after splitting.
    Oversized communities (> 25% of graph nodes, min 10) are split by running
    a second Leiden pass on the subgraph.

    Accepts directed or undirected graphs. DiGraphs are converted to undirected
    internally since Louvain/Leiden require undirected input.

    resolution: passed to Leiden/Louvain. >1.0 = more smaller communities,
        <1.0 = fewer larger communities. Default 1.0.
    exclude_hubs_percentile: if set (0-100), nodes whose degree exceeds this
        percentile are excluded from partitioning and reattached to their
        majority-vote neighbour community afterwards. Useful for staging/utility
        super-hubs that inflate god-node rankings (#919).
    """
    if G.number_of_nodes() == 0:
        return {}
    # Project multigraphs to simple undirected graph so parallel edges
    # don't inflate Louvain/Leiden community detection.
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = project_for_community(G)
    elif G.is_directed():
        G = G.to_undirected()
    if G.number_of_edges() == 0:
        return {i: [n] for i, n in enumerate(_sorted_nodes(G.nodes()))}

    # Compute hub exclusion set before removing anything so degree is based on full graph
    hub_nodes: set[NodeId] = set()
    if exclude_hubs_percentile is not None:
        degrees = sorted(d for _, d in G.degree())
        if degrees:
            idx = max(0, int(len(degrees) * exclude_hubs_percentile / 100) - 1)
            threshold = degrees[idx]
            hub_nodes = {n for n, d in G.degree() if d > threshold}

    # Leiden warns and drops isolates - handle them separately
    # Also exclude hub nodes from partitioning so they don't pull unrelated
    # subsystems into the same community
    excluded = hub_nodes
    isolates = [n for n in G.nodes() if G.degree(n) == 0 and n not in excluded]
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0 and n not in excluded]
    connected = G.subgraph(connected_nodes)

    raw: dict[int, list[NodeId]] = {}
    if connected.number_of_nodes() > 0:
        partition = _partition(connected, resolution=resolution)
        for node, cid in partition.items():
            raw.setdefault(cid, []).append(node)

    # Each isolate becomes its own single-node community
    next_cid = max(raw.keys(), default=-1) + 1
    for node in isolates:
        raw[next_cid] = [node]
        next_cid += 1

    # Reattach excluded hubs by majority-vote neighbour community
    if hub_nodes:
        node_community: dict[NodeId, int] = {n: cid for cid, nodes in raw.items() for n in nodes}
        for hub in _sorted_nodes(hub_nodes):
            votes: dict[int, int] = {}
            for nb in G.neighbors(hub):
                cid = node_community.get(nb)
                if cid is not None:
                    votes[cid] = votes.get(cid, 0) + 1
            if votes:
                best = min(votes, key=lambda c: (-votes[c], c))
                raw.setdefault(best, []).append(hub)
                node_community[hub] = best
            else:
                raw[next_cid] = [hub]
                node_community[hub] = next_cid
                next_cid += 1

    # Split oversized communities
    max_size = max(_MIN_SPLIT_SIZE, int(G.number_of_nodes() * _MAX_COMMUNITY_FRACTION))
    final_communities: list[list[NodeId]] = []
    for nodes in raw.values():
        if len(nodes) > max_size:
            final_communities.extend(_split_community(G, nodes))
        else:
            final_communities.append(nodes)

    # Second pass: re-split low-cohesion communities caused by doc-hub nodes
    # that bridge otherwise-unrelated subsystems (e.g. CLAUDE.md connected to everything).
    second_pass: list[list[NodeId]] = []
    for nodes in final_communities:
        if (
            len(nodes) >= _COHESION_SPLIT_MIN_SIZE
            and cohesion_score(G, nodes) < _COHESION_SPLIT_THRESHOLD
        ):
            splits = _split_community(G, nodes)
            second_pass.extend(splits if len(splits) > 1 else [nodes])
        else:
            second_pass.append(nodes)
    final_communities = second_pass

    # Re-index by size descending. The tuple(sorted(nodes)) tiebreak makes this a
    # TOTAL order, so an identical grouping always gets identical community IDs.
    # Without it, the hundreds of equal-sized small communities are ordered by the
    # partitioner's (not seed-stable) enumeration order, so their integer IDs
    # permute run-to-run - which reads as massive "community churn" in a per-node
    # cid diff even though the actual grouping is reproducible (#1090 follow-up).
    final_communities.sort(key=lambda nodes: (-len(nodes), _member_identity(nodes)))
    return {i: _sorted_nodes(nodes) for i, nodes in enumerate(final_communities)}


def _split_community(G: nx.Graph, nodes: list[NodeId]) -> list[list[NodeId]]:
    """Run a second Leiden pass on a community subgraph to split it further."""
    subgraph = G.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        # No edges - split into individual nodes
        return [[n] for n in _sorted_nodes(nodes)]
    try:
        sub_partition = _partition(subgraph)
        sub_communities: dict[int, list[NodeId]] = {}
        for node, cid in sub_partition.items():
            sub_communities.setdefault(cid, []).append(node)
        if len(sub_communities) <= 1:
            return [_sorted_nodes(nodes)]
        return [_sorted_nodes(v) for v in sub_communities.values()]
    except Exception:
        return [_sorted_nodes(nodes)]


def cohesion_score(G: nx.Graph, community_nodes: Sequence[NodeId]) -> float:
    """Ratio of actual intra-community edges to maximum possible."""
    n = len(community_nodes)
    if n <= 1:
        return 1.0
    # Cohesion is an undirected topology metric. Project every graph class so
    # reciprocal directed records and multigraph parallels each count once.
    subgraph = project_for_community(G.subgraph(community_nodes))
    actual = subgraph.number_of_edges()
    possible = n * (n - 1) / 2
    return actual / possible if possible > 0 else 0.0


def score_all(G: nx.Graph, communities: Mapping[int, Sequence[NodeId]]) -> dict[int, float]:
    return {cid: cohesion_score(G, nodes) for cid, nodes in communities.items()}


def remap_communities_to_previous(
    communities: Mapping[int, Sequence[NodeId]],
    previous_node_community: Mapping[NodeId, int],
) -> dict[int, list[NodeId]]:
    """Remap community IDs to maximize overlap with a previous assignment.

    Maximizes total intersection size globally. Equal optima are resolved by a
    unique bit-weighted preference over stable old IDs and new member identities.
    Unmatched communities receive IDs not present in the prior assignment.
    """
    if not communities:
        return {}

    new_sets = {cid: set(nodes) for cid, nodes in communities.items()}
    old_sets: dict[int, set[NodeId]] = {}
    for node, old_cid in previous_node_community.items():
        old_sets.setdefault(old_cid, set()).add(node)

    overlap_graph = nx.Graph()
    for old_cid, old_nodes in old_sets.items():
        for new_cid, new_nodes in new_sets.items():
            overlap = len(old_nodes & new_nodes)
            if overlap > 0:
                overlap_graph.add_edge(("old", old_cid), ("new", new_cid), overlap=overlap)

    new_to_final: dict[int, int] = {}
    matched_new_ids: set[int] = set()
    if overlap_graph.number_of_edges():
        components = sorted(
            nx.connected_components(overlap_graph),
            key=lambda nodes: min(stable_value_key(node) for node in nodes),
        )
        for component_nodes in components:
            component = overlap_graph.subgraph(component_nodes)
            rows: list[tuple[int, int, int]] = []
            for left, right, data in component.edges(data=True):
                if left[0] == "new":
                    left, right = right, left
                rows.append((int(data["overlap"]), int(left[1]), int(right[1])))
            rows.sort(
                key=lambda row: (
                    -row[0],
                    stable_value_key(row[1]),
                    _member_identity(new_sets[row[2]]),
                )
            )

            # Primary score is total overlap. The sum of all unique tie bits is
            # smaller than `scale`, so it cannot alter that optimum; it only makes
            # the preferred stable edge set uniquely optimal.
            scale = 1 << len(rows)
            weighted = nx.Graph()
            for rank, (overlap, old_cid, new_cid) in enumerate(rows):
                tie_bit = 1 << (len(rows) - rank - 1)
                weighted.add_edge(
                    ("old", old_cid),
                    ("new", new_cid),
                    score=overlap * scale + tie_bit,
                )
            matching = nx.max_weight_matching(weighted, weight="score")
            for left, right in matching:
                if left[0] == "new":
                    left, right = right, left
                old_cid = int(left[1])
                new_cid = int(right[1])
                new_to_final[new_cid] = old_cid
                matched_new_ids.add(new_cid)

    unmatched = [cid for cid in communities if cid not in matched_new_ids]
    unmatched.sort(key=lambda cid: (-len(communities[cid]), _member_identity(communities[cid])))
    reserved_ids = set(old_sets)
    assigned_ids = set(new_to_final.values())
    next_id = 0
    for new_cid in unmatched:
        while next_id in reserved_ids or next_id in assigned_ids:
            next_id += 1
        new_to_final[new_cid] = next_id
        assigned_ids.add(next_id)
        next_id += 1

    remapped: dict[int, list[NodeId]] = {}
    for new_cid, nodes in communities.items():
        remapped[new_to_final[new_cid]] = _sorted_nodes(nodes)
    return dict(sorted(remapped.items(), key=lambda kv: kv[0]))
