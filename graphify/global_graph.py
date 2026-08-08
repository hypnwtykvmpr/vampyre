from __future__ import annotations
import json
import hashlib
import shutil
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

import networkx as nx

from graphify.graph_loader import GRAPHIFY_PROFILE_KEY
from graphify.graph_state import GraphState, NodeId
from graphify.persistence import FileStateTransaction, atomic_write_text, output_state_lock
from graphify.projections import directed_edge_endpoints, normalize_to_multidigraph

_GLOBAL_DIR = Path.home() / ".graphify"
_GLOBAL_GRAPH = _GLOBAL_DIR / "global-graph.json"
_GLOBAL_MANIFEST = _GLOBAL_DIR / "global-manifest.json"

# Graphify graph_type vocabulary (kept byte-identical to graph_loader /
# export so the global graph profile round-trips through node_link_data).
_GRAPH_TYPE_SIMPLE = "simple"
_GRAPH_TYPE_DIGRAPH = "digraph"
_GRAPH_TYPE_MULTIDIGRAPH = "multidigraph"
_GRAPH_TYPES = frozenset({_GRAPH_TYPE_SIMPLE, _GRAPH_TYPE_DIGRAPH, _GRAPH_TYPE_MULTIDIGRAPH})
_REPO_OWNERS_FIELD = "graphify_repo_owners"


def _graph_type_for_instance(G: nx.Graph) -> str:
    """Return the graphify ``graph_type`` token for a live NetworkX instance.

    The instance is authoritative: classify from ``is_multigraph()`` /
    ``is_directed()`` rather than from any stored profile. Mirrors
    :func:`graphify.export._graph_type_for_instance` and the loader's
    :func:`~graphify.graph_loader._set_graph_profile` vocabulary so a
    save/load round-trip is stable.
    """
    if G.is_multigraph():
        return _GRAPH_TYPE_MULTIDIGRAPH
    if G.is_directed():
        return _GRAPH_TYPE_DIGRAPH
    return _GRAPH_TYPE_SIMPLE


def _graph_class_for_type(graph_type: str) -> type[nx.Graph]:
    """Map a graphify ``graph_type`` token to the NetworkX class that realizes it."""
    if graph_type == _GRAPH_TYPE_MULTIDIGRAPH:
        return nx.MultiDiGraph
    if graph_type == _GRAPH_TYPE_DIGRAPH:
        return nx.DiGraph
    return nx.Graph


def _project_to_class(G: nx.Graph, graph_type: str) -> nx.Graph:
    """Return a copy of *G* realized as the NetworkX class for *graph_type*.

    Multigraph targets reuse :func:`normalize_to_multidigraph` so parallel
    keys survive. Simple/digraph targets rebuild the skeleton and replay
    edges with keyless ``add_edge``; when *G* is itself a multigraph this is
    an intentional, caller-warned collapse (parallel edges fold onto one
    ``(u, v)`` pair). Already-correct classes are still copied so callers can
    mutate the result without aliasing the input.
    """
    if graph_type == _GRAPH_TYPE_MULTIDIGRAPH:
        return normalize_to_multidigraph(G)
    target_cls = _graph_class_for_type(graph_type)
    H = target_cls()
    H.graph.update(G.graph)
    H.add_nodes_from((node, attrs.copy()) for node, attrs in G.nodes(data=True))
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, _key, data in G.edges(keys=True, data=True):
            attrs = data.copy()
            if graph_type == _GRAPH_TYPE_DIGRAPH:
                u, v = directed_edge_endpoints(G, u, v, attrs)
                attrs.pop("_src", None)
                attrs.pop("_tgt", None)
            elif G.is_directed():
                attrs.setdefault("_src", u)
                attrs.setdefault("_tgt", v)
            H.add_edge(u, v, **attrs)
    else:
        for u, v, data in G.edges(data=True):
            attrs = data.copy()
            if graph_type == _GRAPH_TYPE_DIGRAPH:
                u, v = directed_edge_endpoints(G, u, v, attrs)
                attrs.pop("_src", None)
                attrs.pop("_tgt", None)
            elif G.is_directed():
                attrs.setdefault("_src", u)
                attrs.setdefault("_tgt", v)
            H.add_edge(u, v, **attrs)
    return H


def _infer_target_type(graphs: list[nx.Graph]) -> str:
    """Infer the composition target type from a list of graphs.

    Multidigraph if ANY input is a multigraph; else digraph if ANY input is
    directed; else simple. This is the no-explicit-target precedence the
    global compose and the merge driver both rely on.
    """
    if any(G.is_multigraph() for G in graphs):
        return _GRAPH_TYPE_MULTIDIGRAPH
    if any(G.is_directed() for G in graphs):
        return _GRAPH_TYPE_DIGRAPH
    return _GRAPH_TYPE_SIMPLE


def normalize_graphs_for_global(
    graphs: list[nx.Graph], *, target_type: str | None = None
) -> tuple[list[nx.Graph], str]:
    """Normalize a list of graphs to one common class for global composition.

    Reusable by both :func:`global_add` and the ``__main__`` merge driver /
    merge-graphs path so class normalization lives in exactly one place.

    - When *target_type* is ``None`` it is inferred via :func:`_infer_target_type`
      (multidigraph if any input is multi; else digraph if any directed; else
      simple). An inferred multidigraph target never loses data.
    - When *target_type* is an EXPLICIT ``"simple"`` / ``"digraph"`` and any
      input is a multigraph, that input is projected down to the simple class
      with an explicit :func:`warnings.warn` + stderr WARNING — graphify never
      silently collapses multigraph input without an explicit simple target.
    - Returns ``(normalized_graphs, resolved_target_type)`` where every graph
      is the same class and ``resolved_target_type`` is in the graph_type
      vocabulary.

    Raises:
        ValueError - *target_type* is not a recognized graph_type token.
    """
    if target_type is not None and target_type not in _GRAPH_TYPES:
        raise ValueError(f"target_type must be one of {sorted(_GRAPH_TYPES)}, got {target_type!r}")

    explicit = target_type is not None
    resolved = target_type if explicit else _infer_target_type(graphs)

    if explicit and resolved != _GRAPH_TYPE_MULTIDIGRAPH:
        # Down-projecting an explicit simple/digraph target: warn loudly for
        # every multigraph input whose parallel edges are about to collapse.
        for G in graphs:
            if G.is_multigraph():
                msg = (
                    f"global compose: projecting multigraph input "
                    f"({G.number_of_edges()} edges) to '{resolved}' target — "
                    f"parallel edges will be collapsed onto single (u, v) pairs. "
                    f"Omit the explicit simple/digraph target to preserve them."
                )
                warnings.warn(msg, stacklevel=2)
                print(f"[graphify global] WARNING: {msg}", file=sys.stderr)

    normalized = [_project_to_class(G, resolved) for G in graphs]
    return normalized, resolved


def detect_pre_profile(data: object) -> bool:
    """Return True when a global-graph JSON dict predates profile/class metadata.

    A "pre-profile" graph JSON LACKS ``graphify_profile`` (at the top level and
    nested under ``"graph"``) AND lacks BOTH explicit ``multigraph`` / ``directed``
    flags. Such a file was written before class normalization existed, so it may
    already be a silently-collapsed simple graph whose lost parallel edges cannot
    be reconstructed. The presence of ANY of those four markers means the writer
    knew the graph class, so it is NOT pre-profile.
    """
    if not isinstance(data, dict):
        return False
    if GRAPHIFY_PROFILE_KEY in data:
        return False
    nested = data.get("graph")
    if isinstance(nested, dict) and GRAPHIFY_PROFILE_KEY in nested:
        return False
    if "multigraph" in data or "directed" in data:
        return False
    return True


class GlobalGraphRecoveryError(RuntimeError):
    """Raised when a global operation would irreversibly upgrade a pre-profile graph."""


def refuse_pre_profile_upgrade(
    data: dict,
    target_type: str,
    *,
    backup_hint: Path | None = None,
    graph_label: str = "global graph",
    graph_path: str = "global-graph.json",
    recovery_hint: str | None = None,
) -> None:
    """Refuse to upgrade a pre-profile global graph to multigraph.

    Reusable guard (callable by the merge driver) that enforces the recovery
    policy: a pre-profile global graph (see :func:`detect_pre_profile`) may
    already be collapsed, so "upgrading" it to a multidigraph target would
    fabricate a keyed graph from data that can no longer carry the lost
    parallel edges. In that case raise :class:`GlobalGraphRecoveryError` with a
    clear recovery message pointing at the backup and the rebuild-from-source
    path (``global remove`` + ``global add``).

    Simple-in -> simple-out (or digraph) operation on a pre-profile graph is
    NOT refused — only an upgrade to ``multidigraph`` is irreversible.
    """
    if target_type != _GRAPH_TYPE_MULTIDIGRAPH:
        return
    if not detect_pre_profile(data):
        return
    backup_line = (
        f" A pre-overwrite backup was saved at {backup_hint}."
        if backup_hint is not None
        else " Check for a dated .bak snapshot of the previous graph."
    )
    if recovery_hint is None:
        recovery_hint = (
            "To rebuild safely, remove the affected repos and re-add them from source "
            "(`graphify global remove <tag>` then `graphify global add`), which "
            "regenerates keyed parallel edges from the per-repo graph.json."
        )
    else:
        recovery_hint = recovery_hint.strip()
        if recovery_hint and not recovery_hint.endswith("."):
            recovery_hint += "."
    recovery_line = f" {recovery_hint}" if recovery_hint else ""
    article = "an" if graph_label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    raise GlobalGraphRecoveryError(
        f"refusing to upgrade {article} pre-profile {graph_label} to a multidigraph: "
        f"{graph_path} has no graphify_profile or multigraph/directed flags, "
        "so it predates class tracking and may already have collapsed parallel "
        "edges that cannot be reconstructed by upgrading in place." + backup_line + recovery_line
    )


def backup_global_graph(*, transaction: FileStateTransaction | None = None) -> Path | None:
    with output_state_lock(_GLOBAL_DIR) as lease:
        if lease is None:  # blocking acquisition cannot normally return None
            raise RuntimeError("could not acquire global publication lock")
        return _backup_global_graph_locked(transaction=transaction)


def _backup_global_graph_locked(*, transaction: FileStateTransaction | None = None) -> Path | None:
    """Snapshot the existing global-graph.json to a dated ``.bak`` before overwrite.

    Mirrors :func:`graphify.export.backup_if_protected`'s dated-snapshot pattern,
    adapted for the single global-graph.json file: the backup is written next to
    it as ``global-graph.<YYYY-MM-DD>.bak``. Idempotent within a day — if today's
    backup already holds byte-identical content the copy is skipped; if the live
    graph changed since the last backup today the snapshot is refreshed in place
    (one backup per day, always the latest pre-overwrite state).

    Returns the backup path, or None when there is nothing to back up (no
    existing global graph) or backup is disabled via ``GRAPHIFY_NO_BACKUP``.
    A protected overwrite is refused when its backup cannot be completed.
    """
    import os

    if os.environ.get("GRAPHIFY_NO_BACKUP"):
        return None
    if not _GLOBAL_GRAPH.exists():
        return None

    today = date.today().isoformat()
    backup_path = _GLOBAL_GRAPH.with_name(f"{_GLOBAL_GRAPH.stem}.{today}.bak")
    backup_transaction = transaction or FileStateTransaction([backup_path])
    owns_transaction = transaction is None
    if transaction is not None:
        transaction.capture(backup_path)
    try:
        if backup_path.exists():
            src_hash = hashlib.sha256(_GLOBAL_GRAPH.read_bytes()).hexdigest()
            bak_hash = hashlib.sha256(backup_path.read_bytes()).hexdigest()
            if src_hash == bak_hash:
                return backup_path  # identical content, nothing to do
        _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_GLOBAL_GRAPH, backup_path)
        if owns_transaction:
            backup_transaction.commit()
        return backup_path
    except Exception as exc:
        if owns_transaction:
            backup_transaction.rollback()
        raise OSError(f"global graph backup failed: {exc}") from exc


def _load_manifest() -> dict:
    if _GLOBAL_MANIFEST.exists():
        try:
            return json.loads(_GLOBAL_MANIFEST.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GlobalGraphRecoveryError(
                f"global manifest at {_GLOBAL_MANIFEST} is corrupt; refusing to "
                f"modify global state. The original file was preserved: {exc}"
            ) from exc
    return {"version": 1, "repos": {}}


def _save_manifest(manifest: dict) -> None:
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(_GLOBAL_MANIFEST, json.dumps(manifest, indent=2))


def _read_global_graph_data() -> dict | None:
    """Return the raw global-graph.json dict (size-capped), or None if absent.

    Reads the on-disk JSON WITHOUT rebuilding the NetworkX graph so callers can
    inspect pre-profile markers (:func:`detect_pre_profile`) before deciding
    whether an operation is safe. The ``edges``->``links`` alias is normalized
    so downstream node_link_graph rehydration is consistent with
    :func:`_load_global_graph`.
    """
    if not _GLOBAL_GRAPH.exists():
        return None
    from graphify.graph_state import read_graph_state_payload

    data = read_graph_state_payload(_GLOBAL_GRAPH)
    if "links" not in data and "edges" in data:
        data = dict(data, links=data["edges"])
    return data


def _load_global_graph(*, allow_pre_profile: bool = False) -> nx.Graph:
    state = _load_global_state(allow_pre_profile=allow_pre_profile)
    return state.graph if state is not None else nx.Graph()


def _load_global_state(*, allow_pre_profile: bool = False) -> GraphState | None:
    if _GLOBAL_GRAPH.exists():
        from graphify.graph_loader import load_graph_state_file
        from graphify.graph_state import DecodeMode

        mode = DecodeMode.READ_ONLY_LEGACY if allow_pre_profile else DecodeMode.MIGRATE_LEGACY
        return load_graph_state_file(_GLOBAL_GRAPH, mode=mode)
    return None


def _stamp_global_profile(G: nx.Graph) -> None:
    """Stamp ``G.graph[GRAPHIFY_PROFILE_KEY]`` with the instance graph_type.

    Existing profile fields are preserved; ``graph_type`` is always overwritten
    to match the live instance (the instance is authoritative), mirroring
    :func:`graphify.export._ensure_graph_profile`. This guarantees the global
    graph JSON always carries an accurate, round-trippable profile.
    """
    existing = G.graph.get(GRAPHIFY_PROFILE_KEY)
    profile = dict(existing) if isinstance(existing, dict) else {}
    profile["graph_type"] = _graph_type_for_instance(G)
    G.graph[GRAPHIFY_PROFILE_KEY] = profile


def _save_global_state(state: GraphState) -> None:
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    from graphify.persistence import publish_graph_state

    publish_graph_state(_GLOBAL_GRAPH, state)


def _rewrite_state_nodes(state: GraphState, nodes: list[dict[str, object]]) -> GraphState:
    """Return ``state`` with canonical node records and matching live attributes."""
    from graphify.graph_state import DecodeMode, decode_graph_state, encode_graph_state

    payload = encode_graph_state(state)
    payload["nodes"] = nodes
    return decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)


def _node_repo_owners(record: dict[str, object]) -> set[str]:
    raw_owners = record.get(_REPO_OWNERS_FIELD)
    owners = (
        {owner for owner in raw_owners if isinstance(owner, str) and owner}
        if isinstance(raw_owners, list)
        else set()
    )
    repo = record.get("repo")
    if isinstance(repo, str) and repo:
        owners.add(repo)
    return owners


def _annotate_incoming_external_owners(state: GraphState, repo_tag: str) -> GraphState:
    nodes: list[dict[str, object]] = []
    for raw in state.nodes:
        record = dict(raw)
        if not record.get("source_file"):
            owners = _node_repo_owners(record)
            owners.add(repo_tag)
            record[_REPO_OWNERS_FIELD] = sorted(owners)
        nodes.append(record)
    return _rewrite_state_nodes(state, nodes)


def _canonicalize_external_owners(state: GraphState) -> GraphState:
    nodes: list[dict[str, object]] = []
    for raw in state.nodes:
        record = dict(raw)
        if not record.get("source_file"):
            owners = _node_repo_owners(record)
            if owners:
                record[_REPO_OWNERS_FIELD] = sorted(owners)
        nodes.append(record)
    return _rewrite_state_nodes(state, nodes)


def _detach_repo_ownership(state: GraphState, repo_tag: str) -> tuple[GraphState, int]:
    """Remove one repo's claims, pruning shared externals only at last owner."""
    from graphify.graph_state import prune_graph_state

    nodes: list[dict[str, object]] = []
    remove_ids: set[object] = set()
    for raw in state.nodes:
        record = dict(raw)
        node_id = record["id"]
        if record.get("source_file"):
            if record.get("repo") == repo_tag:
                remove_ids.add(node_id)
            nodes.append(record)
            continue

        owners = _node_repo_owners(record)
        if repo_tag not in owners:
            nodes.append(record)
            continue
        owners.remove(repo_tag)
        if not owners:
            remove_ids.add(node_id)
            nodes.append(record)
            continue
        record[_REPO_OWNERS_FIELD] = sorted(owners)
        if record.get("repo") == repo_tag:
            record["repo"] = min(owners)
        nodes.append(record)

    rewritten = _rewrite_state_nodes(state, nodes)
    return prune_graph_state(rewritten, remove_ids), len(remove_ids)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def global_add(source_path: Path, repo_tag: str) -> dict:
    with output_state_lock(_GLOBAL_DIR) as lease:
        if lease is None:  # blocking acquisition cannot normally return None
            raise RuntimeError("could not acquire global publication lock")
        return _global_add_locked(source_path, repo_tag)


def _global_add_locked(source_path: Path, repo_tag: str) -> dict:
    """Add or update a project graph in the global graph.

    Returns a summary dict with keys: repo_tag, nodes_added, nodes_removed, skipped.
    Skipped=True means the source graph hasn't changed since last add.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"graph not found: {source_path}")

    manifest = _load_manifest()
    src_hash = _file_hash(source_path)

    existing = manifest["repos"].get(repo_tag, {})
    existing_path = existing.get("source_path", "")
    if existing_path and existing_path != str(source_path.resolve()):
        print(
            f"[graphify global] warning: repo tag '{repo_tag}' previously pointed to "
            f"{existing_path!r}, now updating to {str(source_path.resolve())!r}. "
            f"Use --as <tag> to give it a different name.",
            file=sys.stderr,
        )
    if existing.get("source_hash") == src_hash:
        return {"repo_tag": repo_tag, "nodes_added": 0, "nodes_removed": 0, "skipped": True}

    from graphify.graph_loader import load_graph_state_file
    from graphify.graph_state import DecodeMode

    src_state = load_graph_state_file(source_path, mode=DecodeMode.MIGRATE_LEGACY)
    src_state = _annotate_incoming_external_owners(src_state, repo_tag)

    # Inspect the on-disk global graph BEFORE rehydrating, so the recovery
    # policy can see whether it is a pre-profile file that may already have
    # collapsed parallel edges.
    existing_data = _read_global_graph_data()

    # Load global graph and prune stale nodes for this repo. Pruning happens on
    # the loaded class; the surviving (other-repo) subgraph is what we compose
    # the incoming repo into.
    existing_state = _load_global_state(
        allow_pre_profile=existing_data is not None and detect_pre_profile(existing_data)
    )
    if existing_state is not None:
        existing_state, removed = _detach_repo_ownership(existing_state, repo_tag)
    else:
        removed = 0

    # Resolve the composition target class: multidigraph if EITHER the existing
    # global graph OR the incoming source is multi; else digraph if either is
    # directed; else simple. Inferred (target_type=None) never silently
    # collapses — a simple+multi mix upgrades to multidigraph, which is exactly
    # the go/no-go gate (no class-mismatch crash, no silent collapse).
    from graphify.graph_state import (
        MetadataPolicy,
        NamedGraphState,
        compose_graph_states,
        infer_composition_target_type,
    )

    composition_states = [state for state in (existing_state, src_state) if state is not None]
    target_type = infer_composition_target_type(composition_states)

    # Recovery refusal: if composing would UPGRADE a pre-profile global graph to
    # multidigraph (lost parallel edges unreconstructable), back up first so the
    # refusal can point at the snapshot, then refuse without mutating the file.
    if existing_data is not None and target_type == _GRAPH_TYPE_MULTIDIGRAPH:
        if detect_pre_profile(existing_data):
            backup_hint = backup_global_graph()
            refuse_pre_profile_upgrade(existing_data, target_type, backup_hint=backup_hint)

    # Merge external-library nodes (no source_file) by label to avoid duplication
    external_labels = {
        record.get("label", ""): record["id"]
        for record in (existing_state.nodes if existing_state is not None else ())
        if not record.get("source_file") and record.get("label")
    }
    # Map each deduplicated external onto the existing global node so that
    # edges incident to it can be rewired instead of dropped.
    remap: dict[NodeId, NodeId] = {
        cast(NodeId, record["id"]): cast(NodeId, external_labels[record["label"]])
        for record in src_state.nodes
        if not record.get("source_file") and record.get("label") in external_labels
    }
    named_states = []
    if existing_state is not None:
        named_states.append(
            NamedGraphState(
                f"existing:{repo_tag}",
                existing_state,
                apply_namespace=False,
            )
        )
    named_states.append(NamedGraphState(repo_tag, src_state, node_remap=remap))
    composed_state = compose_graph_states(
        named_states,
        target_type=target_type,
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )
    composed_state = _canonicalize_external_owners(composed_state)

    added = len(src_state.nodes) - len(remap)

    manifest["repos"][repo_tag] = {
        "added_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path.resolve()),
        "node_count": added,
        "edge_count": src_state.graph.number_of_edges(),
        "source_hash": src_hash,
        "graph_type": target_type,
    }
    state_transaction = FileStateTransaction([_GLOBAL_GRAPH, _GLOBAL_MANIFEST])
    try:
        backup_global_graph(transaction=state_transaction)
        _save_global_state(composed_state)
        _save_manifest(manifest)
    except Exception:
        state_transaction.rollback()
        raise
    state_transaction.commit()

    return {"repo_tag": repo_tag, "nodes_added": added, "nodes_removed": removed, "skipped": False}


def global_remove(repo_tag: str) -> int:
    with output_state_lock(_GLOBAL_DIR) as lease:
        if lease is None:  # blocking acquisition cannot normally return None
            raise RuntimeError("could not acquire global publication lock")
        return _global_remove_locked(repo_tag)


def _global_remove_locked(repo_tag: str) -> int:
    """Remove all nodes for repo_tag from the global graph. Returns count removed."""
    manifest = _load_manifest()
    if repo_tag not in manifest["repos"]:
        raise KeyError(f"repo '{repo_tag}' not in global graph")

    state = _load_global_state()
    if state is None:
        raise RuntimeError("global graph is missing")
    pruned_state, removed = _detach_repo_ownership(state, repo_tag)
    del manifest["repos"][repo_tag]
    state_transaction = FileStateTransaction([_GLOBAL_GRAPH, _GLOBAL_MANIFEST])
    try:
        backup_global_graph(transaction=state_transaction)
        _save_global_state(pruned_state)
        _save_manifest(manifest)
    except Exception:
        state_transaction.rollback()
        raise
    state_transaction.commit()
    return removed


def global_list() -> dict:
    """Return the manifest repos dict."""
    with output_state_lock(_GLOBAL_DIR) as lease:
        if lease is None:  # blocking acquisition cannot normally return None
            raise RuntimeError("could not acquire global publication lock")
        return _load_manifest().get("repos", {})


def global_path() -> Path:
    return _GLOBAL_GRAPH
