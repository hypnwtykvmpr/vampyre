# assemble node+edge dicts into a NetworkX graph, preserving edge direction
#
# Node deduplication — three layers:
#
# 1. Within a file (AST): each extractor tracks a `seen_ids` set. A node ID is
#    emitted at most once per file, so duplicate class/function definitions in
#    the same source file are collapsed to the first occurrence.
#
# 2. Between files (build): NetworkX G.add_node() is idempotent — calling it
#    twice with the same ID overwrites the attributes with the second call's
#    values. Nodes are added in extraction order (AST first, then semantic),
#    so if the same entity is extracted by both passes the semantic node
#    silently overwrites the AST node. This is intentional: semantic nodes
#    carry richer labels and cross-file context, while AST nodes have precise
#    source_location. If you need to change the priority, reorder extractions
#    passed to build().
#
# 3. Semantic merge (skill): before calling build(), the skill merges cached
#    and new semantic results using an explicit `seen` set keyed on node["id"],
#    so duplicates across cache hits and new extractions are resolved there
#    before any graph construction happens.
#
from __future__ import annotations
import json
import hashlib
import os
import re
import sys
import unicodedata
from collections.abc import Hashable
from pathlib import Path
import networkx as nx
from .edge_identity import make_stable_key, strip_schema_key
from .graph_state import (
    GraphStateError,
    reconcile_hyperedges,
    repair_legacy_file_node_hyperedge_aliases,
)
from .ids import disambiguate_path_scoped_ids, make_id, normalize_id as _normalize_id
from .paths import (
    default_graph_json as _default_graph_json,
    is_absolute_path,
    resolve_scan_root_marker,
)
from .validate import is_hashable, validate_extraction


# Synonym mapper for known invalid file_type values that LLM subagents commonly
# emit. Keeps semantic intent close (markdown→document, tool→code) and falls
# back to "concept" for any other invalid value (see #840).
_LANG_FAMILY: dict[str, str] = {
    ".py": "py",
    ".pyi": "py",
    ".js": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".jsx": "js",
    ".ts": "js",
    ".tsx": "js",
    ".go": "go",
    ".rs": "rs",
    ".java": "jvm",
    ".kt": "jvm",
    ".scala": "jvm",
    ".groovy": "jvm",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "rb",
    ".php": "php",
    ".cs": "cs",
    ".swift": "swift",
    ".lua": "lua",
}


_FILE_TYPE_SYNONYMS = {
    "markdown": "document",
    "text": "document",
    "tool": "code",
    "library": "code",
    "pattern": "concept",
    "principle": "concept",
    "constraint": "concept",
    "tech": "concept",
    "technology": "concept",
    "data-source": "concept",
    "data_source": "concept",
    "gotcha": "concept",
    "framework": "concept",
}


# Hyperedge member lists are canonically keyed `nodes` (see graphify/llm.py
# extraction spec), but LLM/subagent drift and externally-supplied graph.json
# sometimes emit `members` or `node_ids`. _normalize_hyperedge_members folds
# those aliases into `nodes` at ingest so every downstream consumer reads one
# canonical key — mirroring the `from`/`to` edge-endpoint tolerance below.
_HE_MEMBER_ALIASES = ("members", "node_ids")


def _normalize_hyperedge_members(he: object) -> None:
    """Canonicalize a hyperedge's member list onto the `nodes` key, in place.

    If `nodes` is already a list it wins (canonical), and only stray alias keys
    are dropped. Otherwise the first alias (`members`, then `node_ids`) that is a
    list is moved to `nodes`, deduped preserving order, with a single stderr
    WARNING naming the hyperedge id and alias used. Leftover alias keys are
    always removed so downstream code never re-reads them.
    """
    if not isinstance(he, dict):
        return
    if not isinstance(he.get("nodes"), list):
        for alias in _HE_MEMBER_ALIASES:
            val = he.get(alias)
            if isinstance(val, list):
                seen: set = set()
                deduped: list = []
                for ref in val:
                    try:
                        is_dupe = ref in seen
                    except TypeError:
                        is_dupe = False  # unhashable ref: keep it, validator flags it
                    if is_dupe:
                        continue
                    try:
                        seen.add(ref)
                    except TypeError:
                        pass
                    deduped.append(ref)
                he["nodes"] = deduped
                print(
                    f"[graphify] WARNING: hyperedge "
                    f"'{he.get('id', '?')}' uses field '{alias}' instead of "
                    f"'nodes'; normalizing.",
                    file=sys.stderr,
                )
                break
    # Drop any leftover alias keys regardless of which branch ran above.
    for alias in _HE_MEMBER_ALIASES:
        he.pop(alias, None)


def _norm_source_file(p: str | None, root: str | None = None) -> str | None:
    """Normalize path separators and relativize absolute paths.

    Converts backslashes to forward slashes (Windows compatibility) and, when
    root is provided, strips the absolute prefix from paths produced by semantic
    subagents so source_file is always repo-relative (fixes #932).
    """
    if not p:
        return p
    p = p.replace("\\", "/")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", p):
        return p
    # Canonicalize './' and '..' segments so a stored source_file like './a.py' or
    # 'pkg/../a.py' matches the canonical 'a.py' used in eviction/relativization
    # sets (#1521 F2: a non-canonical path could otherwise let a stale edge escape
    # eviction). normpath may reintroduce OS separators on Windows, so re-posix.
    p = os.path.normpath(p).replace("\\", "/")
    if root and os.path.isabs(p):
        try:
            p = Path(p).relative_to(root).as_posix()
        except ValueError:
            # Lexical relative_to failed. Retry with both sides fully resolved:
            # a symlinked scan root (macOS /var -> /private/var, or a symlinked
            # home/worktree) makes the raw prefixes differ even though they point
            # at the same dir, which otherwise silently defeats prune/replace
            # matching. Only the slow path resolves, so the common lexical match
            # stays filesystem-free.
            try:
                p = Path(p).resolve().relative_to(Path(root).resolve()).as_posix()
            except (ValueError, OSError):
                pass
    return p


def _infer_merge_root(graph_path: Path) -> str | None:
    """Best-effort scan root for relativizing paths in build_merge when the caller
    passes no ``root`` (#1571).

    Prefers the committed ``graphify-out/.graphify_root`` marker — the authoritative
    scan root graphify records at build/watch time (#686/#1423) — then falls back to
    the directory that contains the output dir (``graph.json``'s grandparent, i.e.
    ``<root>/graphify-out/graph.json`` -> ``<root>``). Returns None if neither
    resolves, in which case normalization is a no-op (prior behavior).
    """
    try:
        marker = graph_path.parent / ".graphify_root"
        if marker.exists():
            recorded = resolve_scan_root_marker(marker)
            if recorded is not None:
                return str(recorded)
    except OSError:
        pass
    try:
        return str(graph_path.parent.parent.resolve())
    except Exception:
        return None


def _stable_identity_component(value: object) -> str | None:
    """Normalize malformed edge identity values before stable-key hashing."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        # os.fspath can return bytes for bytes-flavored PathLike; coerce to str
        # so downstream json.dumps / hashing always sees text.
        fs_value = os.fspath(value)
        return (
            fs_value.decode("utf-8", errors="replace") if isinstance(fs_value, bytes) else fs_value
        )
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _make_collision_key(base_key: str, attrs: dict, *, salt: int = 0) -> str:
    payload = {
        "base_key": base_key,
        "attrs": attrs,
    }
    if salt:
        payload["salt"] = salt
    repair_payload = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    repair_digest = hashlib.sha256(repair_payload.encode()).hexdigest()
    return f"{base_key}:alt:{repair_digest}"


def _list_field(data: dict, key: str) -> list:
    """Return ``data[key]`` if it is a list; otherwise warn to stderr and return ``[]``.

    Extraction dicts come from LLM subagents and can contain malformed shapes;
    matching the rest of build_from_json's skip+warn policy keeps a single bad
    field from crashing the whole build.
    """
    value = data.get(key, [])
    if isinstance(value, list):
        return value
    print(
        f"[graphify] WARNING: extraction field '{key}' must be a list, "
        f"got {type(value).__name__}; treating as empty.",
        file=sys.stderr,
    )
    return []


def edge_data(G: nx.Graph, u: Hashable, v: Hashable) -> dict:
    """Return one edge attribute dict for (u, v), tolerating MultiGraph.

    For MultiGraph/MultiDiGraph there can be multiple parallel edges;
    this returns the first one (sufficient for callers that only need
    relation/confidence for rendering). Fixes #796.
    """
    raw = G[u][v]
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return next(iter(raw.values()), {})
    return raw


def edge_datas(G: nx.Graph, u: Hashable, v: Hashable) -> list[dict]:
    """Return every edge attribute dict for (u, v); always a list."""
    raw = G[u][v]
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return list(raw.values())
    return [raw]


def dedupe_nodes(nodes: list[dict]) -> list[dict]:
    """Collapse nodes sharing an ``id``, last-writer-wins on attributes.

    Mirrors what ``build_from_json``'s ``G.add_node`` does implicitly (idempotent;
    a later node overwrites an earlier one's attributes). The ``--no-cluster``
    write path dumps the raw node list without building a graph, so same-id nodes
    — e.g. a Swift ``type=module`` anchor emitted once per importing file (#1327)
    — would otherwise appear as duplicates. Insertion order follows each id's
    first appearance; the retained dict is the last one seen.
    """
    by_id: dict = {}
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        by_id[nid] = n
    return list(by_id.values())


def dedupe_edges(edges: list[dict]) -> list[dict]:
    """Collapse exact parallel edges by ``(source, target, relation)``, keeping the
    first occurrence.

    The clustered build path runs edges through a NetworkX ``DiGraph``, which
    collapses parallel edges automatically. The ``--no-cluster`` and incremental
    ``update`` write paths bypass NetworkX and concatenate edge lists raw, so
    duplicates accumulate and edge counts become non-deterministic across build
    modes / repeated updates (#1317). Deduping on the connectivity identity is
    zero-signal-loss and restores idempotency. Callers that intentionally keep
    parallel edges (multigraph output) must not use this.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in edges:
        key = (e.get("source"), e.get("target"), e.get("relation"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _old_file_stems(rel: Path) -> list[str]:
    """Pre-migration stem forms a semantic fragment may have used for ``rel``.

    Ordered longest-first so prefix stripping is greedy and unambiguous:
      - one-parent form: ``parent.stem``  (the old _file_stem rule, #550-era)
      - zero-parent form: ``stem``        (the old llm.py prompt rule, #1509)
    """
    forms: list[str] = []
    parent = rel.parent.name
    if parent and parent not in (".", ""):
        forms.append(make_id(f"{parent}.{rel.stem}"))
    forms.append(make_id(rel.stem))
    # Dedupe while preserving order (top-level files collapse both forms).
    seen: set[str] = set()
    return [f for f in forms if f and not (f in seen or seen.add(f))]


def _semantic_id_remap(nodes: list, root: str | None) -> dict:
    """Re-derive non-AST node ids from ``source_file`` using the canonical
    full-path stem, so a cached/LLM fragment carrying a pre-migration short id
    reconciles with the AST node instead of spawning a ghost (#1504/#1509).

    Drift-proof by construction: the new id is computed from ``source_file`` in
    code, never trusted from the fragment's own ``id`` string. AST-origin nodes
    are skipped (they are already canonical via the extract() post-pass)."""
    from graphify.extractors.base import _file_stem  # local: avoid import cost at module load

    ast_file_ids = {
        _normalize_id(str(node["id"]))
        for node in nodes
        if isinstance(node, dict)
        and node.get("_origin") == "ast"
        and node.get("id")
        and node.get("source_location") in (None, "", "L1")
        and isinstance(node.get("source_file"), str)
        and bool(node.get("source_file"))
        and str(node.get("label") or "")
        in {
            Path(str(node["source_file"])).name,
            Path(str(node["source_file"])).as_posix(),
        }
    }
    remap: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("_origin") == "ast":
            continue
        nid = node.get("id")
        sf = node.get("source_file")
        if not nid or not isinstance(nid, str) or not sf:
            continue
        sf_norm = _norm_source_file(str(sf), root) or str(sf)
        rel = Path(sf_norm)
        if is_absolute_path(sf_norm):
            continue  # can't relativize (no/failed root) — leave id untouched
        if not rel.name:
            # source_file equals the scan root, so _norm_source_file relativized it
            # to Path('.') — a project-level node with no per-file identity to remap.
            # Leave its id untouched (and avoid _file_stem's empty-name crash, #1618).
            continue
        new_stem = make_id(_file_stem(rel))
        if not new_stem:
            continue
        norm_nid = _normalize_id(nid)
        label = str(node.get("label") or "")
        if node.get("source_location") in (None, "", "L1") and label in {
            rel.name,
            rel.as_posix(),
        }:
            if norm_nid != new_stem:
                remap[nid] = new_stem
            continue
        if label and (
            (norm_nid == new_stem and new_stem in ast_file_ids)
            or norm_nid.startswith(new_stem + "_")
        ):
            canonical_labeled_id = make_id(new_stem, label)
            if canonical_labeled_id != nid:
                remap[nid] = canonical_labeled_id
            continue
        if norm_nid == new_stem or norm_nid.startswith(new_stem + "_"):
            continue
        new_id: str | None = None
        for old_stem in _old_file_stems(rel):
            if old_stem == new_stem:
                continue  # already canonical for this form
            if norm_nid == old_stem:
                new_id = new_stem  # the file node itself
                break
            prefix = old_stem + "_"
            if norm_nid.startswith(prefix):
                entity = norm_nid[len(prefix) :]
                new_id = make_id(new_stem, entity)
                break
        if new_id and new_id != nid:
            remap[nid] = new_id
    return remap


def canonicalize_semantic_fragment(
    fragment: dict,
    root: str | Path | None,
) -> dict:
    """Canonicalize semantic identity before completion-order-independent merge."""

    root_text = str(root) if root is not None else None
    result = dict(fragment)
    nodes = [dict(item) if isinstance(item, dict) else item for item in fragment.get("nodes", [])]
    edges = [dict(item) if isinstance(item, dict) else item for item in fragment.get("edges", [])]
    hyperedges = [
        dict(item) if isinstance(item, dict) else item for item in fragment.get("hyperedges", [])
    ]

    def _source_key(record: dict) -> str:
        source_file = record.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            return ""
        normalized = _norm_source_file(source_file, root_text) or source_file
        record["source_file"] = normalized
        return normalized

    node_rows: list[tuple[dict, str, str, str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        old_id = node.get("id")
        if not isinstance(old_id, str) or not old_id:
            continue
        source_key = _source_key(node)
        desired = _semantic_id_remap([node], root_text).get(old_id, old_id)
        node_rows.append((node, old_id, source_key, desired))

    desired_sources: dict[str, set[str]] = {}
    for _node, _old_id, source_key, desired in node_rows:
        if source_key:
            desired_sources.setdefault(desired, set()).add(source_key)
    scoped_targets = {
        desired: disambiguate_path_scoped_ids(desired, source_keys)
        for desired, source_keys in desired_sources.items()
        if len(source_keys) > 1
    }

    source_remap: dict[tuple[str, str], str] = {}
    old_targets: dict[str, set[str]] = {}
    for node, old_id, source_key, desired in node_rows:
        new_id = scoped_targets.get(desired, {}).get(source_key, desired)
        node["id"] = new_id
        source_remap[(old_id, source_key)] = new_id
        old_targets.setdefault(old_id, set()).add(new_id)

    unambiguous = {
        old_id: next(iter(targets)) for old_id, targets in old_targets.items() if len(targets) == 1
    }
    contested = {old_id for old_id, targets in old_targets.items() if len(targets) > 1}
    final_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    diagnostics = dict(result.get("graphify_identity_diagnostics") or {})
    diagnostics.update(
        {
            "semantic_contested_aliases": sorted(contested),
            "semantic_skipped_edges": 0,
            "semantic_skipped_hyperedge_members": 0,
            "semantic_dropped_hyperedges": 0,
        }
    )

    def _remap_reference(value: object, source_key: str) -> tuple[object, bool]:
        if is_hashable(value) and value in final_ids:
            return value, False
        if isinstance(value, str):
            target = source_remap.get((value, source_key)) or unambiguous.get(value)
            if target is not None:
                return target, False
            if value in contested:
                return value, True
        return value, False

    surviving_edges: list[object] = []
    for edge in edges:
        if not isinstance(edge, dict):
            surviving_edges.append(edge)
            continue
        source_key = _source_key(edge)
        source, source_contested = _remap_reference(edge.get("source"), source_key)
        target, target_contested = _remap_reference(edge.get("target"), source_key)
        if source_contested or target_contested:
            diagnostics["semantic_skipped_edges"] += 1
            continue
        edge["source"] = source
        edge["target"] = target
        surviving_edges.append(edge)

    surviving_hyperedges: list[object] = []
    for hyperedge in hyperedges:
        if not isinstance(hyperedge, dict):
            surviving_hyperedges.append(hyperedge)
            continue
        source_key = _source_key(hyperedge)
        member_field = next(
            (
                field_name
                for field_name in ("nodes", "members", "node_ids")
                if isinstance(hyperedge.get(field_name), list)
            ),
            None,
        )
        if member_field is None:
            surviving_hyperedges.append(hyperedge)
            continue
        members: list[object] = []
        for member in hyperedge[member_field]:
            replacement, member_contested = _remap_reference(member, source_key)
            if member_contested:
                diagnostics["semantic_skipped_hyperedge_members"] += 1
                continue
            members.append(replacement)
        hyperedge[member_field] = list(dict.fromkeys(members))
        if len(hyperedge[member_field]) < 2:
            diagnostics["semantic_dropped_hyperedges"] += 1
            continue
        surviving_hyperedges.append(hyperedge)

    def _fingerprint(record: object) -> str:
        return json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)

    node_by_id: dict[object, tuple[str, dict]] = {}
    other_nodes: list[object] = []
    for node in nodes:
        if not isinstance(node, dict) or not is_hashable(node.get("id")):
            other_nodes.append(node)
            continue
        node_id = node["id"]
        fingerprint = _fingerprint(node)
        previous = node_by_id.get(node_id)
        if previous is not None and previous[0] != fingerprint:
            raise ValueError(f"conflicting semantic node id: {node_id!r}")
        node_by_id[node_id] = (fingerprint, node)

    reconciled_hyperedges: dict[object, tuple[str, object]] = {}
    unkeyed_hyperedges: dict[str, object] = {}
    for hyperedge in surviving_hyperedges:
        fingerprint = _fingerprint(hyperedge)
        explicit_id = hyperedge.get("id") if isinstance(hyperedge, dict) else None
        if explicit_id is None:
            unkeyed_hyperedges[fingerprint] = hyperedge
            continue
        previous = reconciled_hyperedges.get(explicit_id)
        if previous is not None and previous[0] != fingerprint:
            raise ValueError(f"conflicting semantic hyperedge id: {explicit_id!r}")
        reconciled_hyperedges[explicit_id] = (fingerprint, hyperedge)

    result["nodes"] = [
        record for _fingerprint, record in sorted(node_by_id.values(), key=lambda item: item[0])
    ] + sorted(other_nodes, key=_fingerprint)
    result["edges"] = [
        record
        for _fingerprint, record in sorted(
            {_fingerprint(record): record for record in surviving_edges}.items()
        )
    ]
    result["hyperedges"] = [
        record
        for _fingerprint, record in sorted(reconciled_hyperedges.values(), key=lambda item: item[0])
    ] + [record for _fingerprint, record in sorted(unkeyed_hyperedges.items())]
    result["graphify_identity_diagnostics"] = diagnostics
    return result


def graph_has_legacy_ids(nodes: list, root: str | Path | None = None, sample: int = 300) -> bool:
    """Whether a loaded graph still uses pre-#1504 node IDs (parent-dir / filename
    stem) rather than the full repo-relative path. Read-only consumers (query,
    serve) use this to nudge the user to rebuild, since they don't re-extract.

    Heuristic and cheap: only **file-level** nodes (source_location ``L1``) are
    inspected, because their ID is unambiguously the file stem. Symbol nodes are
    skipped — some extractors scope a symbol by package/directory (Go's
    ``_make_id(pkg_dir, name)`` → ``sub_thing``), which can coincide with an old
    file-stem form and would otherwise false-positive. Returns True as soon as one
    file node's ID matches an OLD stem form but not the canonical full-path form."""
    from graphify.extractors.base import _file_stem

    _r = str(root) if root is not None else None
    checked = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("source_location") or "") != "L1":
            continue  # only file-level nodes carry an unambiguous file-stem ID
        nid = node.get("id")
        sf = node.get("source_file")
        if not nid or not isinstance(nid, str) or not sf:
            continue
        rel = Path(_norm_source_file(str(sf), _r) or str(sf))
        if is_absolute_path(str(rel)):
            continue
        if not rel.name:
            continue  # source_file == scan root -> Path('.'), no file stem (#1618)
        new_stem = make_id(_file_stem(rel))
        if not new_stem:
            continue
        norm = _normalize_id(nid)
        if norm == new_stem or norm.startswith(new_stem + "_"):
            checked += 1
        else:
            for old in _old_file_stems(rel):
                if old != new_stem and (norm == old or norm.startswith(old + "_")):
                    return True
            checked += 1
        if checked >= sample:
            break
    return False


def _repair_hyperedge_file_node_aliases_from_records(nodes: list, hyperedges: list) -> list:
    """Resolve legacy semantic file-node members from serialized node records."""
    return repair_legacy_file_node_hyperedge_aliases(nodes, hyperedges)


def _repair_hyperedge_file_node_aliases(G: nx.Graph, hyperedges: list) -> list:
    """Resolve legacy semantic file-node members to unique canonical AST nodes."""
    nodes = [dict(attrs, id=node_id) for node_id, attrs in G.nodes(data=True)]
    return _repair_hyperedge_file_node_aliases_from_records(nodes, hyperedges)


def build_from_json(
    extraction: dict,
    *,
    directed: bool = False,
    root: str | Path | None = None,
    multigraph: bool = False,
) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph:
    """Build a NetworkX graph from an extraction dict.

    directed=True produces a DiGraph that preserves edge direction (source→target).
    directed=False (default) produces an undirected Graph for backward compatibility.
    multigraph=True produces a directed MultiDiGraph with keyed parallel edges for
        internal tests/callers; public CLI exposure is intentionally deferred.
        In this mode, directed is ignored because MultiDiGraph is always directed.
    root: if given, absolute source_file paths from semantic subagents are made
        relative to root so all nodes share a consistent path key (#932).
    """
    if not isinstance(extraction, dict):
        raise TypeError("extraction must be a JSON object")

    _root = str(Path(root).resolve()) if root else None
    # NetworkX <= 3.1 serialised edges as "links"; remap to "edges" for compatibility.
    if "edges" not in extraction and "links" in extraction:
        extraction = dict(extraction, edges=extraction["links"])

    nodes = _list_field(extraction, "nodes")
    edges = _list_field(extraction, "edges")
    extraction = dict(extraction, nodes=nodes, edges=edges)

    # Canonicalize legacy node/edge schema before validation.
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if "source" in node and "source_file" not in node:
            # Count edges that reference this node so the warning is actionable (#479)
            node_id = node.get("id", "?")
            affected_edges = sum(
                1
                for e in edges
                if isinstance(e, dict)
                and (e.get("source") == node_id or e.get("target") == node_id)
            )
            print(
                f"[graphify] WARNING: node '{node_id}' uses field 'source' instead of "
                f"'source_file' — {affected_edges} edge(s) may be misrouted. "
                f"Rename the field to 'source_file' to silence this warning.",
                file=sys.stderr,
            )
            node["source_file"] = node.pop("source")
        # Default missing/None file_type to "concept" so legacy graph.json
        # entries (and stub nodes preserved by `_rebuild_code` from older
        # graphify versions that didn't always populate file_type) don't
        # trigger spurious "invalid file_type 'None'" validator warnings (#660).
        if node.get("file_type") in (None, ""):
            node["file_type"] = "concept"
        ft = node.get("file_type", "")
        if ft and ft not in {"code", "document", "paper", "image", "rationale", "concept"}:
            node["file_type"] = _FILE_TYPE_SYNONYMS.get(ft, "concept")

    # Canonicalize hyperedge member lists (#1561): producers sometimes key the
    # member list `members`/`node_ids` instead of `nodes`. Fold aliases onto
    # `nodes` here — BEFORE validation and the semantic-rekey loop below — so
    # every downstream consumer (rekey, source_file relativize, to_json) reads
    # one canonical key, the same way edge endpoints alias from/to at build.
    for he in extraction.get("hyperedges", []) or []:
        _normalize_hyperedge_members(he)

    errors = validate_extraction(extraction)
    # Dangling edges (stdlib/external imports) are expected - only warn about real schema errors.
    real_errors = [e for e in errors if "does not match any node id" not in e]
    if real_errors:
        print(
            f"[graphify] Extraction warning ({len(real_errors)} issues): {real_errors[0]}",
            file=sys.stderr,
        )
    # Deterministic semantic re-key (#1504/#1509): the node-ID stem is now the
    # full repo-relative path (docs/v1/api/README.md -> docs_v1_api_readme), but
    # the semantic cache is UNVERSIONED, so a cached/LLM fragment can still carry
    # an OLD short id whose stem was just the immediate parent dir (api_readme),
    # or a prompt-drifting id with zero parent dirs (readme). Rather than trust
    # LLM prose to emit the right stem, we re-derive every non-AST node's id from
    # its own source_file in code, so a drifted fragment physically reconciles
    # with the AST node instead of spawning a ghost / a re-bill. AST-origin nodes
    # already carry canonical ids (the extract() id-remap post-pass guarantees it)
    # and are left untouched.
    _rekey: dict[str, str] = _semantic_id_remap(extraction.get("nodes", []), _root)
    if _rekey:
        for node in extraction.get("nodes", []):
            if isinstance(node, dict) and node.get("id") in _rekey:
                node["id"] = _rekey[node["id"]]
        for edge in extraction.get("edges", []):
            if not isinstance(edge, dict):
                continue
            if edge.get("source") in _rekey:
                edge["source"] = _rekey[edge["source"]]
            if edge.get("target") in _rekey:
                edge["target"] = _rekey[edge["target"]]
        for he in extraction.get("hyperedges", []) or []:
            if isinstance(he, dict) and isinstance(he.get("nodes"), list):
                he["nodes"] = [_rekey.get(n, n) for n in he["nodes"]]

    if multigraph:
        from .multigraph_compat import require_multigraph_capabilities

        require_multigraph_capabilities()
    G: nx.Graph = nx.MultiDiGraph() if multigraph else nx.DiGraph() if directed else nx.Graph()
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node:
            continue
        node_id = node["id"]
        if not is_hashable(node_id):
            continue
        if "source_file" in node:
            node["source_file"] = _norm_source_file(
                _stable_identity_component(node["source_file"]), _root
            )
        node_attrs = {k: v for k, v in node.items() if k != "id"}
        # Reject node ids that JSON-serialize but won't round-trip to the same
        # hashable type. Tuples serialize as JSON arrays and come back as lists
        # (unhashable), so they cannot be used as NetworkX node ids after a
        # save/load cycle even though json.dumps would accept them.
        if isinstance(node_id, (list, tuple, set, frozenset, dict)):
            print(
                f"[graphify] WARNING: node id {node_id!r} ({type(node_id).__name__}) "
                f"would not round-trip through JSON as the same hashable type; skipping.",
                file=sys.stderr,
            )
            continue
        # Check id AND attrs are JSON-serializable. NetworkX allows hashable but
        # non-JSON-safe ids (e.g., custom objects); accepting them here would
        # break later node_link_data + json.dump.
        try:
            json.dumps({"id": node_id, **node_attrs}, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            print(
                f"[graphify] WARNING: node {node_id!r} has non-JSON-serializable "
                f"id or attrs ({exc}); skipping.",
                file=sys.stderr,
            )
            continue
        G.add_node(node_id, **node_attrs)
    node_set = set(G.nodes())

    # #1145 (extended): merge LLM ghost-duplicate nodes into AST canonical nodes.
    # Original bug: AST uses parent-qualified IDs (mingpt_bpe_get_pairs) while LLM
    # uses bare-stem IDs (bpe_get_pairs) — different IDs, same symbol.
    # Original fix only caught LLM nodes with source_location=None; LLM now
    # populates source_location, so those ghosts survived. Extended fix: use
    # _origin=="ast" as the canonical signal. AST nodes always win; any non-AST
    # node sharing (basename, label) with an AST node is a ghost.
    _loc_nodes: dict[tuple[str, str], str] = {}  # (basename, label) -> canonical AST node id
    _loc_collisions: set[tuple[str, str]] = set()  # keys shared by 2+ AST nodes

    # Pass 1: collect canonical AST nodes.
    # When 2+ AST nodes share a key (same-named symbols in same-named files across
    # directories, e.g. render in two index.ts), the key is ambiguous: merging a
    # ghost would pick an arbitrary winner via set-iteration order (#1257). Track
    # those keys so Pass 2 skips them — same conservatism as
    # _rewire_unique_stub_nodes, which only merges when exactly one real def exists.
    for nid in node_set:
        if not isinstance(nid, str):
            continue
        attrs = G.nodes[nid]
        label = str(attrs.get("label", "")).strip()
        sf = str(attrs.get("source_file", ""))
        basename = Path(sf).name if sf else ""
        if not label or not basename:
            continue
        is_ast = attrs.get("_origin") == "ast"
        if not is_ast:
            continue
        key = (basename, label)
        if key in _loc_nodes:
            _loc_collisions.add(key)
        _loc_nodes[key] = nid

    # Pass 2: map every semantic ghost to its single unambiguous AST twin.
    _ghost_remap: dict[str, str] = {}  # ghost_id -> canonical_id
    for nid in node_set:
        attrs = G.nodes[nid]
        if attrs.get("_origin") == "ast":
            continue  # AST nodes are never ghosts
        label = str(attrs.get("label", "")).strip()
        sf = str(attrs.get("source_file", ""))
        basename = Path(sf).name if sf else ""
        if not label or not basename:
            continue
        key = (basename, label)
        if key in _loc_collisions:
            continue  # ambiguous key: no safe canonical winner, leave ghost intact
        ast_id = _loc_nodes.get(key)
        if ast_id is not None and ast_id != nid:
            _ghost_remap[nid] = ast_id
    # Remove ghost nodes from the graph; edges will be re-pointed via norm_to_id.
    for ghost_id in _ghost_remap:
        G.remove_node(ghost_id)
        node_set.discard(ghost_id)
    if _ghost_remap:
        for he in extraction.get("hyperedges", []) or []:
            if isinstance(he, dict) and isinstance(he.get("nodes"), list):
                he["nodes"] = [_ghost_remap.get(node_id, node_id) for node_id in he["nodes"]]

    # Exact endpoint IDs always win. Fuzzy aliases are accepted only when one
    # canonical node claims them; contested aliases are skipped rather than
    # assigned according to set/hash iteration order.
    exact_alias_to_id: dict[str, Hashable] = {}
    normalized_alias_to_id: dict[str, Hashable] = {}
    contested_exact_aliases: set[str] = set()
    contested_normalized_aliases: set[str] = set()

    def _register_alias(alias: str, canonical_id: Hashable, *, exact: bool) -> None:
        normalized = _normalize_id(alias)
        prior_normalized = normalized_alias_to_id.get(normalized)
        if normalized in contested_normalized_aliases:
            pass
        elif prior_normalized is None:
            normalized_alias_to_id[normalized] = canonical_id
        elif prior_normalized != canonical_id:
            normalized_alias_to_id.pop(normalized, None)
            contested_normalized_aliases.add(normalized)
        if not exact:
            return
        prior_exact = exact_alias_to_id.get(alias)
        if alias in contested_exact_aliases:
            return
        if prior_exact is None:
            exact_alias_to_id[alias] = canonical_id
        elif prior_exact != canonical_id:
            exact_alias_to_id.pop(alias, None)
            contested_exact_aliases.add(alias)

    for node_id in sorted(
        (node_id for node_id in node_set if isinstance(node_id, str)),
        key=lambda value: (value.casefold(), value),
    ):
        _register_alias(node_id, node_id, exact=False)
    for ghost_id, canonical_id in sorted(_ghost_remap.items()):
        _register_alias(ghost_id, canonical_id, exact=True)
    # Pre-migration alias index (#1504): register each canonical node's OLD-stem id
    # forms as aliases so a stale-id edge endpoint coming from an un-re-keyed
    # fragment (e.g. an incremental update whose fragment references a symbol in a
    # file that was NOT re-extracted) still resolves to the migrated node instead
    # of dangling. Only fills gaps — never overrides a real node id.
    from graphify.extractors.base import _file_stem as _fs

    for nid in node_set:
        if not isinstance(nid, str):
            continue
        attrs = G.nodes[nid]
        sf = attrs.get("source_file")
        if not sf:
            continue
        rel = Path(str(sf))
        if is_absolute_path(str(sf)):
            continue
        raw_stem = _fs(rel)
        if not raw_stem:
            continue
        new_stem = make_id(raw_stem)
        suffix = ""
        if _normalize_id(nid).startswith(new_stem):
            suffix = _normalize_id(nid)[len(new_stem) :]  # leading "_entity" or ""
        for old_stem in _old_file_stems(rel):
            if old_stem == new_stem:
                continue
            alias = old_stem + suffix
            _register_alias(alias, nid, exact=True)
    identity_diagnostics = {
        "contested_exact_aliases": sorted(contested_exact_aliases),
        "contested_normalized_aliases": sorted(contested_normalized_aliases),
        "skipped_contested_endpoints": 0,
    }
    multigraph_groups: dict[tuple[Hashable, Hashable, str], list[dict]] = {}
    multigraph_explicit_keys: set[tuple[Hashable, Hashable, str]] = set()
    multigraph_diagnostics = {"exact_duplicate_edges": 0, "key_collision_edges": 0}

    # Iterate edges in a deterministic order. The graph is undirected and stores
    # direction in _src/_tgt; when two edges collapse onto the same node pair the
    # last write wins, so an unstable iteration order flips _src/_tgt run-to-run
    # and makes the serialized graph churn. Sorting also stabilizes multigraph
    # key-collision grouping before keyed emission.
    def _edge_sort_key(edge: object) -> tuple[str, str, str, str]:
        if not isinstance(edge, dict):
            return ("", "", "", repr(edge))
        return (
            str(edge.get("source", edge.get("from", ""))),
            str(edge.get("target", edge.get("to", ""))),
            str(edge.get("relation", "")),
            json.dumps(edge, sort_keys=True, ensure_ascii=False, default=str),
        )

    for edge in sorted(edges, key=_edge_sort_key):
        if not isinstance(edge, dict):
            continue
        if "source" not in edge and "from" in edge:
            edge["source"] = edge["from"]
        if "target" not in edge and "to" in edge:
            edge["target"] = edge["to"]
        if "source" not in edge or "target" not in edge:
            continue
        src, tgt = edge["source"], edge["target"]
        srcis_hashable = is_hashable(src)
        tgtis_hashable = is_hashable(tgt)
        if not srcis_hashable or not tgtis_hashable:
            endpoint = "source" if not srcis_hashable else "target"
            endpoint_value = src if not srcis_hashable else tgt
            print(
                "[graphify] WARNING: skipped edge with unhashable "
                f"{endpoint} endpoint ({type(endpoint_value).__name__})",
                file=sys.stderr,
            )
            continue
        # Remap mismatched IDs via normalization before dropping the edge.
        if isinstance(src, str) and src not in node_set:
            exact_target = exact_alias_to_id.get(src)
            normalized = _normalize_id(src)
            if exact_target is not None:
                src = exact_target
            elif normalized not in contested_normalized_aliases:
                src = normalized_alias_to_id.get(normalized, src)
            else:
                identity_diagnostics["skipped_contested_endpoints"] += 1
        if isinstance(tgt, str) and tgt not in node_set:
            exact_target = exact_alias_to_id.get(tgt)
            normalized = _normalize_id(tgt)
            if exact_target is not None:
                tgt = exact_target
            elif normalized not in contested_normalized_aliases:
                tgt = normalized_alias_to_id.get(normalized, tgt)
            else:
                identity_diagnostics["skipped_contested_endpoints"] += 1
        if src not in node_set or tgt not in node_set:
            continue  # skip edges to external/stdlib nodes - expected, not an error
        # Exclude legacy from/to alongside source/target so they don't survive
        # as ordinary edge attrs after legacy-shape remap above.
        base_attrs = {k: v for k, v in edge.items() if k not in ("source", "target", "from", "to")}
        raw_key, attrs = strip_schema_key(base_attrs)
        # Backfill source_file from the endpoint nodes (every node carries one).
        # Semantic/LLM edges occasionally omit it, which downstream validation
        # flags and leaves query results with no file reference (#1279).
        if not attrs.get("source_file"):
            attrs["source_file"] = (
                G.nodes[src].get("source_file") or G.nodes[tgt].get("source_file") or ""
            )
        if "source_file" in attrs:
            attrs["source_file"] = _norm_source_file(
                _stable_identity_component(attrs["source_file"]), _root
            )
        # Drop cross-language INFERRED `calls` edges — same short names (render,
        # parse, etc.) appear across language boundaries in multi-language chunks,
        # producing phantom edges that don't represent real call relationships.
        if attrs.get("relation") == "calls" and attrs.get("confidence") == "INFERRED":
            _LANG_FAMILY: dict[str, str] = {
                ".py": "py",
                ".pyi": "py",
                ".js": "js",
                ".mjs": "js",
                ".cjs": "js",
                ".jsx": "js",
                ".ts": "js",
                ".tsx": "js",
                ".mts": "js",
                ".cts": "js",
                ".go": "go",
                ".rs": "rs",
                ".java": "jvm",
                ".kt": "jvm",
                ".scala": "jvm",
                ".groovy": "jvm",
                # C, C++, and ObjC interoperate within one compilation unit: a method
                # declared in a shared `.h` is defined/called from a `.c`/`.cpp`/`.m`
                # sibling, so a cross-file INFERRED call from impl to its header decl
                # is legitimate, not a phantom name-collision across languages. Treat
                # the whole C family as one so the receiver-typed C++/ObjC member-call
                # resolvers' header-targeting edges survive build (#1547/#1556).
                ".c": "c",
                ".h": "c",
                ".cc": "c",
                ".cpp": "c",
                ".hpp": "c",
                ".cxx": "c",
                ".hh": "c",
                ".hxx": "c",
                ".cu": "c",
                ".cuh": "c",
                ".metal": "c",
                ".m": "c",
                ".mm": "c",
                ".rb": "rb",
                ".php": "php",
                ".cs": "cs",
                ".swift": "swift",
                ".lua": "lua",
            }
            src_ext = Path(G.nodes[src].get("source_file") or "").suffix.lower()
            tgt_ext = Path(G.nodes[tgt].get("source_file") or "").suffix.lower()
            if src_ext and tgt_ext and _LANG_FAMILY.get(src_ext) != _LANG_FAMILY.get(tgt_ext):
                continue
        # Preserve original edge direction - undirected graphs lose it otherwise,
        # causing display functions to show edges backwards.
        attrs["_src"] = src
        attrs["_tgt"] = tgt
        # Refuse to store any edge whose attrs cannot round-trip through JSON.
        # Mutating attrs in place would silently change the user's stored value;
        # skipping with a warning matches the rest of the build's defensive policy
        # and prevents later json.dump crashes during export, identically in
        # simple-graph and multigraph modes.
        try:
            json.dumps(attrs, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            print(
                f"[graphify] WARNING: edge ({src}->{tgt}) has non-JSON-serializable "
                f"attrs ({exc}); skipping.",
                file=sys.stderr,
            )
            continue
        if multigraph:
            if raw_key is not None and not isinstance(raw_key, str):
                raise TypeError(
                    f"multigraph edge 'key' must be a string, got "
                    f"{type(raw_key).__name__} ({raw_key!r})"
                )
            base_key = (
                raw_key
                if raw_key is not None
                else make_stable_key(
                    _stable_identity_component(attrs.get("relation")),
                    _stable_identity_component(attrs.get("source_file")),
                    _stable_identity_component(attrs.get("source_location")),
                )
            )
            if raw_key is not None:
                multigraph_explicit_keys.add((src, tgt, base_key))
            multigraph_groups.setdefault((src, tgt, base_key), []).append(dict(attrs))
        else:
            # When the graph is undirected and the same node pair appears twice with
            # the same relation but opposite directions (e.g. a `calls` b and b `calls` a),
            # nx.Graph collapses them into one edge. The deterministic sort above means
            # the lexicographically-later direction would systematically overwrite the
            # earlier one's _src/_tgt, silently flipping the surviving edge's caller
            # and callee. First-seen direction wins instead — drop the redundant
            # reverse-direction duplicate so the original direction is preserved (#1061).
            if not G.is_directed() and G.has_edge(src, tgt):
                existing = edge_data(G, src, tgt)
                if existing.get("relation") == attrs.get("relation") and (
                    existing.get("_src") == tgt and existing.get("_tgt") == src
                ):
                    continue
            G.add_edge(src, tgt, **attrs)
    if multigraph:
        singleton_groups: list[tuple[Hashable, Hashable, str, dict]] = []
        multi_groups: list[tuple[Hashable, Hashable, str, list[dict]]] = []
        used_keys_by_pair: dict[tuple[Hashable, Hashable], set[str]] = {}
        for (src, tgt, base_key), group_attrs in multigraph_groups.items():
            unique_attrs: list[dict] = []
            seen_attr_fingerprints: set[str] = set()
            for attrs in group_attrs:
                attr_fingerprint = json.dumps(
                    attrs, sort_keys=True, ensure_ascii=False, default=str
                )
                if attr_fingerprint in seen_attr_fingerprints:
                    multigraph_diagnostics["exact_duplicate_edges"] += 1
                else:
                    seen_attr_fingerprints.add(attr_fingerprint)
                    unique_attrs.append(attrs)
            if len(unique_attrs) > 1:
                multigraph_diagnostics["key_collision_edges"] += 1
                unique_attrs.sort(
                    key=lambda attrs: json.dumps(
                        attrs, sort_keys=True, ensure_ascii=False, default=str
                    )
                )
                multi_groups.append((src, tgt, base_key, unique_attrs))
            elif unique_attrs:
                # Reserve the singleton's base_key so any later multi-attr
                # collision-repair on the same (src, tgt) avoids it.
                used_keys_by_pair.setdefault((src, tgt), set()).add(base_key)
                singleton_groups.append((src, tgt, base_key, unique_attrs[0]))
        # Sort both lists deterministically.
        singleton_groups.sort(
            key=lambda item: (
                repr(item[0]),
                repr(item[1]),
                item[2],
                json.dumps(item[3], sort_keys=True, ensure_ascii=False, default=str),
            )
        )
        multi_groups.sort(
            key=lambda item: (
                repr(item[0]),
                repr(item[1]),
                item[2],
                json.dumps(item[3], sort_keys=True, ensure_ascii=False, default=str),
            )
        )
        # Emit singletons first: they use base_key directly and were reserved
        # in the pre-loop above, so collision-repair from multi groups will
        # see those reservations and salt around them.
        for src, tgt, base_key, attrs in singleton_groups:
            G.add_edge(src, tgt, key=base_key, **attrs)
        # Then emit multi-attr groups with collision-repair salting against
        # both reserved singleton base_keys and earlier multi-group repair
        # keys on the same (src, tgt) pair.
        for src, tgt, base_key, unique_attrs in multi_groups:
            used_keys = used_keys_by_pair.setdefault((src, tgt), set())
            preserve_explicit = (src, tgt, base_key) in multigraph_explicit_keys
            for index, attrs in enumerate(unique_attrs):
                # When the user passed an explicit `key` shared across multiple
                # distinct edges, preserve it on the first emit so at least one
                # edge per group keeps the canonical user-supplied key.
                # Derived base_keys (from make_stable_key) always go through
                # collision-repair so emission stays order-independent.
                if preserve_explicit and index == 0 and base_key not in used_keys:
                    key = base_key
                else:
                    key = _make_collision_key(base_key, attrs)
                    salt = 0
                    while key in used_keys:
                        salt += 1
                        key = _make_collision_key(base_key, attrs, salt=salt)
                used_keys.add(key)
                G.add_edge(src, tgt, key=key, **attrs)
    hyperedges = _repair_hyperedge_file_node_aliases(G, extraction.get("hyperedges", []))
    if hyperedges:
        # Relativize hyperedge source_file the same way nodes and edges are
        # (above), so to_json — which has no root and writes G.graph["hyperedges"]
        # verbatim — never leaks an absolute path from a semantic subagent (#1418).
        for he in hyperedges:
            if isinstance(he, dict) and he.get("source_file"):
                he["source_file"] = _norm_source_file(he["source_file"], _root)
        G.graph["hyperedges"] = list(reconcile_hyperedges(hyperedges, live_node_ids=set(G.nodes)))
    else:
        G.graph["hyperedges"] = []
    if multigraph:
        G.graph["graphify_multigraph_diagnostics"] = multigraph_diagnostics
    G.graph["graphify_identity_diagnostics"] = identity_diagnostics
    return G


def build(
    extractions: list[dict],
    *,
    directed: bool = False,
    dedup: bool = True,
    dedup_llm_backend: str | None = None,
    root: str | Path | None = None,
    multigraph: bool = False,
) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph:
    """Merge multiple extraction results into one graph.

    directed=True produces a DiGraph that preserves edge direction (source→target).
    directed=False (default) produces an undirected Graph for backward compatibility.
    dedup=True (default) runs entity deduplication before building the graph.
    dedup_llm_backend: if set (e.g. "gemini", "claude", or "kimi"), uses LLM to resolve
        ambiguous pairs in the 75–92 Jaro-Winkler score zone.
    root: if given, absolute source_file paths are made relative to root (#932).

    Extractions are merged in order. For nodes with the same ID, the last
    extraction's attributes win (NetworkX add_node overwrites). Pass AST
    results before semantic results so semantic labels take precedence, or
    reverse the order if you prefer AST source_location precision to win.
    """
    from graphify.dedup import deduplicate_entities

    combined = _combine_extractions(extractions)
    dedup_diagnostics: dict = {}
    dedup_node_remap: dict[str, str] = {}
    if dedup and combined["nodes"]:
        combined["nodes"], combined["edges"] = deduplicate_entities(
            combined["nodes"],
            combined["edges"],
            communities={},
            dedup_llm_backend=dedup_llm_backend,
            diagnostics=dedup_diagnostics,
            node_remap=dedup_node_remap,
        )
        combined["hyperedges"] = list(
            reconcile_hyperedges(
                combined["hyperedges"],
                live_node_ids={node["id"] for node in combined["nodes"]},
                node_remap=dedup_node_remap,
            )
        )
    G = build_from_json(combined, directed=directed, root=root, multigraph=multigraph)
    if multigraph and dedup_diagnostics:
        existing = G.graph.get("graphify_multigraph_diagnostics", {})
        existing.update(dedup_diagnostics)
        G.graph["graphify_multigraph_diagnostics"] = existing
    return G


def _combine_extractions(extractions: list[dict]) -> dict:
    combined: dict = {
        "nodes": [],
        "edges": [],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for ext in extractions:
        combined["nodes"].extend(ext.get("nodes", []))
        combined["edges"].extend(ext.get("edges", []))
        combined["hyperedges"].extend(ext.get("hyperedges", []))
        combined["input_tokens"] += ext.get("input_tokens", 0)
        combined["output_tokens"] += ext.get("output_tokens", 0)
    return combined


def _norm_label(label: str | None) -> str:
    """Canonical dedup key — Unicode-aware, preserves CJK/word characters."""
    if not isinstance(label, str):
        label = "" if label is None else str(label)
    label = unicodedata.normalize("NFKC", label)
    return re.sub(r"[\W_ ]+", " ", label.casefold(), flags=re.UNICODE).strip()


def deduplicate_by_label(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge nodes that share a normalised label, rewriting edge references.

    Prefers IDs without chunk suffixes (_c\\d+) and shorter IDs when tied.
    Drops self-loops created by the merge.

    Dormant: this is NOT wired into ``build()`` — the active dedup path is
    ``deduplicate_entities`` (imported and called in ``build``), which supersedes
    it. The previous "Called in build() automatically" note was never true. It
    also merges by label alone with no ``file_type`` guard, so it must not be
    enabled for code nodes: same-label symbols from different files/packages
    (e.g. two ``Account`` types) would collapse into one — the cross-file
    conflation ``deduplicate_entities`` deliberately avoids for code (#1205).
    """
    _CHUNK_SUFFIX = re.compile(r"_c\d+$")
    canonical: dict[str, dict] = {}  # norm_label -> surviving node
    remap: dict[str, str] = {}  # old_id -> surviving_id

    for node in nodes:
        key = _norm_label(node.get("label", node.get("id", "")))
        if not key:
            continue
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = node
        else:
            has_suffix = bool(_CHUNK_SUFFIX.search(node["id"]))
            existing_has_suffix = bool(_CHUNK_SUFFIX.search(existing["id"]))
            if has_suffix and not existing_has_suffix:
                remap[node["id"]] = existing["id"]
            elif existing_has_suffix and not has_suffix:
                remap[existing["id"]] = node["id"]
                canonical[key] = node
            elif len(node["id"]) < len(existing["id"]):
                remap[existing["id"]] = node["id"]
                canonical[key] = node
            else:
                remap[node["id"]] = existing["id"]

    if not remap:
        return nodes, edges

    print(f"[graphify] Deduplicated {len(remap)} duplicate node(s) by label.", file=sys.stderr)
    deduped_nodes = list(canonical.values())
    deduped_edges = []
    for edge in edges:
        e = dict(edge)
        e["source"] = remap.get(e["source"], e["source"])
        e["target"] = remap.get(e["target"], e["target"])
        if e["source"] != e["target"]:
            deduped_edges.append(e)
    return deduped_nodes, deduped_edges


def _chunk_has_graph_records(chunk: dict) -> bool:
    return bool(
        chunk.get("nodes") or chunk.get("edges") or chunk.get("links") or chunk.get("hyperedges")
    )


def build_merge(
    new_chunks: list[dict],
    graph_path: str | Path | None = None,
    prune_sources: list[str] | None = None,
    *,
    directed: bool | None = None,
    dedup: bool = True,
    dedup_llm_backend: str | None = None,
    root: str | Path | None = None,
    multigraph: bool | None = None,
) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph:
    """Load existing graph.json, merge new chunks into it, and return the merged graph.

    Persistence is the caller's responsibility (e.g., via ``export.to_json``);
    this function does not write back to disk.

    Re-extracted files REPLACE their prior contribution: any source_file present
    in new_chunks is dropped from the loaded graph before merging, so a changed
    file's stale nodes/edges don't accumulate. Files absent from new_chunks are
    preserved unchanged; deleted files are removed via prune_sources.
    Safe to call repeatedly.
    root: if given, absolute source_file paths in new_chunks are made relative (#932).

    ``directed`` defaults to inheriting the saved graph's flag when an
    existing graph.json is present, so updating a directed simple graph with
    default args no longer silently downgrades it to undirected.

    ``multigraph`` likewise defaults to inheriting the saved graph's flag. When
    the saved graph.json has ``multigraph: true`` the merge produces a
    MultiDiGraph that preserves keyed parallel edges end-to-end — existing edges
    keep their stored ``key`` (so distinct parallel edges between the same pair
    survive the re-feed), new chunks are merged without collapsing parallels, and
    the result round-trips back out as multigraph. There is no silent fallback to
    simple-graph behavior.
    """
    graph_path = Path(graph_path if graph_path is not None else _default_graph_json())
    existing_state = None
    if graph_path.exists():
        from graphify.graph_loader import load_graph_state_file
        from graphify.graph_state import DecodeMode

        try:
            state = load_graph_state_file(graph_path, mode=DecodeMode.MIGRATE_LEGACY)
        except GraphStateError as exc:
            raise RuntimeError(
                f"Cannot read {graph_path} for incremental merge: {exc}. "
                "Delete the file and run a full rebuild."
            ) from exc
        saved_multigraph = state.graph_type == "multidigraph"
        if multigraph is None:
            multigraph = saved_multigraph
        elif multigraph != saved_multigraph:
            print(
                f"[graphify] WARNING: build_merge multigraph={multigraph} overrides "
                f"saved graph.json multigraph={saved_multigraph}",
                file=sys.stderr,
            )
        saved_directed = state.graph_type in {"digraph", "multidigraph"}
        if directed is None:
            directed = saved_directed
        elif directed != saved_directed:
            print(
                f"[graphify] WARNING: build_merge directed={directed} overrides "
                f"saved graph.json directed={saved_directed}",
                file=sys.stderr,
            )
        existing_nodes = list(state.nodes)
        existing_edges = list(state.edges)
        existing_hyperedges = list(state.hyperedges)
        existing_state = state
        had_graph = True
    else:
        if directed is None:
            directed = False
        if multigraph is None:
            multigraph = False
        existing_nodes = []
        existing_edges = []
        existing_hyperedges = []
        had_graph = False

    # Effective root for relativizing absolute source_file / prune paths back to the
    # stored relative source_file keys. When the caller passes root we use it;
    # otherwise fall back to the graph's recorded scan root, so absolute
    # prune_sources and new-chunk paths still match even when a caller omits root
    # (#1571 — the skill's --update runbook calls build_merge without root, so
    # absolute deleted-file paths never matched the relative node keys and their
    # nodes survived as ghosts).
    _eff_root = str(Path(root).resolve()) if root is not None else _infer_merge_root(graph_path)

    # Re-extracted files REPLACE their prior contribution. Every source_file
    # present in new_chunks is dropped from the loaded base before merging, so a
    # CHANGED file's stale nodes/edges don't accumulate across incremental
    # updates. Without this, build() merges old+new for the same file and only
    # exact-duplicate edges collapse — edges/nodes that disappeared from the new
    # version survive forever. Brand-new files aren't in base, so this is a no-op
    # for them; genuinely deleted files are still handled via prune_sources.
    # Matched in both raw and _norm_source_file form because new_chunks may carry
    # absolute win32 paths while the stored graph keeps relative posix (#1007).
    _replace_root = _eff_root
    new_sources: set[str] = set()
    for ch in new_chunks:
        for n in ch.get("nodes", []):
            sf = n.get("source_file")
            if not sf:
                continue
            new_sources.add(sf)
            norm = _norm_source_file(sf, _replace_root)
            if norm:
                new_sources.add(norm)
    if new_sources:

        def _kept(item: dict) -> bool:
            sf = item.get("source_file")
            return sf not in new_sources and _norm_source_file(sf, _replace_root) not in new_sources

        existing_nodes = [n for n in existing_nodes if _kept(n)]
        existing_edges = [e for e in existing_edges if _kept(e)]

    base = [{"nodes": existing_nodes, "edges": existing_edges}] if had_graph else []

    incoming_chunks = list(new_chunks)
    incoming_has_records = any(_chunk_has_graph_records(chunk) for chunk in incoming_chunks)
    dedup_diagnostics: dict = {}
    if graph_path.exists() and dedup:
        effective_dedup = False
        if incoming_has_records:
            from graphify.dedup import deduplicate_entities

            incoming = _combine_extractions(incoming_chunks)
            if incoming["nodes"]:
                incoming_remap: dict[str, str] = {}
                incoming["nodes"], incoming["edges"] = deduplicate_entities(
                    incoming["nodes"],
                    incoming["edges"],
                    communities={},
                    dedup_llm_backend=dedup_llm_backend,
                    diagnostics=dedup_diagnostics,
                    node_remap=incoming_remap,
                )
                incoming["hyperedges"] = list(
                    reconcile_hyperedges(
                        incoming["hyperedges"],
                        live_node_ids={node["id"] for node in existing_nodes + incoming["nodes"]},
                        node_remap=incoming_remap,
                    )
                )
            all_chunks = base + [incoming]
        else:
            all_chunks = base + incoming_chunks
    else:
        effective_dedup = dedup
        all_chunks = base + incoming_chunks
    G = build(
        all_chunks,
        directed=directed,
        dedup=effective_dedup,
        dedup_llm_backend=dedup_llm_backend,
        root=root,
        multigraph=multigraph,
    )
    if multigraph and dedup_diagnostics:
        existing = G.graph.get("graphify_multigraph_diagnostics", {})
        existing.update(dedup_diagnostics)
        G.graph["graphify_multigraph_diagnostics"] = existing

    # Prune set for deleted source files — both the raw form (matches nodes that
    # kept absolute source_file) and the normalised relative form (matches nodes
    # relativised by _norm_source_file at build time). .resolve() (via _eff_root)
    # handles symlinked roots and ".." / "./" segments so Path.relative_to()
    # succeeds even when the scan root is a symlink. (#1007, #1571)
    prune_set: set[str] = set()
    for p in prune_sources or []:
        if not p:
            continue
        prune_set.add(p)
        norm = _norm_source_file(p, _eff_root)
        if norm:
            prune_set.add(norm)

    # Carry forward hyperedges from files that were neither re-extracted nor
    # deleted (#1574). build() only sees the new chunks' hyperedges, so without
    # this every --update collapses the graph's hyperedge set down to just the
    # changed files'. Re-extracted files' prior hyperedges are dropped (their new
    # version is already in G — replace-per-source, like nodes/edges); deleted
    # files' are dropped via prune_set. id-dedup (attach_hyperedges) so a carried
    # hyperedge never duplicates one the new chunks re-emitted. Mirrors watch.py,
    # which already preserves existing hyperedges across a rebuild.
    if existing_hyperedges:
        carried = []
        for he in existing_hyperedges:
            if not isinstance(he, dict):
                continue
            sf_value = he.get("source_file")
            sf = str(sf_value) if sf_value is not None else None
            norm = _norm_source_file(sf, _eff_root)
            if sf in new_sources or norm in new_sources:
                continue  # re-extracted — replaced by the new chunk's version
            if sf in prune_set or norm in prune_set:
                continue  # deleted — pruned
            carried.append(he)
        if carried:
            from graphify.export import attach_hyperedges

            attach_hyperedges(G, _repair_hyperedge_file_node_aliases(G, carried))

    # Prune nodes and edges from deleted source files
    if prune_sources:
        to_remove = [n for n, d in G.nodes(data=True) if d.get("source_file") in prune_set]
        G.remove_nodes_from(to_remove)
        n_files = len(prune_sources)
        n_nodes = len(to_remove)
        if n_nodes:
            print(
                f"[graphify] Pruned {n_nodes} node(s) from {n_files} deleted source file(s).",
                file=sys.stderr,
            )

        # Prune edges belonging to changed/deleted source files. On a
        # MultiDiGraph a single (u, v) pair can carry MULTIPLE parallel edges
        # from DIFFERENT source files, so removal MUST be keyed: drop only the
        # parallel edges whose source_file is in prune_set and leave parallel
        # edges from other files between the same pair intact. The two-tuple
        # remove_edges_from used by simple graphs would drop only one edge per
        # pair on a multigraph (first key) and could evict the wrong file's edge.
        # remove_all_parallel_edges is deliberately NOT used here — it is too
        # broad and would delete other-file parallels between the same pair.
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            keyed_to_remove = [
                (u, v, k)
                for u, v, k, d in G.edges(keys=True, data=True)
                if d.get("source_file") in prune_set
            ]
            for u, v, k in keyed_to_remove:
                G.remove_edge(u, v, key=k)
            n_edges_removed = len(keyed_to_remove)
        else:
            edges_to_remove = [
                (u, v) for u, v, d in G.edges(data=True) if d.get("source_file") in prune_set
            ]
            if edges_to_remove:
                G.remove_edges_from(edges_to_remove)
            n_edges_removed = len(edges_to_remove)
        if n_edges_removed:
            print(
                f"[graphify] Pruned {n_edges_removed} edge(s) from deleted source file(s).",
                file=sys.stderr,
            )

        if not n_nodes and not n_edges_removed:
            print(
                f"[graphify] {n_files} source file(s) deleted since last run — "
                f"no matching nodes or edges in graph, already clean.",
                file=sys.stderr,
            )

    G.graph["hyperedges"] = list(
        reconcile_hyperedges(G.graph.get("hyperedges", []), live_node_ids=set(G.nodes))
    )

    if existing_state is not None:
        from graphify.graph_state import (
            TOP_LEVEL_METADATA_CARRIER,
            merge_graph_metadata,
        )

        hyperedges = G.graph.get("hyperedges", [])
        graph_type = (
            "multidigraph" if G.is_multigraph() else "digraph" if G.is_directed() else "simple"
        )
        merged_metadata = merge_graph_metadata(
            existing_state.graph_metadata,
            G.graph,
            target_type=graph_type,
        )
        G.graph.clear()
        G.graph.update(merged_metadata)
        G.graph["hyperedges"] = hyperedges
        G.graph[TOP_LEVEL_METADATA_CARRIER] = dict(existing_state.top_level_metadata)

    # Safety check: refuse to shrink the graph silently (#479).
    # Stateful dedup applies only to incoming chunks, so only explicit pruning
    # may reduce the saved graph's node count.
    if graph_path.exists() and not prune_sources:
        existing_n = len(existing_nodes)
        new_n = G.number_of_nodes()
        if new_n < existing_n:
            raise ValueError(
                f"graphify: build_merge would shrink graph from {existing_n} → {new_n} nodes. "
                f"Pass prune_sources explicitly if you intend to remove nodes."
            )

    # No write to graph_path here; persistence is the caller's responsibility.
    return G


def prefix_graph_for_global(G: nx.Graph, repo_tag: str) -> nx.Graph:
    """Return a copy of G with all node IDs prefixed with repo_tag::.

    Labels are preserved unchanged (for display). A 'local_id' attribute
    is added to each node so the original ID can be recovered. Edges are
    rewritten to match the new prefixed IDs. The 'repo' attribute is set
    on every node.
    """
    relabel = {n: f"{repo_tag}::{n}" for n in G.nodes}
    H = nx.relabel_nodes(G, relabel, copy=True)
    for node, data in H.nodes(data=True):
        data["repo"] = repo_tag
        data.setdefault("local_id", node.split("::", 1)[1])
    H.graph["hyperedges"] = list(
        reconcile_hyperedges(
            G.graph.get("hyperedges", []),
            live_node_ids=set(H.nodes),
            node_remap=relabel,
        )
    )
    return H


def prune_repo_from_graph(G: nx.Graph, repo_tag: str) -> int:
    """Remove all nodes tagged with repo_tag from G in-place. Returns count removed."""
    to_remove = [n for n, d in G.nodes(data=True) if d.get("repo") == repo_tag]
    G.remove_nodes_from(to_remove)
    G.graph["hyperedges"] = list(
        reconcile_hyperedges(G.graph.get("hyperedges", []), live_node_ids=set(G.nodes))
    )
    return len(to_remove)
