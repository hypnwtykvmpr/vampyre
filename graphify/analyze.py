"""Graph analysis: god nodes (most connected), surprising connections (cross-community), suggested questions."""

from __future__ import annotations
from collections.abc import Hashable
from pathlib import Path
import networkx as nx

from graphify.build import edge_data
from graphify.detect import CODE_EXTENSIONS, IMAGE_EXTENSIONS, PAPER_EXTENSIONS
from graphify.projections import distinct_neighbor_degree, project_for_community, stable_value_key

# Builtin/mock names that can appear as annotation-derived nodes in pre-existing
# graphs. Excluded from god-node ranking so they don't displace real abstractions
# even if they weren't filtered at extraction time (#1147).
_BUILTIN_NOISE_LABELS = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "object",
        "True",
        "False",
        "MagicMock",
        "Mock",
        "AsyncMock",
        "NonCallableMock",
        "NonCallableMagicMock",
        "PropertyMock",
        "patch",
        "sentinel",
        # Python stdlib types commonly confused for project symbols
        "Path",
        "Any",
        "Optional",
        "List",
        "Dict",
        "Set",
        "Tuple",
        "Union",
        "Callable",
        "Type",
        "ClassVar",
        "Final",
        "Literal",
        "Protocol",
        "Counter",
        "defaultdict",
        "OrderedDict",
        "datetime",
        "Enum",
        "os",
        "sys",
        "re",
        "json",
        "io",
        "abc",
        "typing",
    }
)

# Language families — extensions sharing a runtime can legitimately call each other
_LANG_FAMILY: dict[str, str] = {
    **{e: "python" for e in (".py", ".pyw")},
    **{
        e: "js"
        for e in (".js", ".jsx", ".mjs", ".ejs", ".ts", ".tsx", ".mts", ".cts", ".vue", ".svelte")
    },
    **{e: "go" for e in (".go",)},
    **{e: "rust" for e in (".rs",)},
    **{e: "jvm" for e in (".java", ".kt", ".kts", ".scala")},
    **{e: "c" for e in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp")},
    **{e: "ruby" for e in (".rb",)},
    **{e: "swift" for e in (".swift",)},
    **{e: "dotnet" for e in (".cs",)},
    **{e: "php" for e in (".php",)},
    **{e: "r" for e in (".r",)},
}


def _cross_language(src_a: str, src_b: str) -> bool:
    """Return True if two source files belong to different language families."""
    ext_a = Path(src_a).suffix.lower()
    ext_b = Path(src_b).suffix.lower()
    fam_a = _LANG_FAMILY.get(ext_a)
    fam_b = _LANG_FAMILY.get(ext_b)
    if fam_a is None or fam_b is None:
        return False
    return fam_a != fam_b


def _node_community_map(communities: dict[int, list[str]]) -> dict[str, int]:
    """Invert communities dict: node_id -> community_id."""
    return {n: cid for cid, nodes in communities.items() for n in nodes}


def _is_file_node(G: nx.Graph, node_id: Hashable) -> bool:
    """
    Return True if this node is a file-level hub node (e.g. 'client', 'models')
    or an AST method stub (e.g. '.auth_flow()', '.__init__()').

    These are synthetic nodes created by the AST extractor and should be excluded
    from god nodes, surprising connections, and knowledge gap reporting.
    """
    attrs = G.nodes[node_id]
    label = attrs.get("label", "")
    if not label:
        return False
    # File-level hub: label matches the actual source filename (not just any label ending in .py)
    source_file = attrs.get("source_file", "")
    if source_file:
        from pathlib import Path as _Path

        if label == _Path(source_file).name:
            return True
    # Method stub: AST extractor labels methods as '.method_name()'
    if label.startswith(".") and label.endswith("()"):
        return True
    # Module-level function stub: labeled 'function_name()' - only has a contains edge
    # These are real functions but structurally isolated by definition; not a gap worth flagging
    if label.endswith("()") and distinct_neighbor_degree(G, node_id) <= 1:
        return True
    return False


_JSON_NOISE_LABELS: frozenset[str] = frozenset(
    {
        "start",
        "end",
        "name",
        "id",
        "type",
        "properties",
        "value",
        "key",
        "data",
        "items",
        "title",
        "description",
        "version",
        "dependencies",
        "devdependencies",
        "peerdependencies",
        "optionaldependencies",
        "bundleddependencies",
        "bundledependencies",
    }
)


def _is_json_key_node(G: nx.Graph, node_id: str) -> bool:
    attrs = G.nodes[node_id]
    src = (attrs.get("source_file") or "").lower()
    if not src.endswith(".json"):
        return False
    label = (attrs.get("label") or "").strip().lower()
    return label in _JSON_NOISE_LABELS


def god_nodes(G: nx.Graph, top_n: int = 10) -> list[dict]:
    """Return the top_n most-connected real entities - the core abstractions.

    File-level hub nodes are excluded: they accumulate import/contains edges
    mechanically and don't represent meaningful architectural abstractions.
    """
    degree = {n: distinct_neighbor_degree(G, n) for n in G.nodes()}
    sorted_nodes = sorted(degree.items(), key=lambda item: (-item[1], stable_value_key(item[0])))
    result = []
    for node_id, deg in sorted_nodes:
        if (
            _is_file_node(G, node_id)
            or _is_concept_node(G, node_id)
            or _is_json_key_node(G, node_id)
        ):
            continue
        if G.nodes[node_id].get("label", "") in _BUILTIN_NOISE_LABELS:
            continue
        result.append(
            {
                "id": node_id,
                "label": G.nodes[node_id].get("label", node_id),
                "degree": deg,
            }
        )
        if len(result) >= top_n:
            break
    return result


def surprising_connections(
    G: nx.Graph,
    communities: dict[int, list[str]] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Find connections that are genuinely surprising - not obvious from file structure.

    Strategy:
    - Multi-file corpora: cross-file edges between real entities (not concept nodes).
      Sorted AMBIGUOUS → INFERRED → EXTRACTED.
    - Single-file / single-source corpora: cross-community edges that bridge
      distant parts of the graph (betweenness centrality on edges).
      These reveal non-obvious structural couplings.

    Concept nodes (empty source_file, or injected semantic annotations) are excluded
    from surprising connections because they are intentional, not discovered.
    """
    # Identify unique source files (ignore empty/null source_file)
    source_files = {
        data.get("source_file", "") for _, data in G.nodes(data=True) if data.get("source_file", "")
    }
    is_multi_source = len(source_files) > 1

    if is_multi_source:
        return _cross_file_surprises(G, communities or {}, top_n)
    else:
        return _cross_community_surprises(G, communities or {}, top_n)


def _is_concept_node(G: nx.Graph, node_id: str) -> bool:
    """
    Return True if this node is a manually-injected semantic concept node
    rather than a real entity found in source code.

    Signals:
    - Empty source_file
    - source_file doesn't look like a real file path (no extension)
    """
    data = G.nodes[node_id]
    source = data.get("source_file", "")
    if not source:
        return True
    # Has no file extension → probably a concept label, not a real file
    if "." not in source.split("/")[-1]:
        return True
    return False


def _file_category(path: str) -> str:
    ext = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path else ""
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in PAPER_EXTENSIONS:
        return "paper"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "doc"


def _top_level_dir(path: str) -> str:
    """Return the first path component - used to detect cross-repo edges."""
    return path.split("/")[0] if "/" in path else path


def _surprise_score(
    G: nx.Graph,
    u: str,
    v: str,
    data: dict,
    node_community: dict[str, int],
    u_source: str,
    v_source: str,
    degrees: dict[str, int] | None = None,
) -> tuple[int, list[str]]:
    """Score how surprising a cross-file edge is. Returns (score, reasons)."""
    score = 0
    reasons: list[str] = []

    # 1. Confidence weight - uncertain connections are more noteworthy
    conf = data.get("confidence", "EXTRACTED")
    relation = data.get("relation", "")
    conf_bonus = {"AMBIGUOUS": 3, "INFERRED": 2, "EXTRACTED": 1}.get(conf, 1)

    cat_u = _file_category(u_source)
    cat_v = _file_category(v_source)

    # Suppress all structural bonuses for INFERRED calls/uses that cross language
    # boundaries or connect code to a doc file.  Both cases are resolver pollution:
    # label-matching fires across language families in monorepos, and code→doc
    # "calls" edges are extraction artefacts, not real architecture.
    # Excludes `semantically_similar_to` (genuine cross-boundary insight) and all
    # AMBIGUOUS/EXTRACTED edges (not from the resolver path).
    _suppress_structural = (
        conf == "INFERRED"
        and relation in ("calls", "uses")
        and (_cross_language(u_source, v_source) or {cat_u, cat_v} == {"code", "doc"})
    )
    if _suppress_structural:
        conf_bonus = 0

    score += conf_bonus
    if conf in ("AMBIGUOUS", "INFERRED"):
        reasons.append(f"{conf.lower()} connection - not explicitly stated in source")

    # 2. Cross file-type bonus - code↔paper or code↔image is non-obvious
    if cat_u != cat_v and not _suppress_structural:
        score += 2
        reasons.append(f"crosses file types ({cat_u} ↔ {cat_v})")

    # 3. Cross-repo bonus - different top-level directory
    if _top_level_dir(u_source) != _top_level_dir(v_source) and not _suppress_structural:
        score += 2
        reasons.append("connects across different repos/directories")

    # 4. Cross-community bonus - Leiden says these are structurally distant
    cid_u = node_community.get(u)
    cid_v = node_community.get(v)
    if cid_u is not None and cid_v is not None and cid_u != cid_v and not _suppress_structural:
        score += 1
        reasons.append("bridges separate communities")

    # 4b. Semantic similarity bonus - non-obvious conceptual links score higher
    if data.get("relation") == "semantically_similar_to":
        score = int(score * 1.5)
        reasons.append("semantically similar concepts with no structural link")

    # 5. Peripheral→hub: a low-degree node connecting to a high-degree one
    deg_u = degrees[u] if degrees is not None else distinct_neighbor_degree(G, u)
    deg_v = degrees[v] if degrees is not None else distinct_neighbor_degree(G, v)
    if min(deg_u, deg_v) <= 2 and max(deg_u, deg_v) >= 5:
        score += 1
        peripheral = G.nodes[u].get("label", u) if deg_u <= 2 else G.nodes[v].get("label", v)
        hub = G.nodes[v].get("label", v) if deg_u <= 2 else G.nodes[u].get("label", u)
        reasons.append(f"peripheral node `{peripheral}` unexpectedly reaches hub `{hub}`")

    return score, reasons


def _cross_file_surprises(G: nx.Graph, communities: dict[int, list[str]], top_n: int) -> list[dict]:
    """
    Cross-file edges between real code/doc entities, ranked by a composite
    surprise score rather than confidence alone.

    Surprise score accounts for:
    - Confidence (AMBIGUOUS > INFERRED > EXTRACTED)
    - Cross file-type (code↔paper is more surprising than code↔code)
    - Cross-repo (different top-level directory)
    - Cross-community (Leiden says structurally distant)
    - Peripheral→hub (low-degree node reaching a god node)

    Each result includes a 'why' field explaining what makes it non-obvious.
    """
    node_community = _node_community_map(communities)
    degrees = {n: distinct_neighbor_degree(G, n) for n in G.nodes()}
    candidates = []

    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        if relation in ("imports", "imports_from", "contains", "method"):
            continue
        if _is_concept_node(G, u) or _is_concept_node(G, v):
            continue
        if _is_file_node(G, u) or _is_file_node(G, v):
            continue

        u_source = G.nodes[u].get("source_file", "")
        v_source = G.nodes[v].get("source_file", "")

        if not u_source or not v_source or u_source == v_source:
            continue

        score, reasons = _surprise_score(G, u, v, data, node_community, u_source, v_source, degrees)
        src_id = data.get("_src", u)
        if src_id not in G.nodes:
            src_id = u
        tgt_id = data.get("_tgt", v)
        if tgt_id not in G.nodes:
            tgt_id = v
        candidates.append(
            {
                "_score": score,
                "_identity": (
                    stable_value_key(src_id),
                    stable_value_key(tgt_id),
                    stable_value_key(relation),
                    stable_value_key(data.get("source_file")),
                    stable_value_key(data.get("source_location")),
                    stable_value_key(data.get("context")),
                    stable_value_key(data.get("confidence")),
                ),
                "source": G.nodes[src_id].get("label", src_id),
                "target": G.nodes[tgt_id].get("label", tgt_id),
                "source_files": [
                    G.nodes[src_id].get("source_file", ""),
                    G.nodes[tgt_id].get("source_file", ""),
                ],
                "confidence": data.get("confidence", "EXTRACTED"),
                "relation": relation,
                "why": "; ".join(reasons) if reasons else "cross-file semantic connection",
            }
        )

    candidates.sort(key=lambda item: (-item["_score"], item["_identity"]))
    for c in candidates:
        c.pop("_score")
        c.pop("_identity")

    if candidates:
        return candidates[:top_n]

    return _cross_community_surprises(G, communities, top_n)


def _cross_community_surprises(
    G: nx.Graph,
    communities: dict[int, list[str]],
    top_n: int,
) -> list[dict]:
    """
    For single-source corpora: find edges that bridge different communities.
    These are surprising because Leiden grouped everything else tightly -
    these edges cut across the natural structure.

    Falls back to high-betweenness edges if no community info is provided.
    """
    if not communities:
        # No community info - use edge betweenness centrality
        if G.number_of_edges() == 0:
            return []
        if G.number_of_nodes() > 5000:
            return []
        # Project to simple graph so betweenness returns 2-tuple keys
        # and parallel edges don't inflate centrality scores.
        simple = project_for_community(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G
        betweenness = nx.edge_betweenness_centrality(simple)
        top_edges = sorted(
            betweenness.items(),
            key=lambda item: (
                -item[1],
                stable_value_key(item[0][0]),
                stable_value_key(item[0][1]),
            ),
        )[:top_n]
        result = []
        for (u, v), score in top_edges:
            data = edge_data(G, u, v)
            result.append(
                {
                    "source": G.nodes[u].get("label", u),
                    "target": G.nodes[v].get("label", v),
                    "source_files": [
                        G.nodes[u].get("source_file", ""),
                        G.nodes[v].get("source_file", ""),
                    ],
                    "confidence": data.get("confidence", "EXTRACTED"),
                    "relation": data.get("relation", ""),
                    "note": f"Bridges graph structure (betweenness={score:.3f})",
                }
            )
        return result

    # Build node → community map
    node_community = _node_community_map(communities)

    surprises = []
    for u, v, data in G.edges(data=True):
        cid_u = node_community.get(u)
        cid_v = node_community.get(v)
        if cid_u is None or cid_v is None or cid_u == cid_v:
            continue
        # Skip file hub nodes and plain structural edges
        if _is_file_node(G, u) or _is_file_node(G, v):
            continue
        relation = data.get("relation", "")
        if relation in ("imports", "imports_from", "contains", "method"):
            continue
        # This edge crosses community boundaries - interesting
        confidence = data.get("confidence", "EXTRACTED")
        src_id = data.get("_src", u)
        if src_id not in G.nodes:
            src_id = u
        tgt_id = data.get("_tgt", v)
        if tgt_id not in G.nodes:
            tgt_id = v
        surprises.append(
            {
                "source": G.nodes[src_id].get("label", src_id),
                "target": G.nodes[tgt_id].get("label", tgt_id),
                "source_files": [
                    G.nodes[src_id].get("source_file", ""),
                    G.nodes[tgt_id].get("source_file", ""),
                ],
                "confidence": confidence,
                "relation": relation,
                "note": f"Bridges community {cid_u} → community {cid_v}",
                "_pair": tuple(sorted([cid_u, cid_v])),
                "_identity": (
                    stable_value_key(src_id),
                    stable_value_key(tgt_id),
                    stable_value_key(relation),
                    stable_value_key(data.get("source_file")),
                    stable_value_key(data.get("source_location")),
                    stable_value_key(data.get("context")),
                    stable_value_key(confidence),
                ),
            }
        )

    # Sort: AMBIGUOUS first, then INFERRED, then EXTRACTED
    order = {"AMBIGUOUS": 0, "INFERRED": 1, "EXTRACTED": 2}
    surprises.sort(key=lambda item: (order.get(item["confidence"], 3), item["_identity"]))

    # Deduplicate by community pair - one representative edge per (A→B) boundary.
    # Without this, a single high-betweenness god node dominates all results.
    seen_pairs: set[tuple] = set()
    deduped = []
    for s in surprises:
        pair = s.pop("_pair")
        s.pop("_identity")
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduped.append(s)
    return deduped[:top_n]


def suggest_questions(
    G: nx.Graph,
    communities: dict[int, list[str]],
    community_labels: dict[int, str],
    top_n: int = 7,
) -> list[dict]:
    """
    Generate questions the graph is uniquely positioned to answer.
    Based on: AMBIGUOUS edges, bridge nodes, underexplored god nodes, isolated nodes.
    Each question has a 'type', 'question', and 'why' field.
    """
    if community_labels:
        community_labels = {
            int(k) if isinstance(k, str) else k: v for k, v in community_labels.items()
        }

    questions = []
    node_community = _node_community_map(communities)

    # 1. AMBIGUOUS edges → unresolved relationship questions
    ambiguous_edges = sorted(
        G.edges(data=True),
        key=lambda edge: (
            stable_value_key(edge[0]),
            stable_value_key(edge[1]),
            stable_value_key(edge[2].get("relation")),
            stable_value_key(edge[2].get("source_location")),
            stable_value_key(edge[2].get("context")),
        ),
    )
    for u, v, data in ambiguous_edges:
        if data.get("confidence") == "AMBIGUOUS":
            ul = G.nodes[u].get("label", u)
            vl = G.nodes[v].get("label", v)
            relation = data.get("relation", "related to")
            questions.append(
                {
                    "type": "ambiguous_edge",
                    "question": f"What is the exact relationship between `{ul}` and `{vl}`?",
                    "why": f"Edge tagged AMBIGUOUS (relation: {relation}) - confidence is low.",
                }
            )

    # 2. Bridge nodes (high betweenness) → cross-cutting concern questions
    if G.number_of_edges() > 0:
        k = min(100, G.number_of_nodes()) if G.number_of_nodes() > 1000 else None
        betweenness = nx.betweenness_centrality(G, k=k, seed=42)
        # Top bridge nodes that are NOT file-level hubs
        bridges = sorted(
            [
                (n, s)
                for n, s in betweenness.items()
                if not _is_file_node(G, n) and not _is_concept_node(G, n) and s > 0
            ],
            key=lambda item: (-item[1], stable_value_key(item[0])),
        )[:3]
        for node_id, score in bridges:
            label = G.nodes[node_id].get("label", node_id)
            cid = node_community.get(node_id)
            comm_label = (
                community_labels.get(cid, f"Community {cid}") if cid is not None else "unknown"
            )
            neighbors = list(G.neighbors(node_id))
            neighbor_comms: set[int] = set()
            for neighbor in neighbors:
                neighbor_community = node_community.get(neighbor)
                if neighbor_community is not None and neighbor_community != cid:
                    neighbor_comms.add(neighbor_community)
            if neighbor_comms:
                other_labels = [
                    community_labels.get(c, f"Community {c}") for c in sorted(neighbor_comms)
                ]
                questions.append(
                    {
                        "type": "bridge_node",
                        "question": f"Why does `{label}` connect `{comm_label}` to {', '.join(f'`{other_label}`' for other_label in other_labels)}?",
                        "why": f"High betweenness centrality ({score:.3f}) - this node is a cross-community bridge.",
                    }
                )

    # 3. God nodes with many INFERRED edges → verification questions
    degree = {n: distinct_neighbor_degree(G, n) for n in G.nodes()}
    top_nodes = sorted(
        [(n, d) for n, d in degree.items() if not _is_file_node(G, n)],
        key=lambda item: (-item[1], stable_value_key(item[0])),
    )[:5]
    for node_id, _ in top_nodes:
        inferred = sorted(
            [
                (u, v, d)
                for u, v, d in G.edges(node_id, data=True)
                if d.get("confidence") == "INFERRED"
            ],
            key=lambda edge: (
                stable_value_key(edge[0]),
                stable_value_key(edge[1]),
                stable_value_key(edge[2].get("relation")),
                stable_value_key(edge[2].get("source_location")),
                stable_value_key(edge[2].get("context")),
            ),
        )
        if len(inferred) >= 2:
            label = G.nodes[node_id].get("label", node_id)
            # Use _src/_tgt to get the correct direction; fall back to v (the other node)
            others = []
            for u, v, d in inferred[:2]:
                src_id = d.get("_src", u)
                if src_id not in G.nodes:
                    src_id = u
                tgt_id = d.get("_tgt", v)
                if tgt_id not in G.nodes:
                    tgt_id = v
                other_id = tgt_id if src_id == node_id else src_id
                others.append(G.nodes[other_id].get("label", other_id))
            questions.append(
                {
                    "type": "verify_inferred",
                    "question": f"Are the {len(inferred)} inferred relationships involving `{label}` (e.g. with `{others[0]}` and `{others[1]}`) actually correct?",
                    "why": f"`{label}` has {len(inferred)} INFERRED edges - model-reasoned connections that need verification.",
                }
            )

    # 4. Isolated or weakly-connected nodes → exploration questions
    isolated = [
        n
        for n in G.nodes()
        if distinct_neighbor_degree(G, n) <= 1
        and not _is_file_node(G, n)
        and not _is_concept_node(G, n)
    ]
    isolated.sort(key=stable_value_key)
    if isolated:
        labels = [G.nodes[n].get("label", n) for n in isolated[:3]]
        questions.append(
            {
                "type": "isolated_nodes",
                "question": f"What connects {', '.join(f'`{isolated_label}`' for isolated_label in labels)} to the rest of the system?",
                "why": f"{len(isolated)} weakly-connected nodes found - possible documentation gaps or missing edges.",
            }
        )

    # 5. Low-cohesion communities → structural questions
    from .cluster import cohesion_score

    for cid, nodes in sorted(communities.items(), key=lambda item: stable_value_key(item[0])):
        score = cohesion_score(G, nodes)
        if score < 0.15 and len(nodes) >= 5:
            label = community_labels.get(cid, f"Community {cid}")
            questions.append(
                {
                    "type": "low_cohesion",
                    "question": f"Should `{label}` be split into smaller, more focused modules?",
                    "why": f"Cohesion score {score} - nodes in this community are weakly interconnected.",
                }
            )

    if not questions:
        return [
            {
                "type": "no_signal",
                "question": None,
                "why": (
                    "Not enough signal to generate questions. "
                    "This usually means the corpus has no AMBIGUOUS edges, no bridge nodes, "
                    "no INFERRED relationships, and all communities are tightly cohesive. "
                    "Add more files or run with --mode deep to extract richer edges."
                ),
            }
        ]

    return questions[:top_n]


def graph_diff(G_old: nx.Graph, G_new: nx.Graph) -> dict:
    """Compare two graph snapshots and return what changed.

    Returns:
        {
          "new_nodes": [{"id": ..., "label": ...}],
          "removed_nodes": [{"id": ..., "label": ...}],
          "new_edges": [{"source": ..., "target": ..., "relation": ..., "confidence": ...}],
          "removed_edges": [...],
          "summary": "3 new nodes, 5 new edges, 1 node removed"
        }
    """
    old_nodes = set(G_old.nodes())
    new_nodes = set(G_new.nodes())

    added_node_ids = new_nodes - old_nodes
    removed_node_ids = old_nodes - new_nodes

    new_nodes_list = [
        {"id": node, "label": G_new.nodes[node].get("label", node)}
        for node in sorted(added_node_ids, key=stable_value_key)
    ]
    removed_nodes_list = [
        {"id": node, "label": G_old.nodes[node].get("label", node)}
        for node in sorted(removed_node_ids, key=stable_value_key)
    ]

    provenance_fields = (
        "relation",
        "source_file",
        "source_location",
        "confidence",
        "confidence_score",
        "context",
        "provenance",
        "_origin",
        "_src",
        "_tgt",
    )

    def edge_records(graph: nx.Graph) -> dict[tuple, dict]:
        if isinstance(graph, nx.MultiGraph | nx.MultiDiGraph):
            rows = graph.edges(keys=True, data=True)
        else:
            rows = ((u, v, None, data) for u, v, data in graph.edges(data=True))
        records: dict[tuple, dict] = {}
        for u, v, edge_key, data in rows:
            if not graph.is_directed() and stable_value_key(v) < stable_value_key(u):
                u, v = v, u
            identity = (
                stable_value_key(u),
                stable_value_key(v),
                stable_value_key(edge_key),
                tuple(
                    (field, stable_value_key(data[field]))
                    for field in provenance_fields
                    if field in data
                ),
            )
            record = {"source": u, "target": v}
            if edge_key is not None:
                record["key"] = edge_key
            for field in provenance_fields:
                if field in data:
                    record[field] = data[field]
            record.setdefault("relation", "")
            record.setdefault("confidence", "")
            records[identity] = record
        return records

    old_edges = edge_records(G_old)
    new_edges = edge_records(G_new)
    new_edges_list = [new_edges[key] for key in sorted(new_edges.keys() - old_edges.keys())]
    removed_edges_list = [old_edges[key] for key in sorted(old_edges.keys() - new_edges.keys())]

    parts = []
    if new_nodes_list:
        parts.append(f"{len(new_nodes_list)} new node{'s' if len(new_nodes_list) != 1 else ''}")
    if new_edges_list:
        parts.append(f"{len(new_edges_list)} new edge{'s' if len(new_edges_list) != 1 else ''}")
    if removed_nodes_list:
        parts.append(
            f"{len(removed_nodes_list)} node{'s' if len(removed_nodes_list) != 1 else ''} removed"
        )
    if removed_edges_list:
        parts.append(
            f"{len(removed_edges_list)} edge{'s' if len(removed_edges_list) != 1 else ''} removed"
        )
    summary = ", ".join(parts) if parts else "no changes"

    return {
        "new_nodes": new_nodes_list,
        "removed_nodes": removed_nodes_list,
        "new_edges": new_edges_list,
        "removed_edges": removed_edges_list,
        "summary": summary,
    }


def find_import_cycles(
    G: nx.Graph,
    max_cycle_length: int = 5,
    top_n: int = 20,
) -> list[dict]:
    """Detect circular import dependencies at the file level.

    Collapses symbol-level nodes to their parent file (using source_file attr
    or 'contains' edges), builds a directed file-level graph from imports_from
    edges, then finds simple cycles.

    Args:
        G: The full knowledge graph (may be undirected or directed).
        max_cycle_length: Only report cycles with at most this many files.
        top_n: Maximum number of cycles to return (shortest first).

    Returns:
        List of cycle records with stable structure:
        {
          "cycle": ["a.ts", "b.ts"],
          "length": 2,
          "why": "circular dependency"
        }
    """

    def _endpoint_source_file(node_id: str) -> str:
        attrs = G.nodes.get(node_id, {})
        src_file = attrs.get("source_file", "")
        return src_file if isinstance(src_file, str) else ""

    # Step 1: Build a directed file-level graph from import/re-export edges.
    # IMPORTANT: resolve endpoints using source_file only; never infer from label/id.
    dependency_edges: set[tuple[str, str]] = set()

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        if rel not in ("imports_from", "re_exports"):
            continue

        src_file_attr = data.get("source_file", "")
        if not isinstance(src_file_attr, str) or not src_file_attr:
            continue

        u_file = _endpoint_source_file(u)
        v_file = _endpoint_source_file(v)

        # Works for both DiGraph and Graph inputs:
        # orient edge from edge.source_file endpoint to the opposite endpoint.
        if u_file == src_file_attr:
            tgt_file = v_file
        elif v_file == src_file_attr:
            tgt_file = u_file
        else:
            # Fallback: if source endpoint cannot be matched exactly,
            # still treat edge.source_file as source and pick the opposite endpoint
            # only if one endpoint has a real source_file.
            tgt_file = v_file if v_file and v_file != src_file_attr else u_file

        if not tgt_file:
            continue

        dependency_edges.add((src_file_attr, tgt_file))

    # NetworkX cycle enumeration follows adjacency insertion order. Build the
    # projection canonically before applying the bounded enumeration below.
    file_graph = nx.DiGraph()
    file_graph.add_nodes_from(
        sorted({node for edge in dependency_edges for node in edge}, key=stable_value_key)
    )
    file_graph.add_edges_from(
        sorted(
            dependency_edges,
            key=lambda edge: (stable_value_key(edge[0]), stable_value_key(edge[1])),
        )
    )

    if not file_graph.edges():
        return []

    # Step 2: Find simple cycles, bounded by length.
    # Pass length_bound so networkx prunes during enumeration rather than
    # enumerating all elementary cycles and post-filtering — avoids exponential
    # blowup on dense graphs with many long cycles (#1196).
    cycles: list[list[str]] = []
    for cycle in nx.simple_cycles(file_graph, length_bound=max_cycle_length):
        if len(cycle) <= max_cycle_length:
            cycles.append(cycle)
        if len(cycles) >= top_n * 10:
            # Stop early to avoid combinatorial explosion
            break

    # Step 3: Sort by length (shortest = tightest coupling), then deduplicate.
    cycles.sort(key=lambda cycle: (len(cycle), tuple(stable_value_key(node) for node in cycle)))

    # Deduplicate rotations: normalize each cycle by starting from the
    # lexicographically smallest element.
    seen: set[tuple[str, ...]] = set()
    unique_cycles: list[list[str]] = []
    for cycle in cycles:
        core = list(cycle)
        if not core:
            continue
        min_idx = core.index(min(core))
        normalized = tuple(core[min_idx:] + core[:min_idx])
        if normalized not in seen:
            seen.add(normalized)
            unique_cycles.append(list(normalized))
            if len(unique_cycles) >= top_n:
                break

    result: list[dict] = []
    for cycle in unique_cycles:
        result.append(
            {
                "cycle": cycle,
                "length": len(cycle),
                "why": "circular dependency",
            }
        )

    return result
