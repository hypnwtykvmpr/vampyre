# generate GRAPH_REPORT.md - the human-readable audit trail
from __future__ import annotations
import re
from collections.abc import Hashable, Mapping, Sequence

import networkx as nx
from graphify.projections import distinct_neighbor_degree, stable_value_key


def _safe_community_name(label: str) -> str:
    """Mirrors export.safe_name so community hub filenames and report wikilinks always agree."""
    cleaned = re.sub(
        r'[\\/*?:"<>|#^[\]]', "", label.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    ).strip()
    cleaned = re.sub(r"\.(md|mdx|markdown)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned or "unnamed"


def load_learning_for_report(graph_path) -> dict | None:
    """Assemble the report's work-memory inputs from sibling artifacts.

    Reads the ``.graphify_learning.json`` overlay (preferred sources) next to
    ``graph_path`` and re-aggregates the memory docs for the query-scoped
    dead-ends. Best-effort: returns None if neither is available, so the report
    simply omits the section. Never raises.
    """
    from pathlib import Path as _Path

    try:
        gp = _Path(graph_path)
        from graphify.reflect import load_learning_overlay, load_memory_docs, aggregate_lessons

        overlay = load_learning_overlay(gp)
        dead_ends: list[dict] = []
        mem = gp.parent / "memory"
        if mem.is_dir():
            agg = aggregate_lessons(load_memory_docs(mem))
            dead_ends = agg.get("dead_ends", [])
        if not overlay and not dead_ends:
            return None
        return {"overlay": overlay, "dead_ends": dead_ends}
    except Exception:
        return None


def _learning_section(lines: list, learning: dict | None, top_n: int = 10) -> None:
    """Append the ``## Work-memory lessons`` section, or nothing when empty."""
    if not learning:
        return
    overlay = learning.get("overlay") or {}
    dead_ends = learning.get("dead_ends") or []
    preferred = [
        (nid, e)
        for nid, e in overlay.items()
        if isinstance(e, dict) and e.get("status") == "preferred"
    ]
    # Most-corroborated first (uses desc), then by score, then id for stability.
    preferred.sort(
        key=lambda kv: (-kv[1].get("uses", 0), -float(kv[1].get("score", 0) or 0), kv[0])
    )
    if not preferred and not dead_ends:
        return
    lines += ["", "## Work-memory lessons"]
    if preferred:
        lines += ["", "**Preferred sources** — corroborated by past sessions; start here."]
        for nid, e in preferred[:top_n]:
            label = e.get("label") or nid
            stale = " _(code changed — re-verify)_" if e.get("stale") else ""
            lines.append(
                f"- `{label}` ({e.get('uses', 0)}× useful, score={e.get('score', 0)}){stale}"
            )
    if dead_ends:
        lines += ["", "**Known dead ends** — questions that led nowhere; don't re-derive."]
        for d in dead_ends:
            nodes = ", ".join(f"`{n}`" for n in d.get("nodes", []))
            lines.append(f'- "{d.get("question", "")}"' + (f" -> {nodes}" if nodes else ""))


def generate(
    G: nx.Graph,
    communities: Mapping[int, Sequence[Hashable]],
    cohesion_scores: dict[int, float],
    community_labels: dict[int, str],
    god_node_list: list[dict],
    surprise_list: list[dict],
    detection_result: dict,
    token_cost: dict,
    root: str,
    suggested_questions: list[dict] | None = None,
    min_community_size: int = 3,
    built_at_commit: str | None = None,
    learning: dict | None = None,
) -> str:
    # JSON deserialization produces string keys; normalize to int so .get(cid) works.
    if community_labels:
        community_labels = {
            int(k) if isinstance(k, str) else k: v for k, v in community_labels.items()
        }

    confidences = [d.get("confidence", "EXTRACTED") for _, _, d in G.edges(data=True)]
    total = len(confidences) or 1
    ext_pct = round(confidences.count("EXTRACTED") / total * 100)
    inf_pct = round(confidences.count("INFERRED") / total * 100)
    amb_pct = round(confidences.count("AMBIGUOUS") / total * 100)

    inf_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("confidence") == "INFERRED"]
    inf_scores = [d.get("confidence_score", 0.5) for _, _, d in inf_edges]
    inf_avg = round(sum(inf_scores) / len(inf_scores), 2) if inf_scores else None

    file_groups = detection_result.get("files")
    if isinstance(file_groups, dict) and all(
        isinstance(paths, list) for paths in file_groups.values()
    ):
        corpus_file_count = sum(len(paths) for paths in file_groups.values())
    else:
        corpus_file_count = detection_result.get("total_files", 0)

    lines = [
        f"# Graph Report - {root}",
        "",
        "## Corpus Check",
    ]
    if detection_result.get("warning"):
        lines.append(f"- {detection_result['warning']}")
    else:
        lines += [
            f"- {corpus_file_count} files · ~{detection_result.get('total_words', 0):,} words",
            "- Verdict: corpus is large enough that graph structure adds value.",
        ]

    from .analyze import _is_file_node as _ifn

    real_communities = {
        cid: sorted(
            [node for node in nodes if node in G and not _ifn(G, node)],
            key=stable_value_key,
        )
        for cid, nodes in communities.items()
    }
    rendered_communities = {
        cid: nodes for cid, nodes in real_communities.items() if len(nodes) >= min_community_size
    }
    shown_count = len(rendered_communities)
    omitted_count = len(communities) - shown_count

    lines += [
        "",
        "## Summary",
        f"- {G.number_of_nodes()} nodes · {G.number_of_edges()} edges · {len(communities)} communities"
        + (f" ({shown_count} shown, {omitted_count} omitted)" if omitted_count else ""),
        f"- Extraction: {ext_pct}% EXTRACTED · {inf_pct}% INFERRED · {amb_pct}% AMBIGUOUS"
        + (
            f" · INFERRED: {len(inf_edges)} edges (avg confidence: {inf_avg})"
            if inf_avg is not None
            else ""
        ),
        f"- Token cost: {token_cost.get('input', 0):,} input · {token_cost.get('output', 0):,} output",
    ]

    if built_at_commit:
        lines += [
            "",
            "## Graph Freshness",
            f"- Built from commit: `{built_at_commit[:8]}`",
            "- Run `git rev-parse HEAD` and compare to check if the graph is stale.",
            "- Run `graphify update .` after code changes (no API cost).",
        ]

    # Community hub navigation - links to _COMMUNITY_*.md files in the Obsidian vault.
    # Without these, GRAPH_REPORT.md is a dead-end and the vault splits into disconnected components.
    if rendered_communities:
        lines += ["", "## Community Hubs (Navigation)"]
        for cid in sorted(rendered_communities, key=stable_value_key):
            label = community_labels.get(cid, f"Community {cid}")
            safe = _safe_community_name(label)
            lines.append(f"- [[_COMMUNITY_{safe}|{label}]]")

    lines += [
        "",
        "## God Nodes (most connected - your core abstractions)",
    ]
    for i, node in enumerate(god_node_list, 1):
        lines.append(f"{i}. `{node['label']}` - {node['degree']} distinct neighbors")

    lines += ["", "## Surprising Connections (you probably didn't know these)"]
    if surprise_list:
        for s in surprise_list:
            relation = s.get("relation", "related_to")
            note = s.get("note", "")
            files = s.get("source_files", ["", ""])
            conf = s.get("confidence", "EXTRACTED")
            cscore = s.get("confidence_score")
            if conf == "INFERRED" and cscore is not None:
                conf_tag = f"INFERRED {cscore:.2f}"
            else:
                conf_tag = conf
            sem_tag = " [semantically similar]" if relation == "semantically_similar_to" else ""
            lines += [
                f"- `{s['source']}` --{relation}--> `{s['target']}`  [{conf_tag}]{sem_tag}",
                f"  {files[0]} → {files[1]}" + (f"  _{note}_" if note else ""),
            ]
    else:
        lines.append("- None detected - all connections are within the same source files.")

    # Circular imports surfaced from file-level dependency graph.
    from .analyze import find_import_cycles

    cycles = find_import_cycles(G)
    lines += ["", "## Import Cycles"]
    if cycles:
        for c in cycles:
            cycle = c.get("cycle", [])
            length = c.get("length", len(cycle))
            if not cycle:
                continue
            cycle_path = " -> ".join(str(node) for node in cycle + [cycle[0]])
            lines.append(f"- {length}-file cycle: `{cycle_path}`")
    else:
        lines.append("- None detected.")

    hyperedges = G.graph.get("hyperedges", [])
    if hyperedges:
        lines += ["", "## Hyperedges (group relationships)"]
        for h in hyperedges:
            node_labels = ", ".join(str(node) for node in h.get("nodes", []))
            conf = h.get("confidence", "INFERRED")
            cscore = h.get("confidence_score")
            conf_tag = f"{conf} {cscore:.2f}" if cscore is not None else conf
            lines.append(f"- **{h.get('label', h.get('id', ''))}** — {node_labels} [{conf_tag}]")

    lines += ["", f"## Communities ({len(communities)} total, {omitted_count} omitted)"]
    for cid, real_nodes in sorted(
        rendered_communities.items(), key=lambda item: stable_value_key(item[0])
    ):
        label = community_labels.get(cid, f"Community {cid}")
        score = cohesion_scores.get(cid, 0.0)
        display = [str(G.nodes[n].get("label", n)) for n in real_nodes[:8]]
        suffix = f" (+{len(real_nodes) - 8} more)" if len(real_nodes) > 8 else ""
        lines += [
            "",
            f'### Community {cid} - "{label}"',
            f"Cohesion: {score:.2f}",
            f"Nodes ({len(real_nodes)}): {', '.join(display)}{suffix}",
        ]

    ambiguous = sorted(
        [(u, v, d) for u, v, d in G.edges(data=True) if d.get("confidence") == "AMBIGUOUS"],
        key=lambda edge: (
            stable_value_key(edge[0]),
            stable_value_key(edge[1]),
            stable_value_key(edge[2].get("relation")),
            stable_value_key(edge[2].get("source_location")),
        ),
    )
    if ambiguous:
        lines += ["", "## Ambiguous Edges - Review These"]
        for u, v, d in ambiguous:
            ul = G.nodes[u].get("label", u)
            vl = G.nodes[v].get("label", v)
            lines += [
                f"- `{ul}` → `{vl}`  [AMBIGUOUS]",
                f"  {d.get('source_file', '')} · relation: {d.get('relation', 'unknown')}",
            ]

    # --- Gaps section ---
    from .analyze import _is_concept_node, _is_file_node

    isolated = [
        n
        for n in G.nodes()
        if distinct_neighbor_degree(G, n) <= 1
        and not _is_file_node(G, n)
        and not _is_concept_node(G, n)
        and G.nodes[n].get("file_type") != "rationale"
    ]
    thin_communities = {
        cid: nodes for cid, nodes in real_communities.items() if 0 < len(nodes) < min_community_size
    }
    gap_count = len(isolated) + len(thin_communities)
    isolated.sort(key=stable_value_key)

    if gap_count > 0 or amb_pct > 20:
        lines += ["", "## Knowledge Gaps"]
        if isolated:
            isolated_labels = [G.nodes[n].get("label", n) for n in isolated[:5]]
            suffix = f" (+{len(isolated) - 5} more)" if len(isolated) > 5 else ""
            lines.append(
                f"- **{len(isolated)} isolated node(s):** {', '.join(f'`{isolated_label}`' for isolated_label in isolated_labels)}{suffix}"
            )
            lines.append(
                "  These have ≤1 connection - possible missing edges or undocumented components."
            )
        if thin_communities:
            lines.append(
                f"- **{len(thin_communities)} thin communities (<{min_community_size} nodes) omitted from report** — run `graphify query` to explore isolated nodes."
            )
        if amb_pct > 20:
            lines.append(
                f"- **High ambiguity: {amb_pct}% of edges are AMBIGUOUS.** Review the Ambiguous Edges section above."
            )

    # --- Work-memory lessons (derived overlay) ---
    # Preferred sources come from the .graphify_learning.json sidecar; the
    # query-scoped dead-ends come from the reflect aggregate. Section omitted
    # entirely when neither is present, so a graph with no work-memory is
    # byte-identical to the pre-feature report.
    _learning_section(lines, learning)

    if suggested_questions:
        lines += ["", "## Suggested Questions"]
        no_signal = (
            len(suggested_questions) == 1 and suggested_questions[0].get("type") == "no_signal"
        )
        if no_signal:
            lines.append(f"_{suggested_questions[0]['why']}_")
        else:
            lines.append("_Questions this graph is uniquely positioned to answer:_")
            lines.append("")
            for q in suggested_questions:
                if q.get("question"):
                    lines.append(f"- **{q['question']}**")
                    lines.append(f"  _{q['why']}_")

    return "\n".join(lines)
