# Wiki export - Wikipedia-style markdown articles from the knowledge graph
# Generates an agent-crawlable wiki: index.md + one article per community + god node articles
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from urllib.parse import quote
import networkx as nx

from graphify.paths import portable_filename_stem
from graphify.persistence import FileStateTransaction, atomic_write_text, output_state_lock
from graphify.projections import (
    distinct_neighbor_degree,
    format_relationship_envelope,
    stable_value_key,
)

_WIKI_MANIFEST = ".graphify_wiki_manifest.json"


def _safe_filename(name: str) -> str:
    """Make a label safe for use as a filename across platforms.

    Substitutes characters that Windows reserves in filenames
    (< > : " / \\ | ? *) and strips trailing dots/spaces, also reserved.
    Falls back to 'unnamed' for empty results and caps length at 200
    chars to stay well under common filesystem limits.
    """
    import re

    s = name.replace("/", "-").replace(" ", "_").replace(":", "-")
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = s.strip(". ")
    return portable_filename_stem(s if s else "unnamed")[:200]


def _md_link(label: str, resolver: dict[str, str]) -> str:
    """Render a link to another wiki article as a portable relative markdown link.

    ``resolver`` maps an article's display label to the slug (filename stem) it
    was written under. When the label has an article, emit a standard
    ``[label](slug.md)`` link, URL-encoding the target so any spaces, parens, &
    or # in the slug survive every CommonMark renderer (GitHub, GitLab, VS Code
    preview, a plain browser) and Obsidian alike. The old ``[[label]]`` form
    only resolved inside Obsidian, because the on-disk filename differs from the
    label — _safe_filename turns spaces into underscores and substitutes
    reserved characters — so e.g. ``[[Domain Data Models]]`` pointed at a
    non-existent ``Domain Data Models.md`` everywhere else.

    Labels with no article — most node-level links, since only communities and
    god nodes get article files — render as plain text instead of a dead link
    that points nowhere even inside Obsidian.
    """
    text = label.replace("[", r"\[").replace("]", r"\]")
    slug = resolver.get(label)
    if slug is None:
        return text
    return f"[{text}]({quote(f'{slug}.md')})"


def _md_link_to_slug(label: str, slug: str) -> str:
    """Render a link when the caller already owns the exact article slug."""
    text = label.replace("[", r"\[").replace("]", r"\]")
    return f"[{text}]({quote(f'{slug}.md')})"


def _all_neighbors(G: nx.Graph, node_id: str) -> set[str]:
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        return set(G.successors(node_id)) | set(G.predecessors(node_id))
    return set(G.neighbors(node_id))


def _cross_community_links(
    G: nx.Graph,
    nodes: list[str],
    own_cid: int,
    labels: dict[int, str],
    node_community: dict[str, int],
) -> list[tuple[str, int]]:
    """Return (community_label, edge_count) pairs for cross-community connections, sorted descending."""
    counts: dict[str, int] = Counter()
    for nid in nodes:
        for neighbor in _all_neighbors(G, nid):
            ncid = node_community.get(neighbor)
            if ncid is not None and ncid != own_cid:
                counts[labels.get(ncid, f"Community {ncid}")] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _community_article(
    G: nx.Graph,
    cid: int,
    nodes: list[str],
    label: str,
    labels: dict[int, str],
    cohesion: float | None,
    node_community: dict[str, int] | None = None,
    resolver: dict[str, str] | None = None,
) -> str:
    resolver = resolver or {}
    top_nodes = sorted(
        nodes,
        key=lambda node: (-distinct_neighbor_degree(G, node), stable_value_key(node)),
    )[:25]
    cross = _cross_community_links(G, nodes, cid, labels, node_community or {})

    # Edge confidence breakdown
    conf_counts: Counter = Counter()
    node_set = set(nodes)
    for source, target, data in G.edges(data=True):
        if source in node_set or target in node_set:
            conf_counts[data.get("confidence", "EXTRACTED")] += 1
    total_edges = sum(conf_counts.values()) or 1

    sources = sorted({G.nodes[n].get("source_file") or "" for n in nodes} - {""})

    lines: list[str] = []
    lines += [f"# {label}", ""]

    meta_parts = [f"{len(nodes)} nodes"]
    if cohesion is not None:
        meta_parts.append(f"cohesion {cohesion:.2f}")
    lines += [f"> {' · '.join(meta_parts)}", ""]

    lines += ["## Key Concepts", ""]
    for nid in top_nodes:
        d = G.nodes[nid]
        node_label = d.get("label", nid)
        src = d.get("source_file", "")
        degree = distinct_neighbor_degree(G, nid)
        edge_records = G.degree(nid)
        src_str = f" — `{src}`" if src else ""
        lines.append(
            f"- **{node_label}** ({degree} distinct neighbors; {edge_records} edge records)"
            f"{src_str}"
        )
    remaining = len(nodes) - len(top_nodes)
    if remaining > 0:
        lines.append(f"- *... and {remaining} more nodes in this community*")
    lines.append("")

    lines += ["## Relationships", ""]
    if cross:
        for other_label, count in cross[:12]:
            lines.append(f"- {_md_link(other_label, resolver)} ({count} shared connections)")
    else:
        lines.append("- No strong cross-community connections detected")
    lines.append("")

    if sources:
        lines += ["## Source Files", ""]
        for src in sources[:20]:
            lines.append(f"- `{src}`")
        lines.append("")

    lines += ["## Audit Trail", ""]
    for conf in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
        n = conf_counts.get(conf, 0)
        pct = round(n / total_edges * 100)
        lines.append(f"- {conf}: {n} ({pct}%)")
    lines.append("")

    lines += [
        "---",
        "",
        f"*Part of the graphify knowledge wiki. See {_md_link('index', resolver)} to navigate.*",
    ]
    return "\n".join(lines)


def _god_node_article(
    G: nx.Graph,
    nid: str,
    labels: dict[int, str],
    node_community: dict[str, int] | None = None,
    resolver: dict[str, str] | None = None,
) -> str:
    resolver = resolver or {}
    d = G.nodes[nid]
    node_label = d.get("label", nid)
    src = d.get("source_file", "")
    cid = (node_community or {}).get(nid)
    community_name = labels.get(cid, f"Community {cid}") if cid is not None else None

    lines: list[str] = []
    lines += [f"# {node_label}", ""]
    lines += [
        f"> God node · {distinct_neighbor_degree(G, nid)} distinct neighbors · "
        f"{G.degree(nid)} edge records · `{src}`",
        "",
    ]

    if community_name:
        lines += [f"**Community:** {_md_link(community_name, resolver)}", ""]

    def add_connections(heading: str, pairs: list[tuple[str, str]]) -> None:
        lines.extend([f"## {heading}", ""])
        if not pairs:
            lines.extend(["- None", ""])
            return
        for source, target in pairs[:20]:
            neighbor = target if source == nid else source
            neighbor_label = G.nodes[neighbor].get("label", neighbor)
            summary = format_relationship_envelope(
                G,
                source,
                target,
                directed_only=True,
            )
            lines.append(f"- {_md_link(str(neighbor_label), resolver)} — {summary}")
        lines.append("")

    def neighbor_key(node: str) -> tuple:
        return (-distinct_neighbor_degree(G, node), stable_value_key(node))

    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        outgoing = [
            (nid, neighbor) for neighbor in sorted(set(G.successors(nid)), key=neighbor_key)
        ]
        incoming = [
            (neighbor, nid) for neighbor in sorted(set(G.predecessors(nid)), key=neighbor_key)
        ]
        add_connections("Outgoing Connections", outgoing)
        add_connections("Incoming Connections", incoming)
    else:
        connections = [
            (nid, neighbor) for neighbor in sorted(set(G.neighbors(nid)), key=neighbor_key)
        ]
        add_connections("Connections", connections)

    lines += [
        "---",
        "",
        f"*Part of the graphify knowledge wiki. See {_md_link('index', resolver)} to navigate.*",
    ]
    return "\n".join(lines)


def _index_md(
    communities: dict[int, list[str]],
    labels: dict[int, str],
    god_nodes_data: list[dict],
    total_nodes: int,
    total_edges: int,
    resolver: dict[str, str] | None = None,
    community_slugs: dict[int, str] | None = None,
    god_slugs: dict[object, str] | None = None,
) -> str:
    resolver = resolver or {}
    community_slugs = community_slugs or {}
    god_slugs = god_slugs or {}
    lines: list[str] = [
        "# Knowledge Graph Index",
        "",
        "> Auto-generated by graphify. Start here — read community articles for context, then drill into god nodes for detail.",
        "",
        f"**{total_nodes} nodes · {total_edges} edge records · {len(communities)} communities**",
        "",
        "---",
        "",
        "## Communities",
        "(sorted by size, largest first)",
        "",
    ]

    for cid, nodes in sorted(communities.items(), key=lambda x: -len(x[1])):
        label = labels.get(cid, f"Community {cid}")
        rendered = (
            _md_link_to_slug(label, community_slugs[cid])
            if cid in community_slugs
            else _md_link(label, resolver)
        )
        lines.append(f"- {rendered} — {len(nodes)} nodes")
    lines.append("")

    if god_nodes_data:
        lines += ["## God Nodes", "(most connected concepts — the load-bearing abstractions)", ""]
        for node in god_nodes_data:
            slug = god_slugs.get(node.get("id"))
            rendered = (
                _md_link_to_slug(str(node["label"]), slug)
                if slug is not None
                else _md_link(str(node["label"]), resolver)
            )
            lines.append(f"- {rendered} — {node['degree']} distinct neighbors")
        lines.append("")

    lines += [
        "---",
        "",
        "*Generated by [Vampyre](https://github.com/hypnwtykvmpr/vampyre)*",
    ]
    return "\n".join(lines)


def to_wiki(
    G: nx.Graph,
    communities: dict[int, list[str]],
    output_dir: str | Path,
    community_labels: dict[int, str] | None = None,
    cohesion: dict[int, float] | None = None,
    god_nodes_data: list[dict] | None = None,
) -> int:
    """Generate a Wikipedia-style wiki from the graph.

    Writes:
      - index.md            — agent entry point, catalog of all articles
      - <CommunityName>.md  — one article per community
      - <GodNodeLabel>.md   — one article per god node

    Returns the number of articles written (excluding index.md).
    """
    out = Path(output_dir)

    if not communities:
        raise ValueError(
            "communities dict is empty — refusing to clear wiki/. "
            "Run `graphify extract .` or `graphify cluster-only .` first."
        )

    # Filter stale node IDs that exist in communities but not in G.
    # Analysis JSON can drift from the graph after dedup / re-extract / update.
    # NetworkX 3.x returns DegreeView({}) for missing nodes instead of raising,
    # which crashes sorted() with TypeError; G.neighbors()/G.nodes[] also raise.
    import sys as _sys

    _g_nodes = set(G.nodes)
    _orig_total = sum(len(ns) for ns in communities.values())
    communities = {cid: [n for n in nodes if n in _g_nodes] for cid, nodes in communities.items()}
    communities = {cid: nodes for cid, nodes in communities.items() if nodes}
    _kept_total = sum(len(ns) for ns in communities.values())
    if _kept_total < _orig_total:
        print(
            f"wiki: dropped {_orig_total - _kept_total} stale node ID(s) not in graph "
            f"({len(communities)} communities remaining)",
            file=_sys.stderr,
        )

    if not communities:
        raise ValueError(
            "all community node IDs are stale — none exist in the graph. "
            "Re-run `graphify extract .` to regenerate .graphify_analysis.json."
        )

    labels = community_labels or {cid: f"Community {cid}" for cid in communities}
    cohesion = cohesion or {}
    god_nodes_data = god_nodes_data or []

    # Build node->community lookup once; node attrs never carry community (it lives in
    # the communities dict), so _cross_community_links and _god_node_article need this.
    node_community: dict[str, int] = {n: cid for cid, nodes in communities.items() for n in nodes}

    count = 0
    used_slugs: set[str] = set()

    def _unique_slug(base: str) -> str:
        # Fold case in the collision check: two labels differing only by case
        # (e.g. "Parser" vs "parser") resolve to one path on case-insensitive
        # filesystems (macOS/APFS, Windows/NTFS), so they must dedup against each
        # other while still emitting the original-case filename.
        slug = base
        n = 2
        while slug.lower() in used_slugs:
            slug = f"{base}_{n}"
            n += 1
        used_slugs.add(slug.lower())
        return slug

    # First pass: assign every article its slug before rendering any body, so the
    # bodies can link to one another. A link's target is the on-disk filename (the
    # slug), which differs from the label — _safe_filename turns spaces into
    # underscores and substitutes reserved chars, and a slug may pick up a numeric
    # suffix from collision dedup — so the final slug must be known up front.
    # resolver maps display label -> slug; labels with no article are absent, so
    # _md_link renders them as plain text. Communities are slugged before god nodes
    # (and setdefault keeps the first), preserving the filename-assignment order
    # the case-collision dedup relies on.
    resolver: dict[str, str] = {"index": "index"}

    community_slugs: dict[int, str] = {}
    for cid in sorted(communities, key=stable_value_key):
        label = labels.get(cid, f"Community {cid}")
        slug = _unique_slug(_safe_filename(label))
        community_slugs[cid] = slug
        resolver.setdefault(label, slug)

    god_articles: list[tuple[str, str]] = []  # (node_id, slug)
    god_slugs: dict[object, str] = {}
    for node_data in sorted(
        god_nodes_data,
        key=lambda item: (stable_value_key(item.get("id")), stable_value_key(item.get("label"))),
    ):
        nid = node_data.get("id")
        if nid and nid in G:
            slug = _unique_slug(_safe_filename(node_data["label"]))
            god_articles.append((nid, slug))
            god_slugs[nid] = slug
            resolver.setdefault(node_data["label"], slug)

    articles: dict[str, str] = {}

    # Second pass: render every article with the full resolver in hand before
    # mutating the destination.
    for cid, nodes in sorted(communities.items(), key=lambda item: stable_value_key(item[0])):
        label = labels.get(cid, f"Community {cid}")
        article = _community_article(
            G, cid, nodes, label, labels, cohesion.get(cid), node_community, resolver
        )
        articles[f"{community_slugs[cid]}.md"] = article
        count += 1

    for nid, slug in god_articles:
        article = _god_node_article(G, nid, labels, node_community, resolver)
        articles[f"{slug}.md"] = article
        count += 1

    articles["index.md"] = _index_md(
        communities,
        labels,
        god_nodes_data,
        G.number_of_nodes(),
        G.number_of_edges(),
        resolver,
        community_slugs,
        god_slugs,
    )

    manifest_path = out / _WIKI_MANIFEST

    def load_owned() -> set[str]:
        if not manifest_path.exists():
            return set()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("wiki ownership manifest is unreadable; refusing publication") from exc
        files = payload.get("files") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(files, list)
        ):
            raise ValueError("wiki ownership manifest is invalid; refusing publication")
        owned: set[str] = set()
        for name in files:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".md"):
                raise ValueError("wiki ownership manifest contains an unsafe path")
            owned.add(name)
        return owned

    def unowned_collisions(owned: set[str]) -> list[str]:
        return sorted(name for name in articles if (out / name).exists() and name not in owned)

    preflight_collisions = unowned_collisions(load_owned())
    if preflight_collisions:
        shown = ", ".join(preflight_collisions[:5])
        raise ValueError(
            f"refusing to overwrite {len(preflight_collisions)} unowned wiki file(s): {shown}"
        )

    with output_state_lock(out) as lease:
        if lease is None:
            raise RuntimeError("could not acquire wiki publication lock")
        owned = load_owned()
        collisions = unowned_collisions(owned)
        if collisions:
            shown = ", ".join(collisions[:5])
            raise ValueError(
                f"refusing to overwrite {len(collisions)} unowned wiki file(s): {shown}"
            )
        stale = owned - set(articles)
        transaction = FileStateTransaction(
            [manifest_path, *(out / name for name in sorted(set(articles) | stale))],
            lease=lease,
        )
        try:
            for name, article in sorted(articles.items()):
                atomic_write_text(out / name, article)
            for name in sorted(stale):
                (out / name).unlink(missing_ok=True)
            atomic_write_text(
                manifest_path,
                json.dumps({"version": 1, "files": sorted(articles)}, indent=2) + "\n",
            )
        except Exception:
            transaction.rollback()
            raise
        transaction.commit()

    return count
