import json
import re
from pathlib import Path
import networkx as nx
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections
from graphify.report import generate

FIXTURES = Path(__file__).parent / "fixtures"


def make_inputs():
    extraction = json.loads((FIXTURES / "extraction.json").read_text(encoding="utf-8"))
    G = build_from_json(extraction)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G)
    detection = {"total_files": 4, "total_words": 62400, "needs_graph": True, "warning": None}
    tokens = {"input": extraction["input_tokens"], "output": extraction["output_tokens"]}
    return G, communities, cohesion, labels, gods, surprises, detection, tokens


def test_report_contains_header():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "# Graph Report" in report
    assert re.search(r"^# Graph Report - \./project$", report, re.MULTILINE)


def test_report_uses_file_list_as_corpus_count_authority():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    detection.update(
        total_files=1,
        files={"code": ["a.py", "b.py"], "document": ["README.md"]},
    )

    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )

    assert "3 files · ~62,400 words" in report


def test_report_shown_community_count_matches_rendered_headings():
    G = nx.Graph()
    for node_id in ("a", "b", "c"):
        G.add_node(node_id, label=node_id, source_file=f"{node_id}.py", source_location="L2")
    for node_id in ("thin-a", "thin-b"):
        G.add_node(node_id, label=node_id, source_file="thin.py", source_location="L2")
    G.add_node("file", label="only.py", source_file="only.py", source_location="L1")
    G.add_edges_from([("a", "b"), ("b", "c"), ("thin-a", "thin-b")])
    communities = {0: ["a", "b", "c"], 1: ["thin-a", "thin-b"], 2: ["file"]}

    report = generate(
        G,
        communities,
        {0: 0.5, 1: 0.5, 2: 0.0},
        {0: "Shown", 1: "Thin", 2: "File only"},
        [],
        [],
        {"total_files": 4, "total_words": 100, "warning": None},
        {"input": 0, "output": 0},
        "./project",
        min_community_size=3,
    )

    assert "3 communities (1 shown, 2 omitted)" in report
    assert report.count("### Community ") == 1


def test_report_gap_counts_use_renderable_members_and_configured_threshold():
    graph = nx.Graph()
    for node_id in ("a", "b", "c"):
        graph.add_node(
            node_id,
            label=node_id,
            source_file="module.py",
            source_location="L2",
        )
    graph.add_edges_from([("a", "b"), ("b", "c")])

    report = generate(
        graph,
        {0: ["a", "b", "c", "stale-node"]},
        {0: 0.5},
        {0: "Below custom threshold"},
        [],
        [],
        {"total_files": 1, "total_words": 10, "warning": None},
        {"input": 0, "output": 0},
        "./project",
        min_community_size=4,
    )

    assert "1 communities (0 shown, 1 omitted)" in report
    assert "1 thin communities (<4 nodes) omitted from report" in report


def test_report_contains_corpus_check():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## Corpus Check" in report


def test_report_contains_god_nodes():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## God Nodes" in report


def test_report_contains_surprising_connections():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## Surprising Connections" in report


def test_report_contains_communities():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## Communities" in report


def test_report_renders_mixed_json_node_ids_without_type_errors():
    graph = nx.MultiDiGraph()
    graph.add_node(1, label="one-int", source_file="one.py", source_location="L2")
    graph.add_node("1", label="one", source_file="two.py", source_location="L2")
    graph.add_edge(1, "1", key="k", relation="calls", confidence="EXTRACTED")
    graph.graph["hyperedges"] = [{"id": "mixed", "nodes": [1, "1"]}]

    report = generate(
        graph,
        {0: [1, "1"]},
        {0: 1.0},
        {0: "Mixed IDs"},
        [],
        [],
        {"total_files": 2, "total_words": 10, "warning": None},
        {"input": 0, "output": 0},
        ".",
        min_community_size=1,
    )

    assert "**mixed** — 1, 1" in report
    assert "Nodes (2): one-int, one" in report


def test_report_contains_ambiguous_section():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## Ambiguous Edges" in report


def test_report_shows_token_cost():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "Token cost" in report
    assert "1,200" in report


def test_report_shows_raw_cohesion_scores():
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    report = generate(
        G,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        "./project",
        min_community_size=1,
    )
    assert "Cohesion:" in report
    assert "✓" not in report
    assert "⚠" not in report


# --- work-memory lessons section ----------------------------------------------


def test_report_work_memory_section_present_with_overlay_and_dead_ends():
    """When a work-memory overlay (preferred sources) and query-scoped dead-ends
    are supplied, the report grows a `## Work-memory lessons` section listing the
    preferred sources and, separately, the dead-ends as question -> nodes."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    learning = {
        "overlay": {
            "auth_login": {
                "status": "preferred",
                "uses": 3,
                "score": 2.4,
                "label": "login()",
                "stale": False,
            },
            "redis": {
                "status": "tentative",
                "uses": 1,
                "score": 0.5,
                "label": "RedisClient",
                "stale": False,
            },
        },
        "dead_ends": [
            {"question": "does it use websockets?", "nodes": ["WSServer"], "date": "2026-05-01"},
        ],
    }
    report = generate(
        G,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        "./project",
        learning=learning,
    )
    assert "## Work-memory lessons" in report
    assert "**Preferred sources**" in report
    assert "`login()`" in report
    # Tentative is not listed in the report's preferred block.
    assert "RedisClient" not in report
    # Dead-ends are query-scoped: question -> nodes, NOT a node-level status.
    assert "**Known dead ends**" in report
    assert "does it use websockets?" in report
    assert "`WSServer`" in report


def test_report_work_memory_section_absent_without_overlay():
    """No learning input => no section; report identical to pre-feature."""
    G, communities, cohesion, labels, gods, surprises, detection, tokens = make_inputs()
    before = generate(
        G, communities, cohesion, labels, gods, surprises, detection, tokens, "./project"
    )
    assert "## Work-memory lessons" not in before
    # Explicit empty learning also omits the section.
    empty = generate(
        G,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        "./project",
        learning={"overlay": {}, "dead_ends": []},
    )
    assert "## Work-memory lessons" not in empty
    assert before == empty
