"""Tests for serve.py - MCP graph query helpers (no mcp package required)."""

import json
import os
import subprocess
import sys
import pytest
import networkx as nx
from networkx.readwrite import json_graph

from graphify.serve import (
    InsufficientBudgetError,
    _estimate_tokens_for_budget,
    _IDF_CACHE,
    _TRIGRAM_INDEX_CACHE,
    _communities_from_graph,
    _score_nodes,
    _compute_idf,
    _pick_seeds,
    _bfs,
    _dfs,
    _find_node,
    _trigrams,
    _node_search_text,
    _get_trigram_index,
    _trigram_candidates,
    _filter_graph_by_context,
    _infer_context_filters,
    _query_terms,
    _query_graph_text,
    _resolve_context_filters,
    _subgraph_to_text,
    _load_graph,
    _community_header,
)


def _make_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("n1", label="extract", source_file="extract.py", source_location="L10", community=0)
    G.add_node("n2", label="cluster", source_file="cluster.py", source_location="L5", community=0)
    G.add_node("n3", label="build", source_file="build.py", source_location="L1", community=1)
    G.add_node("n4", label="report", source_file="report.py", source_location="L1", community=1)
    G.add_node("n5", label="isolated", source_file="other.py", source_location="L1", community=2)
    G.add_edge("n1", "n2", relation="calls", confidence="INFERRED", context="call")
    G.add_edge("n2", "n3", relation="imports", confidence="EXTRACTED", context="import")
    G.add_edge("n3", "n4", relation="uses", confidence="EXTRACTED")
    return G


# --- _communities_from_graph ---


def test_communities_from_graph_basic():
    G = _make_graph()
    communities = _communities_from_graph(G)
    assert 0 in communities
    assert 1 in communities
    assert "n1" in communities[0]
    assert "n2" in communities[0]
    assert "n3" in communities[1]


def test_communities_from_graph_no_community_attr():
    G = nx.Graph()
    G.add_node("a", label="foo")  # no community attr
    communities = _communities_from_graph(G)
    assert communities == {}


def test_communities_from_graph_isolated():
    G = _make_graph()
    communities = _communities_from_graph(G)
    assert 2 in communities
    assert "n5" in communities[2]


# --- _score_nodes ---


def test_score_nodes_exact_label_match():
    G = _make_graph()
    scored = _score_nodes(G, ["extract"])
    nids = [nid for _, nid in scored]
    assert "n1" in nids
    assert scored[0][1] == "n1"  # highest score first


def test_score_nodes_no_match():
    G = _make_graph()
    scored = _score_nodes(G, ["xyzzy"])
    assert scored == []


def test_score_nodes_source_file_partial():
    G = _make_graph()
    # "cluster.py" contains "cluster" - should score 0.5 for source match
    scored = _score_nodes(G, ["cluster"])
    nids = [nid for _, nid in scored]
    assert "n2" in nids


def test_score_nodes_ignores_trailing_punctuation():
    G = _make_graph()
    scored = _score_nodes(G, ["extract?"])
    assert scored[0][1] == "n1"


def test_score_nodes_multiword_exact_label_outranks_superset():
    """A multi-word query equal to a whole label must resolve uniquely.

    Regression for the `graphify path` "No path found" bug: every node sharing
    the query's token set scored identically (no single token equals a
    multi-word label, so the per-token exact tier never fired), the tie broke by
    arbitrary node-id sort, and a wrong/disconnected endpoint was chosen. The
    full-query tier in _score_nodes must make the exact label win strictly.
    """
    G = nx.Graph()

    # Reproduce the real graph: norm_label keeps punctuation (strip_diacritics +
    # lower, NOT tokenized), so the ':' survives. A tokenized query can never
    # equal that, which is exactly why the first-cut fix was a no-op for
    # punctuated labels. The exact node must still win via the label's tokenized
    # form.
    def _add(nid, label, src):
        G.add_node(nid, label=label, norm_label=label.lower(), source_file=src, community=0)

    _add("exact", "UOCE: Dehumidifier Driver", "uoce_dehumidifier.yaml")
    _add("super", "UOCE: Dehumidifier Driver State Machine", "uoce_dehumidifier.yaml")
    _add("decoy", "Dehumidifier Driver Helper", "uoce_dehumidifier.yaml")

    # CLI resolves endpoints as [t.lower() for t in label.split()].
    scored = _score_nodes(G, [t.lower() for t in "UOCE: Dehumidifier Driver".split()])

    # Resolves uniquely to the exact label, strictly ahead of the superset.
    assert scored[0][1] == "exact"
    assert scored[0][0] > scored[1][0], (
        "exact label must strictly outrank superset/token-bag matches"
    )


def test_find_node_ignores_trailing_punctuation():
    G = _make_graph()
    assert _find_node(G, "extract?") == ["n1"]


def test_find_node_matches_full_punctuated_unicode_label():
    G = nx.Graph()
    G.add_node("n1", label="Skill /auditar — Auditoría inquisitiva de enlaces")

    assert _find_node(G, "Skill /auditar — Auditoría inquisitiva de enlaces") == ["n1"]


def test_find_node_uses_defined_diacritic_normalizer_without_stored_norm_label():
    graph = nx.Graph()
    graph.add_node("n1", label="Auditoría inquisitiva")

    assert _find_node(graph, "Auditoria inquisitiva") == ["n1"]


# --- trigram candidate prefilter (the trigram index that shrinks the O(N) scan) ---


def _force_full_scan(monkeypatch):
    """Disable the prefilter so a call exercises the original full-node scan."""
    monkeypatch.setattr("graphify.serve._trigram_candidates", lambda *a, **k: None)


def _make_big_graph(n: int = 150) -> nx.Graph:
    """A graph large enough that the selectivity guard lets the fast-path fire for
    rare terms and fall back for common ones. Most labels share the 'item'/'node'
    stem (common), plus a few distinctive rare labels and one punctuated label."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"id{i}", label=f"item node {i}", source_file=f"pkg/item_{i}.py")
    G.add_node("rareA", label="ZebraQuokkaWidget", source_file="zoo/zqw.py")
    G.add_node("rareB", label="MarmosetGadget handler", source_file="zoo/marmoset.py")
    G.add_node("punct", label="Foo.Bar:Baz", source_file="pkg/foobar.py")
    return G


def test_trigrams_basic():
    assert _trigrams("foobar") == {"foo", "oob", "oba", "bar"}
    assert _trigrams("ab") == {"ab"}  # <3 chars -> whole string is the key
    assert _trigrams("") == set()


def test_node_search_text_includes_all_matched_fields():
    G = _make_big_graph()
    text = _node_search_text(G.nodes["punct"], "punct")
    # norm_label, tokenized label, nid, raw source, and tokenized source are all
    # present, NUL-separated so trigrams can't span fields.
    parts = text.split("\x00")
    assert parts[0] == "foo.bar:baz"  # norm_label (punctuation kept)
    assert parts[1] == "foo bar baz"  # label_tokens (tokenized)
    assert parts[2] == "punct"  # nid
    assert parts[3] == "pkg/foobar.py"  # source_file
    assert parts[4] == "pkg foobar py"  # source_file tokens


def test_trigram_candidates_fast_path_fires_for_rare_term():
    G = _make_big_graph()
    cand = _trigram_candidates(G, ["zebraquokkawidget"])
    assert cand is not None  # selective -> fast-path used
    assert "rareA" in cand
    assert len(cand) < G.number_of_nodes()  # a real shrink, not the whole graph


def test_trigram_candidates_falls_back_on_common_term():
    G = _make_big_graph()
    # 'item' is in the label of every one of the 150 'item node N' nodes -> the
    # rarest trigram is still common -> guard returns None (full-scan fallback).
    assert _trigram_candidates(G, ["item"]) is None


def test_trigram_candidates_falls_back_on_short_token():
    G = _make_big_graph()
    assert _trigram_candidates(G, ["ab"]) is None  # <3 chars -> can't trigram-filter


def test_score_nodes_prefilter_is_identical_to_full_scan(monkeypatch):
    G = _make_big_graph()
    queries = [
        "zebraquokkawidget",
        "marmosetgadget handler",
        "foo bar baz",
        "item",
        "node 42",
        "nonexistentxyz",
    ]
    for q in queries:
        terms = _query_terms(q)
        fast = _score_nodes(G, terms)
        _force_full_scan(monkeypatch)
        full = _score_nodes(G, terms)
        monkeypatch.undo()
        assert fast == full, f"prefilter diverged from full scan for {q!r}"


def test_find_node_prefilter_is_identical_to_full_scan(monkeypatch):
    G = _make_big_graph()
    # includes the punctuated label, exercised via its tokenized (label_tokens) form
    for label in [
        "ZebraQuokkaWidget",
        "MarmosetGadget handler",
        "Foo Bar Baz",
        "item node 7",
        "missing",
    ]:
        fast = _find_node(G, label)
        _force_full_scan(monkeypatch)
        full = _find_node(G, label)
        monkeypatch.undo()
        assert fast == full, f"_find_node prefilter diverged (order!) for {label!r}"


def test_find_node_label_tokens_branch_covered_by_index():
    # "foo bar baz" matches label "Foo.Bar:Baz" only via the tokenized label_tokens
    # form (the dotted/colon norm_label never contains the spaced query). The index
    # must surface this node as a candidate, or the prefilter would silently drop it.
    G = _make_big_graph()
    assert _find_node(G, "Foo Bar Baz") == ["punct"]


def test_find_node_source_file_path_prefers_file_level_node():
    G = _make_big_graph()
    source_file = "app/api/example/route.ts"
    # Insert the function node first to prove source-file lookup reorders the
    # file-level node ahead of other nodes from the same file.
    G.add_node(
        "example_route_get",
        label="GET()",
        source_file=source_file,
        source_location="L42",
    )
    G.add_node(
        "example_route",
        label="route.ts",
        source_file=source_file,
        source_location="L1",
    )

    matches = _find_node(G, source_file)

    assert matches[0] == "example_route"
    assert "example_route_get" in matches


def test_trigram_index_cached_and_rebuilt_per_graph():
    G = _make_big_graph()
    idx1 = _get_trigram_index(G)
    assert idx1 is _get_trigram_index(G)  # cached per graph object
    # The cache is keyed by graph identity OUTSIDE graph state: the postings hold
    # array('i') values, which are not JSON representable, so storing them in
    # G.graph made a query mutate graph state and broke re-encoding.
    assert "_trigram_index" not in G.graph
    assert _TRIGRAM_INDEX_CACHE[G] is idx1
    G2 = _make_big_graph()
    assert _get_trigram_index(G2) is not idx1  # a fresh graph rebuilds (reload safety)


def test_query_terms_strips_search_punctuation():
    # "what" is a question stopword (dropped); punctuation is still stripped from "extract?".
    assert _query_terms("what calls extract?") == ["calls", "extract"]


def test_query_terms_drops_question_stopwords():
    # Natural-language question words are dropped so content words drive seeding:
    # "how does the frontier cache work" must reduce to the content terms, or it
    # seeds on "how"/"the"/"work" (which prefix-match prose labels) instead.
    assert _query_terms("how does the frontier cache work") == ["frontier", "cache"]


def test_query_terms_all_stopwords_falls_back_to_unfiltered():
    # An all-stopword query keeps its terms rather than seeding on nothing.
    assert _query_terms("how does it work") == ["how", "does", "work"]


def test_query_terms_filters_only_short_english_terms(monkeypatch):
    import graphify.serve as serve_mod

    class FakeJieba:
        def cut_text(self, text):
            return {
                "前端": ["前端"],
                "依赖": ["依赖"],
                "安装": ["安装"],
                "包管理器": ["包", "管理器"],
                "项目约定": ["项目", "约定"],
                "a前": ["a", "前"],
            }[text]

    monkeypatch.setattr(serve_mod, "_chinese_tokenizer", FakeJieba())
    terms = _query_terms("前端 dependency 依赖 install 安装 to of 包管理器 项目约定 a前")
    assert terms == [
        "前端",
        "dependency",
        "依赖",
        "install",
        "安装",
        "包",
        "管理器",
        "包管理器",
        "项目",
        "约定",
        "项目约定",
        "前",
        "a前",
    ]


def test_query_graph_text_keeps_short_non_english_terms():
    G = nx.Graph()
    G.add_node(
        "frontend", label="前端", source_file="docs/前端.md", source_location="L1", community=0
    )
    text = _query_graph_text(G, "前端", mode="bfs", depth=1)
    assert "No matching nodes found." not in text
    assert "NODE 前端" in text


def test_infer_context_filters_for_calls_question():
    assert _infer_context_filters("who calls extract") == ["call"]


def test_resolve_context_filters_explicit_overrides_heuristic():
    filters, source = _resolve_context_filters("who calls extract", ["field"])
    assert filters == ["field"]
    assert source == "explicit"


# --- _bfs ---


def test_bfs_depth_1():
    G = _make_graph()
    visited, edges = _bfs(G, ["n1"], depth=1)
    assert "n1" in visited
    assert "n2" in visited  # direct neighbor
    assert "n3" not in visited  # 2 hops away


def test_bfs_depth_2():
    G = _make_graph()
    visited, edges = _bfs(G, ["n1"], depth=2)
    assert "n3" in visited  # n1 -> n2 -> n3


def test_bfs_disconnected():
    G = _make_graph()
    visited, edges = _bfs(G, ["n5"], depth=3)
    assert visited == {"n5"}  # isolated node


def test_bfs_returns_edges():
    G = _make_graph()
    visited, edges = _bfs(G, ["n1"], depth=1)
    assert len(edges) >= 1
    assert any(u == "n1" or v == "n1" for u, v in edges)


def test_filter_graph_by_context_limits_traversal():
    G = _make_graph()
    filtered = _filter_graph_by_context(G, ["call"])
    visited, edges = _bfs(filtered, ["n1"], depth=2)
    assert "n2" in visited
    assert "n3" not in visited
    assert edges == [("n1", "n2")]


def test_filter_graph_by_context_preserves_graph_metadata():
    G = _make_graph()
    G.graph.update(
        graphify_profile={"graph_type": "multidigraph"},
        corpus_name="fixture",
        custom_meta={"owner": "tests"},
    )

    filtered = _filter_graph_by_context(G, ["call"])

    assert filtered.graph == G.graph


@pytest.mark.parametrize("traversal", [_bfs, _dfs])
def test_traversal_hub_detection_uses_distinct_neighbors(traversal):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    for index in range(100):
        graph.add_edge("a", "b", key=f"parallel-{index}", relation="calls")
    graph.add_edge("b", "c", key="continuation", relation="calls")

    visited, _edges = traversal(graph, ["a"], depth=2)

    assert visited == {"a", "b", "c"}


def test_query_output_is_stable_across_insertion_order_and_mixed_ids():
    def make_graph(order):
        graph = nx.Graph()
        graph.graph["corpus_name"] = "stable"
        for node_id in order:
            graph.add_node(
                node_id,
                label="target" if node_id == "seed" else f"peer-{node_id}",
                source_file=f"{node_id}.py",
                source_location="L1",
                community=0,
            )
        for node_id in order:
            if node_id != "seed":
                graph.add_edge("seed", node_id, relation="calls", confidence="EXTRACTED")
        return graph

    first = make_graph(["seed", "z", 1, "a"])
    second = make_graph(["seed", "a", 1, "z"])

    first_min = _minimum_budget(
        lambda b: _query_graph_text(first, "target", depth=1, token_budget=b)
    )
    second_min = _minimum_budget(
        lambda b: _query_graph_text(second, "target", depth=1, token_budget=b)
    )
    assert first_min == second_min, (first_min, second_min)
    assert _query_graph_text(first, "target", depth=1, token_budget=first_min) == _query_graph_text(
        second, "target", depth=1, token_budget=first_min
    )


def test_query_truncation_is_stable_across_hash_seeds():
    # The refusal minimum is itself part of the determinism contract: catch the
    # refusal, retry at the reported minimum, and compare BOTH across seeds.
    code = """
import networkx as nx
from graphify.serve import _query_graph_text, InsufficientBudgetError

graph = nx.Graph()
graph.add_node("seed", label="target", source_file="seed.py", source_location="L1")
for node_id in {"z", "a", "m", "q", "b", "y"}:
    graph.add_node(node_id, label=f"peer-{node_id}", source_file=f"{node_id}.py", source_location="L1")
    graph.add_edge("seed", node_id, relation="calls", confidence="EXTRACTED")
try:
    _query_graph_text(graph, "target", depth=1, token_budget=45)
    raise SystemExit("expected refusal at budget 45")
except InsufficientBudgetError as exc:
    minimum = exc.required_minimum
print(minimum)
print(_query_graph_text(graph, "target", depth=1, token_budget=minimum))
"""
    outputs = []
    for seed in ("1", "2", "3", "42"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONIOENCODING": "utf-8",
            },
        )
        outputs.append(result.stdout)

    assert len(set(outputs)) == 1
    assert "truncated" in outputs[0]


# --- _dfs ---


def test_dfs_depth_1():
    G = _make_graph()
    visited, edges = _dfs(G, ["n1"], depth=1)
    assert "n1" in visited
    assert "n2" in visited
    assert "n3" not in visited


def test_dfs_full_chain():
    G = _make_graph()
    visited, edges = _dfs(G, ["n1"], depth=5)
    assert {"n1", "n2", "n3", "n4"}.issubset(visited)


# --- _subgraph_to_text ---


def test_subgraph_to_text_contains_labels():
    G = _make_graph()
    text = _subgraph_to_text(G, {"n1", "n2"}, [("n1", "n2")])
    assert "extract" in text
    assert "cluster" in text


def test_subgraph_to_text_truncates():
    """At the viable boundary a large expansion still drops optional units.

    The former four-node fixture can no longer show this: its entire output now
    fits as soon as the primary unit plus notice fits, so truncation needs a
    fixture with genuine optional expansion.
    """
    G = _seed_starved_graph()

    def render(budget):
        return _query_graph_text(G, "target", depth=2, token_budget=budget)

    assert "truncated" in render(_minimum_budget(render))


def test_subgraph_to_text_refuses_budget_below_one_complete_record():
    """A budget too small for one whole record refuses instead of splitting it.

    The previous renderer sliced the joined output at a character offset, so a
    tiny budget emitted a partial NODE line as though it were evidence.
    """
    G = _make_graph()
    with pytest.raises(InsufficientBudgetError) as excinfo:
        _subgraph_to_text(G, {"n1", "n2", "n3", "n4"}, [("n1", "n2")], token_budget=1)
    assert excinfo.value.required_minimum > 1


def test_subgraph_to_text_never_emits_a_partial_record():
    """Every emitted line is a complete NODE or EDGE record at any viable budget."""
    G = _make_graph()
    for budget in range(1, 200, 3):
        try:
            text = _subgraph_to_text(G, {"n1", "n2", "n3", "n4"}, [("n1", "n2")], budget)
        except InsufficientBudgetError:
            continue
        assert len(text) <= budget * 3, (budget, len(text))
        for line in text.splitlines():
            if line.startswith("..."):
                continue
            assert line.startswith(("NODE ", "EDGE ")), (budget, line)
            assert line.endswith(("]", ")")) or "-->" in line, (budget, line)


def test_subgraph_to_text_edge_included():
    G = _make_graph()
    text = _subgraph_to_text(G, {"n1", "n2"}, [("n1", "n2")])
    assert "EDGE" in text
    assert "calls" in text


def test_subgraph_to_text_includes_edge_context():
    G = _make_graph()
    text = _subgraph_to_text(G, {"n1", "n2"}, [("n1", "n2")])
    assert "context=call" in text


# --- work-memory overlay annotation on NODE lines -----------------------------


def test_subgraph_to_text_annotates_node_with_learning_status():
    """An annotated node gets a `learning=<status>` suffix inside its NODE
    bracket; an un-annotated node gets none."""
    G = _make_graph()
    G.graph["_learning_overlay"] = {
        "n1": {"status": "preferred", "stale": False},
    }
    text = _subgraph_to_text(G, {"n1", "n2"}, [("n1", "n2")])
    lines = {line.split()[1]: line for line in text.splitlines() if line.startswith("NODE ")}
    assert "learning=preferred]" in lines["extract"]
    assert "learning=" not in lines["cluster"]  # un-annotated node


def test_subgraph_to_text_marks_stale_status():
    G = _make_graph()
    G.graph["_learning_overlay"] = {"n1": {"status": "contested", "stale": True}}
    text = _subgraph_to_text(G, {"n1"}, [])
    assert "learning=contested:stale]" in text


def test_subgraph_to_text_learning_suffix_counts_against_budget():
    """The learning= suffix is part of the NODE line BEFORE the budget cut, so it
    is included in the char_budget accounting (a budget tight enough to fit the
    bare line but not the suffixed line forces truncation)."""
    # Needs enough optional evidence that the ANNOTATED primary unit plus notice
    # still fits while the full annotated render does not. A small fixture can no
    # longer show this: admission is unit-based, so annotation that overflows the
    # primary unit refuses outright instead of truncating.
    G = _seed_starved_graph()
    bare_full = _query_graph_text(G, "target", depth=2, token_budget=4000)
    # token_budget chosen so the un-annotated render fits without truncation...
    budget = (len(bare_full) // 3) + 1
    assert "truncated" not in _query_graph_text(G, "target", depth=2, token_budget=budget)
    # ...but once every node carries a learning= suffix, the same budget overflows.
    G.graph["_learning_overlay"] = {
        node: {"status": "preferred", "stale": False} for node in G.nodes()
    }
    annotated = _query_graph_text(G, "target", depth=2, token_budget=budget)
    assert "learning=preferred" in annotated
    assert "truncated" in annotated


def test_subgraph_to_text_no_overlay_is_unchanged():
    """With no overlay on the graph, NODE lines carry no learning= suffix."""
    G = _make_graph()
    text = _subgraph_to_text(G, {"n1", "n2"}, [("n1", "n2")])
    assert "learning=" not in text


def test_query_graph_text_explicit_context_filter_changes_traversal():
    G = _make_graph()
    text = _query_graph_text(
        G, "extract", mode="bfs", depth=2, token_budget=2000, context_filters=["call"]
    )
    assert "Context: call (explicit)" in text
    assert "cluster" in text
    assert "build" not in text


def test_query_graph_text_heuristic_context_filter_changes_traversal():
    G = _make_graph()
    text = _query_graph_text(G, "who calls extract", mode="bfs", depth=2, token_budget=2000)
    assert "Context: call (heuristic)" in text
    assert "cluster" in text
    assert "build" not in text


# --- _load_graph ---


def test_load_graph_roundtrip(tmp_path):
    G = _make_graph()
    data = json_graph.node_link_data(G, edges="links")
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    G2 = _load_graph(str(p))
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()


def test_load_graph_preserves_keyed_multigraph_state_and_metadata(tmp_path):
    payload = {
        "schema_version": 1,
        "directed": True,
        "multigraph": True,
        "graph": {
            "graphify_profile": {"graph_type": "multidigraph"},
            "corpus_name": "serve-fixture",
        },
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [
            {
                "source": "a",
                "target": "b",
                "key": "call-L5",
                "relation": "calls",
                "context": "first",
                "_origin": "ast",
            },
            {
                "source": "a",
                "target": "b",
                "key": "call-L9",
                "relation": "calls",
                "context": "second",
                "_origin": "ast",
            },
        ],
        "hyperedges": [{"id": "flow", "nodes": ["a", "b"], "label": "Flow"}],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    graph = _load_graph(str(path))

    assert type(graph) is nx.MultiDiGraph
    assert set(graph["a"]["b"]) == {"call-L5", "call-L9"}
    assert {data["context"] for data in graph["a"]["b"].values()} == {"first", "second"}
    assert graph.graph["corpus_name"] == "serve-fixture"
    assert graph.graph["hyperedges"] == [{"id": "flow", "nodes": ["a", "b"], "label": "Flow"}]


def test_load_graph_missing_file(tmp_path):
    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    with pytest.raises(SystemExit):
        _load_graph(str(graphify_dir / "nonexistent.json"))


def test_load_graph_rejects_oversized_file(monkeypatch, tmp_path, capsys):
    # #F4: oversized graph.json must fail fast (SystemExit) with a clear error.
    G = _make_graph()
    data = json_graph.node_link_data(G, edges="links")
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 16)
    with pytest.raises(SystemExit):
        _load_graph(str(p))
    err = capsys.readouterr().err
    assert "exceeds" in err
    assert "byte cap" in err


def test_load_graph_accepts_under_cap(monkeypatch, tmp_path):
    # Verifies the cap path does not regress the normal load.
    G = _make_graph()
    data = json_graph.node_link_data(G, edges="links")
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    # Cap well above the actual file size — load proceeds.
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 10 * 1024 * 1024)
    G2 = _load_graph(str(p))
    assert G2.number_of_nodes() == G.number_of_nodes()


# --- #874: MCP hot-reload ---


def _write_graph(path, nodes: list[str]) -> None:
    """Write a minimal graph.json with the given node IDs."""
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n, label=n, community=0)
    data = json_graph.node_link_data(G, edges="links")
    path.write_text(json.dumps(data), encoding="utf-8")


def test_maybe_reload_detects_graph_change(tmp_path):
    """serve() picks up a new graph.json written after startup (#874)."""
    import time

    out = tmp_path / "graphify-out"
    out.mkdir()
    graph_path = out / "graph.json"
    _write_graph(graph_path, ["alpha", "beta"])

    # Bootstrap _load_graph + _communities_from_graph to verify the reload path
    G1 = _load_graph(str(graph_path))
    assert set(G1.nodes()) == {"alpha", "beta"}

    # Simulate file changing (bump mtime by touching)
    time.sleep(0.01)
    _write_graph(graph_path, ["alpha", "beta", "gamma"])

    G2 = _load_graph(str(graph_path))
    assert "gamma" in G2.nodes()


def test_load_graph_cache_key_changes_with_content(tmp_path):
    """mtime_ns + size uniquely identifies a graph version (#874)."""
    import time

    out = tmp_path / "graphify-out"
    out.mkdir()
    graph_path = out / "graph.json"
    _write_graph(graph_path, ["a"])

    s1 = graph_path.stat()
    key1 = (s1.st_mtime_ns, s1.st_size)

    time.sleep(0.01)
    _write_graph(graph_path, ["a", "b"])

    s2 = graph_path.stat()
    key2 = (s2.st_mtime_ns, s2.st_size)

    assert key1 != key2, "stat key must change when file content changes"


# --- IDF weighting tests (#897) ---


def _make_noisy_graph() -> nx.Graph:
    """20 error-handler nodes + 1 rare identifier: FooBarService."""
    G = nx.Graph()
    for i in range(20):
        G.add_node(f"err{i}", label=f"error_handler_{i}", source_file=f"err{i}.py", community=0)
        if i > 0:
            G.add_edge(f"err{i - 1}", f"err{i}", relation="calls", confidence="EXTRACTED")
    G.add_node("fbs", label="FooBarService", source_file="service.py", community=1)
    G.add_node("fbs_dep", label="ServiceClient", source_file="client.py", community=1)
    G.add_edge("fbs", "fbs_dep", relation="uses", confidence="EXTRACTED")
    return G


def test_idf_downweights_common_terms():
    """'error' matches 20 nodes, 'foobarservice' matches 1 — IDF should make
    FooBarService rank first despite error's higher raw frequency."""
    G = _make_noisy_graph()
    scored = _score_nodes(G, ["foobarservice", "error"])
    assert scored, "should have results"
    assert scored[0][1] == "fbs", f"FooBarService should rank first, got {scored[0][1]}"


def test_idf_cached_per_graph_outside_graph_state():
    """IDF results are memoized per graph object so repeated queries don't recompute.

    The cache is held outside G.graph: a query must never mutate graph state.
    """
    G = _make_graph()
    _score_nodes(G, ["extract"])
    assert "_idf_cache" not in G.graph
    assert "extract" in _IDF_CACHE[G]


def test_idf_new_graph_starts_fresh():
    """Two separate graph instances must not share an IDF cache."""
    G1 = _make_graph()
    G2 = _make_graph()
    _score_nodes(G1, ["extract"])
    assert G2 not in _IDF_CACHE
    assert "_idf_cache" not in G2.graph


def test_idf_rare_term_gets_high_weight():
    """A term matching only 1 of N nodes should get IDF > 1."""
    G = _make_graph()  # 5 nodes
    idf = _compute_idf(G, ["extract"])
    # extract matches only n1: IDF = log(1 + 5/2) ≈ 1.25
    assert idf["extract"] > 1.0


def test_idf_common_term_gets_low_weight():
    """A term matching most nodes should get IDF < 1."""
    G = nx.Graph()
    # 'handle' in every node label
    for i in range(20):
        G.add_node(f"n{i}", label=f"handle_{i}", source_file=f"f{i}.py")
    idf = _compute_idf(G, ["handle"])
    assert idf["handle"] < 1.0


# --- _pick_seeds tests (#897) ---


def test_pick_seeds_dominant_identifier_gives_one_seed():
    """FooBarService at 1000 vs error nodes at 1.0 → only 1 seed chosen."""
    scored = [(1000.0, "fbs"), (1.0, "err1"), (0.9, "err2")]
    seeds = _pick_seeds(scored)
    assert seeds == ["fbs"]


def test_pick_seeds_close_scores_keeps_multiple():
    """When all scores are within 20% of the top, keep up to 3 seeds."""
    scored = [(10.0, "a"), (9.0, "b"), (8.5, "c")]
    seeds = _pick_seeds(scored)
    assert len(seeds) == 3


def test_pick_seeds_empty():
    assert _pick_seeds([]) == []


def test_pick_seeds_single():
    assert _pick_seeds([(5.0, "x")]) == ["x"]


def test_pick_seeds_respects_max_k():
    """Never return more than max_k seeds even when all scores are close."""
    scored = [(10.0, f"n{i}") for i in range(10)]
    seeds = _pick_seeds(scored, max_k=3)
    assert len(seeds) == 3


# --- actionable truncation hint (#897) ---


def test_subgraph_to_text_truncation_hint_is_actionable():
    """Truncation message must tell Claude what to do, not just say truncated."""
    G = _seed_starved_graph()

    def render(budget):
        return _query_graph_text(G, "target", depth=2, token_budget=budget)

    text = render(_minimum_budget(render))
    assert "truncated" in text
    assert "get_node" in text or "context_filter" in text


def test_insufficient_budget_error_carries_actionable_minimum():
    """The refusal must name a retry budget that actually works."""
    G = _make_graph()
    with pytest.raises(InsufficientBudgetError) as excinfo:
        _subgraph_to_text(G, {"n1", "n2", "n3", "n4"}, [("n1", "n2")], token_budget=1)
    retry = excinfo.value.required_minimum
    # Retrying at the reported minimum must not raise again.
    text = _subgraph_to_text(G, {"n1", "n2", "n3", "n4"}, [("n1", "n2")], token_budget=retry)
    assert text.splitlines()[0].startswith("NODE ")


# --- integration: identifier + noise query seeds from identifier (#897) ---


def test_query_seeds_from_identifier_not_noise():
    """'FooBarService error handling' should expand from FooBarService,
    not from error-handler nodes, so ServiceClient appears in results."""
    G = _make_noisy_graph()
    text = _query_graph_text(G, "FooBarService error handling", mode="bfs", depth=2)
    assert "FooBarService" in text
    assert "ServiceClient" in text


def test_query_graph_text_parameter_type_context_filter_changes_traversal():
    import networkx as nx
    from graphify.serve import _query_graph_text

    graph = nx.Graph()
    graph.add_node("process", label="process", source_file="sample.cs", source_location="L20")
    graph.add_node("payload", label="Payload", source_file="sample.cs", source_location="L5")
    graph.add_node("other", label="PayloadFactory", source_file="sample.cs", source_location="L40")
    graph.add_edge(
        "process",
        "payload",
        relation="references",
        context="parameter_type",
        confidence="EXTRACTED",
    )
    graph.add_edge("process", "other", relation="calls", context="call", confidence="EXTRACTED")

    text = _query_graph_text(graph, "who accepts Payload", context_filters=["parameter_type"])

    assert "parameter_type" in text
    assert "Payload" in text
    assert "PayloadFactory" not in text


def test_query_graph_text_context_filter_aliases_resolve():
    from graphify.serve import _normalize_context_filters

    assert _normalize_context_filters(["param"]) == ["parameter_type"]
    assert _normalize_context_filters(["parameter"]) == ["parameter_type"]
    assert _normalize_context_filters(["return"]) == ["return_type"]
    assert _normalize_context_filters(["returns"]) == ["return_type"]
    assert _normalize_context_filters(["generic"]) == ["generic_arg"]
    assert _normalize_context_filters(["generics"]) == ["generic_arg"]
    assert _normalize_context_filters(["annotation"]) == ["attribute"]
    assert _normalize_context_filters(["decorator"]) == ["attribute"]
    # Pass-through for already-canonical values
    assert _normalize_context_filters(["parameter_type"]) == ["parameter_type"]
    assert _normalize_context_filters(["field"]) == ["field"]


# --- Chinese segmentation ---


def test_query_terms_chinese_segments_with_cached_tokenizer(monkeypatch):
    """Chinese text should use the cached tokenizer and keep the original term."""
    import graphify.serve as serve_mod

    class FakeJieba:
        def cut_text(self, text):
            assert text == "页面路由"
            return ["页面", "路由"]

    monkeypatch.setattr(serve_mod, "_chinese_tokenizer", FakeJieba())
    terms = _query_terms("页面路由")
    assert terms == ["页面", "路由", "页面路由"]


def test_query_terms_chinese_mixed():
    """Mixed Chinese and English text should be handled correctly."""
    terms = _query_terms("前端 router 路由配置")
    assert "前端" in terms
    assert "router" in terms
    assert "路由" in terms
    assert "配置" in terms


def test_query_terms_non_chinese_scripts_are_not_segmented():
    """Japanese kana and Hangul are kept as terms but not segmented as Chinese."""
    import graphify.serve as serve_mod

    assert not serve_mod._has_chinese("かなカナ한글")
    assert serve_mod._query_terms("かなカナ한글") == ["かなカナ한글"]


def test_query_terms_chinese_no_tokenizer_fallback(monkeypatch):
    """When the tokenizer is not installed, fall back to character bigrams."""
    import graphify.serve as serve_mod

    monkeypatch.setattr(serve_mod, "_chinese_tokenizer", None)
    terms = serve_mod._query_terms("页面路由")
    # bigram fallback: ["页面", "面路", "路由"] + original "页面路由"
    assert "页面" in terms
    assert "路由" in terms
    assert "页面路由" in terms
    assert len(terms) == 4


def test_score_nodes_chinese_substring_match():
    """Searching for '路由' should match a node with label containing '路由'."""
    G = nx.Graph()
    G.add_node("n1", label="路由桥接核对表", source_file="doc.md", community=0)
    G.add_node("n2", label="其他内容", source_file="doc.md", community=0)
    scored = _score_nodes(G, ["路由"])
    nids = [nid for _, nid in scored]
    assert "n1" in nids
    assert "n2" not in nids


def test_query_text_chinese_finds_routing_nodes():
    """Full pipeline: '页面路由' should find nodes with '路由' in label."""
    G = nx.Graph()
    G.add_node(
        "parent", label="页面路由规范", source_file="doc.md", source_location="L1", community=0
    )
    G.add_node(
        "child", label="路由桥接核对表", source_file="doc.md", source_location="L10", community=0
    )
    G.add_edge("parent", "child", relation="contains", confidence="EXTRACTED")
    text = _query_graph_text(G, "页面路由", mode="bfs", depth=2)
    assert "No matching nodes found." not in text
    assert "路由" in text


# --- get_community header (#1448): show the community name, no placeholder doubling ---


def test_community_header_shows_real_name():
    assert _community_header(12, "Auth & Sessions") == "Community 12 — Auth & Sessions"


def test_community_header_skips_placeholder_name():
    # community_name is written as the "Community N" placeholder for unnamed
    # communities; the header must not read "Community 12 — Community 12".
    assert _community_header(12, "Community 12") == "Community 12"


def test_community_header_falls_back_when_no_name():
    assert _community_header(7, None) == "Community 7"
    assert _community_header(7, "") == "Community 7"


def test_community_header_sanitizes_name():
    # control characters in an LLM-derived name are stripped (F-010)
    out = _community_header(3, "Pay\x00ments\x1b[31m")
    assert out.startswith("Community 3 — ")
    assert "\x00" not in out and "\x1b" not in out


# --- D0: seed-first rendering (queried symbol must survive the budget) ---


def _seed_starved_graph() -> nx.Graph:
    """Low-degree queried seed surrounded by high-degree peers.

    ``_subgraph_to_text`` orders the expansion by descending distinct-neighbour
    degree, so without seed-first rendering the queried symbol sorts last and a
    realistic small budget cuts it out of its own answer.
    """
    graph = nx.Graph()
    graph.add_node("seed", label="target", source_file="seed.py", source_location="L1", community=0)
    for peer in ("p1", "p2"):
        graph.add_node(
            peer, label=f"peer-{peer}", source_file=f"{peer}.py", source_location="L1", community=0
        )
        graph.add_edge("seed", peer, relation="calls", confidence="EXTRACTED", context="call")
    # Inflate peer degree so both outrank the seed in the degree-sorted expansion.
    for filler in ("f1", "f2", "f3", "f4", "f5", "f6"):
        graph.add_node(
            filler, label=f"filler-{filler}", source_file=f"{filler}.py", source_location="L1"
        )
        for peer in ("p1", "p2"):
            graph.add_edge(peer, filler, relation="calls", confidence="EXTRACTED", context="call")
    return graph


def _minimum_budget(render) -> int:
    """Lowest budget at which ``render(budget)`` succeeds.

    Budgets are derived, never hardcoded: the minimum depends on fixture label
    widths and on the truncation notice, which embeds the budget itself. The
    refusal reports the minimum, so this is O(1); that the reported value really
    is the first viable budget is asserted separately by
    ``test_reported_minimum_equals_first_viable_budget``.
    """
    try:
        render(1)
    except InsufficientBudgetError as excinfo:
        return excinfo.required_minimum
    return 1


def test_subgraph_to_text_seed_argument_renders_seed_first():
    """Control: the seed-first capability itself works when the argument is passed.

    This must pass before and after the D0 fix. If it ever fails, the D0
    regression tests below are proving nothing.
    """
    graph = _seed_starved_graph()
    nodes = {"seed", "p1", "p2"}
    edges = [("seed", "p1"), ("seed", "p2")]

    def render(budget):
        return _subgraph_to_text(graph, nodes, edges, budget, seeds=["seed"])

    out = render(_minimum_budget(render))
    node_lines = [line for line in out.splitlines() if line.startswith("NODE ")]
    assert node_lines, "control produced no NODE lines; the fixture is broken"
    assert node_lines[0].startswith("NODE target ["), node_lines


def test_query_graph_text_keeps_queried_seed_at_its_minimum_budget():
    """D0: the queried symbol must appear in the evidence body, not only the header."""
    graph = _seed_starved_graph()

    def render(budget):
        return _query_graph_text(graph, "target", depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    header, _, body = out.partition("\n\n")
    assert "target" in header, "fixture no longer selects the intended seed"
    assert "NODE target [" in body, f"queried seed missing from evidence body:\n{out}"


@pytest.mark.parametrize("budget", [20, 50])
def test_reproduced_budgets_now_refuse_with_one_stable_minimum(budget):
    """The shipped-CLI reproduction budgets are below one complete primary unit.

    They previously returned truncated apparent success without the seed. They
    must now refuse, and every refusal must report the same actionable minimum.
    """
    graph = _seed_starved_graph()
    with pytest.raises(InsufficientBudgetError) as excinfo:
        _query_graph_text(graph, "target", depth=1, token_budget=budget)
    minimum = excinfo.value.required_minimum
    assert minimum > budget

    with pytest.raises(InsufficientBudgetError) as other:
        _query_graph_text(graph, "target", depth=1, token_budget=1)
    assert other.value.required_minimum == minimum, "minimum varies with requested budget"


def test_minimum_budget_is_an_exact_stable_fixed_point():
    """Below the minimum always refuses; at the minimum the response fits."""
    graph = _seed_starved_graph()

    def render(budget):
        return _query_graph_text(graph, "target", depth=2, token_budget=budget)

    minimum = _minimum_budget(render)
    for budget in (1, minimum // 2, minimum - 1):
        if budget < 1:
            continue
        with pytest.raises(InsufficientBudgetError) as excinfo:
            render(budget)
        assert excinfo.value.required_minimum == minimum, (budget, excinfo.value.required_minimum)

    out = render(minimum)
    assert len(out) <= minimum * 3, (len(out), minimum * 3)
    assert "NODE target [" in out
    assert "EDGE " in out


@pytest.mark.parametrize("mode", ["bfs", "dfs"])
def test_query_graph_text_seed_first_in_both_traversal_modes(mode):
    graph = _seed_starved_graph()

    def render(budget):
        return _query_graph_text(graph, "target", mode=mode, depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    _, _, body = out.partition("\n\n")
    node_lines = [line for line in body.splitlines() if line.startswith("NODE ")]
    assert node_lines, f"mode={mode} produced no NODE lines"
    assert node_lines[0].startswith("NODE target ["), node_lines


def test_query_graph_text_seed_first_is_insertion_order_stable():
    """Seed-first rendering must not reintroduce an ordering dependency."""

    def build(order):
        graph = nx.Graph()
        graph.add_node(
            "seed", label="target", source_file="seed.py", source_location="L1", community=0
        )
        for peer in order:
            graph.add_node(
                peer,
                label=f"peer-{peer}",
                source_file=f"{peer}.py",
                source_location="L1",
                community=0,
            )
            graph.add_edge("seed", peer, relation="calls", confidence="EXTRACTED")
        return graph

    first, second = build(["z", "a", "m"]), build(["m", "z", "a"])
    first_min = _minimum_budget(
        lambda b: _query_graph_text(first, "target", depth=1, token_budget=b)
    )
    second_min = _minimum_budget(
        lambda b: _query_graph_text(second, "target", depth=1, token_budget=b)
    )
    assert first_min == second_min, (first_min, second_min)
    shared = first_min
    assert _query_graph_text(first, "target", depth=1, token_budget=shared) == _query_graph_text(
        second, "target", depth=1, token_budget=shared
    )


# --- D0: evidence-unit atomicity and omission accounting ---


def _edge_endpoint_labels(line: str) -> tuple[str, str]:
    body = line[len("EDGE ") :]
    return body.split(" --", 1)[0], body.rsplit("--> ", 1)[1]


def test_reported_minimum_equals_first_viable_budget():
    """The minimum must be the LESSER of the complete response and primary+notice.

    Solving only the primary-plus-notice fixed point over-reported the minimum
    whenever rendering everything was cheaper than rendering a truncated form.
    """
    for depth in (1, 2):
        graph = _seed_starved_graph()

        def render(budget, _depth=depth, _graph=graph):
            return _query_graph_text(_graph, "target", depth=_depth, token_budget=budget)

        # Scan only up to the reported minimum: every budget below it must
        # refuse with that same value, and the minimum itself must succeed.
        minimum = _minimum_budget(render)
        reported = set()
        for budget in range(1, minimum):
            with pytest.raises(InsufficientBudgetError) as excinfo:
                render(budget)
            reported.add(excinfo.value.required_minimum)
        assert reported in ({minimum}, set()), (depth, reported, minimum)
        render(minimum)


def test_complete_response_cheaper_than_truncated_is_not_refused():
    """A budget that fits everything must never refuse on truncated-form cost."""
    graph = _make_graph()
    selected = {"n1", "n2", "n3", "n4"}
    edges = [("n1", "n2")]
    complete = _subgraph_to_text(graph, selected, edges, token_budget=4000)
    exact = _estimate_tokens_for_budget(len(complete))
    out = _subgraph_to_text(graph, selected, edges, token_budget=exact)
    assert "truncated" not in out
    assert len(out) <= exact * 3


def test_no_relationship_is_emitted_without_both_endpoint_nodes():
    """Expansion units are atomic: an EDGE cannot outlive its endpoint records."""
    graph = _seed_starved_graph()
    base = _minimum_budget(lambda b: _query_graph_text(graph, "target", depth=2, token_budget=b))
    for budget in range(base, base + 90, 3):
        try:
            out = _query_graph_text(graph, "target", depth=2, token_budget=budget)
        except InsufficientBudgetError:
            continue
        labels = {
            line.split("NODE ", 1)[1].split(" [")[0]
            for line in out.splitlines()
            if line.startswith("NODE ")
        }
        for line in out.splitlines():
            if not line.startswith("EDGE "):
                continue
            source, target = _edge_endpoint_labels(line)
            assert source in labels, (budget, line, sorted(labels))
            assert target in labels, (budget, line, sorted(labels))


def test_omission_accounting_counts_keyed_records_not_bundles():
    """Three parallel keyed records omitted must report three, not one bundle."""
    graph = nx.MultiDiGraph()
    graph.add_node("a", label="target", source_file="a.py", source_location="L1")
    graph.add_node("b", label="bee", source_file="b.py", source_location="L1")
    graph.add_node("c", label="cee", source_file="c.py", source_location="L1")
    graph.add_edge("a", "b", key="k1", relation="r1", confidence="EXTRACTED")
    for key, relation in (("k1", "p1"), ("k2", "p2"), ("k3", "p3")):
        graph.add_edge("a", "c", key=key, relation=relation, confidence="EXTRACTED")

    def render(budget):
        return _query_graph_text(graph, "target", depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    assert "3 relationships omitted" in out, out


def test_primary_edge_direction_is_preserved_on_directed_graphs():
    graph = nx.DiGraph()
    graph.add_node("a", label="target", source_file="a.py", source_location="L1")
    graph.add_node("b", label="callee", source_file="b.py", source_location="L1")
    graph.add_edge("a", "b", relation="calls", confidence="EXTRACTED")

    def render(budget):
        return _query_graph_text(graph, "target", depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    edges = [line for line in out.splitlines() if line.startswith("EDGE ")]
    assert edges, out
    source, target = _edge_endpoint_labels(edges[0])
    assert (source, target) == ("target", "callee"), edges


@pytest.mark.parametrize("seed_count", [1, 2, 3])
def test_primary_unit_belongs_to_the_primary_seed(seed_count):
    """With several seeds the primary unit must come from the FIRST seed.

    Selecting the first rendered edge globally picked a secondary seed's
    relationship and an unrelated node as its counterpart.
    """
    graph = nx.Graph()
    for index in range(seed_count):
        graph.add_node(
            f"s{index}", label="target", source_file=f"s{index}.py", source_location="L1"
        )
        graph.add_node(
            f"c{index}", label=f"counter{index}", source_file=f"c{index}.py", source_location="L1"
        )
        graph.add_edge(f"s{index}", f"c{index}", relation=f"rel{index}", confidence="EXTRACTED")

    def render(budget):
        return _query_graph_text(graph, "target", depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    node_lines = [line for line in out.splitlines() if line.startswith("NODE ")]
    assert node_lines[0].startswith("NODE target ["), out
    for line in out.splitlines():
        if line.startswith("EDGE "):
            source, target = _edge_endpoint_labels(line)
            assert source in {node.split("NODE ", 1)[1].split(" [")[0] for node in node_lines}, out


def _multi_seed_expansion_graph() -> nx.Graph:
    """Two exact seeds where the primary has several competing expansions."""
    graph = nx.Graph()
    graph.add_node("p", label="target", source_file="p.py", source_location="L1")
    graph.add_node("s2", label="target", source_file="s2.py", source_location="L1")
    for node, relation in (("a", "primary"), ("b", "expand-1"), ("c", "expand-2")):
        graph.add_node(node, label=node * 3, source_file=f"{node}.py", source_location="L1")
        graph.add_edge("p", node, relation=relation, confidence="EXTRACTED")
    graph.add_node("z", label="zzz", source_file="z.py", source_location="L1")
    graph.add_edge("s2", "z", relation="secondary", confidence="EXTRACTED")
    return graph


def test_secondary_seeds_outrank_further_primary_expansion():
    """Allocation order is policy: at most ONE extra primary expansion precedes
    the selected secondary seeds.

    Admitting every relationship expansion before secondary seeds let expansion
    from one seed consume the packet before another selected seed appeared —
    the original discovery failure, one level down.
    """
    graph = _multi_seed_expansion_graph()
    checked = 0
    for budget in range(1, 400, 5):
        try:
            out = _query_graph_text(graph, "target", depth=1, token_budget=budget)
        except InsufficientBudgetError:
            continue
        lines = out.splitlines()
        if not any(line.startswith("...") for line in lines):
            continue  # only truncated responses constrain allocation order
        second_seed_present = any(
            line.startswith("NODE ") and "src=s2.py" in line for line in lines
        )
        primary_expansions = sum(
            1 for line in lines if line.startswith("EDGE ") and "expand-" in line
        )
        checked += 1
        if primary_expansions >= 2:
            assert second_seed_present, (
                f"budget={budget}: a second primary expansion was admitted while a "
                f"selected secondary seed was still missing:\n{out}"
            )
    assert checked, "fixture never produced a truncated response"


def _evidence_lines(out: str) -> frozenset[str]:
    return frozenset(line for line in out.splitlines() if line.startswith(("NODE ", "EDGE ")))


def test_admitted_evidence_is_monotonic_in_budget():
    """Evidence present at budget N must still be present at N+1.

    Allocation skipped a unit that did not fit and kept going, so a lower
    priority secondary seed could appear at one budget and vanish at a larger
    one once an earlier expansion became affordable. Raising a budget must never
    remove evidence.
    """
    graph = _multi_seed_expansion_graph()
    previous = None
    checked = 0
    for budget in range(1, 400):
        try:
            out = _query_graph_text(graph, "target", depth=1, token_budget=budget)
        except InsufficientBudgetError:
            continue
        current = _evidence_lines(out)
        if previous is not None:
            lost = previous - current
            assert not lost, f"budget={budget} dropped evidence present at a smaller budget: {lost}"
            checked += 1
        previous = current
    assert checked, "fixture produced no comparable budget pairs"


def test_admitted_evidence_is_a_prefix_of_the_priority_list():
    """No lower-priority unit may appear while a higher-priority unit is omitted.

    Priority here: the primary unit, then one further primary-seed expansion,
    then the secondary seed. A secondary seed present while the single permitted
    primary expansion is absent means allocation skipped rather than stopped.
    """
    graph = _multi_seed_expansion_graph()
    checked = 0
    for budget in range(1, 400):
        try:
            out = _query_graph_text(graph, "target", depth=1, token_budget=budget)
        except InsufficientBudgetError:
            continue
        if not any(line.startswith("...") for line in out.splitlines()):
            continue  # only truncated responses constrain the prefix
        lines = out.splitlines()
        secondary_present = any(line.startswith("NODE ") and "src=s2.py" in line for line in lines)
        primary_expansion_present = any(
            line.startswith("EDGE ") and "expand-" in line for line in lines
        )
        checked += 1
        if secondary_present:
            assert primary_expansion_present, (
                f"budget={budget}: secondary seed admitted while the higher-priority "
                f"primary expansion was omitted — admission is not a prefix:\n{out}"
            )
    assert checked, "fixture never produced a truncated response"


# --- graph-state cleanliness, asserted on canonical bytes ---


def test_query_leaves_canonical_graph_state_byte_identical():
    """Snapshot canonical state, run real queries, recapture, compare bytes.

    Naming the two known cache keys is implementation-coupled: a future cache
    stored under a different graph-metadata key would pass that check while
    still mutating graph payload. Recapturing the live graph and comparing
    encoded bytes catches any such key.
    """
    from graphify.graph_state import encode_graph_state_bytes, graph_state_from_graph

    graph = _seed_starved_graph()
    before = encode_graph_state_bytes(graph_state_from_graph(graph))

    for mode in ("bfs", "dfs"):
        for budget in (200, 400, 4000):
            _query_graph_text(graph, "target", mode=mode, depth=2, token_budget=budget)
    _score_nodes(graph, ["target"])
    _get_trigram_index(graph)
    _find_node(graph, "target")

    after = encode_graph_state_bytes(graph_state_from_graph(graph))
    assert after == before, "a query mutated canonical graph state"


def test_recapture_comparison_detects_any_graph_metadata_key():
    """Control: prove the byte comparison above is not vacuous.

    If this stops failing on an injected key, the regression proves nothing.
    """
    from graphify.graph_state import encode_graph_state_bytes, graph_state_from_graph

    graph = _seed_starved_graph()
    before = encode_graph_state_bytes(graph_state_from_graph(graph))
    graph.graph["_some_future_cache"] = {"derived": True}
    assert encode_graph_state_bytes(graph_state_from_graph(graph)) != before


# --- D0 integrated budget-boundary matrix ---


def _disconnected_seed_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("lonely", label="target", source_file="lonely.py", source_location="L1")
    for peer in ("x", "y", "z"):
        graph.add_node(peer, label=f"other-{peer}", source_file=f"{peer}.py", source_location="L1")
    graph.add_edge("x", "y", relation="calls", confidence="EXTRACTED", context="call")
    return graph


def _high_degree_seed_graph(fanout: int = 40) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("hub", label="target", source_file="hub.py", source_location="L1")
    for index in range(fanout):
        node = f"n{index}"
        graph.add_node(node, label=f"leaf-{index}", source_file=f"{node}.py", source_location="L1")
        graph.add_edge("hub", node, relation="calls", confidence="EXTRACTED", context="call")
    return graph


@pytest.mark.parametrize(
    "factory",
    [_seed_starved_graph, _disconnected_seed_graph, _high_degree_seed_graph],
    ids=["seed-starved", "disconnected-primary", "high-degree-primary"],
)
def test_budget_boundary_minimum_and_minimum_plus_one(factory):
    """minimum-1 refuses; minimum and minimum+1 succeed, fit, and keep the seed."""
    graph = factory()

    def render(budget):
        return _query_graph_text(graph, "target", depth=2, token_budget=budget)

    minimum = _minimum_budget(render)
    if minimum > 1:
        with pytest.raises(InsufficientBudgetError) as excinfo:
            render(minimum - 1)
        assert excinfo.value.required_minimum == minimum

    for budget in (minimum, minimum + 1):
        out = render(budget)
        assert len(out) <= budget * 3, (budget, len(out))
        _, _, body = out.partition("\n\n")
        assert "NODE target [" in body, (budget, out)


@pytest.mark.parametrize("filter_kind", ["explicit", "inferred"])
def test_budget_boundary_with_context_filtering(filter_kind):
    """Context filtering must not break allocation or lose the queried seed."""
    graph = _seed_starved_graph()
    question = "who calls target" if filter_kind == "inferred" else "target"
    filters = None if filter_kind == "inferred" else ["call"]

    def render(budget):
        return _query_graph_text(
            graph, question, depth=2, token_budget=budget, context_filters=filters
        )

    minimum = _minimum_budget(render)
    for budget in (minimum, minimum + 1, minimum + 40):
        out = render(budget)
        assert len(out) <= budget * 3, (filter_kind, budget, len(out))
        _, _, body = out.partition("\n\n")
        assert "NODE target [" in body, (filter_kind, budget, out)


def test_relationship_between_two_selected_seeds_is_not_lost():
    """Both endpoints selected as seeds must still yield their relationship.

    Every seed starts the traversal already visited, so the frontier loop never
    recorded an edge between two seeds and a real relationship disappeared from
    the evidence exactly when the query matched both of its endpoints.
    """
    graph = nx.Graph()
    graph.add_node("a", label="target", source_file="a.py", source_location="L1")
    graph.add_node("b", label="target", source_file="b.py", source_location="L1")
    graph.add_edge("a", "b", relation="calls", confidence="EXTRACTED")

    def render(budget):
        return _query_graph_text(graph, "target", depth=1, token_budget=budget)

    out = render(_minimum_budget(render))
    assert any(line.startswith("EDGE ") for line in out.splitlines()), out
    assert "calls" in out, out
