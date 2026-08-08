import json
import os
import subprocess
import sys
from typing import Any
import networkx as nx
import pytest
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import (
    _partition,
    cluster,
    cohesion_score,
    label_communities_by_hub,
    remap_communities_to_previous,
    score_all,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_graph():
    return build_from_json(json.loads((FIXTURES / "extraction.json").read_text(encoding="utf-8")))


def test_cluster_returns_dict():
    G = make_graph()
    communities = cluster(G)
    assert isinstance(communities, dict)


def test_cluster_covers_all_nodes():
    G = make_graph()
    communities = cluster(G)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)


def test_cohesion_score_complete_graph():
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes})
    score = cohesion_score(G, list(G.nodes))
    assert score == 1.0


def test_cohesion_score_single_node():
    G = nx.Graph()
    G.add_node("a")
    score = cohesion_score(G, ["a"])
    assert score == 1.0


def test_cohesion_score_disconnected():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == 0.0


def test_cohesion_score_range():
    G = make_graph()
    communities = cluster(G)
    for cid, nodes in communities.items():
        score = cohesion_score(G, nodes)
        assert 0.0 <= score <= 1.0


def test_cohesion_directed_reciprocal_edges_use_undirected_topology():
    graph = nx.DiGraph()
    graph.add_edges_from(
        (source, target) for source in "abc" for target in "abc" if source != target
    )

    assert cohesion_score(graph, list("abc")) == 1.0


def test_hub_label_tie_break_is_type_aware_and_insertion_independent():
    first = nx.Graph()
    first.add_nodes_from([1, "1"])
    first.nodes[1]["label"] = "integer"
    first.nodes["1"]["label"] = "string"
    second = nx.Graph()
    second.add_nodes_from(["1", 1])
    second.nodes[1]["label"] = "integer"
    second.nodes["1"]["label"] = "string"

    assert label_communities_by_hub(first, {0: [1, "1"]}) == label_communities_by_hub(
        second, {0: ["1", 1]}
    )


def test_score_all_keys_match_communities():
    G = make_graph()
    communities = cluster(G)
    scores = score_all(G, communities)
    assert set(scores.keys()) == set(communities.keys())


def test_cluster_does_not_write_to_stdout(capsys):
    """Clustering should not emit ANSI escape codes or other output.

    The native Leiden call must not leak progress or diagnostic output into
    PowerShell's scroll buffer (issue #19).
    """
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    assert captured.out == "", f"cluster() wrote to stdout: {captured.out!r}"


def test_cluster_does_not_write_to_stderr(capsys):
    """Same as above but for stderr — ANSI codes can go to either stream."""
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    # Allow logging output (starts with [graphify]) but no raw ANSI codes
    for line in captured.err.splitlines():
        assert "\x1b" not in line, f"cluster() wrote ANSI to stderr: {line!r}"


def test_remap_communities_to_previous_reuses_old_ids():
    communities = {
        10: ["a", "b", "c"],
        11: ["d", "e"],
    }
    previous = {"a": 5, "b": 5, "c": 5, "d": 1, "e": 1}
    remapped = remap_communities_to_previous(communities, previous)
    assert set(remapped.keys()) == {1, 5}
    assert remapped[5] == ["a", "b", "c"]
    assert remapped[1] == ["d", "e"]


def test_remap_communities_to_previous_assigns_deterministic_new_ids():
    communities = {
        7: ["x", "y", "z"],
        8: ["m"],
    }
    previous = {"a": 3}
    remapped = remap_communities_to_previous(communities, previous)
    assert list(remapped.keys()) == [0, 1]
    assert remapped[0] == ["x", "y", "z"]
    assert remapped[1] == ["m"]


def test_remap_communities_uses_global_maximum_overlap():
    """One locally largest overlap must not block a better total assignment."""
    communities = {
        10: ["a", "b", "c", "d", "h", "i", "j"],
        11: ["e", "f", "g"],
    }
    previous = {
        "a": 0,
        "b": 0,
        "c": 0,
        "d": 0,
        "e": 0,
        "f": 0,
        "g": 0,
        "h": 1,
        "i": 1,
        "j": 1,
    }

    remapped = remap_communities_to_previous(communities, previous)

    assert remapped[0] == ["e", "f", "g"]
    assert remapped[1] == ["a", "b", "c", "d", "h", "i", "j"]


def test_remap_communities_equal_overlap_tie_is_insertion_order_independent():
    previous = {"a": 4, "b": 4, "c": 7, "d": 7}
    forward = {20: ["a", "c"], 21: ["b", "d"]}
    reversed_input = {21: ["d", "b"], 20: ["c", "a"]}

    expected = remap_communities_to_previous(forward, previous)
    assert (
        remap_communities_to_previous(reversed_input, dict(reversed(list(previous.items()))))
        == expected
    )


def _capture_native_edges(
    monkeypatch: pytest.MonkeyPatch, graph: nx.Graph
) -> list[tuple[str, str, float]]:
    import graspologic_native

    captured: list[tuple[str, str, float]] = []

    def fake_leiden(*, edges: list[tuple[str, str, float]], **_kwargs: Any):
        captured.extend(edges)
        nodes = {node for edge in edges for node in edge[:2]}
        return 0.0, {node: 0 for node in nodes}

    monkeypatch.setattr(graspologic_native, "leiden", fake_leiden)
    _partition(graph)
    assert captured, "fixture must exercise native Leiden with at least one edge"
    return captured


def test_partition_canonicalizes_undirected_orientation_before_sort(monkeypatch):
    first = nx.Graph()
    first.add_nodes_from(["b", "a", "z"])
    first.add_edge("a", "z", weight=1.0)
    first.add_edge("b", "a", weight=1.0)

    second = nx.Graph()
    second.add_nodes_from(["z", "a", "b"])
    second.add_edge("z", "a", weight=1.0)
    second.add_edge("a", "b", weight=1.0)

    first_edges = _capture_native_edges(monkeypatch, first)
    second_edges = _capture_native_edges(monkeypatch, second)

    assert first_edges == second_edges


def test_partition_native_supports_mixed_node_id_types(monkeypatch):
    graph = nx.Graph()
    graph.add_edges_from([(1, "1"), ("1", 2)])

    partition = _partition(graph)

    assert set(partition) == {1, "1", 2}


def test_partition_louvain_supports_mixed_node_id_types(monkeypatch):
    monkeypatch.setitem(sys.modules, "graspologic_native", None)
    graph = nx.Graph()
    graph.add_edges_from([(1, "1"), ("1", 2)])

    partition = _partition(graph)

    assert set(partition) == {1, "1", 2}


def test_cluster_edgeless_mixed_node_id_types():
    graph = nx.Graph()
    graph.add_nodes_from([1, "1", 2])

    communities = cluster(graph)

    assert {node for members in communities.values() for node in members} == {1, "1", 2}


@pytest.mark.parametrize("backend", ["native", "louvain"])
@pytest.mark.parametrize(
    "case",
    ["connected", "disconnected", "edgeless", "parallel", "malformed", "all_zero", "mixed"],
)
def test_cluster_is_stable_across_backend_and_graph_shape(monkeypatch, backend, case):
    native_calls = 0
    if backend == "native":
        import graspologic_native

        real_leiden = graspologic_native.leiden

        def counted_leiden(*args, **kwargs):
            nonlocal native_calls
            native_calls += 1
            return real_leiden(*args, **kwargs)

        monkeypatch.setattr(graspologic_native, "leiden", counted_leiden)
    if backend == "louvain":
        monkeypatch.setitem(sys.modules, "graspologic_native", None)

    graph: nx.Graph
    if case == "connected":
        graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    elif case == "disconnected":
        graph = nx.Graph([("a", "b"), ("c", "d")])
        graph.add_node("isolated")
    elif case == "edgeless":
        graph = nx.Graph()
        graph.add_nodes_from(["a", 1, "1"])
    elif case == "parallel":
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(["a", "b", "c"])
        for index in range(8):
            graph.add_edge("a", "b", key=f"ab-{index}", confidence="EXTRACTED")
        for index in range(5):
            graph.add_edge("b", "c", key=f"bc-{index}", confidence="INFERRED")
    elif case == "malformed":
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=None)
        graph.add_edge("b", "c", weight=float("nan"))
        graph.add_edge("c", "d", weight=float("inf"))
        graph.add_edge("d", "a", weight=-1.0)
    elif case == "all_zero":
        graph = nx.Graph()
        graph.add_weighted_edges_from([("a", "b", 0.0), ("b", "c", 0.0)])
    else:
        graph = nx.Graph([(1, "1"), ("1", 2), (2, "2")])

    expected = cluster(graph)
    for _ in range(3):
        assert cluster(graph) == expected
    assert {node for members in expected.values() for node in members} == set(graph.nodes)
    if backend == "native":
        expected_calls = 0 if graph.number_of_edges() == 0 else 4
        assert native_calls == expected_calls


@pytest.mark.parametrize("backend", ["native", "louvain"])
def test_cluster_is_stable_across_processes_and_hash_seeds(backend):
    fallback = "sys.modules['graspologic_native'] = None" if backend == "louvain" else ""
    code = f"""
import json
import networkx as nx
import sys
from graphify.cluster import cluster
{fallback}
graph = nx.Graph()
for source, target in {{('alpha', 'beta'), ('alpha', 'gamma'), ('beta', 'delta'),
                       ('gamma', 'delta'), ('delta', 'epsilon'), ('zeta', 'eta')}}:
    graph.add_edge(source, target, weight=1.0)
print(json.dumps(cluster(graph), sort_keys=True))
"""
    outputs = []
    for seed in ("1", "17", "31337"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert len(set(outputs)) == 1


def test_remap_communities_split_keeps_only_best_overlap_on_old_id():
    communities = {10: ["a", "b", "c"], 11: ["d", "e"]}
    previous = {node: 4 for node in "abcde"}

    remapped = remap_communities_to_previous(communities, previous)

    assert remapped[4] == ["a", "b", "c"]
    assert remapped[0] == ["d", "e"]


def test_remap_communities_merge_uses_largest_prior_overlap():
    communities = {10: ["a", "b", "c", "d", "e"]}
    previous = {"a": 3, "b": 3, "c": 3, "d": 8, "e": 8}

    assert remap_communities_to_previous(communities, previous) == {3: ["a", "b", "c", "d", "e"]}


def test_remap_communities_empty_and_mixed_ids():
    assert remap_communities_to_previous({}, {"old": 1}) == {}
    communities = {9: [1, "1"], 10: [2, "2"]}
    previous = {1: 6, "1": 6, 2: 7, "2": 7}

    assert remap_communities_to_previous(communities, previous) == {
        6: [1, "1"],
        7: [2, "2"],
    }


# --- MultiDiGraph safety tests (PR 4B) ---


def _make_multigraph_triangle():
    """MultiDiGraph with nodes {a, b, c}: 5 parallel edges a->b, 3 parallel edges b->c."""
    G = nx.MultiDiGraph()
    G.add_nodes_from(["a", "b", "c"])
    for i in range(5):
        G.add_edge("a", "b", key=f"ab-{i}", relation=f"rel-{i}")
    for i in range(3):
        G.add_edge("b", "c", key=f"bc-{i}", relation=f"rel-{i}")
    return G


def test_cohesion_multigraph_stays_bounded():
    """Cohesion must be <= 1.0 even when parallel edges outnumber unique pairs."""
    G = _make_multigraph_triangle()
    # 3 nodes, 8 total edge records, but only 2 unique pairs -> must not exceed 1.0
    score = cohesion_score(G, ["a", "b", "c"])
    assert score <= 1.0, f"cohesion {score} exceeds 1.0 on multigraph"
    assert score >= 0.0


def test_cohesion_multigraph_equals_simple_graph_cohesion():
    """Cohesion on a multigraph should equal cohesion on the equivalent simple graph."""
    # Build a MultiDiGraph: a-b, b-c, a-c each with 3 parallel edges
    MG = nx.MultiDiGraph()
    MG.add_nodes_from(["a", "b", "c"])
    for pair in [("a", "b"), ("b", "c"), ("a", "c")]:
        for i in range(3):
            MG.add_edge(pair[0], pair[1], key=f"{pair[0]}{pair[1]}-{i}")

    # Build equivalent simple graph: a-b, b-c, a-c (1 edge each)
    SG = nx.Graph()
    SG.add_nodes_from(["a", "b", "c"])
    SG.add_edge("a", "b")
    SG.add_edge("b", "c")
    SG.add_edge("a", "c")

    multi_score = cohesion_score(MG, ["a", "b", "c"])
    simple_score = cohesion_score(SG, ["a", "b", "c"])
    assert multi_score == simple_score, f"multi={multi_score} != simple={simple_score}"


def test_cluster_multigraph_produces_valid_communities():
    """cluster() on a MultiDiGraph with clear community structure should detect communities."""
    G = nx.MultiDiGraph()
    # Two triangles connected by a weak bridge, with parallel edges and
    # confidence data so projected weights are non-zero.
    for pair in [("a", "b"), ("b", "c"), ("a", "c")]:
        for k in range(3):
            G.add_edge(pair[0], pair[1], key=f"{pair[0]}{pair[1]}-{k}", confidence="EXTRACTED")
    for pair in [("d", "e"), ("e", "f"), ("d", "f")]:
        for k in range(3):
            G.add_edge(pair[0], pair[1], key=f"{pair[0]}{pair[1]}-{k}", confidence="EXTRACTED")
    G.add_edge("c", "d", key="bridge", confidence="AMBIGUOUS")

    communities = cluster(G)
    assert isinstance(communities, dict)
    assert len(communities) > 0
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes), "Not all nodes assigned to communities"


def test_cluster_multigraph_does_not_crash():
    """Smoke test: cluster() on a MultiDiGraph with parallel edges must not raise."""
    G = nx.MultiDiGraph()
    nodes = ["a", "b", "c", "d", "e"]
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, min(i + 3, len(nodes))):
            for k in range(4):
                G.add_edge(
                    nodes[i], nodes[j], key=f"{nodes[i]}-{nodes[j]}-{k}", confidence="EXTRACTED"
                )
    # Must not raise
    communities = cluster(G)
    assert isinstance(communities, dict)


def test_cluster_multigraph_without_confidence_does_not_panic():
    """A multigraph whose edges carry NO confidence must still cluster.

    project_for_community(weight_mode="confidence") scores an unrecognized/absent
    `confidence` as 0.0, so such a graph projects to all-zero edge weights and
    native Leiden aborts with a PanicException — a BaseException the
    ImportError/SyntaxError fallback in _partition cannot catch, so the whole
    process dies. Only the multigraph path reaches this: the simple path never
    sets `weight`. Sibling test above passes confidence on every edge, which is
    exactly why it never caught this.
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(["a", "b", "c", "d"])
    G.add_edge("a", "b", key="k1", relation="calls")
    G.add_edge("a", "b", key="k2", relation="imports")
    G.add_edge("a", "c", relation="calls")
    G.add_edge("c", "d", relation="calls")

    # BaseException, not Exception: PanicException does not subclass Exception.
    communities = cluster(G)
    assert isinstance(communities, dict)
    assert {n for members in communities.values() for n in members} == set(G.nodes)


def test_effective_weight_matches_partitioner_default():
    """Unusable weights count as 1.0 (the established default), bools excluded."""
    from graphify.cluster import _effective_weight

    assert _effective_weight({}) == 1.0
    assert _effective_weight({"weight": None}) == 1.0
    assert _effective_weight({"weight": "0.5"}) == 1.0
    assert _effective_weight({"weight": True}) == 1.0
    assert _effective_weight({"weight": float("nan")}) == 1.0
    assert _effective_weight({"weight": float("inf")}) == 1.0
    assert _effective_weight({"weight": -3.0}) == 1.0
    assert _effective_weight({"weight": 0}) == 0.0
    assert _effective_weight({"weight": 0.25}) == 0.25


def test_is_usable_weight_accepts_only_finite_nonnegative_reals():
    from graphify.cluster import _is_usable_weight

    assert _is_usable_weight(0) and _is_usable_weight(0.0) and _is_usable_weight(2.5)
    assert not _is_usable_weight(None)
    assert not _is_usable_weight("1.0")
    assert not _is_usable_weight(True)
    assert not _is_usable_weight(-0.1)
    assert not _is_usable_weight(float("nan"))
    assert not _is_usable_weight(float("inf"))
    assert not _is_usable_weight(float("-inf"))


@pytest.mark.parametrize(
    "weight_expr",
    ["None", "'abc'", "float('nan')", "float('inf')", "float('-inf')", "-5.0"],
)
def test_cluster_survives_malformed_edge_weights(weight_expr):
    """A corrupt `weight` in graph.json must not take the process down.

    The native engine receives weights in Rust: None/str raise during
    float() coercion, NaN and negatives trigger an unwrap-on-None PanicException
    (a BaseException — it kills the process, the Louvain fallback cannot catch
    it), and +/-inf raise ParameterRangeError. The SIMPLE path is the live
    vector: cluster() only calls to_undirected(), so stored attrs pass through
    untouched. Run in a subprocess because a PanicException aborts the
    interpreter and would take the whole test session with it.
    """
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import networkx as nx\n"
        "from graphify.cluster import cluster\n"
        "G = nx.DiGraph()\n"
        "G.add_nodes_from('abcd')\n"
        "w = %s\n"
        "G.add_edge('a','b',weight=w)\n"
        "G.add_edge('a','c',weight=w)\n"
        "G.add_edge('c','d',weight=w)\n"
        "c = cluster(G)\n"
        "assert isinstance(c, dict) and c\n"
        "print('OK')\n"
    ) % (str(Path(__file__).resolve().parent.parent), weight_expr)
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, encoding="utf-8"
    )
    assert r.returncode == 0, f"weight={weight_expr} crashed:\n{r.stdout}\n{r.stderr}"
    assert "OK" in r.stdout


def test_stateful_class_detection_is_stricter_than_read_only_legacy_loading():
    """Stateful writes require complete declarations; read-only legacy loads may rescue."""
    from graphify.graph_state import GraphStateError
    from graphify.graph_loader import load_graph
    from graphify.watch import _existing_graph_type

    stale_profile = {"graphify_profile": {"graph_type": "multidigraph"}}
    base_nodes = [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}]
    base_links = [{"source": "a", "target": "b", "relation": "calls"}]

    contradictory = {
        "directed": True,
        "multigraph": False,
        "graph": stale_profile,
        "nodes": base_nodes,
        "links": base_links,
    }
    with pytest.raises(GraphStateError, match="profile conflicts with class flags"):
        _existing_graph_type(contradictory)
    with pytest.raises(GraphStateError, match="profile conflicts with class flags"):
        load_graph(contradictory)

    # A read-only inspection can rescue an unambiguous legacy class declaration.
    # A stateful rebuild must refuse it rather than overwrite the payload with a
    # class inferred from incomplete fields.
    rescued = {"directed": True, "graph": stale_profile, "nodes": base_nodes, "links": base_links}
    with pytest.raises(GraphStateError, match="directed and multigraph flags are required"):
        _existing_graph_type(rescued)
    assert isinstance(load_graph(rescued), nx.MultiDiGraph)

    explicit_multi = {
        "directed": True,
        "multigraph": True,
        "graph": stale_profile,
        "nodes": base_nodes,
        "links": base_links,
    }
    assert _existing_graph_type(explicit_multi) == "multidigraph"
    assert isinstance(load_graph(explicit_multi), nx.MultiDiGraph)


def test_cluster_only_preserves_multigraph_class(tmp_path):
    """`graphify cluster-only` must not downgrade a multidigraph graph.json.

    It loaded with build_from_json(directed=...) but never passed multigraph, so a
    multidigraph file rebuilt as a simple DiGraph — collapsing every parallel edge
    — and the to_json write-back persisted that collapse AND re-stamped
    graphify_profile as "digraph", so every later update inherited the downgrade.
    """
    import subprocess

    out = tmp_path / "graphify-out"
    out.mkdir(parents=True)
    (out / "graph.json").write_text(
        json.dumps(
            {
                "directed": True,
                "multigraph": True,
                "graph": {"graphify_profile": {"graph_type": "multidigraph"}},
                "nodes": [
                    {"id": n, "label": n, "type": "function", "source_file": f"{n}.py"}
                    for n in ("a", "b", "c", "d")
                ],
                "links": [
                    {
                        "source": "a",
                        "target": "b",
                        "relation": "calls",
                        "key": "k1",
                        "confidence": "EXTRACTED",
                        "source_file": "a.py",
                    },
                    {
                        "source": "a",
                        "target": "b",
                        "relation": "imports",
                        "key": "k2",
                        "confidence": "EXTRACTED",
                        "source_file": "a.py",
                    },
                    {
                        "source": "a",
                        "target": "c",
                        "relation": "calls",
                        "key": "k3",
                        "confidence": "EXTRACTED",
                        "source_file": "a.py",
                    },
                    {
                        "source": "c",
                        "target": "d",
                        "relation": "calls",
                        "key": "k4",
                        "confidence": "EXTRACTED",
                        "source_file": "c.py",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    r = subprocess.run(
        [sys.executable, "-m", "graphify", "cluster-only", str(tmp_path), "--no-viz", "--no-label"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        encoding="utf-8",
    )
    assert r.returncode == 0, f"cluster-only failed:\n{r.stdout}\n{r.stderr}"

    after = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    links = after.get("links", after.get("edges", []))
    assert after.get("multigraph") is True, "multigraph flag lost — graph downgraded"
    assert after.get("directed") is True
    profile = (after.get("graph") or {}).get("graphify_profile") or {}
    assert profile.get("graph_type") == "multidigraph", "profile marker downgraded"
    parallel = [e for e in links if e.get("source") == "a" and e.get("target") == "b"]
    assert len(parallel) == 2, f"parallel a->b edges collapsed: {parallel}"
    assert len(links) == 4


def test_cluster_only_preserves_graph_metadata_and_is_idempotent(tmp_path):
    """Graph-level metadata must survive cluster-only, and survive REPEATED runs.

    Reading graph.json back through build_from_json (an extraction-dict API)
    dropped every graph attribute the loader preserves: custom top-level keys and
    any graphify_profile field beyond graph_type. Three runs pin that the class,
    parallel edges, hyperedges and metadata are all fixed points, not merely
    correct on the first pass.
    """
    out = tmp_path / "graphify-out"
    out.mkdir(parents=True)
    payload = {
        "directed": True,
        "multigraph": True,
        "graph": {
            "graphify_profile": {
                "graph_type": "multidigraph",
                "schema_version": 7,
                "producer": "vampyre-test",
            },
            "custom_meta": {"keep": "me"},
            "corpus_name": "my-corpus",
        },
        "hyperedges": [{"id": "he1", "label": "trio", "nodes": ["a", "b", "c"]}],
        "nodes": [
            {"id": n, "label": n, "type": "function", "source_file": f"{n}.py"}
            for n in ("a", "b", "c", "d")
        ],
        "links": [
            {
                "source": "a",
                "target": "b",
                "relation": "calls",
                "key": "k1",
                "confidence": "EXTRACTED",
                "source_file": "a.py",
            },
            {
                "source": "a",
                "target": "b",
                "relation": "imports",
                "key": "k2",
                "confidence": "EXTRACTED",
                "source_file": "a.py",
            },
            {
                "source": "c",
                "target": "d",
                "relation": "calls",
                "key": "k4",
                "confidence": "EXTRACTED",
                "source_file": "c.py",
            },
        ],
    }
    (out / "graph.json").write_text(json.dumps(payload), encoding="utf-8")

    for run in range(1, 4):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphify",
                "cluster-only",
                str(tmp_path),
                "--no-viz",
                "--no-label",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
        )
        assert r.returncode == 0, f"run {run} failed:\n{r.stdout}\n{r.stderr}"
        after = json.loads((out / "graph.json").read_text(encoding="utf-8"))
        graph_meta = after.get("graph") or {}

        assert after.get("multigraph") is True, f"run {run}: class downgraded"
        assert graph_meta.get("corpus_name") == "my-corpus", f"run {run}: custom key lost"
        assert graph_meta.get("custom_meta") == {"keep": "me"}, f"run {run}: custom dict lost"
        prof = graph_meta.get("graphify_profile") or {}
        assert prof.get("graph_type") == "multidigraph", f"run {run}: profile downgraded"
        assert prof.get("schema_version") == 7, f"run {run}: profile field lost"
        assert prof.get("producer") == "vampyre-test", f"run {run}: profile field lost"

        links = after.get("links", after.get("edges", []))
        parallel = [e for e in links if e.get("source") == "a" and e.get("target") == "b"]
        assert len(parallel) == 2, f"run {run}: parallel edges collapsed"
        assert {e.get("key") for e in parallel} == {"k1", "k2"}, f"run {run}: edge keys lost"
        assert [h.get("id") for h in after.get("hyperedges", [])] == ["he1"], (
            f"run {run}: hyperedges dropped"
        )
