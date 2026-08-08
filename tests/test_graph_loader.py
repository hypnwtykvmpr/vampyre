"""Tests for graphify.graph_loader — schema-aware graph loading.

Seven required PR 2 scenarios from the Wave 3 handoff guardrails.
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import cast
from unittest.mock import patch

import networkx as nx
import pytest

from graphify.graph_loader import GRAPHIFY_PROFILE_KEY, load_graph
from graphify.graph_state import DecodeMode, GraphStateError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NODES = [
    {"id": "a", "label": "A", "file_type": "code", "source_file": "a.py"},
    {"id": "b", "label": "B", "file_type": "code", "source_file": "b.py"},
]

_SIMPLE_EDGE = {
    "source": "a",
    "target": "b",
    "relation": "calls",
    "confidence": "EXTRACTED",
    "confidence_score": 1.0,
    "source_file": "a.py",
    "weight": 1.0,
    "_origin": "ast",
}

_KEYED_EDGE = {**_SIMPLE_EDGE, "key": "calls:a.py:L1"}

_KEYED_EDGE_2 = {
    "source": "a",
    "target": "b",
    "relation": "imports",
    "confidence": "EXTRACTED",
    "confidence_score": 1.0,
    "source_file": "a.py",
    "key": "imports:a.py:L5",
    "weight": 1.0,
    "_origin": "ast",
}


def _current_multidigraph_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "directed": True,
        "multigraph": True,
        "graph": {
            "graphify_profile": {"graph_type": "multidigraph", "extension": "kept"},
            "corpus_name": "reader-contract",
        },
        "nodes": _NODES,
        "links": [_KEYED_EDGE, _KEYED_EDGE_2],
        "hyperedges": [{"id": "flow", "nodes": ["a", "b"], "label": "Flow"}],
        "custom_top": {"owner": "fixture"},
        "graphify_state_diagnostics": [],
    }


def test_product_readers_agree_on_canonical_multidigraph_state(tmp_path):
    from graphify.affected import load_graph as load_affected_graph
    from graphify.callflow_html import load_graph as load_callflow_graph
    from graphify.diagnostics import _read_json_file
    from graphify.graph_loader import load_graph_file, load_graph_state_file
    from graphify.prs import _load_graph_json
    from graphify.serve import _load_graph as load_served_graph

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(_current_multidigraph_payload()), encoding="utf-8")

    state = load_graph_state_file(path, mode=DecodeMode.STRICT_CURRENT)
    graphs = [load_graph_file(path), load_affected_graph(path), load_served_graph(str(path))]
    for graph in graphs:
        assert type(graph) is nx.MultiDiGraph
        assert graph.number_of_edges() == 2
        assert graph.graph[GRAPHIFY_PROFILE_KEY] == state.graph_metadata[GRAPHIFY_PROFILE_KEY]
        assert graph.graph["corpus_name"] == "reader-contract"
        assert graph.graph["hyperedges"] == list(state.hyperedges)
        assert set(graph["a"]["b"]) == {"calls:a.py:L1", "imports:a.py:L5"}

    nodes, edges, hyperedges, metadata = load_callflow_graph(path)
    assert {node["id"] for node in nodes} == {"a", "b"}
    assert {edge["key"] for edge in edges} == {"calls:a.py:L1", "imports:a.py:L5"}
    assert hyperedges == list(state.hyperedges)
    assert metadata[GRAPHIFY_PROFILE_KEY] == state.graph_metadata[GRAPHIFY_PROFILE_KEY]
    assert metadata["corpus_name"] == "reader-contract"

    for payload in (_load_graph_json(path), _read_json_file(path)):
        assert payload is not None
        assert payload["multigraph"] is True
        assert payload["directed"] is True
        assert payload["graph"][GRAPHIFY_PROFILE_KEY]["extension"] == "kept"
        assert payload["hyperedges"] == list(state.hyperedges)
        assert {edge["key"] for edge in payload["links"]} == {
            "calls:a.py:L1",
            "imports:a.py:L5",
        }


def test_product_readers_refuse_conflicting_current_edge_lists(tmp_path, capsys):
    from graphify.affected import load_graph as load_affected_graph
    from graphify.callflow_html import load_graph as load_callflow_graph
    from graphify.diagnostics import _read_json_file
    from graphify.graph_loader import load_graph_file
    from graphify.prs import _load_graph_json
    from graphify.serve import _load_graph as load_served_graph

    payload = _current_multidigraph_payload()
    payload["edges"] = [{**_KEYED_EDGE, "relation": "conflicts"}]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GraphStateError, match="edges and links contain conflicting records"):
        load_graph_file(path)
    with pytest.raises(RuntimeError, match="conflicting records"):
        load_affected_graph(path)
    with pytest.raises(SystemExit):
        load_served_graph(str(path))
    assert "conflicting records" in capsys.readouterr().err
    with pytest.raises(GraphStateError, match="conflicting records"):
        load_callflow_graph(path)
    with pytest.raises(GraphStateError, match="conflicting records"):
        _read_json_file(path)
    assert _load_graph_json(path) is None


def test_graph_state_reader_policy_allows_only_classified_adapters():
    project_root = Path(__file__).resolve().parents[1]
    allowed_node_link = {"callflow_html.py", "multigraph_compat.py"}
    offenders: list[str] = []
    for source_path in sorted((project_root / "graphify").glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "node_link_graph" and source_path.name not in allowed_node_link:
                offenders.append(f"{source_path.name}:{node.lineno}:node_link_graph")
            if node.func.attr != "loads" or not node.args:
                continue
            argument = node.args[0]
            if not isinstance(argument, ast.Call) or not isinstance(argument.func, ast.Attribute):
                continue
            if argument.func.attr != "read_text":
                continue
            receiver = argument.func.value
            receiver_names = {
                child.id for child in ast.walk(receiver) if isinstance(child, ast.Name)
            }
            graph_names = {
                "graph_path",
                "graph_json",
                "existing_graph",
                "existing_graph_path",
                "_GLOBAL_GRAPH",
            }
            if receiver_names & graph_names and source_path.name not in {
                "graph_state.py",
                "callflow_html.py",
                "diagnostics.py",
            }:
                offenders.append(f"{source_path.name}:{node.lineno}:raw graph JSON")
    assert offenders == []


def test_load_graph_state_file_preserves_full_state_and_legacy_diagnostics(tmp_path):
    from graphify.graph_loader import load_graph_state_file

    path = tmp_path / "graph.json"
    payload = {
        "directed": True,
        "multigraph": True,
        "graph": {"graphify_profile": {"graph_type": "multidigraph", "extra": "kept"}},
        "nodes": _NODES,
        "links": [_KEYED_EDGE],
        "hyperedges": [{"nodes": ["a", "b"], "label": "Flow"}],
        "custom_top": "kept",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_graph_state_file(path, mode=DecodeMode.MIGRATE_LEGACY)

    assert type(state.graph) is nx.MultiDiGraph
    profile = state.graph_metadata["graphify_profile"]
    assert isinstance(profile, dict)
    assert profile["extra"] == "kept"
    assert state.top_level_metadata["custom_top"] == "kept"
    hyperedge_id = state.hyperedges[0]["id"]
    assert isinstance(hyperedge_id, str)
    assert hyperedge_id.startswith("hyperedge:v1:")
    assert state.diagnostics


def _simple_links() -> dict:
    """Legacy simple JSON using 'links' key."""
    return {"nodes": _NODES, "links": [_SIMPLE_EDGE]}


def _simple_edges() -> dict:
    """Modern simple JSON using 'edges' key."""
    return {"nodes": _NODES, "edges": [_SIMPLE_EDGE]}


def _multigraph_data() -> dict:
    """Valid multigraph node-link JSON with two keyed parallel edges."""
    return {
        "multigraph": True,
        "nodes": _NODES,
        "links": [_KEYED_EDGE, _KEYED_EDGE_2],
    }


def _multigraph_missing_keys() -> dict:
    """Multigraph JSON where edges lack 'key' fields."""
    edge_no_key = {k: v for k, v in _SIMPLE_EDGE.items() if k != "key"}
    return {"multigraph": True, "nodes": _NODES, "links": [edge_no_key]}


# ---------------------------------------------------------------------------
# Scenario 1: legacy 'links' loads as nx.Graph
# ---------------------------------------------------------------------------


def test_load_graph_rejects_non_object_payload():
    with pytest.raises(TypeError, match="serialized graph data must be a JSON object"):
        load_graph([])


def test_load_graph_rejects_non_list_nodes():
    data = {**_simple_links(), "nodes": 123}

    with pytest.raises(GraphStateError, match="nodes must be a list"):
        load_graph(data)


def test_legacy_links_loads_as_simple_graph():
    G = load_graph(_simple_links())
    assert type(G) is nx.Graph
    assert not G.is_multigraph()
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


# ---------------------------------------------------------------------------
# Scenario 2: modern 'edges' loads as nx.Graph
# ---------------------------------------------------------------------------


def test_modern_edges_loads_as_simple_graph():
    G = load_graph(_simple_edges())
    assert type(G) is nx.Graph
    assert not G.is_multigraph()
    assert G.number_of_edges() == 1


# ---------------------------------------------------------------------------
# Scenario 3: valid multigraph JSON with keyed parallel edges → nx.MultiDiGraph
# ---------------------------------------------------------------------------


def test_valid_multigraph_loads_as_multidigraph():
    G = load_graph(_multigraph_data())
    assert type(G) is nx.MultiDiGraph
    assert G.is_multigraph()
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 2  # both parallel edges preserved


# ---------------------------------------------------------------------------
# Scenario 4: malformed multigraph (missing keys) repairs explicitly, not silently
# ---------------------------------------------------------------------------


def test_malformed_multigraph_missing_keys_repairs_explicitly(capsys):
    G = load_graph(_multigraph_missing_keys())
    # Must produce a MultiDiGraph (not silently fall back to simple)
    assert type(G) is nx.MultiDiGraph
    assert G.number_of_edges() == 1
    # Must warn to stderr
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "key" in captured.err.lower()


# ---------------------------------------------------------------------------
# Scenario 5: edge 'key' is stripped from attrs — not stored as an edge attribute
# ---------------------------------------------------------------------------


def test_schema_key_stripped_from_edge_attrs():
    G = load_graph(_multigraph_data())
    assert isinstance(G, nx.MultiDiGraph)
    for u, v, k, data in G.edges(keys=True, data=True):
        assert "key" not in data, (
            f"Edge ({u},{v},key={k!r}) must not store 'key' inside its attrs dict"
        )


# ---------------------------------------------------------------------------
# Scenario 6: G.graph["graphify_profile"] is present after load
# ---------------------------------------------------------------------------


def test_graph_profile_metadata_round_trips():
    G = load_graph(_simple_links())
    assert GRAPHIFY_PROFILE_KEY in G.graph
    profile = G.graph[GRAPHIFY_PROFILE_KEY]
    assert isinstance(profile, dict)
    assert "graph_type" in profile


def test_graph_profile_type_for_multidigraph():
    G = load_graph(_multigraph_data())
    assert G.graph[GRAPHIFY_PROFILE_KEY]["graph_type"] == "multidigraph"


def test_graph_profile_type_for_simple():
    G = load_graph(_simple_links())
    assert G.graph[GRAPHIFY_PROFILE_KEY]["graph_type"] == "simple"


# ---------------------------------------------------------------------------
# Scenario 7: capability probe failure raises clearly; simple loading unaffected
# ---------------------------------------------------------------------------


def test_capability_probe_failure_raises_clear_error():
    with patch(
        "graphify.graph_loader.require_multigraph_capabilities",
        side_effect=RuntimeError("MultiDiGraph not supported: simulated failure"),
    ):
        with pytest.raises(RuntimeError, match="MultiDiGraph not supported"):
            load_graph(_multigraph_data(), require_capabilities=True)


def test_capability_probe_failure_does_not_affect_simple_load():
    with patch(
        "graphify.graph_loader.require_multigraph_capabilities",
        side_effect=RuntimeError("should not be called"),
    ):
        # Simple JSON must not trigger the capability probe at all
        G = load_graph(_simple_links(), require_capabilities=True)
    assert type(G) is nx.Graph


# ---------------------------------------------------------------------------
# Blocker 2: missing-key repair must preserve distinct parallel edges
# ---------------------------------------------------------------------------


def _two_missing_key_parallel_edges() -> dict:
    """Multigraph with two missing-key edges sharing relation/file but different attrs."""
    return {
        "multigraph": True,
        "nodes": _NODES,
        "links": [
            {
                "source": "a",
                "target": "b",
                "relation": "calls",
                "source_file": "a.py",
                "confidence": "EXTRACTED",
                "weight": 1.0,
                "context": "one",
            },
            {
                "source": "a",
                "target": "b",
                "relation": "calls",
                "source_file": "a.py",
                "confidence": "EXTRACTED",
                "weight": 1.0,
                "context": "two",
            },
        ],
    }


def test_missing_key_repair_preserves_distinct_parallel_edges(capsys):
    G = load_graph(_two_missing_key_parallel_edges())
    assert type(G) is nx.MultiDiGraph
    assert G.number_of_edges() == 2, (
        f"Both missing-key parallel edges must survive repair; got {G.number_of_edges()}"
    )
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "key" in captured.err.lower()


# ---------------------------------------------------------------------------
# Blocker 3: simple loader must respect serialized directedness
# ---------------------------------------------------------------------------


def test_directed_true_loads_as_digraph():
    data = {
        "directed": True,
        "multigraph": False,
        "nodes": _NODES,
        "edges": [_SIMPLE_EDGE],
    }
    G = load_graph(data)
    assert type(G) is nx.DiGraph


def test_directed_false_explicitly_loads_as_graph():
    data = {
        "directed": False,
        "multigraph": False,
        "nodes": _NODES,
        "edges": [_SIMPLE_EDGE],
    }
    G = load_graph(data)
    assert type(G) is nx.Graph


def test_directed_true_profile_graph_type():
    data = {
        "directed": True,
        "multigraph": False,
        "nodes": _NODES,
        "edges": [_SIMPLE_EDGE],
    }
    G = load_graph(data)
    assert G.graph[GRAPHIFY_PROFILE_KEY]["graph_type"] == "digraph"


# ---------------------------------------------------------------------------
# Blocker 4: malformed JSON must fail cleanly without dropping records
# ---------------------------------------------------------------------------


def test_non_dict_edge_entries_are_rejected():
    data = {"nodes": _NODES, "edges": ["not-a-dict", None, 42]}
    with pytest.raises(GraphStateError, match="must be an object"):
        load_graph(data)


def test_edges_value_not_a_list_raises():
    data = {"nodes": _NODES, "edges": "not-a-list"}
    with pytest.raises((TypeError, ValueError)):
        load_graph(data)


def test_non_dict_graphify_profile_is_rejected():
    data = {
        "nodes": _NODES,
        "edges": [_SIMPLE_EDGE],
        GRAPHIFY_PROFILE_KEY: "bad-profile",
    }
    with pytest.raises(GraphStateError, match="profile must be an object"):
        load_graph(data)


def test_edge_missing_source_or_target_rejected():
    data = {
        "nodes": _NODES,
        "edges": [
            {"target": "b", "relation": "calls"},
            {"source": "a", "relation": "calls"},
        ],
    }
    with pytest.raises(GraphStateError, match="missing source or target"):
        load_graph(data)


# ---------------------------------------------------------------------------
# Non-string multigraph key values must raise before NetworkX sees them
# ---------------------------------------------------------------------------


def _multigraph_with_key(key_value: object) -> dict:
    return {
        "multigraph": True,
        "nodes": _NODES,
        "links": [{**_SIMPLE_EDGE, "key": key_value}],
    }


def test_multigraph_list_key_raises():
    with pytest.raises((TypeError, ValueError)):
        load_graph(_multigraph_with_key(["bad"]))


def test_multigraph_dict_key_raises():
    with pytest.raises((TypeError, ValueError)):
        load_graph(_multigraph_with_key({"bad": 1}))


def test_multigraph_int_key_is_preserved():
    graph = cast(nx.MultiDiGraph, load_graph(_multigraph_with_key(123)))
    assert list(graph.edges(keys=True))[0][2] == 123


def test_load_simple_edge_with_empty_string_source_not_shadowed_by_from():
    # An edge with source="" AND from="a" must not silently use "from" as the
    # source — an explicitly-set empty source means the edge is invalid.
    data = {
        "nodes": _NODES,
        "links": [{"source": "", "from": "a", "target": "b", "relation": "calls"}],
    }
    with pytest.raises(GraphStateError, match="dangling endpoint"):
        load_graph(data)


def test_load_simple_edge_with_from_key_loaded():
    # Edges using legacy "from"/"to" keys should load correctly as long as
    # the IDs are non-empty and present in the node set.
    data = {
        "nodes": _NODES,
        "links": [{"from": "a", "to": "b", "relation": "calls"}],
    }
    G = load_graph(data)
    assert G.number_of_edges() == 1


def test_load_simple_preserves_falsy_hashable_ids():
    """Falsy-but-hashable node IDs like 0 or False must survive the loader."""
    data = {
        "directed": False,
        "multigraph": False,
        "nodes": [{"id": 0}, {"id": ""}, {"id": "x"}],
        "links": [
            {"source": 0, "target": "", "relation": "calls"},
            {"source": 0, "target": "x", "relation": "imports"},
        ],
    }
    G = load_graph(data)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_edge(0, "")
    assert G.has_edge(0, "x")


def test_load_directed_preserves_falsy_hashable_ids():
    data = {
        "directed": True,
        "multigraph": False,
        "nodes": [{"id": 0}, {"id": "y"}],
        "links": [{"source": 0, "target": "y", "relation": "calls"}],
    }
    G = load_graph(data)
    assert G.number_of_edges() == 1
    assert G.has_edge(0, "y")


def test_load_multigraph_preserves_falsy_hashable_ids():
    data = {
        "directed": True,
        "multigraph": True,
        "nodes": [{"id": 0}, {"id": 1}],
        "links": [
            {"source": 0, "target": 1, "key": "k1", "relation": "calls"},
            {"source": 0, "target": 1, "key": "k2", "relation": "imports"},
        ],
    }
    G = load_graph(data)
    assert G.number_of_edges() == 2
    assert G.has_edge(0, 1)


def test_graph_attributes_round_trip_through_node_link_data():
    """G.graph[...] attrs must survive node_link_data → load_graph round-trip.

    NetworkX serializes graph-level metadata under data["graph"]; the loader
    must read from there, not only from top-level keys.
    """
    import networkx as nx
    from networkx.readwrite import json_graph

    G_out = nx.DiGraph()
    G_out.add_node("a")
    G_out.add_node("b")
    G_out.add_edge("a", "b", relation="calls")
    G_out.graph["graphify_profile"] = {"graph_type": "digraph", "extra": "value"}
    G_out.graph["hyperedges"] = [{"members": ["a", "b"]}]
    G_out.graph["graphify_multigraph_diagnostics"] = {"collapsed": 0}

    data = json_graph.node_link_data(G_out, edges="links")
    G_in = load_graph(data)

    assert G_in.graph["graphify_profile"]["extra"] == "value"
    assert G_in.graph["hyperedges"][0]["nodes"] == ["a", "b"]
    assert G_in.graph["hyperedges"][0]["id"].startswith("hyperedge:v1:")
    assert G_in.graph["graphify_multigraph_diagnostics"] == {"collapsed": 0}


def test_graph_attributes_round_trip_through_multigraph_node_link_data():
    """Same round-trip guarantee for multigraph exports."""
    import networkx as nx
    from networkx.readwrite import json_graph

    G_out = nx.MultiDiGraph()
    G_out.add_node("a")
    G_out.add_node("b")
    G_out.add_edge("a", "b", key="k1", relation="calls")
    G_out.add_edge("a", "b", key="k2", relation="imports")
    G_out.graph["graphify_profile"] = {"graph_type": "multidigraph"}
    G_out.graph["graphify_multigraph_diagnostics"] = {"exact_duplicates": 0}

    data = json_graph.node_link_data(G_out, edges="links")
    G_in = load_graph(data, require_capabilities=False)

    assert G_in.graph["graphify_profile"]["graph_type"] == "multidigraph"
    assert G_in.graph["graphify_multigraph_diagnostics"] == {"exact_duplicates": 0}
    assert G_in.number_of_edges() == 2


def test_load_rejects_unhashable_node_ids():
    data = {
        "directed": True,
        "multigraph": False,
        "nodes": [{"id": "ok"}, {"id": ["unhashable"]}, {"id": {"also": "unhashable"}}],
        "links": [{"source": "ok", "target": "ok", "relation": "self"}],
    }
    with pytest.raises(GraphStateError, match="hashable"):
        load_graph(data)


def test_load_rejects_edges_with_unhashable_endpoints():
    data = {
        "directed": True,
        "multigraph": False,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [
            {"source": "a", "target": "b", "relation": "calls"},
            {"source": ["unhashable"], "target": "b", "relation": "bogus"},
            {"source": "a", "target": {"also": "unhashable"}, "relation": "bogus"},
        ],
    }
    with pytest.raises(GraphStateError, match="unhashable endpoint"):
        load_graph(data)


def test_load_multigraph_rejects_unhashable_endpoints():
    data = {
        "directed": True,
        "multigraph": True,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [
            {"source": "a", "target": "b", "key": "k1", "relation": "calls"},
            {"source": ["bad"], "target": "b", "key": "k2", "relation": "calls"},
        ],
    }
    with pytest.raises(GraphStateError, match="unhashable endpoint"):
        load_graph(data, require_capabilities=False)


def test_load_multigraph_duplicate_keys_with_conflicting_attrs_are_rejected():
    data = {
        "directed": True,
        "multigraph": True,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [
            {"source": "a", "target": "b", "key": "same", "relation": "calls", "context": "one"},
            {"source": "a", "target": "b", "key": "same", "relation": "calls", "context": "two"},
        ],
    }
    with pytest.raises(GraphStateError, match="duplicate keyed edge identity"):
        load_graph(data, require_capabilities=False)


def test_load_graph_rejects_non_bool_multigraph_field():
    """String 'false' or other non-bool 'multigraph' must be rejected, not coerced."""
    data = {**_simple_links(), "multigraph": "false"}
    with pytest.raises(GraphStateError, match="multigraph must be a boolean"):
        load_graph(data)


def test_load_graph_rejects_non_bool_directed_field():
    data = {**_simple_links(), "directed": "true"}
    with pytest.raises(GraphStateError, match="directed must be a boolean"):
        load_graph(data)


def test_load_multigraph_with_omitted_directed_does_not_warn(capsys):
    """Missing 'directed' alongside 'multigraph: true' must not trigger the false warning."""
    data = {
        "multigraph": True,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "key": "k", "relation": "calls"}],
    }
    load_graph(data, require_capabilities=False)
    captured = capsys.readouterr()
    assert "multigraph=true but directed=false" not in captured.err


def test_load_graph_rejects_stale_graph_type_in_profile():
    data = {
        "multigraph": True,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "key": "k", "relation": "calls"}],
        "graph": {"graphify_profile": {"graph_type": "simple"}},
    }
    with pytest.raises(GraphStateError, match="profile conflicts"):
        load_graph(data, require_capabilities=False)
