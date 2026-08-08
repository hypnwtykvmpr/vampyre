from __future__ import annotations

import copy
import ast
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import networkx as nx
import pytest

from graphify.graph_state import (
    CompatibilityStatus,
    DecodeMode,
    GraphStateError,
    MetadataPolicy,
    NamedGraphState,
    StateDiagnostic,
    compose_graph_states,
    decode_graph_state,
    encode_graph_state,
    encode_graph_state_bytes,
    merge_graph_states_three_way,
    reconcile_hyperedges,
    repair_legacy_file_node_hyperedge_aliases,
    validate_graph_state,
)


def _payload(graph_type: str = "simple") -> dict:
    multigraph = graph_type == "multidigraph"
    directed = graph_type in {"digraph", "multidigraph"}
    edge = {"source": "a", "target": "b", "relation": "calls", "_origin": "ast"}
    if multigraph:
        edge["key"] = "calls:a.py:L1"
    return {
        "schema_version": 1,
        "directed": directed,
        "multigraph": multigraph,
        "graph": {
            "corpus_name": "fixture",
            "graphify_profile": {"graph_type": graph_type, "extension": "kept"},
        },
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [edge],
        "hyperedges": [],
        "built_at_commit": "abc123",
    }


def test_compose_requires_nonempty_unique_effective_namespaces():
    state = decode_graph_state(_payload(), mode=DecodeMode.STRICT_CURRENT)

    with pytest.raises(GraphStateError, match="non-empty"):
        compose_graph_states(
            [NamedGraphState("", state)],
            target_type="simple",
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )
    with pytest.raises(GraphStateError, match="duplicate implicit namespace"):
        compose_graph_states(
            [NamedGraphState("repo", state), NamedGraphState("repo", state)],
            target_type="simple",
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )
    with pytest.raises(GraphStateError, match="duplicate effective composition namespace"):
        compose_graph_states(
            [
                NamedGraphState("first", state, alias="shared"),
                NamedGraphState("second", state, alias="shared"),
            ],
            target_type="simple",
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )


def test_compose_aliases_preserve_records_and_all_metadata():
    first_payload = _payload("multidigraph")
    first_payload["graph"]["tags"] = ["shared", "first"]
    first_payload["graph"]["owner"] = "alpha"
    first_payload["top"] = "one"
    first_payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"]}]
    second_payload = copy.deepcopy(first_payload)
    second_payload["graph"]["tags"] = ["shared", "second"]
    second_payload["graph"]["owner"] = "beta"
    second_payload["top"] = "two"
    first = decode_graph_state(first_payload, mode=DecodeMode.STRICT_CURRENT)
    second = decode_graph_state(second_payload, mode=DecodeMode.STRICT_CURRENT)

    composed = compose_graph_states(
        [
            NamedGraphState("repo", first, alias="first"),
            NamedGraphState("repo", second, alias="second"),
        ],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    assert type(composed.graph) is nx.MultiDiGraph
    assert set(composed.graph.nodes) == {"first::a", "first::b", "second::a", "second::b"}
    assert composed.graph.number_of_edges() == 2
    assert set(composed.graph["first::a"]["first::b"]) == {"calls:a.py:L1"}
    assert {tuple(cast(list[object], item["nodes"])) for item in composed.hyperedges} == {
        ("first::a", "first::b"),
        ("second::a", "second::b"),
    }
    assert {item["id"] for item in composed.hyperedges} == {
        "first::flow",
        "second::flow",
    }
    assert composed.graph_metadata["tags"] == ["shared", "first", "second"]
    assert composed.graph_metadata["owner"] == "alpha"
    assert composed.top_level_metadata["top"] == "one"
    provenance = composed.graph_metadata["graphify_composition_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["graph.owner"] == {"first": "alpha", "second": "beta"}
    assert provenance["top.top"] == {"first": "one", "second": "two"}


def test_reapplying_same_namespace_preserves_composed_identity():
    payload = _payload("multidigraph")
    payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"]}]
    source = decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)

    first = compose_graph_states(
        [NamedGraphState("repo", source)],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )
    second = compose_graph_states(
        [NamedGraphState("repo", first)],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    assert encode_graph_state(second) == encode_graph_state(first)


def test_compose_uses_direction_rescue_and_refuses_ambiguous_lift():
    payload = _payload("simple")
    payload["links"][0].update({"source": "b", "target": "a", "_src": "a", "_tgt": "b"})
    state = decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)

    composed = compose_graph_states(
        [NamedGraphState("repo", state)],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    assert composed.graph.has_edge("repo::a", "repo::b")
    assert not composed.graph.has_edge("repo::b", "repo::a")
    assert "_src" not in composed.edges[0]
    assert "_tgt" not in composed.edges[0]

    ambiguous_payload = _payload("simple")
    ambiguous = decode_graph_state(ambiguous_payload, mode=DecodeMode.STRICT_CURRENT)
    with pytest.raises(GraphStateError, match="unambiguous _src/_tgt"):
        compose_graph_states(
            [NamedGraphState("repo", ambiguous)],
            target_type="multidigraph",
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )


def test_direction_rescue_survives_simple_directed_round_trip():
    payload = _payload("simple")
    payload["links"][0].update({"source": "b", "target": "a", "_src": "a", "_tgt": "b"})
    source = decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)

    directed = compose_graph_states(
        [NamedGraphState("repo", source)],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )
    with pytest.warns(UserWarning, match="parallel edges will be collapsed"):
        projected = compose_graph_states(
            [NamedGraphState("projected", directed, apply_namespace=False)],
            target_type="simple",
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )
    restored = compose_graph_states(
        [NamedGraphState("restored", projected, apply_namespace=False)],
        target_type="multidigraph",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    assert restored.edges == directed.edges
    assert restored.graph.has_edge("repo::a", "repo::b")
    assert not restored.graph.has_edge("repo::b", "repo::a")


@pytest.mark.parametrize("target_type", ["simple", "digraph"])
def test_compose_refuses_conflicting_edge_claims_between_inputs(target_type):
    first = _payload(target_type)
    second = copy.deepcopy(first)
    first["links"][0]["relation"] = "calls"
    second["links"][0]["relation"] = "imports"

    with pytest.raises(GraphStateError, match="conflicting edge identity"):
        compose_graph_states(
            [
                NamedGraphState(
                    "first",
                    decode_graph_state(first, mode=DecodeMode.STRICT_CURRENT),
                    apply_namespace=False,
                ),
                NamedGraphState(
                    "second",
                    decode_graph_state(second, mode=DecodeMode.STRICT_CURRENT),
                    apply_namespace=False,
                ),
            ],
            target_type=target_type,
            metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
        )


def test_repeated_composition_accumulates_conflict_provenance():
    first = _payload()
    second = _payload()
    third = _payload()
    first["graph"]["owner"] = "alpha"
    second["graph"]["owner"] = "beta"
    third["graph"]["owner"] = "gamma"

    first_round = compose_graph_states(
        [
            NamedGraphState(
                "alpha",
                decode_graph_state(first, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
            NamedGraphState(
                "beta",
                decode_graph_state(second, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
        ],
        target_type="simple",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )
    second_round = compose_graph_states(
        [
            NamedGraphState("existing", first_round, apply_namespace=False),
            NamedGraphState(
                "gamma",
                decode_graph_state(third, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
        ],
        target_type="simple",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    provenance = second_round.graph_metadata["graphify_composition_provenance"]
    assert isinstance(provenance, dict)
    owner_claims = provenance["graph.owner"]
    assert isinstance(owner_claims, dict)
    assert owner_claims["alpha"] == "alpha"
    assert owner_claims["beta"] == "beta"
    assert owner_claims["gamma"] == "gamma"


def test_repeated_namespace_conflicts_use_full_content_digests():
    first = _payload()
    second = _payload()
    third = _payload()
    for payload, owner in ((first, "alpha"), (second, "beta"), (third, "gamma")):
        payload["graph"]["owner"] = owner

    first_round = compose_graph_states(
        [
            NamedGraphState(
                "same",
                decode_graph_state(first, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
            NamedGraphState(
                "other",
                decode_graph_state(second, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
        ],
        target_type="simple",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )
    second_round = compose_graph_states(
        [
            NamedGraphState("existing", first_round, apply_namespace=False),
            NamedGraphState(
                "same",
                decode_graph_state(third, mode=DecodeMode.STRICT_CURRENT),
                apply_namespace=False,
            ),
        ],
        target_type="simple",
        metadata_policy=MetadataPolicy.NAMESPACE_CONFLICTS,
    )

    provenance = second_round.graph_metadata["graphify_composition_provenance"]
    assert isinstance(provenance, dict)
    claims = provenance["graph.owner"]
    assert isinstance(claims, dict)
    digest_keys = [key for key in claims if key.startswith("same:")]
    assert len(digest_keys) == 1
    assert len(digest_keys[0].split(":", 1)[1]) == 64


def test_legacy_migration_preserves_stored_node_ids_without_guessing_boundaries():
    payload = _payload()
    payload.pop("schema_version")
    payload["nodes"][0]["id"] = "legacy_flattened_id"
    payload["links"][0]["source"] = "legacy_flattened_id"

    migrated = decode_graph_state(payload, mode=DecodeMode.MIGRATE_LEGACY)
    encoded = encode_graph_state(migrated)
    encoded_nodes = cast(list[dict[str, object]], encoded["nodes"])

    assert {record["id"] for record in encoded_nodes} == {"legacy_flattened_id", "b"}


def test_graph_state_from_graph_is_pure_and_consumes_top_level_carrier():
    import copy

    from graphify.graph_state import (
        TOP_LEVEL_METADATA_CARRIER,
        graph_state_from_graph,
    )

    graph = nx.Graph()
    graph.add_node("b", label="B")
    graph.add_node("a", label="A")
    graph.add_edge("b", "a", relation="calls", _src="a", _tgt="b")
    graph.graph["custom_meta"] = {"owner": "team"}
    graph.graph[TOP_LEVEL_METADATA_CARRIER] = {"producer": "fixture"}
    before = copy.deepcopy(graph)

    state = graph_state_from_graph(graph, top_level_metadata={"built_at_commit": "abc"})

    assert nx.utils.graphs_equal(graph, before)
    assert state.graph_metadata["custom_meta"] == {"owner": "team"}
    assert TOP_LEVEL_METADATA_CARRIER not in state.graph_metadata
    assert state.top_level_metadata == {"producer": "fixture", "built_at_commit": "abc"}
    assert state.edges == (
        {
            "source": "b",
            "target": "a",
            "relation": "calls",
            "_src": "a",
            "_tgt": "b",
            "_origin": "legacy",
        },
    )


@pytest.mark.parametrize(
    ("graph_type", "expected_type"),
    [("simple", nx.Graph), ("digraph", nx.DiGraph), ("multidigraph", nx.MultiDiGraph)],
)
def test_strict_current_constructs_exact_graph_class(graph_type, expected_type):
    state = decode_graph_state(_payload(graph_type), mode=DecodeMode.STRICT_CURRENT)

    assert type(state.graph) is expected_type
    assert state.graph_type == graph_type
    assert state.compatibility_status is CompatibilityStatus.CURRENT
    assert state.graph_metadata["corpus_name"] == "fixture"
    assert state.graph.graph["graphify_profile"]["extension"] == "kept"
    assert state.top_level_metadata["built_at_commit"] == "abc123"
    validate_graph_state(state)


def test_strict_current_requires_canonical_integer_schema_version():
    data = _payload()
    data["schema_version"] = "1"

    with pytest.raises(GraphStateError, match="schema version"):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)


def test_multidigraph_preserves_integer_edge_key():
    data = _payload("multidigraph")
    data["links"][0]["key"] = 7

    state = decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)
    encoded = encode_graph_state(state)

    assert state.edges[0]["key"] == 7
    encoded_links = cast(list[dict[str, object]], encoded["links"])
    assert encoded_links[0]["key"] == 7


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.pop("directed"),
        lambda data: data.__setitem__("directed", "true"),
        lambda data: data.__setitem__("multigraph", 1),
        lambda data: data.__setitem__("directed", False) or data.__setitem__("multigraph", True),
        lambda data: data["graph"]["graphify_profile"].__setitem__("graph_type", "digraph"),
    ],
)
def test_strict_current_refuses_missing_malformed_or_conflicting_class(mutation):
    data = _payload("simple")
    mutation(data)

    with pytest.raises(GraphStateError):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)


def test_equivalent_dual_edge_fields_are_accepted_order_independently():
    data = _payload("multidigraph")
    second = {
        "source": "a",
        "target": "b",
        "key": "imports:a.py:L2",
        "relation": "imports",
        "_origin": "ast",
    }
    data["links"].append(second)
    data["edges"] = list(reversed(copy.deepcopy(data["links"])))

    state = decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)

    assert state.graph.number_of_edges() == 2


def test_conflicting_dual_edge_fields_are_refused():
    data = _payload()
    data["edges"] = [{"source": "b", "target": "a", "relation": "imports"}]

    with pytest.raises(GraphStateError, match="edges.*links|links.*edges"):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)


def test_top_level_hyperedges_are_authoritative_and_conflicts_refuse():
    data = _payload()
    data["graph"]["hyperedges"] = [{"id": "nested", "nodes": ["a", "b"], "relation": "group"}]

    with pytest.raises(GraphStateError, match="hyperedge"):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)

    data.pop("schema_version")
    del data["hyperedges"]
    state = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)
    assert len(state.hyperedges) == 1
    assert state.compatibility_status is CompatibilityStatus.MIGRATED


def test_strict_state_refuses_dangling_or_degenerate_hyperedges():
    dangling = _payload()
    dangling["hyperedges"] = [{"id": "flow", "nodes": ["a", "missing"]}]
    with pytest.raises(GraphStateError, match="dangling member"):
        decode_graph_state(dangling, mode=DecodeMode.STRICT_CURRENT)

    degenerate = _payload()
    degenerate["hyperedges"] = [{"id": "flow", "nodes": ["a", "a"]}]
    with pytest.raises(GraphStateError, match="fewer than two"):
        decode_graph_state(degenerate, mode=DecodeMode.STRICT_CURRENT)


def test_state_assigns_missing_hyperedge_id_and_refuses_conflicts():
    data = _payload()
    data["hyperedges"] = [{"label": "Flow", "nodes": ["a", "b"]}]

    state = decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)

    hyperedge_id = state.hyperedges[0]["id"]
    assert isinstance(hyperedge_id, str)
    assert hyperedge_id.startswith("hyperedge:v1:")

    data["hyperedges"] = [
        {"id": "same", "label": "One", "nodes": ["a", "b"]},
        {"id": "same", "label": "Two", "nodes": ["a", "b"]},
    ]
    with pytest.raises(GraphStateError, match="conflicting hyperedge id"):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)


def test_implicit_hyperedge_identity_treats_members_as_an_unordered_set():
    first = _payload()
    first["hyperedges"] = [{"label": "Flow", "nodes": ["a", "b"]}]
    second = _payload()
    second["hyperedges"] = [{"label": "Flow", "nodes": ["b", "a"]}]

    first_state = decode_graph_state(first, mode=DecodeMode.STRICT_CURRENT)
    second_state = decode_graph_state(second, mode=DecodeMode.STRICT_CURRENT)

    assert first_state.hyperedges == second_state.hyperedges


def test_hyperedge_canonicalization_preserves_ordered_metadata_and_is_idempotent():
    source = {
        "id": "flow",
        "nodes": ["b", "a", "b"],
        "provenance": ["second observation", "first observation"],
    }

    first = reconcile_hyperedges([source], live_node_ids={"a", "b"})
    second = reconcile_hyperedges(first, live_node_ids={"a", "b"})

    assert first == second
    assert first[0]["nodes"] == ["a", "b"]
    assert first[0]["provenance"] == ["second observation", "first observation"]
    assert source == {
        "id": "flow",
        "nodes": ["b", "a", "b"],
        "provenance": ["second observation", "first observation"],
    }


def test_legacy_file_node_alias_collision_is_order_independent():
    from graphify.ids import make_id

    claimants = [
        {
            "id": "canonical_a",
            "label": "app.py",
            "source_file": "src/app.py",
            "source_location": "L1",
            "_origin": "ast",
        },
        {
            "id": "canonical_b",
            "label": "app.py",
            "source_file": "src/app.py",
            "source_location": "L1",
            "_origin": "ast",
        },
        {"id": "stable"},
    ]
    alias = make_id("src/app.py")
    hyperedges = [{"id": "flow", "nodes": [alias, "stable"]}]

    forward = repair_legacy_file_node_hyperedge_aliases(claimants, hyperedges)
    reverse = repair_legacy_file_node_hyperedge_aliases(
        [*reversed(claimants[:2]), claimants[2]], hyperedges
    )

    assert forward == reverse == hyperedges


@pytest.mark.parametrize("base_has_hyperedge", [False, True])
def test_three_way_hyperedge_conflicts_refuse_instead_of_preferring_missing(
    base_has_hyperedge,
):
    base_payload = _payload("multidigraph")
    current_payload = copy.deepcopy(base_payload)
    other_payload = copy.deepcopy(base_payload)
    if base_has_hyperedge:
        base_payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"], "label": "base"}]
        current_payload["hyperedges"] = []
        other_payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"], "label": "modified"}]
        expected = "delete/modify conflict"
    else:
        current_payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"], "label": "current"}]
        other_payload["hyperedges"] = [{"id": "flow", "nodes": ["a", "b"], "label": "other"}]
        expected = "conflicting additions"

    states = [
        decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)
        for payload in (base_payload, current_payload, other_payload)
    ]
    with pytest.raises(GraphStateError, match=expected):
        merge_graph_states_three_way(*states)


def test_three_way_metadata_delete_modify_conflict_is_recorded_and_idempotent():
    base_payload = _payload()
    base_payload["top_note"] = "base"
    current_payload = copy.deepcopy(base_payload)
    current_payload.pop("top_note")
    other_payload = copy.deepcopy(base_payload)
    other_payload["top_note"] = "modified"
    base, current, other = (
        decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)
        for payload in (base_payload, current_payload, other_payload)
    )

    first = merge_graph_states_three_way(base, current, other)
    second = merge_graph_states_three_way(base, first, other)
    first_payload = encode_graph_state(first)

    assert "top_note" not in first_payload
    graph_metadata = cast(dict[str, object], first_payload["graph"])
    provenance = cast(dict[str, object], graph_metadata["graphify_composition_provenance"])
    assert provenance["top.top_note"] == {
        "current": {"deleted": True},
        "other": "modified",
    }
    assert encode_graph_state_bytes(first) == encode_graph_state_bytes(second)


def test_legacy_migration_reidentifies_conflicting_hyperedges_losslessly():
    data = _payload()
    data.pop("schema_version")
    data["hyperedges"] = [
        {"id": "same", "label": "One", "nodes": ["a", "b"]},
        {"id": "same", "label": "Two", "nodes": ["a", "b"]},
    ]

    state = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)

    assert len(state.hyperedges) == 2
    assert len({record["id"] for record in state.hyperedges}) == 2
    hyperedge_ids = [record["id"] for record in state.hyperedges]
    assert all(isinstance(record_id, str) for record_id in hyperedge_ids)
    assert all(cast(str, record_id).startswith("hyperedge:v1:") for record_id in hyperedge_ids)
    assert "legacy-hyperedge-id-conflict" in {item.code for item in state.diagnostics}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["nodes"].append({"id": "a"}),
        lambda data: data["links"].append(
            {"source": "a", "target": "missing", "relation": "calls"}
        ),
        lambda data: data["nodes"].append("bad"),
        lambda data: data["links"].append("bad"),
    ],
)
def test_strict_current_refuses_malformed_nodes_edges_and_dangling_endpoints(mutate):
    data = _payload()
    mutate(data)
    with pytest.raises(GraphStateError):
        decode_graph_state(data, mode=DecodeMode.STRICT_CURRENT)


def test_duplicate_keyed_identity_collapses_exact_duplicate_but_refuses_conflict():
    exact = _payload("multidigraph")
    exact["links"].append(copy.deepcopy(exact["links"][0]))
    state = decode_graph_state(exact, mode=DecodeMode.STRICT_CURRENT)
    assert state.graph.number_of_edges() == 1

    conflict = _payload("multidigraph")
    conflict_edge = copy.deepcopy(conflict["links"][0])
    conflict_edge["context"] = "different"
    conflict["links"].append(conflict_edge)
    with pytest.raises(GraphStateError, match="duplicate.*key|keyed"):
        decode_graph_state(conflict, mode=DecodeMode.STRICT_CURRENT)


def test_lossless_legacy_migration_repairs_missing_multigraph_key_deterministically():
    data = _payload("multidigraph")
    data.pop("schema_version")
    data["links"][0].pop("key")

    first = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)
    second = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)

    assert first.compatibility_status is CompatibilityStatus.MIGRATED
    first_graph = cast(nx.MultiDiGraph, first.graph)
    second_graph = cast(nx.MultiDiGraph, second.graph)
    assert tuple(first_graph.edges(keys=True)) == tuple(second_graph.edges(keys=True))
    assert first.diagnostics


def test_legacy_migration_marks_missing_edge_provenance_explicitly():
    data = _payload("multidigraph")
    data.pop("schema_version")
    data["links"][0].pop("_origin")

    state = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)
    encoded = encode_graph_state(state)

    assert state.edges[0]["_origin"] == "legacy"
    assert cast(list[dict[str, object]], encoded["links"])[0]["_origin"] == "legacy"
    assert "legacy-missing-edge-provenance" in {item.code for item in state.diagnostics}


def test_strict_current_refuses_missing_or_invalid_edge_provenance():
    missing = _payload()
    missing["links"][0].pop("_origin")
    with pytest.raises(GraphStateError, match="provenance"):
        decode_graph_state(missing, mode=DecodeMode.STRICT_CURRENT)

    invalid = _payload()
    invalid["links"][0]["_origin"] = ""
    with pytest.raises(GraphStateError, match="provenance"):
        decode_graph_state(invalid, mode=DecodeMode.STRICT_CURRENT)


def test_legacy_undirected_multigraph_uses_explicit_direction_rescue():
    data = _payload("multidigraph")
    data.pop("schema_version")
    data["directed"] = False
    data["links"][0]["source"] = "b"
    data["links"][0]["target"] = "a"
    data["links"][0]["_src"] = "a"
    data["links"][0]["_tgt"] = "b"

    state = decode_graph_state(data, mode=DecodeMode.MIGRATE_LEGACY)

    assert type(state.graph) is nx.MultiDiGraph
    assert state.graph.has_edge("a", "b")
    assert not state.graph.has_edge("b", "a")


def test_read_only_legacy_accepts_unmarked_simple_without_claiming_current():
    state = decode_graph_state(
        {"nodes": [{"id": "a"}], "edges": []},
        mode=DecodeMode.READ_ONLY_LEGACY,
    )

    assert type(state.graph) is nx.Graph
    assert state.compatibility_status is CompatibilityStatus.READ_ONLY_LEGACY
    assert state.diagnostics


def _canonical_payload_variant(*, reverse: bool) -> dict:
    nodes = [
        {"id": "z", "label": "Zulu", "metadata": {"b": 2, "a": 1}},
        {"id": 2, "label": "Deux", "metadata": {"a": 1, "b": 2}},
        {"id": 1.5, "label": "Élan"},
    ]
    edges = [
        {"source": "z", "target": 2, "key": "k-z", "relation": "calls", "_origin": "ast"},
        {"source": 2, "target": 1.5, "key": "k-2", "relation": "uses", "_origin": "ast"},
    ]
    hyperedges = [{"id": "flow", "nodes": ["z", 2, 1.5], "label": "Élan flow"}]
    if reverse:
        nodes.reverse()
        edges.reverse()
        graph = {
            "custom": {"second": 2, "first": 1},
            "graphify_profile": {"extension": "kept", "graph_type": "multidigraph"},
        }
    else:
        graph = {
            "graphify_profile": {"graph_type": "multidigraph", "extension": "kept"},
            "custom": {"first": 1, "second": 2},
        }
    return {
        "schema_version": 1,
        "directed": True,
        "multigraph": True,
        "graph": graph,
        "nodes": nodes,
        "links": edges,
        "hyperedges": hyperedges,
        "built_at_commit": "fixed",
    }


def test_encode_graph_state_is_canonical_utf8_and_pure():
    first = decode_graph_state(
        _canonical_payload_variant(reverse=False), mode=DecodeMode.STRICT_CURRENT
    )
    second = decode_graph_state(
        _canonical_payload_variant(reverse=True), mode=DecodeMode.STRICT_CURRENT
    )
    first = replace(
        first,
        diagnostics=(StateDiagnostic("z-last", "Zulu"), StateDiagnostic("a-first", "Élan")),
    )
    second = replace(second, diagnostics=tuple(reversed(first.diagnostics)))
    graph_before = copy.deepcopy(first.graph)
    records_before = copy.deepcopy((first.nodes, first.edges, first.hyperedges))
    metadata_before = copy.deepcopy((first.graph_metadata, first.top_level_metadata))

    first_payload = encode_graph_state(first)
    first_bytes = encode_graph_state_bytes(first)
    second_bytes = encode_graph_state_bytes(second)

    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert "Élan" in first_bytes.decode("utf-8")
    assert first_payload["schema_version"] == 1
    assert first_payload["directed"] is True
    assert first_payload["multigraph"] is True
    encoded_nodes = cast(list[dict[str, object]], first_payload["nodes"])
    encoded_diagnostics = cast(list[dict[str, object]], first_payload["graphify_state_diagnostics"])
    assert [item["id"] for item in encoded_nodes] == [2, 1.5, "z"]
    assert [item["code"] for item in encoded_diagnostics] == [
        "a-first",
        "z-last",
    ]
    assert first.graph.nodes == graph_before.nodes
    assert first.graph.edges == graph_before.edges
    assert (first.nodes, first.edges, first.hyperedges) == records_before
    assert (first.graph_metadata, first.top_level_metadata) == metadata_before


def test_encode_graph_state_bytes_are_hash_seed_independent(tmp_path):
    script = tmp_path / "encode.py"
    script.write_text(
        """
import hashlib
from graphify.graph_state import DecodeMode, decode_graph_state, encode_graph_state_bytes

node_ids = {"z", "a", "m"}
nodes = [{"id": node_id, "label": node_id} for node_id in node_ids]
links = [
    {"source": node_id, "target": "z", "key": f"k-{node_id}", "_origin": "ast"}
    for node_id in node_ids if node_id != "z"
]
payload = {
    "schema_version": 1,
    "directed": True,
    "multigraph": True,
    "graph": {"graphify_profile": {"graph_type": "multidigraph"}},
    "nodes": nodes,
    "links": links,
    "hyperedges": [{"id": "all", "nodes": ["z", "a", "m"]}],
}
state = decode_graph_state(payload, mode=DecodeMode.STRICT_CURRENT)
print(hashlib.sha256(encode_graph_state_bytes(state)).hexdigest())
""".strip()
        + "\n",
        encoding="utf-8",
    )
    digests = []
    for seed in ("1", "77", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        digests.append(result.stdout.strip())
    assert len(set(digests)) == 1


def test_graph_state_writers_do_not_bypass_the_authority():
    project = Path(__file__).parents[1]
    allowed_node_link_encoders = {
        ("graphify/watch.py", "_topology_from_graph"),
        ("graphify/multigraph_compat.py", "_probe_node_link_round_trip"),
    }
    graph_target_names = {
        "existing_graph",
        "graph_json_path",
        "merge_target",
    }
    violations: list[str] = []

    for path in sorted((project / "graphify").glob("*.py")):
        relative = path.relative_to(project).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        parents: list[str] = []

        class Visitor(ast.NodeVisitor):
            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                parents.append(node.name)
                self.generic_visit(node)
                parents.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def visit_Call(self, node: ast.Call) -> None:
                function_name = ""
                if isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    function_name = node.func.id

                owner = parents[-1] if parents else "<module>"
                if (
                    function_name == "node_link_data"
                    and (
                        relative,
                        owner,
                    )
                    not in allowed_node_link_encoders
                ):
                    violations.append(f"{relative}:{node.lineno}: node_link_data in {owner}")

                if function_name in {
                    "write_text",
                    "write_bytes",
                    "replace",
                    "atomic_write_text",
                    "atomic_write_bytes",
                }:
                    target = node.func.value if isinstance(node.func, ast.Attribute) else None
                    first_arg = node.args[0] if node.args else None
                    names = {
                        candidate.id
                        for candidate in (target, first_arg)
                        if isinstance(candidate, ast.Name)
                    }
                    if names & graph_target_names:
                        violations.append(
                            f"{relative}:{node.lineno}: direct graph-state write via {function_name}"
                        )
                self.generic_visit(node)

        Visitor().visit(tree)

    assert violations == []
