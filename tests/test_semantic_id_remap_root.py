"""A node whose source_file equals the scan root must not crash build (#1618).

`_norm_source_file` relativizes an absolute source_file that equals the scan root
to `Path('.')`. `_semantic_id_remap` then fed that into `_file_stem`, whose
`path.with_suffix("")` raises `ValueError: '.' has an empty name` — crashing the
final graph assembly AFTER all LLM extraction cost was spent, writing no graph at
all. A project-level node (source_file == root) has no per-file identity to remap,
so its id is left untouched.
"""

from __future__ import annotations

from pathlib import Path

from graphify.build import _semantic_id_remap, build_from_json, canonicalize_semantic_fragment
from graphify.extractors.base import _file_stem


def test_file_stem_handles_dot_path():
    assert _file_stem(Path(".")) == ""  # no raise
    assert _file_stem(Path("src/foo.py")) == "src/foo"


def test_semantic_id_remap_root_equal_source_file_no_crash():
    root = "/some/project/root"
    node = {"id": "some_concept", "source_file": root, "_origin": "semantic"}
    remap = _semantic_id_remap([node], root)  # must not raise
    # a root-equal node has no file stem, so its id is left untouched (not remapped)
    assert "some_concept" not in remap


def test_build_from_json_with_root_level_concept_node():
    root = "/proj"
    combined = {
        "nodes": [
            {
                "id": "proj_concept",
                "label": "Project",
                "file_type": "concept",
                "source_file": root,
                "_origin": "semantic",
            },
            {
                "id": "src_foo",
                "label": "foo",
                "file_type": "code",
                "source_file": "src/foo.py",
                "_origin": "ast",
            },
        ],
        "edges": [],
    }
    G = build_from_json(combined, root=root)  # previously crashed here
    assert G.number_of_nodes() == 2


def test_normal_semantic_remap_still_works():
    # regression guard: a real per-file node still gets remap consideration (#1504)
    remap = _semantic_id_remap(
        [{"id": "foo", "source_file": "src/foo.py", "_origin": "semantic"}], "/proj"
    )
    assert isinstance(remap, dict)


def test_labeled_semantic_node_does_not_collide_with_its_ast_file_node():
    source_file = "tools/skillgen/expected/graphify__skills__amp__references__update.md"
    file_id = "tools_skillgen_expected_graphify_skills_amp_references_update"
    label = "graphify reference: incremental update and cluster-only (amp)"
    semantic_id = "tools_skillgen_expected_graphify__skills__amp__references__update"

    remap = _semantic_id_remap(
        [
            {
                "id": file_id,
                "label": Path(source_file).name,
                "source_file": source_file,
                "source_location": "L1",
                "_origin": "ast",
            },
            {
                "id": semantic_id,
                "label": label,
                "source_file": source_file,
                "file_type": "concept",
                "_origin": "semantic",
            },
        ],
        "/repo",
    )

    assert remap[semantic_id] != file_id
    assert remap[semantic_id].startswith(file_id + "_")


def test_semantic_path_collisions_use_the_ast_disambiguation_contract():
    fragment = {
        "nodes": [
            {
                "id": "foo_bar_baz",
                "label": "bar_baz.md",
                "source_file": "foo/bar_baz.md",
                "file_type": "document",
                "source_location": "L1",
            },
            {
                "id": "foo_bar_baz",
                "label": "baz.md",
                "source_file": "foo_bar/baz.md",
                "file_type": "document",
                "source_location": "L1",
            },
            {
                "id": "sink",
                "label": "sink",
                "source_file": "sink.md",
                "file_type": "document",
            },
        ],
        "edges": [
            {
                "source": "foo_bar_baz",
                "target": "sink",
                "relation": "references",
                "source_file": "foo/bar_baz.md",
            },
            {
                "source": "foo_bar_baz",
                "target": "sink",
                "relation": "references",
                "source_file": "foo_bar/baz.md",
            },
        ],
        "hyperedges": [],
    }

    canonical = canonicalize_semantic_fragment(fragment, "/repo")
    file_ids = {node["id"] for node in canonical["nodes"] if node["id"] != "sink"}

    assert len(file_ids) == 2
    assert {edge["source"] for edge in canonical["edges"]} == file_ids
    assert canonical["graphify_identity_diagnostics"]["semantic_contested_aliases"] == [
        "foo_bar_baz"
    ]
