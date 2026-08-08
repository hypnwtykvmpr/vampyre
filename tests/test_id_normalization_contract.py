"""Drift guard for the node-ID normalization contract.

Three independent producers must agree on node IDs or the graph splits one entity
into disconnected ghost nodes: the AST extractor (``extract._make_id``), the
semantic subagents (the skill prompt's node-ID spec), and the graph builder
(``build._normalize_id``, which reconciles edge endpoints). The recipe used to be
copy-pasted into ``_make_id`` and ``_normalize_id`` and kept in sync only by
mirrored docstrings — exactly how the recurring ID-drift bug class crept in
(#811 Unicode collapse, #550 same-filename collisions, #1033 AST-vs-LLM file-node
mismatch, #1104).

Both callers now delegate to :mod:`graphify.ids`, so they share one
implementation and cannot diverge. These tests lock that contract: if a future
change re-forks the normalization (a new local helper, an inlined regex, a
dropped ``casefold``), they fail.
"""

import json
import os
import random
import re
import string
import subprocess
import sys

import pytest

from graphify.build import _normalize_id
from graphify.extract import _make_id
from graphify.ids import make_id, normalize_id

# Inputs that previously diverged or are easy to get wrong. The single-part form
# of `_make_id` must equal `_normalize_id` for every one of these.
CONTRACT_CASES = [
    "Session_ValidateToken",  # casing
    "session.validate-token",  # punctuation -> underscore
    "foo__bar..baz",  # repeated separators collapse
    "  Leading_Trailing__  ",  # strip stray underscores/space
    "A/B\\C",  # path separators both directions
    "MixedCASE",  # #811: casefold
    "café",  # composed accented Latin (NFKC)
    "café",  # decomposed e + combining acute -> same as 'café'
    "日本語クラス",  # #811: CJK letters survive, not collapsed
    "Кириллица",  # Cyrillic survives
    "naïve_Über",  # mixed accented Latin
    "x_c1",  # must NOT be treated as a chunk suffix here
    "__dunder__",  # leading/trailing underscores stripped
    "tab\tnewline\nspace ",  # whitespace runs -> single underscore
]


@pytest.mark.parametrize("raw", CONTRACT_CASES)
def test_make_id_matches_normalize_id(raw):
    """The AST id-maker and the builder's reconciler must agree, char for char."""
    assert _make_id(raw) == _normalize_id(raw), (
        f"ID drift for {raw!r}: extract._make_id -> {_make_id(raw)!r} but "
        f"build._normalize_id -> {_normalize_id(raw)!r}"
    )


@pytest.mark.parametrize("raw", CONTRACT_CASES)
def test_normalize_id_is_idempotent(raw):
    once = normalize_id(raw)
    assert normalize_id(once) == once, f"normalize_id not idempotent for {raw!r}"


def test_make_id_joins_then_normalizes():
    """Multipart IDs retain their boundaries rather than flattening them."""
    parts = ("auth", "session.py", "ValidateToken")
    result = make_id(*parts)
    assert result == normalize_id(result)
    assert result.startswith("auth_")


def test_multipart_boundaries_are_injective():
    assert make_id("mod", "a_b") != make_id("mod_a", "b")
    assert make_id("a", "b_c", "d") != make_id("a_b", "c", "d")


def test_make_id_rejects_empty_canonical_output():
    with pytest.raises(ValueError, match="empty"):
        make_id("_")


def test_turkish_dotted_i_is_idempotent_and_grammar_safe():
    once = normalize_id("İ")
    assert once == "i"
    assert normalize_id(once) == once
    assert re.fullmatch(r"\w+", once)


def test_unicode_identifiers_do_not_collapse_to_empty():
    """#811: non-ASCII identifiers must yield distinct, non-empty IDs rather than
    collapsing to a single per-file node."""
    a = _make_id("クラスА")
    b = _make_id("クラスB")
    assert a and b and a != b


def test_normalized_ids_are_safe_node_ids():
    """Output is lowercase and contains no path/punctuation separators."""
    for raw in CONTRACT_CASES:
        out = normalize_id(raw)
        assert out == out.casefold()
        assert not re.search(r"[./\\\s]", out), f"unsafe char in id {out!r}"
        assert not out.startswith("_") and not out.endswith("_")


def test_both_callers_share_one_implementation():
    """Guard against re-forking: the two public callers must resolve to the same
    underlying function object as graphify.ids.normalize_id."""
    # build._normalize_id is imported directly from graphify.ids.
    assert _normalize_id is normalize_id
    # extract._make_id wraps make_id; prove it round-trips through the shared core.
    assert _make_id("Foo.Bar") == normalize_id("Foo.Bar")
    # The other two live ID producers — MCP config ingestion and bash symbol
    # resolution — must also resolve to the shared recipe, or the "single source
    # of truth" leaks back into copy-pasted forks (#1378).
    from graphify.mcp_ingest import _make_id as _mcp_make_id
    from graphify.symbol_resolution import _bash_make_id

    for fn in (_make_id, _mcp_make_id, _bash_make_id):
        assert fn("Foo.Bar", "baz") == make_id("Foo.Bar", "baz")
        assert fn("Ångström", "Ⅳ") == make_id("Ångström", "Ⅳ")


def _generated_normalization_inputs() -> list[str]:
    """Return a reproducible mix of ASCII, separators, and Unicode identifiers."""
    alphabet = list(
        string.ascii_letters
        + string.digits
        + " _-./\\\t\n:#@$%&*()[]{}"
        + "cafe\u0301naiveUberAngstrom"
        + "日本語クラスКириллицаⅣ"
    )
    rng = random.Random(1378)
    samples = ["", *CONTRACT_CASES]
    for _ in range(512):
        samples.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(65))))
    return samples


def test_generated_inputs_keep_all_id_producers_aligned():
    for raw in _generated_normalization_inputs():
        if not _normalize_id(raw):
            with pytest.raises(ValueError, match="empty"):
                _make_id(raw)
            continue
        assert _make_id(raw) == _normalize_id(raw)


def test_generated_inputs_normalize_idempotently():
    for raw in _generated_normalization_inputs():
        once = normalize_id(raw)
        assert normalize_id(once) == once


def test_contested_alias_resolution_is_stable_across_hash_seeds():
    script = r"""
import json
from graphify.build import build_from_json

nodes = [
    {"id": node_id, "label": node_id, "file_type": "code", "source_file": source_file}
    for node_id, source_file in {
        ("A-B", "a.py"),
        ("A_B", "b.py"),
        ("sink", "sink.py"),
    }
]
graph = build_from_json({
    "nodes": nodes,
    "edges": [
        {"source": "A-B", "target": "sink", "relation": "exact", "source_file": "a.py"},
        {"source": "a b", "target": "sink", "relation": "fuzzy", "source_file": "x.py"},
    ],
})
print(json.dumps({
    "nodes": sorted(graph.nodes),
    "edges": sorted(
        (data.get("_src", u), data.get("_tgt", v), data.get("relation"))
        for u, v, data in graph.edges(data=True)
    ),
    "diagnostics": graph.graph["graphify_identity_diagnostics"],
}, sort_keys=True))
"""
    outputs = set()
    for seed in range(32):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        outputs.add(json.dumps(json.loads(completed.stdout), sort_keys=True))

    assert len(outputs) == 1
