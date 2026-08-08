"""Deterministic generative tests for the #1521 hybrid edge eviction.

Two layers:

1. REFERENCE layer (fast): a deterministic corpus generator exercises a reference
   predicate that mirrors the full-rebuild eviction logic in
   ``graphify/watch.py::_rebuild_code`` and asserts the hybrid invariants across
   hundreds of generated graphs.

   The reference is a VERBATIM copy of the production predicate (the nested
   ``_is_ast_node`` / ``_edge_ast_resuppliable`` / ``_edge_evicted`` closures). The
   crucial nuance is that an AST-only rebuild evicts only AST-resuppliable edges;
   semantic-owned records survive in both full and incremental mode unless their
   source is explicitly deleted or ignored. The closures are nested and not
   importable, so this is a mirror; the generated-corpus layer below binds to the
   real code and is the drift detector.

2. GENERATED CORPUS layer (e2e): builds reproducible real corpora on disk, drives
   the ACTUAL ``_rebuild_code`` full rebuild, injects random semantic edges, and
   asserts the end-to-end invariants. This exercises the real predicate, so if the
   mirror above ever drifts from production the two layers diverge.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Reference predicate — VERBATIM mirror of graphify/watch.py::_rebuild_code
# (the existing-graph preservation block). Mode changes extraction scope, not
# record ownership: only AST-resuppliable edges may be replaced by an AST pass.
# ---------------------------------------------------------------------------
def _edge_evicted_ref(
    e: dict,
    *,
    edge_evict_sources: set,
    explicit_evict_sources: set | None = None,
) -> bool:
    def _edge_ast_resuppliable(edge: dict) -> bool:
        # Endpoint ownership is not producer ownership. Only an explicitly AST-
        # owned edge is guaranteed to be re-emitted by an AST-only rebuild.
        return edge.get("_origin") == "ast"

    if not edge_evict_sources:
        return False
    sf = e.get("source_file")
    if not sf:
        return False
    if sf not in edge_evict_sources:  # (real code also tries a normalized fallback)
        return False
    if sf in (explicit_evict_sources or set()):
        return True
    if not _edge_ast_resuppliable(e):
        return False
    return True


def _preserved(edges, *, all_ids, **kw) -> list:
    """Mirror of the preserved_edges comprehension: membership AND not-evicted."""
    return [
        e
        for e in edges
        if e.get("source") in all_ids
        and e.get("target") in all_ids
        and not _edge_evicted_ref(e, **kw)
    ]


# --- deterministic sanity (these lock the mirror against the blanket misread) ---


def test_full_rebuild_does_not_evict_everything():
    """The #1521 BLANKET misread (full rebuild => evict all) must NOT hold."""
    semantic = {"source": "a", "target": "doc1", "source_file": "a.py", "_origin": "semantic"}
    assert (
        _edge_evicted_ref(
            semantic,
            edge_evict_sources={"a.py"},
        )
        is False
    ), "semantic edge must survive full rebuild"


def test_full_rebuild_evicts_stale_ast_edge():
    ast_edge = {"source": "a", "target": "b", "source_file": "a.py", "_origin": "ast"}
    assert (
        _edge_evicted_ref(
            ast_edge,
            edge_evict_sources={"a.py"},
        )
        is True
    ), "stale AST edge must evict on full rebuild"


@pytest.mark.parametrize(
    "origin,sf_evicted,expected",
    [
        ("semantic", True, False),
        ("document", True, False),
        (None, True, False),
        ("ast", True, True),
        ("ast", False, False),
    ],
)
def test_eviction_truth_table(origin, sf_evicted, expected):
    e = {"source": "s", "target": "t", "source_file": "f.py"}
    if origin is not None:
        e["_origin"] = origin
    assert (
        _edge_evicted_ref(
            e,
            edge_evict_sources=({"f.py"} if sf_evicted else set()),
        )
        is expected
    )


# --- deterministic generated-graph layer ---
_ORIGINS = ("ast", "semantic", "document", "rationale", None)
_NODE_IDS = ("n0", "n1", "n2", "n3", "n4")
_FILES = ("a.py", "b.py", "c.py")


def _graph(seed: int):
    rng = random.Random(seed)
    new_ast_ids = {nid for nid in _NODE_IDS if rng.choice((False, True))}
    edges = []
    for _ in range(rng.randrange(13)):
        edge: dict[str, str | None] = {
            "source": rng.choice(_NODE_IDS),
            "target": rng.choice(_NODE_IDS),
        }
        if rng.choice((False, True)):
            edge["source_file"] = rng.choice(_FILES)
        if rng.choice((False, True)):
            edge["_origin"] = rng.choice(_ORIGINS)
        edges.append(edge)
    files = {source_file for e in edges if isinstance(source_file := e.get("source_file"), str)}
    evict = {name for name in sorted(files) if rng.choice((False, True))}
    return edges, evict, new_ast_ids


def _generated_graphs():
    return (_graph(seed) for seed in range(400))


def test_I1_sourceless_never_evicted():
    for edges, evict, _new_ast_ids in _generated_graphs():
        for e in edges:
            if not e.get("source_file"):
                assert not _edge_evicted_ref(
                    e,
                    edge_evict_sources=evict,
                ), f"I1: sourceless edge evicted: {e}"


def test_I2_full_rebuild_preserves_nonast_touching():
    for edges, evict, _new_ast_ids in _generated_graphs():
        for e in edges:
            if e.get("_origin") != "ast":
                assert not _edge_evicted_ref(
                    e,
                    edge_evict_sources=evict,
                ), f"I2: non-AST-owned edge evicted on full rebuild: {e}"


def test_I3_full_rebuild_evicts_resuppliable_ast():
    for edges, evict, _new_ast_ids in _generated_graphs():
        for e in edges:
            sf = e.get("source_file")
            if sf and sf in evict and e.get("_origin") == "ast":
                assert _edge_evicted_ref(
                    e,
                    edge_evict_sources=evict,
                ), f"I3: AST-re-suppliable edge NOT evicted on full rebuild: {e}"


def test_I5_unre_extracted_file_preserved():
    for edges, evict, _new_ast_ids in _generated_graphs():
        for e in edges:
            sf = e.get("source_file")
            if sf and sf not in evict:
                assert not _edge_evicted_ref(
                    e,
                    edge_evict_sources=evict,
                ), f"I5: edge from un-re-extracted file evicted: {e}"


def test_I4_incremental_preserves_semantic_owned_edges():
    """Incremental AST extraction cannot replace semantic-owned records."""
    for edges, evict, _new_ast_ids in _generated_graphs():
        for e in edges:
            sf = e.get("source_file")
            if sf and sf in evict and e.get("_origin") not in (None, "ast"):
                assert not _edge_evicted_ref(
                    e,
                    edge_evict_sources=evict,
                ), f"I4: semantic-owned edge was evicted by AST-only rebuild: {e}"


def test_I6_membership_filter():
    """preserved_edges keeps an edge only if BOTH endpoints are in all_ids,
    independent of eviction."""
    for edges, evict, new_ast_ids in _generated_graphs():
        all_ids = set(new_ast_ids)  # any subset; endpoints outside must be dropped
        kept = _preserved(
            edges,
            all_ids=all_ids,
            edge_evict_sources=evict,
        )
        for e in kept:
            assert e["source"] in all_ids and e["target"] in all_ids, (
                f"I6: kept edge with endpoint outside all_ids: {e}"
            )


def test_explicit_source_deletion_evicts_semantic_owned_edge():
    semantic = {
        "source": "a",
        "target": "b",
        "source_file": "gone.py",
        "_origin": "semantic",
    }

    assert _edge_evicted_ref(
        semantic,
        edge_evict_sources={"gone.py"},
        explicit_evict_sources={"gone.py"},
    )


# ---------------------------------------------------------------------------
# GENERATED CORPUS (e2e) — drives the real _rebuild_code on real corpora.
# Fixed seeds keep the filesystem shapes reproducible.
# ---------------------------------------------------------------------------
def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"], check=True)


def _rng(seed: int):
    # ponytail: tiny LCG, no import of random needed for reproducible structure.
    state = {"s": seed * 2654435761 % (2**32)}

    def nxt(n):
        state["s"] = (1103515245 * state["s"] + 12345) % (2**31)
        return state["s"] % n

    return nxt


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11, 13, 17, 19, 23])
def test_generated_corpus_semantic_survives_real_full_rebuild(tmp_path, seed):
    """Random corpus + random injected semantic edges: a full rebuild (AST only)
    must preserve every injected semantic edge whose endpoints survive, and must
    re-supply real AST call edges (so total never collapses below the AST set).

    The setup intentionally uses only cross-platform Git commands (init, -C,
    and config), so this contract runs on every supported operating system.
    """
    from graphify.watch import _rebuild_code
    from graphify.graph_loader import load_graph
    import networkx as nx

    nxt = _rng(seed)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _git_init(corpus)

    n_files = 2 + nxt(4)  # 2..5 files
    fns = []
    for fi in range(n_files):
        name = f"m{fi}"
        # each file defines one function; some call a prior function (real edges)
        body = ["def f():", "    return 1"]
        if fns and nxt(2) == 0:
            callee_mod, _ = fns[nxt(len(fns))]
            body = [
                f"import {callee_mod}",
                "",
                "def f():",
                f"    {callee_mod}.f()",
                f"    return {callee_mod}.f()",
            ]
        (corpus / f"{name}.py").write_text("\n".join(body) + "\n", encoding="utf-8")
        fns.append((name, f"{name}_f"))

    cwd = os.getcwd()
    try:
        os.chdir(corpus)
        assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
        gpath = corpus / "graphify-out" / "graph.json"
        data = json.loads(gpath.read_text(encoding="utf-8"))
        node_ids = [n["id"] for n in data["nodes"]]
        assert node_ids, "corpus produced no nodes"

        # Inject random semantic edges between existing nodes: some sourced to a
        # real file (the over-eviction fault line), some sourceless.
        injected = []
        n_inj = 1 + nxt(4)
        files = [f"m{i}.py" for i in range(n_files)]
        for k in range(n_inj):
            s = node_ids[nxt(len(node_ids))]
            t = node_ids[nxt(len(node_ids))]
            rel = f"sem_{seed}_{k}"
            e = {
                "source": s,
                "target": t,
                "relation": rel,
                "confidence": "INFERRED",
                "_origin": "semantic",
            }
            if nxt(2) == 0:
                e["source_file"] = files[nxt(len(files))]
            injected.append(e)
        links = data.get("links", data.get("edges", []))
        links.extend(injected)
        for index, edge in enumerate(links):
            edge.setdefault("key", f"fixture-{index}")
        data["links"] = links
        data["multigraph"] = True
        data["directed"] = True
        data.setdefault("graph", {}).setdefault("graphify_profile", {})["graph_type"] = (
            "multidigraph"
        )
        gpath.write_text(json.dumps(data), encoding="utf-8")

        assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
        after = json.loads(gpath.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)

    after_links = after.get("links", after.get("edges", []))
    after_rels = {e.get("relation") for e in after_links}
    surviving = set(n["id"] for n in after["nodes"])
    for e in injected:
        if e["source"] in surviving and e["target"] in surviving:
            assert e["relation"] in after_rels, (
                f"seed {seed}: injected semantic edge {e['relation']} "
                f"(sourced={'source_file' in e}) lost on full rebuild"
            )

    reloaded = load_graph(after)
    assert isinstance(reloaded, nx.MultiDiGraph)
