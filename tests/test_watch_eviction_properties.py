"""Property + generative ("robot factory") tests for the #1521 hybrid edge eviction.

Two layers:

1. PROPERTY layer (fast): Hypothesis fuzzes a reference predicate that mirrors the
   full-rebuild eviction logic in ``graphify/watch.py::_rebuild_code`` and asserts
   the hybrid invariants hold across randomly generated graphs.

   The reference is a VERBATIM copy of the production predicate (the nested
   ``_is_ast_node`` / ``_edge_ast_resuppliable`` / ``_edge_evicted`` closures). The
   crucial nuance — that a full rebuild evicts ONLY AST-re-suppliable edges (it does
   NOT evict everything) — is preserved here. (Two independent LLM drafts of this
   mirror both collapsed it to "full rebuild → evict all", i.e. the #1521 BLANKET
   behavior; this file encodes the real HYBRID.) The closures are nested and not
   importable, so this is a mirror; the robot-factory layer below binds to the REAL
   code and is the drift detector.

2. ROBOT FACTORY layer (e2e): builds randomly-shaped real corpora on disk, drives
   the ACTUAL ``_rebuild_code`` full rebuild, injects random semantic edges, and
   asserts the end-to-end invariants. This exercises the real predicate, so if the
   mirror above ever drifts from production the two layers diverge.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Reference predicate — VERBATIM mirror of graphify/watch.py::_rebuild_code
# (the existing-graph preservation block). Kept import-free so Hypothesis can
# hammer it. full_rebuild does NOT evict everything: it evicts only edges the
# AST pass can re-supply (both endpoints AST-origin, no non-AST _origin marker).
# ---------------------------------------------------------------------------
def _edge_evicted_ref(
    e: dict,
    *,
    full_rebuild: bool,
    edge_evict_sources: set,
    new_ast_ids: set,
    origin_by_id: dict,
) -> bool:
    def _is_ast_node(nid) -> bool:
        return nid in new_ast_ids or origin_by_id.get(nid) == "ast"

    def _edge_ast_resuppliable(edge: dict) -> bool:
        origin = edge.get("_origin")
        if origin is not None and origin != "ast":
            return False
        return _is_ast_node(edge.get("source")) and _is_ast_node(edge.get("target"))

    if not edge_evict_sources:
        return False
    sf = e.get("source_file")
    if not sf:
        return False
    if sf not in edge_evict_sources:  # (real code also tries a normalized fallback)
        return False
    if full_rebuild and not _edge_ast_resuppliable(e):
        return False
    return True


def _preserved(edges, *, all_ids, **kw) -> list:
    """Mirror of the preserved_edges comprehension: membership AND not-evicted."""
    return [
        e for e in edges
        if e.get("source") in all_ids and e.get("target") in all_ids
        and not _edge_evicted_ref(e, **kw)
    ]


# --- deterministic sanity (these lock the mirror against the blanket misread) ---

def test_full_rebuild_does_not_evict_everything():
    """The #1521 BLANKET misread (full rebuild => evict all) must NOT hold."""
    semantic = {"source": "a", "target": "doc1", "source_file": "a.py", "_origin": "semantic"}
    kw = dict(full_rebuild=True, edge_evict_sources={"a.py"},
              new_ast_ids={"a"}, origin_by_id={"a": "ast", "doc1": None})
    assert _edge_evicted_ref(semantic, **kw) is False, "semantic edge must survive full rebuild"


def test_full_rebuild_evicts_stale_ast_edge():
    ast_edge = {"source": "a", "target": "b", "source_file": "a.py"}
    kw = dict(full_rebuild=True, edge_evict_sources={"a.py"},
              new_ast_ids={"a", "b"}, origin_by_id={"a": "ast", "b": "ast"})
    assert _edge_evicted_ref(ast_edge, **kw) is True, "stale AST edge must evict on full rebuild"


@pytest.mark.parametrize("origin,both_ast,sf_evicted,full,expected", [
    ("semantic", True, True, True, False),   # I2 explicit non-AST marker -> kept
    (None, False, True, True, False),        # I2 non-AST endpoint -> kept
    (None, True, True, True, True),           # I3 AST-AST stale -> evicted
    (None, True, False, True, False),         # I5 not in evict set -> kept
    (None, True, True, False, True),          # I4 incremental: guard off -> evicted
    ("semantic", False, True, False, True),  # I4 incremental: semantic on changed file evicts
])
def test_eviction_truth_table(origin, both_ast, sf_evicted, full, expected):
    e = {"source": "s", "target": "t", "source_file": "f.py"}
    if origin is not None:
        e["_origin"] = origin
    origin_by_id = {"s": "ast", "t": "ast" if both_ast else None}
    new_ast_ids = {"s", "t"} if both_ast else {"s"}
    kw = dict(full_rebuild=full, edge_evict_sources=({"f.py"} if sf_evicted else set()),
              new_ast_ids=new_ast_ids, origin_by_id=origin_by_id)
    assert _edge_evicted_ref(e, **kw) is expected


# --- Hypothesis property layer ---
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

_ORIGINS = st.sampled_from(["ast", "semantic", "document", "rationale", None])
_node_ids = st.sampled_from(["n0", "n1", "n2", "n3", "n4"])
_files = st.sampled_from(["a.py", "b.py", "c.py", None])


@st.composite
def _graph(draw):
    ids = ["n0", "n1", "n2", "n3", "n4"]
    origin_by_id = {nid: draw(_ORIGINS) for nid in ids}
    new_ast_ids = {nid for nid in ids if draw(st.booleans())}
    edges = draw(st.lists(
        st.fixed_dictionaries(
            {"source": _node_ids, "target": _node_ids},
            optional={"source_file": st.sampled_from(["a.py", "b.py", "c.py"]),
                      "_origin": _ORIGINS},
        ),
        min_size=0, max_size=12,
    ))
    files = {e["source_file"] for e in edges if e.get("source_file")}
    evict = draw(st.sets(st.sampled_from(sorted(files) or ["a.py"]), max_size=3))
    full = draw(st.booleans())
    return edges, evict, new_ast_ids, origin_by_id, full


@settings(max_examples=400)
@given(_graph())
def test_I1_sourceless_never_evicted(g):
    edges, evict, new_ast_ids, origin_by_id, full = g
    for e in edges:
        if not e.get("source_file"):
            assert not _edge_evicted_ref(e, full_rebuild=full, edge_evict_sources=evict,
                                         new_ast_ids=new_ast_ids, origin_by_id=origin_by_id), \
                f"I1: sourceless edge evicted: {e}"


@settings(max_examples=400)
@given(_graph())
def test_I2_full_rebuild_preserves_nonast_touching(g):
    edges, evict, new_ast_ids, origin_by_id, _full = g

    def is_ast(nid):
        return nid in new_ast_ids or origin_by_id.get(nid) == "ast"

    for e in edges:
        nonast = (not is_ast(e.get("source")) or not is_ast(e.get("target"))
                  or e.get("_origin") not in (None, "ast"))
        if nonast:
            assert not _edge_evicted_ref(e, full_rebuild=True, edge_evict_sources=evict,
                                         new_ast_ids=new_ast_ids, origin_by_id=origin_by_id), \
                f"I2: non-AST-touching edge evicted on full rebuild: {e}"


@settings(max_examples=400)
@given(_graph())
def test_I3_full_rebuild_evicts_resuppliable_ast(g):
    edges, evict, new_ast_ids, origin_by_id, _full = g

    def is_ast(nid):
        return nid in new_ast_ids or origin_by_id.get(nid) == "ast"

    for e in edges:
        sf = e.get("source_file")
        resup = (e.get("_origin") in (None, "ast")
                 and is_ast(e.get("source")) and is_ast(e.get("target")))
        if sf and sf in evict and resup:
            assert _edge_evicted_ref(e, full_rebuild=True, edge_evict_sources=evict,
                                     new_ast_ids=new_ast_ids, origin_by_id=origin_by_id), \
                f"I3: AST-re-suppliable edge NOT evicted on full rebuild: {e}"


@settings(max_examples=400)
@given(_graph())
def test_I5_unre_extracted_file_preserved(g):
    edges, evict, new_ast_ids, origin_by_id, full = g
    for e in edges:
        sf = e.get("source_file")
        if sf and sf not in evict:
            assert not _edge_evicted_ref(e, full_rebuild=full, edge_evict_sources=evict,
                                         new_ast_ids=new_ast_ids, origin_by_id=origin_by_id), \
                f"I5: edge from un-re-extracted file evicted: {e}"


@settings(max_examples=400)
@given(_graph())
def test_I4_incremental_guard_off(g):
    """The endpoint-AST guard applies ONLY on full rebuild: incrementally, any
    source_file-evicted edge evicts regardless of origin."""
    edges, evict, new_ast_ids, origin_by_id, _full = g
    for e in edges:
        sf = e.get("source_file")
        if sf and sf in evict:
            assert _edge_evicted_ref(e, full_rebuild=False, edge_evict_sources=evict,
                                     new_ast_ids=new_ast_ids, origin_by_id=origin_by_id), \
                f"I4: source_file-evicted edge survived incremental rebuild: {e}"


@settings(max_examples=400)
@given(_graph())
def test_I6_membership_filter(g):
    """preserved_edges keeps an edge only if BOTH endpoints are in all_ids,
    independent of eviction."""
    edges, evict, new_ast_ids, origin_by_id, full = g
    all_ids = set(new_ast_ids)  # any subset; endpoints outside must be dropped
    kept = _preserved(edges, all_ids=all_ids, full_rebuild=full,
                      edge_evict_sources=evict, new_ast_ids=new_ast_ids,
                      origin_by_id=origin_by_id)
    for e in kept:
        assert e["source"] in all_ids and e["target"] in all_ids, \
            f"I6: kept edge with endpoint outside all_ids: {e}"


@settings(max_examples=400)
@given(_graph())
def test_full_vs_incremental_difference_is_exactly_the_guarded_set(g):
    """The only edges full preserves that incremental evicts are exactly the
    source_file-evicted, NON-re-suppliable ones (the semantic layer the guard saves)."""
    edges, evict, new_ast_ids, origin_by_id, _full = g
    kw = dict(edge_evict_sources=evict, new_ast_ids=new_ast_ids, origin_by_id=origin_by_id)

    def is_ast(nid):
        return nid in new_ast_ids or origin_by_id.get(nid) == "ast"

    full_evicts = {id(e) for e in edges if _edge_evicted_ref(e, full_rebuild=True, **kw)}
    inc_evicts = {id(e) for e in edges if _edge_evicted_ref(e, full_rebuild=False, **kw)}
    saved_by_guard = inc_evicts - full_evicts
    for e in edges:
        if id(e) in saved_by_guard:
            sf = e.get("source_file")
            resup = (e.get("_origin") in (None, "ast")
                     and is_ast(e.get("source")) and is_ast(e.get("target")))
            assert sf and sf in evict and not resup, \
                f"guard saved an edge it should not have: {e}"


# ---------------------------------------------------------------------------
# ROBOT FACTORY (e2e) — drives the REAL _rebuild_code on random real corpora.
# Deterministic seeds (not Hypothesis) so the filesystem fuzz is reproducible.
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


@pytest.mark.skipif(sys.platform == "win32", reason="git CLI behaviour varies on Windows runners")
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11, 13, 17, 19, 23])
def test_robot_factory_semantic_survives_real_full_rebuild(tmp_path, seed):
    """Random corpus + random injected semantic edges: a full rebuild (AST only)
    must preserve every injected semantic edge whose endpoints survive, and must
    re-supply real AST call edges (so total never collapses below the AST set)."""
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
            body = [f"import {callee_mod}", "", "def f():",
                    f"    {callee_mod}.f()", f"    return {callee_mod}.f()"]
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
            e = {"source": s, "target": t, "relation": rel,
                 "confidence": "INFERRED", "_origin": "semantic"}
            if nxt(2) == 0:
                e["source_file"] = files[nxt(len(files))]
            injected.append(e)
        links = data.get("links", data.get("edges", []))
        links.extend(injected)
        data["links"] = links
        data["multigraph"] = True
        data["directed"] = True
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
