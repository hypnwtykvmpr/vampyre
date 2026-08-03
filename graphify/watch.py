# monitor a folder and auto-trigger --update when files change
from __future__ import annotations
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

from graphify.detect import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PAPER_EXTENSIONS,
    _is_ignored,
    _load_graphifyignore,
)
from graphify.installation import uv_tool_install_command
from graphify.persistence import (
    FileStateTransaction,
    atomic_write_text,
    locked_file,
    write_pending_signal,
)

# Single source of truth in graphify.paths (#1423); re-exported as _GRAPHIFY_OUT.
from graphify.paths import (
    GRAPHIFY_OUT as _GRAPHIFY_OUT,
    write_scan_root_marker,
)

_PENDING_FILENAME = ".pending_changes"
_PENDING_INFLIGHT_GLOB = ".pending_changes.inflight.*"
_PENDING_GUARD_FILENAME = ".pending_changes.guard"
_PENDING_DRAIN_MAX_PASSES = 20


def _queue_pending(out_dir: Path, changed_paths: list[Path]) -> None:
    """Append path-safe JSON records to ``out_dir/.pending_changes``.

    Used by a post-commit hook process that cannot acquire ``_rebuild_lock``
    so its change set is not silently dropped (#1059). The lock-holding
    process drains this file before and after its rebuild and merges the
    contents with its own change set.

    A trailing newline keeps a torn final record isolated. The queue guard
    serializes appenders on every supported platform; JSON encoding preserves
    legal path characters such as embedded newlines.
    """
    if not changed_paths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = out_dir / _PENDING_FILENAME
    payload = "".join(
        json.dumps({"path": os.fspath(path)}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for path in changed_paths
    )
    with locked_file(out_dir / _PENDING_GUARD_FILENAME) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-change queue lock")
        with open(pending, "a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())


def _pending_paths(raw_batches: list[str]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for raw in raw_batches:
        for line in raw.splitlines():
            value = line.strip()
            if not value:
                continue
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, dict) and isinstance(decoded.get("path"), str):
                value = decoded["path"]
            if value in seen:
                continue
            seen.add(value)
            paths.append(Path(value))
    return paths


def _claim_pending(out_dir: Path) -> tuple[list[Path], list[Path]]:
    """Move queued changes into durable in-flight files without acknowledging them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = out_dir / _PENDING_GUARD_FILENAME
    with locked_file(guard) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-change queue lock")
        pending = out_dir / _PENDING_FILENAME
        if pending.exists():
            claim = out_dir / f"{_PENDING_FILENAME}.inflight.{os.getpid()}.{time.time_ns()}"
            os.replace(pending, claim)
        claim_files = sorted(out_dir.glob(_PENDING_INFLIGHT_GLOB))
        raw_batches: list[str] = []
        readable_claims: list[Path] = []
        for path in claim_files:
            try:
                raw_batches.append(path.read_text(encoding="utf-8"))
                readable_claims.append(path)
            except OSError:
                continue
    return _pending_paths(raw_batches), readable_claims


def _ack_pending(out_dir: Path, claim_files: list[Path]) -> None:
    if not claim_files:
        return
    with locked_file(out_dir / _PENDING_GUARD_FILENAME) as acquired:
        if not acquired:
            raise RuntimeError("could not acquire pending-change queue lock")
        for path in claim_files:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


def _drain_pending(out_dir: Path) -> list[Path]:
    """Compatibility helper that immediately acknowledges a claimed batch."""
    paths, claims = _claim_pending(out_dir)
    _ack_pending(out_dir, claims)
    return paths


def _merge_changed_paths(*sources: "list[Path] | None") -> list[Path]:
    """Concatenate path lists, preserving order and dropping duplicates.

    Used to combine a hook process's own ``changed_paths`` with the drained
    contents of ``.pending_changes`` so the lock-holding rebuild covers
    every queued commit's worth of files (#1059).
    """
    seen: set[str] = set()
    out: list[Path] = []
    for src in sources:
        if not src:
            continue
        for p in src:
            key = os.fspath(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


@contextlib.contextmanager
def _rebuild_lock(out_dir: Path, *, blocking: bool = False):
    """Per-repo advisory lock around a rebuild.

    Yields True if acquired, False if another rebuild is already running and
    ``blocking`` is False. The persistent guard uses the platform's native
    advisory lock, which is released automatically if the process is killed.

    While the lock is held, ``.rebuild.lock`` contains the owning PID followed
    by a newline so external pollers (publish scripts, etc.) can read it.
    On successful release the file is unlinked so downstream tooling that
    waits for the lock to clear by polling for its absence unblocks promptly.

    Windows uses ``msvcrt.locking``; POSIX uses ``fcntl.flock``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / ".rebuild.lock"
    guard_path = out_dir / ".rebuild.guard"
    with locked_file(guard_path, blocking=blocking) as acquired:
        if not acquired:
            yield False
            return
        atomic_write_text(status_path, f"{os.getpid()}\n")
        try:
            yield True
        finally:
            with contextlib.suppress(OSError):
                status_path.unlink()


def _apply_resource_limits() -> None:
    """Best-effort nice + memory cap. Called from inline hook scripts.

    GRAPHIFY_REBUILD_MEMORY_LIMIT_MB caps RSS-ish memory. Uses RLIMIT_DATA on
    macOS (RLIMIT_AS is unreliable under Apple's libmalloc) and RLIMIT_AS on
    Linux. Silently skips if the platform doesn't support it.
    """
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass
    mb = os.environ.get("GRAPHIFY_REBUILD_MEMORY_LIMIT_MB", "").strip()
    if not mb:
        return
    try:
        limit = int(mb) * 1024 * 1024
    except ValueError:
        return
    try:
        import resource

        which = resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS
        soft, hard = resource.getrlimit(which)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < limit else limit
        resource.setrlimit(which, (limit, new_hard))
    except (ImportError, ValueError, OSError):
        pass


def _git_head() -> str | None:
    """Return current git HEAD commit hash, or None outside a repo."""
    import subprocess as _sp

    try:
        r = _sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


_WATCHED_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | PAPER_EXTENSIONS | IMAGE_EXTENSIONS
_CODE_EXTENSIONS = CODE_EXTENSIONS


def _report_root_label(watch_path: Path) -> str:
    if watch_path.is_absolute():
        return watch_path.name or str(watch_path)
    return Path.cwd().name if watch_path == Path(".") else str(watch_path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _changed_path_candidates(raw: Path, *, change_root: Path, watch_root: Path) -> list[Path]:
    """Return plausible absolute locations for a hook-provided changed path.

    Git hooks pass paths relative to the repository root. Watch callers may
    also pass paths relative to the watched root. Keep both interpretations so
    a graph rooted at ``src`` accepts ``src/app.py`` and ``app.py``.
    """
    if raw.is_absolute():
        return [raw.resolve()]

    candidates: list[Path] = []
    seen: set[str] = set()
    for base in (change_root, watch_root):
        cand = (base / raw).resolve()
        key = os.fspath(cand)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)
    return candidates


def _relativize_source_files(payload: dict, root: Path) -> None:
    # Include "links" alongside "edges": modern NetworkX serialises the edge list
    # under "links", and FIX 3 (source_file-aware edge eviction) compares each
    # edge's source_file against an eviction set built from repo-relative paths.
    # Without relativising edge source_files here, an absolute-pathed preserved
    # edge from an older graph would never match the relative evict_sources and
    # would survive as stale. Nodes were already relativised; this aligns edges.
    for bucket in ("nodes", "edges", "links", "hyperedges"):
        for item in payload.get(bucket, []):
            source = item.get("source_file")
            if not source:
                continue
            source_path = Path(source)
            if not source_path.is_absolute():
                continue
            try:
                item["source_file"] = source_path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue


def _node_community_map(graph_data: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in graph_data.get("nodes", []):
        node_id = node.get("id")
        cid = node.get("community")
        if node_id is None or cid is None:
            continue
        try:
            out[str(node_id)] = int(cid)
        except (TypeError, ValueError):
            print(
                f"[graphify watch] Skipping node with invalid community id: "
                f"node_id={node_id!r} community={cid!r}",
                file=sys.stderr,
            )
            continue
    return out


def _canonical_graph_for_compare(graph_data: dict) -> dict:
    canonical = dict(graph_data)
    canonical.pop("built_at_commit", None)
    # Graph-level metadata under the top-level "graph" key (graphify_profile, a
    # duplicate hyperedges copy, NetworkX bookkeeping) is NOT graph structure.
    # export.to_json now persists graphify_profile there (PR 7 Phase A); without
    # this strip the on-disk graph ({"graph": {"graphify_profile": ...}}) would
    # never compare equal to a candidate that lacks it, defeating the
    # no-change short-circuit. Hyperedge topology is preserved via the
    # authoritative top-level "hyperedges" key sorted below.
    canonical.pop("graph", None)
    # Fold the legacy edge-list key so a graph.json written under the old "edges"
    # key compares equal to a modern candidate keyed "links". Without this fold a
    # legacy file ({"edges": [...]}) and the fresh candidate ({"links": [...]})
    # canonicalise to dicts with DIFFERENT keys and the no-change short-circuit
    # flaps — every watcher tick rewrites graph.json forever (RISK 3). Mirrors the
    # edge handling in _canonical_topology_for_compare; the deliberate difference
    # is that node fields (community/norm_label) are NOT stripped here so existing
    # watch graph-compare tests keep passing.
    if "links" not in canonical and "edges" in canonical:
        canonical["links"] = canonical.pop("edges")
    # Treat a missing hyperedges key as an empty list so a null-vs-[] history
    # (older writers omitted the key; modern ones persist []) does not register
    # as a change.
    if "hyperedges" not in canonical:
        canonical["hyperedges"] = []

    links = canonical.get("links")
    if isinstance(links, list):
        norm_links = []
        for edge in links:
            if not isinstance(edge, dict):
                norm_links.append(edge)
                continue
            e = dict(edge)
            # to_json overwrites source/target with the canonical _src/_tgt before
            # serialising, so the on-disk graph has no _src/_tgt while a candidate
            # fresh from node_link_data still does. Pop and reassign so both sides
            # compare on the same directed endpoints (existing gets no-op pops).
            true_src = e.pop("_src", None)
            true_tgt = e.pop("_tgt", None)
            if true_src is not None and true_tgt is not None:
                e["source"] = true_src
                e["target"] = true_tgt
            # VOLATILE: confidence_score is recomputed from confidence on every
            # export, so a legacy file that stamped it must not differ from a
            # candidate that has not yet had it recomputed.
            e.pop("confidence_score", None)
            # PRESERVE key: NetworkX guarantees `key` is unique within a
            # (source, target) pair, so parallel edges differ only by key and must
            # stay distinct in the sorted comparison. The json.dumps sort key below
            # already includes it because we never strip it; this is explicit.
            if "key" in edge:
                e["key"] = edge["key"]
            norm_links.append(e)
        canonical["links"] = norm_links

    for key in ("nodes", "links", "hyperedges"):
        if key in canonical and isinstance(canonical[key], list):
            canonical[key] = sorted(
                canonical[key],
                key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str),
            )
    return canonical


def _canonical_topology_for_compare(graph_data: dict) -> dict:
    canonical = dict(graph_data)
    canonical.pop("built_at_commit", None)
    # Graph-level metadata under the top-level "graph" key (graphify_profile, a
    # duplicate hyperedges copy, NetworkX bookkeeping) is NOT topology. Phase A's
    # export.to_json persists graphify_profile there; the on-disk graph then has
    # {"graph": {"graphify_profile": ...}} while a fresh candidate from
    # _topology_from_graph has {"graph": {}}, which would otherwise be read as a
    # spurious topology change and needlessly re-run cluster(). Strip it just like
    # built_at_commit. Hyperedge topology is still compared via the authoritative
    # top-level "hyperedges" key normalised below.
    canonical.pop("graph", None)
    # Fold the legacy edge-list key so a graph.json written under the old "edges"
    # key compares equal to a modern candidate keyed "links". The clustered
    # compare otherwise normalises "edges" and "links" under their OWN keys, so a
    # legacy file ({"edges": [...]}) and a fresh candidate ({"links": [...]})
    # canonicalise to dicts with DIFFERENT keys and the topology compare flaps —
    # needlessly re-running cluster() on every tick (same root cause as the
    # no-cluster RISK 3 flap fixed in _canonical_graph_for_compare).
    if "links" not in canonical and "edges" in canonical:
        canonical["links"] = canonical.pop("edges")
    # Treat a missing hyperedges key as an empty list so a legacy file that
    # dropped the key does not differ from a candidate carrying [].
    if "hyperedges" not in canonical:
        canonical["hyperedges"] = []

    nodes = canonical.get("nodes")
    if isinstance(nodes, list):
        norm_nodes = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            n = dict(node)
            n.pop("community", None)
            n.pop("norm_label", None)
            norm_nodes.append(n)
        canonical["nodes"] = sorted(
            norm_nodes,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str),
        )

    for key in ("links",):
        items = canonical.get(key)
        if not isinstance(items, list):
            continue
        norm_edges = []
        for edge in items:
            if not isinstance(edge, dict):
                continue
            e = dict(edge)
            # to_json writes _src/_tgt as the canonical directed endpoints and
            # overwrites source/target with them before serialising, so the
            # on-disk graph has no _src/_tgt. The candidate topology (fresh from
            # node_link_data) still has them. Popping and reassigning here makes
            # both sides comparable: existing gets no-op pops (None), candidate
            # gets source/target overwritten from _src/_tgt — same result.
            true_src = e.pop("_src", None)
            true_tgt = e.pop("_tgt", None)
            if true_src is not None and true_tgt is not None:
                e["source"] = true_src
                e["target"] = true_tgt
            # VOLATILE fields — derived/recomputed every rebuild, so they must NOT
            # drive a "topology changed" verdict. confidence_score is recomputed
            # from confidence on every export (see export._CONFIDENCE_SCORE_DEFAULTS).
            e.pop("confidence_score", None)
            # IDENTITY fields are everything that survives: source, target,
            # relation, confidence, source_file, source_location, weight, and —
            # critically for MultiDiGraphs — `key`. NetworkX guarantees `key` is
            # unique within a (source, target) pair, so two parallel edges that
            # share the same relation/source_file/source_location but differ only
            # in `key` MUST stay distinct in the sorted comparison; otherwise an
            # unchanged multigraph with parallel edges would read as "changed"
            # (or a real parallel-edge add/remove would be silently missed). The
            # json.dumps sort key below already includes `key` because we never
            # strip it — this assignment makes that contract explicit and guards
            # against a future edit accidentally dropping it from the canonical
            # edge. Simple graphs have no `key`, so this is a no-op for them.
            if "key" in edge:
                e["key"] = edge["key"]
            norm_edges.append(e)
        canonical[key] = sorted(
            norm_edges,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str),
        )

    hyperedges = canonical.get("hyperedges")
    if isinstance(hyperedges, list):
        canonical["hyperedges"] = sorted(
            hyperedges,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str),
        )

    return canonical


def _dedupe_rebuilt_edge_records(edges: list[dict]) -> list[dict]:
    """Remove duplicate edge records introduced by preserving + re-extracting.

    A full AST rebuild re-extracts all code edges and also preserves existing
    graph links so semantic/non-code relationships survive. Without this pass,
    raw ``--no-cluster`` rebuilds append the same AST links on every run.

    Distinct keyed parallels remain distinct. When the same relationship appears
    once with a key and once without one, prefer the keyed record because it
    carries the stable MultiDiGraph identity from the previous graph.
    """

    def fingerprint(edge: dict) -> str:
        comparable = dict(edge)
        comparable.pop("key", None)
        comparable.pop("confidence_score", None)
        # _origin is provenance, not connectivity identity — exclude it so a fresh
        # AST-stamped edge and a preserved legacy (un-stamped) edge for the same
        # relationship still collapse on the first rebuild after edge stamping
        # lands; otherwise tens of thousands of duplicate edge records inflate the
        # graph until the second rebuild (#1521). Same rationale as popping key.
        comparable.pop("_origin", None)
        return json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)

    kept: list[dict] = []
    by_fingerprint: dict[str, list[int]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            kept.append(edge)
            continue
        fp = fingerprint(edge)
        key = edge.get("key")
        existing_indexes = by_fingerprint.get(fp, [])
        if not existing_indexes:
            by_fingerprint[fp] = [len(kept)]
            kept.append(edge)
            continue

        if key is None:
            # A keyless duplicate is either an exact repeat from a previous raw
            # no-cluster rebuild or a fresh AST record that matches a preserved
            # keyed edge. In both cases the existing record is authoritative.
            continue

        same_key = False
        replace_keyless_index: int | None = None
        for idx in existing_indexes:
            existing = kept[idx]
            if not isinstance(existing, dict):
                continue
            existing_key = existing.get("key")
            if existing_key == key:
                same_key = True
                break
            if existing_key is None and replace_keyless_index is None:
                replace_keyless_index = idx
        if same_key:
            continue
        if replace_keyless_index is not None:
            kept[replace_keyless_index] = edge
            continue
        existing_indexes.append(len(kept))
        kept.append(edge)
    return kept


def _topology_from_graph(G) -> dict:
    from networkx.readwrite import json_graph

    try:
        data = json_graph.node_link_data(G, edges="links")
    except TypeError:
        data = json_graph.node_link_data(G)
    data["hyperedges"] = getattr(G, "graph", {}).get("hyperedges", [])
    return data


def _existing_is_multigraph(graph_data: dict) -> bool:
    """Return True when an on-disk ``graph.json`` payload is a MultiDiGraph.

    Mirrors :func:`graphify.graph_loader.load_graph`'s detection so an
    incremental ``_rebuild_code`` rebuilds with the SAME graph class the file
    was saved as. Without inheriting this flag, the rebuild would re-emit a
    simple ``DiGraph`` whose serialized ``graph.json`` declares ``multigraph``
    unset — a deferred silent fallback: the preserved parallel-edge *records*
    survive the write, but the next ``load_graph`` collapses them to one edge
    per pair (the PR 7 go/no-go violation).

    The top-level ``multigraph`` boolean is authoritative WHENEVER IT IS PRESENT,
    exactly as ``load_graph`` treats it — including an explicit ``false``. A
    serialized ``graphify_profile.graph_type == "multidigraph"`` is consulted
    only when the top-level flag is ABSENT, rescuing a graph whose
    profile-stamping writer omitted it so a multigraph is never misread as
    simple. Letting a stale profile override an explicit ``multigraph: false``
    would make this helper disagree with the loader on the same file — the
    rebuild would emit a MultiDiGraph for a payload every reader loads as a
    DiGraph.
    """
    if not isinstance(graph_data, dict):
        return False
    if "multigraph" in graph_data:
        return graph_data["multigraph"] is True
    graph_meta = graph_data.get("graph")
    if isinstance(graph_meta, dict):
        from graphify.graph_loader import GRAPHIFY_PROFILE_KEY

        profile = graph_meta.get(GRAPHIFY_PROFILE_KEY)
        if isinstance(profile, dict) and profile.get("graph_type") == "multidigraph":
            return True
    return False


def _existing_graph_type(graph_data: dict) -> str:
    """Return the graph class declared by an existing serialized payload."""
    if _existing_is_multigraph(graph_data):
        return "multidigraph"
    if isinstance(graph_data, dict) and graph_data.get("directed") is True:
        return "digraph"
    return "simple"


def _stamp_graph_type(graph_data: dict, graph_type: str) -> None:
    """Persist an explicit loader profile on raw no-cluster output."""
    from graphify.graph_loader import GRAPHIFY_PROFILE_KEY

    graph_data["multigraph"] = graph_type == "multidigraph"
    graph_data["directed"] = graph_type in {"digraph", "multidigraph"}
    graph_meta = graph_data.get("graph")
    if not isinstance(graph_meta, dict):
        graph_meta = {}
    profile = graph_meta.get(GRAPHIFY_PROFILE_KEY)
    profile = dict(profile) if isinstance(profile, dict) else {}
    profile["graph_type"] = graph_type
    graph_meta[GRAPHIFY_PROFILE_KEY] = profile
    graph_data["graph"] = graph_meta


def _check_shrink(
    force: bool,
    existing_data: dict,
    new_data: dict,
    tmp: "Path | None" = None,
    *,
    had_explicit_deletions: bool = False,
    rebuilt_sources: "set[str] | None" = None,
) -> bool:
    """Return True (ok to proceed) or False (shrink refused).

    When False, cleans up *tmp* if provided and prints a warning to stderr.

    The shrink-guard exists to catch SILENT shrinkage from failed extraction
    chunks (a half-written semantic pass leaving thousands of nodes
    unaccounted for). When ``had_explicit_deletions`` is True, the caller
    has declared which files were removed (e.g. the post-commit hook saw
    a ``D`` in ``git diff --name-only``) and a smaller graph is the expected
    outcome — skip the guard so legitimate refactors don't require ``--force``.

    ``rebuilt_sources`` (when given) is the set of source files re-extracted this
    run. A net shrink is legitimate — not a failed chunk — when every *lost* node
    belonged to one of those files (a symbol removed from a re-extracted file) or
    carries no source_file. Only an unexplained loss (a node from a file we did
    NOT touch — e.g. a dropped semantic/doc node) refuses the write. This lets a
    plain ``graphify update`` after deleting a function refresh the graph without
    ``--force`` (#1116 left stale nodes write-blocked even though build dropped them).
    """
    existing_n = len(existing_data.get("nodes", [])) if existing_data else 0
    new_n = len(new_data.get("nodes", []))
    # ABSOLUTE 0-floor: a populated graph must NEVER be overwritten with an empty
    # (0-node) one. A 0-node candidate over a populated graph is the signature of
    # a failed/aborted extraction (a crashed or half-written pass), not a real
    # result — even a total delete-all leaves the corpus build at the "no code
    # files" early return rather than a 0-node write. This floor sits BEFORE the
    # force/had_explicit_deletions short-circuit on purpose: neither --force nor a
    # declared deletion is a license to wipe a populated graph to nothing.
    if existing_n > 0 and new_n == 0:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        print(
            f"[graphify] ERROR: refusing to overwrite a populated graph.json "
            f"({existing_n} nodes) with an EMPTY (0-node) graph - this is a "
            f"failed/aborted extraction, not a real result. The previous graph "
            f"is preserved.",
            file=sys.stderr,
        )
        return False
    if force or not existing_data or had_explicit_deletions:
        return True
    existing_nodes = existing_data.get("nodes", [])
    new_nodes = new_data.get("nodes", [])
    if len(new_nodes) >= len(existing_nodes):
        return True
    if rebuilt_sources is not None:
        from graphify.build import _norm_source_file

        new_ids = {n.get("id") for n in new_nodes}
        lost = [n for n in existing_nodes if n.get("id") not in new_ids]

        def _accounted(n: dict) -> bool:
            sf = n.get("source_file")
            return not sf or sf in rebuilt_sources or _norm_source_file(sf) in rebuilt_sources

        if all(_accounted(n) for n in lost):
            return True
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    print(
        f"[graphify] WARNING: new graph has {len(new_nodes)} nodes but existing "
        f"graph.json has {len(existing_nodes)}. Refusing to overwrite — you may be "
        f"missing chunk files from a previous session. "
        f"Pass --force to override.",
        file=sys.stderr,
    )
    return False


def _report_for_compare(report_text: str) -> str:
    return re.sub(r"^- Built from commit: `[^`]+`\n?", "", report_text, flags=re.MULTILINE)


def _json_text(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _missing_local_source(source_file: object, *roots: Path) -> bool:
    """Return whether a path-like local source is absent from every scan root."""
    if not isinstance(source_file, str) or not source_file.strip():
        return False
    source = source_file.strip()
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", source):
        return False
    # A Windows absolute path cannot be proven stale from a non-Windows host.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", source):
        return False

    path = Path(source).expanduser()
    known_extensions = CODE_EXTENSIONS | DOC_EXTENSIONS | PAPER_EXTENSIONS | IMAGE_EXTENSIONS
    if not path.is_absolute() and path.suffix.lower() not in known_extensions:
        return False
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    return bool(candidates) and not any(candidate.exists() for candidate in candidates)


def _rebuild_code(
    watch_path: Path,
    *,
    output_dir: Path | None = None,
    output_root: Path | None = None,
    graph_type: str | None = None,
    changed_paths: list[Path] | None = None,
    follow_symlinks: bool = False,
    force: bool = False,
    no_cluster: bool = False,
    no_viz: bool = False,
    acquire_lock: bool = True,
    block_on_lock: bool = False,
) -> bool:
    """Re-run AST extraction + build + optional cluster + report for code files. No LLM needed.

    When ``force`` is True the node-count safety check in ``to_json`` is bypassed
    so the rebuilt graph overwrites graph.json even if it has fewer nodes.
    Use this after refactors that legitimately delete code.

    When ``changed_paths`` is provided, only those files are re-extracted; nodes
    for unchanged files are preserved from the existing graph. Deleted paths
    in ``changed_paths`` (paths that no longer exist on disk) are dropped from
    the preserved set. When ``changed_paths`` is None the full code corpus is
    re-extracted (used by the watcher and post-checkout hook).

    ``acquire_lock`` (default True) takes a non-blocking per-repo flock around
    the rebuild so concurrent post-commit hooks across multiple repos do not
    pile up. Returns False with a log line if the lock is held. Pass
    ``block_on_lock=True`` to wait instead of skip (used by the interactive
    ``graphify update`` CLI).

    ``no_cluster`` skips community detection and writes raw merged extraction
    JSON to graphify-out/graph.json (mirrors ``extract --no-cluster``).

    ``no_viz`` skips graph.html generation while still refreshing graph.json
    and GRAPH_REPORT.md.

    Returns True on success, False on error or skipped-due-to-lock.
    """
    out = Path(output_dir).resolve() if output_dir is not None else watch_path / _GRAPHIFY_OUT
    storage_root = Path(output_root).resolve() if output_root is not None else watch_path.resolve()
    if acquire_lock:
        # #1059: incremental (changed_paths is not None) hooks must not drop
        # their change set when another rebuild is already running. Queue
        # before attempting the lock so a non-blocking failure still records
        # the work; the lock-holder drains the queue and merges it in. Full-
        # corpus rebuilds skip the queue entirely — they already cover every
        # file, so there is nothing to merge.
        if changed_paths is not None and not block_on_lock:
            _queue_pending(out, list(changed_paths))
        with _rebuild_lock(out, blocking=block_on_lock) as got:
            if not got:
                print(
                    "[graphify watch] Rebuild already in progress for "
                    f"{watch_path.resolve()} - changes queued."
                )
                return False
            # Lock acquired. Claim anything queued by earlier contenders
            # (including the paths we just queued ourselves), but acknowledge
            # the claim only after the rebuild succeeds. A crash or refusal
            # therefore leaves an in-flight batch for the next process.
            pending, claim_files = _claim_pending(out)
            if changed_paths is not None:
                merged = _merge_changed_paths(changed_paths, pending)
            else:
                # Full-corpus rebuild supersedes any queued incremental work.
                merged = None
            ok = _rebuild_code(
                watch_path,
                output_dir=out,
                output_root=storage_root,
                graph_type=graph_type,
                changed_paths=merged,
                follow_symlinks=follow_symlinks,
                force=force,
                no_cluster=no_cluster,
                no_viz=no_viz,
                acquire_lock=False,
            )
            if not ok:
                return False
            _ack_pending(out, claim_files)
            # Late-arrival drain: another hook may have queued work while we
            # were rebuilding. Loop up to _PENDING_DRAIN_MAX_PASSES times so a
            # storm of commits eventually quiesces without livelocking. A full
            # rebuild already saw everything, so skip this for changed_paths is None.
            if merged is not None:
                for _ in range(_PENDING_DRAIN_MAX_PASSES):
                    late, late_claims = _claim_pending(out)
                    if not late:
                        break
                    ok = _rebuild_code(
                        watch_path,
                        output_dir=out,
                        output_root=storage_root,
                        graph_type=graph_type,
                        changed_paths=late,
                        follow_symlinks=follow_symlinks,
                        force=force,
                        no_cluster=no_cluster,
                        no_viz=no_viz,
                        acquire_lock=False,
                    )
                    if not ok:
                        return False
                    _ack_pending(out, late_claims)
            return ok

    watch_root = watch_path.resolve()
    project_root = Path.cwd().resolve() if not watch_path.is_absolute() else watch_root
    report_root = _report_root_label(watch_path)
    try:
        from graphify.extract import extract, _get_extractor
        from graphify.detect import detect, snapshot_source_hashes
        from graphify.build import build_from_json, _norm_source_file as _nsf
        from graphify.cluster import cluster, remap_communities_to_previous, score_all
        from graphify.analyze import god_nodes, surprising_connections, suggest_questions
        from graphify.report import generate
        from graphify.export import to_json, to_html
        from graphify.security import check_graph_file_size_cap

        detected = detect(watch_path, follow_symlinks=follow_symlinks)
        code_files = [Path(f) for f in detected["files"]["code"]]

        # Include document files that have AST extractors (e.g. .md, .mdx, .qmd)
        for doc_file in detected["files"].get("document", []):
            p = Path(doc_file)
            if _get_extractor(p) is not None:
                code_files.append(p)

        if not code_files:
            print("[graphify watch] No code files found - nothing to rebuild.")
            return False

        # Incremental path: when the caller passed an explicit change list,
        # extract only changed-and-still-existing files. Deleted paths are
        # tracked separately so their stale nodes can be evicted below.
        deleted_paths: set[str] = set()

        def _add_deleted_source(path: Path) -> None:
            for root in (project_root, watch_root):
                deleted_paths.add(_nsf(str(path), str(root)) or str(path))

        if changed_paths is not None:
            code_set = {p.resolve() for p in code_files}
            wanted: list[Path] = []
            change_root = Path.cwd().resolve()
            for raw in changed_paths:
                candidates = _changed_path_candidates(
                    raw,
                    change_root=change_root,
                    watch_root=watch_root,
                )
                tracked = next(
                    (cand for cand in candidates if cand.exists() and cand in code_set), None
                )
                if tracked is not None:
                    if tracked not in wanted:
                        wanted.append(tracked)
                    continue

                existing_in_root = next(
                    (
                        cand
                        for cand in candidates
                        if cand.exists() and _is_relative_to(cand, watch_root)
                    ),
                    None,
                )
                if existing_in_root is not None:
                    # The path exists under the watched root but detect filtered
                    # it out. Evict any stale nodes that still claim it.
                    _add_deleted_source(existing_in_root)
                    continue

                deleted_in_root = next(
                    (cand for cand in candidates if _is_relative_to(cand, watch_root)),
                    None,
                )
                if deleted_in_root is not None:
                    # File was deleted or renamed away inside the watched root.
                    # Evict preserved nodes that still claim this source path.
                    _add_deleted_source(deleted_in_root)
            if not wanted and not deleted_paths:
                print("[graphify watch] No tracked code files in change set - skipping rebuild.")
                return True
            extract_targets = wanted
        else:
            extract_targets = code_files

        if changed_paths is None:
            manifest_files = detected["files"]
        else:
            rebuilt_paths = {path.resolve() for path in extract_targets}
            manifest_files = {
                kind: [path for path in paths if Path(path).resolve() in rebuilt_paths]
                for kind, paths in detected["files"].items()
            }

        try:
            source_snapshot = snapshot_source_hashes(
                [path for paths in manifest_files.values() for path in paths]
            )
        except OSError as exc:
            print(f"[graphify watch] Rebuild refused: {exc}", file=sys.stderr)
            return False

        commit = _git_head()
        result = (
            extract(
                extract_targets,
                cache_root=storage_root,
                source_root=project_root,
                defer_cache_writes=True,
            )
            if extract_targets
            else {
                "nodes": [],
                "edges": [],
                "hyperedges": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        failed_files = list(result.get("failed_files", []))
        if failed_files:
            preview = ", ".join(failed_files[:3])
            if len(failed_files) > 3:
                preview += f", and {len(failed_files) - 3} more"
            print(
                f"[graphify watch] Rebuild refused: AST extraction failed for "
                f"{len(failed_files)} file(s): {preview}",
                file=sys.stderr,
            )
            return False

        deferred_cache_entries = result.pop("_deferred_cache_entries", [])
        result.pop("failed_files", None)

        cache_committed = False

        def _commit_ast_cache() -> None:
            nonlocal cache_committed
            if cache_committed:
                return
            cache_committed = True
            from graphify.cache import save_cached

            for entry in deferred_cache_entries:
                save_cached(
                    Path(entry["path"]),
                    entry["result"],
                    storage_root,
                    source_root=project_root,
                )

        # Preserve semantic nodes/edges from a previous full run.
        # AST-only rebuild replaces nodes for changed files; everything else is kept.
        # Filter by node ID membership in the new AST output, not by file_type —
        # INFERRED/AMBIGUOUS nodes extracted from code files also carry file_type="code"
        # and would be wrongly dropped by a file_type-based filter.
        # When the caller supplied changed_paths, also evict preserved nodes whose
        # source_file matches a path that was changed (re-extracted) or deleted —
        # otherwise the old nodes for those files would survive forever.
        existing_graph = out / "graph.json"
        existing_graph_data: dict = {}
        existing_graph_metadata: dict = {}
        if existing_graph.exists():
            try:
                check_graph_file_size_cap(existing_graph)
                existing = json.loads(existing_graph.read_text(encoding="utf-8"))
                existing_graph_data = existing
                raw_graph_metadata = existing.get("graph")
                if isinstance(raw_graph_metadata, dict):
                    existing_graph_metadata = {
                        key: value
                        for key, value in raw_graph_metadata.items()
                        if key != "hyperedges"
                    }
                new_ast_ids = {n["id"] for n in result["nodes"]}
                _relativize_source_files(existing, project_root)
                evict_sources: set[str] = set(deleted_paths)
                if changed_paths is not None:
                    for p in extract_targets:
                        for root in (project_root, watch_root):
                            evict_sources.add(_nsf(str(p), str(root)) or str(p))
                else:
                    # Full re-extraction: reconcile against current code files to
                    # evict nodes from files deleted since the last run (#1007).
                    # This must include semantic document sources: an AST-only
                    # rebuild does not re-run their producer, but preserving a
                    # record whose local source is gone keeps invalid graph data
                    # forever and can leave dangling hyperedges.
                    _root_str = str(project_root)
                    current_sources = {
                        _nsf(str(p.relative_to(project_root)), _root_str)
                        for p in code_files
                        if p.is_relative_to(project_root)
                    }
                    sourced_records = (
                        list(existing.get("nodes", []))
                        + list(existing.get("links", existing.get("edges", [])))
                        + list(existing.get("hyperedges", []))
                    )
                    for record in sourced_records:
                        if not isinstance(record, dict):
                            continue
                        sf = record.get("source_file")
                        if not sf:
                            continue
                        source_file = str(sf)
                        norm = _nsf(source_file, _root_str) or source_file
                        stale_code_source = (
                            Path(source_file).suffix.lower() in _CODE_EXTENSIONS
                            and norm not in current_sources
                        )
                        if stale_code_source or _missing_local_source(
                            source_file, project_root, watch_root
                        ):
                            evict_sources.add(source_file)
                            evict_sources.add(norm)
                            deleted_paths.add(norm)
                # On a full re-extraction every code file is re-extracted, so
                # new_ast_ids is the complete current AST set. Any AST-marked node
                # missing from it is stale and must be dropped even if its source
                # file still exists (a symbol removed from a surviving file, #1116).
                # Gate on full_rebuild: in incremental mode an AST node from an
                # unchanged file is legitimately absent from new_ast_ids. Semantic
                # nodes lack the "_origin" marker, so they are never dropped here —
                # only by the deleted-file eviction in evict_sources above.
                full_rebuild = changed_paths is None
                preserved_nodes: list[dict] = []
                for node in existing.get("nodes", []):
                    if node["id"] in new_ast_ids:
                        continue
                    if full_rebuild and node.get("_origin") == "ast":
                        continue
                    if evict_sources and node.get("source_file") in evict_sources:
                        continue
                    if not node.get("label"):
                        label = node.get("name") or node.get("id")
                        if label:
                            node = {**node, "label": str(label)}
                    preserved_nodes.append(node)
                all_ids = new_ast_ids | {n["id"] for n in preserved_nodes}
                # An edge is OWNED by the file it was extracted from (its
                # source_file). When that file is re-extracted, its prior edges must
                # not be carried forward — the fresh extraction re-emits whichever
                # ones still exist. Preserving by endpoint membership alone keeps a
                # removed import's edge alive forever whenever both endpoint nodes
                # survive (e.g. `a` no longer imports from `b`, but both `a` and `b`
                # are still present), producing phantom circular dependencies
                # (#1521). So drop preserved edges whose source_file was re-extracted
                # this run (or deleted). Unlike the node-level evict set, this MUST
                # cover the full-rebuild case too — there every file is re-extracted
                # but `evict_sources` only lists deleted files, so a removed import
                # in a surviving file would never be pruned. Edges with no
                # source_file, or owned by a file that was NOT re-extracted, are
                # kept exactly as before, so cross-file edges that merely point at a
                # re-extracted file (#1402 sourceless stubs / cross-file rewire) are
                # not over-pruned — only edges the re-extracted file itself produced.
                edge_evict_sources: set[str] = set(evict_sources)
                for p in extract_targets:
                    for _root in (project_root, watch_root):
                        edge_evict_sources.add(_nsf(str(p), str(_root)) or str(p))
                # #1521 hybrid scope. On a full rebuild `extract_targets` is EVERY
                # code file, so the source_file eviction above would also drop
                # semantic / LLM-inferred edges — but a CLI full rebuild is AST-only
                # (the LLM pass is not re-run), so those edges are never re-supplied
                # and would be silently lost forever (~15% of edges on the live
                # graph). So on a full rebuild, only evict an edge the fresh AST pass
                # can actually re-supply: one with no non-AST `_origin` marker whose
                # BOTH endpoints are AST-origin nodes. A stale structural edge (a
                # removed import between two surviving AST nodes) still evicts and
                # re-appears only if it still exists; an edge marked non-AST, or
                # touching a non-AST (document / rationale / concept / LLM) node, is
                # preserved. Incremental mode is unchanged: the user edited that file,
                # so its own edges go regardless of origin.
                origin_by_id = {n["id"]: n.get("_origin") for n in existing.get("nodes", [])}

                def _is_ast_node(nid: object) -> bool:
                    return nid in new_ast_ids or origin_by_id.get(nid) == "ast"

                def _edge_ast_resuppliable(e: dict) -> bool:
                    origin = e.get("_origin")
                    if origin is not None and origin != "ast":
                        return False
                    return _is_ast_node(e.get("source")) and _is_ast_node(e.get("target"))

                def _edge_evicted(e: dict) -> bool:
                    if not edge_evict_sources:
                        return False
                    sf = e.get("source_file")
                    if not sf:
                        return False
                    sf_evicted = sf in edge_evict_sources
                    if not sf_evicted:
                        norm = _nsf(sf, str(project_root))
                        sf_evicted = bool(norm) and norm in edge_evict_sources
                    if not sf_evicted:
                        return False
                    if full_rebuild and not _edge_ast_resuppliable(e):
                        return False
                    return True

                preserved_edges = [
                    e
                    for e in existing.get("links", existing.get("edges", []))
                    if e.get("source") in all_ids
                    and e.get("target") in all_ids
                    and not _edge_evicted(e)
                ]
                from graphify.build import _repair_hyperedge_file_node_aliases_from_records

                existing_hyperedges = _repair_hyperedge_file_node_aliases_from_records(
                    result["nodes"] + preserved_nodes,
                    existing.get("hyperedges", []),
                )
                preserved_hyperedges: list[dict] = []
                for hyperedge in existing_hyperedges:
                    if not isinstance(hyperedge, dict):
                        continue
                    sf = hyperedge.get("source_file")
                    norm_sf = _nsf(str(sf), str(project_root)) if sf else None
                    if sf in evict_sources or (norm_sf and norm_sf in evict_sources):
                        continue
                    members = hyperedge.get("nodes")
                    if not isinstance(members, list):
                        members = next(
                            (
                                hyperedge.get(alias)
                                for alias in ("members", "node_ids")
                                if isinstance(hyperedge.get(alias), list)
                            ),
                            None,
                        )
                    if not isinstance(members, list):
                        continue
                    surviving_members = [
                        member
                        for member in members
                        if isinstance(member, str) and member in all_ids
                    ]
                    surviving_members = list(dict.fromkeys(surviving_members))
                    if len(surviving_members) < 2:
                        continue
                    if surviving_members != members or "nodes" not in hyperedge:
                        hyperedge = {
                            key: value
                            for key, value in hyperedge.items()
                            if key not in ("members", "node_ids")
                        }
                        hyperedge["nodes"] = surviving_members
                    preserved_hyperedges.append(hyperedge)

                result = {
                    "nodes": result["nodes"] + preserved_nodes,
                    "edges": result["edges"] + preserved_edges,
                    "hyperedges": preserved_hyperedges,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            except Exception as exc:
                print(
                    f"[graphify] error: could not preserve existing graph data: {exc}",
                    file=sys.stderr,
                )
                return False

        resolved_graph_type = graph_type or _existing_graph_type(existing_graph_data)

        _relativize_source_files(result, project_root)
        # Source files re-extracted this run — their symbol sets may legitimately
        # shrink (a removed function), so the shrink-guard should not block the
        # write when every lost node belongs to one of them (or a deleted file).
        _rebuilt_root = str(project_root)
        rebuilt_sources: set[str] = set()
        if changed_paths is None:
            for code_file in code_files:
                if not code_file.is_relative_to(project_root):
                    continue
                relative_source = str(code_file.relative_to(project_root))
                rebuilt_sources.add(_nsf(relative_source, _rebuilt_root) or relative_source)
        else:
            rebuilt_sources = {(_nsf(str(p), _rebuilt_root) or str(p)) for p in extract_targets}
        rebuilt_sources |= set(deleted_paths)
        result["edges"] = _dedupe_rebuilt_edge_records(result.get("edges", []))
        out.mkdir(exist_ok=True)

        if no_cluster:
            # Normalise to "links" key so schema is consistent with the full clustered path.
            # Dedupe parallel edges (the clustered path's DiGraph collapses them implicitly);
            # without it, --no-cluster + repeated `update` accumulate duplicates and edge
            # counts diverge across build modes (#1317).
            from graphify.build import dedupe_edges as _dedupe_edges, dedupe_nodes as _dedupe_nodes

            candidate_graph_data = {
                **{k: v for k, v in result.items() if k not in ("edges", "nodes")},
                "nodes": _dedupe_nodes(result.get("nodes", [])),
                # Multigraph keeps parallel edges; only collapse for simple graphs (#1317).
                "links": (
                    result.get("edges", [])
                    if resolved_graph_type == "multidigraph"
                    else _dedupe_edges(result.get("edges", []))
                ),
            }
            if existing_graph_metadata:
                candidate_graph_data["graph"] = existing_graph_metadata
            # The no-cluster path writes raw merged extraction JSON directly (it
            # never goes through build_from_json/to_json), so it would otherwise
            # emit a graph.json with no multigraph flag — the same deferred
            # collapse as the clustered path. Stamp the resolved class explicitly
            # so every rewrite reloads with the same graph semantics. Preserved
            # multigraph records keep their key; new AST edges receive one on load.
            _stamp_graph_type(candidate_graph_data, resolved_graph_type)
            candidate_graph_text = _json_text(candidate_graph_data)
            same_graph = False
            if existing_graph.exists():
                try:
                    check_graph_file_size_cap(existing_graph)
                    existing_payload = json.loads(existing_graph.read_text(encoding="utf-8"))
                    same_graph = json.dumps(
                        _canonical_graph_for_compare(existing_payload),
                        sort_keys=True,
                        ensure_ascii=False,
                    ) == json.dumps(
                        _canonical_graph_for_compare(candidate_graph_data),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                except Exception:
                    same_graph = False
            with FileStateTransaction(
                [existing_graph, out / "manifest.json", out / ".graphify_root"]
            ) as state_transaction:
                if not same_graph:
                    if not _check_shrink(
                        force,
                        existing_graph_data,
                        candidate_graph_data,
                        had_explicit_deletions=bool(deleted_paths),
                        rebuilt_sources=rebuilt_sources,
                    ):
                        return False
                    from graphify.export import backup_if_protected as _backup

                    _backup(out, transaction=state_transaction)
                    atomic_write_text(existing_graph, candidate_graph_text)

                from graphify.detect import save_manifest

                save_manifest(
                    manifest_files,
                    manifest_path=str(out / "manifest.json"),
                    kind="ast",
                    root=project_root,
                    expected_hashes=source_snapshot,
                )
                write_scan_root_marker(out / ".graphify_root", project_root)
                state_transaction.commit()
            _commit_ast_cache()

            if same_graph:
                print(
                    "[graphify watch] No code-graph changes detected (--no-cluster); outputs left untouched."
                )
            else:
                print(
                    "[graphify watch] Rebuilt (no clustering): "
                    f"{len(candidate_graph_data.get('nodes', []))} nodes, "
                    f"{len(candidate_graph_data.get('links', []))} edges"
                )
                print(f"[graphify watch] graph.json updated in {out}")
            return True

        detection = {
            "files": {
                "code": [str(f) for f in code_files],
                "document": [],
                "paper": [],
                "image": [],
            },
            "total_files": len(code_files),
            "total_words": detected.get("total_words", 0),
        }

        # Rebuild with the resolved saved class. MultiDiGraph records retain their
        # keys, and directed simple graphs remain directed across the write/load
        # cycle instead of being downgraded to Graph.
        saved_is_multigraph = resolved_graph_type == "multidigraph"
        G = build_from_json(
            result,
            directed=resolved_graph_type == "digraph",
            multigraph=saved_is_multigraph,
        )
        if existing_graph_metadata:
            G.graph.update(existing_graph_metadata)
        candidate_topology = _topology_from_graph(G)
        if existing_graph_data:
            try:
                same_topology = json.dumps(
                    _canonical_topology_for_compare(existing_graph_data),
                    sort_keys=True,
                    ensure_ascii=False,
                ) == json.dumps(
                    _canonical_topology_for_compare(candidate_topology),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            except Exception:
                same_topology = False
            if same_topology:
                with FileStateTransaction(
                    [out / "manifest.json", out / ".graphify_root"]
                ) as state_transaction:
                    from graphify.detect import save_manifest

                    save_manifest(
                        manifest_files,
                        manifest_path=str(out / "manifest.json"),
                        kind="ast",
                        root=project_root,
                        expected_hashes=source_snapshot,
                    )
                    write_scan_root_marker(out / ".graphify_root", project_root)
                    state_transaction.commit()
                _commit_ast_cache()
                print(
                    "[graphify watch] No code-graph topology changes detected; outputs left untouched."
                )
                return True

        communities = cluster(G)
        previous_node_community = _node_community_map(existing_graph_data)
        if previous_node_community:
            communities = remap_communities_to_previous(communities, previous_node_community)
        cohesion = score_all(G, communities)
        gods = god_nodes(G)
        surprises = surprising_connections(G, communities)
        labels_file = out / ".graphify_labels.json"
        try:
            raw = (
                json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}
            )
            labels = {int(k): v for k, v in raw.items() if int(k) in communities}
        except Exception:
            raw = {}
            labels = {}
        missing = {cid: members for cid, members in communities.items() if cid not in labels}
        if missing:
            # Deterministic hub name (highest-degree member) beats a bare "Community N"
            # placeholder for any community without a saved label.
            from graphify.cluster import label_communities_by_hub

            labels.update(label_communities_by_hub(G, missing))
        questions = suggest_questions(G, communities, labels)
        from graphify.report import load_learning_for_report as _llfr

        report = generate(
            G,
            communities,
            cohesion,
            labels,
            gods,
            surprises,
            detection,
            {"input": 0, "output": 0},
            report_root,
            suggested_questions=questions,
            built_at_commit=commit,
            learning=_llfr(out / "graph.json"),
        )
        report_path = out / "GRAPH_REPORT.md"
        labels_json = (
            json.dumps({str(k): v for k, v in sorted(labels.items())}, ensure_ascii=False, indent=2)
            + "\n"
        )
        graph_tmp = out / ".graph.tmp.json"
        # NOTE: Guard 1 in to_json (the empty-merge floor that refuses to
        # overwrite a populated graph.json with 0 nodes) is INERT here.
        # graph_tmp is a not-yet-existing temp file, so existing_path.exists()
        # is False and the guard never engages.  The RISK 4 protection on this
        # code path comes from Guard 2 (_check_shrink, called below), which
        # fires before the force/had_explicit_deletions short-circuit and
        # refuses the graph_tmp.replace(existing_graph) when the candidate has
        # 0 nodes and the on-disk graph is populated.
        json_written = to_json(G, communities, str(graph_tmp), force=True, built_at_commit=commit)
        if not json_written:
            return False
        candidate_graph_data = json.loads(graph_tmp.read_text(encoding="utf-8"))
        same_graph = False
        same_report = False
        if existing_graph.exists():
            try:
                check_graph_file_size_cap(existing_graph)
                existing_payload = json.loads(existing_graph.read_text(encoding="utf-8"))
                same_graph = json.dumps(
                    _canonical_graph_for_compare(existing_payload),
                    sort_keys=True,
                    ensure_ascii=False,
                ) == json.dumps(
                    _canonical_graph_for_compare(candidate_graph_data),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            except Exception:
                same_graph = False
        if report_path.exists():
            old_report = report_path.read_text(encoding="utf-8")
            same_report = _report_for_compare(old_report) == _report_for_compare(report)
        no_change = same_graph and same_report
        with FileStateTransaction(
            [
                existing_graph,
                report_path,
                labels_file,
                out / "manifest.json",
                out / ".graphify_root",
            ]
        ) as state_transaction:
            if no_change:
                graph_tmp.unlink(missing_ok=True)
                print(
                    "[graphify watch] No code-graph changes detected; "
                    "graph.json/GRAPH_REPORT.md left untouched."
                )
            else:
                if not _check_shrink(
                    force,
                    existing_graph_data,
                    candidate_graph_data,
                    tmp=graph_tmp,
                    had_explicit_deletions=bool(deleted_paths),
                    rebuilt_sources=rebuilt_sources,
                ):
                    return False
                from graphify.export import backup_if_protected as _backup

                _backup(out, transaction=state_transaction)
                graph_tmp.replace(existing_graph)
                atomic_write_text(report_path, report)
                atomic_write_text(labels_file, labels_json)

            from graphify.detect import save_manifest

            save_manifest(
                manifest_files,
                manifest_path=str(out / "manifest.json"),
                kind="ast",
                root=project_root,
                expected_hashes=source_snapshot,
            )
            write_scan_root_marker(out / ".graphify_root", project_root)
            state_transaction.commit()
        _commit_ast_cache()

        # to_html raises ValueError for graphs > MAX_NODES_FOR_VIZ (5000).
        # Wrap so core outputs (graph.json + GRAPH_REPORT.md) always land.
        html_written = False
        if no_viz:
            stale = out / "graph.html"
            if stale.exists():
                stale.unlink()
                print("[graphify watch] --no-viz: removed stale graph.html")
        elif not no_change:
            try:
                to_html(G, communities, str(out / "graph.html"), community_labels=labels or None)
                html_written = True
            except ValueError as viz_err:
                print(f"[graphify watch] Skipped graph.html: {viz_err}")
                stale = out / "graph.html"
                if stale.exists():
                    stale.unlink()

        # Regenerate callflow HTML if the user previously generated one —
        # opt-in by existence so users who never ran callflow-html aren't affected.
        callflow_files = list(out.glob("*-callflow.html"))
        if callflow_files and not no_change:
            try:
                from graphify.callflow_html import write_callflow_html

                for cf in callflow_files:
                    write_callflow_html(
                        graph=out / "graph.json",
                        report=out / "GRAPH_REPORT.md",
                        labels=out / ".graphify_labels.json",
                        output=cf,
                        verbose=False,
                    )
            except Exception as cf_err:
                print(f"[graphify watch] callflow HTML update skipped: {cf_err}")

        if not no_change:
            print(
                f"[graphify watch] Rebuilt: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges, {len(communities)} communities"
            )
            products = (
                "graph.json" + (", graph.html" if html_written else "") + " and GRAPH_REPORT.md"
            )
            if callflow_files:
                products += f", {len(callflow_files)} callflow HTML"
            print(f"[graphify watch] {products} updated in {out}")
        return True

    except Exception as exc:
        with contextlib.suppress(OSError):
            (out / ".graph.tmp.json").unlink()
        print(f"[graphify watch] Rebuild failed: {exc}")
        return False


def check_update(watch_path: Path, *, output_dir: Path | None = None) -> bool:
    """Check for pending semantic update flag and notify the user if set.

    Cron-safe: always returns True so cron jobs do not alarm.
    Non-code file changes (docs, papers, images) require LLM-backed
    re-extraction via `/graphify --update` — this function only signals
    that the update is needed.
    """
    flag = (
        Path(output_dir).resolve() if output_dir is not None else Path(watch_path) / _GRAPHIFY_OUT
    ) / "needs_update"
    if flag.exists():
        print(f"[graphify check-update] Pending non-code changes in {watch_path}.")
        command = f"graphify update {Path(watch_path).resolve()} --remap"
        if output_dir is not None:
            command += f" --out {Path(output_dir).resolve().parent}"
        print(f"[graphify check-update] Run `{command}` to apply semantic re-extraction.")
    return True


def _notify_only(watch_path: Path, *, output_dir: Path | None = None) -> None:
    """Write a flag file and print a notification (fallback for non-code-only corpora)."""
    flag = (
        Path(output_dir).resolve() if output_dir is not None else watch_path / _GRAPHIFY_OUT
    ) / "needs_update"
    write_pending_signal(flag)
    print(f"\n[graphify watch] New or changed files detected in {watch_path}")
    print("[graphify watch] Non-code files changed - semantic re-extraction requires LLM.")
    print("[graphify watch] Run `/graphify --update` in Claude Code to update the graph.")
    print(f"[graphify watch] Flag written to {flag}")


def _has_non_code(changed_paths: list[Path]) -> bool:
    return any(p.suffix.lower() not in _CODE_EXTENSIONS for p in changed_paths)


def watch(
    watch_path: Path,
    debounce: float = 3.0,
    *,
    output_dir: Path | None = None,
    output_root: Path | None = None,
    graph_type: str | None = None,
) -> None:
    """
    Watch watch_path for new or modified files and auto-update the graph.

    For code-only changes: re-runs AST extraction + rebuild immediately (no LLM).
    For doc/paper/image changes: writes a needs_update flag and notifies the user
    to run /graphify --update (LLM extraction required).

    debounce: seconds to wait after the last change before triggering (avoids
    running on every keystroke when many files are saved at once).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.observers.polling import PollingObserver
        from watchdog.events import FileSystemEventHandler
    except ImportError as e:
        raise ImportError(f"watchdog not installed. Run: {uv_tool_install_command('watch')}") from e

    last_trigger: float = 0.0
    pending: bool = False
    changed: set[Path] = set()

    # Load .graphifyignore patterns ONCE at startup so the handler does not
    # re-parse the file on every filesystem event. Watchdog's handler runs on
    # the observer thread and is invoked for every event the OS delivers
    # (Time Machine writes, Docker/Colima VM I/O, Spotlight indexing, …) —
    # without this short-circuit a busy volume can saturate a CPU core
    # discarding events one extension at a time. (gh-928)
    watch_root_for_ignore = watch_path.resolve()
    ignore_patterns = _load_graphifyignore(watch_root_for_ignore)

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal last_trigger, pending
            if event.is_directory:
                return
            path = Path(os.fsdecode(event.src_path))
            # Check .graphifyignore BEFORE the extension/dotfile/out filters so
            # the cheapest short-circuit for users with broad ignore patterns
            # (node_modules/, .venv/, build/, …) fires first. _is_ignored
            # tolerates absolute paths outside watch_root via its internal
            # relative_to guard, so a stray symlinked event won't raise.
            if ignore_patterns and _is_ignored(path, watch_root_for_ignore, ignore_patterns):
                return
            if path.suffix.lower() not in _WATCHED_EXTENSIONS:
                return
            try:
                filter_parts = path.relative_to(watch_root_for_ignore).parts
            except ValueError:
                filter_parts = path.parts
            if any(part.startswith(".") for part in filter_parts):
                return
            if _GRAPHIFY_OUT in filter_parts:
                return
            last_trigger = time.monotonic()
            pending = True
            changed.add(path)

    handler = Handler()
    # Use polling observer on macOS — FSEvents can miss rapid saves in some editors
    observer = PollingObserver() if sys.platform == "darwin" else Observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    observer.start()

    print(f"[graphify watch] Watching {watch_path.resolve()} - press Ctrl+C to stop")
    print(
        "[graphify watch] Code changes rebuild graph automatically. "
        "Doc/image changes require /graphify --update."
    )
    print(f"[graphify watch] Debounce: {debounce}s")

    try:
        while True:
            time.sleep(0.5)
            if pending and (time.monotonic() - last_trigger) >= debounce:
                pending = False
                batch = list(changed)
                changed.clear()
                print(f"\n[graphify watch] {len(batch)} file(s) changed")
                has_non_code = _has_non_code(batch)
                has_code = any(p.suffix.lower() in _CODE_EXTENSIONS for p in batch)
                if has_code:
                    _rebuild_code(
                        watch_path,
                        output_dir=output_dir,
                        output_root=output_root,
                        graph_type=graph_type,
                        changed_paths=batch,
                    )
                if has_non_code:
                    _notify_only(watch_path, output_dir=output_dir)
    except KeyboardInterrupt:
        print("\n[graphify watch] Stopped.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch a folder and auto-update the graphify graph"
    )
    parser.add_argument("path", nargs="?", default=".", help="Folder to watch (default: .)")
    parser.add_argument(
        "--debounce",
        type=float,
        default=3.0,
        help="Seconds to wait after last change before updating (default: 3)",
    )
    args = parser.parse_args()
    watch(Path(args.path), debounce=args.debounce)
